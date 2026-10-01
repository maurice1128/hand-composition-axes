"""experiment_paired_composition.py on a PERMUTED label grid: a within-dataset zero-truth control.

Why
---
Every penalty in this study is read against one synthetic zero-truth control (+2.6% at budget 256). Nothing
establishes that the paired design's bias is dataset-independent: it plausibly depends on cell count, group sizes,
trajectory length, error scale and how much of its budget the informed arm can fill. A referee's objection, and a
fair one: if OakInk-Image's own bias were +8%, much of Table 1 would evaporate.

This wrapper supplies the missing control on the real dataset itself. It permutes the SECOND factor of the coarse
label across fine groups, so that

  - the number of trajectories, their poses and their lengths are untouched;
  - each factor's marginal distribution over fine groups is untouched;
  - the cell count and the distribution of group sizes are close to the real grid's;
  - but the pairing of the two factors carries no relationship that a prior could exploit.

Holding out a cell of the permuted grid therefore removes a set of trajectories that is structured exactly as a real
held-out cell is, while the informed arm's extra trajectories are no longer motion of the target's own composition.

Reading. If the real grid's penalty is compositional, the permuted grid returns the zero-truth control's value. If
the permuted grid returns the real grid's value, then the measurement responds to any structured cell holdout and
the word "compositional" is not earned. This is declared before any model is trained; see
runs/PREREG_sham_grid.md.

How
---
The permutation is applied to FINE GROUPS, not to trajectories, because ``build_paired_split`` assigns whole fine
groups to one side of the split. Permuting trajectories individually would put one fine label in several sham cells
and break the disjointness the leak gate enforces. For each distinct fine label, its coarse label ``a->b`` keeps
``a`` and receives a ``b`` drawn without replacement from the multiset of ``b`` over all fine groups. Seeded by
``--sham-seed`` so the grid is one fixed object across the sweep's seeds, exactly as the real grid is; the sweep's
own seeds then vary the split within it, as they do for the real grid.

A permutation is REJECTED and redrawn if it leaves the candidate pool (compositions carried by at least
``--min-chains`` distinct fine groups) smaller than the real grid's, because a sham grid that cannot supply a split
is not a matched control. The realised pool size for both grids is printed and recorded.

Usage: exactly the arguments of experiment_paired_composition.py, plus

  --sham-seed N    seed of the permutation itself (default 12345). One grid per sweep.
  --dry-run        train nothing; print the real and sham grids side by side (cells, pool, group sizes), and for
                   every seed the pool sizes, the informed arm's coverage and realised exposure, and the label leak.
  --dry-run-out P  where the dry run writes its JSON (default runs/gates/sham_grid_dryrun.json).

A sidecar ``<out>/sham_grid.json`` records the permutation and the per-seed pool sizes. results.json is written by
the module and is unchanged.

What this does NOT do
---------------------
- It is not a test of whether the prior composes. It is a test of what the measurement reads when there is nothing
  to compose.
- It does not match the informed arm's realised exposure by construction; it matches the structure that determines
  exposure. The realised value is measured and reported for both grids, and a large difference is itself a finding.
- Permuting the second factor leaves a fine label's own two factors intact, so the FINE composition of a target is
  still held out of both arms by the same leak rule. The sham grid changes only which fine groups are considered to
  share a cell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_STATE: dict = {"log": [], "sidecar": None, "sham_seed": 12345, "mapping": None, "grids": None}


def _split_coarse(lab: str) -> tuple[str, str]:
    """A coarse label of this study is ``a->b``; a chain label has more arrows and is refused."""
    parts = lab.split("->")
    if len(parts) != 2:
        raise ValueError(
            f"the sham grid needs two-factor coarse labels of the form 'a->b'; got {lab!r} with "
            f"{len(parts)} parts. Chain axes (OakInk2 annotated transitions) have no second factor to permute.")
    return parts[0], parts[1]


def permute_grid(coarse: list[str], fine: list[str], sham_seed: int, min_chains: int) -> dict:
    """Return {fine label -> sham coarse label} and a description of both grids.

    The second factor is permuted over DISTINCT FINE GROUPS. Redraws until the sham grid's candidate pool is at
    least the real grid's, so the control is matched on the quantity that decides whether a split exists at all.
    """
    import numpy as np

    by_fine: dict[str, str] = {}
    for f, c in zip(fine, coarse):
        if by_fine.setdefault(f, c) != c:
            raise ValueError(f"fine label {f!r} carries two coarse labels ({by_fine[f]!r}, {c!r}); "
                             "the sham grid assumes a fine label sits in exactly one cell")
    groups = sorted(by_fine)
    firsts = [_split_coarse(by_fine[g])[0] for g in groups]
    seconds = [_split_coarse(by_fine[g])[1] for g in groups]

    def pool_size(mapping: dict[str, str]) -> int:
        chains: dict[str, set[str]] = {}
        for f in groups:
            chains.setdefault(mapping[f], set()).add(f)
        return sum(len(v) >= min_chains for v in chains.values())

    real = {g: by_fine[g] for g in groups}
    real_pool = pool_size(real)

    rng = np.random.default_rng(sham_seed)
    for attempt in range(200):
        perm = rng.permutation(len(seconds))
        sham = {g: f"{firsts[i]}->{seconds[perm[i]]}" for i, g in enumerate(groups)}
        sham_pool = pool_size(sham)
        if sham_pool >= real_pool:
            break
    else:
        raise ValueError(
            f"200 permutations all left the sham candidate pool below the real grid's ({real_pool}); "
            f"best was {sham_pool}. This grid cannot supply a matched control.")

    def describe(mapping: dict[str, str]) -> dict:
        cells = Counter(mapping.values())
        per_cell = Counter()
        for f, c in zip(fine, coarse):
            per_cell[mapping[f]] += 1
        sizes = sorted(per_cell.values())
        return {"cells": len(cells), "fine_groups": len(mapping),
                "candidate_pool": pool_size(mapping),
                "median_trajectories_per_cell": sizes[len(sizes) // 2] if sizes else 0,
                "max_trajectories_per_cell": max(sizes) if sizes else 0}

    return {"mapping": sham, "attempts": attempt + 1,
            "real": describe(real), "sham": describe(sham),
            "moved": sum(sham[g] != real[g] for g in groups), "n_fine_groups": len(groups)}


def permute_cells(coarse: list[str], fine: list[str], sham_seed: int, min_chains: int) -> dict:
    """Reassign whole fine groups to cells, keeping every cell's number of fine groups exactly.

    For a concentrated grid. TACO's action-by-tool grid is 16% occupied because a tool affords few actions; permuting
    the tool factor scatters its 151 triplets and left at most 3 cells spanned by four triplets against 17, so that
    control is not matched there and ``permute_grid`` refuses. Here the multiset of cell labels over fine groups is
    shuffled instead: the cells, their sizes in fine groups, the candidate pool and each factor's marginal over fine
    groups are all EXACTLY the real grid's, and only which triplets sit together changes.

    What differs from ``permute_grid``: a fine group's own factors no longer match its cell's, in either factor, so
    this is a stronger scrambling. Trajectories per cell are not preserved exactly, because fine groups differ in
    size; the realised distribution is reported beside the real one.
    """
    import numpy as np

    by_fine: dict[str, str] = {}
    for f, c in zip(fine, coarse):
        if by_fine.setdefault(f, c) != c:
            raise ValueError(f"fine label {f!r} carries two coarse labels")
    groups = sorted(by_fine)
    cells = [by_fine[g] for g in groups]
    rng = np.random.default_rng(sham_seed)
    perm = rng.permutation(len(cells))
    sham = {g: cells[perm[i]] for i, g in enumerate(groups)}
    real = {g: by_fine[g] for g in groups}

    def describe(mapping):
        from collections import Counter as C
        chains: dict[str, set[str]] = {}
        for g in groups:
            chains.setdefault(mapping[g], set()).add(g)
        per_cell = C()
        for f in fine:
            per_cell[mapping[f]] += 1
        sizes = sorted(per_cell.values())
        return {"cells": len(chains), "fine_groups": len(groups),
                "candidate_pool": sum(len(v) >= min_chains for v in chains.values()),
                "median_trajectories_per_cell": sizes[len(sizes) // 2], "max_trajectories_per_cell": max(sizes)}

    r, s_ = describe(real), describe(sham)
    assert (r["cells"], r["candidate_pool"]) == (s_["cells"], s_["candidate_pool"]), (r, s_)
    return {"mapping": sham, "attempts": 1, "real": r, "sham": s_, "mode": "cells",
            "moved": sum(sham[g] != real[g] for g in groups), "n_fine_groups": len(groups)}


_PERMUTE = {"factor": None, "cells": permute_cells}


def install(E, sham_seed: int, min_chains: int) -> None:
    """Replace the module's ``build_paired_split`` with one that first permutes the grid."""
    original = E.build_paired_split

    def build_paired_split_sham(bundle, n_held_compositions, seed=0, *a, **kw):
        fine = kw.get("fine_labels") or (a[1] if len(a) > 1 else None)
        if fine is None:
            raise ValueError("the sham grid needs fine_labels; run with a coarsened --granularity")
        if _STATE["mapping"] is None:
            info = (_PERMUTE[_STATE.get('mode', 'factor')] or permute_grid)(
                list(bundle.labels), list(fine), sham_seed, min_chains)
            _STATE["mapping"], _STATE["grids"] = info["mapping"], info
            print(f"[sham] permuted the second factor over {info['n_fine_groups']} fine groups "
                  f"({info['moved']} moved, {info['attempts']} draw(s))")
            print(f"[sham] real grid: {info['real']}")
            print(f"[sham] sham grid: {info['sham']}")
        mapping = _STATE["mapping"]
        real_labels = list(bundle.labels)
        bundle.labels = [mapping[f] for f in fine]
        try:
            split = original(bundle, n_held_compositions, seed, *a, **kw)
        finally:
            # main() scores and reports with bundle.labels; leave the sham grid in place for the
            # rest of this seed so coverage and leak checks see the grid the split was made on.
            bundle.labels = [mapping[f] for f in fine]
        split["sham_grid"] = True
        split["real_labels_replaced"] = len(real_labels)
        _STATE["log"].append({"seed": int(seed), "naive_pool": len(split["naive_pool"]),
                              "informed_extra": len(split["informed_extra"]),
                              "held": list(split["held_compositions"])})
        if _STATE["sidecar"] is not None:
            p = Path(_STATE["sidecar"])
            p.parent.mkdir(parents=True, exist_ok=True)
            prior = []
            if p.exists():
                try:
                    prior = json.loads(p.read_text(encoding="utf-8")).get("seeds", [])
                except (OSError, ValueError):
                    prior = []
            rows = {r["seed"]: r for r in prior}
            rows.update({r["seed"]: r for r in _STATE["log"]})
            p.write_text(json.dumps({
                "control": "the second factor of the coarse label is permuted over fine groups, so cell "
                           "structure is preserved and the pairing carries no relationship",
                "wrapper": "scripts/experiment_paired_sham_grid.py",
                "sham_seed": sham_seed, "grids": _STATE["grids"],
                "mapping": _STATE["mapping"],
                "seeds": [rows[k] for k in sorted(rows)]}, indent=1), encoding="utf-8")
        return split

    build_paired_split_sham.__wrapped__ = original
    E.build_paired_split = build_paired_split_sham


def dry_run(E, a, out_path: Path, sham_seed: int) -> int:
    import numpy as np
    from caredex.data.base import TrajectoryBundle

    bundle = TrajectoryBundle.load(a.bundle)
    fine = list(bundle.labels)
    if a.granularity == "fine":
        print("REFUSED: --granularity fine has no coarse grid to permute.")
        return 2
    bundle.labels = E.coarsen_labels(bundle.labels, a.granularity)
    info = (_PERMUTE[_STATE.get('mode', 'factor')] or permute_grid)(
        list(bundle.labels), fine, sham_seed, a.min_chains)
    print(f"{a.bundle}: {len(fine)} trajectories, {info['n_fine_groups']} fine groups")
    print(f"real grid: {info['real']}")
    print(f"sham grid: {info['sham']}  ({info['moved']} groups moved, {info['attempts']} draw(s))")

    original = E.build_paired_split
    seeds = a.seeds if a.seeds else [a.seed]
    rows = []
    print(f"\n{'seed':>5} {'grid':<5} {'budget':>6} | {'pool':>5} {'extra':>6} | {'cov':>7} {'per':>5} "
          f"{'status':<9} | {'exposure':>8} | leak n/i")
    for seed in seeds:
        for which in ("real", "sham"):
            labels = (E.coarsen_labels(list(fine), a.granularity) if which == "real"
                      else [info["mapping"][f] for f in fine])
            bundle.labels = labels
            try:
                split = original(bundle, a.held_compositions, seed, fine_labels=fine,
                                 min_chains=a.min_chains)
            except ValueError as exc:
                rows.append({"seed": seed, "grid": which, "error": str(exc)})
                print(f"{seed:>5} {which:<5} ERROR {exc}")
                continue
            for budget in a.budgets:
                if budget > len(split["naive_pool"]):
                    rows.append({"seed": seed, "grid": which, "budget": budget, "feasible": False})
                    print(f"{seed:>5} {which:<5} {budget:>6} | INFEASIBLE (pool {len(split['naive_pool'])})")
                    continue
                naive, informed = E.sample_pools(split, budget, seed, labels, a.min_per_composition)
                sound = E.assert_split_sound(split, naive, informed, fine, labels)
                held = set(split["held_compositions"])
                seen = Counter()
                for i in informed:
                    for x, y in E.transitions_of(labels[i]):
                        if f"{x}->{y}" in held:
                            seen[f"{x}->{y}"] += 1
                per = float(np.mean(list(seen.values()))) if seen else 0.0
                ok = len(seen) / max(len(held), 1) > 0.8 and per >= 3.0
                # Realised exposure: the share of the informed arm's budget drawn from held-out cells.
                exposure = len([i for i in informed if held & {f"{x}->{y}" for x, y in
                                                               E.transitions_of(labels[i])}]) / budget
                rows.append({"seed": seed, "grid": which, "budget": budget, "feasible": True,
                             "naive_pool": len(split["naive_pool"]),
                             "informed_extra": len(split["informed_extra"]),
                             "covered": len(seen), "held": len(held), "per_comp": per,
                             "coverage_ok": bool(ok), "exposure": exposure,
                             "leak": {"naive": sound["leak_naive"], "informed": sound["leak_informed"]}})
                print(f"{seed:>5} {which:<5} {budget:>6} | {len(split['naive_pool']):>5} "
                      f"{len(split['informed_extra']):>6} | {len(seen):>3}/{len(held):<3} {per:>5.2f} "
                      f"{'ok' if ok else 'TOO THIN':<9} | {exposure:>7.3f}  | "
                      f"{sound['leak_naive']:.3f}/{sound['leak_informed']:.3f}")

    summary = {}
    for which in ("real", "sham"):
        good = [r for r in rows if r.get("grid") == which and r.get("feasible")]
        if not good:
            continue
        summary[which] = {
            "seeds": len(good),
            "coverage_fail": sum(not r["coverage_ok"] for r in good),
            "mean_exposure": float(np.mean([r["exposure"] for r in good])),
            "mean_covered": float(np.mean([r["covered"] for r in good])),
            "max_leak": max(max(r["leak"].values()) for r in good),
            "min_naive_pool": min(r["naive_pool"] for r in good),
        }
        print(f"\n{which}: coverage fails {summary[which]['coverage_fail']} of {len(good)}; "
              f"mean informed exposure {summary[which]['mean_exposure']:.3f}; "
              f"mean cells covered {summary[which]['mean_covered']:.2f}; "
              f"max label leak {summary[which]['max_leak']:.3f}")
    if "real" in summary and "sham" in summary:
        d = summary["sham"]["mean_exposure"] - summary["real"]["mean_exposure"]
        print(f"\nexposure sham - real = {d:+.3f}. The control is matched on structure, not on this "
              f"quantity; a large gap is reported with the result.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"bundle": a.bundle, "granularity": a.granularity,
                                    "sham_seed": sham_seed, "grids": info,
                                    "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(f"wrote {out_path}\nDRY RUN: nothing was trained.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        print("=" * 78 + "\nArguments of the wrapped experiment_paired_composition.py follow.\n" + "=" * 78)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sham-seed", type=int, default=12345)
    pre.add_argument("--sham-mode", choices=("factor", "cells"), default="factor",
                     help="factor: permute the second factor over fine groups (the registered control). "
                          "cells: reassign fine groups to cells keeping cell sizes, for concentrated grids.")
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--dry-run-out", default="runs/gates/sham_grid_dryrun.json")
    mine, rest = pre.parse_known_args(argv)
    _STATE["mode"] = mine.sham_mode

    if mine.dry_run:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("OMP_NUM_THREADS", "4")

    import experiment_paired_composition as E

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--bundle", default="data/bundles/oakink.npz")
    peek.add_argument("--out", default="runs/paired")
    peek.add_argument("--budgets", type=int, nargs="*", default=[32, 64, 128, 256])
    peek.add_argument("--held-compositions", type=int, default=8)
    peek.add_argument("--min-chains", type=int, default=4)
    peek.add_argument("--min-per-composition", type=int, default=4)
    peek.add_argument("--granularity", default="fine")
    peek.add_argument("--seed", type=int, default=0)
    peek.add_argument("--seeds", type=int, nargs="*", default=None)
    a, _ = peek.parse_known_args(rest)

    if mine.dry_run and not ("-h" in argv or "--help" in argv):
        return dry_run(E, a, ROOT / mine.dry_run_out, mine.sham_seed)

    install(E, mine.sham_seed, a.min_chains)
    assert E.build_paired_split.__name__ == "build_paired_split_sham"
    assert E.main.__globals__["build_paired_split"] is E.build_paired_split
    _STATE["sidecar"] = str(Path(a.out) / "sham_grid.json")
    sys.argv = [sys.argv[0], *rest]
    return E.main()


if __name__ == "__main__":
    raise SystemExit(main())
