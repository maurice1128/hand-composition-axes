"""experiment_paired_composition.py with recording-disjoint training pools.

Why. ``oakink2_primseg.npz`` cuts 609 OakInk2 recordings into 2,128 primitive
segments. ``build_paired_split`` keeps FINE LABELS disjoint between the target
set and both training arms, but segments of one recording carry different fine
labels, so about 45% of target segments share a recording (never a frame) with
some training segment. The label-leak gate cannot see that.

What this does. It imports ``experiment_paired_composition`` and replaces its
module-level ``build_paired_split`` with a wrapper that runs the original
unchanged and then removes from ``naive_pool`` and ``informed_extra`` every
index whose recording also supplies a ``target`` index. It then calls the
module's own ``main()``. ``main()`` resolves ``build_paired_split`` as a module
global at call time, so the patch takes effect without editing that file.
Target set, held compositions, windowing, models, scoring, per-row resume and
the results.json format are the module's own code, untouched.

Recording identity: ``bundle.meta['segments'][i]['sequence']`` (written by
``scripts/build_oakink2_diagnostics.py``; one row per trajectory). Any bundle
whose meta has that table works; one without it is refused.

Usage: exactly the arguments of experiment_paired_composition.py, plus

  --dry-run            train nothing. For every seed and budget print the pool
                       sizes before and after the filter, the informed arm's
                       coverage (check_informed_coverage.py's rule), and the
                       share of target segments sharing a recording with each
                       arm before and after (after is asserted to be 0.0).
  --frame-leak-seeds N in a dry run, also report verbatim target frames in each
                       arm for the first N seeds (check_frame_leak.py identity).
  --dry-run-out PATH   where the dry run writes its JSON
                       (default runs/gates/recording_disjoint_dryrun.json).

In a real run a sidecar ``<out>/recording_filter.json`` records per seed what the
filter removed; results.json itself is written by the module and is unchanged.

What this does NOT do
---------------------
- It does not make the split the same as the original sweep's. The target set
  and held compositions for a seed are identical, but ``sample_pools`` draws
  from smaller pools, so the training sets differ in more than the removed
  recordings.
- It removes whole recordings from training, so the training arms lose segments
  of the target's scene/subject sessions in both arms equally. If error rises in
  both arms, that is the filter, not composition.
- A budget larger than the filtered naive pool is SKIPPED by the module's
  ``main()`` with one printed line, as before. This wrapper prints a louder
  warning but does not change that behaviour; run ``--dry-run`` first.
- Different recordings of the same subject in the same scene remain on both
  sides. "Recording-disjoint" is not "subject-disjoint" or "scene-disjoint".
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

_STATE: dict = {"log": [], "sidecar": None, "budgets": None}


def recordings_of(bundle) -> list[str]:
    meta = getattr(bundle, "meta", None) or {}
    rows = meta.get("segments")
    n = len(bundle.labels)
    if not rows or len(rows) != n or not all("sequence" in r for r in rows):
        raise ValueError(
            "recording-disjoint split needs bundle.meta['segments'][i]['sequence'] for "
            f"every trajectory (found {0 if not rows else len(rows)} rows for {n})")
    return [str(r["sequence"]) for r in rows]


def filter_split(split: dict, rec: list[str]) -> dict:
    """Drop from both training pools every index whose recording supplies a target."""
    target_rec = {rec[i] for i in split["target"]}
    before = {"naive_pool": len(split["naive_pool"]),
              "informed_extra": len(split["informed_extra"])}
    out = dict(split)
    out["naive_pool"] = [i for i in split["naive_pool"] if rec[i] not in target_rec]
    out["informed_extra"] = [i for i in split["informed_extra"] if rec[i] not in target_rec]
    assert not ({rec[i] for i in out["naive_pool"]} & target_rec)
    assert not ({rec[i] for i in out["informed_extra"]} & target_rec)
    assert out["target"] == split["target"]
    assert out["held_compositions"] == split["held_compositions"]
    if not out["informed_extra"]:
        raise ValueError("recording filter left informed_extra empty")
    out["recording_disjoint"] = True
    out["recording_filter"] = {
        "target_recordings": len(target_rec),
        "naive_pool_before": before["naive_pool"], "naive_pool_after": len(out["naive_pool"]),
        "informed_extra_before": before["informed_extra"],
        "informed_extra_after": len(out["informed_extra"]),
    }
    return out


def install(E) -> None:
    original = E.build_paired_split

    def build_paired_split_recording_disjoint(bundle, n_held_compositions, seed=0, *a, **kw):
        split = filter_split(original(bundle, n_held_compositions, seed, *a, **kw),
                             recordings_of(bundle))
        f = split["recording_filter"]
        print(f"[recording-disjoint] seed {seed}: naive_pool {f['naive_pool_before']} -> "
              f"{f['naive_pool_after']}, informed_extra {f['informed_extra_before']} -> "
              f"{f['informed_extra_after']} ({f['target_recordings']} target recordings)")
        budgets = _STATE["budgets"] or []
        for b in budgets:
            if b > f["naive_pool_after"]:
                print(f"[recording-disjoint] WARNING seed {seed}: budget {b} exceeds the "
                      f"filtered naive pool ({f['naive_pool_after']}); main() will SKIP it")
        _STATE["log"].append({"seed": int(seed), **f})
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
                "filter": "training indices whose meta.segments[i].sequence also supplies a "
                          "target index are removed from naive_pool and informed_extra",
                "wrapper": "scripts/experiment_paired_recording_disjoint.py",
                "seeds": [rows[k] for k in sorted(rows)]}, indent=1), encoding="utf-8")
        return split

    build_paired_split_recording_disjoint.__wrapped__ = original
    E.build_paired_split = build_paired_split_recording_disjoint


def _frame_keys(traj):
    import numpy as np
    return {np.round(row, 5).tobytes() for row in np.asarray(traj, dtype=np.float32)}


def dry_run(E, a, out_path: Path, frame_leak_seeds: int, min_examples: float = 3.0) -> int:
    import numpy as np
    from caredex.data.base import TrajectoryBundle

    original = E.build_paired_split.__wrapped__
    bundle = TrajectoryBundle.load(a.bundle)
    fine = list(bundle.labels)
    if a.granularity != "fine":
        bundle.labels = E.coarsen_labels(bundle.labels, a.granularity)
    rec = recordings_of(bundle)
    per_rec = Counter(rec)
    print(f"{a.bundle}: {len(rec)} segments from {len(per_rec)} recordings "
          f"(sum {sum(per_rec.values())}, max {max(per_rec.values())} per recording)")
    seeds = a.seeds if a.seeds else [a.seed]

    def share(target, idx):
        r = {rec[i] for i in idx}
        return sum(rec[i] in r for i in target) / max(len(target), 1)

    def coverage(split, informed):
        held = set(split["held_compositions"])
        seen = Counter()
        for i in informed:
            for x, y in E.transitions_of(bundle.labels[i]):
                if f"{x}->{y}" in held:
                    seen[f"{x}->{y}"] += 1
        per = float(np.mean(list(seen.values()))) if seen else 0.0
        ok = len(seen) / max(len(held), 1) > 0.8 and per >= min_examples
        return len(seen), len(held), per, ok

    rows = []
    print(f"\n{'seed':>5} {'budget':>6} | {'pool':>5}->{'pool':<5} {'extra':>5}->{'extra':<5} | "
          f"{'cov':>5} {'per':>5} {'status':<9} | shared-rec before n/i    after n/i | frame leak n/i")
    for k, seed in enumerate(seeds):
        raw = original(bundle, a.held_compositions, seed, fine_labels=fine,
                       min_chains=a.min_chains)
        try:
            flt = E.build_paired_split(bundle, a.held_compositions, seed, fine_labels=fine,
                                       min_chains=a.min_chains)
        except ValueError as exc:
            rows.append({"seed": seed, "error": str(exc)})
            print(f"{seed:>5} ERROR {exc}")
            continue
        for budget in a.budgets:
            row = {"seed": seed, "budget": budget, **flt["recording_filter"]}
            if budget > len(flt["naive_pool"]):
                row["feasible"] = False
                rows.append(row)
                print(f"{seed:>5} {budget:>6} | {len(raw['naive_pool']):>5}->"
                      f"{len(flt['naive_pool']):<5} INFEASIBLE: budget exceeds filtered pool")
                continue
            n0, i0 = E.sample_pools(raw, budget, seed, bundle.labels, a.min_per_composition)
            n1, i1 = E.sample_pools(flt, budget, seed, bundle.labels, a.min_per_composition)
            sound = E.assert_split_sound(flt, n1, i1, fine, bundle.labels)
            cov, held, per, ok = coverage(flt, i1)
            cov0, _, per0, ok0 = coverage(raw, i0)
            sb = (share(raw["target"], n0), share(raw["target"], i0))
            sa = (share(flt["target"], n1), share(flt["target"], i1))
            assert sa == (0.0, 0.0), f"seed {seed}: recording still shared after filter {sa}"
            assert len(n1) == len(i1) == budget
            row.update(feasible=True, covered=cov, held=held, per_comp=per, coverage_ok=bool(ok),
                       covered_unfiltered=cov0, per_comp_unfiltered=per0,
                       coverage_ok_unfiltered=bool(ok0),
                       shared_recording_before={"naive": sb[0], "informed": sb[1]},
                       shared_recording_after={"naive": sa[0], "informed": sa[1]},
                       label_leak={"naive": sound["leak_naive"], "informed": sound["leak_informed"]},
                       swapped=len([i for i in i1 if i not in set(n1)]))
            fl = ""
            if k < frame_leak_seeds:
                tgt = set().union(*(_frame_keys(bundle.trajectories[i]) for i in flt["target"]))
                fn = set().union(*(_frame_keys(bundle.trajectories[i]) for i in n1))
                fi = set().union(*(_frame_keys(bundle.trajectories[i]) for i in i1))
                row["frame_leak"] = {"naive": len(tgt & fn) / len(tgt),
                                     "informed": len(tgt & fi) / len(tgt)}
                fl = f"{100 * row['frame_leak']['naive']:.2f}% / {100 * row['frame_leak']['informed']:.2f}%"
                del tgt, fn, fi
            rows.append(row)
            print(f"{seed:>5} {budget:>6} | {len(raw['naive_pool']):>5}->{len(flt['naive_pool']):<5} "
                  f"{len(raw['informed_extra']):>5}->{len(flt['informed_extra']):<5} | "
                  f"{cov:>3}/{held} {per:>5.2f} {'ok' if ok else 'TOO THIN':<9} | "
                  f"{sb[0]:.3f} / {sb[1]:.3f}        {sa[0]:.3f} / {sa[1]:.3f} | {fl}")

    summary = {}
    for budget in a.budgets:
        rs = [r for r in rows if r.get("budget") == budget]
        good = [r for r in rs if r.get("feasible")]
        summary[str(budget)] = {
            "seeds": len(rs),
            "infeasible_seeds": [r["seed"] for r in rs if not r.get("feasible")],
            "min_naive_pool_after": min((r["naive_pool_after"] for r in rs), default=None),
            "min_naive_pool_before": min((r["naive_pool_before"] for r in rs), default=None),
            "min_informed_extra_after": min((r["informed_extra_after"] for r in rs), default=None),
            "coverage_fail": sum(not r["coverage_ok"] for r in good),
            "coverage_fail_seeds": [r["seed"] for r in good if not r["coverage_ok"]],
            "coverage_fail_unfiltered": sum(not r["coverage_ok_unfiltered"] for r in good),
            "mean_shared_recording_before": {
                arm: float(np.mean([r["shared_recording_before"][arm] for r in good])) if good else None
                for arm in ("naive", "informed")},
            "max_shared_recording_after": max(
                (max(r["shared_recording_after"].values()) for r in good), default=None),
            "max_frame_leak": max((max(r["frame_leak"].values()) for r in good if "frame_leak" in r),
                                  default=None),
        }
        s = summary[str(budget)]
        print(f"\nbudget {budget}: min naive pool {s['min_naive_pool_before']} -> "
              f"{s['min_naive_pool_after']}; infeasible seeds {s['infeasible_seeds']}; coverage fails "
              f"{s['coverage_fail']} of {len(good)} (unfiltered {s['coverage_fail_unfiltered']}); "
              f"shared recording before {s['mean_shared_recording_before']}, after max "
              f"{s['max_shared_recording_after']}; max frame leak {s['max_frame_leak']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "bundle": a.bundle, "granularity": a.granularity, "held": a.held_compositions,
        "min_chains": a.min_chains, "min_per_composition": a.min_per_composition,
        "recordings": len(per_rec), "segments": len(rec),
        "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(f"wrote {out_path}\nDRY RUN: nothing was trained.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        print("=" * 78 + "\nArguments of the wrapped experiment_paired_composition.py follow.\n" + "=" * 78)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--frame-leak-seeds", type=int, default=0)
    pre.add_argument("--dry-run-out", default="runs/gates/recording_disjoint_dryrun.json")
    mine, rest = pre.parse_known_args(argv)

    if mine.dry_run:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("OMP_NUM_THREADS", "4")

    import experiment_paired_composition as E

    install(E)
    assert E.build_paired_split.__name__ == "build_paired_split_recording_disjoint"
    # main() looks the name up in its own module globals at call time; prove that
    # is the object just installed, or the filter would silently not apply.
    assert E.main.__globals__["build_paired_split"] is E.build_paired_split

    # Read, never consume, the module's own arguments that the wrapper needs.
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
    _STATE["budgets"] = list(a.budgets)

    if mine.dry_run and not ("-h" in argv or "--help" in argv):
        return dry_run(E, a, ROOT / mine.dry_run_out, mine.frame_leak_seeds)

    _STATE["sidecar"] = str(Path(a.out) / "recording_filter.json")
    sys.argv = [sys.argv[0], *rest]
    return E.main()


if __name__ == "__main__":
    raise SystemExit(main())
