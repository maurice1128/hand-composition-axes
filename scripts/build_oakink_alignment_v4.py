"""Alignment pair v4: two-clip trajectories whose constituents the split can see.

Why. In ``oakink_long3_aligned`` (v2) a trajectory is three clips sharing a
category and an intent, labelled with the FIRST clip's ``object->intent``. The
paired split keeps fine labels disjoint, but only the first clip's: clips 2 and
3 can be other objects, and the informed arm trains on clips with exactly those
``object->intent`` pairs (``scripts/check_constituent_leak.py``: 88% of target
trajectories for the informed arm, 0% for the naive arm, seeds 0-9). The aligned
sweep's excess over the unmodified baseline is therefore inflated.

This builds, from ``data/bundles/oakink.npz`` with numpy seed 0:

``oakink_pair_aligned``     2 clips with the SAME ``object->intent``; label = it.
    Every constituent carries the trajectory's own fine label, so the split's
    fine-label disjointness covers all constituents. Size: each fine label with
    n clips yields floor(n/2) trajectories; that is the maximum under the rules.
``oakink_pair_misaligned``  2 clips of the same category, the second with a
    DIFFERENT intent; label = first clip's ``object->intent``. Drawn only from
    the aligned bundle's categories, same trajectory count, per-category counts
    matched to the aligned bundle's as closely as the clips allow.

Shared rules, asserted: no source clip used twice in a bundle; no frame (rounded
to 1e-5) in two trajectories; v2's 4-frame linear cross-fade, imported not
copied; ``meta["source_clips"]`` records the composition.

The builder then runs ``check_constituent_leak.measure`` on both and ASSERTS
measure (a) is exactly 0 for both arms of the aligned bundle on seeds 0-9.

What this does NOT fix or show
------------------------------
- The misaligned bundle's constituent exposure (a) is NOT zero and cannot be
  made zero by this construction: its second clip is another object->intent the
  split does not know about. It is reported, not gated.
- Aligned only uses fine labels that have >= 2 clips; misaligned can use any
  clip of the category. The two bundles therefore draw different source clips
  and different objects, and have different fine-label multiplicities.
- The misaligned label's intent is expected to be nearly uninformative about
  pose (v2 caveat, screen ``right_only``): "label does not describe the window"
  and "label is uninformative" remain inseparable.
- Two clips, not three: trajectories are shorter than v2's, so this pair is not
  comparable in length with ``pc_oakink_long3_*``.

    python scripts/build_oakink_alignment_v4.py build
    python scripts/build_oakink_alignment_v4.py screen

Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402

import build_oakink_alignment_v2 as V2  # noqa: E402
from experiment_paired_composition import coarsen_labels  # noqa: E402

OUT_A = V2.OUT_DIR / "oakink_pair_aligned.npz"
OUT_M = V2.OUT_DIR / "oakink_pair_misaligned.npz"
REPORT = ROOT / "runs" / "build_oakink_alignment_v4.json"
FACTOR = 2


def cat_of(c: str) -> str:
    return c.partition("->")[0]


def int_of(c: str) -> str:
    return c.partition("->")[2]


def pairs_aligned(fine: list[str], rng: np.random.Generator) -> list[list[int]]:
    by: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(fine):
        by[f].append(i)
    out = []
    for f in sorted(by):
        clips = [int(x) for x in rng.permutation(by[f])]
        for j in range(len(clips) // FACTOR):
            out.append(clips[j * FACTOR:(j + 1) * FACTOR])
    assert len(out) == sum(len(v) // FACTOR for v in by.values())
    return out


def pairs_misaligned_cat(clips_by_intent: dict[str, list[int]], rng: np.random.Generator) -> list[list[int]]:
    """All different-intent pairs a category can give: repeatedly take one clip
    from each of the two largest remaining intents (ties at random). Reaches
    min(floor(n/2), n - c_max), asserted. Which of the two is first (the label
    clip) is random."""
    queues = {i: [int(x) for x in rng.permutation(v)] for i, v in clips_by_intent.items()}
    n = sum(len(v) for v in queues.values())
    bound = min(n // 2, n - max(len(v) for v in queues.values()))
    out = []
    while True:
        ks = [i for i in sorted(queues) if queues[i]]
        if len(ks) < 2:
            break
        tie = rng.permutation(len(ks))
        r = [ks[j] for j in sorted(range(len(ks)), key=lambda j: (-len(queues[ks[j]]), tie[j]))]
        p = [queues[r[0]].pop(), queues[r[1]].pop()]
        out.append(p if int(rng.integers(2)) == 0 else p[::-1])
    assert len(out) == bound, f"greedy made {len(out)}, bound {bound}"
    return out


def pairs_misaligned(coarse: list[str], target: Counter, rng: np.random.Generator):
    by: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(coarse):
        if cat_of(c) in target:
            by[cat_of(c)][int_of(c)].append(i)
    avail = {c: pairs_misaligned_cat(by[c], rng) if len(by[c]) >= 2 else [] for c in sorted(target)}
    take = {c: min(len(avail[c]), target[c]) for c in avail}
    short = sum(target.values()) - sum(take.values())
    # Close a shortfall from categories with spare pairs, always the one whose
    # count is currently least above its aligned count (ties alphabetical).
    while short > 0:
        spare = [c for c in sorted(avail) if len(avail[c]) > take[c]]
        if not spare:
            break
        c = min(spare, key=lambda c: (take[c] - target[c], c))
        take[c] += 1
        short -= 1
    out = []
    for c in sorted(avail):
        sel = np.sort(rng.choice(len(avail[c]), take[c], replace=False)) if take[c] else []
        out += [avail[c][int(j)] for j in sel]
    return out, {c: len(avail[c]) for c in avail}, short


def make_bundle(b, pairs, name, rule, extra):
    return V2.make_bundle(b, pairs, name, rule, {
        "ablation": "label/window alignment v4: two clips, constituent-clean aligned half",
        "builder": "scripts/build_oakink_alignment_v4.py", "concat_factor": FACTOR, **extra})


def check_rules(b, coarse, out, pairs, aligned: bool) -> None:
    used = Counter(i for t in pairs for i in t)
    assert max(used.values()) == 1, "a source clip is used more than once"
    assert all(len(t) == FACTOR for t in pairs)
    for t, lab in zip(pairs, out.labels):
        assert lab == b.labels[t[0]]
        assert cat_of(coarse[t[0]]) == cat_of(coarse[t[1]]), "clips must share the category"
        if aligned:
            assert b.labels[t[0]] == b.labels[t[1]], "aligned: both clips carry the label's object->intent"
        else:
            assert int_of(coarse[t[0]]) != int_of(coarse[t[1]]), "misaligned: 2nd intent must differ"
    owner: dict[bytes, int] = {}
    for j, tr in enumerate(out.trajectories):
        for k in V2.frame_keys(tr):
            assert owner.setdefault(k, j) == j, f"frame shared by trajectories {owner[k]} and {j}"


def run_build() -> int:
    b = TrajectoryBundle.load(V2.SRC)
    fine = list(b.labels)
    coarse = coarsen_labels(fine, V2.GRANULARITY)
    n = len(fine)
    rng = np.random.default_rng(V2.SEED)

    pa = pairs_aligned(fine, rng)
    per_cat_a = Counter(cat_of(coarse[t[0]]) for t in pa)
    pm, avail_m, short = pairs_misaligned(coarse, per_cat_a, rng)
    pa = [pa[i] for i in rng.permutation(len(pa))]
    pm = [pm[i] for i in rng.permutation(len(pm))]
    per_cat_m = Counter(cat_of(coarse[t[0]]) for t in pm)

    out_a = make_bundle(b, pa, "oakink_pair_aligned",
                        "both clips carry the same object->intent; label = it", {})
    out_m = make_bundle(b, pm, "oakink_pair_misaligned",
                        "first clip's object->intent; 2nd clip shares its category and has a "
                        "different intent; categories restricted to oakink_pair_aligned's",
                        {"matched_categories": sorted(per_cat_a)})
    check_rules(b, coarse, out_a, pa, aligned=True)
    check_rules(b, coarse, out_m, pm, aligned=False)
    assert set(per_cat_m) <= set(per_cat_a)
    for out, path in ((out_a, OUT_A), (out_m, OUT_M)):
        out.save(path)
        back = TrajectoryBundle.load(path)
        assert list(back.labels) == list(out.labels)
        assert all(np.array_equal(x, y) for x, y in zip(back.trajectories, out.trajectories))

    in_cats = sum(cat_of(c) in per_cat_a for c in coarse)
    diffs = {c: per_cat_m[c] - per_cat_a[c] for c in sorted(per_cat_a)}
    report = {
        "source": {"n_traj": n, "fine_labels": len(set(fine)),
                   "fine_labels_with_2plus_clips": sum(v >= 2 for v in Counter(fine).values())},
        "aligned": V2.stats(out_a, n, pa),
        "misaligned": V2.stats(out_m, n, pm),
        "misaligned_shortfall_vs_aligned": short,
        "misaligned_source_clips_in_matched_categories": in_cats,
        "misaligned_clips_unused_within_matched_categories": in_cats - 2 * len(pm),
        "per_category": {c: {"aligned": per_cat_a[c], "misaligned": per_cat_m[c],
                             "misaligned_available": avail_m[c]} for c in sorted(per_cat_a)},
        "largest_per_category_difference": max(abs(v) for v in diffs.values()),
        "categories_with_difference": {c: v for c, v in diffs.items() if v},
        "shared_source_clips_between_bundles": len({i for t in pa for i in t} & {i for t in pm for i in t}),
    }

    # Constituent-level check. (a) must be exactly zero for both arms of the
    # aligned bundle; for the misaligned bundle it is recorded only.
    import check_constituent_leak as C
    failed = []
    leak = {}
    for key, path in (("aligned", OUT_A), ("misaligned", OUT_M)):
        try:
            m = C.measure(path, range(10))
            leak[key] = m["summary_mean_over_seeds"]
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for aligned below
            leak[key] = {"error": f"{type(exc).__name__}: {exc}"}
            failed.append(f"{key}: {leak[key]['error']}")
    report["constituent_leak_seeds_0_9"] = leak
    report["failed_assertions"] = failed
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    a = leak["aligned"]
    assert "error" not in a, a
    assert a["naive.a_target_traj_exposed.max"] == 0.0 and a["informed.a_target_traj_exposed.max"] == 0.0, \
        f"aligned pair bundle has constituent exposure: {a}"
    print("ASSERTED: aligned (a) == 0 for both arms on seeds 0-9")
    return 0


def run_screen(out: Path) -> int:
    from screen_axes import screen

    cfg = V2.SCREEN_CFG
    rows = []
    for name in ("oakink", "oakink_pair_aligned", "oakink_pair_misaligned"):
        r = screen(V2.OUT_DIR / f"{name}.npz", V2.GRANULARITY, cfg["window"], cfg["stride"],
                   cfg["max_per_traj"], cfg["n_perm"], cfg["seed"])
        if r is None or "error" in r:
            print(f"{name}: {r}")
            return 1
        r["bundle"] = name
        rows.append(r)
        print(f"{name:<24} traj {r['n_traj']:>4} cells {r['n_cells']:>4}  "
              f"left {r['left_only']:.4f}  right {r['right_only']:.4f}  "
              f"excess {r['excess']:+.4f}  z {r['z']:.2f}")
    ref = rows[0]
    ok = abs(ref["excess"] - V2.REFERENCE_EXCESS) < 1e-6 and abs(ref["z"] - V2.REFERENCE_Z) < 1e-3
    print(f"reference reproduction: {'MATCH' if ok else 'MISMATCH'}")
    out.write_text(json.dumps({
        "config": {**cfg, "granularity": V2.GRANULARITY},
        "reference": {"excess": V2.REFERENCE_EXCESS, "z": V2.REFERENCE_Z, "reproduced": bool(ok)},
        "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "screen"))
    ap.add_argument("--out", default="runs/axis_screen_alignment_v4.json")
    args = ap.parse_args()
    return run_screen(ROOT / args.out) if args.mode == "screen" else run_build()


if __name__ == "__main__":
    raise SystemExit(main())
