"""Category-matched misaligned bundle for the OakInk-Image alignment test.

The v2 pair (``build_oakink_alignment_v2.py``) differs in more than alignment:
``oakink_long3_aligned`` covers 14 object categories and
``oakink_long3_misaligned`` covers 33. This builds

``oakink_long3_misaligned_cm``  the v2 misaligned construction, verbatim
    (3 clips per trajectory sharing a category; 2nd and 3rd clips with intents
    different from the label's; label = first clip's ``object->intent``; 4-frame
    cross-fade; numpy seed 0; no source clip used twice), but drawing ONLY from
    the categories that occur in ``oakink_long3_aligned.npz``.

Everything that does the construction is imported from v2, not copied:
``triples_misaligned``, ``concat``/``make_bundle``, ``check_rules``, ``stats``.

    python scripts/build_oakink_alignment_v3.py build
    python scripts/build_oakink_alignment_v3.py screen

What this does NOT fix or show
------------------------------
- The category SET is matched; the per-category trajectory counts are not. A
  category's aligned yield is sum over intents of floor(n_ci/3), its misaligned
  yield is min(floor(n/3), n - c_max); the two differ per category.
- The source clips still differ from the aligned bundle's: aligned uses clips in
  same-intent triples, this uses mixed-intent triples from the same categories.
- The rng stream differs from v2's misaligned build. v2 drew the aligned triples
  first from the same generator; here the generator starts at the misaligned
  construction. Same seed (0), different draws.
- If the greedy bound over the matched categories is below 210, the bundle is
  smaller than the aligned one and that is reported, not patched.
- The v2 caveat that a misaligned label is nearly uninformative about intent
  (screen ``right_only``) is a property of the construction and is expected to
  remain. Category matching does not separate "label does not describe the
  window" from "label is uninformative".

Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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

NAME = "oakink_long3_misaligned_cm"
OUT = V2.OUT_DIR / f"{NAME}.npz"
REPORT = ROOT / "runs" / "build_oakink_alignment_v3.json"


def aligned_categories() -> list[str]:
    a = TrajectoryBundle.load(V2.OUT_ALIGNED)
    coarse = coarsen_labels(list(a.labels), V2.GRANULARITY)
    return sorted({c.partition("->")[0] for c in coarse})


def run_build() -> int:
    b = TrajectoryBundle.load(V2.SRC)
    coarse = coarsen_labels(list(b.labels), V2.GRANULARITY)
    n = len(b.trajectories)
    cats = aligned_categories()
    aligned_n = len(TrajectoryBundle.load(V2.OUT_ALIGNED).trajectories)

    # Restrict to the matched categories, run v2's construction on the restricted
    # list, then map its local indices back to indices into oakink.npz.
    keep = [i for i, c in enumerate(coarse) if c.partition("->")[0] in set(cats)]
    sub = [coarse[i] for i in keep]
    rng = np.random.default_rng(V2.SEED)
    tri_local = V2.triples_misaligned(sub, rng)
    triples = [[keep[j] for j in t] for t in tri_local]

    per_cat_src = Counter(c.partition("->")[0] for c in sub)
    per_cat_unc = Counter(coarse[t[0]].partition("->")[0] for t in triples)
    uncapped = len(triples)
    cap_note = "none"
    if uncapped > aligned_n:
        sel = np.sort(rng.choice(uncapped, aligned_n, replace=False))
        triples = [triples[i] for i in sel]
        cap_note = (f"subsampled uniformly at random from {uncapped} to {aligned_n} "
                    f"trajectories to match aligned")
    elif uncapped < aligned_n:
        cap_note = (f"greedy bound over the matched categories is {uncapped}, below the "
                    f"aligned bundle's {aligned_n}; nothing was added to close the gap")
    triples = [triples[i] for i in rng.permutation(len(triples))]

    out = V2.make_bundle(
        b, triples, NAME,
        "first clip's object->intent; all three share its category; 2nd and 3rd have "
        "intents different from the label's; categories restricted to those of "
        "oakink_long3_aligned",
        {"uncapped_sizes": {"misaligned_cm": uncapped, "aligned": aligned_n},
         "size_cap": cap_note, "matched_categories": cats,
         "builder": "scripts/build_oakink_alignment_v3.py",
         "ablation": "label/window alignment v3: category-matched misaligned"})
    V2.check_rules(b, coarse, out, triples, aligned=False)

    got = sorted({c.partition("->")[0] for c in coarsen_labels(list(out.labels), V2.GRANULARITY)})
    assert set(got) <= set(cats), "a trajectory fell outside the matched categories"
    for t in triples:
        assert all(coarse[i].partition("->")[0] in set(cats) for i in t)

    out.save(OUT)
    back = TrajectoryBundle.load(OUT)
    assert list(back.labels) == list(out.labels)
    assert all(np.array_equal(x, y) for x, y in zip(back.trajectories, out.trajectories))

    st = V2.stats(out, n, triples)
    per_cat = Counter(c.partition("->")[0]
                      for c in coarsen_labels(list(out.labels), V2.GRANULARITY))
    a_lab = coarsen_labels(list(TrajectoryBundle.load(V2.OUT_ALIGNED).labels), V2.GRANULARITY)
    per_cat_aligned = Counter(c.partition("->")[0] for c in a_lab)
    used = {i for t in triples for i in t}
    report = {
        "bundle": str(OUT.relative_to(ROOT)),
        "matched_categories": cats,
        "categories_present_in_output": got,
        "categories_missing_from_output": sorted(set(cats) - set(got)),
        "source_clips_in_matched_categories": len(keep),
        "clips_unused_within_matched_categories": len(keep) - len(used),
        "uncapped_size": uncapped, "aligned_size": aligned_n, "size_cap": cap_note,
        "per_category": {c: {"source_clips": per_cat_src[c], "uncapped": per_cat_unc[c],
                             "final": per_cat[c], "aligned_bundle": per_cat_aligned[c]}
                         for c in cats},
        "stats": st,
    }
    print(json.dumps(report, indent=1))
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {OUT}\nwrote {REPORT}")
    return 0


def run_screen(out: Path) -> int:
    from screen_axes import screen

    cfg = V2.SCREEN_CFG
    rows = []
    for name in ("oakink", "oakink_long3_aligned", "oakink_long3_misaligned", NAME):
        r = screen(V2.OUT_DIR / f"{name}.npz", V2.GRANULARITY, cfg["window"], cfg["stride"],
                   cfg["max_per_traj"], cfg["n_perm"], cfg["seed"])
        if r is None or "error" in r:
            print(f"{name}: {r}")
            return 1
        r["bundle"] = name
        rows.append(r)
        print(f"{name:<28} traj {r['n_traj']:>4} cells {r['n_cells']:>4}  "
              f"left {r['left_only']:.4f}  right {r['right_only']:.4f}  "
              f"excess {r['excess']:+.4f}  z {r['z']:.2f}")
    ref = rows[0]
    ok = (abs(ref["excess"] - V2.REFERENCE_EXCESS) < 1e-6
          and abs(ref["z"] - V2.REFERENCE_Z) < 1e-3)
    print(f"reference reproduction: {'MATCH' if ok else 'MISMATCH'}")
    out.write_text(json.dumps({
        "config": {**cfg, "granularity": V2.GRANULARITY},
        "reference": {"excess": V2.REFERENCE_EXCESS, "z": V2.REFERENCE_Z, "reproduced": bool(ok)},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "screen"))
    ap.add_argument("--out", default="runs/axis_screen_alignment_v3.json")
    args = ap.parse_args()
    if args.mode == "screen":
        return run_screen(ROOT / args.out)
    return run_build()


if __name__ == "__main__":
    raise SystemExit(main())
