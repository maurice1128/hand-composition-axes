"""Build the TACO bundle, its three two-factor axis bundles, and the build report.

Why a named entry point: ``runs/PREREG_taco_prediction.md`` commits to recording
clip durations, the inferred conventions, DOF health and each candidate axis's
label grid **before any model is trained**, and to choosing the confirmatory
axis by a fixed rule. ``runs/taco_build.json`` is that record. It is written
from the data alone; nothing here trains or reads a training result.

The three axis bundles reuse a label workaround instead of adding a granularity
mode to the experiment: ``coarsen_labels(..., "oakink2_scene_verb")`` reads a
label shaped ``<fine>@<scene>@<subject>@<verb>`` and returns ``scene->verb``.
Writing ``<action>|<tool>|<object>@<left>@-@<right>`` therefore yields the cell
``left->right`` with the whole string as the fine label, and no experiment code
changes. The same workaround carried OakInk-Image category x subject and was
validated by ``scripts/verify_category_subject_split.py``. It is re-checked here
for every label, not assumed: the coarse label must equal ``left->right`` and
``transitions_of`` must yield exactly one pair.

    python scripts/build_taco_bundle.py
    python scripts/build_taco_bundle.py --from-bundle      # re-derive axes + report only

What this does NOT do
---------------------
* The "candidate pool" counted here is the number of cells spanned by at least
  ``--min-chains`` distinct triplets. It says a split *can* be drawn, not that
  the informed arm will be adequately covered; that is
  ``check_informed_coverage.py``'s job.
* DOF health is the repo's ``check_dof_health.audit``: pinned more than 25% of
  frames, or normalised sd below 0.01. It does not show that a healthy DOF is
  *correct*, only that it is neither constant nor saturated.
* The duration statistics cannot tell one long action from several short ones.
  The pre-registration's premise ("one action per sequence") is only checked
  against the 10 s threshold it names.
* The triplet field order (action, tool, object) is taken from the directory
  names. The per-position vocabularies are printed and stored so a reader can
  confirm the first position holds verbs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.mano_retarget import MANO_UNAVAILABLE_DOF  # noqa: E402

FIELDS = ("action", "tool", "object")
#: In the order the pre-registration fixes.
AXES = (("action", "tool"), ("action", "object"), ("tool", "object"))
GRANULARITY = "oakink2_scene_verb"


def axis_labels(fine: list[str], left: str, right: str) -> list[str]:
    li, ri = FIELDS.index(left), FIELDS.index(right)
    out = []
    for f in fine:
        parts = f.split("|")
        out.append(f"{f}@{parts[li]}@-@{parts[ri]}")
    return out


def verify_workaround(labels: list[str], left: str, right: str) -> dict:
    """The coarse label must be ``left->right`` and carry exactly one pair."""
    from experiment_paired_composition import coarsen_labels, transitions_of

    li, ri = FIELDS.index(left), FIELDS.index(right)
    coarse = coarsen_labels(labels, GRANULARITY)
    bad = 0
    for lab, c in zip(labels, coarse):
        parts = lab.split("@")[0].split("|")
        pairs = list(transitions_of(c))
        if c != f"{parts[li]}->{parts[ri]}" or pairs != [(parts[li], parts[ri])]:
            bad += 1
    if bad:
        raise RuntimeError(f"{left} x {right}: {bad} labels do not coarsen to one left->right pair")
    return {"labels_checked": len(labels), "mismatches": bad, "example": [labels[0], coarse[0]]}


def grid_stats(fine: list[str], left: str, right: str, min_chains: int) -> dict:
    li, ri = FIELDS.index(left), FIELDS.index(right)
    trajs, groups = Counter(), {}
    for f in fine:
        p = f.split("|")
        cell = (p[li], p[ri])
        trajs[cell] += 1
        groups.setdefault(cell, set()).add(f)
    pool = sorted(c for c, g in groups.items() if len(g) >= min_chains)
    n_left, n_right = len({c[0] for c in trajs}), len({c[1] for c in trajs})
    return {
        "cells": len(trajs),
        "grid": [n_left, n_right],
        "occupancy": len(trajs) / (n_left * n_right),
        "fine_groups": len(set(fine)),
        "trajectories_per_cell_median": float(np.median(list(trajs.values()))),
        "fine_groups_per_cell_max": max(len(g) for g in groups.values()),
        "min_chains": min_chains,
        "candidate_pool": len(pool),
        "candidate_pool_cells": [f"{a}->{b} ({len(groups[(a, b)])} triplets, {trajs[(a, b)]} traj)"
                                 for a, b in pool],
        "trajectories_in_candidate_cells": int(sum(trajs[c] for c in pool)),
    }


def dof_health(path: Path) -> dict:
    from check_dof_health import DEAD_SD, PINNED_FRACTION, audit

    rows, meta = audit(path)
    expected_dead = set(MANO_UNAVAILABLE_DOF) if meta["mano_derived"] else set()
    bad = [r for r in rows if r["dof"] not in expected_dead
           and (r["pinned"] > PINNED_FRACTION or r["sd"] < DEAD_SD)]
    return {
        "n_dof": len(rows),
        "n_unusable": len(bad),
        "unusable": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in bad],
        "expected_dead_not_counted": sorted(expected_dead),
        "post_clamp_fraction_at_a_limit": float(np.mean([r["pinned"] for r in rows
                                                         if r["dof"] not in expected_dead])),
        "thresholds": {"pinned_fraction": PINNED_FRACTION, "dead_sd": DEAD_SD},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"D:\datasets\taco\Hand_Poses.zip")
    ap.add_argument("--out", default="data/bundles/taco.npz")
    ap.add_argument("--report", default="runs/taco_build.json")
    ap.add_argument("--pose-mean", choices=("infer", "add", "flat"), default="infer")
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--min-chains", type=int, default=4)
    ap.add_argument("--max-sequences", type=int, default=None)
    ap.add_argument("--no-compare-hands", action="store_true",
                    help="skip reading the left hand (halves I/O, loses the weak handedness flag)")
    ap.add_argument("--from-bundle", action="store_true",
                    help="skip the zip; rebuild axis bundles and the report from --out")
    args = ap.parse_args()

    out = ROOT / args.out
    t0 = time.time()
    if args.from_bundle:
        bundle = TrajectoryBundle.load(out)
    else:
        from caredex.data.taco import TacoSource

        bundle = TacoSource(
            root=args.root, pose_mean=args.pose_mean, min_frames=args.min_frames,
            max_sequences=args.max_sequences, compare_hands=not args.no_compare_hands,
        ).load()
        bundle.save(out)
    minutes = (time.time() - t0) / 60

    fine = list(bundle.labels)
    meta = bundle.meta
    lengths = np.array([len(t) for t in bundle.trajectories])
    fps = bundle.fps

    vocab = {f: Counter(l.split("|")[i] for l in fine) for i, f in enumerate(FIELDS)}
    dropped = meta.get("dropped", {})
    n_found = meta.get("format", {}).get("n_sequences_found", len(fine))

    axes = {}
    for left, right in AXES:
        name = f"taco_{left}_{right}"
        labels = axis_labels(fine, left, right)
        check = verify_workaround(labels, left, right)
        path = out.with_name(f"{name}.npz")
        TrajectoryBundle(
            trajectories=bundle.trajectories, fps=fps, labels=labels,
            meta={**{k: v for k, v in meta.items() if k not in ("right_vs_left_motion",)},
                  "axis": f"{left} x {right}", "label_format": f"<fine>@<{left}>@-@<{right}>",
                  "read_with_granularity": GRANULARITY},
        ).save(path)
        axes[f"{left}_x_{right}"] = {
            "bundle": str(path.relative_to(ROOT)).replace("\\", "/"),
            "granularity": GRANULARITY,
            "workaround_check": check,
            **grid_stats(fine, left, right, args.min_chains),
        }

    ratio = np.array([r for r in meta.get("right_vs_left_motion", []) if r is not None and np.isfinite(r)])
    seconds = lengths / fps
    report = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "build_minutes": round(minutes, 1),
        "source_zip": args.root,
        "sequences": {
            "found": n_found,
            "kept": len(fine),
            "dropped": {k: {"n": len(v), "examples": v[:5]} for k, v in dropped.items()},
        },
        "frames": {
            "total": int(lengths.sum()), "fps": fps,
            "median": float(np.median(lengths)), "min": int(lengths.min()), "max": int(lengths.max()),
            "median_s": float(np.median(seconds)), "min_s": float(seconds.min()),
            "max_s": float(seconds.max()),
            "fraction_longer_than_10s": float(np.mean(seconds > 10.0)),
            "prereg_premise_median_le_10s": bool(np.median(seconds) <= 10.0),
        },
        "format_as_found": meta.get("format", {}),
        "pose_mean": meta.get("pose_mean", {}),
        "conventions": {k: meta[k] for k in (
            "flexion_axis", "flexion_sign", "flexion_axis_dominance",
            "pip_in_range_positive", "pip_in_range_negative",
            "abduction_axis", "abduction_sign", "finger_abd_in_range_chosen",
            "thumb_flexion_axis", "thumb_flexion_sign", "thumb_axis_dominance",
            "thumb_ip_in_range_positive", "thumb_ip_in_range_negative",
            "thumb_abduction_axis", "thumb_abduction_sign", "thumb_abd_in_range_chosen",
        ) if k in meta},
        "projection_residual_deg": meta.get("projection_residual_deg"),
        "limit_clip_fraction_before_clamp": meta.get("limit_clip_fraction"),
        "dof_health": dof_health(out),
        "handedness_flag": {
            "what": "mean |d pose| of the right hand over the left, per sequence; weak flag only",
            "n": int(len(ratio)),
            "median": float(np.median(ratio)) if len(ratio) else None,
            "fraction_right_moves_less_than_half_of_left": float(np.mean(ratio < 0.5)) if len(ratio) else None,
        },
        "vocabulary": {
            **{f"{f}s": len(vocab[f]) for f in FIELDS},
            "triplets": len(set(fine)),
            **{f"{f}_counts": dict(vocab[f].most_common()) for f in FIELDS},
        },
        "axes": axes,
    }
    rp = ROOT / args.report
    rp.write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    print(f"\nwrote {args.out} (+3 axis bundles) in {minutes:.1f} min; report {args.report}")
    print(f"  sequences   found {n_found}, kept {len(fine)}; dropped "
          f"{ {k: len(v) for k, v in dropped.items() if v} }")
    print(f"  frames      median {np.median(lengths):.0f} ({np.median(seconds):.1f} s), "
          f"min {lengths.min()} ({seconds.min():.1f} s), max {lengths.max()} ({seconds.max():.1f} s); "
          f"{np.mean(seconds > 10):.1%} longer than 10 s")
    print(f"  vocabulary  {len(vocab['action'])} actions, {len(vocab['tool'])} tools, "
          f"{len(vocab['object'])} objects, {len(set(fine))} triplets")
    print(f"  actions     {list(vocab['action'])}")
    h = report["dof_health"]
    print(f"  DOF health  {h['n_unusable']} of {h['n_dof']} unusable: {[r['dof'] for r in h['unusable']]}")
    print(f"  clip        {meta.get('limit_clip_fraction', float('nan')):.2%} outside a limit before clamping")
    for k, a in axes.items():
        print(f"  {k:<16} {a['cells']} cells ({a['occupancy']:.0%} of {a['grid'][0]}x{a['grid'][1]}), "
              f"{a['fine_groups']} fine groups, candidate pool {a['candidate_pool']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
