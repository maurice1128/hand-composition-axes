"""What direct joint-angle retargeting costs, per dataset, with an artefact.

The paper's per-dataset retargeting figures -- clip fractions, penetration rates,
what the abduction correction resolves and at what cost -- were carried in the
draft's prose with nothing under `runs/` behind them, and three of them reach
the abstract. An audit found no file and no entry point producing any of them:
`experiment_robot_transfer.py` only does the seen-versus-unseen split on one
bundle. Whatever produced those numbers is gone.

This regenerates them. If the result disagrees with what the draft says, the
draft is what changes: a number with a file behind it beats a number that was
once true.

Kinematic throughout, as in `robot_hand.py`: poses are placed with `mj_forward`
and contacts are read off. Nothing here measures force.

    python scripts/experiment_retarget_survey.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.robot_hand import PenetrationResolver, ShadowHand  # noqa: E402

DATASETS = {
    "OakInk-Image": "data/bundles/oakink.npz",
    "DexYCB": "data/bundles/dexycb_shape.npz",
    "GRAB": "data/bundles/grab.npz",
}


def colliding_pairs(hand: ShadowHand, qpos: np.ndarray, tol_mm: float,
                    cap: int = 4000) -> Counter:
    """Which body pairs actually overlap, so the failure can be named."""
    mj, tol = hand._mj, tol_mm / 1000.0
    seen: Counter = Counter()
    for frame in qpos[:cap]:
        hand.data.qpos[:] = frame
        mj.mj_forward(hand.model, hand.data)
        for i in range(int(hand.data.ncon)):
            c = hand.data.contact[i]
            if float(c.dist) >= -tol:
                continue
            names = tuple(sorted(
                mj.mj_id2name(hand.model, mj.mjtObj.mjOBJ_BODY,
                              hand.model.geom_bodyid[g]) or f"geom{g}"
                for g in (c.geom1, c.geom2)))
            seen[names] += 1
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trajectories", type=int, default=40,
                    help="per dataset; the draft's figures used 40")
    ap.add_argument("--tol-mm", type=float, default=0.5)
    ap.add_argument("--out", default="runs/retarget_survey.json")
    args = ap.parse_args()

    hand = ShadowHand()
    resolver = PenetrationResolver(hand, tol_mm=args.tol_mm)
    report: dict = {"n_trajectories": args.n_trajectories, "tol_mm": args.tol_mm,
                    "datasets": {}}

    print(f"{'dataset':<14}{'frames':>9}{'clipped':>9}{'penetr.':>9}"
          f"{'after':>9}{'resolved':>10}{'abduction':>11}")
    print("-" * 71)

    for name, rel in DATASETS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"{name:<14} bundle missing: {rel}")
            continue
        bundle = TrajectoryBundle.load(path)
        trajs = bundle.trajectories[: args.n_trajectories]
        q = np.concatenate(trajs, axis=0)

        qpos, clip = hand.retarget(q)
        rep = hand.replay(q, penetration_tol_mm=args.tol_mm)
        _, stats = resolver.resolve_trajectory(q)

        before = stats["penetrating_before"]
        after = stats["penetrating_after"]
        resolved = (before - after) / before if before else float("nan")
        cost = stats.get("mean_abduction_change_deg",
                         stats.get("mean_change_deg", float("nan")))

        pairs = colliding_pairs(hand, qpos, args.tol_mm)
        top = pairs.most_common(6)
        report["datasets"][name] = {
            "n_frames": int(len(q)),
            "clip_fraction": float(rep.clip_fraction),
            "penetrating_before": float(before),
            "penetrating_after": float(after),
            "resolved_fraction": float(resolved),
            "abduction_cost_deg": float(cost),
            "worst_joint_clip": sorted(rep.per_joint_clip.items(),
                                       key=lambda kv: -kv[1])[:4],
            "top_colliding_pairs": [[list(k), int(v)] for k, v in top],
            "penetrating_contacts_total": int(sum(pairs.values())),
        }
        print(f"{name:<14}{len(q):>9,}{rep.clip_fraction:>8.2%}"
              f"{before:>9.1%}{after:>9.1%}{resolved:>9.0%}{cost:>10.2f}°")
        if top:
            a, b = top[0][0]
            print(f"{'':<14}worst pair {a} ~ {b}: {top[0][1]:,} of "
                  f"{sum(pairs.values()):,} penetrating contacts")

    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("These supersede any per-dataset retargeting figure in the draft that "
          "\nno file supports. Where they differ, the draft is what changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
