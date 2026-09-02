"""Do the prior's compositions produce grasps a robot hand can hold?

Every earlier robot measurement here is geometric. This one has forces and a
success rate, which is what a robotics venue asks for and what the paper did not
have.

Four arms, all executed on the same MuJoCo scene:

    fist        a scripted full closure -- positive control, must succeed
    open        the hand held open      -- negative control, must fail
    seen        primitive pairs the prior's assignments contain
    unseen      pairs they never contain

The controls are not decoration. The object size and the success criterion were
*calibrated* against them: a 3.5 cm sphere was launched by the closing fingers
in both control arms, which is a scene that measures nothing. Only the box
separates the controls under this setup; sphere and cylinder roll out of the
hand regardless of what the fingers do, and are excluded rather than reported as
failures.

Success is displacement under 10 cm *and* contact with the hand at the end. Both
are needed: displacement alone passes an object that slid free and stopped
nearby, contact alone passes one brushing a fingertip on the way down.

    python scripts/experiment_grasp_success.py --run-dir runs/bricks_s4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.grasp_task import GraspTask  # noqa: E402
from caredex.hand_model import DOF_NAMES, LIMITS_HI  # noqa: E402

from demo_composition import load_model, observed_transitions  # noqa: E402
from experiment_robot_transfer import decode_pairs  # noqa: E402


def scripted_fist(n_frames: int) -> np.ndarray:
    fist = np.zeros(len(DOF_NAMES))
    for i, name in enumerate(DOF_NAMES):
        if "flex" in name and "wrist" not in name:
            fist[i] = LIMITS_HI[i] * 0.95
    return np.linspace(np.zeros(len(DOF_NAMES)), fist, n_frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/bricks_s4")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--shape", default="box")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/grasp_success.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ROOT / args.run_dir, "best.pt", device)
    bundle = TrajectoryBundle.load(ROOT / args.bundle)
    k, window = model.cfg.n_primitives, model.cfg.window

    seen, _ = observed_transitions(model, bundle, window, device)
    seen_list = sorted(seen)
    unseen = [(a, b) for a in range(k) for b in range(k) if a != b and (a, b) not in seen]
    rng = np.random.default_rng(args.seed)
    n = min(args.n_pairs, len(seen_list), len(unseen))

    task = GraspTask(args.shape, args.difficulty)
    report: dict = {"run_dir": args.run_dir, "shape": args.shape, "difficulty": args.difficulty, "n_per_arm": n}

    ctrl = {"fist": task.run(scripted_fist(window)),
            "open": task.run(np.zeros((window, len(DOF_NAMES))))}
    print(f"controls   fist held={ctrl['fist'].held} drop={ctrl['fist'].drop_m:.3f}  |  "
          f"open held={ctrl['open'].held} drop={ctrl['open'].drop_m:.3f}")
    if not (ctrl["fist"].held and not ctrl["open"].held):
        print("STOP: controls do not separate; the scene measures nothing.")
        return 1
    report["controls"] = {k2: {"held": v.held, "drop_m": v.drop_m} for k2, v in ctrl.items()}

    arms: dict[str, np.ndarray] = {}
    for label, pool in (("seen", seen_list), ("unseen", unseen)):
        pairs = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
        clips = decode_pairs(model, pairs, device)
        res = [task.run(c) for c in clips]
        arms[label] = np.array([r.held for r in res], dtype=float)
        report[label] = {
            "success_rate": float(arms[label].mean()),
            "mean_drop_m": float(np.mean([r.drop_m for r in res])),
            "held": [bool(r.held) for r in res],
        }
        print(f"{label:<8} success {arms[label].mean():.1%}  "
              f"mean drop {report[label]['mean_drop_m']:.3f} m")

    t, p = stats.ttest_ind(arms["seen"], arms["unseen"], equal_var=False)
    odds = stats.fisher_exact([[int(arms["seen"].sum()), int(n - arms["seen"].sum())],
                               [int(arms["unseen"].sum()), int(n - arms["unseen"].sum())]])
    report["p_ttest"], report["p_fisher"] = float(p), float(odds[1])
    print(f"\nseen vs unseen: t-test p = {p:.3f}, Fisher exact p = {odds[1]:.3f}")
    print("\nA null says compositions the prior never saw grasp as well as ones it "
          "\ndid. A significant difference is the more interesting result and must "
          "\nnot be reported as 'no difference'.")

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
