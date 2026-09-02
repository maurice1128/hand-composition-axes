"""Do unseen compositions transfer, on the pipeline the field actually uses?

`experiment_robot_transfer.py` answered this by copying joint angles onto the
Shadow Hand. That comparison is internally valid -- both arms get the same
transfer -- but its absolute numbers describe a method nobody uses, and the
limitations section says so. This runs the same comparison through
dex-retargeting's DexPilotOptimizer, which is the method people use.

Three processes, because the retargeter needs Pinocchio and the prior needs the
venv's cu128 torch, and those cannot share an interpreter:

    this script (venv)     decode primitive pairs -> 21 MANO keypoints -> .npy
    dexpilot_bridge (conda) DexPilot fit          -> joint angles      -> .npy
    this script (venv)     map to MuJoCo qpos, measure penetration, test

Keypoint order is MANO's 21: wrist, then thumb, index, middle, ring, pinky, four
joints each ending at the tip. The optimizer's own
``target_link_human_indices`` confirms it -- destinations 4, 8, 12, 16, 20 are
the five fingertips and the last five origins are 0, the wrist, which is
DexPilot's five wrist-to-fingertip plus ten inter-fingertip vectors.

    python scripts/experiment_dexpilot_transfer.py --run-dir runs/bricks_s4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.kinematics import all_chains  # noqa: E402
from caredex.robot_hand import ShadowHand  # noqa: E402

from demo_composition import load_model, observed_transitions  # noqa: E402
from experiment_robot_transfer import decode_pairs  # noqa: E402

DEXRET_PYTHON = Path(r"C:\Users\maurice\miniforge3\envs\dexret2\python.exe")

#: MANO's 21-keypoint order after the wrist.
_DIGIT_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def keypoints_21(q_deg: np.ndarray) -> np.ndarray:
    """``(T, 27)`` anatomical pose -> ``(T, 21, 3)`` MANO-ordered keypoints.

    ``all_chains`` returns four joints per digit ending at the tip, and the
    wrist is the origin of that frame, which is exactly the 21-keypoint layout
    the retargeter indexes into.
    """
    chains = all_chains(np.asarray(q_deg, dtype=np.float64))
    n = len(q_deg)
    out = np.zeros((n, 21, 3))
    for i, d in enumerate(_DIGIT_ORDER):
        out[:, 1 + 4 * i : 5 + 4 * i, :] = chains[d]
    return out


def match_joints(dex_names: list[str], hand: ShadowHand) -> list[int]:
    """Map dex-retargeting joint names onto MuJoCo qpos addresses.

    The two models name the same joints differently (``WRJ2`` against
    ``rh_WRJ2``), so matching is on the suffix after any prefix, uppercased. A
    name that matches nothing maps to -1 and its angle is dropped rather than
    written to an arbitrary slot.
    """
    lookup = {n.split("_")[-1].upper(): a for n, a in hand.joint_qpos.items()}
    return [lookup.get(n.split("_")[-1].upper(), -1) for n in dex_names]


def penetration(hand: ShadowHand, qpos: np.ndarray, tol_mm: float = 0.5) -> float:
    tol, pen = tol_mm / 1000.0, 0
    for frame in qpos:
        hand.data.qpos[:] = frame
        hand._mj.mj_forward(hand.model, hand.data)
        n = int(hand.data.ncon)
        if n and min(float(hand.data.contact[i].dist) for i in range(n)) < -tol:
            pen += 1
    return pen / max(len(qpos), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/bricks_s4")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dexpilot_transfer.json")
    args = ap.parse_args()

    if not DEXRET_PYTHON.exists():
        print(f"{DEXRET_PYTHON} not found. Run scripts/setup_dexpilot_env.sh first.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ROOT / args.run_dir, "best.pt", device)
    bundle = TrajectoryBundle.load(ROOT / args.bundle)
    k = model.cfg.n_primitives

    seen, _ = observed_transitions(model, bundle, model.cfg.window, device)
    seen_list = sorted(seen)
    unseen = [(a, b) for a in range(k) for b in range(k) if a != b and (a, b) not in seen]
    if not seen_list or not unseen:
        print(f"cannot split: {len(seen_list)} seen, {len(unseen)} unseen")
        return 1

    rng = np.random.default_rng(args.seed)
    n = min(args.n_pairs, len(seen_list), len(unseen))
    print(f"{k} primitives | {len(seen_list)} seen, {len(unseen)} unseen | {n} per arm\n")

    hand = ShadowHand()
    report = {"run_dir": args.run_dir, "n_per_arm": n}
    arms: dict[str, np.ndarray] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for label, pool in (("seen", seen_list), ("unseen", unseen)):
            pairs = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
            clips = decode_pairs(model, pairs, device)
            kp = np.stack([keypoints_21(c) for c in clips])

            src, dst = Path(tmp) / f"{label}_in.npy", Path(tmp) / f"{label}_out.npy"
            np.save(src, kp)
            print(f"[{label}] retargeting {kp.shape[0]} clips x {kp.shape[1]} frames")
            proc = subprocess.run(
                [str(DEXRET_PYTHON), str(ROOT / "scripts" / "dexpilot_bridge.py"),
                 str(src), str(dst)],
                capture_output=True, text=True)
            if proc.returncode != 0 or not dst.exists():
                print(proc.stdout[-1500:] + proc.stderr[-1500:])
                return 1

            angles = np.load(dst)
            names = json.loads(dst.with_suffix(".joints.json").read_text(encoding="utf-8"))
            addr = match_joints(names, hand)
            if sum(a >= 0 for a in addr) < 20:
                print(f"only {sum(a >= 0 for a in addr)} of {len(names)} joints matched "
                      f"MuJoCo: {names[:6]}")
                return 1
            rates = []
            for clip in angles:
                qpos = np.zeros((len(clip), hand.model.nq))
                for j, a in enumerate(addr):
                    if a >= 0:
                        qpos[:, a] = clip[:, j]
                rates.append(penetration(hand, qpos))
            arms[label] = np.array(rates)
            report[label] = {"penetrating_mean": float(np.mean(rates)),
                             "per_clip": [float(r) for r in rates]}
            print(f"[{label}] penetrating {np.mean(rates):.1%}\n")

    t, p = stats.ttest_ind(arms["seen"], arms["unseen"], equal_var=False)
    report["p_penetration"] = float(p)
    print(f"penetration after DexPilot retargeting: seen {arms['seen'].mean():.3f}  "
          f"unseen {arms['unseen'].mean():.3f}  p = {p:.3f}")
    print("\nA null here says compositions the prior never saw reach the robot as "
          "\nwell as ones it did, on the pipeline the field uses. A significant "
          "\ndifference is the more interesting result and must not be reported "
          "\nas 'no difference'.")

    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
