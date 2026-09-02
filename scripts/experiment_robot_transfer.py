"""Do compositions the prior never saw transfer to a robot hand as well as ones it did?

This is the experiment that joins the two halves of the project. One half shows
a modular prior composes primitive sequences it was never trained on without
degrading -- measured in the 27-DOF anatomical representation. The other half
shows what a trip to the Shadow Hand costs: adjacent-finger interpenetration in
a quarter to three quarters of frames, fixable by a collision-aware abduction
correction at 1-5.5 degrees.

Neither half implies the other. A composition can be ergonomically valid as
human motion and still be unreachable on a robot, because the robot's fingers
are thicker and its thumb has less travel. So the claim worth making is not
"the prior composes" and not "human motion retargets", but:

> unseen primitive compositions reach the robot as well as seen ones.

That is what makes a learned primitive library useful to someone building a
hand controller, and it is a property nothing measured so far guarantees.

Design. Primitive pairs are split by whether they ever occur in the model's own
assignments over real data -- the same `observed_transitions` the composition
demo uses, so "seen" means the same thing in both. Equal numbers of pairs are
drawn from each side, decoded, retargeted, and put through the correction.
Everything downstream is identical between the arms; only the pair's
seen/unseen status differs.

Reported per arm: clip fraction, penetration before and after correction, and
the angular cost the correction incurred. A difference in *any* of those is a
difference in transferability, and the last one matters most -- if unseen
compositions need much larger corrections, the library composes on paper and
not on hardware.

    python scripts/experiment_robot_transfer.py --run-dir runs/prior_oakink2_consist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import N_DOF, denormalize  # noqa: E402
from caredex.robot_hand import PenetrationResolver, ShadowHand  # noqa: E402

from demo_composition import load_model, observed_transitions  # noqa: E402


@torch.no_grad()
def decode_pairs(model, pairs: list[tuple[int, int]], device) -> np.ndarray:
    """Decode each primitive pair to native-unit DOF, ``(n, T, 27)``."""
    out = []
    for a, b in pairs:
        z = model.sample(1, temperature=1.0, device=device, sequence=[int(a), int(b)])
        out.append(z[0].cpu().numpy()[:, :N_DOF])
    return denormalize(np.stack(out))


def arm(resolver: PenetrationResolver, clips: np.ndarray) -> dict:
    """Retarget and correct every clip; summarise what the trip cost."""
    stats = [resolver.resolve_trajectory(c)[1] for c in clips]
    return {
        "n_clips": len(clips),
        "n_frames": int(sum(s["n_frames"] for s in stats)),
        "clip_fraction": float(np.mean([s["clip_fraction"] for s in stats])),
        "penetrating_before": float(np.mean([s["penetrating_before"] for s in stats])),
        "penetrating_after": float(np.mean([s["penetrating_after"] for s in stats])),
        "abduction_change_deg": float(np.mean([s["mean_abduction_change_deg"] for s in stats])),
        # Per-clip, so the two arms can be compared with a test rather than by
        # eye. The paired design is not available here -- a pair is seen or
        # unseen, never both -- so these are independent samples.
        "per_clip_after": [s["penetrating_after"] for s in stats],
        "per_clip_change": [s["mean_abduction_change_deg"] for s in stats],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/prior_oakink2_consist")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--n-pairs", type=int, default=48,
                    help="pairs drawn per arm; both arms get the same number")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/robot_transfer.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.run_dir), args.checkpoint, device)
    bundle = TrajectoryBundle.load(ROOT / args.bundle)
    k = model.cfg.n_primitives

    seen_pairs, _ = observed_transitions(model, bundle, model.cfg.window, device)
    all_pairs = [(a, b) for a in range(k) for b in range(k) if a != b]
    unseen_pairs = [p for p in all_pairs if p not in seen_pairs]
    seen_list = sorted(seen_pairs)
    if not seen_list or not unseen_pairs:
        print(f"cannot split: {len(seen_list)} seen, {len(unseen_pairs)} unseen")
        return 1

    rng = np.random.default_rng(args.seed)
    n = min(args.n_pairs, len(seen_list), len(unseen_pairs))
    draw = lambda pool: [pool[i] for i in rng.choice(len(pool), n, replace=False)]  # noqa: E731

    print(f"{k} primitives | {len(seen_list)} pairs seen, {len(unseen_pairs)} unseen "
          f"| drawing {n} from each\n")

    hand = ShadowHand()
    resolver = PenetrationResolver(hand)
    report = {"run_dir": args.run_dir, "n_primitives": k,
              "pairs_seen": len(seen_list), "pairs_unseen": len(unseen_pairs)}

    for label, pool in (("seen", seen_list), ("unseen", unseen_pairs)):
        clips = decode_pairs(model, draw(pool), device)
        report[label] = arm(resolver, clips)
        r = report[label]
        print(f"{label:<8} {r['n_clips']} clips, {r['n_frames']:,} frames | "
              f"clipped {r['clip_fraction']:.2%} | penetrating "
              f"{r['penetrating_before']:.1%} -> {r['penetrating_after']:.1%} | "
              f"abduction {r['abduction_change_deg']:.2f} deg")

    from scipy import stats as st

    print()
    for key, name in (("per_clip_after", "penetration after correction"),
                      ("per_clip_change", "abduction change needed")):
        a = np.array(report["seen"][key])
        b = np.array(report["unseen"][key])
        t, p = st.ttest_ind(a, b, equal_var=False)
        report[f"{key}_p"] = float(p)
        verdict = ("no detectable difference" if p >= 0.05 else
                   "unseen is WORSE" if b.mean() > a.mean() else "unseen is better")
        print(f"  {name:<32} seen {a.mean():.4f}  unseen {b.mean():.4f}  "
              f"p={p:.3f}  {verdict}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nThe claim this supports, if both tests come back null: a learned "
          "\nprimitive library composes to sequences it never saw AND those "
          "\nsequences reach a robot hand no worse than the ones it did. A "
          "\nsignificant difference in either direction is the more interesting "
          "\nresult and must not be reported as 'no difference'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
