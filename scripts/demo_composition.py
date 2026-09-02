"""Are the learned primitives actually reusable bricks?

MotionBricks' claim is not that its latent space is modular -- it is that you
can *assemble* motion from named units, plug-and-play, without retraining. A
model can satisfy every modularity regulariser (12 primitives in use, balanced,
segmented assignments) and still fail that claim, if primitive k means one
thing after primitive 3 and something unrelated after primitive 7. Then it is a
context-dependent code, not a brick.

So this script tests the property that makes composition an *interface* rather
than an artefact of the bottleneck:

1. **Composition validity.** Decode many primitive sequences, including
   combinations never seen in training, and score the motion on the same
   ergonomic axes as the training data (joint limits, DIP/PIP coupling,
   smoothness, mesh self-intersection). Unseen compositions must not be worse
   than seen ones -- otherwise "plug-and-play" is false.

2. **Primitive identity.** Decode the same primitive under many different
   predecessors and initial poses. If it is a reusable unit, the motion it
   produces should cluster by primitive identity. Quantified as a
   between-primitive / within-primitive variance ratio: >1 means identity
   dominates context, <1 means context dominates and the "library" is an
   illusion.

3. **Segmentation fidelity.** Encode real trajectories and check whether the
   assignment boundaries land where the motion actually changes, rather than at
   arbitrary points. Measured as the ratio of joint-angle speed at assignment
   switches to speed elsewhere.

Check 2 is the one with teeth. Nothing in the training objective asks for
context-independence, so passing it is evidence and failing it is a real
negative result about this architecture.

    python scripts/demo_composition.py --run-dir runs/prior_oakink_modular
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.pipeline import WindowedTrajectoryDataset  # noqa: E402
from caredex.hand_model import ARTICULATED_SLICE, denormalize  # noqa: E402
from caredex.models.modular_prior import ModularConfig, ModularPrimitivePrior  # noqa: E402
from caredex.train.checkpoint import CheckpointManager  # noqa: E402
from caredex.validate.ergonomics import validate_poses  # noqa: E402


def load_model(run_dir: Path, checkpoint: str, device: torch.device):
    mgr = CheckpointManager(run_dir)
    path = run_dir / checkpoint
    cfg_dict = mgr.peek(path).get("config", {}).get("model", {})
    cfg = ModularConfig(**{
        k: v for k, v in cfg_dict.items() if k in ModularConfig.__dataclass_fields__
    })
    model = ModularPrimitivePrior(cfg)
    mgr.load(model, path=path, restore_rng=False)
    return model.to(device).eval(), mgr.peek(path)


@torch.no_grad()
def observed_transitions(model, bundle: TrajectoryBundle, window: int, device, cap: int = 400):
    """Which ordered primitive pairs the model actually uses on real data."""
    ds = WindowedTrajectoryDataset(bundle.trajectories, window, stride=window, labels=bundle.labels)
    idx = np.linspace(0, len(ds) - 1, min(cap, len(ds))).astype(int)
    batch = torch.stack([ds[i] for i in idx]).to(device)
    weights = model.assign(model.encode(batch)["logits"])
    hard = weights.argmax(-1).cpu().numpy()

    seen = set()
    for row in hard:
        runs = [row[0]]
        for a, b in zip(row[:-1], row[1:]):
            if a != b:
                runs.append(b)
        seen.update(zip(runs[:-1], runs[1:]))
    return seen, hard


@torch.no_grad()
def check_composition_validity(model, bundle, seen_pairs, device, n=192, seed=0):
    """Ergonomics of decoded sequences, split by whether the pair was ever used."""
    rng = np.random.default_rng(seed)
    k = model.cfg.n_primitives
    all_pairs = [(a, b) for a in range(k) for b in range(k) if a != b]
    unseen = [p for p in all_pairs if p not in seen_pairs]
    if not unseen:
        return None, "every primitive pair already occurs in the data; no unseen compositions to test"

    out = {}
    for label, pairs in (("seen", sorted(seen_pairs)), ("unseen", unseen)):
        if not pairs:
            continue
        picks = [pairs[i] for i in rng.integers(0, len(pairs), size=n)]
        windows = []
        for a, b in picks:
            z = model.sample(1, temperature=1.0, device=device, sequence=[int(a), int(b)])
            windows.append(z[0].cpu().numpy())
        native = denormalize(np.stack(windows))
        out[label] = validate_poses(
            native, fps=bundle.fps, label=f"{label} compositions",
            limits_are_structural=model.cfg.bounded_output,
        )
    return out, None


@torch.no_grad()
def check_primitive_identity(model, device, n_contexts=24, seed=0):
    """Between-primitive vs within-primitive variance of decoded motion.

    Each primitive is decoded after many different predecessors and from many
    different initial poses. A reusable brick produces motion that clusters by
    identity; a context-dependent code does not.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    k = model.cfg.n_primitives
    rng = np.random.default_rng(seed)

    per_primitive = []
    for target in range(k):
        reps = []
        for _ in range(n_contexts):
            pred = int(rng.integers(0, k))
            init = torch.from_numpy(
                rng.uniform(-0.6, 0.6, size=(1, 27)).astype(np.float32)
            ).to(device)
            w = model.sample(
                1, initial_pose=init, temperature=1.0, device=device,
                generator=g, sequence=[pred, target],
            )[0]
            # Score only the second half -- the part the target primitive drives.
            half = w[w.shape[0] // 2 :, ARTICULATED_SLICE]
            # Velocity profile, which is what a "motion primitive" should own;
            # absolute pose is dominated by the random initial condition.
            reps.append((half[1:] - half[:-1]).cpu().numpy().ravel())
        per_primitive.append(np.stack(reps))

    arr = np.stack(per_primitive)                      # (k, n_contexts, D)
    centroids = arr.mean(axis=1)                       # (k, D)
    grand = centroids.mean(axis=0)
    between = float(np.mean(np.sum((centroids - grand) ** 2, axis=-1)))
    within = float(np.mean(np.sum((arr - centroids[:, None]) ** 2, axis=-1)))
    return {
        "between_primitive_variance": between,
        "within_primitive_variance": within,
        # >1 means primitive identity explains more of the motion than the
        # context it is placed in -- the property that makes it a brick.
        "identity_ratio": between / max(within, 1e-12),
        "n_contexts": n_contexts,
    }


@torch.no_grad()
def check_segmentation(model, bundle, window, device, cap=200):
    """Do assignment switches land where the motion actually changes?"""
    ds = WindowedTrajectoryDataset(bundle.trajectories, window, stride=window, labels=bundle.labels)
    idx = np.linspace(0, len(ds) - 1, min(cap, len(ds))).astype(int)
    batch = torch.stack([ds[i] for i in idx]).to(device)
    hard = model.assign(model.encode(batch)["logits"]).argmax(-1).cpu().numpy()

    native = denormalize(batch.cpu().numpy())[..., ARTICULATED_SLICE]
    speed = np.abs(np.diff(native, axis=1)).mean(-1)          # (B, T-1)
    switch = hard[:, 1:] != hard[:, :-1]                       # (B, T-1)

    if switch.sum() == 0 or (~switch).sum() == 0:
        return {"switch_speed_ratio": float("nan"), "switch_rate": float(switch.mean())}
    return {
        "speed_at_switch": float(speed[switch].mean()),
        "speed_elsewhere": float(speed[~switch].mean()),
        # >1 means boundaries coincide with fast motion, i.e. real transitions.
        "switch_speed_ratio": float(speed[switch].mean() / max(speed[~switch].mean(), 1e-12)),
        "switch_rate": float(switch.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--n-samples", type=int, default=192)
    ap.add_argument("--contexts", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    model, peeked = load_model(run_dir, args.checkpoint, device)
    bundle = TrajectoryBundle.load(args.bundle)
    window = model.cfg.window

    print(f"model    : {run_dir/args.checkpoint}  epoch {peeked.get('state', {}).get('epoch', '?')}")
    print(f"           {model.cfg.n_primitives} primitives, {model.n_parameters:,} parameters")
    print(f"data     : {args.bundle}  {len(bundle.trajectories)} trajectories\n")

    seen_pairs, hard = observed_transitions(model, bundle, window, device)
    k = model.cfg.n_primitives
    print(f"primitive pairs observed on real data: {len(seen_pairs)} of {k * (k - 1)} possible")
    print(f"primitives actually used: {len(np.unique(hard))} of {k}\n")

    report: dict = {"n_primitives": k, "pairs_observed": len(seen_pairs)}

    print("--- 1. composition validity ---")
    val, note = check_composition_validity(model, bundle, seen_pairs, device, args.n_samples)
    if note:
        print(f"  {note}")
    else:
        for label, rep in val.items():
            print(f"  {label:<8} coupling {rep.coupling_residual_mean:6.2f} deg   "
                  f"|v| mean {rep.velocity_mean:7.2f} deg/s   "
                  f"|jerk| p99 {rep.jerk_p99:.2e}   interpen {rep.interpenetration_rate:.2%}")
            report[f"validity_{label}"] = rep.to_dict()
        if "seen" in val and "unseen" in val:
            d = val["unseen"].coupling_residual_mean - val["seen"].coupling_residual_mean
            print(f"  unseen minus seen coupling residual: {d:+.3f} deg "
                  f"({'plug-and-play holds' if abs(d) < 2.0 else 'unseen compositions degrade'})")

    print("\n--- 2. primitive identity (the check with teeth) ---")
    ident = check_primitive_identity(model, device, args.contexts)
    report["identity"] = ident
    print(f"  between-primitive variance : {ident['between_primitive_variance']:.6f}")
    print(f"  within-primitive variance  : {ident['within_primitive_variance']:.6f}")
    print(f"  identity ratio             : {ident['identity_ratio']:.3f}")
    print("  " + (
        "PASS - primitive identity dominates context: these behave as reusable bricks"
        if ident["identity_ratio"] > 1.0 else
        "FAIL - context dominates identity: the bank is a context-dependent code, "
        "not a library, and 'composable primitives' should not be claimed"
    ))

    print("\n--- 3. segmentation fidelity ---")
    seg = check_segmentation(model, bundle, window, device)
    report["segmentation"] = seg
    if np.isfinite(seg.get("switch_speed_ratio", float("nan"))):
        print(f"  speed at switches {seg['speed_at_switch']:.3f} vs elsewhere "
              f"{seg['speed_elsewhere']:.3f}  ratio {seg['switch_speed_ratio']:.3f}")
        print("  " + (
            "boundaries coincide with real motion changes"
            if seg["switch_speed_ratio"] > 1.05 else
            "boundaries are not aligned with motion changes -- segmentation is arbitrary"
        ))
    else:
        print("  no switches observed; the model assigns one primitive per window")

    out = Path(args.out or run_dir / "composition.json")
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
