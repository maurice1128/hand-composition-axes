"""Ergonomic validation of a trained hand prior.

Samples the prior's latent space and scores the decoded windows against the
training data on four axes: joint limits, DIP/PIP coupling, self-intersection,
and smoothness. The data's own scores are the reference -- a prior cannot be
expected to beat the motion it was fit to.

Two of the four checks are weaker than they look and the report says so:
joint limits are structurally enforced by the tanh output, and
self-intersection is a capsule-model proxy. The checks with teeth are the
coupling residual and the smoothness statistics, neither of which is enforced
anywhere in the architecture.

Sampling z ~ N(0, I) rather than encoding real data is deliberate: Phase 4's RL
policy will explore the whole latent space, so a prior that only behaves on the
data manifold is not usable as an action space.

    python scripts/validate_prior.py --run-dir runs/prior_monolithic
    python scripts/validate_prior.py --run-dir runs/prior_modular --model modular
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.config import get, load_config  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.models.latent_prior import LatentActionPrior, PriorConfig  # noqa: E402
from caredex.models.modular_prior import ModularConfig, ModularPrimitivePrior  # noqa: E402
from caredex.train.checkpoint import CheckpointManager  # noqa: E402
from caredex.validate.ergonomics import compare, validate_poses  # noqa: E402


def windows_from_bundle(bundle: TrajectoryBundle, window: int, n: int, seed: int) -> np.ndarray:
    """Sample ``n`` random native-unit windows from the data, for reference."""
    rng = np.random.default_rng(seed)
    usable = [t for t in bundle.trajectories if len(t) >= window]
    if not usable:
        raise ValueError(f"no trajectory is {window} frames long")
    out = np.empty((n, window, usable[0].shape[1]), dtype=np.float32)
    for i in range(n):
        tr = usable[rng.integers(len(usable))]
        s = int(rng.integers(0, len(tr) - window + 1))
        out[i] = tr[s : s + window]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", choices=("monolithic", "modular"), default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--temperatures", type=float, nargs="*", default=[0.5, 1.0, 1.5])
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--mano",
        default=r"D:\datasets\mano\MANO_RIGHT.pkl",
        help="MANO model for mesh-level self-intersection; skipped if absent",
    )
    ap.add_argument("--mesh-frames", type=int, default=128)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        print(f"no checkpoint at {ckpt_path}")
        return 1

    mgr = CheckpointManager(run_dir)
    peeked = mgr.peek(ckpt_path)
    saved_cfg = peeked.get("config", {})
    model_cfg = saved_cfg.get("model", {})
    data_meta = saved_cfg.get("data", {})

    # Prefer the checkpoint's own record of what it is; a mismatch between the
    # flag and the file would silently score the wrong architecture.
    kind = args.model or data_meta.get("model_kind")
    if kind is None:
        kind = "modular" if "n_primitives" in model_cfg else "monolithic"

    if kind == "modular":
        model = ModularPrimitivePrior(ModularConfig(**{
            k: v for k, v in model_cfg.items() if k in ModularConfig.__dataclass_fields__
        }))
    else:
        model = LatentActionPrior(PriorConfig(**{
            k: v for k, v in model_cfg.items() if k in PriorConfig.__dataclass_fields__
        }))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mgr.load(model, path=ckpt_path, restore_rng=False)
    model.to(device).eval()

    bundle_path = Path(args.bundle or data_meta.get("bundle") or get(cfg, "data.cache"))
    bundle = TrajectoryBundle.load(bundle_path)
    window = model.cfg.window
    n_samples = int(get(cfg, "validate.n_samples", 512))

    print(f"model      : {kind}  ({model.n_parameters:,} parameters)")
    print(f"checkpoint : {ckpt_path}  (epoch {peeked.get('state', {}).get('epoch', '?')})")
    print(f"data       : {bundle_path}  {len(bundle.trajectories)} trajectories\n")

    mesh = _mesh_checker(args.mano, bundle)

    ref_windows = windows_from_bundle(bundle, window, n_samples, seed=0)
    ref = validate_poses(ref_windows, fps=bundle.fps, label="training data")
    print(ref)
    ref_mesh = mesh(ref_windows, args.mesh_frames) if mesh else None
    if ref_mesh:
        print(_fmt_mesh("training data", ref_mesh))

    reports = {"reference": {**ref.to_dict(), "mesh": ref_mesh}}
    for temp in args.temperatures:
        samples = model.sample(n_samples, temperature=temp, device=device)
        native = _denorm(samples)
        rep = validate_poses(
            native,
            fps=bundle.fps,
            label=f"prior T={temp:g}",
            limits_are_structural=model.cfg.bounded_output,
        )
        print("\n" + str(rep))
        print("\n" + compare(ref, rep))
        m = mesh(native, args.mesh_frames) if mesh else None
        if m:
            print(_fmt_mesh(f"prior T={temp:g}", m))
        reports[f"T={temp:g}"] = {**rep.to_dict(), "mesh": m}

    # The modular model can also be asked to compose primitives it was never
    # asked to compose during training -- the interface a monolithic latent
    # cannot offer. Sampling explicit sequences checks those decode sanely.
    if kind == "modular":
        rng = np.random.default_rng(0)
        k = model.cfg.n_primitives
        seq = [int(x) for x in rng.integers(0, k, size=3)]
        composed = model.sample(n_samples, temperature=1.0, device=device, sequence=seq)
        rep = validate_poses(
            _denorm(composed),
            fps=bundle.fps,
            label=f"composed {seq}",
            limits_are_structural=model.cfg.bounded_output,
        )
        print("\n" + str(rep))
        print("\n" + compare(ref, rep))
        reports["composed"] = rep.to_dict()

    out = Path(args.out or run_dir / "ergonomics.json")
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    print("\nREADING THIS REPORT")
    print("  limit violations  : structurally zero with bounded_output; not evidence")
    print("  capsule interpen. : coarse proxy; the mesh number below supersedes it")
    print("  mesh interpen.    : compare against the TRAINING DATA row, not against 0 --")
    print("                      the 27-DOF projection itself puts real hand data above zero")
    print("  coupling + smooth : NOT enforced anywhere -- these are the real checks")
    return 0


def _mesh_checker(mano_path: str, bundle: TrajectoryBundle):
    """Return a callable scoring mesh self-intersection, or None if unavailable."""
    if not Path(mano_path).exists():
        print(f"[mesh] {mano_path} not found -- skipping mesh-level check\n")
        return None
    from caredex.mano import load_mano
    from caredex.mesh_collision import MeshSelfIntersection, mesh_interpenetration_rate

    model = load_mano(mano_path)
    checker = MeshSelfIntersection(model)
    # Reuse whatever convention the bundle was converted with, so the bridge
    # back to MANO is the exact inverse of how the data was read.
    axis = "xyz".find(bundle.meta.get("flexion_axis", "z"))
    sign = float(bundle.meta.get("flexion_sign", 1.0))

    def run(windows: np.ndarray, max_frames: int) -> dict:
        return mesh_interpenetration_rate(
            windows.reshape(-1, windows.shape[-1]),
            model,
            checker=checker,
            flexion_axis=axis,
            flexion_sign=sign,
            max_frames=max_frames,
        )

    return run


def _fmt_mesh(label: str, m: dict) -> str:
    return (
        f"mesh self-intersection [{label}]  "
        f"rate {m['mesh_interpenetration_rate']:.2%} over {m['frames_checked']} frames  "
        f"(mean {m['mean_intersecting_pairs']:.2f} pairs, max {m['max_intersecting_pairs']})"
        + (f"  digits: {m['digit_pairs_involved']}" if m["digit_pairs_involved"] else "")
    )


def _denorm(x: torch.Tensor) -> np.ndarray:
    from caredex.hand_model import denormalize

    return denormalize(x.float().cpu().numpy())


if __name__ == "__main__":
    raise SystemExit(main())
