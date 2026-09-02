"""Train a hand prior, resumably.

Re-running the same command after an interruption picks up from ``last.pt``
with no extra flags -- the session-lasts-hours / training-lasts-days problem is
the reason the checkpoint carries RNG state too.

    python scripts/train_prior.py
    python scripts/train_prior.py --model modular --set train.epochs=400
    python scripts/train_prior.py --bundle data/bundles/oakink.npz --run-dir runs/prior_oakink
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.config import dump, get, load_config  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.pipeline import make_dataloaders  # noqa: E402
from caredex.models.latent_prior import LatentActionPrior, PriorConfig  # noqa: E402
from caredex.models.modular_prior import ModularConfig, ModularPrimitivePrior  # noqa: E402
from caredex.train.trainer import PriorTrainer, TrainConfig  # noqa: E402


def build(kind: str, cfg: dict, window: int):
    m = get(cfg, "model", {})
    if kind == "monolithic":
        return LatentActionPrior(
            PriorConfig(
                window=window,
                latent_dim=m.get("latent_dim", 12),
                hidden_dim=m.get("hidden_dim", 256),
                n_layers=m.get("n_layers", 2),
                dropout=m.get("dropout", 0.0),
                bounded_output=m.get("bounded_output", True),
                condition_on_initial=m.get("condition_on_initial", True),
                beta=m.get("beta", 1.0),
                free_bits=m.get("free_bits", 0.02),
                smoothness_weight=m.get("smoothness_weight", 0.1),
                # Was omitted, so `--set model.per_frame_latent=true` parsed
                # correctly and then went nowhere: two task-level comparisons
                # were run against a window-level baseline while their logs
                # recorded the flag as set. A config field the builder does not
                # read is worse than one that does not exist, because the run
                # directory then contains a record of a setting that never
                # applied.
                per_frame_latent=m.get("per_frame_latent", False),
            )
        )
    if kind == "modular":
        return ModularPrimitivePrior(
            ModularConfig(
                window=window,
                n_primitives=m.get("n_primitives", 12),
                primitive_dim=m.get("primitive_dim", 32),
                style_dim=m.get("style_dim", 4),
                hidden_dim=m.get("hidden_dim", 256),
                n_layers=m.get("n_layers", 2),
                dropout=m.get("dropout", 0.0),
                bounded_output=m.get("bounded_output", True),
                beta=m.get("beta", 1.0),
                free_bits=m.get("free_bits", 0.02),
                smoothness_weight=m.get("smoothness_weight", 0.1),
                consistency_weight=m.get("consistency_weight", 0.0),
                consistency_contexts=m.get("consistency_contexts", 4),
            )
        )
    raise ValueError(f"unknown model {kind!r}; expected 'monolithic' or 'modular'")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--model", choices=("monolithic", "modular"), default="monolithic")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    bundle_path = Path(args.bundle or get(cfg, "data.cache"))
    if not bundle_path.exists():
        print(f"no bundle at {bundle_path}; run scripts/generate_synthetic.py first")
        return 1

    bundle = TrajectoryBundle.load(bundle_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ld = get(cfg, "loader", {})
    window = ld.get("window", 32)

    train_loader, val_loader, test_loader, sizes = make_dataloaders(
        bundle,
        window=window,
        batch_size=ld.get("batch_size", 256),
        stride=ld.get("stride", 2),
        val_frac=ld.get("val_frac", 0.1),
        test_frac=ld.get("test_frac", 0.1),
        seed=ld.get("split_seed", 0),
        num_workers=ld.get("num_workers", 0),
        device=device,
    )
    print(
        f"[data] {bundle_path}  {len(bundle.trajectories)} trajectories -> "
        f"windows train/val/test = {sizes.train}/{sizes.val}/{sizes.test}"
    )

    model = build(args.model, cfg, window)
    t = get(cfg, "train", {})
    run_dir = Path(args.run_dir or f"{get(cfg, 'run_dir', 'runs/prior')}_{args.model}")

    trainer = PriorTrainer(
        model,
        train_loader,
        val_loader,
        run_dir=run_dir,
        cfg=TrainConfig(
            epochs=t.get("epochs", 200),
            lr=t.get("lr", 1e-3),
            weight_decay=t.get("weight_decay", 0.0),
            grad_clip=t.get("grad_clip", 1.0),
            kl_warmup_epochs=t.get("kl_warmup_epochs", 20),
            lr_min_factor=t.get("lr_min_factor", 0.05),
            save_every=t.get("save_every", 10),
            log_every=t.get("log_every", 0),
            patience=t.get("patience", 0),
            amp=t.get("amp", True),
            seed=t.get("seed", 0),
        ),
        device=device,
        data_meta={**bundle.meta, "bundle": str(bundle_path), "model_kind": args.model},
    )

    print(f"[config]\n{dump({'model': cfg.get('model', {}), 'train': cfg.get('train', {})})}\n")
    trainer.fit(resume=not args.fresh)

    if trainer.ckpt.best_path.exists():
        trainer.ckpt.load(model, path=trainer.ckpt.best_path, restore_rng=False)
        model.to(device)
    test_metrics = trainer.evaluate(test_loader, beta=model.cfg.beta)
    print("\n[test] " + "  ".join(f"{k}={v:.5f}" for k, v in test_metrics.items()))
    print(f"[test] checkpoint: {trainer.ckpt.best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
