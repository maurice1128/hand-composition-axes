"""Training loop for the latent action prior.

Resumable by construction: :meth:`PriorTrainer.fit` picks up from ``last.pt``
if one exists in the run directory, so the same command can be re-run after an
interruption with no extra flags.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from caredex.models.latent_prior import LatentActionPrior
from caredex.train.checkpoint import CheckpointManager, TrainState


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    #: Epochs over which beta ramps 0 -> cfg.beta. Ramping avoids the VAE
    #: spending its early capacity satisfying the KL term and collapsing z
    #: before the decoder can use it.
    kl_warmup_epochs: int = 20
    #: Cosine decay to this fraction of lr. 1.0 disables the schedule.
    lr_min_factor: float = 0.05
    save_every: int = 10
    log_every: int = 50
    #: Stop if val loss has not improved for this many epochs. 0 disables.
    patience: int = 0
    amp: bool = True
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PriorTrainer:
    def __init__(
        self,
        model: LatentActionPrior,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: str | Path,
        cfg: TrainConfig | None = None,
        device: torch.device | None = None,
        data_meta: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg or TrainConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_meta = data_meta or {}

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.cfg.epochs, 1), eta_min=self.cfg.lr * self.cfg.lr_min_factor
        )
        self.use_amp = self.cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ckpt = CheckpointManager(run_dir)
        self.state = TrainState()

    # -- schedule -----------------------------------------------------------

    def beta_at(self, epoch: int) -> float:
        """Linear KL warmup from 0 to ``model.cfg.beta``."""
        target = self.model.cfg.beta
        if self.cfg.kl_warmup_epochs <= 0:
            return target
        return target * min(1.0, epoch / self.cfg.kl_warmup_epochs)

    # -- run ----------------------------------------------------------------

    def full_config(self) -> dict[str, Any]:
        return {
            "model": self.model.cfg.to_dict(),
            "train": self.cfg.to_dict(),
            "data": self.data_meta,
        }

    def fit(self, resume: bool = True) -> TrainState:
        if resume and self.ckpt.exists():
            self.state = self.ckpt.load(self.model, self.optimizer, self.scheduler)
            print(
                f"[train] resumed from {self.ckpt.last_path} "
                f"at epoch {self.state.epoch}, step {self.state.global_step}"
            )
        else:
            torch.manual_seed(self.cfg.seed)
            print(f"[train] starting fresh in {self.ckpt.run_dir}")

        print(
            f"[train] device={self.device} amp={self.use_amp} "
            f"params={self.model.n_parameters:,} "
            f"train_batches={len(self.train_loader)} val_batches={len(self.val_loader)}"
        )

        since_improved = 0
        for epoch in range(self.state.epoch, self.cfg.epochs):
            self.state.epoch = epoch
            t0 = time.time()
            beta = self.beta_at(epoch)

            train_metrics = self._train_epoch(beta)
            # Validation is always scored at the FINAL beta, never at the
            # warmup value. Scoring it at the current beta compares a different
            # objective every epoch, and "best" then always lands on epoch 0
            # where beta is 0 and the KL term is free.
            val_metrics = self.evaluate(self.val_loader, self.model.cfg.beta)

            self.scheduler.step()
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "beta": beta,
                "lr": self.scheduler.get_last_lr()[0],
                "seconds": round(elapsed, 2),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.state.history.append(record)

            is_best = val_metrics["loss"] < self.state.best_metric
            if is_best:
                self.state.best_metric = val_metrics["loss"]
                self.state.best_epoch = epoch
                since_improved = 0
            else:
                since_improved += 1

            print(
                f"[{epoch:4d}/{self.cfg.epochs}] "
                f"train {train_metrics['loss']:.5f} | val {val_metrics['loss']:.5f} "
                f"(recon {val_metrics['recon']:.5f}, kl {val_metrics['kl']:.3f}, "
                f"active {val_metrics['active_units']:.2f}) "
                f"beta={beta:.3f} {elapsed:.1f}s{'  *best' if is_best else ''}"
            )

            # state.epoch is stored as the *next* epoch to run, so a resume does
            # not repeat the epoch that was just completed.
            self.state.epoch = epoch + 1
            periodic = (epoch + 1) % self.cfg.save_every == 0
            if periodic or is_best or epoch + 1 == self.cfg.epochs:
                self.ckpt.save(
                    self.model,
                    self.optimizer,
                    self.state,
                    self.full_config(),
                    self.scheduler,
                    is_best=is_best,
                    tag_epoch=periodic,
                )

            if self.cfg.patience and since_improved >= self.cfg.patience:
                print(f"[train] early stop: no improvement for {since_improved} epochs")
                break

        print(
            f"[train] done. best val {self.state.best_metric:.6f} "
            f"at epoch {self.state.best_epoch} -> {self.ckpt.best_path}"
        )
        return self.state

    # -- epochs -------------------------------------------------------------

    def _train_epoch(self, beta: float) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        n = 0

        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, metrics = self.model.loss(batch, beta=beta)

            if not math.isfinite(metrics["loss"]):
                raise RuntimeError(
                    f"non-finite loss at epoch {self.state.epoch} step {self.state.global_step}: "
                    f"{metrics}"
                )

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = batch.shape[0]
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            n += bs
            self.state.global_step += 1

            if self.cfg.log_every and i % self.cfg.log_every == 0 and i > 0:
                print(f"    step {i}/{len(self.train_loader)} loss {metrics['loss']:.5f}")

        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, beta: float | None = None) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, metrics = self.model.loss(batch, beta=beta)
            bs = batch.shape[0]
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            n += bs
        return {k: v / max(n, 1) for k, v in totals.items()}
