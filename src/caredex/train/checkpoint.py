"""Checkpointing built for interrupted runs.

The operating constraint of this project: an agent session lasts hours, a
training run lasts days. Every run must therefore survive being killed at an
arbitrary step and resumed later, with no silent change in behaviour.

"No silent change" is the hard part and it is why RNG state is saved. Without
it, a resumed run reshuffles its batches and redraws its VAE noise from a fresh
stream; the loss curve still looks plausible, so the corruption is invisible.
:meth:`CheckpointManager.save` captures the Python, numpy, torch CPU and torch
CUDA generators alongside the weights.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_VERSION = 1


@dataclass
class TrainState:
    """Everything about a run that is not weights or optimiser state."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = -1
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainState":
        return cls(
            epoch=d.get("epoch", 0),
            global_step=d.get("global_step", 0),
            best_metric=d.get("best_metric", float("inf")),
            best_epoch=d.get("best_epoch", -1),
            history=d.get("history", []),
        )


def _capture_rng() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except (RuntimeError, ValueError):
            # Resuming on a machine with a different GPU count. The weights are
            # still valid; only the CUDA noise stream diverges.
            print("[checkpoint] warning: could not restore CUDA RNG state (device mismatch)")


class CheckpointManager:
    """Writes ``last.pt``, ``best.pt`` and rolling ``epoch_XXXX.pt`` into a run dir.

    ``last.pt`` is always the resume target. It is written atomically (temp file
    then replace) so that a crash mid-save cannot leave an unreadable
    checkpoint -- the failure mode that costs a day of GPU time.
    """

    def __init__(self, run_dir: str | Path, keep_last: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainState,
        config: dict[str, Any],
        scheduler: Any | None = None,
        is_best: bool = False,
        tag_epoch: bool = True,
    ) -> Path:
        payload = {
            "version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "state": state.to_dict(),
            "config": config,
            "rng": _capture_rng(),
        }
        self._atomic_save(payload, self.last_path)
        if is_best:
            self._atomic_save(payload, self.best_path)
        if tag_epoch:
            self._atomic_save(payload, self.run_dir / f"epoch_{state.epoch:04d}.pt")
            self._prune()
        (self.run_dir / "history.json").write_text(
            json.dumps(state.history, indent=2), encoding="utf-8"
        )
        return self.last_path

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> TrainState:
        """Restore in place and return the :class:`TrainState` to continue from."""
        ckpt_path = Path(path) if path else self.last_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no checkpoint at {ckpt_path}")

        payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        version = payload.get("version", 0)
        if version != CHECKPOINT_VERSION:
            print(
                f"[checkpoint] warning: file version {version} != "
                f"current {CHECKPOINT_VERSION}; loading anyway"
            )

        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if restore_rng and "rng" in payload:
            _restore_rng(payload["rng"])

        return TrainState.from_dict(payload["state"])

    def exists(self) -> bool:
        return self.last_path.exists()

    def peek(self, path: str | Path | None = None) -> dict[str, Any]:
        """Read the config and state of a checkpoint without building a model."""
        ckpt_path = Path(path) if path else self.last_path
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return {"config": payload.get("config", {}), "state": payload.get("state", {})}

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def _prune(self) -> None:
        snaps = sorted(self.run_dir.glob("epoch_*.pt"))
        for old in snaps[: max(0, len(snaps) - self.keep_last)]:
            old.unlink(missing_ok=True)
