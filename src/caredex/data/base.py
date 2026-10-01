"""Pluggable trajectory sources.

Every source -- synthetic, DexYCB, OakInk -- returns the same
:class:`TrajectoryBundle`, so swapping the data behind the prior is a config
change, not a code change. This matters because the real datasets are gated
behind licence forms and will arrive after the pipeline is already running on
synthetic data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from caredex.hand_model import N_DOF, DOF_NAMES


@dataclass
class TrajectoryBundle:
    """A set of variable-length hand trajectories in native DOF units.

    Attributes
    ----------
    trajectories:
        List of ``(T_i, 27)`` float32 arrays. Lengths may differ; windowing into
        fixed-length training samples happens downstream in the dataset.
    fps:
        Sampling rate, needed to make velocity/acceleration smoothness metrics
        comparable across sources.
    labels:
        Optional per-trajectory string tag (grasp type, sequence id). Used for
        stratified splits and for colouring latent-space plots.
    meta:
        Free-form provenance -- source name, generator seed, dataset version.
        Written into checkpoints so a trained prior can be traced to its data.
    """

    trajectories: list[np.ndarray]
    fps: float
    labels: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    #: Optional per-frame channels alongside the 27 DOF, one ``(T, K)`` array per
    #: trajectory. Kept *beside* the pose rather than inside it because
    #: :mod:`caredex.hand_model` is the single source of truth for the action
    #: layout, and widening that vector would change what every model in the
    #: repo means by "an action". Used for GRAB's per-region contact.
    aux: list[np.ndarray] | None = None
    aux_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("TrajectoryBundle is empty")
        for i, tr in enumerate(self.trajectories):
            if tr.ndim != 2 or tr.shape[1] != N_DOF:
                raise ValueError(
                    f"trajectory {i} has shape {tr.shape}, expected (T, {N_DOF})"
                )
            if not np.isfinite(tr).all():
                raise ValueError(f"trajectory {i} contains non-finite values")
        if self.aux is not None:
            if len(self.aux) != len(self.trajectories):
                raise ValueError(
                    f"{len(self.aux)} aux arrays for {len(self.trajectories)} trajectories"
                )
            k = self.aux[0].shape[1]
            for i, (tr, a) in enumerate(zip(self.trajectories, self.aux)):
                if a.ndim != 2 or a.shape[0] != len(tr) or a.shape[1] != k:
                    raise ValueError(
                        f"aux {i} has shape {a.shape}, expected ({len(tr)}, {k})"
                    )
                if not np.isfinite(a).all():
                    raise ValueError(f"aux {i} contains non-finite values")
            if self.aux_names and len(self.aux_names) != k:
                raise ValueError(f"{len(self.aux_names)} aux names for {k} channels")
        if self.labels and len(self.labels) != len(self.trajectories):
            raise ValueError(
                f"{len(self.labels)} labels for {len(self.trajectories)} trajectories"
            )
        if not self.labels:
            self.labels = ["unlabelled"] * len(self.trajectories)

    @property
    def n_frames(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def stacked(self) -> np.ndarray:
        """All frames concatenated, ``(n_frames, 27)``. Used to fit PCA/stats."""
        return np.concatenate(self.trajectories, axis=0)

    def describe(self) -> str:
        arr = self.stacked()
        lines = [
            f"source={self.meta.get('source', '?')}  "
            f"{len(self.trajectories)} trajectories  {self.n_frames} frames  {self.fps:g} fps",
            f"{'dof':<24} {'min':>8} {'mean':>8} {'max':>8} {'std':>8}",
        ]
        for i, name in enumerate(DOF_NAMES):
            c = arr[:, i]
            lines.append(
                f"{name:<24} {c.min():>8.2f} {c.mean():>8.2f} {c.max():>8.2f} {c.std():>8.2f}"
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Persist as a single .npz (ragged trajectories stored with offsets)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.array([len(t) for t in self.trajectories], dtype=np.int64)
        payload = dict(
            frames=self.stacked().astype(np.float32),
            lengths=lengths,
            labels=np.array(self.labels, dtype=object),
            fps=np.float64(self.fps),
            meta=np.array(repr(self.meta), dtype=object),
        )
        # Written only when present, so bundles without aux stay byte-comparable
        # to the ones already on disk and older readers keep working.
        if self.aux is not None:
            payload["aux"] = np.concatenate(self.aux, axis=0).astype(np.float32)
            payload["aux_names"] = np.array(self.aux_names, dtype=object)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryBundle":
        import ast

        data = np.load(Path(path), allow_pickle=True)
        frames = data["frames"]
        lengths = data["lengths"]
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        trajectories = [
            frames[bounds[i] : bounds[i + 1]] for i in range(len(lengths))
        ]
        aux = None
        aux_names: list[str] = []
        if "aux" in data.files:
            flat = data["aux"]
            aux = [flat[bounds[i] : bounds[i + 1]] for i in range(len(lengths))]
            aux_names = [str(x) for x in data["aux_names"].tolist()]
        return cls(
            trajectories=trajectories,
            fps=float(data["fps"]),
            labels=[str(x) for x in data["labels"].tolist()],
            meta=ast.literal_eval(str(data["meta"])),
            aux=aux,
            aux_names=aux_names,
        )


class TrajectorySource(ABC):
    """Base class for anything that can produce a :class:`TrajectoryBundle`."""

    name: str = "base"

    @abstractmethod
    def load(self, **kwargs: Any) -> TrajectoryBundle:
        """Produce the bundle. May be slow; callers cache the result to disk."""


_REGISTRY: dict[str, Callable[[], TrajectorySource]] = {}


def register_source(name: str) -> Callable[[type], type]:
    """Class decorator registering a source under ``name`` for config lookup."""

    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return deco


def get_source(name: str, **init_kwargs: Any) -> TrajectorySource:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown source {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**init_kwargs)  # type: ignore[operator]
