"""Eigengrasp basis: PCA over hand poses.

The oldest hand-synergy method there is (Santello et al. 1998; Ciocarlie &
Allen's eigengrasps, 2007 onward). It earns its place here for two reasons:

1. It is the honest floor for the latent prior. If a sequence VAE cannot beat a
   linear projection at reconstructing hand poses, the VAE is not doing
   anything and the paper should say so.
2. Its explained-variance curve picks the latent dimension. Choosing that by
   hand is the kind of unjustified number a reviewer will ask about.

Deliberately numpy/SVD only -- no torch, no fitting loop, nothing to tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from caredex.hand_model import ARTICULATED_SLICE, N_ARTICULATED, N_DOF


@dataclass
class EigengraspBasis:
    """A fitted PCA basis over the articulated (or full) DOF, in normalised units.

    Attributes
    ----------
    mean:
        ``(d,)`` mean pose.
    components:
        ``(k, d)`` orthonormal rows, ordered by decreasing explained variance.
    explained_variance_ratio:
        ``(k,)`` fraction of total variance per component.
    articulated_only:
        If True the basis covers DOF 0..20 and ignores the 6 wrist DOF. This is
        the default: the wrist is commanded directly by the task policy and has
        no synergy structure to discover.
    """

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    articulated_only: bool = True

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def dim(self) -> int:
        return N_ARTICULATED if self.articulated_only else N_DOF

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        poses: np.ndarray,
        n_components: int | None = None,
        articulated_only: bool = True,
    ) -> "EigengraspBasis":
        """Fit on ``(n_frames, 27)`` *normalised* poses.

        Pass poses through :func:`caredex.hand_model.normalize` first; fitting
        on raw degrees would let the wrist's 160-degree range dominate the
        fingers' 90.
        """
        if poses.ndim != 2 or poses.shape[1] != N_DOF:
            raise ValueError(f"expected (n_frames, {N_DOF}), got {poses.shape}")
        x = poses[:, ARTICULATED_SLICE] if articulated_only else poses
        x = np.asarray(x, dtype=np.float64)

        d = x.shape[1]
        k = d if n_components is None else min(n_components, d)
        if len(x) <= d:
            raise ValueError(
                f"need more frames than DOF to fit a full basis: {len(x)} frames, {d} DOF"
            )

        mean = x.mean(axis=0)
        centred = x - mean
        # SVD rather than an eigendecomposition of the covariance: better
        # conditioned when components carry near-zero variance, which is exactly
        # what happens here thanks to the DIP/PIP coupling.
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        variance = s**2 / max(len(x) - 1, 1)
        total = variance.sum()
        ratio = variance / total if total > 0 else np.zeros_like(variance)

        return cls(
            mean=mean.astype(np.float32),
            components=vt[:k].astype(np.float32),
            explained_variance_ratio=ratio[:k].astype(np.float32),
            articulated_only=articulated_only,
        )

    # -- projection ---------------------------------------------------------

    def _select(self, poses: np.ndarray) -> np.ndarray:
        if poses.shape[-1] == N_DOF:
            return poses[..., ARTICULATED_SLICE] if self.articulated_only else poses
        if poses.shape[-1] == self.dim:
            return poses
        raise ValueError(
            f"expected trailing dim {N_DOF} or {self.dim}, got {poses.shape}"
        )

    def encode(self, poses: np.ndarray, k: int | None = None) -> np.ndarray:
        """Poses -> ``(..., k)`` eigengrasp amplitudes."""
        k = self.n_components if k is None else min(k, self.n_components)
        x = self._select(np.asarray(poses, dtype=np.float32))
        return (x - self.mean) @ self.components[:k].T

    def decode(self, amplitudes: np.ndarray, full: bool = True) -> np.ndarray:
        """Amplitudes -> poses. With ``full``, returns 27 DOF (wrist left at mean)."""
        a = np.asarray(amplitudes, dtype=np.float32)
        k = a.shape[-1]
        if k > self.n_components:
            raise ValueError(f"got {k} amplitudes, basis has {self.n_components}")
        x = a @ self.components[:k] + self.mean
        if not full or not self.articulated_only:
            return x.astype(np.float32)
        out = np.zeros(x.shape[:-1] + (N_DOF,), dtype=np.float32)
        out[..., ARTICULATED_SLICE] = x
        return out

    def reconstruct(self, poses: np.ndarray, k: int | None = None) -> np.ndarray:
        """Round-trip through the first ``k`` components, in the input's layout."""
        rec = self.decode(self.encode(poses, k), full=False)
        if poses.shape[-1] != N_DOF or not self.articulated_only:
            return rec
        out = np.array(poses, dtype=np.float32, copy=True)
        out[..., ARTICULATED_SLICE] = rec
        return out

    # -- diagnostics --------------------------------------------------------

    def n_components_for(self, variance: float = 0.95) -> int:
        """Smallest component count reaching ``variance`` of the total.

        This is the number that should justify the latent dimension in the paper.
        """
        if not 0 < variance <= 1:
            raise ValueError("variance must be in (0, 1]")
        cum = np.cumsum(self.explained_variance_ratio)
        idx = int(np.searchsorted(cum, variance) + 1)
        return min(idx, self.n_components)

    def variance_table(self) -> str:
        cum = np.cumsum(self.explained_variance_ratio)
        lines = [f"{'k':>3}  {'var':>8}  {'cumulative':>10}"]
        for i, (v, c) in enumerate(zip(self.explained_variance_ratio, cum), start=1):
            lines.append(f"{i:>3}  {v:>8.4f}  {c:>10.4f}")
        return "\n".join(lines)

    # -- io -----------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
            articulated_only=np.bool_(self.articulated_only),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EigengraspBasis":
        d = np.load(Path(path))
        return cls(
            mean=d["mean"],
            components=d["components"],
            explained_variance_ratio=d["explained_variance_ratio"],
            articulated_only=bool(d["articulated_only"]),
        )
