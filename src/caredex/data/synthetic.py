"""Synthetic hand trajectory generator.

Stand-in for DexYCB / OakInk while those are behind licence forms. It is not a
throwaway: it gives the pipeline a dataset whose *intrinsic dimensionality is
known by construction*, which is the only way to tell a working hand prior from
a broken one before real data arrives.

Construction
------------
Each trajectory is a sequence of transitions between grasp primitives. A
primitive is a fixed 21-D articulated pose (a "grasp"); a trajectory picks a
few primitives and interpolates between them with minimum-jerk timing, while
the 6-D wrist pose follows an independent smooth random path. Then:

* per-primitive shape noise is drawn from a low-rank subspace, so the data has
  a genuine low-dimensional manifold rather than isotropic jitter;
* DIP flexion is overwritten by the 2/3 * PIP coupling;
* everything is clipped into the joint limit box.

Consequences a validator can check:
    - PCA on the articulated DOF should hit ~95% variance well below 21
      components (the primitives plus the low-rank noise span far less);
    - the DIP/PIP coupling residual should be ~0;
    - no limit violations, and bounded jerk.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from caredex.data.base import TrajectoryBundle, TrajectorySource, register_source
from caredex.hand_model import (
    ARTICULATED_SLICE,
    DOF_INDEX,
    GLOBAL_SLICE,
    LIMITS_HI,
    LIMITS_LO,
    N_ARTICULATED,
    N_DOF,
    apply_dip_pip_coupling,
    clamp_to_limits,
)

# ---------------------------------------------------------------------------
# Grasp primitives
# ---------------------------------------------------------------------------
# Values are degrees for the 21 articulated DOF, written as
# (mcp_flex, mcp_abd, pip_flex, dip_flex) per finger then the five thumb DOF.
# The taxonomy follows the coarse grasp families used in hand-synergy work
# (power / precision / lateral / hook / open), plus two caregiving-flavoured
# poses -- a flat palm for smoothing cloth and a pinch-and-drag for gripping a
# fabric edge -- so the prior sees the kind of postures Phase 3 will need.

_PRIMITIVES: dict[str, dict[str, float]] = {
    "open_flat": {},  # all zeros: neutral flat hand
    "power_cylindrical": {
        "index_mcp_flex": 60, "index_pip_flex": 80,
        "middle_mcp_flex": 62, "middle_pip_flex": 85,
        "ring_mcp_flex": 60, "ring_pip_flex": 82,
        "pinky_mcp_flex": 55, "pinky_pip_flex": 78,
        "thumb_cmc_flex": 30, "thumb_cmc_abd": 25, "thumb_mcp_flex": 30,
        "thumb_ip_flex": 25,
    },
    "power_spherical": {
        "index_mcp_flex": 45, "index_mcp_abd": 10, "index_pip_flex": 55,
        "middle_mcp_flex": 48, "middle_pip_flex": 58,
        "ring_mcp_flex": 45, "ring_mcp_abd": -8, "ring_pip_flex": 55,
        "pinky_mcp_flex": 42, "pinky_mcp_abd": -14, "pinky_pip_flex": 50,
        "thumb_cmc_flex": 25, "thumb_cmc_abd": 40, "thumb_mcp_flex": 20,
        "thumb_ip_flex": 15,
    },
    "precision_pinch": {
        "index_mcp_flex": 40, "index_pip_flex": 45, "middle_mcp_flex": 20,
        "middle_pip_flex": 25, "ring_mcp_flex": 12, "pinky_mcp_flex": 8,
        "thumb_cmc_flex": 35, "thumb_cmc_abd": 35, "thumb_mcp_flex": 25,
        "thumb_ip_flex": 20,
    },
    "tripod": {
        "index_mcp_flex": 38, "index_pip_flex": 42,
        "middle_mcp_flex": 36, "middle_pip_flex": 40,
        "ring_mcp_flex": 15, "pinky_mcp_flex": 10,
        "thumb_cmc_flex": 32, "thumb_cmc_abd": 38, "thumb_mcp_flex": 22,
        "thumb_ip_flex": 18,
    },
    "lateral_key": {
        "index_mcp_flex": 55, "index_pip_flex": 70, "index_dip_flex": 45,
        "middle_mcp_flex": 58, "middle_pip_flex": 75,
        "ring_mcp_flex": 55, "ring_pip_flex": 72,
        "pinky_mcp_flex": 50, "pinky_pip_flex": 68,
        "thumb_cmc_flex": 20, "thumb_cmc_abd": 8, "thumb_mcp_flex": 35,
        "thumb_mcp_abd": -6, "thumb_ip_flex": 10,
    },
    "hook": {
        "index_mcp_flex": 25, "index_pip_flex": 95,
        "middle_mcp_flex": 28, "middle_pip_flex": 100,
        "ring_mcp_flex": 26, "ring_pip_flex": 97,
        "pinky_mcp_flex": 22, "pinky_pip_flex": 92,
        "thumb_cmc_flex": 5, "thumb_cmc_abd": 5,
    },
    # Caregiving-flavoured postures (Phase 3 relevance, not clinically validated).
    "palm_smooth": {
        "index_mcp_flex": 12, "index_mcp_abd": 6,
        "middle_mcp_flex": 12, "ring_mcp_flex": 12, "ring_mcp_abd": -6,
        "pinky_mcp_flex": 12, "pinky_mcp_abd": -12,
        "thumb_cmc_abd": 30, "thumb_cmc_flex": 5,
    },
    "edge_pinch_drag": {
        "index_mcp_flex": 48, "index_pip_flex": 60, "index_mcp_abd": 4,
        "middle_mcp_flex": 44, "middle_pip_flex": 55,
        "ring_mcp_flex": 30, "ring_pip_flex": 35,
        "pinky_mcp_flex": 25, "pinky_pip_flex": 30,
        "thumb_cmc_flex": 38, "thumb_cmc_abd": 30, "thumb_mcp_flex": 30,
        "thumb_ip_flex": 30,
    },
}

PRIMITIVE_NAMES: tuple[str, ...] = tuple(_PRIMITIVES)


def primitive_matrix() -> np.ndarray:
    """``(n_primitives, 21)`` array of articulated primitive poses in degrees."""
    mat = np.zeros((len(_PRIMITIVES), N_ARTICULATED), dtype=np.float32)
    for row, spec in enumerate(_PRIMITIVES.values()):
        for name, value in spec.items():
            idx = DOF_INDEX[name]
            if idx >= N_ARTICULATED:
                raise ValueError(f"{name} is not an articulated DOF")
            mat[row, idx] = value
    return mat


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def minimum_jerk(n: int) -> np.ndarray:
    """Minimum-jerk interpolation weights over ``n`` samples, from 0 to 1.

    Human reaching and grasping follow a bell-shaped velocity profile; using it
    here means the synthetic velocity/acceleration statistics are in the right
    regime for the smoothness validator to be meaningful.
    """
    if n < 2:
        return np.ones(max(n, 1), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _smooth_random_walk(
    rng: np.random.Generator, n_steps: int, n_dim: int, smoothness: int
) -> np.ndarray:
    """Random walk low-pass filtered by repeated moving average, in [-1, 1]."""
    x = rng.standard_normal((n_steps + 4 * smoothness, n_dim)).astype(np.float32)
    kernel = np.ones(smoothness, dtype=np.float32) / smoothness
    for _ in range(3):
        x = np.stack(
            [np.convolve(x[:, d], kernel, mode="same") for d in range(n_dim)], axis=1
        )
    x = x[2 * smoothness : 2 * smoothness + n_steps]
    peak = np.abs(x).max(axis=0, keepdims=True)
    return x / np.maximum(peak, 1e-6)


@register_source("synthetic")
class SyntheticSource(TrajectorySource):
    """Generate grasp-primitive trajectories with known intrinsic structure.

    Parameters
    ----------
    n_trajectories, fps:
        Dataset size and sampling rate.
    seg_frames:
        ``(lo, hi)`` frame count for one primitive-to-primitive transition.
    segments_per_traj:
        ``(lo, hi)`` number of transitions per trajectory.
    noise_rank:
        Dimension of the subspace the per-pose shape noise is drawn from. Keep
        it below ``N_ARTICULATED`` so the manifold stays genuinely low-rank.
    noise_deg:
        Scale of that shape noise, in degrees.
    sensor_noise_deg:
        Per-frame independent noise, standing in for mocap/annotation jitter.
        Small -- it is the one component that is *not* low-rank.
    """

    def __init__(
        self,
        n_trajectories: int = 512,
        fps: float = 30.0,
        seg_frames: tuple[int, int] = (25, 60),
        segments_per_traj: tuple[int, int] = (3, 7),
        noise_rank: int = 6,
        noise_deg: float = 7.0,
        sensor_noise_deg: float = 0.4,
        transition_via_deg: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not 1 <= noise_rank < N_ARTICULATED:
            raise ValueError(f"noise_rank must be in [1, {N_ARTICULATED}), got {noise_rank}")
        self.n_trajectories = n_trajectories
        self.fps = fps
        self.seg_frames = seg_frames
        self.segments_per_traj = segments_per_traj
        self.noise_rank = noise_rank
        self.noise_deg = noise_deg
        self.sensor_noise_deg = sensor_noise_deg
        #: Magnitude of a per-ORDERED-PAIR via-point detour, in degrees.
        #:
        #: Zero reproduces plain interpolation between primitives -- and that
        #: turned out to make the dataset useless as a positive control for
        #: compositional generalisation. Interpolating between two endpoints a
        #: model has already seen separately is trivially generalisable, so a
        #: model that never saw the pair (a, b) still reconstructs it, and the
        #: measured compositional penalty is zero *by construction*.
        #:
        #: With a non-zero value each ordered pair gets its own fixed detour
        #: drawn from the low-rank subspace, applied at the midpoint of the
        #: transition. The path from a to b is then NOT derivable from a and b
        #: alone, so a model that never saw the pair genuinely cannot produce
        #: it. That is what a positive control has to look like.
        self.transition_via_deg = transition_via_deg
        self.seed = seed

    def load(self, **kwargs: Any) -> TrajectoryBundle:
        rng = np.random.default_rng(self.seed)
        prims = primitive_matrix()
        n_prims = len(prims)

        # One fixed low-rank basis for the whole dataset: this is the subspace
        # the prior is expected to discover.
        basis = rng.standard_normal((self.noise_rank, N_ARTICULATED)).astype(np.float32)
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)

        # One fixed via-point per ordered primitive pair. Drawn once for the
        # whole dataset so the same transition always detours the same way --
        # that consistency is what makes it learnable, and its pair-specificity
        # is what makes it unlearnable from the endpoints alone.
        via = np.zeros((n_prims, n_prims, N_ARTICULATED), dtype=np.float32)
        if self.transition_via_deg > 0:
            coeff = rng.standard_normal((n_prims, n_prims, self.noise_rank)).astype(np.float32)
            via = self.transition_via_deg * (coeff @ basis)

        trajectories: list[np.ndarray] = []
        labels: list[str] = []

        for _ in range(self.n_trajectories):
            n_seg = int(rng.integers(*self.segments_per_traj, endpoint=True))
            order = rng.integers(0, n_prims, size=n_seg + 1)

            # Each waypoint is a primitive plus a draw from the low-rank subspace.
            coeffs = rng.standard_normal((n_seg + 1, self.noise_rank)).astype(np.float32)
            waypoints = prims[order] + self.noise_deg * (coeffs @ basis)

            segments = []
            for k in range(n_seg):
                n = int(rng.integers(*self.seg_frames, endpoint=True))
                w = minimum_jerk(n)[:, None]
                seg = waypoints[k] * (1 - w) + waypoints[k + 1] * w
                if self.transition_via_deg > 0:
                    # Bump peaking at the midpoint and vanishing at both ends,
                    # so the endpoints stay exactly the primitive poses and only
                    # the path between them carries the pair-specific signature.
                    bump = np.sin(np.pi * w)
                    seg = seg + bump * via[order[k], order[k + 1]]
                segments.append(seg)
            articulated = np.concatenate(segments, axis=0)

            total = len(articulated)
            traj = np.zeros((total, N_DOF), dtype=np.float32)
            traj[:, ARTICULATED_SLICE] = articulated
            traj[:, GLOBAL_SLICE] = self._wrist_path(rng, total)

            traj += self.sensor_noise_deg * rng.standard_normal(traj.shape).astype(np.float32)
            traj = apply_dip_pip_coupling(traj)
            trajectories.append(clamp_to_limits(traj).astype(np.float32))
            labels.append("->".join(PRIMITIVE_NAMES[i] for i in order))

        return TrajectoryBundle(
            trajectories=trajectories,
            fps=self.fps,
            labels=labels,
            meta={
                "source": "synthetic",
                "seed": self.seed,
                "noise_rank": self.noise_rank,
                "noise_deg": self.noise_deg,
                "sensor_noise_deg": self.sensor_noise_deg,
                "transition_via_deg": self.transition_via_deg,
                # Without a via-point the transitions are pure interpolation and
                # carry NO compositional difficulty, whatever the labels suggest.
                "has_hard_compositions": self.transition_via_deg > 0,
                "n_primitives": n_prims,
                "primitives": list(PRIMITIVE_NAMES),
                # The ceiling a correct prior should approach: the primitive
                # simplex plus the noise subspace.
                "expected_intrinsic_dim": min(n_prims - 1 + self.noise_rank, N_ARTICULATED),
            },
        )

    def _wrist_path(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Smooth bounded wrist motion over the 6 global DOF."""
        lo = LIMITS_LO[GLOBAL_SLICE]
        hi = LIMITS_HI[GLOBAL_SLICE]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # 0.6 keeps the wrist off its limits, so limit violations in a trained
        # model are attributable to the model rather than to the data.
        walk = _smooth_random_walk(rng, n, 6, smoothness=max(3, n // 8))
        return mid + 0.6 * half * walk
