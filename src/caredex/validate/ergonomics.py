"""Ergonomic validation of decoded hand trajectories.

The gate the handover document sets for the hand prior: no hyperextension, no
interpenetration, biomechanically fluent motion. This module turns each of
those into a number, computed the same way for the training data and for
samples drawn from the prior -- the data's own scores are the reference, since
a prior cannot be expected to beat the motion it was fit to.

Read the caveats. Two of the four checks are weaker than they look:

* **Joint limits** are enforced structurally when the model uses a tanh output
  (``bounded_output=True``, the default), so a score of zero says nothing about
  what the model learned. ``report.limits_are_structural`` records this.
* **Interpenetration** is a capsule-model proxy, not mesh collision. See
  :mod:`caredex.kinematics`.

The checks with teeth are the DIP/PIP coupling residual and the smoothness
statistics: neither is enforced anywhere in the architecture, so failing them
means the prior really did not learn the structure in the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from caredex.hand_model import (
    ARTICULATED_SLICE,
    DIP_PIP_RATIO,
    coupling_residual,
    denormalize,
    limit_violations,
)
from caredex.kinematics import interpenetration
from caredex.models.latent_prior import LatentActionPrior


@dataclass
class ErgonomicReport:
    n_windows: int
    n_frames: int
    fps: float

    # Joint limits, in native units (degrees / metres).
    limit_violation_rate: float
    limit_violation_max: float
    limits_are_structural: bool

    # DIP/PIP coupling, degrees.
    coupling_residual_mean: float
    coupling_residual_p95: float
    coupling_residual_max: float

    # Capsule-model self-intersection, metres.
    interpenetration_rate: float
    interpenetration_max: float

    # Smoothness of the articulated DOF, in degrees per second and per second^2.
    velocity_mean: float
    velocity_p99: float
    acceleration_p99: float
    jerk_p99: float

    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        star = " (structural, see docstring)" if self.limits_are_structural else ""
        return "\n".join(
            [
                f"--- ergonomic report: {self.label or 'unnamed'} ---",
                f"windows={self.n_windows}  frames={self.n_frames}  fps={self.fps:g}",
                "",
                f"joint limits    violation rate {self.limit_violation_rate:.4%}  "
                f"max {self.limit_violation_max:.3f}{star}",
                f"DIP/PIP coupling  mean {self.coupling_residual_mean:6.3f} deg  "
                f"p95 {self.coupling_residual_p95:6.3f}  max {self.coupling_residual_max:6.3f}",
                f"interpenetration  rate {self.interpenetration_rate:.4%}  "
                f"max depth {self.interpenetration_max * 1000:.2f} mm",
                f"smoothness      |v| mean {self.velocity_mean:7.2f} deg/s  "
                f"p99 {self.velocity_p99:7.2f}",
                f"                |a| p99 {self.acceleration_p99:9.1f} deg/s^2  "
                f"|jerk| p99 {self.jerk_p99:.3e} deg/s^3",
            ]
        )


def validate_poses(
    windows: np.ndarray,
    fps: float,
    label: str = "",
    limits_are_structural: bool = False,
    interpenetration_tol: float = 1e-4,
) -> ErgonomicReport:
    """Score ``(N, T, 27)`` windows given in NATIVE units (degrees / metres)."""
    if windows.ndim != 3:
        raise ValueError(f"expected (N, T, 27), got {windows.shape}")
    n_windows, T, _ = windows.shape

    viol = limit_violations(windows)
    coup = coupling_residual(windows, DIP_PIP_RATIO)
    pen = interpenetration(windows)

    # Derivatives on the articulated DOF only: mixing degrees and metres into
    # one "velocity" number would be meaningless.
    art = windows[..., ARTICULATED_SLICE]
    dt = 1.0 / fps
    vel = np.abs(np.diff(art, axis=1)) / dt
    acc = np.abs(np.diff(art, n=2, axis=1)) / dt**2 if T > 2 else np.zeros((1,))
    jerk = np.abs(np.diff(art, n=3, axis=1)) / dt**3 if T > 3 else np.zeros((1,))

    return ErgonomicReport(
        n_windows=n_windows,
        n_frames=n_windows * T,
        fps=fps,
        limit_violation_rate=float((viol > 0).any(axis=-1).mean()),
        limit_violation_max=float(viol.max()),
        limits_are_structural=limits_are_structural,
        coupling_residual_mean=float(coup.mean()),
        coupling_residual_p95=float(np.percentile(coup, 95)),
        coupling_residual_max=float(coup.max()),
        interpenetration_rate=float((pen > interpenetration_tol).mean()),
        interpenetration_max=float(pen.max()),
        velocity_mean=float(vel.mean()),
        velocity_p99=float(np.percentile(vel, 99)),
        acceleration_p99=float(np.percentile(acc, 99)),
        jerk_p99=float(np.percentile(jerk, 99)),
        label=label,
    )


@torch.no_grad()
def validate_prior(
    model: LatentActionPrior,
    fps: float,
    n_samples: int = 512,
    temperature: float = 1.0,
    initial_pose: torch.Tensor | None = None,
    device: torch.device | None = None,
    seed: int = 0,
    label: str = "prior samples",
) -> ErgonomicReport:
    """Sample the prior and score the decoded windows.

    Sampling z ~ N(0, I) rather than encoding real data is the point: it tests
    whether *the whole latent space* decodes to plausible hand motion, which is
    what Phase 4's RL policy will rely on when it explores. A prior that only
    behaves on the data manifold is not usable as an action space.
    """
    device = device or next(model.parameters()).device
    model.eval()

    generator = torch.Generator(device=device).manual_seed(seed)
    samples = model.sample(
        n_samples,
        initial_pose=initial_pose,
        temperature=temperature,
        device=device,
        generator=generator,
    )
    native = denormalize(samples.float().cpu().numpy())

    report = validate_poses(
        native,
        fps=fps,
        label=label,
        limits_are_structural=model.cfg.bounded_output,
    )
    report.extra["temperature"] = temperature
    report.extra["latent_dim"] = model.cfg.latent_dim
    return report


def compare(reference: ErgonomicReport, candidate: ErgonomicReport) -> str:
    """Side-by-side table. ``reference`` is normally the training data."""
    rows = [
        ("limit violation rate", "limit_violation_rate", "{:.4%}"),
        ("coupling residual mean (deg)", "coupling_residual_mean", "{:.3f}"),
        ("coupling residual p95 (deg)", "coupling_residual_p95", "{:.3f}"),
        ("interpenetration rate", "interpenetration_rate", "{:.4%}"),
        ("interpenetration max (m)", "interpenetration_max", "{:.5f}"),
        ("|v| mean (deg/s)", "velocity_mean", "{:.2f}"),
        ("|v| p99 (deg/s)", "velocity_p99", "{:.2f}"),
        ("|a| p99 (deg/s^2)", "acceleration_p99", "{:.1f}"),
        ("|jerk| p99 (deg/s^3)", "jerk_p99", "{:.3e}"),
    ]
    width = max(len(r[0]) for r in rows)
    lines = [
        f"{'metric':<{width}}  {reference.label or 'reference':>18}  {candidate.label or 'candidate':>18}",
        "-" * (width + 42),
    ]
    for name, attr, fmt in rows:
        a = fmt.format(getattr(reference, attr))
        b = fmt.format(getattr(candidate, attr))
        lines.append(f"{name:<{width}}  {a:>18}  {b:>18}")
    return "\n".join(lines)
