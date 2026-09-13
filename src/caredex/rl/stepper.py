"""Run a trained prior's decoder one frame at a time.

Both priors were trained on 32-frame windows and decode a whole window in one
GRU pass. A policy cannot use that: it decides one frame, sees what happened,
and decides the next. Both decoders are unidirectional GRUs, so the same
weights can be stepped autoregressively with an explicit hidden state, and the
per-frame outputs are bit-identical to the windowed pass (see ``tests``).

Two things carry over from training and are kept deliberately:

* The decoder is conditioned on the window's *initial pose*. Beyond 32 frames
  the hidden state is extrapolating past anything it saw in training, so the
  stepper re-anchors every ``rewindow`` frames: it resets the hidden state
  with the current commanded pose as the new initial pose. That keeps every
  frame the policy asks for inside the decoder's training distribution.
* The modular arm's style code was one 4-D vector per window. Here the policy
  may emit it per frame (``style_mode="frame"``) or the stepper may hold it
  fixed until the next re-anchor (``style_mode="window"``). The latter is the
  faithful one; the former gives the policy more channel, and the rate
  accounting in the experiment scripts charges it accordingly.

Action layout, in normalised units the policy sees:

* modular:     ``[K logits | style_dim]``  -> softmax(logits) @ primitives
* continuous:  ``[latent_dim]``            -> fed straight to the decoder
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from caredex.models.latent_prior import LatentActionPrior
from caredex.models.modular_prior import ModularPrimitivePrior


@dataclass
class StepperSpec:
    kind: str
    action_dim: int
    #: Per-component bounds handed to the environment's action space.
    low: np.ndarray
    high: np.ndarray


class _Base:
    rewindow: int
    n_channels: int

    def __init__(self, rewindow: int = 32, device: str | torch.device = "cpu"):
        self.rewindow = int(rewindow)
        self.device = torch.device(device)
        self._h: torch.Tensor | None = None
        self._anchor: torch.Tensor | None = None
        self._t = 0
        self._last_pose: torch.Tensor | None = None

    # -- interface ----------------------------------------------------------

    @property
    def spec(self) -> StepperSpec:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self, initial_pose: np.ndarray) -> None:
        """``initial_pose`` is normalised ``(27,)``; the window anchor."""
        pose = torch.as_tensor(np.asarray(initial_pose, dtype=np.float32),
                               device=self.device).view(1, -1)
        self._anchor = pose
        self._last_pose = pose
        self._h = None
        self._t = 0

    @torch.no_grad()
    def step(self, action: np.ndarray) -> np.ndarray:
        if self._anchor is None:
            raise RuntimeError("call reset() before step()")
        if self._t > 0 and self._t % self.rewindow == 0:
            # Re-anchor: the pose we last commanded becomes the new window start.
            self._anchor = self._last_pose
            self._h = None
        a = torch.as_tensor(np.asarray(action, dtype=np.float32),
                            device=self.device).view(1, -1)
        pose = self._step(a)
        self._last_pose = pose
        self._t += 1
        return pose[0].cpu().numpy()

    def _step(self, a: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class ModularStepper(_Base):
    def __init__(self, model: ModularPrimitivePrior, rewindow: int = 32,
                 style_mode: str = "window", logit_scale: float = 1.0,
                 device: str | torch.device = "cpu"):
        super().__init__(rewindow, device)
        if style_mode not in ("window", "frame"):
            raise ValueError(style_mode)
        self.model = model.to(self.device).eval()
        self.cfg = model.cfg
        self.n_channels = self.cfg.n_channels
        self.style_mode = style_mode
        self.logit_scale = float(logit_scale)
        self._style: torch.Tensor | None = None

    @property
    def spec(self) -> StepperSpec:
        K, S = self.cfg.n_primitives, self.cfg.style_dim
        low = np.concatenate([np.full(K, -5.0), np.full(S, -3.0)]).astype(np.float32)
        return StepperSpec("modular", K + S, low, -low)

    def reset(self, initial_pose: np.ndarray) -> None:
        super().reset(initial_pose)
        self._style = None

    def weights(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logit_scale * logits, dim=-1)

    def _step(self, a: torch.Tensor) -> torch.Tensor:
        K = self.cfg.n_primitives
        logits, style_in = a[:, :K], a[:, K:]
        if self.style_mode == "frame" or self._style is None or self._h is None:
            self._style = style_in
        style = self._style
        ctx = torch.cat([style, self._anchor], dim=-1)
        if self._h is None:
            h0 = self.model.decoder_init(ctx)
            self._h = h0.view(-1, self.cfg.n_layers, self.cfg.hidden_dim).permute(1, 0, 2).contiguous()
        prim = self.weights(logits) @ self.model.primitives
        x = torch.cat([prim, ctx], dim=-1).unsqueeze(1)
        out, self._h = self.model.decoder_rnn(x, self._h)
        pose = self.model.to_pose(out[:, 0])
        return torch.tanh(pose) if self.cfg.bounded_output else pose


class ContinuousStepper(_Base):
    """Per-frame latent baseline (``per_frame_latent=True`` checkpoints).

    The windowed decoder initialises its hidden state from the *first* frame's
    latent together with the initial pose, so the first action after each
    re-anchor plays that role here too.
    """

    def __init__(self, model: LatentActionPrior, rewindow: int = 32,
                 device: str | torch.device = "cpu"):
        super().__init__(rewindow, device)
        if not model.cfg.per_frame_latent:
            raise ValueError("ContinuousStepper needs a per-frame-latent prior; "
                             "a window-level latent has no per-step action.")
        if not model.cfg.condition_on_initial:
            raise ValueError("expected condition_on_initial=True")
        self.model = model.to(self.device).eval()
        self.cfg = model.cfg
        self.n_channels = self.cfg.n_channels

    @property
    def spec(self) -> StepperSpec:
        D = self.cfg.latent_dim
        low = np.full(D, -3.0, dtype=np.float32)
        return StepperSpec("continuous", D, low, -low)

    def _step(self, a: torch.Tensor) -> torch.Tensor:
        z = a
        if self._h is None:
            ctx0 = torch.cat([z, self._anchor], dim=-1)
            h0 = self.model.decoder_init(ctx0)
            self._h = h0.view(-1, self.cfg.n_layers, self.cfg.hidden_dim).permute(1, 0, 2).contiguous()
        x = torch.cat([z, self._anchor], dim=-1).unsqueeze(1)
        out, self._h = self.model.decoder_rnn(x, self._h)
        pose = self.model.to_pose(out[:, 0])
        return torch.tanh(pose) if self.cfg.bounded_output else pose


class KeyframeStepper(_Base):
    """MotionBricks-style interface: the action is a target hand pose.

    The policy emits a keyframe (21 articulated joints, normalised units)
    that the decoder is asked to reach ``horizon`` frames after the current
    anchor; the decoder produces the data's way of getting there, frame by
    frame. Re-anchoring every ``horizon`` frames makes each macro step a
    fresh in-betweening problem from the pose actually commanded last.
    """

    def __init__(self, model, horizon: int = 16, device: str | torch.device = "cpu"):
        super().__init__(rewindow=horizon, device=device)
        self.model = model.to(self.device).eval()
        self.cfg = model.cfg
        self.n_channels = self.cfg.n_channels
        self.horizon = int(horizon)
        self._target: torch.Tensor | None = None

    @property
    def spec(self) -> StepperSpec:
        from caredex.hand_model import N_ARTICULATED
        low = np.full(N_ARTICULATED, -1.0, dtype=np.float32)
        return StepperSpec("keyframe", N_ARTICULATED, low, -low)

    def _step(self, a: torch.Tensor) -> torch.Tensor:
        from caredex.hand_model import N_ARTICULATED
        C = self.n_channels
        if self._h is None:
            # New macro step: latch the target for the whole horizon.
            tgt = torch.zeros(1, C, device=self.device)
            tgt[:, :N_ARTICULATED] = a[:, :N_ARTICULATED]
            self._target = tgt
            self._t_in = 0
        mask = torch.zeros(1, C, device=self.device); mask[:, :N_ARTICULATED] = 1.0
        remaining = torch.tensor([[max(self.horizon - self._t_in, 0) / self.cfg.window]], device=self.device)
        x = torch.cat([self._anchor, self._target * mask, mask, remaining], dim=-1)
        if self._h is None:
            h0 = self.model.decoder_init(x)
            self._h = h0.view(-1, self.cfg.n_layers, self.cfg.hidden_dim).permute(1, 0, 2).contiguous()
        out, self._h = self.model.decoder_rnn(x.unsqueeze(1), self._h)
        self._t_in += 1
        pose = self.model.to_pose(out[:, 0])
        return torch.tanh(pose) if self.cfg.bounded_output else pose


class LinearKeyframeStepper(_Base):
    """Control for the keyframe interface: no learned prior, straight-line
    interpolation from the anchor to the target over the horizon."""

    def __init__(self, horizon: int = 16, device: str | torch.device = "cpu"):
        super().__init__(rewindow=horizon, device=device)
        self.n_channels = 27
        self.horizon = int(horizon)
        self.model = None
        self._target = None

    @property
    def spec(self) -> StepperSpec:
        from caredex.hand_model import N_ARTICULATED
        low = np.full(N_ARTICULATED, -1.0, dtype=np.float32)
        return StepperSpec("linear", N_ARTICULATED, low, -low)

    def _step(self, a: torch.Tensor) -> torch.Tensor:
        from caredex.hand_model import N_ARTICULATED
        if self._h is None:
            tgt = self._anchor.clone(); tgt[:, :N_ARTICULATED] = a[:, :N_ARTICULATED]
            self._target = tgt; self._t_in = 0; self._h = torch.zeros(1)
        self._t_in += 1
        w = min(self._t_in / self.horizon, 1.0)
        return (1 - w) * self._anchor + w * self._target


# -- loading ------------------------------------------------------------------

def load_stepper(run_dir, arm: str, device: str | torch.device = "cpu",
                 untrained: bool = False, untrained_seed: int = 0,
                 **kw) -> ModularStepper | ContinuousStepper:
    """Load ``best.pt`` from a run directory into the matching stepper.

    ``untrained=True`` is a control: it builds the same architecture from the
    checkpoint's config but keeps the random initialisation. A policy acting
    through an untrained decoder tells you how much of whatever the trained
    arms achieve is owed to the prior at all, which is what LAMP's ``Raw``
    baseline asks; here it is asked per architecture so the two arms' floors
    can differ.
    """
    from pathlib import Path

    from caredex.models.latent_prior import PriorConfig
    from caredex.models.modular_prior import ModularConfig
    from caredex.train.checkpoint import CheckpointManager

    if arm == "linear":
        return LinearKeyframeStepper(device=device, **{k: v for k, v in kw.items() if k == "horizon"})
    run_dir = Path(run_dir)
    mgr = CheckpointManager(run_dir)
    path = run_dir / "best.pt"
    cfg = mgr.peek(path).get("config", {}).get("model", {})
    if untrained:
        torch.manual_seed(untrained_seed)
    if arm == "keyframe":
        from caredex.models.keyframe_prior import KeyframeConfig, KeyframePrior
        keys = KeyframeConfig.__dataclass_fields__.keys()
        model = KeyframePrior(KeyframeConfig(**{k: v for k, v in cfg.items() if k in keys}))
        if not untrained:
            mgr.load(model, path=path, restore_rng=False)
        return KeyframeStepper(model, device=device, **{k: v for k, v in kw.items() if k == "horizon"})
    if arm == "modular":
        keys = ModularConfig.__dataclass_fields__.keys()
        model = ModularPrimitivePrior(ModularConfig(**{k: v for k, v in cfg.items() if k in keys}))
        if not untrained:
            mgr.load(model, path=path, restore_rng=False)
        return ModularStepper(model, device=device, **kw)
    if arm == "continuous":
        keys = PriorConfig.__dataclass_fields__.keys()
        model = LatentActionPrior(PriorConfig(**{k: v for k, v in cfg.items() if k in keys}))
        if not untrained:
            mgr.load(model, path=path, restore_rng=False)
        return ContinuousStepper(model, device=device, **kw)
    raise ValueError(f"unknown arm {arm!r}")
