"""A keyframe-conditioned hand-motion in-betweener, after MotionBricks.

MotionBricks' bricks are not motion clips. Its backbone is an in-betweener:
given a few context frames and a sparse set of target keyframes at arbitrary
times, it fills in the motion, and during training "a random number of
keyframe constraints are sampled at arbitrary temporal positions" so that
any constraint density works. Its smart primitives only *produce* keyframes.
Composition happens at the level of goals; the network owns the how.

This module is that backbone for a hand: a GRU decoder conditioned per frame
on the window's initial pose, the pose of the next keyframe ahead (masked
per joint, so a keyframe may specify only some joints), the mask, and the
time remaining to that keyframe. Frames after the last keyframe see it with
zero time remaining, which teaches the model to hold. No latent code: the
goal is the interface. A policy that acts through it emits a target hand
pose every few frames and the decoder produces the data's way of getting
there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from caredex.hand_model import N_ARTICULATED, N_DOF


@dataclass
class KeyframeConfig:
    window: int = 32
    n_channels: int = N_DOF
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0
    bounded_output: bool = True
    #: Keyframes sampled per training window, inclusive bounds.
    min_keyframes: int = 1
    max_keyframes: int = 3
    #: Earliest frame a keyframe may sit at (frame 0 is the context pose).
    min_offset: int = 4
    #: Probability a training keyframe specifies every articulated joint;
    #: otherwise each joint is kept with ``joint_keep_prob``.
    full_mask_prob: float = 0.5
    joint_keep_prob: float = 0.7
    #: Extra weight on hitting the keyframes' specified joints.
    keyframe_weight: float = 5.0
    smoothness_weight: float = 0.1
    #: Unused by this model; kept so the shared trainer's ``beta`` schedule
    #: has something to read.
    beta: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class KeyframePrior(nn.Module):
    def __init__(self, cfg: KeyframeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or KeyframeConfig()
        c = self.cfg
        drop = c.dropout if c.n_layers > 1 else 0.0
        # per-frame input: anchor | target*mask | mask | time remaining (0..1)
        self.in_dim = 3 * c.n_channels + 1
        self.decoder_init = nn.Linear(self.in_dim, c.n_layers * c.hidden_dim)
        self.decoder_rnn = nn.GRU(self.in_dim, c.hidden_dim, num_layers=c.n_layers,
                                  batch_first=True, dropout=drop)
        self.to_pose = nn.Linear(c.hidden_dim, c.n_channels)

    # -- core -------------------------------------------------------------------

    def cond(self, anchor: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
             remaining: torch.Tensor) -> torch.Tensor:
        """Build the per-frame conditioning tensor ``(B, T, in_dim)``."""
        T = target.shape[1]
        a = anchor.unsqueeze(1).expand(-1, T, -1)
        return torch.cat([a, target * mask, mask, remaining.unsqueeze(-1)], dim=-1)

    def decode(self, anchor, target, mask, remaining) -> torch.Tensor:
        x = self.cond(anchor, target, mask, remaining)
        h0 = self.decoder_init(x[:, 0]).view(-1, self.cfg.n_layers, self.cfg.hidden_dim).permute(1, 0, 2)
        out, _ = self.decoder_rnn(x, h0.contiguous())
        pose = self.to_pose(out)
        return torch.tanh(pose) if self.cfg.bounded_output else pose

    # -- training ---------------------------------------------------------------

    @torch.no_grad()
    def sample_constraints(self, window: torch.Tensor):
        """Random sparse keyframes for a batch of windows, MotionBricks-style.

        Returns per-frame target ``(B, T, C)``, mask ``(B, T, C)``, remaining
        ``(B, T)`` and a per-frame flag ``(B, T)`` marking keyframe frames.
        """
        B, T, C = window.shape
        c = self.cfg
        dev = window.device
        target = torch.zeros_like(window)
        mask = torch.zeros_like(window)
        remaining = torch.zeros(B, T, device=dev)
        is_key = torch.zeros(B, T, dtype=torch.bool, device=dev)
        n_key = torch.randint(c.min_keyframes, c.max_keyframes + 1, (B,))
        t_idx = torch.arange(T, device=dev)
        for b in range(B):
            pos = torch.randperm(T - c.min_offset)[: int(n_key[b])] + c.min_offset
            pos, _ = torch.sort(pos)
            # per-keyframe joint mask
            if torch.rand(()) < c.full_mask_prob:
                jm = torch.ones(C, device=dev)
            else:
                jm = (torch.rand(C, device=dev) < c.joint_keep_prob).float()
            jm[N_ARTICULATED:] = 0.0  # wrist channels are never specified
            prev = -1
            for p in pos.tolist():
                seg = (t_idx > prev) & (t_idx <= p)
                target[b, seg] = window[b, p]
                mask[b, seg] = jm
                remaining[b, seg] = (p - t_idx[seg]).float() / T
                is_key[b, p] = True
                prev = p
            tail = t_idx > prev
            target[b, tail] = window[b, prev]
            mask[b, tail] = jm
            remaining[b, tail] = 0.0
        return target, mask, remaining, is_key

    def forward(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        target, mask, remaining, is_key = self.sample_constraints(window)
        recon = self.decode(window[:, 0], target, mask, remaining)
        return {"recon": recon, "target": target, "mask": mask, "is_key": is_key}

    def loss(self, window: torch.Tensor, beta: float | None = None):
        out = self.forward(window)
        recon, mask, is_key = out["recon"], out["mask"], out["is_key"]
        recon_loss = F.mse_loss(recon, window)
        key_err = ((recon - window) ** 2 * mask)[is_key]
        key_loss = key_err.sum() / mask[is_key].sum().clamp(min=1.0)
        dv_true = window[:, 1:] - window[:, :-1]
        dv_pred = recon[:, 1:] - recon[:, :-1]
        smooth_loss = F.mse_loss(dv_pred, dv_true)
        total = recon_loss + self.cfg.keyframe_weight * key_loss + self.cfg.smoothness_weight * smooth_loss
        return total, {
            "loss": float(total.detach()), "recon": float(recon_loss.detach()),
            "keyframe": float(key_loss.detach()), "smooth": float(smooth_loss.detach()),
            # The shared trainer logs these two; the in-betweener has no latent.
            "beta": 0.0, "kl": 0.0, "active_units": 1.0,
        }

    # -- sampling (validator compatibility) -------------------------------------------

    @torch.no_grad()
    def sample(self, n: int, initial_pose=None, temperature: float = 1.0, device=None,
               generator=None) -> torch.Tensor:
        """Windows that go from a pose to a random target pose and hold it."""
        device = device or next(self.parameters()).device
        c = self.cfg
        anchor = torch.zeros(n, c.n_channels, device=device) if initial_pose is None else initial_pose
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0).expand(n, -1)
        tgt = torch.rand(n, c.n_channels, device=device, generator=generator) * 2 - 1
        T = c.window
        target = tgt.unsqueeze(1).expand(-1, T, -1)
        mask = torch.zeros_like(target); mask[..., :N_ARTICULATED] = 1.0
        remaining = torch.clamp((T // 2 - torch.arange(T, device=device)).float() / T, min=0).unsqueeze(0).expand(n, -1)
        return self.decode(anchor, target, mask, remaining)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
