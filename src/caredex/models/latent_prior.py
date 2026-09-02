"""Latent action prior over short hand trajectory windows.

ADOPTED METHOD, NOT A CONTRIBUTION. This is a sequence VAE of the kind LAMP
(2026) established for latent action priors, and the paper must describe it as
"we adopt / following LAMP", never as "we propose". The contribution of this
project is the caregiving benchmark and the clinically calibrated safety
formalisation; this module exists so that Phase 4's RL policy has a
low-dimensional, biomechanically plausible action space to act in.

Interface, and why it is shaped this way
----------------------------------------
    encode(window)              -> q(z | window)
    decode(z, initial_pose)     -> window

Conditioning the decoder on the *initial pose* is what makes the prior usable
at RL time: the policy observes the current hand configuration, emits a latent
z, and the decoder expands it into the next ``window`` frames of motion. A
decoder that ignored the current pose would produce chunks that teleport.

Joint limits are enforced structurally by a tanh output when
``bounded_output`` is set (poses live in [-1, 1] after normalisation). That
makes zero limit violations an architectural guarantee, not a learned
behaviour -- do not report it as evidence the model learned anatomy. The
biomechanical claims worth making are about the DIP/PIP coupling and the
smoothness statistics, which are *not* structurally enforced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from caredex.hand_model import N_DOF


@dataclass
class PriorConfig:
    window: int = 32
    #: Width of the per-frame signal the model reconstructs. Defaults to the 27
    #: anatomical DOF; the contact experiment widens it to 27 + 16 contact
    #: regions. Kept as config rather than read from `hand_model` at import time
    #: so a model trained on one width cannot silently be fed the other.
    n_channels: int = N_DOF
    latent_dim: int = 12
    #: One latent per frame instead of one per window.
    #:
    #: The window-level default is what LAMP does and is the honest description
    #: of a monolithic prior. But it makes an unfair baseline for the modular
    #: model, which emits a primitive assignment *every frame*: matched
    #: parameter counts hide a large gap in information bandwidth, and the
    #: first sweep's "modular wins" may have been nothing but extra bits
    #: getting through. Setting this gives the monolithic model a per-frame
    #: latent sequence too, so the comparison isolates modularity (discrete
    #: reusable units) from bandwidth.
    per_frame_latent: bool = False
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0
    bounded_output: bool = True
    condition_on_initial: bool = True
    #: KL weight at the end of annealing (beta-VAE).
    beta: float = 1.0
    #: Nats per latent dimension exempted from the KL penalty. Without this a
    #: sequence VAE with a strong autoregressive-free decoder still tends to
    #: collapse onto the prior and ignore z.
    free_bits: float = 0.02
    #: Weight on a first-difference penalty. The data is 30 fps human motion;
    #: without it the decoder is free to produce jittery chunks that score well
    #: on per-frame MSE but are useless as robot actions.
    smoothness_weight: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class LatentActionPrior(nn.Module):
    """Sequence VAE mapping ``(B, T, 27)`` windows to a ``latent_dim`` code."""

    def __init__(self, cfg: PriorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PriorConfig()
        c = self.cfg
        drop = c.dropout if c.n_layers > 1 else 0.0

        self.encoder_rnn = nn.GRU(
            input_size=c.n_channels,
            hidden_size=c.hidden_dim,
            num_layers=c.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop,
        )
        self.to_latent = nn.Linear(2 * c.hidden_dim, 2 * c.latent_dim)

        dec_in = c.latent_dim + (c.n_channels if c.condition_on_initial else 0)
        self.decoder_rnn = nn.GRU(
            input_size=dec_in,
            hidden_size=c.hidden_dim,
            num_layers=c.n_layers,
            batch_first=True,
            dropout=drop,
        )
        self.decoder_init = nn.Linear(dec_in, c.n_layers * c.hidden_dim)
        self.to_pose = nn.Linear(c.hidden_dim, c.n_channels)

    # -- core -------------------------------------------------------------

    def encode(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, 27)`` -> ``(mu, logvar)``.

        Shape is ``(B, latent_dim)`` normally, ``(B, T, latent_dim)`` when
        ``per_frame_latent`` is set.
        """
        self._check_window(window)
        out, h = self.encoder_rnn(window)
        if self.cfg.per_frame_latent:
            mu, logvar = self.to_latent(out).chunk(2, dim=-1)
            return mu, logvar.clamp(-10.0, 10.0)
        # h is (n_layers * 2, B, H); concatenate the last layer's two directions.
        h = h.view(self.cfg.n_layers, 2, window.shape[0], self.cfg.hidden_dim)[-1]
        h = torch.cat([h[0], h[1]], dim=-1)
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        # Clamped for numerical safety: exp(logvar/2) overflows in fp16 above ~22.
        return mu, logvar.clamp(-10.0, 10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(
        self,
        z: torch.Tensor,
        initial_pose: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        """``(B, latent_dim)`` -> ``(B, T, 27)``."""
        T = window or self.cfg.window
        if z.dim() == 3:
            # Per-frame latent: feed the sequence directly rather than tiling
            # one code across the window.
            T = z.shape[1]
            ctx0 = z[:, 0]
            if self.cfg.condition_on_initial:
                if initial_pose is None:
                    raise ValueError("condition_on_initial=True requires initial_pose")
                ctx0 = torch.cat([ctx0, initial_pose], dim=-1)
                seq = torch.cat([z, initial_pose.unsqueeze(1).expand(-1, T, -1)], dim=-1)
            else:
                seq = z
            h0 = self.decoder_init(ctx0)
            h0 = h0.view(-1, self.cfg.n_layers, self.cfg.hidden_dim).permute(1, 0, 2)
            out, _ = self.decoder_rnn(seq, h0.contiguous())
            poses = self.to_pose(out)
            return torch.tanh(poses) if self.cfg.bounded_output else poses

        if self.cfg.condition_on_initial:
            if initial_pose is None:
                raise ValueError(
                    "condition_on_initial=True requires initial_pose of shape (B, 27)"
                )
            if initial_pose.shape[-1] != self.cfg.n_channels or initial_pose.dim() != 2:
                raise ValueError(
                    f"initial_pose must be (B, {self.cfg.n_channels}), got {tuple(initial_pose.shape)}"
                )
            ctx = torch.cat([z, initial_pose], dim=-1)
        else:
            ctx = z

        h0 = self.decoder_init(ctx)
        h0 = h0.view(-1, self.cfg.n_layers, self.cfg.hidden_dim).permute(1, 0, 2)
        out, _ = self.decoder_rnn(ctx.unsqueeze(1).expand(-1, T, -1), h0.contiguous())
        poses = self.to_pose(out)
        return torch.tanh(poses) if self.cfg.bounded_output else poses

    def forward(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(window)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, window[:, 0], window.shape[1])
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    # -- sampling ----------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n: int,
        initial_pose: torch.Tensor | None = None,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw ``n`` windows from the prior. Used by the ergonomic validator."""
        device = device or next(self.parameters()).device
        shape = (
            (n, self.cfg.window, self.cfg.latent_dim)
            if self.cfg.per_frame_latent
            else (n, self.cfg.latent_dim)
        )
        z = temperature * torch.randn(*shape, device=device, generator=generator)
        if self.cfg.condition_on_initial and initial_pose is None:
            # Zero in normalised units is the midpoint of every joint's range.
            initial_pose = torch.zeros(n, self.cfg.n_channels, device=device)
        elif initial_pose is not None and initial_pose.dim() == 1:
            initial_pose = initial_pose.unsqueeze(0).expand(n, -1)
        return self.decode(z, initial_pose)

    # -- loss --------------------------------------------------------------

    def loss(
        self, window: torch.Tensor, beta: float | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Total loss plus a dict of scalars for logging."""
        out = self.forward(window)
        recon, mu, logvar = out["recon"], out["mu"], out["logvar"]

        recon_loss = F.mse_loss(recon, window)

        # Per-dimension KL with free bits, then summed over dimensions. Leading
        # dims are flattened so a per-frame latent (B, T, D) averages over both
        # batch and time -- summing over T instead would scale the KL term by
        # the window length and silently change the objective.
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        flat = kl_per_dim.reshape(-1, kl_per_dim.shape[-1])
        kl_raw = kl_per_dim.sum(dim=-1).mean()
        kl_clamped = torch.clamp(flat.mean(dim=0), min=self.cfg.free_bits).sum()

        dv_true = window[:, 1:] - window[:, :-1]
        dv_pred = recon[:, 1:] - recon[:, :-1]
        smooth_loss = F.mse_loss(dv_pred, dv_true)

        b = self.cfg.beta if beta is None else beta
        total = recon_loss + b * kl_clamped + self.cfg.smoothness_weight * smooth_loss

        return total, {
            "loss": float(total.detach()),
            "recon": float(recon_loss.detach()),
            "kl": float(kl_raw.detach()),
            "kl_clamped": float(kl_clamped.detach()),
            "smooth": float(smooth_loss.detach()),
            "beta": float(b),
            # Fraction of latent dims carrying signal. Near zero means the
            # posterior collapsed and z is being ignored.
            "active_units": float((flat.mean(dim=0) > 0.01).float().mean()),
        }

    # -- misc ---------------------------------------------------------------

    def _check_window(self, window: torch.Tensor) -> None:
        if window.dim() != 3 or window.shape[-1] != self.cfg.n_channels:
            raise ValueError(
                f"expected window of shape (B, T, {self.cfg.n_channels}), got {tuple(window.shape)}"
            )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
