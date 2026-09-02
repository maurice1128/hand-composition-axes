"""Primitives as velocity fields — an ADOPTED formulation, used as an instrument.

Provenance, stated first so it is not mistaken for a claim
----------------------------------------------------------
This is a switching nonlinear dynamical system. A bank of learned velocity
fields with per-frame discrete assignment is not new and is not presented as
new:

* switching linear dynamical systems for motion — Pavlovic et al., NIPS 2000;
* recurrent SLDS — Linderman et al., AISTATS 2017; SNLDS — Dong et al., ICML 2020;
* velocity fields as motion primitives — DMPs (Ijspeert et al.), SEDS
  (Khansari-Zadeh & Billard 2011), Motion Fields (Lee et al. 2010);
* hard one-hot per-timestep selection over K learned dynamics modules, trained
  end-to-end without segmentation labels — DHAL, arXiv:2503.01842;
* the soft alternative, blending expert *weights* per frame — MANN (Zhang,
  Starke, Komura & Saito, SIGGRAPH 2018), implemented here as ``blend="weight"``
  because it is the baseline this architecture has to beat;
* discrete identity plus a continuous modulating variable — WARHMM, NeurIPS 2022.

"Primitive identity is context-independent by construction" is the *definition*
of this model class, not a property added to it, and must never be written as a
contribution. The reason to use it here is instrumental: a hard switch over a
bank makes assignment and primitive usage directly observable, which is what
lets the modularity hypothesis be measured at all. That is an honest reason to
choose it and not a novelty claim.

The problem this solves
-----------------------
:class:`~caredex.models.modular_prior.ModularPrimitivePrior` decodes with a
shared GRU that receives the primitive embedding as an input token. A GRU
carries hidden state, so what primitive k *does* necessarily depends on
everything that preceded it. Context dependence is not a training failure
there; it is structural. Adding a consistency loss fights the architecture,
which is why it needed a large weight and behaved unstably, and why the
measured identity ratio swung 12x (0.086 to 1.060) across identical runs.

Here a primitive is a **function**, not a token::

    v_t = f_k(q_t, style)
    q_{t+1} = clamp(q_t + v_t)

Primitive k applies the same map wherever it is invoked. Starting states differ,
so trajectories differ -- but the *policy* is identical, which is what "reusable
unit" has to mean. Context independence needs no loss term and no measurement:
it holds by definition.

This is the Dynamic Movement Primitive idea (Ijspeert et al., 2002 onward) and
is claimed as nothing else. What is being tested is not the formulation but
whether a *learned discrete library* of such fields, fitted to hand data, keeps
enough capacity to be a usable prior -- the GRU has far more expressive power,
and giving it up should cost reconstruction quality. That cost is the number
worth reporting.

Two properties follow that the GRU version cannot offer:

* **Composition is exact.** Running primitive 3 then 7 means applying f_3 then
  f_7. Nothing about the sequence is approximated by a sequence model.
* **Primitives are executable.** Each is a closed-loop policy from state to
  velocity, so Phase 4's RL can select among them directly rather than decoding
  latent codes into open-loop chunks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from caredex.hand_model import N_DOF


@dataclass
class FieldConfig:
    window: int = 32
    #: Width of the per-frame signal. 27 anatomical DOF by default; the
    #: contact experiment widens it to 27 + 16 contact regions.
    n_channels: int = N_DOF
    n_primitives: int = 12
    #: Width of each primitive's velocity field MLP. Small on purpose: the
    #: library should carry the structure, not one over-parameterised expert.
    field_hidden: int = 128
    field_layers: int = 2
    #: Continuous within-primitive variation, shared across the window.
    style_dim: int = 4
    #: Encoder width (used only to infer assignments and style; it may be as
    #: large as convenient since it never runs at deployment).
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0
    #: Cap on per-step velocity in normalised units. Bounds how far one step can
    #: move and keeps rollouts from diverging; 0 disables.
    max_step: float = 0.25
    bounded_state: bool = True
    #: How the assignment combines the K experts.
    #:
    #: ``"output"`` evaluates every field and mixes their velocities -- with
    #: near-one-hot assignments this is a hard switch, which is the switching
    #: nonlinear dynamical system formulation (Pavlovic et al. 2000; rSLDS,
    #: AISTATS 2017; SNLDS, ICML 2020; DHAL, arXiv:2503.01842). Adopted, not
    #: proposed.
    #:
    #: ``"weight"`` is Mode-Adaptive Neural Networks (Zhang, Starke, Komura &
    #: Saito, SIGGRAPH 2018): the assignment blends the experts' *weight
    #: tensors*, synthesising one interpolated network per frame, and only then
    #: runs a single forward pass. This is a materially different inductive
    #: bias -- a convex combination of parameters, not of outputs -- and it is
    #: the mandatory baseline for any claim that discrete switching buys
    #: something soft gating does not. A prior-art review was explicit that
    #: omitting it makes the comparison rejectable on its own.
    blend: str = "output"

    beta: float = 1.0
    free_bits: float = 0.02
    smoothness_weight: float = 0.1
    assignment_smooth_weight: float = 0.05
    entropy_weight: float = 0.02
    usage_entropy_weight: float = 0.5
    load_balance_weight: float = 0.5
    gumbel_tau: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class PrimitiveFieldBank(nn.Module):
    """K independent velocity fields, evaluated in parallel via grouped weights.

    A loop over K small MLPs would be correct and slow. Batched weight tensors
    compute all primitives at once and then select, which matters because the
    fields are evaluated at every timestep of every rollout.
    """

    def __init__(self, cfg: FieldConfig) -> None:
        super().__init__()
        self.cfg = cfg
        k = cfg.n_primitives
        dims = ([cfg.n_channels + cfg.style_dim]
                + [cfg.field_hidden] * cfg.field_layers + [cfg.n_channels])

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            w = torch.randn(k, a, b) * (1.0 / a**0.5)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(k, b)))
        # Start near zero velocity so early training does not fling the state
        # around before the assignments mean anything.
        with torch.no_grad():
            self.weights[-1].mul_(0.01)

    def forward(self, q: torch.Tensor, style: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``q`` ``(B, 27)``, ``style`` ``(B, S)``, ``weights`` ``(B, K)`` -> ``(B, 27)``."""
        h = torch.cat([q, style], dim=-1)

        if self.cfg.blend == "weight":
            # MANN: blend the expert parameters, then run one network.
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                w_mix = torch.einsum("bk,kde->bde", weights, w)
                b_mix = weights @ b
                h = torch.einsum("bd,bde->be", h, w_mix) + b_mix
                if i < len(self.weights) - 1:
                    h = F.silu(h)
            v = h
        else:
            # Switching: evaluate every field, mix the velocities. Near-one-hot
            # assignments make this a selection among fixed policies.
            h = h.unsqueeze(1).expand(-1, self.cfg.n_primitives, -1)
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                h = torch.einsum("bkd,kde->bke", h, w) + b
                if i < len(self.weights) - 1:
                    h = F.silu(h)
            v = torch.einsum("bk,bkd->bd", weights, h)

        if self.cfg.max_step > 0:
            v = self.cfg.max_step * torch.tanh(v / self.cfg.max_step)
        return v


class FieldPrimitivePrior(nn.Module):
    """Modular prior whose primitives are velocity fields."""

    def __init__(self, cfg: FieldConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FieldConfig()
        c = self.cfg
        drop = c.dropout if c.n_layers > 1 else 0.0

        self.encoder_rnn = nn.GRU(
            input_size=c.n_channels, hidden_size=c.hidden_dim, num_layers=c.n_layers,
            batch_first=True, bidirectional=True, dropout=drop,
        )
        self.to_assignment = nn.Linear(2 * c.hidden_dim, c.n_primitives)
        self.to_style = nn.Linear(2 * c.hidden_dim, 2 * c.style_dim)
        self.bank = PrimitiveFieldBank(c)

    # -- core ---------------------------------------------------------------

    def encode(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        if window.dim() != 3 or window.shape[-1] != self.cfg.n_channels:
            raise ValueError(
                f"expected (B, T, {self.cfg.n_channels}), got {tuple(window.shape)}")
        h, _ = self.encoder_rnn(window)
        mu, logvar = self.to_style(h.mean(dim=1)).chunk(2, dim=-1)
        return {"logits": self.to_assignment(h), "mu": mu, "logvar": logvar.clamp(-10.0, 10.0)}

    def assign(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        if self.training and self.cfg.gumbel_tau > 0:
            return F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=hard, dim=-1)
        return F.softmax(logits, dim=-1)

    def rollout(
        self, weights: torch.Tensor, style: torch.Tensor, initial_pose: torch.Tensor
    ) -> torch.Tensor:
        """Integrate the selected fields forward from ``initial_pose``.

        Autoregressive by nature: each step's velocity depends on the state the
        previous steps produced. That is what makes the primitives executable
        policies rather than a decoded chunk, and it is also why errors compound
        -- a field prior is judged on rollout, not on teacher-forced prediction.
        """
        T = weights.shape[1]
        q = initial_pose
        out = [q]
        for t in range(T - 1):
            v = self.bank(q, style, weights[:, t])
            q = q + v
            if self.cfg.bounded_state:
                q = torch.tanh(q)
            out.append(q)
        return torch.stack(out, dim=1)

    def forward(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encode(window)
        w = self.assign(enc["logits"])
        style = enc["mu"]
        if self.training:
            style = style + torch.randn_like(style) * torch.exp(0.5 * enc["logvar"])
        return {**enc, "weights": w, "style": style,
                "recon": self.rollout(w, style, window[:, 0])}

    # -- sampling and composition -------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n: int,
        initial_pose: torch.Tensor | None = None,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        sequence: list[int] | None = None,
    ) -> torch.Tensor:
        """Roll out a named primitive sequence. This is the composition interface."""
        device = device or next(self.parameters()).device
        c = self.cfg
        if initial_pose is None:
            initial_pose = torch.zeros(n, self.cfg.n_channels, device=device)
        elif initial_pose.dim() == 1:
            initial_pose = initial_pose.unsqueeze(0).expand(n, -1)

        style = temperature * torch.randn(n, c.style_dim, device=device, generator=generator)
        weights = torch.zeros(n, c.window, c.n_primitives, device=device)
        if sequence:
            bounds = torch.linspace(0, c.window, len(sequence) + 1).round().long()
            for i, prim in enumerate(sequence):
                weights[:, bounds[i] : bounds[i + 1], prim] = 1.0
        else:
            idx = torch.randint(0, c.n_primitives, (n,), device=device, generator=generator)
            weights[torch.arange(n, device=device), :, idx] = 1.0
        return self.rollout(weights, style, initial_pose)

    @torch.no_grad()
    def field_at(self, q: torch.Tensor, primitive: int, style: torch.Tensor | None = None):
        """The velocity primitive ``k`` applies at states ``q``.

        Exists so the library can be *inspected*: a primitive is a named policy,
        and its field can be plotted, compared, and reused without running the
        encoder at all.
        """
        device = next(self.parameters()).device
        q = q.to(device)
        if style is None:
            style = torch.zeros(len(q), self.cfg.style_dim, device=device)
        w = torch.zeros(len(q), self.cfg.n_primitives, device=device)
        w[:, primitive] = 1.0
        return self.bank(q, style, w)

    # -- loss ---------------------------------------------------------------

    def loss(
        self, window: torch.Tensor, beta: float | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.forward(window)
        recon, mu, logvar, w = out["recon"], out["mu"], out["logvar"], out["weights"]

        recon_loss = F.mse_loss(recon, window)

        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        flat = kl_per_dim.reshape(-1, kl_per_dim.shape[-1])
        kl_raw = kl_per_dim.sum(dim=-1).mean()
        kl_clamped = torch.clamp(flat.mean(dim=0), min=self.cfg.free_bits).sum()

        dv_true = window[:, 1:] - window[:, :-1]
        dv_pred = recon[:, 1:] - recon[:, :-1]
        smooth_loss = F.mse_loss(dv_pred, dv_true)

        switch = (w[:, 1:] - w[:, :-1]).abs().sum(-1).mean()
        eps = 1e-8
        entropy = -(w * (w + eps).log()).sum(-1).mean()
        usage = w.mean(dim=(0, 1))
        usage_entropy = -(usage * (usage + eps).log()).sum()

        c = self.cfg
        k = c.n_primitives
        hard = w.argmax(-1)
        counts = F.one_hot(hard, k).float().mean(dim=(0, 1))
        balance = k * (counts * usage).sum()

        b = c.beta if beta is None else beta
        total = (
            recon_loss + b * kl_clamped
            + c.smoothness_weight * smooth_loss
            + c.assignment_smooth_weight * switch
            + c.entropy_weight * entropy
            - c.usage_entropy_weight * usage_entropy
            + c.load_balance_weight * balance
        )

        with torch.no_grad():
            switches = (hard[:, 1:] != hard[:, :-1]).float().sum(-1).mean()
            n_used = (usage > 0.01).float().sum()
            # How much motion the fields actually produce. A bank that has
            # collapsed to near-zero velocity reconstructs a static hand and
            # would otherwise look fine on the modularity diagnostics.
            step = (recon[:, 1:] - recon[:, :-1]).abs().mean()

        return total, {
            "loss": float(total.detach()), "recon": float(recon_loss.detach()),
            "kl": float(kl_raw.detach()), "kl_clamped": float(kl_clamped.detach()),
            "smooth": float(smooth_loss.detach()), "beta": float(b),
            "assign_entropy": float(entropy.detach()),
            "usage_entropy": float(usage_entropy.detach()),
            "load_balance": float(balance.detach()),
            "primitives_used": float(n_used),
            "switches_per_window": float(switches),
            "mean_step": float(step),
            "active_units": float((flat.mean(dim=0) > 0.01).float().mean()),
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
