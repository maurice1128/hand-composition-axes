"""Modular primitive prior — the hand analogue of a MotionBricks-style backbone.

Why this exists
---------------
LAMP's hand prior is a single continuous latent bottleneck: one vector explains
a whole window, and nothing in the architecture says that two windows built
from the same underlying motions share structure. MotionBricks (SIGGRAPH 2026)
takes the opposite view for whole-body motion — a modular latent backbone whose
primitives compose plug-and-play — and gets its result from 350,000 motion
clips. Note that its primitives are *authored* through an interface, not
learned; what is borrowed here is the modular backbone, and the bank below is
the part under test.

Hand datasets have roughly 800 sequences. Three orders of magnitude less. So
porting the architecture is not the interesting part; the interesting question
is whether **modularity buys compositional generalisation, and whether that
matters most exactly when data is scarce**. MotionBricks never had to make that
argument because it had the clips. Here it is the whole hypothesis, and it is
falsifiable: if this model does not beat the monolithic one on unseen primitive
transitions at small data budgets, there is no result.

Architecture
------------
    encoder  : window -> per-timestep assignment logits over K primitives
                       + one continuous style latent for the window
    bank     : K learned primitive embeddings (the "bricks")
    decoder  : (primitive embedding at t, style, initial pose) -> pose at t

Composition is the assignment *sequence*: the model expresses a window as a
succession of primitives rather than as one point in a latent space. Two
regularisers give the assignments the properties that make them primitives
rather than an arbitrary soft mixture:

* **temporal smoothness** — assignments should change rarely, producing
  segments; without it every timestep picks its own blend and there are no
  reusable units;
* **confidence** — low per-timestep entropy, so a segment commits to one
  primitive instead of averaging several.

Both are reported so a reader can check the primitives are real. A model whose
assignments are neither smooth nor confident has learned a mixture, not a
library, and its "modularity" claim is empty.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from caredex.hand_model import N_DOF


@dataclass
class ModularConfig:
    window: int = 32
    #: Width of the per-frame signal the model reconstructs. Defaults to the 27
    #: anatomical DOF; the contact experiment widens it to 27 + 16 contact
    #: regions. Kept as config rather than read from `hand_model` at import time
    #: so a model trained on one width cannot silently be fed the other.
    n_channels: int = N_DOF
    #: Size of the primitive library. The synthetic generator uses 9 grasp
    #: primitives, so K a little above that is the honest setting for it.
    n_primitives: int = 12
    primitive_dim: int = 32
    #: Continuous within-primitive variation. Deliberately much smaller than
    #: the monolithic model's latent: the discrete bank is meant to carry the
    #: structure, and an oversized style code would let the model bypass it.
    style_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0
    bounded_output: bool = True

    beta: float = 1.0
    free_bits: float = 0.02
    smoothness_weight: float = 0.1
    #: Penalty on assignment changes between adjacent frames. Kept an order of
    #: magnitude below the balancing terms: at 0.5 the cheapest way to avoid
    #: switching is to never switch, and the bank collapses to one primitive.
    #: That is not hypothetical -- it is what the first run of the data
    #: efficiency experiment produced (primitives_used = 1.0), which made the
    #: modular model a monolithic model with extra steps.
    assignment_smooth_weight: float = 0.05
    #: Penalty on per-timestep assignment entropy (drives commitment).
    entropy_weight: float = 0.02
    #: Reward for using the whole library, via the entropy of the marginal usage.
    usage_entropy_weight: float = 0.5
    #: Switch-Transformer style load balancing: K * sum_i (usage_i * prob_i),
    #: minimised at 1.0 by uniform usage. More robust than marginal entropy
    #: alone because it couples the hard assignment counts to the soft
    #: probabilities, so a bank that is nominally used but never actually
    #: selected still gets penalised.
    load_balance_weight: float = 0.5

    #: Weight on the categorical KL of the assignment against a uniform prior,
    #: ``ln K - H(q)``. Zero reproduces every run in ``runs/``; the bank was
    #: written without it, so its assignment channel pays nothing for the ~2.3
    #: nats per frame it carries while the baseline's latent is charged for
    #: 0.21. Set it to ``beta`` for the rate-matched control of
    #: ``scripts/rate_matched_control.sh``.
    assignment_kl_weight: float = 0.0
    #: Gumbel-softmax temperature. 0 disables sampling (plain softmax).
    gumbel_tau: float = 1.0
    #: Weight on the context-consistency loss (0 disables it).
    #:
    #: Nothing else in this objective asks a primitive to *mean the same thing*
    #: in different contexts, and measurement says it does not: decoding the
    #: same primitive after different predecessors and from different initial
    #: poses gives an identity ratio of 0.055 -- context explains 18x more of
    #: the resulting motion than primitive identity does. Segmentation is fine
    #: (boundaries land on real motion changes, ratio 1.72); it is the *labels*
    #: that carry no stable meaning. A bank like that cannot confer
    #: compositional generalisation, which is exactly what the paired
    #: experiments failed to find.
    #:
    #: This term asks the velocity profile a primitive produces to be similar
    #: across contexts, which is the property that makes a unit reusable rather
    #: than a context-dependent code.
    consistency_weight: float = 0.0
    #: Contexts sampled per primitive when computing that loss. Higher is a
    #: better estimate and linearly more expensive.
    consistency_contexts: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


class ModularPrimitivePrior(nn.Module):
    """Sequence model with a discrete primitive bank and a small style latent."""

    def __init__(self, cfg: ModularConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModularConfig()
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
        # Per-timestep head: which primitive is active at each frame.
        self.to_assignment = nn.Linear(2 * c.hidden_dim, c.n_primitives)
        # Window-level head: the continuous residual the bank cannot express.
        self.to_style = nn.Linear(2 * c.hidden_dim, 2 * c.style_dim)

        self.primitives = nn.Parameter(torch.randn(c.n_primitives, c.primitive_dim) * 0.1)

        dec_in = c.primitive_dim + c.style_dim + c.n_channels
        self.decoder_rnn = nn.GRU(
            input_size=dec_in,
            hidden_size=c.hidden_dim,
            num_layers=c.n_layers,
            batch_first=True,
            dropout=drop,
        )
        self.decoder_init = nn.Linear(c.style_dim + c.n_channels, c.n_layers * c.hidden_dim)
        self.to_pose = nn.Linear(c.hidden_dim, c.n_channels)

    # -- core ---------------------------------------------------------------

    def encode(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        if window.dim() != 3 or window.shape[-1] != self.cfg.n_channels:
            raise ValueError(f"expected (B, T, {self.cfg.n_channels}), got {tuple(window.shape)}")
        h, _ = self.encoder_rnn(window)
        logits = self.to_assignment(h)
        pooled = h.mean(dim=1)
        mu, logvar = self.to_style(pooled).chunk(2, dim=-1)
        return {"logits": logits, "mu": mu, "logvar": logvar.clamp(-10.0, 10.0)}

    def assign(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Assignment weights ``(B, T, K)``.

        Gumbel noise during training keeps gradients flowing while pushing the
        assignments toward one-hot; at eval time the argmax-free softmax is used
        so the reported numbers are deterministic.
        """
        if self.training and self.cfg.gumbel_tau > 0:
            return F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=hard, dim=-1)
        return F.softmax(logits, dim=-1)

    def decode(
        self,
        weights: torch.Tensor,
        style: torch.Tensor,
        initial_pose: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, T, K)`` assignments + style + initial pose -> ``(B, T, 27)``."""
        T = weights.shape[1]
        prim = weights @ self.primitives  # (B, T, primitive_dim)
        ctx = torch.cat([style, initial_pose], dim=-1)

        h0 = self.decoder_init(ctx)
        h0 = h0.view(-1, self.cfg.n_layers, self.cfg.hidden_dim).permute(1, 0, 2)
        seq = torch.cat([prim, ctx.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        out, _ = self.decoder_rnn(seq, h0.contiguous())
        poses = self.to_pose(out)
        return torch.tanh(poses) if self.cfg.bounded_output else poses

    def forward(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encode(window)
        weights = self.assign(enc["logits"])
        style = enc["mu"]
        if self.training:
            style = style + torch.randn_like(style) * torch.exp(0.5 * enc["logvar"])
        recon = self.decode(weights, style, window[:, 0])
        return {**enc, "weights": weights, "style": style, "recon": recon}

    # -- sampling -----------------------------------------------------------

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
        """Draw windows. ``sequence`` composes named primitives explicitly.

        Passing ``sequence=[3, 7, 1]`` splits the window into three equal
        segments driven by primitives 3, 7 and 1 — this is the plug-and-play
        composition interface, and it is what a monolithic latent cannot offer.
        """
        device = device or next(self.parameters()).device
        c = self.cfg
        if initial_pose is None:
            initial_pose = torch.zeros(n, c.n_channels, device=device)
        elif initial_pose.dim() == 1:
            initial_pose = initial_pose.unsqueeze(0).expand(n, -1)

        style = temperature * torch.randn(n, c.style_dim, device=device, generator=generator)

        weights = torch.zeros(n, c.window, c.n_primitives, device=device)
        if sequence:
            bounds = torch.linspace(0, c.window, len(sequence) + 1).round().long()
            for k, prim in enumerate(sequence):
                weights[:, bounds[k] : bounds[k + 1], prim] = 1.0
        else:
            idx = torch.randint(
                0, c.n_primitives, (n,), device=device, generator=generator
            )
            weights[torch.arange(n, device=device), :, idx] = 1.0
        return self.decode(weights, style, initial_pose)

    # -- context consistency -------------------------------------------------

    def consistency_loss(
        self, initial_pose: torch.Tensor, n_contexts: int | None = None
    ) -> torch.Tensor:
        """Penalise a primitive's motion depending on what surrounds it.

        For each primitive, decode it after several different predecessors and
        from several different initial poses, take the velocity profile of the
        segment it drives, and penalise the spread across contexts. Velocity
        rather than absolute pose: the pose is dominated by the initial
        condition, so matching poses would be asking every primitive to ignore
        where the hand started, which is not what reusability means.

        This mirrors ``scripts/demo_composition.py``'s identity check, so the
        training signal and the evaluation measure the same quantity -- which
        is a reason to read the resulting identity ratio as "we optimised it"
        rather than as independent confirmation.
        """
        c = self.cfg
        n = n_contexts or c.consistency_contexts
        if n < 2:
            raise ValueError("consistency needs at least 2 contexts")

        device = initial_pose.device
        k = c.n_primitives
        T = c.window
        half = T // 2

        # Reuse real initial poses from the batch so the contexts stay on the
        # data manifold rather than being uniform noise.
        idx = torch.randint(0, initial_pose.shape[0], (n,), device=device)
        inits = initial_pose[idx]                                   # (n, 27)
        preds = torch.randint(0, k, (n,), device=device)

        targets = torch.arange(k, device=device).repeat_interleave(n)  # (k*n,)
        pred_rep = preds.repeat(k)
        init_rep = inits.repeat(k, 1)

        weights = torch.zeros(k * n, T, k, device=device)
        rows = torch.arange(k * n, device=device)
        weights[rows, :half, pred_rep] = 1.0
        weights[rows, half:, targets] = 1.0

        style = torch.zeros(k * n, c.style_dim, device=device)
        decoded = self.decode(weights, style, init_rep)
        seg = decoded[:, half:]
        vel = (seg[:, 1:] - seg[:, :-1]).reshape(k, n, -1)

        centroids = vel.mean(dim=1)                                    # (k, D)
        within = ((vel - centroids[:, None]) ** 2).sum(-1).mean()
        grand = centroids.mean(dim=0, keepdim=True)
        between = ((centroids - grand) ** 2).sum(-1).mean()

        # Contrastive, not just attractive. Penalising within-primitive spread
        # alone has a global minimum at "every primitive produces no motion",
        # and the model finds it: a weight sweep drove between-primitive
        # variance from 0.168 to 0.00087 (200x collapse) while the identity
        # *ratio* rose to 0.995 -- numerator and denominator vanishing together,
        # which looks like success and is not. The between term supplies the
        # negatives that make the collapse unprofitable.
        return within / (between + 1e-8)

    # -- loss ---------------------------------------------------------------

    def loss(
        self, window: torch.Tensor, beta: float | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.forward(window)
        recon, mu, logvar, w = out["recon"], out["mu"], out["logvar"], out["weights"]

        recon_loss = F.mse_loss(recon, window)

        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl_raw = kl_per_dim.sum(dim=-1).mean()
        kl_clamped = torch.clamp(kl_per_dim.mean(dim=0), min=self.cfg.free_bits).sum()

        dv_true = window[:, 1:] - window[:, :-1]
        dv_pred = recon[:, 1:] - recon[:, :-1]
        smooth_loss = F.mse_loss(dv_pred, dv_true)

        # Segment structure: adjacent frames should usually share a primitive.
        switch = (w[:, 1:] - w[:, :-1]).abs().sum(-1).mean()

        eps = 1e-8
        entropy = -(w * (w + eps).log()).sum(-1).mean()
        usage = w.mean(dim=(0, 1))
        usage_entropy = -(usage * (usage + eps).log()).sum()

        # Rate of the assignment channel, in nats per frame: the mutual
        # information between the input and the primitive choice, estimated as
        # H(marginal) - E[H(conditional)]. This is the quantity the baseline
        # pays a KL for and the bank, as originally written, did not.
        #
        # Measured across three sweeps the gap is about tenfold -- baseline
        # 0.207 to 0.223 nats/frame against 2.230 to 2.399 here -- so the two
        # arms were never rate-matched, only capacity-matched, in the direction
        # that favours this model. Note the baseline's own KL penalty is
        # *inactive*: its per-dimension KL (0.017 to 0.019) sits below its
        # free-bits floor of 0.02, so the clamp returns a constant and the
        # gradient is zero. Raising that floor therefore controls nothing;
        # the binding fix is on this side, charging the assignment channel for
        # the information it carries, which is what ``assignment_kl_weight``
        # does. At weight = beta it is the categorical KL of a proper VAE
        # against a uniform prior.
        assignment_kl = usage_entropy.new_tensor(0.0)
        if self.cfg.assignment_kl_weight:
            assignment_kl = math.log(self.cfg.n_primitives) - entropy

        c = self.cfg
        k = c.n_primitives
        hard = w.argmax(-1)
        counts = F.one_hot(hard, k).float().mean(dim=(0, 1))
        balance = k * (counts * usage).sum()

        b = c.beta if beta is None else beta
        total = (
            recon_loss
            + b * kl_clamped
            + c.smoothness_weight * smooth_loss
            + c.assignment_smooth_weight * switch
            + c.entropy_weight * entropy
            - c.usage_entropy_weight * usage_entropy
            + c.load_balance_weight * balance
            + self.cfg.assignment_kl_weight * assignment_kl
        )

        consistency = torch.zeros((), device=window.device)
        if c.consistency_weight > 0:
            consistency = self.consistency_loss(window[:, 0])
            total = total + c.consistency_weight * consistency

        with torch.no_grad():
            switches_per_window = (hard[:, 1:] != hard[:, :-1]).float().sum(-1).mean()
            n_used = (usage > 0.01).float().sum()

        return total, {
            "loss": float(total.detach()),
            "recon": float(recon_loss.detach()),
            "kl": float(kl_raw.detach()),
            "assignment_kl": float(
                assignment_kl.detach() if torch.is_tensor(assignment_kl) else assignment_kl
            ),
            "kl_clamped": float(kl_clamped.detach()),
            "smooth": float(smooth_loss.detach()),
            "beta": float(b),
            # Diagnostics that decide whether the "library" claim is real.
            "assign_entropy": float(entropy.detach()),
            "usage_entropy": float(usage_entropy.detach()),
            "load_balance": float(balance.detach()),
            "consistency": float(consistency.detach()),
            # If this sits at 1.0 the bank has collapsed and the model is
            # monolithic in disguise -- any comparison against a monolithic
            # baseline is then meaningless, not evidence against modularity.
            "primitives_used": float(n_used),
            "switches_per_window": float(switches_per_window),
            "active_units": float((kl_per_dim.mean(dim=0) > 0.01).float().mean()),
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
