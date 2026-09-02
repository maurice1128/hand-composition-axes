"""Core experiment: does modularity buy compositional generalisation, and does
the advantage grow as data shrinks?

The hypothesis, stated so it can fail
-------------------------------------
MotionBricks gets its result from 350,000 motion clips. Hand datasets have
~800 sequences. If a modular primitive prior is merely a different way to write
the same model, it will match the monolithic prior at every data budget and
there is no paper. The claim worth making is narrower and testable:

    A modular prior generalises to UNSEEN COMPOSITIONS of primitives better
    than a monolithic latent prior, and the gap WIDENS as the data budget
    shrinks.

Design
------
Trajectories are labelled by the primitive sequence that generated them, so
transitions (ordered primitive pairs) can be held out:

    train        trajectories containing no held-out transition
    test_seen    held-out trajectories whose transitions were all seen  (control)
    test_unseen  held-out trajectories containing >=1 held-out transition

``test_seen`` is the control that makes the result interpretable: it isolates
ordinary held-out error from *compositional* error. Reporting only
``test_unseen`` would confound the two.

Each model is then trained at several fractions of the training set, and the
compositional gap (unseen - seen) is compared.

    python scripts/experiment_data_efficiency.py --quick
    python scripts/experiment_data_efficiency.py --epochs 120

Caveat, stated up front: on synthetic data the compositional structure is one
this repo built in, so a positive result there is a sanity check, not evidence.
The real test is OakInk (--bundle), where nobody chose the structure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.pipeline import WindowedTrajectoryDataset  # noqa: E402
from caredex.models.field_prior import FieldConfig, FieldPrimitivePrior  # noqa: E402
from caredex.models.latent_prior import LatentActionPrior, PriorConfig  # noqa: E402
from caredex.models.modular_prior import ModularConfig, ModularPrimitivePrior  # noqa: E402
from caredex.train.trainer import PriorTrainer, TrainConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Compositional split
# ---------------------------------------------------------------------------


def transitions_of(label: str) -> list[tuple[str, str]]:
    """Ordered primitive pairs in a synthetic label like ``a->b->c``."""
    parts = label.split("->")
    return [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]


def compositional_split(
    bundle: TrajectoryBundle, holdout_frac: float = 0.25, seed: int = 0
) -> tuple[list[int], list[int], list[int], dict]:
    """Split trajectory indices into (train, test_seen, test_unseen).

    Held-out transitions are chosen at random, then every trajectory containing
    one is removed from training. The remaining trajectories are split into
    train and a same-distribution control set.
    """
    rng = np.random.default_rng(seed)
    all_trans = sorted({t for lab in bundle.labels for t in transitions_of(lab)})
    if len(all_trans) < 4:
        raise ValueError(
            f"only {len(all_trans)} distinct transitions in this bundle; "
            "a compositional split needs labels of the form 'a->b->c' "
            "(the synthetic source produces them)"
        )

    n_hold = max(1, int(round(holdout_frac * len(all_trans))))
    held = {all_trans[i] for i in rng.choice(len(all_trans), n_hold, replace=False)}

    contaminated, clean = [], []
    for i, lab in enumerate(bundle.labels):
        (contaminated if held & set(transitions_of(lab)) else clean).append(i)

    if len(clean) < 20:
        raise ValueError(
            f"holdout_frac={holdout_frac} left only {len(clean)} clean trajectories; "
            "lower it or generate more data"
        )

    clean = list(rng.permutation(clean))
    n_control = max(10, int(0.15 * len(clean)))
    test_seen, train = clean[:n_control], clean[n_control:]

    info = {
        "n_transitions_total": len(all_trans),
        "n_transitions_held_out": len(held),
        "held_out_transitions": sorted(f"{a}->{b}" for a, b in held),
        "n_train": len(train),
        "n_test_seen": len(test_seen),
        "n_test_unseen": len(contaminated),
    }
    return train, test_seen, contaminated, info


def subset(bundle: TrajectoryBundle, idx: list[int]) -> TrajectoryBundle:
    return TrajectoryBundle(
        trajectories=[bundle.trajectories[i] for i in idx],
        fps=bundle.fps,
        labels=[bundle.labels[i] for i in idx],
        meta=bundle.meta,
        # Auxiliary channels have to follow the same indices or a split would
        # pair one trajectory's pose with another's contact.
        aux=None if bundle.aux is None else [bundle.aux[i] for i in idx],
        aux_names=bundle.aux_names,
    )


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


#: Per-dimension free-bits floor for ``perframe_rate``: 12 x 0.186 = 2.23 nats
#: per frame, the modular arm's measured assignment rate. The default 0.02 gives
#: a 0.24-nat floor, which is where the unmatched baseline sits -- at the floor,
#: with the penalty active, not below it with the penalty off.
RATE_MATCHED_FREE_BITS = 0.186

#: Weight on the modular arm's categorical assignment KL for ``modular_rate``.
#: Equal to beta, so the assignment channel is charged for its ~2.3 nats per
#: frame as the baseline's latent is charged for its 0.21.
#:
#: Note what this does to the bank-shaping terms: the loss already carries
#: ``+ entropy_weight * entropy`` at 0.02, and this adds ``+ w * (ln K -
#: entropy)``, so the net per-frame-entropy coefficient goes 0.02 -> -0.98. That
#: is a sign flip, not an addition, and it pushes assignments toward uniform for
#: reasons unrelated to composition. ``rate_matched_control.sh`` therefore
#: declares a second void condition on reconstruction error; without it a
#: degraded modular arm would look like a controlled one.
ASSIGNMENT_KL_WEIGHT = 1.0


def build_model(kind: str, window: int, latent_dim: int, hidden: int, n_prim: int,
                n_channels: int | None = None):
    # None keeps each config's own default -- the 27 anatomical DOF -- so every
    # existing caller and every finished run is unaffected. The contact
    # experiment passes 43 (27 joints + 16 contact regions).
    width = {} if n_channels is None else {"n_channels": n_channels}
    if kind == "monolithic":
        return LatentActionPrior(
            PriorConfig(window=window, latent_dim=latent_dim, hidden_dim=hidden, **width)
        )
    if kind in ("perframe", "perframe_rate"):
        # Capacity-matched control -- and NOT rate-matched, which is a different
        # thing and the weaker of the two.
        #
        # The window-level monolithic model squeezes 32 frames through one 12-D
        # code while the modular model emits an assignment every frame; matched
        # parameter counts hide that gap. Giving the baseline a per-frame latent
        # equalises how often a code is emitted, and on *nominal* capacity leaves
        # the baseline ahead: 12 continuous dimensions against a 12-way choice.
        #
        # An earlier version of this comment said any remaining difference was
        # therefore attributable to modularity "rather than to the number of bits
        # reaching the decoder". Measured, that is false and backwards. Final-epoch
        # validation, three sweeps:
        #
        #     sweep         baseline KL   modular usage-minus-assign   ratio
        #     category axis    0.207            2.275                  11.0x
        #     pc_hard          0.223            2.399                  10.8x
        #     pc_easy          0.219            2.230                  10.2x
        #
        # The baseline sits at its own free-bits floor (12 x 0.02 = 0.24 nats) while
        # the assignment channel carries no KL term at all. So the modular arm has
        # roughly ten times the realised per-frame rate, in the direction that
        # *creates* a modular advantage rather than hiding one -- and on ``pc_easy``,
        # built to contain no compositional structure, it still wins 5 of 5.
        #
        # Two readings of this were written here before and both were wrong; the
        # measured position is below, and it took three attempts to get right.
        #
        # WRONG #1: "the baseline is penalised and the assignment channel is not,
        # so any residual asymmetry runs against the modular arm." The direction is
        # backwards -- see the table above.
        #
        # WRONG #2: "the baseline's KL penalty is inactive, since its per-dimension
        # KL (0.017 to 0.019) sits below the 0.02 free-bits floor, so raising the
        # floor controls nothing." That reasoning divides a *summed* KL by 12 and
        # then reasons about individual dimensions. ``clamp`` in
        # ``latent_prior.py`` is applied per dimension, so a dimension-average
        # below the floor does not mean every dimension is. ``kl_clamped`` exceeds
        # 12 x 0.02 = 0.24 -- i.e. at least one dimension is above the floor and
        # carrying gradient -- in 102/140, 45/46 and 19/20 of the baselines at the
        # final training epoch. And ``val_active_units`` is 0.996 to 1.000, so the
        # latent is not collapsed either. The penalty is ACTIVE and binding at its
        # floor on most dimensions.
        #
        # So both controls are legitimate and they move the rate from opposite
        # sides. ``perframe_rate`` raises the baseline's free-bits floor to
        # ``RATE_MATCHED_FREE_BITS``; ``modular_rate`` charges the assignment
        # channel the categorical KL a properly specified model of that form would
        # carry. ``scripts/rate_matched_control.sh`` runs both, because a single
        # arm cannot distinguish "the rate gap explains the effect" from "this
        # particular intervention broke one model".
        free_bits = RATE_MATCHED_FREE_BITS if kind == "perframe_rate" else 0.02
        return LatentActionPrior(
            PriorConfig(
                window=window,
                latent_dim=latent_dim,
                hidden_dim=hidden,
                per_frame_latent=True,
                free_bits=free_bits,
                **width,
            )
        )
    if kind in ("modular", "modular_consist", "modular_rate"):
        # style_dim is deliberately small: the discrete bank should carry the
        # structure. Parameter counts are reported so the comparison is fair.
        #
        # `modular_consist` adds the contrastive context-consistency term. Its
        # weight is 0.003, the point on the sweep where the identity ratio
        # crosses 1.0 without a reconstruction penalty. That the *identity
        # ratio* improves is not evidence -- the loss optimises it directly.
        # The independent test is whether compositional generalisation follows,
        # which is what this experiment measures.
        return ModularPrimitivePrior(
            ModularConfig(
                window=window,
                n_primitives=n_prim,
                style_dim=max(2, latent_dim // 3),
                hidden_dim=hidden,
                consistency_weight=0.003 if kind == "modular_consist" else 0.0,
                # ``modular_rate`` charges the assignment channel for the ~2.3
                # nats per frame it carries, which is the control Sec III-B of
                # the paper says the comparison lacks.
                assignment_kl_weight=(
                    ASSIGNMENT_KL_WEIGHT if kind == "modular_rate" else 0.0
                ),
                **width,
            )
        )
    if kind == "mann":
        # Soft weight-space gating (Zhang et al., SIGGRAPH 2018). The mandatory
        # control for any claim that hard switching over a bank buys something
        # soft blending does not: same K experts, same assignment network, only
        # the combination rule differs.
        return FieldPrimitivePrior(
            FieldConfig(
                window=window,
                n_primitives=n_prim,
                style_dim=max(2, latent_dim // 3),
                hidden_dim=hidden,
                blend="weight",
                **width,
            )
        )
    if kind == "field":
        # Primitives as velocity fields: context independence holds by
        # construction rather than being measured (verified exactly zero
        # deviation in scripts/check_field_prior.py). Expected to cost
        # reconstruction quality, since K small MLPs have far less capacity
        # than a shared GRU -- that cost is the point of measuring it.
        return FieldPrimitivePrior(
            FieldConfig(
                window=window,
                n_primitives=n_prim,
                style_dim=max(2, latent_dim // 3),
                hidden_dim=hidden,
                **width,
            )
        )
    raise ValueError(f"unknown model kind {kind!r}")


@torch.no_grad()
def recon_mse(model, loader: DataLoader, device: torch.device,
              channels: slice | None = None) -> float:
    """Reconstruction MSE in normalised units — the comparable quantity.

    ``channels`` restricts the error to part of the feature vector. The contact
    experiment needs this: a 43-channel model and a 27-channel one cannot be
    compared on their raw MSE, but both can be scored on the 27 pose channels,
    and the 43-channel model can be scored on its 16 contact channels alone.
    That split is the whole question -- whether the compositional signal is in
    the joint angles or in what the hand touches.
    """
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred, target = out["recon"], batch
        if channels is not None:
            pred, target = pred[..., channels], target[..., channels]
        err = torch.nn.functional.mse_loss(pred, target, reduction="sum")
        total += float(err)
        n += target.numel()
    return total / max(n, 1)


def run_one(
    kind: str,
    frac: float,
    train_b: TrajectoryBundle,
    seen_b: TrajectoryBundle,
    unseen_b: TrajectoryBundle,
    args,
    device: torch.device,
    out_root: Path,
) -> dict:
    rng = np.random.default_rng(args.seed)
    n_keep = max(4, int(round(frac * len(train_b.trajectories))))
    keep = rng.choice(len(train_b.trajectories), n_keep, replace=False)
    budget_b = subset(train_b, list(keep))

    def make_loader(b: TrajectoryBundle, stride: int, shuffle: bool) -> DataLoader:
        ds = WindowedTrajectoryDataset(b.trajectories, args.window, stride, b.labels)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(budget_b, args.stride, True)
    # A validation split carved from the budget itself: using test data to pick
    # the checkpoint would leak the thing being measured.
    n_val = max(2, int(0.15 * len(budget_b.trajectories)))
    val_loader = make_loader(subset(budget_b, list(range(n_val))), args.window // 2, False)
    seen_loader = make_loader(seen_b, args.window // 2, False)
    unseen_loader = make_loader(unseen_b, args.window // 2, False)

    model = build_model(kind, args.window, args.latent_dim, args.hidden, args.n_primitives)
    run_dir = out_root / f"{kind}_frac{frac:g}"
    trainer = PriorTrainer(
        model,
        train_loader,
        val_loader,
        run_dir=run_dir,
        cfg=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            kl_warmup_epochs=max(1, args.epochs // 10),
            # See experiment_paired_composition.py: tagged epoch snapshots cost
            # gigabytes across a sweep and add nothing over last.pt/best.pt.
            save_every=args.epochs + 1,
            log_every=0,
            amp=False,  # these models are tiny; amp adds noise, not speed
            seed=args.seed,
        ),
        device=device,
        data_meta={"kind": kind, "frac": frac, "n_train_traj": n_keep},
    )

    t0 = time.time()
    state = trainer.fit(resume=not args.fresh)
    elapsed = time.time() - t0

    # Score the best checkpoint, not the last one.
    if trainer.ckpt.best_path.exists():
        trainer.ckpt.load(model, path=trainer.ckpt.best_path, restore_rng=False)
        model.to(device)

    seen = recon_mse(model, seen_loader, device)
    unseen = recon_mse(model, unseen_loader, device)

    result = {
        "kind": kind,
        "frac": frac,
        "n_train_traj": n_keep,
        "n_train_windows": len(train_loader.dataset),
        "n_parameters": model.n_parameters,
        "epochs": args.epochs,
        "seconds": round(elapsed, 1),
        "best_val": state.best_metric,
        "mse_test_seen": seen,
        "mse_test_unseen": unseen,
        # The quantity the hypothesis is about: extra error attributable to
        # composition, normalised so budgets are comparable.
        "compositional_gap": unseen - seen,
        "compositional_ratio": unseen / seen if seen > 0 else float("nan"),
    }
    if hasattr(model, "cfg") and hasattr(model.cfg, "n_primitives"):
        diag = trainer.evaluate(val_loader)
        result["primitives_used"] = diag.get("primitives_used")
        result["switches_per_window"] = diag.get("switches_per_window")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/synthetic.npz")
    ap.add_argument("--out", default="runs/data_efficiency")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.0625, 0.125, 0.25, 0.5, 1.0])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-primitives", type=int, default=12)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--quick", action="store_true", help="tiny run to check plumbing")
    ap.add_argument(
        "--kinds",
        nargs="*",
        default=["monolithic", "perframe", "modular"],
        choices=["monolithic", "perframe", "modular", "modular_consist", "field", "mann"],
        help="'perframe' is the bandwidth-matched control; drop it only knowingly",
    )
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.fracs, args.stride = 6, [0.125, 1.0], 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = TrajectoryBundle.load(args.bundle)
    train_idx, seen_idx, unseen_idx, info = compositional_split(
        bundle, args.holdout_frac, args.seed
    )

    print(f"bundle: {args.bundle}  ({len(bundle.trajectories)} trajectories)")
    print(json.dumps(info, indent=2)[:1200])
    print(f"device: {device}\n")

    train_b = subset(bundle, train_idx)
    seen_b = subset(bundle, seen_idx)
    unseen_b = subset(bundle, unseen_idx)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for frac in args.fracs:
        for kind in args.kinds:
            print(f"\n{'=' * 70}\n{kind}  frac={frac:g}\n{'=' * 70}")
            results.append(
                run_one(kind, frac, train_b, seen_b, unseen_b, args, device, out_root)
            )
            (out_root / "results.json").write_text(
                json.dumps({"split": info, "args": vars(args), "results": results}, indent=2),
                encoding="utf-8",
            )

    width = 62
    print("\n\n" + "=" * width)
    print("RESULT — gap = MSE(unseen compositions) - MSE(seen compositions)")
    print("=" * width)
    print(f"{'frac':>7} {'n_traj':>7} {'model':>11} {'seen':>9} {'unseen':>9} {'gap':>10}  notes")
    print("-" * width)

    for frac in args.fracs:
        row = {r["kind"]: r for r in results if r["frac"] == frac}
        for kind in args.kinds:
            r = row.get(kind)
            if r is None:
                continue
            note = ""
            if kind == "modular":
                used = r.get("primitives_used") or 0
                note = (
                    "VOID: bank collapsed" if used <= 1.5
                    else f"prims={used:.0f} switch={r.get('switches_per_window', 0):.1f}"
                )
            print(
                f"{frac:>7.4g} {r['n_train_traj']:>7} {kind:>11} "
                f"{r['mse_test_seen']:>9.5f} {r['mse_test_unseen']:>9.5f} "
                f"{r['compositional_gap']:>10.5f}  {note}"
            )
        print()

    print("HOW TO READ THIS")
    print("  The hypothesis is that the modular gap is smaller AND the advantage grows")
    print("  as frac shrinks. Two ways the comparison can be void rather than negative:")
    print("   * a collapsed primitive bank -- the modular model was monolithic in disguise;")
    print("   * a baseline whose absolute error is FLAT across budgets -- it never learned,")
    print("     so its near-zero gap means 'cannot distinguish', not 'generalises well'.")
    print("  Check the seen column across budgets before reading any gap.")
    print(f"\nfull results -> {out_root / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
