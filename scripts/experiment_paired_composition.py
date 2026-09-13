"""Paired compositional generalisation, with set difficulty cancelled out.

Why the unpaired design failed
------------------------------
The first design compared ``MSE(unseen compositions) - MSE(seen compositions)``
from a single model. That only measures composition if the two test sets are
otherwise matched, and they are not: ``scripts/check_split_balance.py`` shows
the unseen set's variance differing from the seen set's by -5.9% to +8.7%
depending on the split seed, while the compositional effect being chased is
about 2% of the error. **The confound is three times the signal.** Every model
produced negative gaps on seed 0 for exactly this reason -- the unseen
trajectories were simply easier.

The paired design
-----------------
Evaluate the *same* trajectories under two models that differ only in whether
their training data contained that trajectory's composition:

    target T      trajectories whose composition is held out (never trained on)
    model NAIVE   trained on N trajectories, none with T's compositions
    model INFORMED trained on N trajectories, some with T's compositions
                  (but never T itself)

    compositional penalty = MSE_naive(T) - MSE_informed(T)

Both models see the same number of training trajectories and are scored on
identical test data, so trajectory difficulty, set size and evaluation noise
all cancel. What remains is the cost of never having seen the composition.

The hypothesis then becomes precise: **a modular prior should pay a smaller
compositional penalty than a monolithic one, and the difference should grow as
N shrinks.**

    python scripts/experiment_paired_composition.py --bundle data/bundles/oakink.npz
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import N_DOF  # noqa: E402
from caredex.data.pipeline import WindowedTrajectoryDataset  # noqa: E402
from caredex.train.trainer import PriorTrainer, TrainConfig  # noqa: E402
from experiment_data_efficiency import build_model, recon_mse, subset, transitions_of  # noqa: E402


def coarsen_labels(labels: list[str], mode: str) -> list[str]:
    """Reduce composition granularity so cells hold enough trajectories.

    OakInk has 770 trajectories across 345 ``object->intent`` cells: **2.2
    trajectories per cell**. The paired design has to split those between a
    target set and an informed set, leaving about one example each -- so the
    informed model is never meaningfully informed and the penalty is pinned
    near zero by construction. That is a property of the dataset, not something
    more seeds can fix.

    ``category`` keeps only the leading letter of the object id, giving 20 cells
    at ~38 trajectories each. **It is a bad axis and is kept only so the older
    runs remain reproducible.** The leading character takes five values
    (A, C, O, S, Y) that record which sub-collection an object came from, not
    what it is -- so a result on it says little about manipulation.

    ``oakink_category`` / ``oakink_class`` / ``oakink_attr`` are OakInk's **own**
    taxonomy, read from the ``metaV2.zip`` that shipped with the download and
    resolves all 100 object ids here. Respectively: the dataset's named object
    category (``mug``, ``teapot``, ``trigger_sprayer``, 33 groups), its coarse
    functional class (``container``, ``maniptools``, 5 groups), and its
    affordance attribute (``handled``, ``pourable``, ``squeezable``, 12 groups).
    The last is the most grasp-relevant: what a hand must do with an object is
    closer to its affordance than to its name.

    ``grab_shape_intentclass`` and ``grab_object_intentclass`` coarsen the right
    side as well, replacing GRAB's fine intent with its four documented intent
    classes (:func:`caredex.data.grab.intent_class`). They exist because the
    screen scores them and the sweep could not run them: an axis the screen can
    rank but the experiment cannot measure is a gap between the two, and the
    necessity direction needs exactly these non-positive-interaction axes.

    ``oakink2_<left>_<right>`` crosses a provenance factor of OakInk2 (``scene``
    or ``subject``) with a content factor (the chain's first ``primitive`` or the
    task sentence's leading ``verb``). OakInk2's labels are primitive chains and
    carry neither scene nor subject, so ``screen_oakink2_axes.py --build`` writes
    bundles whose labels are ``<chain>@<scene>@<subject>@<verb>``. The fine label
    is that whole string, so two trajectories are near-duplicates only when the
    same chain was performed in the same scene by the same subject, and the
    coarse label is built from the suffix. Plain ``oakink2.npz`` has no suffix and
    these modes refuse it rather than quietly building a one-cell-per-label axis.

    ``shape`` is the GRAB equivalent: object name -> grasp-relevant shape class
    (:data:`caredex.data.grab.GRAB_SHAPE_CLASS`). GRAB object ids are words, so
    the leading-letter trick would put ``apple`` and ``airplane`` in one cell.
    That one is still constructed here, anchored on the 15 GRAB objects that are
    named geometric primitives.
    """
    if mode == "fine":
        return labels
    if mode == "shape":
        from caredex.data.grab import GRAB_SHAPE_CLASS

        unknown = {l.partition("->")[0] for l in labels} - set(GRAB_SHAPE_CLASS)
        if unknown:
            raise ValueError(
                f"granularity='shape' is GRAB-specific and {len(unknown)} object ids are "
                f"not in GRAB_SHAPE_CLASS: {sorted(unknown)[:5]}"
            )
    if mode.startswith("oakink2_"):
        _, left, right = mode.split("_", 2)
        out = []
        for lab in labels:
            parts = lab.split("@")
            if len(parts) != 4:
                raise ValueError(
                    f"granularity={mode!r} needs labels of the form "
                    f"chain@scene@subject@verb, written by screen_oakink2_axes.py "
                    f"--build; got {lab!r}"
                )
            chain, scene, subject, verb = parts
            fac = {"scene": scene, "subject": subject, "verb": verb,
                   "primitive": chain.partition("->")[0]}
            out.append(f"{fac[left]}->{fac[right]}")
        return out
    if mode.startswith("grab_"):
        from caredex.data.grab import GRAB_SHAPE_CLASS, intent_class

        out = []
        for lab in labels:
            left, _, right = lab.partition("->")
            group = GRAB_SHAPE_CLASS[left] if "shape" in mode else left
            out.append(f"{group}->{intent_class(right)}")
        return out
    if mode.startswith("oakink_"):
        from caredex.data.oakink_meta import coverage, object_group

        cov = coverage([l.partition("->")[0] for l in labels])
        if cov["fraction"] < 1.0:
            raise ValueError(
                f"granularity={mode!r} resolved only {cov['n_resolved']}/{cov['n_ids']} "
                f"object ids in OakInk's metaV2. Unresolved ids would each form "
                f"their own cell and quietly change the axis."
            )
    out = []
    for lab in labels:
        left, _, right = lab.partition("->")
        if mode == "category":
            out.append(f"{left[:1] or left}->{right}")
        elif mode.startswith("oakink_"):
            out.append(f"{object_group(left, mode[len('oakink_'):])}->{right}")
        elif mode == "shape":
            out.append(f"{GRAB_SHAPE_CLASS[left]}->{right}")
        elif mode == "left":
            out.append(f"{left}->any")
        elif mode == "right":
            out.append(f"any->{right}")
        else:
            raise ValueError(f"unknown granularity {mode!r}")
    return out


def discard_checkpoints(run_dir: Path, keep: bool = False) -> int:
    """Delete a finished run's checkpoints once it has been scored.

    A 70-seed sweep trains 280 models and each leaves best.pt and last.pt at
    ~27 MB, so one sweep costs about 15 GB and the C drive filled twice. The
    sweep's product is ``results.json``; the weights are needed only to resume
    or re-score, and neither applies to a run that has just been scored.

    ``--keep-checkpoints`` opts out for anyone who does want to re-score.
    """
    if keep:
        return 0
    n = 0
    for f in run_dir.glob("*.pt"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


def build_paired_split(
    bundle: TrajectoryBundle,
    n_held_compositions: int,
    seed: int = 0,
    target_frac: float = 0.5,
    fine_labels: list[str] | None = None,
    min_chains: int = 4,
) -> dict:
    """Partition trajectories into target / naive pool / informed-only pool.

    ``fine_labels`` must be supplied whenever the split labels have been
    coarsened. Splitting a coarse cell's trajectories at random between the
    target set and the informed pool lets the *same fine composition* land on
    both sides: an audit measured 48-69% of OakInk target trajectories having
    their exact ``object->intent`` pair in the informed model's training set,
    against 0% for the naive model. The measured "compositional penalty" was
    then substantially "did you train on a near-duplicate of the test clip",
    which the naive arm is structurally denied.

    With fine labels available, whole fine groups are assigned to one side or
    the other, so no fine composition can appear in both.
    """
    rng = np.random.default_rng(seed)
    keys_for_pool = fine_labels if fine_labels is not None else bundle.labels

    # Only transitions carried by several DISTINCT fine labels can be held out
    # under a leak-free split. Whole fine groups go to one side or the other, so
    # a transition appearing in a single fine label lands entirely in the target
    # set and the informed model never sees it -- the informed arm is then no
    # better informed than the naive one, and the penalty is pinned near zero
    # by construction.
    #
    # This is a structural requirement, not selection of favourable cases: on
    # OakInk2, 391 transitions exist but only 60 appear in >= 6 distinct chains,
    # and drawing uniformly covered 4-5 of 8 held transitions.
    chains_of: dict[tuple[str, str], set[str]] = {}
    for i, lab in enumerate(bundle.labels):
        for t in transitions_of(lab):
            chains_of.setdefault(t, set()).add(keys_for_pool[i])

    pool = sorted(t for t, chains in chains_of.items() if len(chains) >= min_chains)
    if len(pool) < n_held_compositions:
        raise ValueError(
            f"only {len(pool)} transitions appear in >= {min_chains} distinct groups "
            f"(of {len(chains_of)} total); cannot hold out {n_held_compositions}. "
            f"Lower --held-compositions or --min-chains."
        )
    held = {pool[i] for i in rng.choice(len(pool), n_held_compositions, replace=False)}

    with_held, without_held = [], []
    for i, lab in enumerate(bundle.labels):
        (with_held if held & set(transitions_of(lab)) else without_held).append(i)

    if len(with_held) < 8:
        raise ValueError(f"only {len(with_held)} trajectories carry a held composition")

    keys = fine_labels if fine_labels is not None else bundle.labels
    groups: dict[str, list[int]] = {}
    for i in with_held:
        groups.setdefault(keys[i], []).append(i)

    order = sorted(groups)
    rng.shuffle(order)

    # Fill the target set group by group until it reaches its share, so a fine
    # composition is never split across the two sides.
    n_target_goal = max(4, int(round(target_frac * len(with_held))))
    target: list[int] = []
    informed_extra: list[int] = []
    for key in order:
        if len(target) < n_target_goal:
            target.extend(groups[key])
        else:
            informed_extra.extend(groups[key])

    if not informed_extra or len(target) < 4:
        raise ValueError(
            f"fine-group split left target={len(target)} informed_extra={len(informed_extra)}; "
            f"hold out fewer compositions or coarsen less"
        )

    return {
        "held_compositions": sorted(f"{a}->{b}" for a, b in held),
        "target": target,
        "naive_pool": without_held,
        "informed_extra": informed_extra,
        "n_fine_groups_held": len(groups),
        "fine_disjoint": fine_labels is not None,
        "candidate_pool_size": len(pool),
        "min_chains": min_chains,
    }


def sample_pools(
    split: dict,
    n: int,
    seed: int,
    labels: list[str] | None = None,
    min_per_composition: int = 4,
) -> tuple[list[int], list[int]]:
    """Two training sets of identical size, differing only in composition coverage.

    The informed set is filled **round-robin across held compositions** rather
    than by uniform random draw. Random draws left 75% of held compositions
    absent from the informed model's training data at realistic budgets, which
    made it barely more informed than the naive model and guaranteed a zero
    penalty regardless of architecture -- an artefact, not a finding. Round-robin
    spends the swap budget on breadth first, then depth.
    """
    rng = np.random.default_rng(seed)
    naive_pool = split["naive_pool"]
    extra = split["informed_extra"]
    if n > len(naive_pool):
        raise ValueError(f"budget {n} exceeds naive pool of {len(naive_pool)}")

    naive = list(rng.choice(naive_pool, n, replace=False))

    held = set(split["held_compositions"])
    n_swap = min(len(extra), max(1, n // 2))

    if labels is None:
        chosen = list(rng.choice(extra, n_swap, replace=False))
    else:
        by_comp: dict[str, list[int]] = {}
        for i in extra:
            for a, b in transitions_of(labels[i]):
                key = f"{a}->{b}"
                if key in held:
                    by_comp.setdefault(key, []).append(i)
        for v in by_comp.values():
            rng.shuffle(v)

        chosen, taken = [], set()
        order = sorted(by_comp)
        rng.shuffle(order)
        for depth in range(max(min_per_composition, 1)):
            for key in order:
                if len(chosen) >= n_swap:
                    break
                pool = by_comp[key]
                if depth < len(pool) and pool[depth] not in taken:
                    chosen.append(pool[depth])
                    taken.add(pool[depth])
            if len(chosen) >= n_swap:
                break
        # Top up with anything left if breadth did not consume the budget.
        if len(chosen) < n_swap:
            rest = [i for i in extra if i not in taken]
            rng.shuffle(rest)
            chosen += rest[: n_swap - len(chosen)]

    keep = list(rng.choice(naive_pool, n - len(chosen), replace=False))
    return naive, keep + chosen


def assert_split_sound(
    split: dict,
    naive_idx: list[int],
    informed_idx: list[int],
    fine: list[str],
    labels: list[str],
    tolerance: float = 0.0,
) -> dict:
    """Verify both halves of the paired design before spending any GPU time.

    The design rests on two facts, and a violation of either makes the measured
    penalty mean something other than what it is reported to mean:

    1. the naive arm really has not seen the held compositions, and neither arm
       has trained on a near-duplicate of a target clip (``check_composition_leak``)
    2. the informed arm really has seen them, or the two models are nearly the
       same model and a zero penalty is guaranteed (``check_informed_coverage``)

    Both checks existed as standalone scripts and neither was wired in, so they
    depended on someone remembering to run them after every change of axis. When
    the evidence axis moved to OakInk's official taxonomy they were not re-run,
    and the headline numbers sat ungated until an audit noticed. Splitting
    correctly is claimed in ``build_paired_split``'s docstring; a docstring is
    not a check.

    A leak raises. Thin coverage does not: it biases the penalty *downward*,
    against the effect being claimed, so a run that survives it is conservative
    rather than wrong. It is counted and returned for the record.
    """
    target = split["target"]
    leaks = {}
    for name, idx in (("naive", naive_idx), ("informed", informed_idx)):
        train_fine = {fine[i] for i in idx}
        leaks[name] = sum(1 for i in target if fine[i] in train_fine) / max(len(target), 1)

    if max(leaks.values()) > tolerance + 1e-9:
        raise RuntimeError(
            f"composition leak: naive {leaks['naive']:.1%}, informed "
            f"{leaks['informed']:.1%} of target trajectories have their exact fine "
            f"composition in a training set. The penalty would measure "
            f"near-duplicate memorisation, not composition."
        )

    held = set(split["held_compositions"])
    covered = {
        f"{a}->{b}"
        for i in informed_idx
        for a, b in transitions_of(labels[i])
        if f"{a}->{b}" in held
    }
    return {
        "leak_naive": leaks["naive"],
        "leak_informed": leaks["informed"],
        "held": len(held),
        "covered": len(covered),
        "coverage_thin": len(covered) < len(held),
    }


def train_and_score(
    kind: str,
    train_idx: list[int],
    bundle: TrajectoryBundle,
    target_loader: DataLoader,
    args,
    device: torch.device,
    run_dir: Path,
) -> dict:
    train_b = subset(bundle, train_idx)
    n_aux = 0 if train_b.aux is None else train_b.aux[0].shape[1]
    ds = WindowedTrajectoryDataset(train_b.trajectories, args.window, args.stride,
                                   train_b.labels, aux=train_b.aux)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    n_val = max(2, int(0.15 * len(train_b.trajectories)))
    val_b = subset(train_b, list(range(n_val)))
    val_ds = WindowedTrajectoryDataset(val_b.trajectories, args.window, args.window // 2,
                                       val_b.labels, aux=val_b.aux)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(kind, args.window, args.latent_dim, args.hidden, args.n_primitives,
                        n_channels=None if n_aux == 0 else N_DOF + n_aux)
    trainer = PriorTrainer(
        model, train_loader, val_loader, run_dir=run_dir,
        cfg=TrainConfig(
            epochs=args.epochs, lr=args.lr,
            kl_warmup_epochs=max(1, args.epochs // 10),
            # No periodic epoch snapshots. A sweep is dozens of short runs, and
            # keeping 3 tagged snapshots each cost 11 GB of disk for nothing --
            # last.pt covers resume and best.pt covers scoring.
            save_every=args.epochs + 1,
            log_every=0, amp=False, seed=args.seed,
        ),
        device=device, data_meta={"kind": kind, "n_train": len(train_idx)},
    )
    t0 = time.time()
    trainer.fit(resume=not args.fresh)
    if trainer.ckpt.best_path.exists():
        trainer.ckpt.load(model, path=trainer.ckpt.best_path, restore_rng=False)
        model.to(device)

    out = {
        # Scored on the pose channels in every arm, so a 43-channel run and a
        # 27-channel one are comparable. The contact channels are scored
        # separately -- that split is the experiment.
        "mse_target": recon_mse(model, target_loader, device, channels=slice(0, N_DOF)),
        **({} if n_aux == 0 else {
            "mse_target_contact": recon_mse(
                model, target_loader, device, channels=slice(N_DOF, N_DOF + n_aux)),
            "mse_target_all": recon_mse(model, target_loader, device),
        }),
        "n_train": len(train_idx),
        "seconds": round(time.time() - t0, 1),
        "n_parameters": model.n_parameters,
    }
    if kind == "modular":
        diag = trainer.evaluate(val_loader)
        out["primitives_used"] = diag.get("primitives_used")

    # The weights have served their purpose the moment the metrics above exist.
    # A 70-seed sweep leaves 280 of them at ~27 MB and filled this machine's
    # system drive twice; results.json is what the sweep is for.
    discard_checkpoints(run_dir, keep=getattr(args, "keep_checkpoints", False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--out", default="runs/paired")
    ap.add_argument("--budgets", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument(
        "--kinds", nargs="*",
        default=["perframe", "modular"],
        choices=[
            "monolithic",
            "perframe",
            "perframe_rate",
            "modular",
            "modular_consist",
            "modular_rate",
            "field",
            "mann",
        ],
    )
    ap.add_argument("--held-compositions", type=int, default=8)
    ap.add_argument(
        "--min-chains",
        type=int,
        default=4,
        help="a held-out transition must appear in at least this many distinct "
             "fine groups, or it cannot be split between the target and informed "
             "sets at all. On OakInk2 only 60 of 391 transitions reach 6.",
    )
    ap.add_argument(
        "--min-per-composition",
        type=int,
        default=4,
        help="target examples per held composition in the informed set. Verify "
             "with scripts/check_informed_coverage.py before trusting a null result.",
    )
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-primitives", type=int, default=12)
    ap.add_argument(
        "--granularity",
        choices=("fine", "category", "shape", "grab_shape_intentclass",
                 "grab_object_intentclass", "oakink_category", "oakink_class",
                 "oakink_attr", "oakink2_scene_primitive", "oakink2_subject_primitive",
                 "oakink2_scene_verb", "oakink2_subject_verb", "left", "right"),
        default="fine",
        help="composition cell size. 'fine' uses the raw label; 'category' "
             "coarsens the left side, which OakInk needs because its fine cells "
             "hold only 2.2 trajectories each.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="repeat every configuration under these seeds and report mean +/- sd. "
             "Single-seed runs of this experiment produce effect sizes the same "
             "size as run-to-run variance, so any claim needs several.",
    )
    ap.add_argument("--fresh", action="store_true",
                    help="retrain everything: ignore checkpoints and rows in results.json")
    ap.add_argument(
        "--keep-checkpoints", action="store_true",
        help="keep each run's weights after scoring. Off by default: a 70-seed "
             "sweep leaves 280 models at ~27 MB each, which filled this "
             "machine's system drive twice. results.json holds every number "
             "the sweep produces.")
    ap.add_argument(
        "--ignore-aux", action="store_true",
        help="drop the bundle's auxiliary channels. The contact experiment runs "
             "both arms off the same file so the split is provably identical; "
             "reading a separate 27-channel bundle instead would leave the two "
             "arms one build step apart.")
    args = ap.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = TrajectoryBundle.load(args.bundle)
    if args.ignore_aux and bundle.aux is not None:
        bundle = TrajectoryBundle(
            trajectories=bundle.trajectories, fps=bundle.fps,
            labels=bundle.labels, meta=bundle.meta,
        )
        print("[aux] dropped auxiliary channels (--ignore-aux)")
    # Kept so the split can keep fine compositions disjoint between the target
    # and informed sets even when the held-out cells are coarse.
    fine_labels = list(bundle.labels)
    if args.granularity != "fine":
        bundle.labels = coarsen_labels(bundle.labels, args.granularity)
        cells = len(set(bundle.labels))
        print(
            f"[split] granularity={args.granularity}: {cells} composition cells, "
            f"{len(bundle.labels) / max(cells, 1):.1f} trajectories per cell"
        )

    print(f"bundle : {args.bundle}  ({len(bundle.trajectories)} trajectories)")
    print(f"seeds  : {seeds}")
    print(f"device : {device}\n")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # A sweep resumes at the row level, not only inside one model. Scored
    # models discard their checkpoints, so a restart that began at seed 0 with
    # an empty list retrained every finished seed from scratch and overwrote
    # results.json with the partial rerun; two restarts of one 40-seed sweep
    # lost seeds 0-4 twice over. Rows already in results.json are kept and
    # their (seed, budget, kind) skipped unless --fresh.
    results = []
    done: set[tuple[int, int, str]] = set()
    prior = out_root / "results.json"
    if prior.exists() and not args.fresh:
        results = json.loads(prior.read_text(encoding="utf-8")).get("results", [])
        done = {(r["seed"], r["budget"], r["kind"]) for r in results}
        if done:
            print(f"[resume] {len(done)} scored rows in {prior}, "
                  f"seeds {sorted({k[0] for k in done})} kept; their models are not retrained")

    for seed in seeds:
        # The split is reseeded too, so the reported spread covers both which
        # compositions get held out and how training happens to go. Holding the
        # split fixed would understate the uncertainty.
        split = build_paired_split(
            bundle, args.held_compositions, seed,
            fine_labels=fine_labels, min_chains=args.min_chains,
        )
        target_b = subset(bundle, split["target"])
        target_ds = WindowedTrajectoryDataset(
            target_b.trajectories, args.window, args.window // 2, target_b.labels,
            aux=target_b.aux,
        )
        target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=False)
        print(
            f"[seed {seed}] held={len(split['held_compositions'])} "
            f"target={len(split['target'])} ({len(target_ds)} windows) "
            f"naive_pool={len(split['naive_pool'])} informed_extra={len(split['informed_extra'])}"
        )

        run_args = argparse.Namespace(**{**vars(args), "seed": seed})
        for budget in args.budgets:
            if budget > len(split["naive_pool"]):
                print(f"skip budget {budget}: naive pool has only {len(split['naive_pool'])}")
                continue
            naive_idx, informed_idx = sample_pools(
                split, budget, seed, bundle.labels, args.min_per_composition
            )
            sound = assert_split_sound(
                split, naive_idx, informed_idx, fine_labels, bundle.labels
            )
            if sound["coverage_thin"]:
                print(f"  thin coverage: informed saw {sound['covered']}/"
                      f"{sound['held']} held compositions -- penalty biased low")
            for kind in args.kinds:
                if (seed, budget, kind) in done:
                    print(f"[resume] seed {seed} {kind} budget={budget}: already scored, skipped")
                    continue
                row = {"kind": kind, "budget": budget, "seed": seed,
                       "soundness": sound}
                for cond, idx in (("naive", naive_idx), ("informed", informed_idx)):
                    print(f"\n{'=' * 70}\nseed {seed}  {kind}  budget={budget}  {cond}\n{'=' * 70}")
                    row[cond] = train_and_score(
                        kind, idx, bundle, target_loader, run_args, device,
                        out_root / f"s{seed}_{kind}_b{budget}_{cond}",
                    )
                row["penalty"] = row["naive"]["mse_target"] - row["informed"]["mse_target"]
                if "mse_target_contact" in row["naive"]:
                    row["penalty_contact"] = (row["naive"]["mse_target_contact"]
                                              - row["informed"]["mse_target_contact"])
                    row["penalty_all"] = (row["naive"]["mse_target_all"]
                                          - row["informed"]["mse_target_all"])
                results.append(row)
                results.sort(key=lambda r: (r["seed"], r["budget"], args.kinds.index(r["kind"])))
                (out_root / "results.json").write_text(
                    json.dumps({"args": vars(args), "seeds": seeds, "results": results}, indent=2),
                    encoding="utf-8",
                )

    _summarise(results, args.budgets, args.kinds, len(seeds))
    print(f"\nfull results -> {out_root / 'results.json'}")
    return 0


def _summarise(results: list[dict], budgets: list[int], kinds: list[str], n_seeds: int) -> None:
    print("\n\n" + "=" * 86)
    print("PAIRED RESULT — penalty = MSE(never saw composition) - MSE(saw composition)")
    print("        identical target trajectories, matched training size, per-seed splits")
    print("=" * 86)
    print(
        f"{'budget':>7} {'model':>10} {'naive':>16} {'informed':>16} {'penalty':>18}  verdict"
    )
    print("-" * 86)

    for budget in budgets:
        for kind in kinds:
            rows = [r for r in results if r["budget"] == budget and r["kind"] == kind]
            if not rows:
                continue
            naive = np.array([r["naive"]["mse_target"] for r in rows])
            informed = np.array([r["informed"]["mse_target"] for r in rows])
            pen = np.array([r["penalty"] for r in rows])
            sd = pen.std(ddof=1) if len(pen) > 1 else float("nan")
            # A mean smaller than its own spread is indistinguishable from zero
            # at this sample size; say so rather than reporting a direction.
            verdict = (
                "n/a (1 seed)" if len(pen) < 2
                else ("INDISTINGUISHABLE FROM 0" if abs(pen.mean()) < sd else f"{'positive' if pen.mean() > 0 else 'negative'}")
            )
            print(
                f"{budget:>7} {kind:>10} "
                f"{naive.mean():>8.5f}+-{naive.std(ddof=1) if len(naive) > 1 else 0:<6.5f} "
                f"{informed.mean():>8.5f}+-{informed.std(ddof=1) if len(informed) > 1 else 0:<6.5f} "
                f"{pen.mean():>9.5f}+-{sd:<7.5f} {verdict}"
            )
        print()

    print(f"n = {n_seeds} seed(s) per configuration. Spreads are sample standard deviations.")
    print("A POSITIVE penalty means seeing the composition helped, i.e. the held-out")
    print("compositions are genuinely hard. Penalties indistinguishable from zero for")
    print("every architecture mean the dataset has no compositional difficulty to")
    print("generalise over, and cannot test the hypothesis whatever the architecture.")


if __name__ == "__main__":
    raise SystemExit(main())
