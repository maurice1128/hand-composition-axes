"""Damage OakInk-Image one property at a time, to name what makes it hard.

Compositional difficulty shows up on OakInk-Image (four axes, +12 to +18% of
naive error) and on nothing else (five axes on GRAB and OakInk2, all at the
zero-truth control's ~2.6%). Instrument insensitivity, window stride,
near-duplicate retrieval and grasp-only dilution on GRAB are already ruled out,
so the open question is *which property of OakInk-Image* produces it.

This builds three bundles, each breaking one candidate property and nothing
else on purpose. Re-running the confirmatory axis (``--granularity
oakink_category``, +15.2%) on each says which damage removes the difficulty.

``oakink_long``      label/window alignment. OakInk2's labels cover 613-frame
                     recordings in which most windows show something other than
                     the label; OakInk-Image's cover 72 frames. Clips of one
                     category but mixed intents are concatenated and given the
                     first clip's label, so the label is right for the first
                     eighth of the windows and wrong for the rest.
``oakink_dofdamage`` retargeting quality. OakInk-Image loses 1 of 27 DOF,
                     GRAB loses 6. GRAB's pinning pattern is reproduced on
                     OakInk-Image poses, leaving labels, lengths and counts
                     untouched.
``oakink_sparse``    sampling density. GRAB's shape x fine-intent grid is 27%
                     occupied, OakInk-Image's category x intent grid is 76%.
                     Whole cells are dropped at random until occupancy is as
                     low as a 256-trajectory budget still allows.

    python scripts/build_oakink_ablations.py build
    python scripts/build_oakink_ablations.py screen

Nothing here trains anything. The bundles are inputs to a later sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import DOF_INDEX, LIMITS_HI, LIMITS_LO  # noqa: E402

from experiment_paired_composition import build_paired_split, coarsen_labels  # noqa: E402

SRC = ROOT / "data" / "bundles" / "oakink.npz"
OUT_DIR = ROOT / "data" / "bundles"

#: Axis the confirmatory sweep uses, and the settings it ran at
#: (``runs/oakink_official_category/results.json`` -> args).
GRANULARITY = "oakink_category"
HELD = 5
MIN_CHAINS = 4
BUDGET = 256

#: The screen reading stored in ``runs/axis_screen.json`` for this axis, and the
#: config it was produced with. Reproducing it is the control: a builder that
#: changes the reference bundle's reading has changed something it should not.
SCREEN_CFG = dict(window=32, stride=16, max_per_traj=6, n_perm=20, seed=0)
REFERENCE_EXCESS = 0.023794310930612808
REFERENCE_Z = 8.32239280894553

#: GRAB's pinned fractions, from ``runs/rerun_dof_health.txt``, as
#: (dof, fraction at the lower limit, fraction at the upper limit).
GRAB_PINNING = [
    ("thumb_cmc_flex", 0.63, 0.00),
    ("pinky_mcp_abd", 0.03, 0.44),
    ("thumb_mcp_abd", 0.27, 0.13),
    ("thumb_cmc_abd", 0.35, 0.00),
    ("pinky_dip_flex", 0.28, 0.00),
]
#: GRAB's wrist_tz is dead (sd 0.0041 normalised). Reproduced as an exact constant.
GRAB_DEAD = "wrist_tz"

CROSSFADE = 4  # frames blended at each join; see build_long


# --------------------------------------------------------------------------
# helpers shared by the three builders
# --------------------------------------------------------------------------

def load_source() -> tuple[TrajectoryBundle, list[str]]:
    b = TrajectoryBundle.load(SRC)
    coarse = coarsen_labels(list(b.labels), GRANULARITY)
    return b, coarse


def naive_pool_sizes(bundle: TrajectoryBundle, seeds: range) -> tuple[int, str]:
    """Smallest naive pool over ``seeds``, or the first failure's message.

    The split is built the way the sweep builds it -- coarse labels for the
    cells, fine labels for the leak-free group assignment -- so this measures
    the pool the sweep would actually draw from.

    The budget is spent out of the naive pool, so a bundle whose pool drops
    below 256 cannot be swept at the confirmatory settings at all -- which is a
    property of the damage, not of the sweep, and has to be checked here rather
    than discovered after the GPU is committed.
    """
    from types import SimpleNamespace

    fine = list(bundle.labels)
    view = SimpleNamespace(labels=coarsen_labels(fine, GRANULARITY))
    worst = 10**9
    for s in seeds:
        try:
            split = build_paired_split(view, HELD, s, fine_labels=fine,
                                       min_chains=MIN_CHAINS)
        except ValueError as exc:
            return -1, f"seed {s}: {exc}"
        worst = min(worst, len(split["naive_pool"]))
    return worst, ""


def pinned_fractions(trajs: list[np.ndarray]) -> dict[str, tuple[float, float, float]]:
    """(at_lo, at_hi, sd_normalised) per DOF, measured as check_dof_health does."""
    q = np.concatenate(trajs)
    span = np.maximum(LIMITS_HI - LIMITS_LO, 1e-8)
    x = 2.0 * (q - LIMITS_LO) / span - 1.0
    out = {}
    for name, i in DOF_INDEX.items():
        out[name] = (
            float(np.mean(q[:, i] <= LIMITS_LO[i] + 1e-4)),
            float(np.mean(q[:, i] >= LIMITS_HI[i] - 1e-4)),
            float(x[:, i].std()),
        )
    return out


def occupancy(coarse: list[str]) -> dict:
    cells = set(coarse)
    left = {c.partition("->")[0] for c in coarse}
    right = {c.partition("->")[2] for c in coarse}
    grid = len(left) * len(right)
    return {"cells": len(cells), "left": len(left), "right": len(right),
            "grid": grid, "occupancy": len(cells) / grid if grid else 0.0}


def report(bundle: TrajectoryBundle, name: str) -> dict:
    lengths = np.array([len(t) for t in bundle.trajectories])
    coarse = coarsen_labels(list(bundle.labels), GRANULARITY)
    occ = occupancy(coarse)
    worst, err = naive_pool_sizes(bundle, range(40))
    info = {
        "bundle": name,
        "n_traj": len(bundle.trajectories),
        "n_frames": int(lengths.sum()),
        "median_len": float(np.median(lengths)),
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
        "fine_cells": len(set(bundle.labels)),
        **occ,
        "min_naive_pool_seeds_0_39": worst,
        "split_error": err,
    }
    print(f"  {name}: {info['n_traj']} traj, median {info['median_len']:.0f} frames, "
          f"{info['cells']} cells ({info['occupancy']:.0%} of {info['grid']}), "
          f"min naive pool {worst}{' ' + err if err else ''}")
    return info


# --------------------------------------------------------------------------
# 1. label/window alignment
# --------------------------------------------------------------------------

def build_long(b: TrajectoryBundle, coarse: list[str], n_long: int, factor: int,
               seed: int = 0) -> TrajectoryBundle:
    """Concatenate ``factor`` clips of one category, mixing intents, one label.

    Each long trajectory draws its clips from a single category and cycles
    through that category's intents, so consecutive clips differ in intent and
    the first clip's intent labels at most ``1/factor`` of the windows. This is
    OakInk2's situation: one label over a long recording whose windows mostly
    show something else.

    770 clips cannot be partitioned into 385 eight-clip trajectories, so clips
    are drawn from per-(category, intent) queues that refill when exhausted and
    **each clip is reused in about ``n_long * factor / 770`` trajectories**.
    That is the cost of holding the trajectory count high enough for a
    256-trajectory budget while reaching OakInk2's median length; it is a real
    confound and is reported rather than hidden.
    """
    rng = np.random.default_rng(seed)
    by_cat_intent: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(coarse):
        cat, _, intent = c.partition("->")
        by_cat_intent[cat][intent].append(i)

    # Categories that offer at least two intents; single-intent categories
    # cannot mix and would defeat the point of the ablation.
    cats = [c for c, d in by_cat_intent.items() if len(d) >= 2]
    weights = np.array([sum(len(v) for v in by_cat_intent[c].values()) for c in cats],
                       dtype=np.float64)
    weights /= weights.sum()

    queues: dict[tuple[str, str], list[int]] = {}

    def draw(cat: str, intent: str) -> int:
        key = (cat, intent)
        q = queues.get(key)
        if not q:
            q = list(rng.permutation(by_cat_intent[cat][intent]))
            queues[key] = q
        return int(q.pop())

    trajs, labels, n_clips_used = [], [], Counter()
    for _ in range(n_long):
        cat = cats[int(rng.choice(len(cats), p=weights))]
        intents = sorted(by_cat_intent[cat])
        start = int(rng.integers(len(intents)))
        picks = [draw(cat, intents[(start + j) % len(intents)]) for j in range(factor)]
        for p in picks:
            n_clips_used[p] += 1

        parts = [np.asarray(b.trajectories[picks[0]], dtype=np.float32)]
        for p in picks[1:]:
            nxt = np.asarray(b.trajectories[p], dtype=np.float32)
            k = min(CROSSFADE, len(parts[-1]), len(nxt) - 1)
            if k > 0:
                # Linear cross-fade over k frames, consuming the incoming clip's
                # first k frames, so a join does not create a pose discontinuity
                # larger than anything inside a clip. Nothing else is blended.
                w = np.linspace(0.0, 1.0, k + 2, dtype=np.float32)[1:-1][:, None]
                tail = parts[-1][-k:] * (1.0 - w) + nxt[:k] * w
                parts[-1] = np.concatenate([parts[-1][:-k], tail])
                nxt = nxt[k:]
            parts.append(nxt)
        trajs.append(np.concatenate(parts).astype(np.float32))
        labels.append(b.labels[picks[0]])

    meta = dict(b.meta)
    meta.update({
        "source": "oakink_long",
        "ablation": "label/window alignment",
        "derived_from": "oakink.npz",
        "concat_factor": factor,
        "n_long": n_long,
        "crossfade_frames": CROSSFADE,
        "clip_reuse_mean": float(np.mean(list(n_clips_used.values()))),
        "clip_reuse_max": int(max(n_clips_used.values())),
        "clips_never_used": int(len(b.trajectories) - len(n_clips_used)),
        "label_rule": "first clip's object->intent; the remaining clips share the "
                      "category but carry other intents",
    })
    meta.pop("sequence_ids", None)
    return TrajectoryBundle(trajectories=trajs, fps=b.fps, labels=labels, meta=meta)


# --------------------------------------------------------------------------
# 1b. the aligned control for 1
# --------------------------------------------------------------------------

def build_longaligned(b: TrajectoryBundle, coarse: list[str], n_long: int,
                      factor: int, seed: int = 0,
                      min_cell: int = 2) -> tuple[TrajectoryBundle, dict]:
    """``build_long``'s concatenation, but joining only same-composition clips.

    ``oakink_long`` confounds two things. It lengthens trajectories *and*
    destroys the intent's own marginal signal, because the clips inside one
    trajectory carry different intents than the label: ``right_only`` fell from
    0.045 to 0.003. A null sweep result on it therefore cannot separate "the
    label no longer matches the window" from "the axis itself was degraded".

    This is the control that separates them. Every clip in a trajectory shares
    both the category and the intent of the trajectory's label, so length grows
    exactly as in ``oakink_long`` while label-to-window alignment is preserved.

    Everything else is held at ``build_long``'s settings on purpose: the same
    ``factor``, the same ``CROSSFADE`` blend consuming the incoming clip's first
    frames, cells drawn with probability proportional to their clip count (as
    categories are there), the same refilling per-cell queues, and the same
    label rule -- the first clip's ``object->intent``. The one difference is the
    pool each draw comes from: one (category, intent) cell here, one category
    cycling through its intents there.

    The cost is reuse. A category x intent cell holds a median of 2 clips
    against a category's 7, so a cell smaller than ``factor`` cannot fill a
    trajectory with distinct clips and its queue refills mid-trajectory. The
    queue already maximises distinctness -- every clip of the cell is used once
    before any is used twice -- but the residual repetition is larger than
    ``oakink_long``'s and is measured into the metadata rather than left to be
    discovered later. ``min_cell`` excludes cells too small to concatenate at
    all; at 2 it mirrors ``build_long``'s "at least two intents" eligibility.
    """
    rng = np.random.default_rng(seed)
    by_cell: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(coarse):
        by_cell[c].append(i)

    cells = sorted(c for c, v in by_cell.items() if len(v) >= min_cell)
    weights = np.array([len(by_cell[c]) for c in cells], dtype=np.float64)
    weights /= weights.sum()

    queues: dict[str, list[int]] = {}

    def draw(cell: str) -> int:
        q = queues.get(cell)
        if not q:
            q = list(rng.permutation(by_cell[cell]))
            queues[cell] = q
        return int(q.pop())

    trajs, labels, n_clips_used = [], [], Counter()
    repeat_slots = 0
    repeat_trajs = 0
    worst_multiplicity = 0
    for _ in range(n_long):
        cell = cells[int(rng.choice(len(cells), p=weights))]
        picks = [draw(cell) for _ in range(factor)]
        for p in picks:
            n_clips_used[p] += 1
        counts = Counter(picks)
        repeat_slots += factor - len(counts)
        repeat_trajs += int(len(counts) < factor)
        worst_multiplicity = max(worst_multiplicity, max(counts.values()))

        parts = [np.asarray(b.trajectories[picks[0]], dtype=np.float32)]
        for p in picks[1:]:
            nxt = np.asarray(b.trajectories[p], dtype=np.float32)
            k = min(CROSSFADE, len(parts[-1]), len(nxt) - 1)
            if k > 0:
                # Identical to build_long: linear cross-fade over k frames,
                # consuming the incoming clip's first k frames.
                w = np.linspace(0.0, 1.0, k + 2, dtype=np.float32)[1:-1][:, None]
                tail = parts[-1][-k:] * (1.0 - w) + nxt[:k] * w
                parts[-1] = np.concatenate([parts[-1][:-k], tail])
                nxt = nxt[k:]
            parts.append(nxt)
        trajs.append(np.concatenate(parts).astype(np.float32))
        labels.append(b.labels[picks[0]])

    stats = {
        "clip_reuse_mean": float(np.mean(list(n_clips_used.values()))),
        "clip_reuse_max": int(max(n_clips_used.values())),
        "clips_never_used": int(len(b.trajectories) - len(n_clips_used)),
        "eligible_cells": len(cells),
        "min_cell": min_cell,
        "within_traj_repeat_trajectories": repeat_trajs,
        "within_traj_repeat_slot_fraction": repeat_slots / float(n_long * factor),
        "within_traj_worst_multiplicity": worst_multiplicity,
    }

    meta = dict(b.meta)
    meta.update({
        "source": "oakink_longaligned",
        "ablation": "length without misalignment (control for oakink_long)",
        "derived_from": "oakink.npz",
        "concat_factor": factor,
        "n_long": n_long,
        "crossfade_frames": CROSSFADE,
        "label_rule": "first clip's object->intent; every clip in the trajectory "
                      "shares that label's category AND intent",
        **stats,
    })
    meta.pop("sequence_ids", None)
    return TrajectoryBundle(trajectories=trajs, fps=b.fps, labels=labels,
                            meta=meta), stats


def run_build_aligned(n_long: int = 385, factor: int = 8, min_cell: int = 2,
                      save: bool = True) -> dict:
    """Build ``oakink_longaligned.npz`` and report it as the others are reported.

        python -c "import sys; sys.path.insert(0,'scripts'); \
            from build_oakink_ablations import run_build_aligned; run_build_aligned()"

    Kept out of ``main`` so the three original bundles and their reported
    outputs cannot be perturbed by anything done here.
    """
    b, coarse = load_source()
    out, stats = build_longaligned(b, coarse, n_long, factor, min_cell=min_cell)
    if save:
        out.save(OUT_DIR / "oakink_longaligned.npz")
    info = report(out, "oakink_longaligned")
    info.update(stats)
    print(f"    reuse mean {stats['clip_reuse_mean']:.2f}, max "
          f"{stats['clip_reuse_max']}, {stats['clips_never_used']} clips unused; "
          f"{stats['within_traj_repeat_trajectories']}/{n_long} trajectories repeat "
          f"a clip internally ({stats['within_traj_repeat_slot_fraction']:.1%} of "
          f"slots, worst x{stats['within_traj_worst_multiplicity']})")
    return info


# --------------------------------------------------------------------------
# 2. retargeting quality
# --------------------------------------------------------------------------

def build_dofdamage(b: TrajectoryBundle) -> tuple[TrajectoryBundle, dict]:
    """Reproduce GRAB's pinned DOF on OakInk-Image poses.

    For a target lower-limit fraction ``p``, the ``p``-quantile of the DOF is
    clamped to the limit, which puts exactly ``p`` of frames at it. Where
    OakInk-Image is *already* pinned harder than GRAB, clamping cannot undo it
    and the DOF is left alone -- that is reported, not silently skipped.

    Labels, lengths and trajectory count are untouched. Pose statistics are
    not: clamping a quantile collapses that tail onto the limit, so the mean
    and sd of a damaged DOF move as well.
    """
    q = np.concatenate([np.asarray(t, dtype=np.float32) for t in b.trajectories]).copy()
    before = pinned_fractions(b.trajectories)
    notes = {}

    for name, p_lo, p_hi in GRAB_PINNING:
        i = DOF_INDEX[name]
        cur_lo, cur_hi, _ = before[name]
        if cur_lo + cur_hi >= p_lo + p_hi:
            notes[name] = (f"left alone: OakInk already pinned {cur_lo + cur_hi:.0%} "
                           f"(GRAB {p_lo + p_hi:.0%}); clamping cannot reduce pinning")
            continue
        col = q[:, i]
        if p_lo > cur_lo:
            thr = float(np.quantile(col, p_lo))
            col[col <= thr] = LIMITS_LO[i]
        if p_hi > cur_hi:
            thr = float(np.quantile(col, 1.0 - p_hi))
            col[col >= thr] = LIMITS_HI[i]
        q[:, i] = col
        notes[name] = f"clamped to lo {p_lo:.0%} / hi {p_hi:.0%}"

    j = DOF_INDEX[GRAB_DEAD]
    q[:, j] = float(np.median(q[:, j]))
    notes[GRAB_DEAD] = "set to its median; sd exactly 0"

    bounds = np.concatenate([[0], np.cumsum([len(t) for t in b.trajectories])])
    trajs = [q[bounds[i]:bounds[i + 1]] for i in range(len(b.trajectories))]

    meta = dict(b.meta)
    meta.update({
        "source": "oakink_dofdamage",
        "ablation": "retargeting quality",
        "derived_from": "oakink.npz",
        "target_pinning": {n: [lo, hi] for n, lo, hi in GRAB_PINNING},
        "dead_dof": GRAB_DEAD,
        "notes": notes,
    })
    out = TrajectoryBundle(trajectories=trajs, fps=b.fps, labels=list(b.labels), meta=meta)
    after = pinned_fractions(out.trajectories)
    achieved = {n: {"before": [round(before[n][0], 3), round(before[n][1], 3)],
                    "after": [round(after[n][0], 3), round(after[n][1], 3)],
                    "sd_after": round(after[n][2], 4), "note": notes[n]}
                for n in notes}
    return out, achieved


# --------------------------------------------------------------------------
# 3. sampling density
# --------------------------------------------------------------------------

def build_sparse(b: TrajectoryBundle, coarse: list[str], seed: int = 0,
                 target: float = 0.27) -> tuple[TrajectoryBundle, dict]:
    """Drop whole category x intent cells at random towards GRAB-like occupancy.

    Cells are dropped, not thinned: every trajectory of a surviving cell is
    kept, so within-cell density is unchanged and only the grid's occupancy
    moves. How far it can move is bounded by the budget -- the naive pool must
    still hold 256 trajectories at every seed -- so the number of kept cells is
    the smallest that clears that bar, found by search, with the *identity* of
    the kept cells drawn at random at each size.
    """
    rng = np.random.default_rng(seed)
    cells = sorted(set(coarse))
    base = occupancy(coarse)
    n_target = max(1, int(round(target * base["grid"])))

    trace = []
    for n_keep in range(n_target, len(cells) + 1):
        keep = set(rng_choice(rng, cells, n_keep, seed + n_keep))
        idx = [i for i, c in enumerate(coarse) if c in keep]
        if len(idx) < BUDGET + 8:
            trace.append({"n_keep": n_keep, "n_traj": len(idx), "pool": None,
                          "why": "too few trajectories"})
            continue
        sub = TrajectoryBundle(
            trajectories=[b.trajectories[i] for i in idx], fps=b.fps,
            labels=[b.labels[i] for i in idx], meta=dict(b.meta),
        )
        worst, err = naive_pool_sizes(sub, range(40))
        trace.append({"n_keep": n_keep, "n_traj": len(idx), "pool": worst,
                      "why": err or "ok"})
        if worst >= BUDGET:
            occ = occupancy(coarsen_labels(list(sub.labels), GRANULARITY))
            meta = dict(b.meta)
            meta.update({
                "source": "oakink_sparse",
                "ablation": "sampling density",
                "derived_from": "oakink.npz",
                "cells_kept": n_keep,
                "cells_before": base["cells"],
                "occupancy_before": base["occupancy"],
                "occupancy_after": occ["occupancy"],
                "target_occupancy": target,
                "selection": "whole cells dropped uniformly at random, seed 0",
            })
            meta.pop("sequence_ids", None)
            sub.meta = meta
            return sub, {"trace": trace, "occupancy_after": occ["occupancy"],
                         "cells_kept": n_keep, "cells_before": base["cells"],
                         "occupancy_before": base["occupancy"]}
    raise RuntimeError("no cell subset leaves a workable pool")


def rng_choice(rng: np.random.Generator, items: list[str], k: int, seed: int) -> list[str]:
    """A fresh draw per size, so the search is not a nested-subset ladder."""
    local = np.random.default_rng(seed)
    return [items[i] for i in local.choice(len(items), k, replace=False)]


# --------------------------------------------------------------------------
# screen
# --------------------------------------------------------------------------

def run_screen(out: Path) -> int:
    """Excess interaction and z for the confirmatory axis on all four bundles."""
    from screen_axes import screen

    rows = []
    for name in ("oakink", "oakink_long", "oakink_dofdamage", "oakink_sparse"):
        path = OUT_DIR / f"{name}.npz"
        r = screen(path, GRANULARITY, SCREEN_CFG["window"], SCREEN_CFG["stride"],
                   SCREEN_CFG["max_per_traj"], SCREEN_CFG["n_perm"], SCREEN_CFG["seed"])
        if r is None or "error" in r:
            print(f"{name}: {r}")
            return 1
        r["bundle"] = name
        rows.append(r)
        print(f"{name:<18} cells {r['n_cells']:>4}  left {r['left_only']:.3f}  "
              f"right {r['right_only']:.3f}  inter {r['interaction_r2']:.4f}  "
              f"excess {r['excess']:+.4f}  z {r['z']:.1f}")

    ref = rows[0]
    ok = (abs(ref["excess"] - REFERENCE_EXCESS) < 1e-6
          and abs(ref["z"] - REFERENCE_Z) < 1e-3)
    print(f"\nreference reproduction: excess {ref['excess']:.6f} "
          f"(stored {REFERENCE_EXCESS:.6f}), z {ref['z']:.3f} "
          f"(stored {REFERENCE_Z:.3f}) -> {'MATCH' if ok else 'MISMATCH'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {**SCREEN_CFG, "granularity": GRANULARITY},
        "reference": {"excess": REFERENCE_EXCESS, "z": REFERENCE_Z,
                      "reproduced": bool(ok)},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    assert ok, "unmodified oakink.npz no longer reproduces the stored screen reading"
    return 0


def screen_aligned(out: Path | None = None) -> dict:
    """Screen ``oakink_longaligned`` and append it to the ablations' screen file.

    The existing rows are read and written back untouched: this adds a fifth
    entry, it does not re-run the other four. ``run_screen`` deliberately still
    knows nothing about this bundle, so re-running it reproduces exactly the
    file it produced before.

    The reading to compare against is ``right_only``. On the unmodified bundle
    the intent explains 0.045 of pose variance on its own; ``oakink_long`` left
    0.003 of that, which is what made its null uninterpretable.
    """
    from screen_axes import screen

    path = OUT_DIR / "oakink_longaligned.npz"
    r = screen(path, GRANULARITY, SCREEN_CFG["window"], SCREEN_CFG["stride"],
               SCREEN_CFG["max_per_traj"], SCREEN_CFG["n_perm"], SCREEN_CFG["seed"])
    if r is None or "error" in r:
        raise RuntimeError(f"screen failed: {r}")
    r["bundle"] = "oakink_longaligned"
    print(f"{'oakink_longaligned':<18} cells {r['n_cells']:>4}  "
          f"left {r['left_only']:.3f}  right {r['right_only']:.3f}  "
          f"inter {r['interaction_r2']:.4f}  excess {r['excess']:+.4f}  z {r['z']:.1f}")

    out = out or (ROOT / "runs" / "axis_screen_oakink_ablations.json")
    doc = json.loads(out.read_text(encoding="utf-8"))
    before = [row["bundle"] for row in doc["rows"]]
    doc["rows"] = [row for row in doc["rows"] if row["bundle"] != "oakink_longaligned"]
    doc["rows"].append(r)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"appended to {out}; rows now {[x['bundle'] for x in doc['rows']]} "
          f"(was {before})")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build", "screen"))
    ap.add_argument("--n-long", type=int, default=385)
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--out", default="runs/axis_screen_oakink_ablations.json")
    args = ap.parse_args()

    if args.mode == "screen":
        return run_screen(ROOT / args.out)

    b, coarse = load_source()
    print(f"source {SRC.name}: {len(b.trajectories)} trajectories, "
          f"median {np.median([len(t) for t in b.trajectories]):.0f} frames")
    summary = [report(b, "oakink (reference)")]

    long_b = build_long(b, coarse, args.n_long, args.factor)
    long_b.save(OUT_DIR / "oakink_long.npz")
    info = report(long_b, "oakink_long")
    info["meta"] = {k: long_b.meta[k] for k in
                    ("concat_factor", "n_long", "crossfade_frames", "clip_reuse_mean",
                     "clip_reuse_max", "clips_never_used")}
    summary.append(info)

    dmg_b, achieved = build_dofdamage(b)
    dmg_b.save(OUT_DIR / "oakink_dofdamage.npz")
    info = report(dmg_b, "oakink_dofdamage")
    info["pinning"] = achieved
    summary.append(info)
    for name, row in achieved.items():
        print(f"    {name:<16} lo/hi {row['before'][0]:.0%}/{row['before'][1]:.0%} "
              f"-> {row['after'][0]:.0%}/{row['after'][1]:.0%}  sd {row['sd_after']:.4f}"
              f"   {row['note']}")

    sparse_b, sinfo = build_sparse(b, coarse)
    sparse_b.save(OUT_DIR / "oakink_sparse.npz")
    info = report(sparse_b, "oakink_sparse")
    info["sparsity"] = sinfo
    summary.append(info)
    print(f"    occupancy {sinfo['occupancy_before']:.0%} -> "
          f"{sinfo['occupancy_after']:.0%} "
          f"({sinfo['cells_before']} -> {sinfo['cells_kept']} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
