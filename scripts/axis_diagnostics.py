"""Referee-requested quantities for the nine Table 1 axes, derived without training.

Why this exists
---------------
A referee asked for four things the manuscript does not report, and all four are
recoverable from ``runs/<sweep>/results.json`` plus a re-derivation of each
seed's split. Nothing here trains, samples or touches a GPU.

1. **The denominator.** The headline statistic is a *ratio*,
   ``penalty / naive mse_target``, and the denominator is never printed. A
   +10% on OakInk-Image and a +10% on OakInk2 are only comparable if their
   naive errors are comparable, which is an empirical question, not an
   assumption. Section A reports both absolute errors.
2. **The split's own parameters** (held compositions, ``min_chains``,
   ``min_per_composition``, window, stride) per axis, read from each run's
   stored ``args`` rather than from the manuscript, because two Table 1 rows
   were once reported at a stride the text did not mention.
3. **The operationalisation of "the label describes what is held out."**
   Section 3.4 of the draft argues that a penalty is measurable only when the
   label describes the scored window. That argument is made with *derived*
   datasets (concatenated clips, primitive segments). The referee is entitled to
   ask what the same diagnostic reads on the nine axes actually in Table 1. Three
   quantities operationalise it: how much of the informed arm's budget really
   comes from held-out cells, how much of the naive arm's does (should be none),
   and how much of the target set really lies in held-out cells (should be all).
4. **A within-axis check**: does a seed that happened to give the informed arm
   more held-cell exposure pay a larger penalty?

The honest expectation for (3), stated before running it: on the Table 1 axes
the split is made *directly on trajectory labels*, so naive contamination is 0
and target purity is 1 **by construction** -- ``build_paired_split`` partitions
by label and cannot do otherwise. If that is what comes out, the diagnostic does
not discriminate among Table 1 axes and saying so is the result. This script
does not go looking for variation that the design forbids.

How the splits are reproduced
-----------------------------
``build_paired_split`` / ``sample_pools`` are seeded by ``np.random.default_rng``
and read nothing but the bundle's labels, so calling them with a run's own
``args`` and each of its seeds reproduces that run's split exactly. The claim is
not taken on trust: every derived split is checked against the ``soundness``
block the sweep itself stored (leak fractions, held count, covered count), and a
mismatch is reported loudly rather than averaged away.

    python scripts/axis_diagnostics.py            # -> runs/axis_diagnostics.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_data_efficiency import transitions_of  # noqa: E402
from experiment_paired_composition import (  # noqa: E402
    assert_split_sound,
    build_paired_split,
    coarsen_labels,
    sample_pools,
)

#: (dataset, axis label, run directories, both_arch) exactly as Table 1 lists them.
#: ``both_arch=True`` marks a sweep that also trained the modular bank; the
#: manuscript's seed rule then keeps only seeds where BOTH kinds finished. The
#: rule is copied from ``scripts/verify_alignment_draft.py::rel`` so the seed
#: sets here are the same ones the reported means were taken over.
AXES: list[tuple[str, str, list[str], bool]] = [
    ("OakInk-Image", "functional class x intent", ["oakink_class_rep"], False),  # 40-seed replication; original oakink_class_v1 (12)
    ("OakInk-Image", "category x intent", ["oakink_official_category", "oakink_category_rest"], True),
    ("OakInk-Image", "affordance x intent", ["oakink_official_attr", "oakink_attr_more"], True),
    ("OakInk-Image", "category x subject", ["oakink_category_subject"], False),
    ("GRAB", "shape x fine intent", ["grab_shape_s4"], False),
    ("GRAB", "shape x intent class", ["grab_shapeclass"], True),
    ("OakInk2", "scene x verb", ["oakink2_scene_verb"], True),
    ("OakInk2", "scene x primitive", ["oakink2_scene_primitive"], False),
    ("OakInk2", "annotated transitions", ["oakink2_transitions_rep"], False),  # 40-seed replication; original _s4 (12)
    ("TACO", "action x tool", ["taco_action_tool"], False),
]

BUDGET = 256
KIND = "perframe"

#: Sweeps that are still running or pre-registered-and-unread. Reading them would
#: be optional stopping; this project already has one such defect on record.
FORBIDDEN = {
    "sham_oakink_category",
    "sham_grab_shape",
}

#: Run directory -> full coverage-gate transcript in ``runs/gates/``.
#:
#: Three of the nine axes' runs predate the ``soundness`` block and stored
#: nothing to check a rebuilt split against, and one of them
#: (``oakink_official_attr`` + ``oakink_attr_more``) is a whole Table 1 row. The
#: coverage gate wrote per-seed transcripts for exactly those runs, and it
#: computes ``covered`` and ``per comp`` from the same two functions, so the
#: transcript is an independent record of the same split. It also prints
#: ``swapped`` = |informed \ naive|, which the stored soundness block does not,
#: so where both sources exist this adds a third checked number per seed.
#:
#: Only transcripts whose header agrees with the run's own args are used; a
#: mismatch is refused rather than compared, because the gate was also run at
#: other held-out counts (the ``_h3`` / ``_h4`` files).
GATE_TRANSCRIPTS = {
    "oakink_official_category": "cover_oakink_category_official_full.txt",
    "oakink_category_rest": "cover_oakink_category_rest_full.txt",
    "oakink_official_attr": "cover_oakink_attr_official_full.txt",
    "oakink_attr_more": "cover_oakink_attr_more_full.txt",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _lineno(rel_path: str, needle: str) -> str:
    """``path:line`` for the first line containing ``needle``.

    Line numbers are looked up rather than hard-coded so the evidence pointers in
    the "notes" section stay correct when the files move around.
    """
    p = ROOT / rel_path
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if needle in line:
            return f"{rel_path}:{i}"
    return f"{rel_path}:NOT_FOUND({needle!r})"


_LABEL_CACHE: dict[str, list[str]] = {}


def load_labels(bundle: str) -> list[str]:
    """The bundle's labels, and only its labels.

    ``TrajectoryBundle.load`` would also decompress every frame, and the OakInk2
    bundles are 47 MB each; other jobs hold this machine's RAM and CPU. The split
    functions read nothing but ``bundle.labels``, so the frames are dead weight.
    The conversion is character-for-character the one ``TrajectoryBundle.load``
    performs (``[str(x) for x in data["labels"].tolist()]``).
    """
    key = bundle.replace("\\", "/")
    if key not in _LABEL_CACHE:
        with np.load(ROOT / key, allow_pickle=True) as z:
            _LABEL_CACHE[key] = [str(x) for x in z["labels"].tolist()]
    return _LABEL_CACHE[key]


def read_run(d: str) -> dict:
    if d in FORBIDDEN:
        raise RuntimeError(f"{d} is running or unread; reading it would be optional stopping")
    return json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))


def seed_rows(dirs: list[str], both_arch: bool) -> list[tuple[str, int, dict]]:
    """(run dir, seed, perframe row) under the manuscript's seed rule.

    Copied from ``verify_alignment_draft.rel``: group a run's rows by seed at the
    reported budget, require the perframe row, and on a both-arch sweep require
    the modular row too. The dirs of a multi-part axis are concatenated, which is
    only sound if their seed ranges are disjoint -- checked by the caller.
    """
    out = []
    for d in dirs:
        by: dict[int, dict[str, dict]] = {}
        for r in read_run(d)["results"]:
            if r["budget"] == BUDGET:
                by.setdefault(r["seed"], {})[r["kind"]] = r
        for seed in sorted(by):
            v = by[seed]
            if KIND not in v or (both_arch and "modular" not in v):
                continue
            out.append((d, seed, v[KIND]))
    return out


def gate_rows(run: str, args: dict) -> dict[int, dict]:
    """Per-seed rows from a coverage-gate transcript, keyed by seed.

    Returns an empty dict when there is no transcript for this run, or when its
    header does not match the run's own args -- comparing against a transcript
    produced at a different held-out count would invent mismatches.
    """
    name = GATE_TRANSCRIPTS.get(run)
    if name is None:
        return {}
    text = (ROOT / "runs" / "gates" / name).read_text(encoding="utf-8")
    head = text.split("\n seed", 1)[0]
    if (f"granularity={args['granularity']}:" not in head
            or f"holding out {args['held_compositions']} compositions" not in head):
        return {}
    out: dict[int, dict] = {}
    for line in text.splitlines():
        # seed budget swapped held covered per_comp status
        f = line.split()
        if len(f) >= 7 and all(t.isdigit() for t in f[:5]):
            seed, budget, swapped, held, covered = (int(t) for t in f[:5])
            if budget == BUDGET:
                out[seed] = {"swapped": swapped, "held": held,
                             "covered": covered, "per_comp": float(f[5])}
    return out


def mean_ci(x: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% CI of the mean, t-based. Not a CI on a single seed."""
    if len(x) < 2:
        return (float("nan"), float("nan"))
    sem = x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.975, len(x) - 1)
    return (float(x.mean() - q * sem), float(x.mean() + q * sem))


# ---------------------------------------------------------------------------
# Section B: re-derive one seed's split
# ---------------------------------------------------------------------------


def derive_seed(args: dict, seed: int) -> dict:
    """Rebuild the split this sweep used at this seed and measure it.

    The call sequence mirrors ``experiment_paired_composition.main`` exactly:
    coarsen the labels, keep the fine labels for group-disjointness, build the
    paired split, then draw the two pools. A stand-in namespace is passed in
    place of the bundle because the split functions read only ``.labels``.
    """
    fine = load_labels(args["bundle"])
    gran = args["granularity"]
    coarse = fine if gran == "fine" else coarsen_labels(fine, gran)
    shim = types.SimpleNamespace(labels=coarse)

    split = build_paired_split(
        shim, args["held_compositions"], seed,
        fine_labels=fine, min_chains=args["min_chains"],
    )
    naive_idx, informed_idx = sample_pools(
        split, BUDGET, seed, coarse, args["min_per_composition"]
    )
    sound = assert_split_sound(split, naive_idx, informed_idx, fine, coarse)

    held = set(split["held_compositions"])

    def held_hits(i: int) -> set[str]:
        return {f"{a}->{b}" for a, b in transitions_of(coarse[i])} & held

    # B5 realised informed exposure, B6 naive contamination, B7 target purity.
    # All three are measured by membership rather than inferred from the sampling
    # code, so a change in that code would show up here instead of being assumed
    # away.
    exposure = sum(1 for i in informed_idx if held_hits(i)) / len(informed_idx)
    contamination = sum(1 for i in naive_idx if held_hits(i)) / len(naive_idx)
    purity = sum(1 for i in split["target"] if held_hits(i)) / len(split["target"])

    per_cell: dict[str, int] = {}
    for i in informed_idx:
        for k in held_hits(i):
            per_cell[k] = per_cell.get(k, 0) + 1

    # How much the two arms actually share. The design is described as "identical
    # training size, differing only in composition coverage", and the size part is
    # exact -- but sample_pools draws the informed arm's non-held remainder
    # (``keep``) as a FRESH sample from the naive pool rather than reusing the
    # naive arm's own trajectories, so the two arms also differ in most of their
    # non-held data. The referee should have the number.
    shared = len(set(naive_idx) & set(informed_idx))

    return {
        "seed": seed,
        "held_compositions": split["held_compositions"],
        "exposure": exposure,
        "shared_training_fraction": shared / len(naive_idx),
        "swapped": len(set(informed_idx) - set(naive_idx)),
        "contamination": contamination,
        "purity": purity,
        "covered": len(per_cell),
        "n_held": len(held),
        "examples_per_covered_cell": (
            float(np.mean(list(per_cell.values()))) if per_cell else 0.0
        ),
        "n_target": len(split["target"]),
        "n_naive_pool": len(split["naive_pool"]),
        "n_informed_extra": len(split["informed_extra"]),
        # How many held-out units were even eligible: a transition has to appear
        # in >= min_chains distinct fine groups to be splittable at all. An axis
        # whose pool is barely larger than held_compositions is drawing from
        # almost the whole eligible set every seed, so its seeds are not
        # independent draws over cells.
        "candidate_pool_size": split["candidate_pool_size"],
        "n_fine_groups_held": split["n_fine_groups_held"],
        "derived_soundness": sound,
    }


# ---------------------------------------------------------------------------
# notes 9-11
# ---------------------------------------------------------------------------


def note_error_quantity() -> dict:
    """What ``mse_target`` actually is. It is not a generative score."""
    return {
        "answer": (
            "Teacher-forced reconstruction, not prior-sampled generation: the scored "
            "windows are fed through the encoder, the latent is taken at its mean "
            "(no sampling at eval), the decoder is additionally conditioned on the "
            "window's true first frame, and the reported number is the mean squared "
            "error between that reconstruction and the same window, over the 27 pose "
            "channels in normalised [-1, 1] units."
        ),
        "chain": [
            _lineno("scripts/experiment_paired_composition.py", 'mse_target": recon_mse(')
            + "  mse_target = recon_mse(model, target_loader, device, channels=slice(0, N_DOF))",
            f"{_lineno('scripts/experiment_data_efficiency.py', 'out = model(batch)')}"
            "  recon_mse calls model(batch) on the target windows and compares out['recon'] to batch itself",
            f"{_lineno('scripts/experiment_data_efficiency.py', 'model.eval()')}"
            "  recon_mse puts the model in eval mode first",
            f"{_lineno('src/caredex/models/latent_prior.py', 'if not self.training:')}"
            "  reparameterize returns mu unchanged outside training, so the latent is deterministic",
            f"{_lineno('src/caredex/models/latent_prior.py', 'recon = self.decode(z, window[:, 0], window.shape[1])')}"
            "  forward decodes from the encoded window AND its true first frame",
            f"{_lineno('src/caredex/models/latent_prior.py', 'def sample(')}"
            "  the model does have a prior-sampling path (sample), and no sweep metric uses it",
        ],
        "consequence": (
            "The 'perframe' prior used for every Table 1 row emits one latent per "
            "frame, so the decoder receives 12 numbers per frame plus the true "
            "initial pose. The penalty is therefore the cost of an unseen "
            "composition to an autoencoder's bottleneck, not to a generator asked "
            "to produce the composition unprompted. Any claim phrased as "
            "'the model cannot generate the unseen composition' is unsupported by "
            "this metric."
        ),
    }


def note_synthetic_controls() -> dict:
    """The zero-truth and planted controls, in enough detail to reimplement."""
    out: dict = {
        "generator": "src/caredex/data/synthetic.py, class SyntheticSource (registered as 'synthetic')",
        "built_by": (
            "scripts/generate_synthetic.py (--set data.synthetic.*); defaults in "
            "configs/default.yaml. The generation parameters are stored in each "
            "bundle's meta and are reported below from the bundle itself, not from "
            "the config."
        ),
        "swept_by": _lineno("scripts/queue_confound.cmd", "pc_easy_rerun"),
        "code_evidence": {
            "interpolation": f"{_lineno('src/caredex/data/synthetic.py', 'seg = waypoints[k] * (1 - w) + waypoints[k + 1] * w')}",
            "via_drawn_once": f"{_lineno('src/caredex/data/synthetic.py', 'coeff = rng.standard_normal((n_prims, n_prims, self.noise_rank))')}",
            "via_applied": f"{_lineno('src/caredex/data/synthetic.py', 'bump = np.sin(np.pi * w)')}",
            "minimum_jerk": f"{_lineno('src/caredex/data/synthetic.py', 'return 10 * t**3 - 15 * t**4 + 6 * t**5')}",
            "low_rank_basis": f"{_lineno('src/caredex/data/synthetic.py', 'basis = rng.standard_normal((self.noise_rank, N_ARTICULATED))')}",
            "waypoints": f"{_lineno('src/caredex/data/synthetic.py', 'waypoints = prims[order] + self.noise_deg * (coeffs @ basis)')}",
            "dip_pip_and_clamp": f"{_lineno('src/caredex/data/synthetic.py', 'traj = apply_dip_pip_coupling(traj)')}",
        },
        "process": (
            "Per trajectory: draw n_seg segments; draw n_seg+1 primitive indices "
            "uniformly with replacement from the 9 fixed grasp primitives (21 "
            "articulated DOF each, in degrees); each waypoint = primitive pose + "
            "noise_deg * (N(0,1)^rank @ basis), where basis is ONE (rank, 21) "
            "unit-row matrix fixed for the whole dataset; each segment interpolates "
            "consecutive waypoints with minimum-jerk weights "
            "w = 10t^3 - 15t^4 + 6t^5; the 6 global wrist DOF follow an independent "
            "thrice-moving-averaged random walk at 0.6 of the limit half-range; add "
            "sensor_noise_deg i.i.d. per frame per DOF; overwrite DIP flexion with "
            "the 2/3 x PIP coupling; clamp into the joint-limit box. The label is "
            "the primitive chain joined by '->'."
        ),
        "factor_structure": (
            "NOT a two-factor grid, unlike all nine Table 1 axes. The held-out unit "
            "is an ordered pair of consecutive primitives drawn from a single "
            "9-symbol alphabet, so the cell space is the 9x9 = 81 ordered pairs "
            "(self-pairs included, since indices are drawn independently). A "
            "trajectory carries several cells at once. This is a structural "
            "difference between the controls and the axes they calibrate and should "
            "be stated where the controls are used."
        ),
        "zero_truth_mechanism": (
            "synthetic_big.npz has transition_via_deg = 0 (absent from its meta), so "
            "a segment is exactly a convex combination of its two endpoint "
            "waypoints: seg = w_k (1-w) + w_{k+1} w. Any path between two endpoints "
            "a model has seen separately is therefore reproducible without ever "
            "having seen that ordered pair, and the compositional penalty is zero "
            "BY CONSTRUCTION. Whatever this sweep reads is the instrument's own "
            "offset, which is why it is the zero-truth control rather than a null "
            "result."
        ),
        "planted_mechanism": (
            "synth_hard.npz sets transition_via_deg = 18.0 degrees. One via offset "
            "per ORDERED pair is drawn once for the whole dataset, "
            "via[a,b] = 18.0 * (coeff[a,b] @ basis) with coeff of shape "
            "(9, 9, rank) ~ N(0,1) and basis the same fixed low-rank matrix, and is "
            "added to the segment as bump * via[a,b] with bump = sin(pi * w). The "
            "bump vanishes at w = 0 and w = 1 and peaks at the midpoint, so the "
            "endpoints stay exactly the endpoint waypoints and only the path between "
            "them carries the pair-specific signature. It is fixed per pair (so it "
            "is learnable from examples of that pair) and independent of the "
            "endpoints (so it is not learnable from them), which is what a positive "
            "control has to be."
        ),
        "bundles": {},
    }
    for name, run in (("synthetic_big.npz", "pc_easy_rerun"), ("synth_hard.npz", "pc_hard_rerun")):
        with np.load(ROOT / "data" / "bundles" / name, allow_pickle=True) as z:
            lengths = np.asarray(z["lengths"])
            labels = [str(x) for x in z["labels"].tolist()]
            # base.py stores meta as repr(dict) and reads it back with
            # ast.literal_eval; doing the same keeps this in step with the loader.
            meta = ast.literal_eval(str(z["meta"]))
            fps = float(z["fps"])
        n_seg = np.array([lab.count("->") for lab in labels])
        trans = {t for lab in labels for t in transitions_of(lab)}
        prims = {p for lab in labels for p in lab.split("->")}
        out["bundles"][name] = {
            "used_by_run": run,
            "n_trajectories": int(len(lengths)),
            "fps": fps,
            "frames_per_trajectory": {
                "min": int(lengths.min()), "median": float(np.median(lengths)),
                "max": int(lengths.max()), "mean": float(lengths.mean()),
            },
            "seconds_per_trajectory_mean": float(lengths.mean() / fps),
            "segments_per_trajectory": {"min": int(n_seg.min()), "max": int(n_seg.max())},
            "frames_per_segment_implied": {
                "mean": float((lengths / np.maximum(n_seg, 1)).mean())
            },
            "n_primitives": len(prims),
            "n_distinct_ordered_pairs_present": len(trans),
            "n_possible_ordered_pairs": len(prims) ** 2,
            "n_distinct_chain_labels": len(set(labels)),
            "example_label": labels[0],
            "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()},
        }
    return out


def note_labels(per_axis: list[dict]) -> dict:
    """What a fine label and a coarse cell literally are, per axis."""
    return {
        "coarsen_labels": _lineno(
            "scripts/experiment_paired_composition.py", "def coarsen_labels("
        ),
        "how_the_two_factors_are_formed": (
            "granularity='fine' keeps the raw label and the held-out unit is an "
            "ordered pair of consecutive primitives inside it. 'shape' and "
            "'grab_*' rewrite the left and/or right side of a 'left->right' label "
            "through a lookup table. 'oakink_*' replaces the object id with its "
            "group from OakInk's own metaV2.zip taxonomy. 'oakink2_<left>_<right>' "
            "does not read the label's chain at all: the bundle label is "
            "'chain@scene@subject@verb' and the cell is built from two of that "
            "suffix's fields, so the fine label (the whole string) and the coarse "
            "cell come from different parts of it."
        ),
        "per_axis": [
            {
                "dataset": a["dataset"],
                "axis": a["axis"],
                "bundle": a["bundle"],
                "granularity": a["granularity"],
                **a["labels"],
            }
            for a in per_axis
        ],
    }


#: What the two sides of a held-out cell mean, keyed by AXIS rather than by
#: granularity: ``oakink_category_subject`` reuses the ``oakink2_scene_verb``
#: parser on an OakInk-Image bundle (there is no generic OakInk-Image mode), so
#: the granularity name does not say what the factors are.
FACTOR_NAMES = {
    "functional class x intent": (
        "OakInk object functional class, metaV2 object_id.json (container, maniptools, ...)",
        "OakInk intent id",
    ),
    "category x intent": (
        "OakInk named object category, metaV2 yodaobject_cat.json (mug, teapot, ...)",
        "OakInk intent id",
    ),
    "affordance x intent": (
        "OakInk affordance attribute, first listed in object_id.json. Note that for "
        "some objects OakInk's attribute string equals its category name, so an "
        "example cell can look like a category cell without being one.",
        "OakInk intent id",
    ),
    "category x subject": (
        "OakInk named object category, read out of the label suffix written by the "
        "bundle builder",
        "OakInk-Image capture subject id",
    ),
    "shape x fine intent": (
        "GRAB grasp-relevant shape class, caredex.data.grab.GRAB_SHAPE_CLASS",
        "GRAB fine intent verb, as annotated",
    ),
    "shape x intent class": (
        "GRAB grasp-relevant shape class",
        "GRAB documented intent class, caredex.data.grab.intent_class",
    ),
    "scene x verb": (
        "OakInk2 scene id",
        "leading verb of the OakInk2 task sentence",
    ),
    "scene x primitive": (
        "OakInk2 scene id",
        "FIRST primitive of the recording's chain (not the chain)",
    ),
    "action x tool": (
        "TACO action verb, first element of the (action, tool, object) triplet",
        "TACO tool, second element of the triplet",
    ),
    "annotated transitions": (
        "a primitive in the recording's chain",
        "the next primitive in that chain -- there is no two-factor grid here; the "
        "held-out unit is a consecutive pair inside the chain itself",
    ),
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def analyse_axis(dataset: str, axis: str, dirs: list[str], both_arch: bool) -> dict:
    rows = seed_rows(dirs, both_arch)
    if not rows:
        raise RuntimeError(f"{axis}: no rows at budget {BUDGET}")

    # Multi-part axes are concatenated. Overlapping seed ranges would double-count
    # a seed, so refuse rather than average a duplicate in.
    seen: dict[int, str] = {}
    for d, seed, _ in rows:
        if seed in seen and seen[seed] != d:
            raise RuntimeError(f"{axis}: seed {seed} appears in both {seen[seed]} and {d}")
        seen[seed] = d

    args_by_dir = {d: read_run(d)["args"] for d in dirs}
    # Concatenating dirs is only meaningful if they ran the same axis.
    keys = ("bundle", "granularity", "held_compositions", "min_chains",
            "min_per_composition", "window", "stride")
    base = {k: args_by_dir[dirs[0]][k] for k in keys}
    for d in dirs[1:]:
        diff = {k: (base[k], args_by_dir[d][k]) for k in keys if args_by_dir[d][k] != base[k]}
        if diff:
            raise RuntimeError(f"{axis}: {dirs[0]} and {d} differ in {diff}")

    rel = np.array([100 * r["penalty"] / r["naive"]["mse_target"] for _, _, r in rows])
    naive_abs = np.array([r["naive"]["mse_target"] for _, _, r in rows])
    inf_abs = np.array([r["informed"]["mse_target"] for _, _, r in rows])
    lo, hi = mean_ci(rel)

    # --- B: rebuild every one of those seeds' splits -----------------------
    gates = {d: gate_rows(d, args_by_dir[d]) for d in dirs}
    per_seed, mismatches, n_checks = [], [], 0
    sources: set[str] = set()
    for d, seed, row in rows:
        got = derive_seed(args_by_dir[d], seed)
        stored = row.get("soundness") or {}
        if stored:
            sources.add("results.json soundness")
        # The verification the docstring promises: the sweep stored the leak
        # fractions and the held/covered counts it actually saw. If the rebuild
        # is the same split, these agree exactly. Checks are counted, not
        # assumed to be four per seed -- an old row with no soundness block
        # would otherwise be reported as verified.
        for field in ("leak_naive", "leak_informed", "held", "covered"):
            if field in stored:
                n_checks += 1
                mine = got["derived_soundness"][field]
                if abs(float(stored[field]) - float(mine)) > 1e-9:
                    mismatches.append(
                        {"run": d, "seed": seed, "field": field, "source": "soundness",
                         "stored": stored[field], "derived": mine}
                    )

        # Second source: the coverage gate's own transcript, which exists for the
        # three runs that stored no soundness block.
        g = gates[d].get(seed)
        if g:
            sources.add(f"runs/gates/{GATE_TRANSCRIPTS[d]}")
            for field, mine in (
                ("held", got["n_held"]),
                ("covered", got["covered"]),
                ("swapped", got["swapped"]),
                ("per_comp", got["examples_per_covered_cell"]),
            ):
                n_checks += 1
                # per_comp is printed to two decimals in the transcript.
                if abs(float(g[field]) - float(mine)) > 5e-3:
                    mismatches.append(
                        {"run": d, "seed": seed, "field": field, "source": "gate transcript",
                         "stored": g[field], "derived": mine}
                    )

        got["penalty_pct"] = 100 * row["penalty"] / row["naive"]["mse_target"]
        per_seed.append(got)

    exposure = np.array([s["exposure"] for s in per_seed])
    contamination = np.array([s["contamination"] for s in per_seed])
    purity = np.array([s["purity"] for s in per_seed])

    # --- C: within-axis exposure/penalty correlation ----------------------
    if exposure.std() < 1e-12 or len(exposure) < 3:
        corr = {"n": len(exposure), "r": None, "p": None,
                "note": ("realised exposure is constant across seeds, so no "
                         "correlation is defined" if len(exposure) >= 3
                         else "fewer than 3 seeds")}
    else:
        r, p = stats.pearsonr(exposure, rel)
        corr = {"n": int(len(exposure)), "r": float(r), "p": float(p), "note": None}

    fine = load_labels(base["bundle"])
    coarse = fine if base["granularity"] == "fine" else coarsen_labels(fine, base["granularity"])
    left, right = FACTOR_NAMES[axis]

    # The held-out unit is a TRANSITION in the coarse label, which is the whole
    # cell on the eight two-factor axes and a consecutive pair inside the chain
    # on the transitions axis. Deriving the vocabularies from transitions_of
    # therefore describes both cases with the same code, instead of assuming a
    # coarse label splits on its first '->'.
    units = sorted({(a, b) for lab in coarse for a, b in transitions_of(lab)})
    lefts = sorted({a for a, _ in units})
    rights = sorted({b for _, b in units})

    return {
        "dataset": dataset,
        "axis": axis,
        "runs": dirs,
        "bundle": base["bundle"].replace("\\", "/"),
        "granularity": base["granularity"],
        "seeds": [s["seed"] for s in per_seed],
        "n_seeds": len(per_seed),
        "seed_rule": ("both architectures finished" if both_arch else "perframe only"),
        # A1
        "penalty_pct_of_naive": {
            "mean": float(rel.mean()),
            "sd": float(rel.std(ddof=1)) if len(rel) > 1 else None,
            "ci95_of_mean": [lo, hi],
            "median": float(np.median(rel)),
            "n_positive": int((rel > 0).sum()),
            "n": int(len(rel)),
        },
        # A2 -- the ratio's denominator, and the numerator's other half
        "absolute_mse_target": {
            "naive_mean": float(naive_abs.mean()),
            "naive_sd": float(naive_abs.std(ddof=1)) if len(naive_abs) > 1 else None,
            "informed_mean": float(inf_abs.mean()),
            "informed_sd": float(inf_abs.std(ddof=1)) if len(inf_abs) > 1 else None,
            "units": "MSE in normalised [-1, 1] joint-limit units, 27 pose channels",
        },
        # A3 / A4
        "split_settings": {
            "held_compositions": base["held_compositions"],
            "held_compositions_derived": int(per_seed[0]["n_held"]),
            "min_chains": base["min_chains"],
            "min_per_composition": base["min_per_composition"],
            "budget": BUDGET,
        },
        "windowing": {"window": base["window"], "stride": base["stride"],
                      "target_eval_stride": base["window"] // 2},
        # B
        "alignment_diagnostic": {
            "informed_exposure_mean": float(exposure.mean()),
            "informed_exposure_sd": float(exposure.std(ddof=1)) if len(exposure) > 1 else None,
            "informed_exposure_min": float(exposure.min()),
            "informed_exposure_max": float(exposure.max()),
            "naive_contamination_mean": float(contamination.mean()),
            "naive_contamination_max": float(contamination.max()),
            "target_purity_mean": float(purity.mean()),
            "target_purity_min": float(purity.min()),
            "contamination_trivially_zero": bool(contamination.max() == 0.0),
            "purity_trivially_one": bool(purity.min() == 1.0),
            "shared_training_fraction_mean": float(
                np.mean([s["shared_training_fraction"] for s in per_seed])
            ),
            "shared_training_fraction_note": (
                "the two arms are the same SIZE but not the same data: the "
                "informed arm's non-held remainder is redrawn from the naive pool "
                "rather than copied from the naive arm"
            ),
        },
        "coverage": {
            "cells_covered_mean": float(np.mean([s["covered"] for s in per_seed])),
            "held_cells": int(per_seed[0]["n_held"]),
            "seeds_fully_covered": int(sum(s["covered"] == s["n_held"] for s in per_seed)),
            "examples_per_covered_cell_mean": float(
                np.mean([s["examples_per_covered_cell"] for s in per_seed])
            ),
        },
        "pool_sizes": {
            "target_mean": float(np.mean([s["n_target"] for s in per_seed])),
            "naive_pool_mean": float(np.mean([s["n_naive_pool"] for s in per_seed])),
            "informed_extra_mean": float(np.mean([s["n_informed_extra"] for s in per_seed])),
            "n_trajectories": len(fine),
        },
        # C
        "exposure_penalty_correlation": corr,
        "split_verification": {
            "n_checks": n_checks,
            "sources": sorted(sources),
            "seeds_with_no_source": sorted(
                seed for d, seed, row in rows
                if not (row.get("soundness") or gates[d].get(seed))
            ),
        },
        "labels": {
            "left_factor": left,
            "right_factor": right,
            "example_fine_label": fine[0],
            "example_coarse_label": coarse[0],
            "example_held_unit": f"{units[0][0]}->{units[0][1]}",
            "n_distinct_fine_labels": len(set(fine)),
            "n_distinct_coarse_labels": len(set(coarse)),
            "n_left_values": len(lefts),
            "n_right_values": len(rights),
            "left_values_sample": lefts[:12],
            "right_values_sample": rights[:12],
            # Distinct held-out UNITS present, and of those the ones eligible to be
            # held out at all under this run's min_chains.
            "n_distinct_units_present": len(units),
            "n_units_eligible": int(per_seed[0]["candidate_pool_size"]),
            "trajectories_per_coarse_label": round(len(coarse) / len(set(coarse)), 2),
            "example_held_cells_seed0": per_seed[0]["held_compositions"],
            "fine_groups_carrying_a_held_unit_seed0": int(per_seed[0]["n_fine_groups_held"]),
        },
        "split_reproduction_mismatches": mismatches,
        "per_seed": per_seed,
    }


def markdown(per_axis: list[dict]) -> str:
    lines = []
    lines.append("### A. Penalty, its denominator, and the split settings "
                 f"(budget {BUDGET}, kind '{KIND}')\n")
    lines.append("| Dataset | Axis | n | Penalty % of naive | SD | 95% CI of mean | Median | "
                 "Pos/n | Naive MSE | Informed MSE | Held | min_chains | min/comp | Win/Stride |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in per_axis:
        p, m, s, w = (a["penalty_pct_of_naive"], a["absolute_mse_target"],
                      a["split_settings"], a["windowing"])
        lines.append(
            f"| {a['dataset']} | {a['axis']} | {p['n']} | {p['mean']:+.2f} | "
            f"{p['sd']:.2f} | {p['ci95_of_mean'][0]:+.2f} to {p['ci95_of_mean'][1]:+.2f} | "
            f"{p['median']:+.2f} | {p['n_positive']}/{p['n']} | {m['naive_mean']:.5f} | "
            f"{m['informed_mean']:.5f} | {s['held_compositions']} | {s['min_chains']} | "
            f"{s['min_per_composition']} | {w['window']}/{w['stride']} |"
        )

    lines.append("\n### B. Re-derived split: does the label describe what is held out?\n")
    lines.append("| Dataset | Axis | Informed exposure (mean, min-max) | Naive contamination | "
                 "Target purity | Cells covered / held | Examples per covered cell | "
                 "Arms' shared training data | Target set | Naive pool | Informed extra |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in per_axis:
        d, c, q = a["alignment_diagnostic"], a["coverage"], a["pool_sizes"]
        lines.append(
            f"| {a['dataset']} | {a['axis']} | {d['informed_exposure_mean']:.3f} "
            f"({d['informed_exposure_min']:.3f}-{d['informed_exposure_max']:.3f}) | "
            f"{d['naive_contamination_mean']:.3f} | {d['target_purity_mean']:.3f} | "
            f"{c['cells_covered_mean']:.2f} / {c['held_cells']} | "
            f"{c['examples_per_covered_cell_mean']:.1f} | "
            f"{d['shared_training_fraction_mean']:.3f} | {q['target_mean']:.0f} | "
            f"{q['naive_pool_mean']:.0f} | {q['informed_extra_mean']:.0f} |"
        )

    lines.append("\n### C. Within-axis correlation between realised exposure and penalty\n")
    lines.append("| Dataset | Axis | n | r | p | Note |")
    lines.append("|---|---|---|---|---|---|")
    for a in per_axis:
        c = a["exposure_penalty_correlation"]
        r = "n/a" if c["r"] is None else f"{c['r']:+.3f}"
        # Small p values print as 0.000 at three decimals, which hides the
        # difference between 0.049 and 1e-6.
        p = ("n/a" if c["p"] is None
             else (f"{c['p']:.1e}" if c["p"] < 1e-3 else f"{c['p']:.3f}"))
        lines.append(f"| {a['dataset']} | {a['axis']} | {c['n']} | {r} | {p} | {c['note'] or ''} |")

    lines.append("\n### Item 11. What a label is on each axis\n")
    lines.append("| Dataset | Axis | Granularity | Example fine label | Example held-out cell | "
                 "Fine labels | Left x right | Cells present | Eligible to hold out | Traj/cell |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in per_axis:
        L = a["labels"]
        lines.append(
            f"| {a['dataset']} | {a['axis']} | `{a['granularity']}` | "
            f"`{L['example_fine_label']}` | `{L['example_held_cells_seed0'][0]}` | "
            f"{L['n_distinct_fine_labels']} | "
            f"{L['n_left_values']} x {L['n_right_values']} | "
            f"{L['n_distinct_units_present']} | {L['n_units_eligible']} | "
            f"{L['trajectories_per_coarse_label']} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/axis_diagnostics.json")
    args = ap.parse_args()

    per_axis = [analyse_axis(*a) for a in AXES]

    bad = [m for a in per_axis for m in a["split_reproduction_mismatches"]]
    trivial = all(
        a["alignment_diagnostic"]["contamination_trivially_zero"]
        and a["alignment_diagnostic"]["purity_trivially_one"]
        for a in per_axis
    )

    out = {
        "what_this_is": (
            "Referee-requested quantities for the nine Table 1 axes, derived from "
            "each sweep's results.json plus a re-derivation of its per-seed split. "
            "No training, no GPU."
        ),
        "budget": BUDGET,
        "kind": KIND,
        "seed_rule": (
            "copied from scripts/verify_alignment_draft.py::rel -- rows at the "
            "reported budget, grouped by seed; on sweeps that also trained the "
            "modular bank, only seeds where both kinds finished"
        ),
        "split_reproduction": {
            "method": (
                "build_paired_split / sample_pools / assert_split_sound called with "
                "the run's own stored args and each of its seeds. They are seeded by "
                "np.random.default_rng and read only the bundle's labels, so the "
                "split is deterministic given the seed."
            ),
            "verified_against": (
                "(a) each row's stored 'soundness' block -- leak_naive, "
                "leak_informed, held, covered; and (b) where a full coverage-gate "
                "transcript exists in runs/gates/, its per-seed swapped / held / "
                "covered / per-comp columns. results.json does not store the held "
                "composition STRINGS, so the check is on those numbers, per seed, "
                "exactly. Three runs (oakink_official_category, "
                "oakink_official_attr, oakink_attr_more) predate the soundness "
                "block and are verified by (b) alone."
            ),
            "n_checks": sum(a["split_verification"]["n_checks"] for a in per_axis),
            "n_mismatches": len(bad),
            "mismatches": bad,
            "per_axis": [
                {"axis": a["axis"], **a["split_verification"]} for a in per_axis
            ],
            "seeds_with_no_verification_source": sum(
                len(a["split_verification"]["seeds_with_no_source"]) for a in per_axis
            ),
            "verdict": "REPRODUCED EXACTLY" if not bad else "MISMATCH -- DO NOT TRUST SECTION B",
        },
        "finding_B6_B7": {
            "contamination_zero_and_purity_one_on_every_axis": trivial,
            "verdict": (
                "The alignment diagnostic of Section 3.4 DOES NOT DISCRIMINATE among "
                "the nine Table 1 axes. Naive-arm contamination is exactly 0 and "
                "target purity is exactly 1 on every axis and every seed, because "
                "build_paired_split partitions trajectories by label: the naive pool "
                "is defined as the trajectories with no held cell and the target set "
                "as a subset of those with one. These two numbers are properties of "
                "the split code, not of the dataset, and cannot separate a "
                "well-labelled axis from a badly-labelled one. The draft's evidence "
                "for label-to-window alignment comes from DERIVED bundles "
                "(concatenated clips, primitive segments), where the label's "
                "relation to the scored window was manipulated; it does not come "
                "from these quantities on the Table 1 axes, and the text should not "
                "imply that it does."
                if trivial else
                "Contamination is not zero or purity is not one somewhere; see "
                "per-axis values, because that would be a real defect in a "
                "reported row."
            ),
            "where_variation_does_live": (
                "Realised informed exposure (B5) is the one quantity of the three "
                "that varies, and its ceiling is structural: sample_pools swaps at "
                "most budget//2 trajectories into the informed arm, so exposure is "
                "min(len(informed_extra), budget//2) / budget and is capped at 0.50. "
                "An axis below that cap is one whose held cells could not supply "
                "128 trajectories."
            ),
        },
        "incidental_findings": [
            {
                "what": "The paired arms are size-matched, not data-matched.",
                "detail": (
                    "sample_pools draws the informed arm's non-held remainder as a "
                    "fresh sample from the naive pool instead of reusing the naive "
                    "arm's trajectories, so the two arms share only 19-42% of their "
                    "256 training trajectories, and the fraction is set by pool size "
                    "(GRAB's 880-trajectory pool gives ~0.20, OakInk2's ~548 gives "
                    "~0.42). The draws are exchangeable, so this does not bias the "
                    "penalty, but it does add between-arm variance that 'identical "
                    "training size, differing only in composition coverage' does not "
                    "lead a reader to expect, and it differs systematically across "
                    "the three datasets being compared."
                ),
                "evidence": _lineno(
                    "scripts/experiment_paired_composition.py",
                    "keep = list(rng.choice(naive_pool, n - len(chosen), replace=False))",
                ),
                "values": {
                    a["axis"]: round(
                        a["alignment_diagnostic"]["shared_training_fraction_mean"], 3
                    )
                    for a in per_axis
                },
            },
            {
                "what": "Seeds within an axis are not independent draws over cells.",
                "detail": (
                    "min_chains leaves only a small eligible pool of held-out units "
                    "on some axes -- 12 eligible on OakInk-Image functional class x "
                    "intent, from which every seed draws 4 -- so the per-seed spread "
                    "understates the uncertainty over which cells were held out. The "
                    "'n_units_eligible' field per axis is the pool each seed drew "
                    "from."
                ),
                "values": {
                    a["axis"]: {
                        "eligible": a["labels"]["n_units_eligible"],
                        "held_per_seed": a["split_settings"]["held_compositions"],
                    }
                    for a in per_axis
                },
            },
        ],
        "notes": {
            "9_what_the_error_is": note_error_quantity(),
            "10_synthetic_controls": note_synthetic_controls(),
            "11_labels": note_labels(per_axis),
        },
        "axes": per_axis,
    }

    path = ROOT / args.out
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(markdown(per_axis))
    sr = out["split_reproduction"]
    print(f"\nsplit reproduction: {sr['verdict']} "
          f"({sr['n_checks']} checks over {sum(a['n_seeds'] for a in per_axis)} seeds, "
          f"{len(bad)} mismatches, "
          f"{sr['seeds_with_no_verification_source']} seeds with no stored record to "
          f"check against)")
    for a in per_axis:
        v = a["split_verification"]
        print(f"  {a['axis']:28s} {v['n_checks']:>4} checks  "
              f"{', '.join(v['sources']) or 'NO SOURCE'}"
              + (f"  unverified seeds: {v['seeds_with_no_source']}"
                 if v["seeds_with_no_source"] else ""))
    if bad:
        print(json.dumps(bad[:10], indent=2))
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
