"""A training-free surrogate for the paired compositional penalty.

What this is for
----------------
The measured quantity is a *paired* compositional penalty: the MSE of a prior
whose training trajectories excluded a label pairing, minus the MSE of one that
included it, on the **same** held-out target trajectories, as a percentage of
the naive error. It is large on OakInk-Image (12-18%) and absent on GRAB and
OakInk2 (~0-3%, against a zero-truth control that itself reads +2.59%).

Five explanations are already excluded by experiment -- instrument
insensitivity, retargeting quality, grid sparsity, near-duplicate retrieval and
window stride. What is missing is the *mechanism*. The obvious cheap statistic,
the excess interaction R^2 of ``scripts/screen_axes.py``, failed in both
directions and is deliberately not reused here.

The idea
--------
Rebuild the experiment with no training at all. Replace each prior by the
cheapest predictor that can express the same difference between the two arms --
a group mean over windows -- and keep *everything else* identical: the same
bundles, the same ``coarsen_labels``, the same ``build_paired_split`` and
``sample_pools`` with each sweep's own recorded arguments, the same target
trajectories.

For a target window belonging to a held composition ``(a, b)``:

    naive     mean window over training trajectories carrying ``a`` on the left
              (excluding any carrying ``a->b``)
            + mean window over training trajectories carrying ``b`` on the right
              (excluding any carrying ``a->b``)
            - global mean window
    informed  mean window over the training trajectories that DO carry ``a->b``
              in the informed arm's own training set

The naive predictor is additive by construction: it is the best it can do
knowing each factor separately and nothing about the pairing. The informed
predictor is a cell centroid. So

    surrogate penalty  = err_naive - err_informed
    surrogate % naive  = 100 * (err_naive - err_informed) / err_naive

is, by construction, *what the pairing is worth to a predictor that cannot
learn anything else*. If that tracks the measured penalty across axes, the
mechanism is first-order: the pairing carries mean structure that no additive
model can reach, and the datasets differ in how much.

What the surrogate is NOT
-------------------------
A centroid is linear and the real prior is a VAE; five seeds, not forty; and
windows at stride 16 (``screen_axes``' own setting) against the sweeps' stride
4. It cannot, and does not try to, predict the absolute size of anything.

Two stages, kept separate on purpose
------------------------------------
Stage 1 is the nine originally swept axes. Those numbers were known when this
surrogate was designed, so stage 1 is **description**, not evidence.
Stage 2 is eight results measured after the definition was fixed. Only stage 2
is a real test.

    .venv/Scripts/python.exe scripts/surrogate_penalty.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402

from experiment_paired_composition import (  # noqa: E402
    build_paired_split,
    coarsen_labels,
    sample_pools,
)
from experiment_data_efficiency import transitions_of  # noqa: E402
from screen_axes import features  # noqa: E402


#: One row per axis. ``run`` is the directory under ``runs/`` whose
#: ``results.json`` supplies BOTH the measured number and the arguments; nothing
#: here is invented. Two OakInk axes were swept as two directories with
#: disjoint seed ranges and identical settings, so they are one axis.
AXES = [
    # stage 1 -- descriptive: these nine were known before the surrogate existed
    ("oakink_official_category+oakink_category_rest", "OakInk-Image",
     ["oakink_official_category", "oakink_category_rest"], 15.2, 1),
    ("oakink_official_attr+oakink_attr_more", "OakInk-Image",
     ["oakink_official_attr", "oakink_attr_more"], 12.2, 1),
    ("oakink_n70_v2", "OakInk-Image", ["oakink_n70_v2"], 14.3, 1),
    ("oakink_class_v1", "OakInk-Image", ["oakink_class_v1"], 17.9, 1),
    ("grab_shape_v2", "GRAB", ["grab_shape_v2"], 0.7, 1),
    ("grab_shapeclass", "GRAB", ["grab_shapeclass"], 2.3, 1),
    ("oakink2_scene_primitive", "OakInk2", ["oakink2_scene_primitive"], 2.3, 1),
    ("oakink2_scene_verb", "OakInk2", ["oakink2_scene_verb"], 3.0, 1),
    ("oakink2_paired_v2", "OakInk2", ["oakink2_paired_v2"], 3.4, 1),
    # stage 2 -- held out: measured after the surrogate's definition was fixed
    ("oakink_category_subject", "OakInk-Image", ["oakink_category_subject"], 10.3, 2),
    ("pc_oakink_dofdamage", "OakInk-Image", ["pc_oakink_dofdamage"], 13.7, 2),
    ("pc_oakink_sparse", "OakInk-Image", ["pc_oakink_sparse"], 19.6, 2),
    ("oakink2_primseg_s4", "OakInk2", ["oakink2_primseg_s4"], 9.0, 2),
    ("grab_grasp_s32", "GRAB", ["grab_grasp_s32"], 2.2, 2),
    ("grab_planted_s32", "GRAB", ["grab_planted_s32"], 13.2, 2),
    ("pc_easy_rerun", "synthetic (zero-truth control)", ["pc_easy_rerun"], 2.6, 2),
    ("pc_hard_rerun", "synthetic (planted)", ["pc_hard_rerun"], 9.1, 2),
]


# ---------------------------------------------------------------------------
# small statistics, written out so this script needs nothing beyond numpy
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not fabricate an ordering."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    out = ranks.copy()
    i = 0
    s = v[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            out[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(x), _rank(y))


def _p_two_sided(r: float, n: int) -> float:
    """Two-sided p for a correlation, via the t transform.

    Uses scipy when it is importable and a continued-fraction incomplete beta
    otherwise, so the number exists on a machine without scipy.
    """
    if not np.isfinite(r) or n < 3 or abs(r) >= 1.0:
        return float("nan")
    import math

    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    try:
        from scipy import stats  # type: ignore

        return float(2 * stats.t.sf(t, df))
    except Exception:  # noqa: BLE001 - scipy is optional here
        x = df / (df + t * t)
        return float(_betainc(df / 2.0, 0.5, x))


def _betainc(a: float, b: float, x: float) -> float:
    import math

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


# ---------------------------------------------------------------------------
# the surrogate
# ---------------------------------------------------------------------------


def traj_index(x: np.ndarray, owner: np.ndarray, n_traj: int):
    """Sort windows by trajectory, and precompute per-trajectory sums and counts.

    Everything downstream needs either "the windows of trajectory t" or "the
    mean window of a set of trajectories". Sorting once turns the first into a
    slice and the second into a sum of precomputed rows, instead of a fresh
    scan of ``owner`` for every group of every seed.
    """
    d = x.shape[1]
    counts = np.bincount(owner, minlength=n_traj).astype(np.int64)
    order = np.argsort(owner, kind="mergesort")
    xs = x[order]
    edges = np.searchsorted(owner[order], np.arange(n_traj + 1))
    csum = np.concatenate([np.zeros((1, d)), np.cumsum(xs, axis=0)], axis=0)
    sums = csum[edges[1:]] - csum[edges[:-1]]
    del csum
    return sums, counts, xs, edges


def mean_of(sums: np.ndarray, counts: np.ndarray, idx) -> np.ndarray | None:
    idx = np.asarray(list(idx), dtype=np.intp)
    if len(idx) == 0:
        return None
    n = int(counts[idx].sum())
    if n == 0:
        return None
    return sums[idx].sum(axis=0) / n


def dispersion(xs: np.ndarray, edges: np.ndarray, sums: np.ndarray,
               counts: np.ndarray, groups: dict[str, list[int]],
               global_mean: np.ndarray) -> dict:
    """Within- vs between-group spread of windows, in the units of the errors.

    Within-group spread is computed **leave-one-trajectory-out**: a window is
    scored against its own group's centroid built from the *other* trajectories.
    Scoring it against a centroid it helped define would make a group of
    near-duplicate frames look arbitrarily tight, which is exactly the artefact
    the paired design spends its effort avoiding.
    """
    d = xs.shape[1]
    tot_w, n_w = 0.0, 0
    tot_b, n_b = 0.0, 0
    for tids in groups.values():
        tids = [t for t in tids if counts[t] > 0]
        if not tids:
            continue
        gs = sums[tids].sum(axis=0)
        gn = int(counts[tids].sum())
        cen = gs / gn
        tot_b += float(((cen - global_mean) ** 2).sum() / d) * gn
        n_b += gn
        if len(tids) < 2:
            continue
        for t in tids:
            rest_n = gn - int(counts[t])
            if rest_n <= 0:
                continue
            cen_t = (gs - sums[t]) / rest_n
            w = xs[edges[t] : edges[t + 1]]
            tot_w += float(((w - cen_t) ** 2).sum() / d)
            n_w += len(w)
    within = tot_w / n_w if n_w else float("nan")
    between = tot_b / n_b if n_b else float("nan")
    return {
        "within": within,
        "between": between,
        "ratio_between_within": between / within if within and np.isfinite(within) else float("nan"),
        "n_groups": len(groups),
    }


def surrogate_axis(bundle: TrajectoryBundle, x: np.ndarray, owner: np.ndarray,
                   args: dict, seeds: list[int]) -> dict:
    """Run the paired split for each seed and score the two group predictors."""
    n_traj = len(bundle.trajectories)
    fine = list(bundle.labels)
    gran = args["granularity"]
    coarse = coarsen_labels(fine, gran) if gran != "fine" else list(fine)
    bundle.labels = coarse

    sums, counts, xs, edges = traj_index(x, owner, n_traj)
    total_n = int(counts.sum())
    global_mean = sums.sum(axis=0) / total_n
    d = x.shape[1]

    # Which trajectories carry which transition, and which carry a given
    # primitive on each side. For every coarse axis a label is a single
    # ``L->R``, so these reduce to "cell", "all with left L", "all with right R".
    trans_of = [set(transitions_of(l)) for l in coarse]
    left_of = [{a for a, _ in t} for t in trans_of]
    right_of = [{b for _, b in t} for t in trans_of]

    budget = int(args["budgets"][0])
    per_seed = []
    for seed in seeds:
        try:
            split = build_paired_split(
                bundle, args["held_compositions"], seed,
                fine_labels=fine, min_chains=args["min_chains"],
            )
            naive_idx, informed_idx = sample_pools(
                split, budget, seed, coarse, args["min_per_composition"]
            )
        except Exception as exc:  # noqa: BLE001 - a seed the sweep also could not run
            per_seed.append({"seed": seed, "error": str(exc)[:120]})
            continue

        naive_idx = [int(i) for i in naive_idx]
        informed_idx = [int(i) for i in informed_idx]
        target = [int(i) for i in split["target"]]
        held = {(h.partition("->")[0], h.partition("->")[2])
                for h in split["held_compositions"]}

        # The target set is disjoint from both training sets by construction;
        # assert it rather than trust it, because a target leaking into its own
        # predictor would manufacture the whole effect.
        assert not (set(target) & set(naive_idx)), "target leaked into naive set"
        assert not (set(target) & set(informed_idx)), "target leaked into informed set"

        se_n = se_i = 0.0
        n_scored = 0
        n_cells_covered, n_cells_seen = 0, 0
        n_left_missing = n_right_missing = 0
        for t in target:
            if counts[t] == 0:
                continue
            w = xs[edges[t] : edges[t + 1]]
            mine = sorted(held & trans_of[t])
            if not mine:
                continue
            for (a, b) in mine:
                n_cells_seen += 1
                # additive prediction from the NAIVE arm's training set
                li = [i for i in naive_idx if a in left_of[i] and (a, b) not in trans_of[i]]
                ri = [i for i in naive_idx if b in right_of[i] and (a, b) not in trans_of[i]]
                ml = mean_of(sums, counts, li)
                mr = mean_of(sums, counts, ri)
                if ml is None:
                    n_left_missing += 1
                    ml = global_mean
                if mr is None:
                    n_right_missing += 1
                    mr = global_mean
                pred_n = ml + mr - global_mean

                # cell centroid from the INFORMED arm's training set
                ci = [i for i in informed_idx if (a, b) in trans_of[i]]
                pred_i = mean_of(sums, counts, ci)
                if pred_i is None:
                    # The informed arm never saw this cell either, so it knows
                    # exactly what the naive arm knows. Charging it the same
                    # error is the conservative reading -- it can only shrink
                    # the surrogate penalty, never inflate it.
                    pred_i = pred_n
                else:
                    n_cells_covered += 1

                se_n += float(((w - pred_n) ** 2).sum() / d)
                se_i += float(((w - pred_i) ** 2).sum() / d)
                n_scored += len(w)

        if n_scored == 0:
            per_seed.append({"seed": seed, "error": "no scorable target windows"})
            continue
        en, ei = se_n / n_scored, se_i / n_scored
        per_seed.append({
            "seed": seed,
            "err_naive": en,
            "err_informed": ei,
            "penalty": en - ei,
            "pct_naive": 100.0 * (en - ei) / en if en > 0 else float("nan"),
            "n_target": len(target),
            "n_windows_scored": n_scored,
            "held": len(held),
            "cell_coverage": n_cells_covered / max(n_cells_seen, 1),
            "additive_fallbacks": n_left_missing + n_right_missing,
            "n_naive_train": len(naive_idx),
            "n_informed_train": len(informed_idx),
        })

    ok = [r for r in per_seed if "error" not in r]
    out: dict = {"per_seed": per_seed, "n_seeds_ok": len(ok)}
    if ok:
        pct = np.array([r["pct_naive"] for r in ok])
        out.update({
            "err_naive": float(np.mean([r["err_naive"] for r in ok])),
            "err_informed": float(np.mean([r["err_informed"] for r in ok])),
            "penalty": float(np.mean([r["penalty"] for r in ok])),
            "surrogate_pct_naive": float(pct.mean()),
            "surrogate_pct_sd": float(pct.std(ddof=1)) if len(pct) > 1 else float("nan"),
            "cell_coverage": float(np.mean([r["cell_coverage"] for r in ok])),
            "mean_target": float(np.mean([r["n_target"] for r in ok])),
        })

    # Description of the axis itself, split-free: how tight is a cell relative
    # to how far apart cells are, and the same for each factor on its own.
    # The dispersion groups are built from the SAME transitions the surrogate
    # predicts, not from the raw coarse string. On a coarse axis a label is one
    # ``L->R`` and the two are identical; on a ``fine`` axis the raw string is a
    # whole chain carried by about one trajectory, and a within-cell spread over
    # singleton cells would describe nothing. A trajectory joins every
    # transition group it carries, so the groupings overlap by construction.
    cell_g: dict[str, set[int]] = {}
    left_g: dict[str, set[int]] = {}
    right_g: dict[str, set[int]] = {}
    for i, ts in enumerate(trans_of):
        for (a, b) in ts:
            cell_g.setdefault(f"{a}->{b}", set()).add(i)
            left_g.setdefault(a, set()).add(i)
            right_g.setdefault(b, set()).add(i)
    disp = {}
    for name, g in (("cell", cell_g), ("left", left_g), ("right", right_g)):
        disp[name] = dispersion(
            xs, edges, sums, counts,
            {k: sorted(v) for k, v in g.items()}, global_mean,
        )
    out["dispersion"] = disp
    out["n_cells"] = len(cell_g)
    out["n_left"] = len(left_g)
    out["n_right"] = len(right_g)
    out["n_label_cells"] = len(set(coarse))
    out["traj_per_cell"] = n_traj / max(len(cell_g), 1)
    return out


def measured_pct(dirs: list[str], budget: int) -> tuple[float, int]:
    """The sweep's own number: mean over seeds of (naive-informed)/naive."""
    rows = []
    for d in dirs:
        blob = json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))
        rows += [r for r in blob["results"]
                 if r["kind"] == "perframe" and r["budget"] == budget]
    n = np.array([r["naive"]["mse_target"] for r in rows])
    i = np.array([r["informed"]["mse_target"] for r in rows])
    return float(100.0 * np.mean((n - i) / n)), len(rows)


def correlate(rows: list[dict]) -> dict:
    s = np.array([r["surrogate_pct_naive"] for r in rows])
    m = np.array([r["measured_pct_naive"] for r in rows])
    r_p, r_s = _pearson(s, m), _spearman(s, m)
    return {
        "n": len(rows),
        "pearson": r_p,
        "pearson_p": _p_two_sided(r_p, len(rows)),
        "spearman": r_s,
        "spearman_p": _p_two_sided(r_s, len(rows)),
        "mean_abs_error_pct_points": float(np.mean(np.abs(s - m))),
        "max_abs_error_pct_points": float(np.max(np.abs(s - m))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-per-traj", type=int, default=64,
                    help="cap on windows per trajectory, as in screen_axes; 64 "
                         "leaves every bundle here unsubsampled except OakInk2's "
                         "longest sequences and keeps peak RAM under a GB.")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default="runs/surrogate_penalty.json")
    ap.add_argument("--table", default="runs/gates/surrogate_penalty.txt")
    a = ap.parse_args()

    # Axes sharing a bundle are run together so each file is read and windowed
    # once. OakInk2's bundles are 570k frames and windowing them five times over
    # would dominate the runtime.
    by_bundle: dict[str, list[tuple]] = {}
    meta: dict[str, dict] = {}
    for name, dataset, dirs, measured, stage in AXES:
        blob = json.loads((ROOT / "runs" / dirs[0] / "results.json").read_text(encoding="utf-8"))
        args = blob["args"]
        bundle_path = str(args["bundle"]).replace("\\", "/")
        meta[name] = {
            "axis": name, "dataset": dataset, "runs": dirs, "stage": stage,
            "bundle": bundle_path, "granularity": args["granularity"],
            "held_compositions": args["held_compositions"],
            "min_chains": args["min_chains"],
            "min_per_composition": args["min_per_composition"],
            "budget": int(args["budgets"][0]),
            "sweep_stride": args["stride"],
        }
        by_bundle.setdefault(bundle_path, []).append((name, args, measured))

    results: dict[str, dict] = {}
    t0 = time.time()
    for bundle_path, entries in by_bundle.items():
        b = TrajectoryBundle.load(ROOT / bundle_path)
        rng = np.random.default_rng(a.feature_seed)
        x, owner = features(b, a.window, a.stride, a.max_per_traj, rng)
        print(f"[{bundle_path}] {len(b.trajectories)} traj -> {len(x)} windows "
              f"({x.nbytes / 1e6:.0f} MB)", flush=True)
        orig_labels = list(b.labels)
        for name, args, measured in entries:
            # surrogate_axis coarsens labels in place, exactly as the
            # experiment's own main() does, so each axis on this bundle gets a
            # view carrying the file's original labels.
            fresh = TrajectoryBundle(
                trajectories=b.trajectories, fps=b.fps,
                labels=list(orig_labels), meta=b.meta,
            )
            r = surrogate_axis(fresh, x, owner, args, a.seeds)
            m, n_rows = measured_pct(meta[name]["runs"], meta[name]["budget"])
            r.update(meta[name])
            r["measured_pct_naive"] = m
            r["measured_pct_reported"] = measured
            r["measured_n_seeds"] = n_rows
            results[name] = r
            print(f"   {name:<46} surrogate {r.get('surrogate_pct_naive', float('nan')):>7.2f}%"
                  f"   measured {m:>6.2f}%", flush=True)
        del x, owner, b

    rows = [results[n] for n, *_ in AXES]
    ok = [r for r in rows if "surrogate_pct_naive" in r]
    s1 = [r for r in ok if r["stage"] == 1]
    s2 = [r for r in ok if r["stage"] == 2]
    corr = {"stage1_descriptive": correlate(s1), "stage2_heldout": correlate(s2),
            "pooled": correlate(ok)}

    out = {
        "config": vars(a),
        "definition": (
            "surrogate % naive = 100*(err_naive-err_informed)/err_naive, where "
            "err_naive is the MSE of an additive (left-mean + right-mean - global-mean) "
            "window prediction built from the naive arm's training trajectories and "
            "err_informed the MSE of the held cell's centroid built from the informed "
            "arm's training trajectories, on the identical target windows of "
            "build_paired_split."
        ),
        "axes": rows,
        "correlations": corr,
    }
    outp = ROOT / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = []
    W = "=" * 118
    lines.append(W)
    lines.append("TRAINING-FREE SURROGATE OF THE PAIRED COMPOSITIONAL PENALTY")
    lines.append("nearest-centroid vs additive group means on 32-frame windows, stride 16, "
                 f"seeds {a.seeds}")
    lines.append(W)
    hdr = (f"{'axis':<46}{'dataset':<17}{'cells':>6}{'cov':>6}"
           f"{'surr%':>8}{'sd':>7}{'meas%':>8}{'err':>8}")
    lines.append(hdr)
    lines.append("-" * 118)
    for stage, title in ((1, "STAGE 1 - DESCRIPTIVE (axes known when the surrogate was defined)"),
                         (2, "STAGE 2 - HELD OUT (measured after the definition was fixed)")):
        lines.append("")
        lines.append(title)
        for r in rows:
            if r["stage"] != stage:
                continue
            if "surrogate_pct_naive" not in r:
                lines.append(f"{r['axis']:<46}{r['dataset']:<17}  no usable seed")
                continue
            lines.append(
                f"{r['axis']:<46}{r['dataset']:<17}{r['n_cells']:>6}"
                f"{r['cell_coverage']:>6.2f}{r['surrogate_pct_naive']:>8.2f}"
                f"{r['surrogate_pct_sd']:>7.2f}{r['measured_pct_naive']:>8.2f}"
                f"{r['surrogate_pct_naive'] - r['measured_pct_naive']:>+8.2f}"
            )
    lines.append("")
    lines.append("-" * 118)
    for key, c in corr.items():
        lines.append(
            f"{key:<22} n={c['n']:<3} pearson r={c['pearson']:+.3f} (p={c['pearson_p']:.4f})  "
            f"spearman rho={c['spearman']:+.3f} (p={c['spearman_p']:.4f})  "
            f"MAE={c['mean_abs_error_pct_points']:.2f}pp  max={c['max_abs_error_pct_points']:.2f}pp"
        )
    lines.append("")
    lines.append(W)
    lines.append("DISPERSION -- why. within = leave-one-trajectory-out spread of windows about "
                 "their own group's")
    lines.append("centroid; between = spread of group centroids about the global mean. "
                 "Both in normalised MSE units.")
    lines.append(W)
    lines.append(f"{'axis':<46}{'cell w':>9}{'cell b':>9}{'cell b/w':>10}"
                 f"{'left b/w':>10}{'right b/w':>11}{'surr%':>8}{'meas%':>8}")
    lines.append("-" * 118)
    for r in rows:
        if "surrogate_pct_naive" not in r:
            continue
        d = r["dispersion"]
        lines.append(
            f"{r['axis']:<46}{d['cell']['within']:>9.4f}{d['cell']['between']:>9.4f}"
            f"{d['cell']['ratio_between_within']:>10.3f}"
            f"{d['left']['ratio_between_within']:>10.3f}"
            f"{d['right']['ratio_between_within']:>11.3f}"
            f"{r['surrogate_pct_naive']:>8.2f}{r['measured_pct_naive']:>8.2f}"
        )
    lines.append("")
    lines.append("CAVEATS. Five seeds per axis, not the sweeps' 12-130. The predictor is a group "
                 "mean and is")
    lines.append("therefore linear; the measured prior is a VAE and is not. Windows here are "
                 "stride 16, the sweeps'")
    lines.append("were stride 4 on every axis but grab_*_s32 and oakink2_paired_v2. Stage 1 is "
                 "description -- those")
    lines.append("nine numbers were known when the surrogate was written. Only stage 2 is a test.")
    tab = ROOT / a.table
    tab.parent.mkdir(parents=True, exist_ok=True)
    tab.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {outp}\nwrote {tab}\n{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
