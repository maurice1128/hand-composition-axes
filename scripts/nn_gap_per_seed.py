"""Does the penalty track how much closer the informed training set lies to the targets than the naive one?

A referee's alternative to composition: the informed arm holds other fine labels of the held-out cell, so the
penalty may only measure whether it contains near-duplicates of the target motion. This script measures, for every
seed of a sweep and with no model, the mean over target windows of the nearest-neighbour RMS distance to the naive
arm's and to the informed arm's training windows, and the relative gap (d_naive - d_informed) / d_naive. It reuses the
window, feature and distance code of scripts/check_nearest_neighbour.py (stride-16 windows, all 32 frames x 27 DOF,
normalised units) and rebuilds each split with the experiment's own functions and the sweep's stored arguments; for a
permuted-grid sweep the stored mapping from fine label to sham cell is applied first, as the wrapper does.

The quantities that matter: (1) whether the gap differs between sweeps whose penalties differ (aligned against
misaligned joined clips; real against permuted grids), and (2) whether the per-seed gap predicts the per-seed penalty.

What it does not show: distances are at stride 16, so they are upper bounds on stride-4 distances; RMS weighs every
DOF and frame equally; a gap is compatible with retrieval and does not prove the model uses it.

Output: runs/nn_gap_per_seed.json
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import check_nearest_neighbour as nn  # noqa: E402  (main-guarded)
from caredex.data.base import TrajectoryBundle  # noqa: E402
from experiment_paired_composition import build_paired_split, coarsen_labels, sample_pools  # noqa: E402

SWEEPS = [  # (sweep dir, budget, sham?)
    ("pc_oakink_b64", 64, False),
    ("pc_oakink_pair_aligned", 64, False),
    ("pc_oakink_pair_misaligned", 64, False),
    ("oakink_official_category", 256, False),
    ("oakink_category_rest", 256, False),
    ("sham_oakink_category", 256, True),
    ("taco_action_tool", 256, False),
    ("sham_taco_action_tool", 256, True),
    ("grab_shape_s4", 256, False),
    ("sham_grab_shape", 256, True),
]


def penalties(d, budget):
    return {r["seed"]: 100 * r["penalty"] / r["naive"]["mse_target"]
            for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]
            if r["kind"] == "perframe" and r["budget"] == budget}


out = {}
for sweep, budget, sham in SWEEPS:
    a = json.loads((ROOT / "runs" / sweep / "results.json").read_text(encoding="utf-8"))["args"]
    bundle = TrajectoryBundle.load(ROOT / a["bundle"].replace("\\", "/"))
    fine = list(bundle.labels)
    if sham:
        mapping = json.loads((ROOT / "runs" / sweep / "sham_grid.json").read_text(encoding="utf-8"))["mapping"]
        coarse = [mapping[f] for f in fine]
    else:
        coarse = fine if a["granularity"] == "fine" else coarsen_labels(fine, a["granularity"])
    pen = penalties(sweep, budget)
    trajs = bundle.trajectories
    rows = []
    for seed in sorted(pen):
        shim = types.SimpleNamespace(labels=coarse)
        split = build_paired_split(shim, a["held_compositions"], seed, fine_labels=fine, min_chains=a["min_chains"])
        naive_idx, inf_idx = sample_pools(split, budget, seed, coarse, a["min_per_composition"])
        T_ds, _ = nn.window_table(trajs, list(split["target"]))
        U_idx = sorted(set(naive_idx) | set(inf_idx))
        U_ds, owner = nn.window_table(trajs, U_idx)
        key = np.random.default_rng(0).random(len(owner))
        sets = {"naive": nn.select(owner, set(naive_idx), key), "informed": nn.select(owner, set(inf_idx), key)}
        allpos = np.unique(np.concatenate(list(sets.values())))
        U = nn.features(U_ds, allpos)
        remap = {p: k for k, p in enumerate(allpos)}
        sets = {k: np.array([remap[p] for p in v]) for k, v in sets.items()}
        T = nn.features(T_ds, np.arange(len(T_ds)))
        d = nn.nn_dists(T, U, sets)
        dn, di = float(np.mean(d["naive"])), float(np.mean(d["informed"]))
        rows.append({"seed": seed, "d_naive": dn, "d_informed": di, "rel_gap": (dn - di) / dn,
                     "frac_closer_informed": float(np.mean(d["informed"] < d["naive"] - 1e-6)),
                     "penalty": pen[seed], "n_target_windows": len(T)})
    g = np.array([r["rel_gap"] for r in rows])
    p = np.array([r["penalty"] for r in rows])
    r, pv = stats.pearsonr(g, p) if len(rows) > 2 else (np.nan, np.nan)
    out[sweep] = {"budget": budget, "sham": sham, "n": len(rows), "rel_gap_mean": float(g.mean()),
                  "rel_gap_sd": float(g.std(ddof=1)), "penalty_mean": float(p.mean()),
                  "gap_penalty_r": float(r), "gap_penalty_p": float(pv), "per_seed": rows}
    print(f"{sweep:28s} n {len(rows):3d} gap {100 * g.mean():6.2f} % (sd {100 * g.std(ddof=1):.2f}) "
          f"penalty {p.mean():+6.2f} r {r:+.2f} p {pv:.3g}", flush=True)


def cmp(x, y):
    gx = np.array([r["rel_gap"] for r in out[x]["per_seed"]])
    gy = np.array([r["rel_gap"] for r in out[y]["per_seed"]])
    t = stats.ttest_ind(gx, gy, equal_var=False)
    return {"a": x, "b": y, "gap_a": float(gx.mean()), "gap_b": float(gy.mean()), "p": float(t.pvalue)}


out["_contrasts"] = [cmp("pc_oakink_pair_aligned", "pc_oakink_pair_misaligned"),
                     cmp("sham_oakink_category", "oakink_official_category"),
                     cmp("sham_taco_action_tool", "taco_action_tool"),
                     cmp("sham_grab_shape", "grab_shape_s4")]
for c in out["_contrasts"]:
    print(c)
(ROOT / "runs" / "nn_gap_per_seed.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
