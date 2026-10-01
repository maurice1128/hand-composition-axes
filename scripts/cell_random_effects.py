"""Inference that treats the held-out cell, not the seed, as the unit of replication.

Seeds of one sweep draw 4-6 held cells from a pool of 12-60, so the same cells recur and per-seed penalties are
correlated. Model, per sweep: y_s = mu + (1/k_s) * sum_{c in H_s} u_c + e_s, with u_c ~ N(0, tau^2) a cell effect and
e_s ~ N(0, sigma^2) the rest (arm fill, training noise). Then Cov(y_s, y_t) = tau^2 |H_s & H_t| / (k_s k_t) +
sigma^2 [s = t]. tau^2 and sigma^2 are fitted by maximum likelihood, mu by generalised least squares, and its standard
error is compared with the naive one that treats seeds as independent. Differences between two sweeps are tested
with a normal approximation on the two GLS estimates, which are independent because the sweeps share no models.

Held cells per seed are re-derived with the experiment's own split function and each sweep's stored arguments (and,
for permuted grids, the stored mapping from fine label to sham cell).

Output: runs/cell_random_effects.json
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import axis_diagnostics as ad  # noqa: E402  (main-guarded)
from experiment_paired_composition import build_paired_split, coarsen_labels  # noqa: E402

SWEEPS = {  # name: (dirs, budget, both_arch, sham)
    "control_256": (["pc_easy_rerun"], 256, False, False),
    "control_64": (["pc_easy_rerun"], 64, False, False),
    "planted_synthetic": (["pc_hard_rerun"], 256, False, False),
    "functional class x intent": (["oakink_class_v1"], 256, True, False),
    "category x intent": (["oakink_official_category", "oakink_category_rest"], 256, True, False),
    "affordance x intent": (["oakink_official_attr", "oakink_attr_more"], 256, True, False),
    "category x subject": (["oakink_category_subject"], 256, False, False),
    "action x tool": (["taco_action_tool"], 256, False, False),
    "shape x fine intent": (["grab_shape_s4"], 256, False, False),
    "shape x intent class": (["grab_shapeclass"], 256, True, False),
    "scene x verb": (["oakink2_scene_verb"], 256, True, False),
    "scene x primitive": (["oakink2_scene_primitive"], 256, False, False),
    "annotated transitions": (["oakink2_transitions_s4"], 256, False, False),
    "permuted category x intent": (["sham_oakink_category"], 256, False, True),
    "permuted shape x fine intent": (["sham_grab_shape"], 256, False, True),
    "permuted action x tool": (["sham_taco_action_tool"], 256, False, True),
    "pair unmodified": (["pc_oakink_b64"], 64, False, False),
    "pair aligned": (["pc_oakink_pair_aligned"], 64, False, False),
    "pair misaligned": (["pc_oakink_pair_misaligned"], 64, False, False),
}


def seed_values(dirs, budget, both):
    out = []
    for d in dirs:
        by = {}
        for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]:
            if r["budget"] == budget:
                by.setdefault(r["seed"], {})[r["kind"]] = r
        for s in sorted(by):
            v = by[s]
            if "perframe" in v and (not both or "modular" in v):
                out.append((d, s, 100 * v["perframe"]["penalty"] / v["perframe"]["naive"]["mse_target"]))
    return out


def held_sets(dirs, rows, sham):
    cache = {}
    sets = []
    for d, s, _ in rows:
        if d not in cache:
            a = json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["args"]  # read_run refuses a stale list
            fine = ad.load_labels(a["bundle"])
            if sham:
                m = json.loads((ROOT / "runs" / d / "sham_grid.json").read_text(encoding="utf-8"))["mapping"]
                coarse = [m[f] for f in fine]
            else:
                coarse = fine if a["granularity"] == "fine" else coarsen_labels(fine, a["granularity"])
            cache[d] = (a, fine, coarse)
        a, fine, coarse = cache[d]
        split = build_paired_split(types.SimpleNamespace(labels=coarse), a["held_compositions"], s,
                                   fine_labels=fine, min_chains=a["min_chains"])
        sets.append(frozenset(split["held_compositions"]))
    return sets


def fit(y, H):
    n = len(y)
    A = np.array([[len(H[i] & H[j]) / (len(H[i]) * len(H[j])) for j in range(n)] for i in range(n)])
    one = np.ones(n)

    def nll(theta):
        tau2, sig2 = np.exp(theta)
        V = tau2 * A + sig2 * np.eye(n)
        L = np.linalg.cholesky(V)
        Vi1 = np.linalg.solve(V, one)
        mu = (one @ np.linalg.solve(V, y)) / (one @ Vi1)
        r = y - mu
        return 0.5 * (2 * np.log(np.diag(L)).sum() + r @ np.linalg.solve(V, r))

    v0 = np.var(y, ddof=1)
    best = min((optimize.minimize(nll, np.log([v0 * f, v0 * (1 - f)]), method="Nelder-Mead",
                                  options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000}) for f in (0.1, 0.5, 0.9)),
               key=lambda o: o.fun)
    tau2, sig2 = np.exp(best.x)
    V = tau2 * A + sig2 * np.eye(n)
    Vi1 = np.linalg.solve(V, one)
    mu = (one @ np.linalg.solve(V, y)) / (one @ Vi1)
    se = 1 / np.sqrt(one @ Vi1)
    return {"mu": float(mu), "se_gls": float(se), "se_naive": float(np.std(y, ddof=1) / np.sqrt(n)),
            "tau2": float(tau2), "sigma2": float(sig2), "n": n,
            "distinct_cells": len(set().union(*H)), "cells_per_seed": float(np.mean([len(h) for h in H]))}


res = {}
for name, (dirs, budget, both, sham) in SWEEPS.items():
    rows = seed_values(dirs, budget, both)
    H = held_sets(dirs, rows, sham)
    y = np.array([v for _, _, v in rows])
    res[name] = fit(y, H)
    f = res[name]
    print(f"{name:30s} n {f['n']:3d} cells {f['distinct_cells']:3d} mu {f['mu']:+6.2f} se {f['se_gls']:.2f} "
          f"(naive {f['se_naive']:.2f}) tau {np.sqrt(f['tau2']):.2f} sigma {np.sqrt(f['sigma2']):.2f}", flush=True)


def contrast(a, b):
    d = res[a]["mu"] - res[b]["mu"]
    se = np.hypot(res[a]["se_gls"], res[b]["se_gls"])
    return {"a": a, "b": b, "diff": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se,
            "p": float(2 * stats.norm.sf(abs(d) / se))}


C = [contrast(k, "control_256") for k in SWEEPS if not k.startswith(("control", "pair"))]
C += [contrast("permuted category x intent", "category x intent"), contrast("permuted action x tool", "action x tool"),
      contrast("permuted shape x fine intent", "shape x fine intent"),
      contrast("pair misaligned", "pair aligned"), contrast("pair misaligned", "pair unmodified"),
      contrast("pair aligned", "pair unmodified"), contrast("pair misaligned", "control_64"),
      contrast("pair unmodified", "control_64"), contrast("pair aligned", "control_64")]
for c in C:
    print(f"{c['a']:30s} - {c['b']:26s} {c['diff']:+6.2f} [{c['lo']:+6.2f}, {c['hi']:+6.2f}] p {c['p']:.2g}")
(ROOT / "runs" / "cell_random_effects.json").write_text(json.dumps({"sweeps": res, "contrasts": C}, indent=1),
                                                          encoding="utf-8")
