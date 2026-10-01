"""Read the stride-4 re-runs and the GRAB diagnostics exactly as the manuscript reads Table II.

Statistic: per-seed penalty / naive mse_target x 100 at budget 256, perframe prior.
Tests: Welch two-sample t against the zero-truth control (runs/pc_easy_rerun), Welch against the unmodified
sweep at the same stride where one exists, one-sample t against zero. Readings are declared in
runs/PREREG_stride4_reruns.md and runs/PREREG_grab_diagnostics.md; this script only computes.
Sweeps still running are read as they stand and marked PARTIAL with their seed count.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


# Sweeps of the alignment test v2 run at budget 64; everything else is read at 256.
B64 = {"pc_oakink_b64", "pc_oakink_long3_aligned", "pc_oakink_long3_misaligned", "pc_oakink_long3_misaligned_cm",
       "pc_oakink_pair_aligned", "pc_oakink_pair_misaligned",
       "pc_oakink_b64_modular", "pc_oakink_long3_aligned_modular", "pc_oakink_long3_misaligned_modular",
       "pc_easy_rerun_modular"}
# Sweeps trained with the modular primitive bank are read on their modular rows (runs/PREREG_alignment_v2_modular.md).
MODULAR = {"pc_oakink_b64_modular", "pc_oakink_long3_aligned_modular", "pc_oakink_long3_misaligned_modular",
           "pc_easy_rerun_modular"}


def rel(dirs, expected=None, budget=None):
    vals, seeds = [], []
    for d in dirs:
        p = ROOT / "runs" / d / "results.json"
        if not p.exists():
            continue
        want = budget if budget is not None else (64 if d in B64 else 256)
        kind = "modular" if d in MODULAR else "perframe"
        for r in json.loads(p.read_text(encoding="utf-8"))["results"]:
            if r["kind"] == kind and r["budget"] == want:
                vals.append(100 * r["penalty"] / r["naive"]["mse_target"])
                seeds.append((d, r["seed"]))
    return np.array(vals), seeds


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    t = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q = stats.t.ppf(0.975, dof)
    return t.pvalue, diff, (diff - q * se, diff + q * se)


control, _ = rel(["pc_easy_rerun"])
control64, _ = rel(["pc_easy_rerun"], budget=64)
control64_mod, _ = rel(["pc_easy_rerun_modular"])
if len(control64_mod):
    print(f"modular control pc_easy_rerun_modular: {control64_mod.mean():+.2f}% naive, n {len(control64_mod)} (budget 64)")
print(f"control pc_easy_rerun: {control.mean():+.2f}% naive, n {len(control)} (budget 256); "
      f"{control64.mean():+.2f}%, n {len(control64)} (budget 64)\n")

# name, dirs, expected seeds, baseline dirs (same stride, unmodified)
SWEEPS = [
    ("GRAB shape x fine intent, stride 32 (v7 row)", ["grab_shape_v2"], 40, None),
    ("GRAB shape x fine intent, stride 4 (re-run)", ["grab_shape_s4"], 40, None),
    ("GRAB planted, stride 32", ["grab_planted_s32"], 40, ["grab_shape_v2"]),
    ("GRAB planted, stride 4", ["grab_planted_s4"], 40, ["grab_shape_s4"]),
    ("GRAB grasp-only, stride 32", ["grab_grasp_s32"], 40, ["grab_shape_v2"]),
    ("OakInk2 scene x primitive, primitive segments", ["oakink2_primseg_s4"], 20, ["oakink2_scene_primitive"]),
    ("OakInk-Image category x subject (no interaction)", ["oakink_category_subject"], 40,
     ["oakink_official_category", "oakink_category_rest"]),
    # Ablations: OakInk-Image's own confirmatory axis, with one property damaged at a time.
    # Readings are declared in runs/PREREG_oakink_ablations.md; the baseline is the undamaged axis.
    ("OakInk-Image category x intent, DOF damaged like GRAB", ["pc_oakink_dofdamage"], 40,
     ["oakink_official_category", "oakink_category_rest"]),
    ("OakInk-Image category x intent, grid thinned", ["pc_oakink_sparse"], 40,
     ["oakink_official_category", "oakink_category_rest"]),
    ("OakInk-Image category x intent, long clips misaligned", ["pc_oakink_long"], 12,
     ["oakink_official_category", "oakink_category_rest"]),
    ("OakInk-Image category x intent, long clips aligned", ["pc_oakink_longaligned"], 12,
     ["oakink_official_category", "oakink_category_rest"]),
    # Alignment test v2 (runs/PREREG_oakink_alignment_v2.md), budget 64, no clip reuse. The rows above for the long
    # clips are VOID (frame leak). Note: the control line printed at the top is the budget-256 control; the budget-64
    # control is +2.96% (pc_easy_rerun at budget 64). Compare v2 rows against pc_oakink_b64.
    ("v2 baseline, unmodified OakInk-Image, budget 64", ["pc_oakink_b64"], 40, None),
    ("v2 long3 aligned, budget 64", ["pc_oakink_long3_aligned"], 40, ["pc_oakink_b64"]),
    ("v2 long3 misaligned, budget 64", ["pc_oakink_long3_misaligned"], 40, ["pc_oakink_b64"]),
    # Alignment test v4, no constituent leak (runs/PREREG_oakink_alignment_v4.md). The v2 aligned row is confounded.
    ("v4 pair aligned, budget 64", ["pc_oakink_pair_aligned"], 40, ["pc_oakink_b64"]),
    ("v4 pair misaligned, budget 64", ["pc_oakink_pair_misaligned"], 40, ["pc_oakink_pair_aligned"]),
    # Clean-up sweeps, runs/PREREG_cleanup_runs.md.
    ("v3 long3 misaligned, category-matched, budget 64", ["pc_oakink_long3_misaligned_cm"], 40,
     ["pc_oakink_long3_aligned"]),
    ("OakInk2 primitive segments, recording-disjoint (seeds 200-239)", ["oakink2_primseg_recdisjoint"], 40,
     ["oakink2_scene_primitive"]),
    # Second architecture (modular bank), runs/PREREG_alignment_v2_modular.md; read against the modular control.
    ("v2m modular control, synthetic, budget 64", ["pc_easy_rerun_modular"], 20, None),
    ("v2m modular baseline, budget 64", ["pc_oakink_b64_modular"], 40, None),
    ("v2m modular long3 aligned, budget 64", ["pc_oakink_long3_aligned_modular"], 40, ["pc_oakink_b64_modular"]),
    ("v2m modular long3 misaligned, budget 64", ["pc_oakink_long3_misaligned_modular"], 40,
     ["pc_oakink_long3_aligned_modular"]),
    # Independent replication on fresh seeds 100-139, runs/PREREG_oakink2_primseg_replication.md.
    ("OakInk2 primitive segments, replication (seeds 100-139)", ["oakink2_primseg_rep"], 40,
     ["oakink2_scene_primitive"]),
    ("GRAB grasp-only, stride 4", ["grab_grasp_s4"], 40, ["grab_shape_s4"]),
    ("OakInk2 transitions, stride 16 (v7 row)", ["oakink2_paired_v2"], 12, None),
    ("OakInk2 transitions, stride 4 (re-run)", ["oakink2_transitions_s4"], 12, ["oakink2_paired_v2"]),
    ("OakInk2 subject x verb, held 3", ["oakink2_subject_verb_h3_a", "oakink2_subject_verb_h3_b",
                                        "oakink2_subject_verb_h3_c"], 40, None),
    ("synthetic planted pc_hard_rerun (reference)", ["pc_hard_rerun"], 18, None),
]

out = {}
for name, dirs, expected, base in SWEEPS:
    x, seeds = rel(dirs)
    if len(x) == 0:
        print(f"{name}: no rows yet")
        continue
    status = "complete" if len(x) >= expected else f"PARTIAL {len(x)}/{expected}"
    p0 = stats.ttest_1samp(x, 0).pvalue if len(x) > 1 else float("nan")
    if any(d in MODULAR for d in dirs):
        ctrl = control64_mod if len(control64_mod) > 1 else control64
    elif any(d in B64 for d in dirs):
        ctrl = control64
    else:
        ctrl = control
    pc, dc, ci = welch(x, ctrl)
    line = (f"{name}: {x.mean():+.2f}% naive (sd {x.std(ddof=1) if len(x) > 1 else float('nan'):.2f}) n {len(x)} [{status}]"
            f" | p vs 0 {p0:.3g} | vs control p {pc:.3g}, diff {dc:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    rec = dict(mean=x.mean(), n=len(x), status=status, p_zero=p0, p_control=pc, diff_control=dc, ci_control=ci)
    if base:
        b, _ = rel(base)
        pb, db, cib = welch(x, b)
        line += f" | vs {'+'.join(base)} p {pb:.3g}, diff {db:+.2f} [{cib[0]:+.2f}, {cib[1]:+.2f}]"
        rec.update(p_base=pb, diff_base=db, ci_base=cib)
    print(line)
    out[name] = rec

(ROOT / "runs" / "diagnostics_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
sys.exit(0)
