"""Recompute every number in docs/PAPER_ALIGNMENT_DRAFT.md from the stored per-seed results and require the exact
string to appear in the draft. Reads finished sweeps only; it never touches a sweep that is still running.

Statistic: per-seed penalty / naive mse_target x 100 on the perframe prior. Tests: two-sided Welch on per-seed values.
Seed rule, as in the earlier manuscript: on sweeps that also trained the modular bank, only seeds where both finished.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DRAFT = (ROOT / "docs" / "PAPER_ALIGNMENT_DRAFT.md").read_text(encoding="utf-8")
fails = 0


def rel(dirs, budget=256, both_arch=False):
    vals = []
    for d in dirs:
        by = {}
        for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]:
            if r["budget"] == budget:
                by.setdefault(r["seed"], {})[r["kind"]] = r
        for v in by.values():
            if "perframe" not in v or (both_arch and "modular" not in v):
                continue
            vals.append(100 * v["perframe"]["penalty"] / v["perframe"]["naive"]["mse_target"])
    return np.array(vals)


def welch(a, b):
    t = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q = stats.t.ppf(0.975, df)
    d = a.mean() - b.mean()
    return t.pvalue, d, d - q * se, d + q * se


def sci(p):
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.1f} × 10{sup}"


FLAT = " ".join(DRAFT.split())  # the draft is hard-wrapped, so compare with line breaks collapsed


def check(label, s):
    global fails
    ok = " ".join(s.split()) in FLAT
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label:58s} {s}")


c256 = rel(["pc_easy_rerun"])
c64 = rel(["pc_easy_rerun"], budget=64)
hard = rel(["pc_hard_rerun"])
check("control, budget 256", f"+{c256.mean():.2f} % of the naive error over {len(c256)} seeds at a budget of 256 (SD {c256.std(ddof=1):.1f})")
check("control, budget 64", f"and +{c64.mean():.2f} % at")
check("planted synthetic control", f"returned +{hard.mean():.2f} % over {len(hard)} seeds")

TABLE1 = [
    ("functional class × intent", ["oakink_class_v1"], True),
    ("category × intent", ["oakink_official_category", "oakink_category_rest"], True),
    ("affordance × intent", ["oakink_official_attr", "oakink_attr_more"], True),
    ("category × subject", ["oakink_category_subject"], False),
    ("shape × fine intent", ["grab_shape_s4"], False),
    ("shape × intent class", ["grab_shapeclass"], True),
    ("scene × verb", ["oakink2_scene_verb"], True),
    ("scene × primitive", ["oakink2_scene_primitive"], False),
    ("annotated transitions", ["oakink2_transitions_s4"], False),
]
print("\nTable 1")
# Exposure and held cells come from scripts/axis_diagnostics.py, which re-derived every seed's split (1,760 checks,
# 0 mismatches against stored records). Everything else in the row is recomputed here from results.json.
DIAG = {a["axis"]: a for a in json.loads((ROOT / "runs" / "axis_diagnostics.json").read_text(encoding="utf-8"))["axes"]}


def rows_of(dirs, both_arch):
    out = []
    for d in dirs:
        by = {}
        for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]:
            if r["budget"] == 256:
                by.setdefault(r["seed"], {})[r["kind"]] = r
        out += [v["perframe"] for v in by.values() if "perframe" in v and (not both_arch or "modular" in v)]
    return out


upper, cover = [], {}
for axis, dirs, both in TABLE1:
    R = rows_of(dirs, both)
    x = np.array([100 * r["penalty"] / r["naive"]["mse_target"] for r in R])
    assert np.allclose(np.sort(x), np.sort(rel(dirs, both_arch=both)))
    naive = np.mean([r["naive"]["mse_target"] for r in R])
    n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    p, d, lo, hi = welch(x, c256)
    pstr = "< 10⁻¹²" if p < 1e-12 else (sci(p) if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.2f}"))
    dg = DIAG[axis.replace("×", "x")]
    ps = dg["per_seed"]
    assert dg["n_seeds"] == n, (axis, dg["n_seeds"], n)
    expo_ = np.mean([r["exposure"] for r in ps])
    cover[axis] = (sum(not (r["covered"] / r["n_held"] > 0.8 and r["examples_per_covered_cell"] >= 3.0) for r in ps), n)
    check(f"  {axis}", f"| {axis} | {ps[0]['n_held']} | {n} | +{x.mean():.1f} | {x.std(ddof=1):.1f} | "
                       f"{x.mean() - h:+.1f} to {x.mean() + h:+.1f} | {int((x > 0).sum())}/{n} | {naive:.4f} | "
                       f"{expo_:.2f} | {pstr} |")
    if x.mean() < 5:
        upper.append(hi)
check("largest interval upper bound on the null axes", f"reached at most +{max(upper):.1f} percentage points")
tr4, tr16 = rel(["oakink2_transitions_s4"]), rel(["oakink2_paired_v2"], both_arch=True)
check("transitions at stride 4", f"did (+{tr4.mean():.2f} %, {int((tr4 > 0).sum())} of {len(tr4)} seeds positive)")
check("transitions at stride 16", f"had returned +{tr16.mean():.2f} % (p = {welch(tr16, c256)[0]:.2f} against the control); the two strides differed (p = {welch(tr4, tr16)[0]:.4f})")
check("transitions in the abstract", f"contains carried +{tr4.mean():.1f} %")
print("\nDiagnostics (referee round 1)")
a, c5, f3 = cover["affordance × intent"], cover["category × intent"], cover["functional class × intent"]
check("coverage, recounted per seed", f"it failed on {a[0]} of {a[1]} seeds of affordance by intent, {c5[0]} of {c5[1]} of category by intent, {f3[0]} of {f3[1]} of functional class by intent, and at most 3 of 40 elsewhere")
assert max(v[0] for v in cover.values() if v[1] == 40) <= 3


def absdiff(k):
    v = DIAG[k]["absolute_mse_target"]
    return v["naive_mean"] - v["informed_mean"]


def expo(k):
    return np.mean([r["exposure"] for r in DIAG[k]["per_seed"]])


def rp(k):
    c = DIAG[k]["exposure_penalty_correlation"]
    return c["r"], c["p"]


check("absolute penalties", f"by {absdiff('category x intent'):.4f} on OakInk-Image category by intent, by {absdiff('shape x fine intent'):.4f} on GRAB shape by fine intent and by {absdiff('scene x primitive'):.4f} on OakInk2 scene by primitive")
check("GRAB is the most exposed", f"({expo('shape x fine intent'):.2f} and {expo('shape x intent class'):.2f})")
assert min(expo(k) for k in DIAG if k.startswith("shape")) >= max(expo(k) for k in ("category x intent", "affordance x intent", "category x subject"))
check("OakInk2 exposure range", f"shared an exposure of {expo('scene x verb'):.2f} to {expo('scene x primitive'):.2f}")
r1, p1 = rp("category x intent"); r2, p2 = rp("affordance x intent"); r3, p3 = rp("category x subject")
check("dose-response where a penalty exists", f"category by intent r = {r1:+.2f}, p = {p1:.3f}; affordance by intent r = {r2:+.2f}, p = {sci(p2)}; category by subject r = {r3:+.2f}, p = {sci(p3)}")
r4, p4 = rp("shape x fine intent"); r5, p5 = rp("scene x primitive")
check("no dose-response where none exists", f"GRAB shape by fine intent r = {r4:+.2f}, p = {p4:.2f}; OakInk2 scene by primitive r = {r5:+.2f}, p = {p5:.2f}")
sh = [np.mean([r["shared_training_fraction"] for r in DIAG[k]["per_seed"]]) for k in DIAG]
el = [DIAG[k]["per_seed"][0]["candidate_pool_size"] for k in DIAG]
check("arms share part of their training data", f"the arms shared {100 * min(sh):.0f} % to {100 * max(sh):.0f} % of their training trajectories")
check("eligible cells per axis", f"of which there were {min(el)} to {max(el)} per axis")
MC = json.loads((ROOT / "runs" / "motion_contamination_oakink2.json").read_text(encoding="utf-8"))
LC = json.loads((ROOT / "runs" / "label_coverage_oakink2.json").read_text(encoding="utf-8"))
rec, frm, pur = (100 * MC[k]["mean"] for k in ("naive_motion_contaminated_recordings", "naive_motion_contaminated_frames", "target_frame_purity_first_span"))
check("OakInk2 label coverage", f"covered a median of {100 * LC['first_primitive_share']['median']:.0f} % of the recording's frames; {LC['recordings_with_one_primitive']} of {LC['recordings']} recordings held a single primitive")
check("OakInk2 motion-level contamination", f"{rec:.1f} % of the naive arm's training recordings contained a held-out scene and primitive after their first primitive ({frm:.1f} % of its training frames), and {pur:.0f} % of the target recordings' frames")
check("Table 3, OakInk2 whole recordings", f"| OakInk2, whole recordings | {rec:.1f} % | {pur:.0f} % | +{rel(['oakink2_scene_primitive']).mean():.2f} |")
check("Table 3, OakInk2 segments", f"| OakInk2, primitive segments | 0 % | 100 % | +{rel(['oakink2_primseg_rep']).mean():.2f} |")
check("Introduction, label coverage", f"the labelled primitive covers a median of {100 * LC['first_primitive_share']['median']:.0f} % of a recording's frames")
assert MC["seeds"] == 40 and MC["naive_label_contamination"] == 0.0

# runs/oakink2_transitions_rep (seeds 100-139, runs/PREREG_oakink2_transitions_replication.md) is a separate,
# pre-registered replication. It is deliberately NOT read here until it has all 40 seeds and the draft cites it.

print("\nExclusion experiments")
planted, grab32 = rel(["grab_planted_s32"]), rel(["grab_shape_v2"], both_arch=True)
check("planted GRAB", f"GRAB returned +{planted.mean():.2f} % over {len(planted)} seeds, against +{grab32.mean():.2f} %")
check("planted vs unmodified p", f"(p = {sci(welch(planted, grab32)[0])})")
grab4 = rel(["grab_shape_s4"])
check("stride 32 vs stride 4 on the unmodified GRAB axis", f"value in Table 1 (p = {welch(grab4, grab32)[0]:.2f})")
cat70 = rel(["oakink_official_category", "oakink_category_rest"], both_arch=True)  # same seed rule as Table 1
dof, sparse, grasp = rel(["pc_oakink_dofdamage"]), rel(["pc_oakink_sparse"]), rel(["grab_grasp_s32"])
check("DOF damage", f"+{dof.mean():.2f} %, against +{cat70.mean():.2f} % undamaged (p = {welch(dof, cat70)[0]:.2f})")
check("grid thinning", f"returned +{sparse.mean():.2f} % (p = {welch(sparse, cat70)[0]:.4f}")
check("grasp-only GRAB", f"gave +{grasp.mean():.2f} %")
check("grasp-only vs control and unmodified", f"(p = {welch(grasp, c256)[0]:.2f}) or from the unmodified sweep (p = {welch(grasp, grab32)[0]:.2f})")

print("\nTable 2")
# Version-4 pair: two clips per trajectory, no constituent leak (runs/PREREG_oakink_alignment_v4.md).
base, al, mis = rel(["pc_oakink_b64"], 64), rel(["pc_oakink_pair_aligned"], 64), rel(["pc_oakink_pair_misaligned"], 64)
for name, x in (("Unmodified clips", base), ("Two clips joined, label describes every window", al),
                ("Two clips joined, label describes the first clip", mis)):
    p = welch(x, c64)[0]
    pstr = sci(p) if p < 1e-4 else f"{p:.2f}"
    check(f"  {name[:40]}", f"| {name} | +{x.mean():.2f} | {x.std(ddof=1):.2f} | {int((x > 0).sum())}/{len(x)} | {pstr} |")
p, d, lo, hi = welch(mis, al)
check("misaligned vs aligned", f"{-d:.1f} percentage points below the aligned version (95 % CI {-hi:.1f} to {-lo:.1f}, p = {sci(p)})")
p, d, lo, hi = welch(mis, base)
check("misaligned vs unmodified", f"{-d:.1f} below the unmodified clips (95 % CI {-hi:.1f} to {-lo:.1f}, p = {sci(p)})")
p, d, lo, hi = welch(al, base)
check("aligned vs unmodified", f"(difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}, p = {p:.4f})")
check("abstract, aligned and misaligned", f"the penalty was +{al.mean():.1f} %; when the label described only the first clip, it fell to +{mis.mean():.1f} %")


def rel_by_seed(d):
    R = json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]
    return {r["seed"]: 100 * r["penalty"] / r["naive"]["mse_target"] for r in R if r["kind"] == "perframe" and r["budget"] == 64}


def gate_fails(name):
    import re
    out = set()
    for line in (ROOT / "runs" / "gates" / name).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(.*)$", line)
        if m and not m.group(7).strip().startswith("ok"):
            out.add(int(m.group(1)))
    return out


fa, fm = gate_fails("cover_oakink_pair_aligned.txt"), gate_fails("cover_oakink_pair_misaligned.txt")
a2 = np.array([v for k, v in rel_by_seed("pc_oakink_pair_aligned").items() if k not in fa])
m2 = np.array([v for k, v in rel_by_seed("pc_oakink_pair_misaligned").items() if k not in fm])
p2, d2, _, _ = welch(m2, a2)
check("gate-passing estimates", f"left the aligned version at +{a2.mean():.2f} % and the misaligned version at +{m2.mean():.2f} %, still {-d2:.1f} points apart (p = {sci(p2)})")
check("gate-passing misaligned vs control", f"against the control at p = {welch(m2, c64)[0]:.3f}")
lens_a = np.load(ROOT / "data" / "bundles" / "oakink_pair_aligned.npz", allow_pickle=True)["lengths"]
lens_m = np.load(ROOT / "data" / "bundles" / "oakink_pair_misaligned.npz", allow_pickle=True)["lengths"]
check("bundle sizes and lengths", f"(means {lens_m.mean():.0f} and {lens_a.mean():.0f} frames, medians {np.median(lens_m):g} and {np.median(lens_a):g}, misaligned first)")
check("bundle trajectory counts", f"Both versions had {len(lens_a)} trajectories" if len(lens_a) == len(lens_m) else "IMPOSSIBLE")

print("\nOakInk2")
whole, orig, rep = rel(["oakink2_scene_primitive"]), rel(["oakink2_primseg_s4"]), rel(["oakink2_primseg_rep"])
check("whole recordings", f"whole recordings returned +{whole.mean():.2f} % over {len(whole)} seeds")
check("primitive segments, both sweeps", f"+{orig.mean():.2f} % over {len(orig)} seeds and\n+{rep.mean():.2f} % over {len(rep)} fresh seeds")
p, d, lo, hi = welch(rep, c256)
check("replication vs control", f"(difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}, p = {p:.4f})")
p, d, lo, hi = welch(rep, whole)
check("replication vs whole", f"(difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}, p = {p:.4f})")
check("replication seeds positive", f"with {int((rep > 0).sum())} of {len(rep)} seeds positive")
rd = rel(["oakink2_primseg_recdisjoint"])
p, d, lo, hi = welch(rd, c256)
check("recording-disjoint vs control",
      f"returned +{rd.mean():.2f} % over {len(rd)} seeds ({int((rd > 0).sum())} positive), above the control "
      f"(difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}, p = {p:.3f})")
p, d, lo, hi = welch(rd, whole)
check("recording-disjoint vs whole", f"above whole recordings (difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}, p = {p:.3f})")
check("recording-disjoint vs repetition", f"It was {rep.mean() - rd.mean():.1f} points below the repetition, a difference that was not distinguishable (p = {welch(rd, rep)[0]:.2f})")
check("recording-disjoint in the abstract", f"the penalty was +{rd.mean():.1f} % and still above the control")

print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
