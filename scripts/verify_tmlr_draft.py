"""Recompute every number in docs/tmlr/PAPER_TMLR.md from stored results and bundles, and require the exact string in
the text. Reads finished sweeps only. Sweeps still running (the two 40-seed replications of the dagger rows, and the
volume-matched GRAB sweep until it has 40 seeds) are never read here.

Statistic: per-seed penalty / naive mse_target x 100, perframe prior. Tests: two-sided Welch on per-seed values.
Seed rule: on sweeps that also trained the modular bank, only seeds where both finished (as in every earlier draft).
Split-derived quantities come from runs/axis_diagnostics_tmlr.json (scripts/axis_diagnostics.py), volume and marginal
loss from runs/marginal_loss_volume.json (scripts/marginal_loss_and_volume.py), the joined-clip mechanism numbers from
runs/constituent_leak_pair.json (scripts/check_constituent_leak.py), and descriptive counts from the bundles and
build reports themselves.
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
DRAFT = (ROOT / "docs" / "tmlr" / "PAPER_TMLR.md").read_text(encoding="utf-8")
FLAT = " ".join(DRAFT.split())
fails = 0
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]


def check(label, s):
    global fails
    ok = " ".join(s.split()) in FLAT
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label:40s} {s}")


def load(d):
    return json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]


def rows(dirs, budget=256, both=False, kind="perframe"):
    out = []
    for d in dirs:
        by = {}
        for r in load(d):
            if r["budget"] == budget:
                by.setdefault(r["seed"], {})[r["kind"]] = r
        out += [v[kind] for v in by.values() if kind in v and (not both or "modular" in v)]
    return out


def rel(dirs, budget=256, both=False):
    return np.array([100 * r["penalty"] / r["naive"]["mse_target"] for r in rows(dirs, budget, both)])


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
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{p / 10 ** e:.1f} × 10{sup}"


def pfmt(p):
    return "< 10⁻¹²" if p < 1e-12 else (sci(p) if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.2f}"))


def mse(dirs, arm, budget=256):
    return np.mean([r[arm]["mse_target"] for r in rows(dirs, budget)])


def by_seed(d, budget=64):
    return {r["seed"]: 100 * r["penalty"] / r["naive"]["mse_target"]
            for r in load(d) if r["kind"] == "perframe" and r["budget"] == budget}


def gate_fails(name):
    out = set()
    for line in (ROOT / "runs" / "gates" / name).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(.*)$", line)
        if m and not m.group(7).strip().startswith("ok"):
            out.add(int(m.group(1)))
    return out


def bundle(name):
    return np.load(ROOT / "data" / "bundles" / f"{name}.npz", allow_pickle=True)


def meta(name):
    m = bundle(name)["meta"].item()
    return ast.literal_eval(m) if isinstance(m, str) else m


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %"


# ------------------------------------------------------------------------------------------ descriptive counts
print("Datasets and settings")
for name, n_str, med_str in (("oakink", "770 trajectories at 30 fps, median 72 frames", None),
                             ("grab", "1,048 trajectories at 30 fps, median 254 frames", None),
                             ("oakink2", "609 trajectories, motion capture subsampled by 4 to 30 fps, median 613 frames", None),
                             ("taco_action_tool", "2,317 trajectories at 30 fps, median 148 frames", None)):
    z = bundle(name)
    L = z["lengths"]
    fps = float(z["fps"])
    s = n_str.split(" trajectories")[0].replace(",", "")
    assert int(s) == len(L), (name, len(L))
    assert f"median {np.median(L):.0f} frames" in n_str, (name, np.median(L))
    check(f"  {name} size and median", n_str)
    print(f"      fps {fps}")
gm = meta("grab")
check("GRAB subject archives", "from subject archives s1 to s7 and s10")
assert sorted(gm["subjects"], key=lambda x: int(x[1:])) == ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s10"]
# The OakInk2 loader labels its bundle 7.5 fps, assuming 30 fps motion capture subsampled by 4. The annotation's image
# frame ids step by 4 against the motion-capture frame index, so motion capture is 120 Hz and the bundle is 30 fps
# (Appendix C). Only durations depend on this.
assert float(bundle("oakink2")["fps"]) == 7.5
TRUE_FPS = {"oakink2": 30.0}
for name, sec in (("oakink", 2.4), ("taco_action_tool", 4.9), ("grab", 8.5), ("oakink2", 20)):
    z = bundle(name)
    med_s = np.median(z["lengths"]) / TRUE_FPS.get(name, float(z["fps"]))
    assert round(med_s, 1 if sec < 10 else 0) == sec, (name, med_s)
check("median durations in the Introduction", "median of 2.4 s. A TACO trajectory is described by its authors as one tool-use action and lasts a median of 4.9 s. A GRAB trajectory is a whole interaction of 8.5 s including approach and release, and an OakInk2 recording lasts a median of 20 s")
for name in ("oakink", "grab", "taco_action_tool", "oakink2"):
    assert TRUE_FPS.get(name, float(bundle(name)["fps"])) == 30.0, name
check("window duration", f"A window spans {32 / 30:.2f} s on every dataset")
check("window duration, Introduction", f"A prior trains and scores on windows of {32 / 30:.2f} s")
npar = {r["naive"]["n_parameters"] for r in rows(["taco_action_tool"])}
assert len(npar) == 1
check("prior parameter count", f"({next(iter(npar)):,} parameters)")
syn = bundle("synthetic_big")
sm = meta("synthetic_big")
Ls = syn["lengths"]
check("synthetic size", f"A synthetic dataset of {len(Ls):,} trajectories at 30 fps (mean {Ls.mean():.0f} frames)")
assert sm["n_primitives"] == 9 and sm["noise_rank"] == 6 and sm["noise_deg"] == 7.0 and sm["sensor_noise_deg"] == 0.4
pairs = set()
for lab in syn["labels"]:
    seq = str(lab).split("->")
    pairs |= set(zip(seq, seq[1:]))
check("synthetic ordered pairs", f"all {len(pairs)} ordered pairs of poses occur")
check("synthetic noise", "perturbed by a draw of scale 7° from a fixed six-dimensional subspace")
check("synthetic sensor noise", "independent noise of 0.4° is added to every frame")
from caredex.data.synthetic import SyntheticSource  # noqa: E402
import inspect  # noqa: E402
src = inspect.getsource(SyntheticSource.__init__)
assert "segments_per_traj: tuple[int, int] = (3, 7)" in src
check("synthetic segments", "A trajectory consists of three to seven segments")
assert meta("synth_hard")["transition_via_deg"] == 18.0
check("planted synthetic scale", "drawn at a scale of 18° from the same subspace")
pm = meta("grab_planted")
print("  planted GRAB rule:", pm["planted_rule"], "rms", pm["planted_rms_norm"])
pp = json.loads((ROOT / "runs" / "planted_cells.json").read_text(encoding="utf-8"))
check("planted GRAB cells", f"{pp['cells_with_offset']} of {pp['cells']} cells receive a non-zero offset")
TB = json.loads((ROOT / "runs" / "taco_build.json").read_text(encoding="utf-8"))
v = TB["vocabulary"]
check("TACO vocabulary", f"the {v['triplets']} triplets of the released data combine {v['actions']} actions, {v['tools']} tools and {v['objects']} objects")
check("TACO triplets in Methods", f"shuffled over the {v['triplets']} triplets")
from experiment_paired_composition import coarsen_labels  # noqa: E402
from experiment_data_efficiency import transitions_of  # noqa: E402
tl = [str(x) for x in bundle("taco_action_tool")["labels"]]
tc = [transitions_of(c)[0] for c in coarsen_labels(tl, "oakink2_scene_verb")]
occ = len(set(tc)) / (len({a for a, _ in tc}) * len({b for _, b in tc}))
check("TACO grid occupancy", f"TACO's action-by-tool grid is {100 * occ:.0f} % occupied")
check("TACO mean pose", f"{pct(TB['pose_mean']['clip_fraction_flat'], 2)} of values fall outside a joint limit over all 2,317 sequences, and with it {pct(TB['pose_mean']['clip_fraction_add'], 2)}")
check("TACO DOF health", f"number {TB['dof_health']['n_unusable']} of 27 on TACO")
ol = [str(x) for x in bundle("oakink")["labels"]]
sl = [str(x) for x in bundle("oakink_sparse")["labels"]]


def grid(labels):
    c = [transitions_of(x)[0] for x in coarsen_labels(labels, "oakink_category")]
    cats, ints = {a for a, _ in c}, {b for _, b in c}
    return len(set(c)) / (len(cats) * len(ints)), len(cats)


o_occ, o_cat = grid(ol)
s_occ, s_cat = grid(sl)
check("sparse grid", f"reduce occupancy from {100 * o_occ:.0f} % to {100 * s_occ:.0f} % (which also removed {o_cat - s_cat} categories and {len(ol) - len(sl)} of {len(ol)} trajectories)")
gg = bundle("grab_grasp")
check("GRAB contact segment", f"(8.5 s to {np.median(gg['lengths']) / float(gg['fps']):.1f} s; {len(bundle('grab')['lengths']) - len(gg['lengths'])} of 1,048 trajectories dropped")
B4 = json.loads((ROOT / "runs" / "build_oakink_alignment_v4.json").read_text(encoding="utf-8"))
assert B4["aligned"]["n_traj"] == B4["misaligned"]["n_traj"] == 334
assert B4["aligned"]["categories"] == B4["misaligned"]["categories"] == 33

# ------------------------------------------------------------------------------------------ 4.1 and Table 1
print("\nCalibration")
c256, c64, hard = rel(["pc_easy_rerun"]), rel(["pc_easy_rerun"], 64), rel(["pc_hard_rerun"])
check("control 256", f"returns +{c256.mean():.2f} % of the naive error over {len(c256)} seeds at a budget of 256 (SD {c256.std(ddof=1):.2f})")
check("control 64", f"+{c64.mean():.2f} % at a budget of 64 (SD {c64.std(ddof=1):.2f}, {len(c64)} seeds)")
check("abstract control", f"calibrated with a synthetic dataset whose true penalty is zero (+{c256.mean():.1f} % of the naive error)")
check("discussion control", f"The design returns +{c256.mean():.1f} % when the true value is zero")
p, d, lo, hi = welch(hard, c256)
check("planted synthetic", f"returns +{hard.mean():.2f} % over {len(hard)} seeds, +{d:.1f} points above the zero-truth control (95 % CI {lo:.1f} to {hi:.1f}, p = {sci(p)})")
planted, grab32 = rel(["grab_planted_s32"]), rel(["grab_shape_v2"], both=True)
p2, d2, _, _ = welch(planted, c256)
check("planted GRAB", f"returns +{planted.mean():.2f} % over {len(planted)} seeds at a stride of 32, against +{grab32.mean():.2f} % for the same sweep without it (p = {sci(welch(planted, grab32)[0])}) and +{d2:.1f} points above the zero-truth control (p = {sci(p2)})")

cat = rel(["oakink_official_category", "oakink_category_rest"], both=True)
shc, g4, shg = rel(["sham_oakink_category"]), rel(["grab_shape_s4"]), rel(["sham_grab_shape"])
taco, sht = rel(["taco_action_tool"]), rel(["sham_taco_action_tool"])


def t1(name, x, ref_name, ref):
    p, d, lo, hi = welch(x, ref)
    return (f"| {name} | {x.mean():+.2f} | {x.std(ddof=1):.2f} | {int((x > 0).sum())}/{len(x)} | {ref_name} | "
            f"{d:+.2f} ({lo:.2f} to {hi:.2f}) | {pfmt(p)} |")


check("T1 zero-truth", f"| Zero-truth synthetic (n = {len(c256)}) | +{c256.mean():.2f} | {c256.std(ddof=1):.2f} | {int((c256 > 0).sum())}/{len(c256)} | zero | | |")
check("T1 planted", t1(f"Planted synthetic (n = {len(hard)})", hard, "zero-truth", c256))
for name, x, real in (("OakInk-Image category × intent, permuted", shc, cat),
                      ("GRAB shape × fine intent, permuted", shg, g4),
                      ("TACO action × tool, permuted cells", sht, taco)):
    check(f"  T1 {name[:22]} vs control", t1(name, x, "zero-truth", c256))
    check(f"  T1 {name[:22]} vs real", t1(name, x, "real grid", real))
check("T1 caption", f"The real grid of OakInk-Image category by intent has {len(cat)} seeds.")
# (abstract permuted-grid sentence removed in the 2026-10-01 style rewrite; Table 1 rows are checked)
check("real above permuted", f"the real grids lie {-welch(shc, cat)[1]:.1f} and {-welch(sht, taco)[1]:.1f} points above their permuted grids")

DIAG = {a["axis"]: a for a in json.loads((ROOT / "runs" / "axis_diagnostics_tmlr.json").read_text(encoding="utf-8"))["axes"]}


def expo(k):
    return np.mean([r["exposure"] for r in DIAG[k]["per_seed"]])


SG = json.loads((ROOT / "runs" / "gates" / "sham_grid_cover_grab_shape_full.json").read_text(encoding="utf-8"))["summary"]
check("GRAB permuted exposure", f"more exposed to held-out cells ({SG['sham']['mean_exposure']:.2f} against {SG['real']['mean_exposure']:.2f})")
assert f"{SG['real']['mean_exposure']:.2f}" == f"{expo('shape x fine intent'):.2f}"
check("TACO sham informed worse", f"(mean error {mse(['sham_taco_action_tool'], 'informed'):.4f} against {mse(['sham_taco_action_tool'], 'naive'):.4f})")
assert c256.mean() > sht.mean()  # the declared rule: the larger of the two references is the synthetic control

# ------------------------------------------------------------------------------------------ Table 2
print("\nTable 2")
MV = json.loads((ROOT / "runs" / "marginal_loss_volume.json").read_text(encoding="utf-8"))
VOL = {a["axis"]: a for a in MV["axes"]}
TABLE2 = [
    ("OakInk-Image", "functional class × intent ‡ W", "functional class x intent", ["oakink_class_rep"], False),
    ("OakInk-Image", "category × intent", "category x intent", ["oakink_official_category", "oakink_category_rest"], True),
    ("OakInk-Image", "affordance × intent", "affordance x intent", ["oakink_official_attr", "oakink_attr_more"], True),
    ("OakInk-Image", "category × subject W", "category x subject", ["oakink_category_subject"], False),
    ("TACO", "action × tool W", "action x tool", ["taco_action_tool"], False),
    ("GRAB", "shape × fine intent W", "shape x fine intent", ["grab_shape_s4"], False),
    ("GRAB", "shape × intent class W", "shape x intent class", ["grab_shapeclass"], True),
    ("OakInk2", "scene × verb W", "scene x verb", ["oakink2_scene_verb"], True),
    ("OakInk2", "scene × primitive", "scene x primitive", ["oakink2_scene_primitive"], False),
    ("OakInk2", "annotated transitions ‡ W", "annotated transitions", ["oakink2_transitions_rep"], False),
]
# W = settings and readings written before training; the file that did so, per row.
WRITTEN = {"category x subject": "PREREG_oakink_category_subject.md", "action x tool": "PREREG_taco_prediction.md",
           "shape x fine intent": "PREREG_stride4_reruns.md", "shape x intent class": "PREREG_grab_shapeclass.md",
           "scene x verb": "PREREG_oakink2_scene_verb.md",
           "annotated transitions": "PREREG_oakink2_transitions_replication.md",
           "functional class x intent": "PREREG_oakink_class_replication.md"}
for f in WRITTEN.values():
    assert (ROOT / "runs" / f).exists(), f
upper, covfail = [], {}
for ds, shown, key, dirs, both in TABLE2:
    assert ("W" in shown.split()) == (key in WRITTEN), key
    R = rows(dirs, both=both)
    x = np.array([100 * r["penalty"] / r["naive"]["mse_target"] for r in R])
    n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    p, d, lo, hi = welch(x, c256)
    ps = DIAG[key]["per_seed"]
    assert DIAG[key]["n_seeds"] == n, (key, DIAG[key]["n_seeds"], n)
    covfail[key] = (sum(not (r["covered"] / r["n_held"] > 0.8 and r["examples_per_covered_cell"] >= 3.0) for r in ps), n)
    naive = np.mean([r["naive"]["mse_target"] for r in R])
    win = VOL[key]["expected_windows_per_arm"]
    check(f"  {key}", f"| {ds} | {shown} | {win:,.0f} | {ps[0]['n_held']} | {n} | +{x.mean():.1f} | {x.std(ddof=1):.1f} | "
                      f"{x.mean() - h:+.1f} to {x.mean() + h:+.1f} | {int((x > 0).sum())}/{n} | {naive:.4f} | "
                      f"{expo(key):.2f} | {pfmt(p)} |")
    if x.mean() < 5:
        upper.append(hi)
    if "‡" in shown:
        assert n == 40
check("upper bound on null axes", f"reaches at most +{max(upper):.1f} points")
held = sorted({DIAG[k]["per_seed"][0]["n_held"] for k in DIAG})
check("held cells range", f"For each seed, the protocol holds out {NUM[held[0]]} to {NUM[held[-1]]} cells")
sh = [np.mean([r["shared_training_fraction"] for r in DIAG[k]["per_seed"]]) for k in DIAG]
check("shared training", f"the arms share {100 * min(sh):.0f} % to {100 * max(sh):.0f} % of their training trajectories")
el = [DIAG[k]["per_seed"][0]["candidate_pool_size"] for k in DIAG]
check("eligible cells", f"and {min(el)} to {max(el)} cells per axis are eligible")
SR = json.loads((ROOT / "runs" / "axis_diagnostics_tmlr.json").read_text(encoding="utf-8"))["split_reproduction"]
assert SR["n_mismatches"] == 0 and SR["seeds_with_no_verification_source"] == 0
check("split re-derivation", f"with no mismatch in {SR['n_checks']:,} checks")
a, c5, f3, tcf = (covfail[k] for k in ("affordance x intent", "category x intent", "functional class x intent", "action x tool"))
check("coverage recount", f"Coverage fails on {a[0]} of {a[1]} seeds of affordance by intent, {c5[0]} of {c5[1]} of category by intent, {tcf[0]} of {tcf[1]} on TACO and at most 3 of 40 elsewhere")
assert max(v[0] for k, v in covfail.items() if v[1] == 40 and k != "action x tool") <= 3
tr4 = rel(["oakink2_transitions_rep"])
tr_orig, fc_orig = rel(["oakink2_transitions_s4"]), rel(["oakink_class_v1"], both=True)
assert len(tr_orig) == len(fc_orig) == 12
check("original 12-seed sweeps in the caption", f"the original 12-seed sweeps returned +{fc_orig.mean():.1f} % (functional class by intent) and +{tr_orig.mean():.1f} % (annotated transitions)")
gb64 = rel(["grab_shape_b64"], 64)
assert len(gb64) == 40
_p, _d, _lo, _hi = welch(gb64, c64)
check("GRAB at matched volume", f"returns +{gb64.mean():.2f} % (SD {gb64.std(ddof=1):.2f}, {len(gb64)} seeds), not distinguishable from the zero-truth control at that budget (difference +{_d:.1f}, 95 % CI {_lo:.1f} to {_hi:.1f})")
assert _p > 0.05
check("transitions value, Section 4.2", f"OakInk2's transitions axis carries +{tr4.mean():.1f} % at the largest volume in the table")
wv = {k: VOL[k]["expected_windows_per_arm"] for k in VOL}
check("volume ordering", f"(4,591 and 8,144, against 17,604 on GRAB and 57,919 on OakInk2)")
assert (f"{wv['category x intent']:,.0f}", f"{wv['action x tool']:,.0f}", f"{wv['shape x fine intent']:,.0f}",
        f"{wv['scene x primitive']:,.0f}") == ("4,591", "8,144", "17,604", "57,919")
check("GRAB volume-matched budget", f"GRAB at a budget of 64, about {round(wv['shape x fine intent'] / 4, -2):,.0f} windows per arm and so matched to OakInk-Image")
check("unmodified clips volume at 64", f"which have about {round(wv['category x intent'] / 4, -1):,.0f} windows per arm")
ob64, w256, w64 = rel(["pc_oakink_b64"], 64), rel(["oakink2_scene_primitive"]), rel(["oakink2_scene_primitive_b64"], 64)
check("volume signs", f"the penalty is +{w64.mean():.2f} % at a budget of 64 against +{w256.mean():.2f} % at 256 (Appendix B). Volume does not act in one direction, however. On OakInk-Image a budget of 64 gives +{ob64.mean():.2f} % against +{cat.mean():.2f} % at 256")


def absdiff(k):
    v = DIAG[k]["absolute_mse_target"]
    return v["naive_mean"] - v["informed_mean"]


check("absolute penalties", f"exceeds the informed arm's by {absdiff('category x intent'):.4f} on OakInk-Image category by intent and by {absdiff('action x tool'):.4f} on TACO, against {absdiff('shape x fine intent'):.4f} on GRAB shape by fine intent and {absdiff('scene x primitive'):.4f} on OakInk2 scene by primitive")
check("exposure TACO and GRAB", f"most exposed on TACO ({expo('action x tool'):.2f}) and GRAB ({expo('shape x fine intent'):.2f} and {expo('shape x intent class'):.2f})")
assert expo("action x tool") == max(expo(k) for k in DIAG)
ml = {k: VOL[k].get("marginal_loss_mean") for k in VOL}


def rng(keys):
    vals = sorted(round(100 * ml[k]) for k in keys)
    return f"{vals[0]} %" if vals[0] == vals[-1] else f"{vals[0]} % to {vals[-1]} %"


oi = ["functional class x intent", "category x intent", "affordance x intent", "category x subject"]
check("marginal loss", f"the naive arm loses {rng(oi)} of a held factor's trajectories on OakInk-Image, {rng(['action x tool'])} on TACO, {rng(['shape x fine intent', 'shape x intent class'])} on GRAB and {rng(['scene x verb', 'scene x primitive'])} on OakInk2")
rs = sorted(VOL[k]["loss_penalty_r"] for k in oi)
check("marginal loss correlation", f"on OakInk-Image (r = {rs[0]:+.2f} to {rs[-1]:+.2f}) and on TACO (r = {VOL['action x tool']['loss_penalty_r']:+.2f}, p = {VOL['action x tool']['loss_penalty_p']:.2f})")
dof, sparse = rel(["pc_oakink_dofdamage"]), rel(["pc_oakink_sparse"])
check("appendix B DOF/sparse", f"The penalty was +{dof.mean():.2f} % and +{sparse.mean():.2f} % respectively, against +{cat.mean():.2f} % unmodified (p = {welch(dof, cat)[0]:.2f} and p = {welch(sparse, cat)[0]:.4f})")
gc = rel(["grab_shapeclass"], both=True)
check("GRAB intent class vs control", f"(+{gc.mean():.1f} %, p = {welch(gc, c256)[0]:.2f} against the control)")
assert stats.ttest_1samp(gc, 0).pvalue < 0.05, "claim: a test against zero would call it a penalty"

# ------------------------------------------------------------------------------------------ Tables 3 and 4
print("\nTables 3 and 4")
base, al, mis = rel(["pc_oakink_b64"], 64), rel(["pc_oakink_pair_aligned"], 64), rel(["pc_oakink_pair_misaligned"], 64)
for name, x in (("Unmodified clips", base), ("Aligned: label describes both clips", al), ("Misaligned: label describes the first clip", mis)):
    check(f"  T3 {name}", f"| {name} | +{x.mean():.2f} | {x.std(ddof=1):.2f} | {int((x > 0).sum())}/{len(x)} | {pfmt(welch(x, c64)[0])} |")
check("T3 caption control", f"budget (+{c64.mean():.2f} %, {len(c64)} seeds)")
p, d, lo, hi = welch(mis, al)
check("mis vs al", f"is {-d:.1f} points below the aligned version (95 % CI {-hi:.1f} to {-lo:.1f}, p = {sci(p)})")
p, d, lo, hi = welch(mis, base)
check("mis vs base", f"and {-d:.1f} below the unmodified clips (95 % CI {-hi:.1f} to {-lo:.1f}, p = {sci(p)})")
p, d, lo, hi = welch(mis, c64)
assert p > 0.05
check("mis vs control, results", f"It lies at most {hi:.1f} points above the control (difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}; Table 3)")
check("mis vs control, abstract", f"reduces the penalty from +{al.mean():.1f} % to +{mis.mean():.1f} %, at most {hi:.1f} points above the control")
p, d, lo, hi = welch(al, base)
check("al vs base", f"(difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}, p = {p:.4f})")
check("abstract al/mis", f"reduces the penalty from +{al.mean():.1f} % to +{mis.mean():.1f} %")
check("discussion aligned", f"reduces a penalty of\n+{al.mean():.1f} % to near the control")
fa, fm = gate_fails("cover_oakink_pair_aligned.txt"), gate_fails("cover_oakink_pair_misaligned.txt")
a2 = np.array([v for k, v in by_seed("pc_oakink_pair_aligned").items() if k not in fa])
m2 = np.array([v for k, v in by_seed("pc_oakink_pair_misaligned").items() if k not in fm])
p2, d2, _, _ = welch(m2, a2)
check("gate-passing", f"coverage gate ({len(fa)} aligned, {len(fm)} misaligned) the two versions read +{a2.mean():.2f} % and +{m2.mean():.2f} %, still {-d2:.1f} points apart (p = {sci(p2)}), with the misaligned version against the control at p = {welch(m2, c64)[0]:.3f}")
la = bundle("oakink_pair_aligned")["lengths"]
lm = bundle("oakink_pair_misaligned")["lengths"]
check("bundle sizes", f"Both versions have {len(la)} trajectories" if len(la) == len(lm) else "IMPOSSIBLE")
check("bundle lengths", f"mean lengths of {la.mean():.0f} and {lm.mean():.0f} frames")
ex = MV["extra"]
wa, wm = ex["oakink_pair"]["expected_windows_per_arm"], ex["oakink_pair_misaligned"]["expected_windows_per_arm"]
check("pair volume", f"and {wa:,.0f} and {wm:,.0f} expected training windows per arm")
CL = {Path(e["bundle"]).name: e["summary_mean_over_seeds"] for e in
      json.loads((ROOT / "runs" / "constituent_leak_pair.json").read_text(encoding="utf-8"))}
A_, M_ = CL["oakink_pair_aligned.npz"], CL["oakink_pair_misaligned.npz"]
check("naive contamination", f"{100 * M_['naive.c_train_clips_in_held_cell']:.1f} % of the naive arm's training clips belong to held-out cells")
check("naive-seen targets", f"{100 * M_['naive.a_target_traj_exposed']:.1f} % of target trajectories contain a clip whose object and intent the naive arm has seen")
check("informed dilution", f"falls from {100 * A_['informed.c_train_clips_in_held_cell']:.1f} % in the aligned version to {100 * M_['informed.c_train_clips_in_held_cell']:.1f} %")
assert A_["naive.c_train_clips_in_held_cell"] == 0 and A_["naive.a_target_traj_exposed"] == 0 and A_["d_target_clips_in_held_cell"] == 1
check("purity", f"In the aligned version both\nnaive-arm figures are 0 %. In the misaligned version {100 * M_['d_target_clips_in_held_cell']:.0f} % of target clips belong to a held-out cell, against 100 % in\nthe aligned version")

W = {}
for v_ in ("aligned", "misaligned"):
    J = json.loads((ROOT / "runs" / f"pc_oakink_pair_{v_}_wc" / "window_classes.json").read_text(encoding="utf-8"))["runs"]
    seeds = sorted({int(re.match(r"s(\d+)_", k).group(1)) for k in J})
    assert len(seeds) == 40
    W[v_] = {c: np.array([100 * (J[f"s{s}_perframe_b64_naive"][c]["mse"] - J[f"s{s}_perframe_b64_informed"][c]["mse"])
                          / J[f"s{s}_perframe_b64_naive"][c]["mse"] for s in seeds])
             for c in ("first", "second", "straddle")}
A, M = W["aligned"], W["misaligned"]
ta, tm = rel(["pc_oakink_pair_aligned_wc"], 64), rel(["pc_oakink_pair_misaligned_wc"], 64)
check("wc reproduction", f"reproduced the totals of Table 3 (+{ta.mean():.2f} % and +{tm.mean():.2f} %)")
for name, D in (("Aligned", A), ("Misaligned", M)):
    check(f"  T4 {name}", f"| {name} | +{D['first'].mean():.2f} (SD {D['first'].std(ddof=1):.2f}) | +{D['straddle'].mean():.2f} (SD {D['straddle'].std(ddof=1):.2f}) | +{D['second'].mean():.2f} (SD {D['second'].std(ddof=1):.2f}) |")
check("A1 vs A2", f"(paired difference +{(A['first'] - A['second']).mean():.1f}, p = {stats.ttest_rel(A['first'], A['second']).pvalue:.3f})")
p, d, lo, hi = welch(M["first"], A["first"])
loss = (A["first"].mean() - M["first"].mean()) / (A["first"].mean() - c64.mean())
check("M1 vs A1", f"read {-d:.1f} points below the aligned version's (95 % CI {-hi:.1f} to {-lo:.1f}, p = {sci(p)}), a loss of {100 * loss:.0f} % of their penalty above the control")
_rng = np.random.default_rng(0)
_boot = []
for _ in range(100000):
    a_, m_, c_ = (x[_rng.integers(0, len(x), len(x))].mean() for x in (A["first"], M["first"], c64))
    _boot.append((a_ - m_) / (a_ - c_))
blo, bhi = np.percentile(_boot, [2.5, 97.5]); print(f"      bootstrap bounds {100 * blo:.2f} {100 * bhi:.2f}")
check("82 % in the abstract", f"lose {100 * loss:.0f} % of their penalty above the\ncontrol (95 % CI {100 * blo:.0f} % to {100 * bhi:.0f} %)")
check("82 % bootstrap CI, results", f"(bootstrap 95 % CI {100 * blo:.0f} % to {100 * bhi:.0f} %)")
p, d, lo, hi = welch(M["first"], c64)
check("M1 vs control", f"They remain +{d:.1f} points above the control (95 % CI {lo:.1f} to {hi:.1f}, p = {p:.3f})")
check("M2 vs M1", f"are a further {(M['first'] - M['second']).mean():.1f} points lower (paired p = {stats.ttest_rel(M['second'], M['first']).pvalue:.4f})")

# ------------------------------------------------------------------------------------------ 4.4 TACO
print("\nTACO")
fr = TB["frames"]
check("TACO durations", f"last a median of {fr['median_s']:.2f} s ({fr['min_s']:.2f} to {fr['max_s']:.2f} s; {100 * fr['fraction_longer_than_10s']:.1f} % over 10 s)")
p, d, lo, hi = welch(taco, c256)
check("TACO confirmed", f"+{taco.mean():.2f} % over {len(taco)} seeds, {d:.1f} points above the control (95 % CI {lo:.1f} to {hi:.1f}, p = {sci(p)})")
tfail = {r["seed"] for r in DIAG["action x tool"]["per_seed"]
         if not (r["covered"] / r["n_held"] > 0.8 and r["examples_per_covered_cell"] >= 3.0)}
tk = np.array([v for k, v in by_seed("taco_action_tool", 256).items() if k not in tfail])
check("TACO gate-passing", f"and +{tk.mean():.2f} % without the {len(tfail)} seeds that fail the coverage gate")
check("TACO above its permuted grid", f"It is also {-welch(sht, taco)[1]:.1f} points above TACO's permuted grid")
q1, q3 = np.percentile(taco, [25, 75])
check("TACO per-seed spread", f"They range from {taco.min():.2f} % to +{taco.max():.2f} %, with an interquartile range of +{q1:.2f} % to +{q3:.2f} %; {int((taco <= c256.mean()).sum())} of {len(taco)} fall at or below the mean of the zero-truth control, and {int((taco > 20).sum())} exceed +20 %")
assert "most of TACO's SD" not in FLAT, "council 2: the SD-attribution sentence was deleted"
check("TACO range, discussion", f"any value from {taco.min():.1f} % to +{taco.max():.1f} %")
check("TACO, abstract", f"appears (+{taco.mean():.1f} %)")
pr = sorted([welch(taco, c256)[0], welch(shc, cat)[0], welch(sht, taco)[0], welch(mis, al)[0]])
assert max(pr) * 4 < 1e-5
check("Bonferroni", "All four remain below p = 10⁻⁵ after a Bonferroni correction for four tests")
RE = {(c["a"], c["b"]): c["p"] for c in
      json.loads((ROOT / "runs" / "cell_random_effects.json").read_text(encoding="utf-8"))["contrasts"]}
re4 = [RE[("action x tool", "control_256")], RE[("permuted category x intent", "category x intent")],
       RE[("permuted action x tool", "action x tool")], RE[("pair misaligned", "pair aligned")]]
assert max(re4) < 0.002, re4
check("random effects", "refitting the four with a random effect for each held-out cell leaves\nall four below p = 0.002")

# ------------------------------------------------------------------------------------------ OakInk2 and appendices
print("\nOakInk2 and appendices")
MC = json.loads((ROOT / "runs" / "motion_contamination_oakink2.json").read_text(encoding="utf-8"))
LC = json.loads((ROOT / "runs" / "label_coverage_oakink2.json").read_text(encoding="utf-8"))
rec, frm, pur = (100 * MC[k]["mean"] for k in ("naive_motion_contaminated_recordings",
                                              "naive_motion_contaminated_frames", "target_frame_purity_first_span"))
check("OakInk2 multi-primitive recordings", f"with {LC['recordings'] - LC['recordings_with_one_primitive']} of the {LC['recordings']} used here containing several annotated primitives")
check("OakInk2 label coverage", f"primitive covers a median of {100 * LC['first_primitive_share']['median']:.0f} % of a recording's frames, {LC['recordings_with_one_primitive']} of {LC['recordings']} recordings hold a single primitive")
check("OakInk2 contamination", f"over the sweep's {MC['seeds']} splits {rec:.1f} % of the naive arm's training recordings contain a held-out scene and primitive after their first primitive ({frm:.1f} % of its training frames), while {pur:.0f} % of target frames")
seg, b896 = rel(["oakink2_primseg_rep"]), rel(["oakink2_primseg_b896"], 896)
p, d, _, _ = welch(w64, w256)
check("OakInk2 budget", f"returned +{w256.mean():.2f} % at a budget of 256 and +{w64.mean():.2f} % at a budget of 64 (difference +{d:.1f}, p = {p:.4f}; the latter above its control, p = {welch(w64, c64)[0]:.3f})")
check("segments at 256", f"returned +{seg.mean():.2f} % at a budget of 256")
check("segments vs whole@64", f"did not differ from them (p = {welch(seg, w64)[0]:.2f}), and +{b896.mean():.2f} % at a budget of 896")
_seg_frames = int(round(float(np.mean(bundle("oakink2_primseg")["lengths"])) * 256))
_whole_frames = int(round(float(np.mean(bundle("oakink2")["lengths"])) * 64))
check("frames per arm, segments@256 vs whole@64", f"({_seg_frames:,} against {_whole_frames:,} frames per arm)")
p, d, lo, hi = welch(b896, w256)
check("segments@896 vs whole@256", f"(difference +{d:.1f}, 95 % CI {lo:.1f} to {hi:.1f}, p = {p:.3f})")
check("stride 32 vs 4", f"does not differ from its stride-4 value in Table 2 (p = {welch(g4, grab32)[0]:.2f})")
grasp = rel(["grab_grasp_s32"])
check("grasp-only, appendix", f"gave +{grasp.mean():.2f} %, not different from the control (p = {welch(grasp, c256)[0]:.2f}) or from the unmodified sweep (p = {welch(grasp, grab32)[0]:.2f})")
check("grasp-only, discussion", f"(+{grasp.mean():.2f} %, p = {welch(grasp, c256)[0]:.2f}; Appendix B)")
check("grasp volume", f"about {round(ex['grab_contact_segment']['expected_windows_per_arm'], -2):,.0f} windows per arm")
tr16 = rel(["oakink2_paired_v2"], both=True)
check("transitions at stride 16", f"axis from +{tr16.mean():.2f} % at a stride of 16")

print("\nAppendix D")
rm = rel(["ratematch_a", "ratematch_b", "ratematch_c", "ratematch_d"])
check("second set of models", f"| {len(rm)} | +{rm.mean():.2f} |")
pre = [rel([d]) for d in ("oakink_n70", "oakink_n70_v2", "oakink_coarse", "oakink_noleak")]
check("object-id prefix", f"| {min(len(x) for x in pre)} to {max(len(x) for x in pre)} | +{min(x.mean() for x in pre):.2f} to +{max(x.mean() for x in pre):.2f} |")
f256, f64 = rel(["pc_oakink"]), rel(["pc_oakink"], 64)
check("object x intent", f"| {len(f256)} | +{f256.mean():.2f} and {f64.mean():.2f} |")
d1, d2_ = rel(["dexycb_paired"]), rel(["dexycb_paired_v2"])
check("DexYCB", f"| {len(d1)} | +{d1.mean():.2f} and +{d2_.mean():.2f} |")
cf, cp = rel(["contact_full"]), rel(["contact_pose"])
check("GRAB contact channels", f"| {len(cf)} | {cf.mean():.2f} and +{cp.mean():.2f} |")
g0 = rel(["grab_shape"])
check("GRAB first run", f"| {len(g0)} | {g0.mean():.2f} |")
gs4 = rel(["grab_grasp_s4"])
check("GRAB contact segment s4", f"| {len(gs4)} | {gs4.mean():.2f} |")
o1, o2 = rel(["oakink2_paired"]), rel(["oakink2_paired_v2"])
check("OakInk2 transitions s16", f"| {len(o1)} | +{o1.mean():.2f} and +{o2.mean():.2f} |")
ps4, prd = rel(["oakink2_primseg_s4"]), rel(["oakink2_primseg_recdisjoint"])
check("OakInk2 segments further", f"| {len(ps4)} and {len(prd)} | +{ps4.mean():.2f} and +{prd.mean():.2f} |")
jo = [rel(["pc_oakink_long"]), rel(["pc_oakink_longaligned"]), rel(["pc_oakink_long3_aligned"], 64),
      rel(["pc_oakink_long3_misaligned"], 64), rel(["pc_oakink_long3_misaligned_cm"], 64)]
check("joined clips earlier", f"| {min(len(x) for x in jo)} to {max(len(x) for x in jo)} | +{min(x.mean() for x in jo):.2f} to +{max(x.mean() for x in jo):.2f} |")

print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
