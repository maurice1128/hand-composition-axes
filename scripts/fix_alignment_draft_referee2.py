"""Referee round 1, part 2: the quantities the referee asked for that needed computing, not training.

Sources, all produced by named scripts and re-checked by scripts/verify_alignment_draft.py:
  runs/axis_diagnostics.json            scripts/axis_diagnostics.py  (splits re-derived: 1,760 checks, 0 mismatches)
  runs/label_coverage_oakink2.json      scripts/label_coverage_oakink2.py
  runs/motion_contamination_oakink2.json scripts/motion_contamination_oakink2.py
  runs/<sweep>/results.json

What changes and why
  - Table 1 gains held cells, SD, 95 % CI, seeds positive, the absolute naive error and the informed arm's realised
    exposure (referee M3a, M3b, M11). The penalty is a ratio; its denominator was never reported.
  - The label-level diagnostics of Section 3.4 are 0 / 0 / 1 on every Table 1 axis BY CONSTRUCTION, so they cannot
    operationalise alignment there (referee M2). The draft now says so, and reports what can be measured on a real
    axis instead: motion-level contamination on OakInk2, from its own primitive spans. New Table 3.
  - The Introduction said most scored windows of long recordings show motion other than what the label names. On
    OakInk2 the labelled primitive covers a median 57 % of frames and 259 of 609 recordings hold one primitive. The
    sentence was too strong and is replaced by the measurement.
  - The error is teacher-forced reconstruction, not generation (referee minor 6). Stated in Methods and limitations.
  - The synthetic controls are described fully, including that they are NOT two-factor grids (referee M6).
  - Fine labels, held-cell counts, eligible cells and how much training data the two arms share are stated (M7).
  - The coverage worst case was stale ("at most 12 of 70"); the affordance axis has 130 seeds (referee minor 1).
  - Four experiments are pending and are marked [PENDING]; nothing about their outcome is written.
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "docs" / "PAPER_ALIGNMENT_DRAFT.md"
s = P.read_text(encoding="utf-8")


def rep(old, new):
    global s
    pat = re.compile(r"\s+".join(re.escape(tok) for tok in old.split()))
    hits = pat.findall(s)
    assert len(hits) == 1, f"expected 1, found {len(hits)}: {old[:70]}"
    s = pat.sub(lambda m: new, s, count=1)


def load(d, budget=256, both=False):
    out = []
    by = {}
    for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]:
        if r["budget"] == budget:
            by.setdefault(r["seed"], {})[r["kind"]] = r
    for v in by.values():
        if "perframe" in v and (not both or "modular" in v):
            out.append(v["perframe"])
    return out


def sci(p):
    e = int(np.floor(np.log10(p)))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{p / 10 ** e:.1f} × 10{sup}"


DIAG = {a["axis"]: a for a in json.loads((ROOT / "runs" / "axis_diagnostics.json").read_text(encoding="utf-8"))["axes"]}
ctrl = np.array([100 * r["penalty"] / r["naive"]["mse_target"] for r in load("pc_easy_rerun")])

TABLE1 = [
    ("OakInk-Image", "functional class × intent", "functional class x intent", ["oakink_class_v1"], True),
    ("OakInk-Image", "category × intent", "category x intent", ["oakink_official_category", "oakink_category_rest"], True),
    ("OakInk-Image", "affordance × intent", "affordance x intent", ["oakink_official_attr", "oakink_attr_more"], True),
    ("OakInk-Image", "category × subject", "category x subject", ["oakink_category_subject"], False),
    ("GRAB", "shape × fine intent", "shape x fine intent", ["grab_shape_s4"], False),
    ("GRAB", "shape × intent class", "shape x intent class", ["grab_shapeclass"], True),
    ("OakInk2", "scene × verb", "scene x verb", ["oakink2_scene_verb"], True),
    ("OakInk2", "scene × primitive", "scene x primitive", ["oakink2_scene_primitive"], False),
    ("OakInk2", "annotated transitions", "annotated transitions", ["oakink2_transitions_s4"], False),
]

rows_md, fails = [], {}
for ds, axis, key, dirs, both in TABLE1:
    R = [r for d in dirs for r in load(d, both=both)]
    x = np.array([100 * r["penalty"] / r["naive"]["mse_target"] for r in R])
    naive = float(np.mean([r["naive"]["mse_target"] for r in R]))
    n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_ind(x, ctrl, equal_var=False).pvalue
    pstr = "< 10⁻¹²" if p < 1e-12 else (sci(p) if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.2f}"))
    d = DIAG[key]
    assert d["n_seeds"] == n, (key, d["n_seeds"], n)
    ps = d["per_seed"]
    expo = float(np.mean([r["exposure"] for r in ps]))
    held = ps[0]["n_held"]
    fails[key] = (sum(not (r["covered"] / r["n_held"] > 0.8 and r["examples_per_covered_cell"] >= 3.0) for r in ps), n)
    rows_md.append(f"| {ds} | {axis} | {held} | {n} | +{x.mean():.1f} | {x.std(ddof=1):.1f} | "
                   f"{x.mean() - h:+.1f} to {x.mean() + h:+.1f} | {int((x > 0).sum())}/{n} | {naive:.4f} | "
                   f"{expo:.2f} | {pstr} |")

# --- Table 1 ---------------------------------------------------------------------------------------------------
old_tbl = re.search(r"\| Dataset \| Axis \| Penalty \(% naive\) \| p vs control \| n \|\n\|[-|]+\|\n(?:\|.*\|\n)+", s)
assert old_tbl, "Table 1 not found"
new_tbl = ("| Dataset | Axis | Held cells | n | Penalty (% naive) | SD | 95 % CI | Seeds positive | Naive error | "
           "Informed exposure | p vs control |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
           + "\n".join(rows_md) + "\n")
s = s[:old_tbl.start()] + new_tbl + s[old_tbl.end():]

rep("Table 1. Compositional penalty by dataset and axis at a budget of 256. p is from Welch's test against the "
    "zero-truth control.",
    "Table 1. Compositional penalty by dataset and axis at a budget of 256. Held cells is the number of grid cells "
    "held out per seed. The 95 % CI is of the mean over seeds. Naive error is the naive arm's mean-squared error on "
    "the target windows, the denominator of the penalty. Informed exposure is the share of the informed arm's 256 "
    "trajectories drawn from the held-out cells, re-derived from each seed's split. p is from Welch's test against "
    "the zero-truth control. [PENDING: the two rows with n = 12 are being repeated on 40 fresh seeds each.]")

# --- Results: the denominator and the exposure ------------------------------------------------------------------
cat = DIAG["category x intent"]; gr = DIAG["shape x fine intent"]; sp = DIAG["scene x primitive"]


def absolute(a):
    v = a["absolute_mse_target"]
    return v["naive_mean"] - v["informed_mean"]


def rp(a):
    c = a["exposure_penalty_correlation"]
    return c["r"], c["p"]


r_cat, p_cat = rp(cat); r_aff, p_aff = rp(DIAG["affordance x intent"]); r_sub, p_sub = rp(DIAG["category x subject"])
r_g, p_g = rp(gr); r_s, p_s = rp(sp)
expo = lambda k: float(np.mean([r["exposure"] for r in DIAG[k]["per_seed"]]))

rep("One of the four OakInk-Image axes pairs an object category with the identity of the subject",
    f"Neither the denominator of the penalty nor the informed arm's exposure explains the pattern between datasets. "
    f"The naive error was smaller on the datasets without a penalty (Table 1), so in absolute terms the contrast is "
    f"sharper than in percentages: the naive arm's error exceeded the informed arm's by {absolute(cat):.4f} on "
    f"OakInk-Image category by intent, by {absolute(gr):.4f} on GRAB shape by fine intent and by "
    f"{absolute(sp):.4f} on OakInk2 scene by primitive. The informed arm was most exposed on GRAB "
    f"({expo('shape x fine intent'):.2f} and {expo('shape x intent class'):.2f}), where no penalty was found, and "
    f"OakInk2's three axes shared an exposure of {expo('scene x verb'):.2f} to {expo('scene x primitive'):.2f} while "
    f"one carried a penalty and two did not. Within an axis, per-seed exposure was related to the per-seed penalty "
    f"where a penalty existed (OakInk-Image category by intent r = {r_cat:+.2f}, p = {p_cat:.3f}; affordance by "
    f"intent r = {r_aff:+.2f}, p = {sci(p_aff)}; category by subject r = {r_sub:+.2f}, p = {sci(p_sub)}) and was not "
    f"where none existed (GRAB shape by fine intent r = {r_g:+.2f}, p = {p_g:.2f}; OakInk2 scene by primitive "
    f"r = {r_s:+.2f}, p = {p_s:.2f}), which is the dose-response a real effect predicts. "
    f"One of the four OakInk-Image axes pairs an object category with the identity of the subject")

# --- The mechanism measured on a real axis: new Table 3 ---------------------------------------------------------
MC = json.loads((ROOT / "runs" / "motion_contamination_oakink2.json").read_text(encoding="utf-8"))
LC = json.loads((ROOT / "runs" / "label_coverage_oakink2.json").read_text(encoding="utf-8"))
mc_rec = 100 * MC["naive_motion_contaminated_recordings"]["mean"]
mc_frm = 100 * MC["naive_motion_contaminated_frames"]["mean"]
mc_pur = 100 * MC["target_frame_purity_first_span"]["mean"]
lc_med = 100 * LC["first_primitive_share"]["median"]
one = LC["recordings_with_one_primitive"]; nrec = LC["recordings"]

rep("On OakInk2, whole recordings returned +2.30 % over 40 seeds.",
    f"The same two quantities can be read on a real axis of Table 1, because OakInk2 annotates the frame span of "
    f"every primitive in a recording and its trajectories consist of those spans. On its scene-by-primitive axis a "
    f"recording is labelled by the first primitive of its sequence, which covered a median of {lc_med:.0f} % of the "
    f"recording's frames; {one} of {nrec} recordings held a single primitive. Re-deriving the 40 splits of that "
    f"sweep, no naive-arm recording carried a held-out label, as the split guarantees, but {mc_rec:.1f} % of the "
    f"naive arm's training recordings contained a held-out scene and primitive after their first primitive "
    f"({mc_frm:.1f} % of its training frames), and {mc_pur:.0f} % of the target recordings' frames belonged to "
    f"their held-out primitive (Table 3).\n\n"
    f"Table 3. Whether holding out a label held out the motion. Contamination is the share of the naive arm's "
    f"training trajectories containing held-out motion under another label; purity is the share of what was scored "
    f"that was the held-out composition. OakInk-Image values are over clips and ten seeds, OakInk2 values over "
    f"recordings and frames and 40 seeds. Neither can be read on unmodified OakInk-Image or on GRAB, which have no "
    f"annotation within a recording.\n\n"
    f"| Dataset and version | Contamination | Purity | Penalty (% naive) |\n|---|---|---|---|\n"
    f"| OakInk-Image, label describes every window | 0 % | 100 % | +17.03 |\n"
    f"| OakInk-Image, label describes the first clip | 8.4 % | 61 % | +3.87 |\n"
    f"| OakInk2, whole recordings | {mc_rec:.1f} % | {mc_pur:.0f} % | +2.30 |\n"
    f"| OakInk2, primitive segments | 0 % | 100 % | +8.93 |\n\n"
    f"On OakInk2, whole recordings returned +2.30 % over 40 seeds.")

# --- The Introduction's overstatement --------------------------------------------------------------------------
rep("A prior is trained and scored on short windows, so in the longer recordings most scored windows show motion "
    "other than what the label names.",
    f"A prior is trained and scored on short windows, so in the longer recordings a label names only part of what "
    f"is scored: on OakInk2, where this can be measured, the labelled primitive covers a median of {lc_med:.0f} % of "
    f"a recording's frames (Section 3.4).")

# --- Methods 2.2: what the error is ----------------------------------------------------------------------------
rep("The error throughout is the mean-squared error on normalised pose over all 27 degrees of freedom.",
    "The error throughout is a reconstruction error and not a generation error: each scored window was passed "
    "through the encoder, the latent was taken at its mean, the decoder was additionally given the window's true "
    "first frame, and the mean-squared error between the output and the same window was taken on normalised pose "
    "over all 27 degrees of freedom. The prior's sampling path was not evaluated.")

# --- Methods 2.3: fine labels, shared training data, eligible cells, coverage -----------------------------------
rep("A composition was one cell of a two-factor label grid.",
    "A composition was one cell of a two-factor label grid. A fine label is the most specific label a dataset gives "
    "a trajectory: an object identity and an intent on OakInk-Image, an object and an intent on GRAB, and the "
    "sequence of primitives with its scene, subject and verb on OakInk2. A grid was formed by replacing one side of "
    "the fine label with a coarser group, so that a cell spans several fine labels; on OakInk-Image, for example, "
    "the object identity was replaced by its category, functional class or affordance from the dataset's own "
    "taxonomy. On OakInk2's transitions axis the fine label was kept and a composition was an ordered pair of "
    "consecutive primitives.")

shared = [float(np.mean([r["shared_training_fraction"] for r in DIAG[k]["per_seed"]])) for k in DIAG]
elig = [DIAG[k]["per_seed"][0]["candidate_pool_size"] for k in DIAG]
rep("On sweeps that had also trained a second prior for a comparison not reported here, only seeds on which both "
    "had finished were used.",
    f"On sweeps that had also trained a second prior for a comparison not reported here, only seeds on which both "
    f"had finished were used. The two arms were matched in size and not in content: the informed arm's trajectories "
    f"outside the held-out cells were drawn afresh from the naive pool, so the arms shared {100 * min(shared):.0f} % "
    f"to {100 * max(shared):.0f} % of their training trajectories depending on the size of that pool, which adds "
    f"variance that differs between datasets without biasing the difference. Held-out cells were drawn from those "
    f"spanning enough fine labels to be split, of which there were {min(elig)} to {max(elig)} per axis, so the "
    f"spread over seeds understates the uncertainty over which cells were held.")

worst = max(fails.items(), key=lambda kv: kv[1][0] / kv[1][1])
aff = fails["affordance x intent"]
rep("It failed on a minority of seeds in most sweeps (at most 12 of 70 seeds, and 3 of 12 on functional class by "
    "intent);",
    f"Re-derived for every seed of Table 1, it failed on {aff[0]} of {aff[1]} seeds of affordance by intent, "
    f"{fails['category x intent'][0]} of {fails['category x intent'][1]} of category by intent, "
    f"{fails['functional class x intent'][0]} of {fails['functional class x intent'][1]} of functional class by "
    f"intent, and at most 3 of 40 elsewhere;")

# --- Methods 2.4: the synthetic controls, in full --------------------------------------------------------------
rep("Two synthetic datasets supplied the reference values. In the first, held-out pairings were interpolable by "
    "construction, so the true penalty was zero. In the second, a via-point offset specific to each pairing was "
    "planted.",
    "Two synthetic datasets supplied the reference values. Each held 3,000 trajectories at 30 fps with a mean of "
    "211 frames, built from nine fixed grasp poses: a trajectory visited three to seven of them in sequence, and "
    "each segment between two consecutive poses followed a minimum-jerk profile w(t) = 10t³ - 15t⁴ + 6t⁵. All 81 "
    "ordered pairs of poses occurred. In the first dataset a segment was exactly the convex combination "
    "(1 - w) a + w b of its two end poses, so a pair never seen in training is reproducible from its end poses "
    "alone and the true penalty is zero by construction. In the second, an offset fixed for each ordered pair and "
    "independent of its end poses, of amplitude 18°, was added as sin(πw), vanishing at both ends and largest "
    "midway, so that it can be learned only from trajectories containing that pair. The held-out unit in both is an "
    "ordered pair of poses from one alphabet, as on OakInk2's transitions axis, and not a cell of a two-factor grid "
    "as on the other eight axes. [PENDING: a control on a permuted two-factor grid of OakInk-Image and of GRAB "
    "themselves is running.]")

# --- Limitations ------------------------------------------------------------------------------------------------
rep("One prior architecture was used, so the size of the penalty, though not necessarily its pattern, may differ "
    "for others.",
    "The error is a reconstruction error under teacher forcing, so nothing here shows that a prior cannot generate "
    "an unseen composition; it shows that a prior reconstructs one less well when it has not seen that cell. One "
    "prior architecture was used, so the size of the penalty, though not necessarily its pattern, may differ for "
    "others.")

rep("No downstream task was evaluated, so the consequences of the penalty for a robot policy were not tested.",
    "The motion-level reading of Table 3 exists for OakInk2 and for the constructed OakInk-Image trajectories only, "
    "and its two rows at the control agree on two datasets, which is a small basis. No downstream task was "
    "evaluated, so the consequences of the penalty for a robot policy were not tested. [PENDING: an out-of-sample "
    "test on a fourth dataset, TACO [8], whose prediction was registered before its data were opened.]")

P.write_text(s, encoding="utf-8")
print("referee round 1, part 2 applied")
print("coverage fails:", fails)
print(f"OakInk2: label covers median {lc_med:.0f}% of frames; naive contamination {mc_rec:.1f}% of recordings "
      f"({mc_frm:.1f}% of frames); target purity {mc_pur:.0f}%")
