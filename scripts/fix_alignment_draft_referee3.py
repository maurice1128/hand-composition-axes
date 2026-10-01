"""Referee round 2 (major as it stands, MINOR conditional on the pending runs): framing, readability, and gaps.

The second cold referee's verdict turned on four experiments that are still running, but six of its issues are about
the text whatever those show. This script makes those changes.

  1. WHAT THE QUANTITY IS. The instrument measures the gain from having seen other trajectories of a held-out CELL;
     a lack of coverage would produce it with no composition involved, and the category-by-subject axis shows it
     responding to a non-action factor. Defined as such in Methods; called compositional only where both grid
     factors are factors of the action. The abstract no longer counts the subject axis as compositional.
  2. GRAB'S NULL IS A VALID MEASUREMENT BY THE PAPER'S OWN RULE. It has labels that name one object and intent and
     a recovered planted control. The draft called it "not explained" four times, as a hole in the account. It is a
     result about interpolability; misalignment is one way to a null and interpolability is another, and the
     controls are what separate them. What makes GRAB interpolable is what was not established.
  3. "NECESSARY" WAS OVER-GENERAL. Scoped to the two datasets where the relation could be manipulated.
  4. TABLE 3 IS NOT A MECHANISM. Contamination and purity are collinear across its rows; "The reason is" becomes "A
     candidate reason is", and the small dose (8.4 %) for a 13-point fall is said to point at target purity.
  5. READABILITY. Abstract cut from about 330 to about 200 words with one claim. Section 3.2 gets run-in headings.
     The 450-word limitations block becomes three short groups. Nothing assumes a pending outcome: the n = 12
     transitions row leaves the abstract.
  6. GAPS: stride sensitivity discussed with window counts; multiple comparisons; the one seed the both-priors rule
     dropped; the first-70 affordance estimate; planted control's SD and test; optimiser; which hand; Table 2's
     gate-passing p; "seeds above control" replaces "seeds positive"; TACO described fairly ("reported as point
     estimates", its other two benchmarks named); four more references, each verified on its own arXiv page.

Internal header lines are removed: they name files and breach double-blind review.
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


def between(start, end, new):
    """Replace everything from the line `start` up to (not including) the line `end`."""
    global s
    i, j = s.index(start), s.index(end)
    assert 0 <= i < j
    s = s[:i] + new + s[j:]


def load(d, budget=256, both=False):
    by = {}
    for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]:
        if r["budget"] == budget:
            by.setdefault(r["seed"], {})[r["kind"]] = r
    return {k: v["perframe"] for k, v in by.items() if "perframe" in v and (not both or "modular" in v)}


def pct(rows):
    return np.array([100 * r["penalty"] / r["naive"]["mse_target"] for r in rows.values()])


def sci(p):
    e = int(np.floor(np.log10(p)))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{p / 10 ** e:.1f} × 10{sup}"


def welch_p(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


c256, c64 = pct(load("pc_easy_rerun")), pct(load("pc_easy_rerun", 64))
hard = pct(load("pc_hard_rerun"))
DIAG = {a["axis"]: a for a in json.loads((ROOT / "runs" / "axis_diagnostics.json").read_text(encoding="utf-8"))["axes"]}
MC = json.loads((ROOT / "runs" / "motion_contamination_oakink2.json").read_text(encoding="utf-8"))
mc_rec = 100 * MC["naive_motion_contaminated_recordings"]["mean"]
mc_pur = 100 * MC["target_frame_purity_first_span"]["mean"]

# ---------------------------------------------------------------------------------------------------------------
# Header: internal, and it breaches double-blind review.
between("*Draft 1, 2026-09-20.", "## Abstract", "")

# ---------------------------------------------------------------------------------------------------------------
# Abstract: one claim, about 200 words, nothing resting on a pending run.
cat = pct({**load("oakink_official_category", both=True), **{f"r{k}": v for k, v in load("oakink_category_rest", both=True).items()}})
aff = pct({**load("oakink_official_attr", both=True), **{f"m{k}": v for k, v in load("oakink_attr_more", both=True).items()}})
al, mis = pct(load("pc_oakink_pair_aligned", 64)), pct(load("pc_oakink_pair_misaligned", 64))
whole, seg, rd = pct(load("oakink2_scene_primitive")), pct(load("oakink2_primseg_rep")), pct(load("oakink2_primseg_recdisjoint"))

ABSTRACT = f"""## Abstract

Motion priors for dexterous hands are expected to recombine what they learned separately, and held-out combinations
of label factors are used to test this. We asked when such a test measures anything. On three public hand-motion
datasets the same held-out trajectories were scored by a prior whose training data excluded their label cell and by
one whose data included it, and the difference was read against a synthetic control whose true value is zero, which
itself returned +{c256.mean():.1f} % of the naive error. OakInk-Image's two best-sampled object-by-intent axes carried
a penalty of +{aff.mean():.1f} % and +{cat.mean():.1f} %; four GRAB and OakInk2 axes that label a whole recording by
one pair of factors stayed at the control. A split is made on labels, so it withholds motion only when a label
describes what is scored. Joining OakInk-Image clips so that the label described only the first clip cut the penalty
from +{al.mean():.1f} % to +{mis.mean():.1f} %, not distinguishable from the control. Cutting OakInk2's recordings at
their annotated primitive boundaries raised it from +{whole.mean():.1f} % to +{seg.mean():.1f} %, replicated on fresh
seeds, and to +{rd.mean():.1f} % with recordings kept apart. In OakInk2's whole recordings, {mc_rec:.1f} % of the
training recordings contained held-out motion under another label. A null therefore reflects the prior only where
labels describe the scored motion and a planted difficulty is recovered, as on GRAB, and reflects the labelling
where they do not.

"""
between("## Abstract", "Keywords:", ABSTRACT)

# ---------------------------------------------------------------------------------------------------------------
# Introduction.
rep("OakInk-Image labels an object and an intent [3], GRAB an object shape and an intent [4],",
    "OakInk-Image labels an object and an intent [3], GRAB an object and an intent [4],")

# Related work: fair to TACO, hedged, and four more references.
rep("Held-out pairing splits are established, and this study adopts the design rather than proposing it. In "
    "hand-object interaction, TACO [8] evaluates",
    "Held-out pairing splits are established, and this study adopts the design rather than proposing it. For "
    "hand-object action recognition, Something-Else [13] introduced a split in which the combinations of verb and "
    "noun seen in training do not overlap with those tested. In hand-object interaction, TACO [8] evaluates")
rep("and applies them to forecasting hand and object motion from ten observed frames.",
    "and applies them to forecasting hand and object motion from ten observed frames, as well as to action "
    "recognition and grasp synthesis.")
rep("and with the resulting gap decomposed into marginal, compositional and context terms [11].",
    "and with the resulting gap decomposed into marginal, compositional and context terms [11]; a policy's bias "
    "towards particular instruction factors has also been measured and used to direct data collection [16], which "
    "concerns the policy and not the evaluation.")
rep("What none of these reports is a calibration of the measurement. Their estimates come from one fixed partition "
    "and, in the hand-motion case, from a single run per split, without variation over splits,",
    "What none of these reports, to our knowledge, is a calibration of the measurement. Their estimates come from "
    "one fixed partition and, in the hand-motion case, are reported as point estimates without variation over runs "
    "or splits,")
rep("TACO's finding that its triplet-level split is harder than its geometry-level split, for example, rests on one "
    "run of each.",
    "TACO's finding that its triplet-level split is harder than its geometry-level split, for example, is reported "
    "without an interval.")
rep("so a label is true of its episode by construction, and whether the label describes the motion that is scored "
    "cannot become a question.",
    "so a label is true of its episode by construction, and whether the label describes the motion that is scored "
    "cannot become a question. Work with controlled generative factors has likewise held out combinations of "
    "factors and found that some are recovered by interpolation and others are not [15], and the degree to which a "
    "split is compositional has been quantified from the data alone as the divergence between the compounds of its "
    "two sides [14].")
rep("The present study is an instance in which a suspected cause of such disagreement can be manipulated directly",
    "The present study is an instance, in another modality, in which an analogous cause of such disagreement can be "
    "manipulated directly")

# ---------------------------------------------------------------------------------------------------------------
# Methods.
rep("Each provides MANO hand parameters [6],",
    "The right hand was used throughout. Each provides MANO hand parameters [6],")
rep("and optimisation used a learning rate of 0.001 and batches of 256 windows.",
    "and optimisation used AdamW with cosine annealing from a learning rate of 0.001 and batches of 256 windows; "
    "the weights with the lowest validation loss were scored.")

rep("Both were scored on the identical target windows, and the penalty was the naive error minus the informed "
    "error, expressed as a percentage of the naive error.",
    "Both were scored on the identical target windows, and the penalty was the naive error minus the informed "
    "error, expressed as a percentage of the naive error. The penalty is thus the gain from having seen other "
    "trajectories of a held-out cell. A prior that lacks coverage of that region of poses would show it with no "
    "composition involved, so it is called compositional here only where both factors of the grid are factors of "
    "the action, and one axis on which they are not is reported for that reason (Section 3.2).")

dropped = sum(len(load(d)) - len(load(d, both=True)) for d in
              ("oakink_class_v1", "oakink_official_category", "oakink_category_rest", "oakink_official_attr",
               "oakink_attr_more", "grab_shapeclass", "oakink2_scene_verb"))
assert dropped == 1, dropped
rep("only seeds on which both had finished were used.",
    "only seeds on which both had finished were used, which excluded one seed in the whole study (of category by "
    "intent).")

n_tests = 25
rep("They support statements about whether a given dataset yields a penalty under resampling,",
    f"About {n_tests} such tests share the one control sample and no correction for multiplicity was applied; the "
    f"contrasts on which the argument rests have p below 10⁻⁴ and survive any correction, and those that would not "
    f"are named where they occur. They support statements about whether a given dataset yields a penalty under "
    f"resampling,")

rep("[PENDING: a control on a permuted two-factor grid of OakInk-Image and of GRAB themselves is running.]",
    "[PENDING: a control on OakInk-Image and on GRAB themselves is running. The second factor of the grid is "
    "permuted across fine labels, which keeps every trajectory, each factor's marginal distribution and the cell "
    "structure, and removes any relation between the two factors, so that the trajectories the informed arm gains "
    "are no longer of the target's own composition. Its reading was registered beforehand, including that a "
    "permuted grid reading as high as the real one would withdraw the word compositional from this paper.]")

# ---------------------------------------------------------------------------------------------------------------
# Results 3.1: the planted control's spread and test.
rep("The planted control returned +9.12 % over 18 seeds.",
    f"The planted control returned +{hard.mean():.2f} % over {len(hard)} seeds (SD {hard.std(ddof=1):.1f}), above "
    f"the zero-truth control (p = {sci(welch_p(hard, c256))}). Both budgets of the zero-truth control used "
    f"{len(c64)} seeds.")

# Results 3.2: run-in headings, stride, the first-70 estimate.
rep("The four OakInk-Image axes carried a penalty above the control.",
    "**Pattern.** The four OakInk-Image axes carried a penalty above the control.")

L2 = np.load(ROOT / "data" / "bundles" / "oakink2.npz", allow_pickle=True)["lengths"]
LG = np.load(ROOT / "data" / "bundles" / "grab.npz", allow_pickle=True)["lengths"]
win = lambda L, st: 256 * float(np.maximum((L - 32) // st + 1, 0).mean())
r100 = lambda v: int(round(v, -2))
a70 = pct(load("oakink_official_attr", both=True))
rep("Neither the denominator of the penalty nor the informed arm's exposure explains the pattern between datasets.",
    f"The affordance axis was extended after its first {len(a70)} seeds had been seen (Acknowledgements); those "
    f"{len(a70)} alone gave +{a70.mean():.2f} % (p = {sci(welch_p(a70, c256))}).\n\n"
    f"**Stride.** The transitions axis is the only one on which stride changed the reading; on GRAB a change from 32 "
    f"to 4 did not (Section 3.3). The number of training windows does not account for the difference: each arm had "
    f"about {r100(win(L2, 16)):,} windows on OakInk2 at a stride of 16 and {r100(win(L2, 4)):,} at 4, against "
    f"{r100(win(LG, 32)):,} and {r100(win(LG, 4)):,} on GRAB at 32 and 4, so GRAB's unchanged null was obtained with "
    f"the fewest. A candidate reason is that a transition is localised in time, so that a coarse stride places few "
    f"windows across it, whereas a cell of a two-factor grid is a property of every window of a trajectory. This was "
    f"not tested, and the row is provisional until its repetition is read.\n\n"
    f"**Denominator and exposure.** Neither the denominator of the penalty nor the informed arm's exposure explains "
    f"the pattern between datasets.")
rep("which is the dose-response a real effect predicts. One of the four OakInk-Image axes pairs",
    "which is the dose-response a real effect predicts, whether that effect is compositional or one of coverage."
    "\n\n**The subject axis.** One of the four OakInk-Image axes pairs")

# Table 1: "seeds positive" is uninformative when the zero-truth control is itself positive.
hdr_old = "| Seeds positive | Naive error |"
assert s.count(hdr_old) == 1
s = s.replace(hdr_old, "| Seeds above control | Naive error |")
TABLE1 = [("functional class × intent", ["oakink_class_v1"], True),
          ("category × intent", ["oakink_official_category", "oakink_category_rest"], True),
          ("affordance × intent", ["oakink_official_attr", "oakink_attr_more"], True),
          ("category × subject", ["oakink_category_subject"], False),
          ("shape × fine intent", ["grab_shape_s4"], False), ("shape × intent class", ["grab_shapeclass"], True),
          ("scene × verb", ["oakink2_scene_verb"], True), ("scene × primitive", ["oakink2_scene_primitive"], False),
          ("annotated transitions", ["oakink2_transitions_s4"], False)]
for axis, dirs, both in TABLE1:
    x = np.concatenate([pct(load(d, both=both)) for d in dirs])
    n = len(x)
    old = f"| {int((x > 0).sum())}/{n} |"
    line = next(l for l in s.split("\n") if l.startswith("|") and f"| {axis} |" in l)
    assert line.count(old) == 1, (axis, line)
    s = s.replace(line, line.replace(old, f"| {int((x > c256.mean()).sum())}/{n} |"))
rep("The 95 % CI is of the mean over seeds.",
    f"The 95 % CI is of the mean over seeds. Seeds above control counts the seeds above the zero-truth control's "
    f"mean of +{c256.mean():.2f} %.")

# Table 2: both p-values.


def gate_fails(name):
    out = set()
    for line in (ROOT / "runs" / "gates" / name).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(.*)$", line)
        if m and not m.group(7).strip().startswith("ok"):
            out.add(int(m.group(1)))
    return out


def passing(d, gate):
    bad = gate_fails(gate)
    return pct({k: v for k, v in load(d, 64).items() if k not in bad})


pa = welch_p(passing("pc_oakink_pair_aligned", "cover_oakink_pair_aligned.txt"), c64)
pm = welch_p(passing("pc_oakink_pair_misaligned", "cover_oakink_pair_misaligned.txt"), c64)
rep("| Version | Penalty (% naive) | SD | Seeds positive | p vs control |\n|---|---|---|---|---|",
    "| Version | Penalty (% naive) | SD | Seeds positive | p vs control | p, coverage-passing seeds |\n"
    "|---|---|---|---|---|---|")
for label, extra in (("Unmodified clips", "not checked"),
                     ("Two clips joined, label describes every window", sci(pa)),
                     ("Two clips joined, label describes the first clip", f"{pm:.3f}")):
    line = next(l for l in s.split("\n") if l.startswith(f"| {label} |"))
    s = s.replace(line, line + f" {extra} |")
rep("The two versions were of similar length (means 200 and 204 frames, medians 181 and 141.5, misaligned first).",
    "The two versions were of similar mean length (200 and 204 frames, misaligned first), though not of similar "
    "median (181 and 141.5).")

# The disjoint-recording contrasts would not survive a correction; say so where they occur.
rep("It was 1.8 points below the repetition,",
    "These two contrasts would not survive a correction for the number of tests in this study, unlike the "
    "repetition's. It was 1.8 points below the repetition,")

# ---------------------------------------------------------------------------------------------------------------
# Discussion, limitations and conclusion, rewritten.
DISCUSSION = f"""## 4. Discussion

This study asked when a penalty measured by holding out a cell of a label grid means anything. With one prior and
one measurement, a penalty was present on OakInk-Image's object-by-intent axes and absent on four axes of GRAB and
OakInk2 that label a whole recording by one pair of factors. In the two datasets where the relation between a label
and the scored motion could be manipulated, a penalty appeared only when the label described that motion: removing
the relation on OakInk-Image reduced the penalty to a level not distinguishable from the control, and supplying it
on OakInk2 produced one.

A candidate reason is that a split is made on labels. When a label does not describe what a trajectory contains,
holding out the label does not hold out the motion. Table 3 shows both consequences in the two cases that sat at the
control: about a tenth of the naive arm's training trajectories contained held-out motion under another label, and
only 40 % to 61 % of what was scored was the held-out composition. The table cannot say which of the two matters
more, because they vary together across its rows. A contamination of 8 % is a small dose for a fall of 13 points,
which suggests that the purity of the targets carries much of the effect.

Three points follow. First, a null is a measurement only under two conditions, and they separate the nulls found
here. GRAB meets both: its labels name a single object and intent, and a planted interaction was recovered on its
own poses. Its null is therefore a result about the data and the prior, namely that on GRAB's shape-by-intent axes
this prior needs no example of a held-out cell in order to reconstruct it, at least for a difficulty of the planted
size. OakInk2's whole-recording axes meet neither condition, and the same poses returned a penalty once each segment
carried its own label. Misalignment is one way to obtain a null and interpolability is another, and the controls are
what tell them apart. What makes GRAB interpolable where OakInk-Image is not was not established; GRAB records whole
interactions with their approach and retreat, and OakInk-Image records the grasp alone. Second, the requirement is on
the label and not on the length of the recording. Shortening GRAB trajectories to their contact segment left the
penalty at the control, and the aligned and misaligned OakInk-Image trajectories were of similar mean length. Third,
the differences between datasets that might be blamed first were not responsible. Damaging OakInk-Image's
retargeting to GRAB's level left its penalty in place, thinning its label grid did not reduce it, and neither the
denominator of the penalty nor the informed arm's exposure followed the pattern between datasets.

The disagreement between compositional benchmarks reported in language [12] has an analogue here that can be stated
concretely: two datasets can be given the same split design and return different answers because their labels stand
in different relations to the motion that is scored. For the collection of new data, the implication is that each
recorded trajectory should contain the one action its label names, as OakInk-Image's single-grasp clips do. For
existing datasets, recordings of whole tasks can be made usable by cutting them at annotated action boundaries and
labelling each segment separately. In either case a null should be read only with a zero-truth control, because the
paired design returned +{c256.mean():.1f} % when the true value was zero, and with a planted positive control on the
dataset in question.

This study has limitations of three kinds. *What is measured.* The error is a reconstruction error under teacher
forcing, so nothing here concerns generation. The quantity is the gain from having seen other trajectories of a
held-out cell, which a lack of coverage would produce without composition, as the category-by-subject axis shows.
*Scope.* One prior architecture was used. Three datasets were examined, two of them from one collection effort, and
the reading of Table 3 exists for two. The OakInk-Image manipulation ran at a budget of 64, with its own baseline and
control. No downstream task was evaluated. *Untested.* The zero of the measurement comes from synthetic data and not
from each dataset [PENDING]. OakInk2 has no planted control, because most of its primitives occur in a single scene.
Segmenting OakInk2 also cut each arm's training frames to a third and enlarged the grid from 62 to 79 cells, so the
label was not the only thing that changed. In the misaligned OakInk-Image trajectories, a label that fails to
describe the window and one that is uninformative about it were not separated. The aligned version exceeded the
unmodified clips for a reason that was not tested, and the stride sensitivity of the transitions axis is unexplained.
[PENDING: an out-of-sample test on a fourth dataset, TACO [8]. The prediction, registered before its data were
opened, is that its first testable axis among action by tool, action by object and tool by object carries a penalty
above the control, because each of its sequences is one tool-use action named by its label; a null there would be a
second interpolable dataset and would leave the rule without a demonstrated prediction.]

In conclusion, holding out a label withheld the motion only where the label described what was scored. Where it did,
a penalty could be measured or, as on GRAB, a null could be trusted; where it did not, the same prior and the same
poses returned a null that reflected the labelling. Whether a prior appears to generalise to unseen compositions
therefore depends on how the data were cut and labelled as well as on the prior.

"""
between("## 4. Discussion", "## Acknowledgements", DISCUSSION)

# Acknowledgements: "Finally" stood before the fourth of five.
rep("Finally, a first version of the concatenated OakInk-Image trajectories reused source clips,",
    "A first version of the concatenated OakInk-Image trajectories reused source clips,")

# References 13-16, each verified on its own arXiv page.
rep("12. Sun K, Williams A, Hupkes D. The validity of evaluation results: assessing concurrence across "
    "compositionality benchmarks. CoNLL 2023. arXiv:2310.17514.",
    "12. Sun K, Williams A, Hupkes D. The validity of evaluation results: assessing concurrence across "
    "compositionality benchmarks. CoNLL 2023. arXiv:2310.17514.\n"
    "13. Materzynska J, Xiao T, Herzig R, Xu H, Wang X, Darrell T. Something-Else: compositional action recognition "
    "with spatial-temporal interaction networks. CVPR 2020. arXiv:1912.09930.\n"
    "14. Keysers D, Schärli N, Scales N, Buisman H, Furrer D, Kashubin S, et al. Measuring compositional "
    "generalization: a comprehensive method on realistic data. ICLR 2020. arXiv:1912.09713.\n"
    "15. Schott L, von Kügelgen J, Träuble F, Gehler P, Russell C, Bethge M, Schölkopf B, Locatello F, Brendel W. "
    "Visual representation learning does not generalize strongly within the same domain. ICLR 2022. "
    "arXiv:2107.08221.\n"
    "16. Qi Y, Ye Z, Xu X, Lu Y, Sandhu A, Hu B, Huang H, Tremblay J, Wong LLS. Scale up strategically: learning "
    "compositional generalization via bias-aware evaluation and data collection for robotic manipulation. "
    "arXiv:2607.21582, 2026.")

P.write_text(s, encoding="utf-8")
words = len(ABSTRACT.split()) - 2
print(f"referee round 2 applied; abstract {words} words; gate-passing p: aligned {sci(pa)}, misaligned {pm:.3f}")
