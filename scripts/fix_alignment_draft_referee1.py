"""Referee round 1: the corrections that need no new result.

The cold referee returned major revision. The single most important item: the abstract and the conclusion state the
alignment condition as sufficient ("a dataset can measure ... when each label describes the motion"), while Section
3.3 reports a GRAB axis that satisfies it and stays at the control. The paper contains its own counterexample. This
script restates the claim as necessary-but-not-sufficient wherever it appears, confronts the category-by-subject
axis, names the inferential unit, and points at the artefacts.

Items that need numbers being computed or trained (per-axis diagnostics, the permuted-grid control, the two n = 12
replications, related work) are NOT touched here.

Matching ignores how the draft is line-wrapped; each `old` must match exactly once.
"""
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1] / "docs" / "PAPER_ALIGNMENT_DRAFT.md"
s = P.read_text(encoding="utf-8")


def rep(old, new):
    global s
    pat = re.compile(r"\s+".join(re.escape(tok) for tok in old.split()))
    hits = pat.findall(s)
    assert len(hits) == 1, f"expected 1, found {len(hits)}: {old[:70]}"
    s = pat.sub(lambda m: new, s, count=1)


# --- The sufficiency over-claim, in all three places it appears. -----------------------------------------------

rep("A dataset can therefore measure compositional generalisation when each label describes the motion that is "
    "scored, and a dataset that labels long recordings as a whole can return a null that reflects the labelling "
    "and not the prior.",
    "Describing the scored motion was therefore necessary but not sufficient: a GRAB axis whose labels already "
    "name a single object and intent stayed at the control, and trimming its recordings to their contact segment "
    "did not change that. A dataset that labels long recordings as a whole can return a null that reflects its "
    "labelling rather than its prior, and a null obtained under labels that do describe the scored motion still "
    "needs a positive control on that dataset before it is read as a property of the prior.")

rep("In conclusion, a hand-motion dataset measured compositional generalisation when each label described the "
    "motion that was scored. Whether a prior appears to generalise to unseen compositions therefore depends on how "
    "the data were cut and labelled as well as on the prior.",
    "In conclusion, whether a hand-motion dataset can measure compositional generalisation depended on whether each "
    "label described the motion that was scored: removing that relation removed the penalty and supplying it "
    "produced one. The relation was necessary and was not sufficient, since a GRAB axis that satisfies it stayed at "
    "the control for reasons this study did not establish. Whether a prior appears to generalise to unseen "
    "compositions therefore depends on how the data were cut and labelled as well as on the prior.")

# The third statement of it, in the Introduction's framing of question (3), is already conditional; the Discussion's
# "Three points follow" opener is where a reader meets the rule, so it is scoped there too.
rep("Three points follow. First, a null on a dataset of long, whole-recording labels should not be read as evidence "
    "that a prior composes well.",
    "Three points follow, and the first two are conditions on the data rather than guarantees. First, a null on a "
    "dataset of long, whole-recording labels should not be read as evidence that a prior composes well.")

# --- The category-by-subject axis is not a composition of action factors. -------------------------------------

rep("The 95 % intervals on the difference from the control reached at most +3.0 percentage points on the four GRAB "
    "and OakInk2 axes without a penalty.",
    "The 95 % intervals on the difference from the control reached at most +3.0 percentage points on the four GRAB "
    "and OakInk2 axes without a penalty. One of the four OakInk-Image axes pairs an object category with the "
    "identity of the subject who performed the grasp, which is not a factor of the action. Its penalty of +10.3 % "
    "therefore shows that the measurement responds to a held-out cell of a two-factor grid whose second factor "
    "names the performer, and not only to a held-out composition of object and intent. It is reported here for that "
    "reason and no claim about composition rests on it.")

rep("This study has several limitations. One prior architecture was used, so the size of the penalty, though not "
    "necessarily its pattern, may differ for others.",
    "This study has several limitations. The quantity measured is the improvement a prior gains from having seen "
    "other trajectories of a held-out cell. Holding out a cell removes a structured region of the pose "
    "distribution, so a prior that merely lacks coverage of that region, with no composition involved, would also "
    "produce a positive value; the category-by-subject axis (Section 3.2) shows that the measurement responds when "
    "the second factor is not an action factor at all. The synthetic zero-truth control bounds this only on "
    "synthetic data, where held-out pairings are interpolable by construction. One prior architecture was used, so "
    "the size of the penalty, though not necessarily its pattern, may differ for others.")

# --- What the p-values are over. -------------------------------------------------------------------------------

rep("Every penalty was compared with the zero-truth control at the same budget by a two-sided Welch test on the "
    "per-seed values, because a test against zero detects the bias of the paired design itself.",
    "Every penalty was compared with the zero-truth control at the same budget by a two-sided Welch test on the "
    "per-seed values, because a test against zero detects the bias of the paired design itself. A seed draws a "
    "split, so these tests quantify variability over splits of one fixed dataset. They support statements about "
    "whether a given dataset yields a penalty under resampling, and not about datasets with a given property in "
    "general: each dataset here contributes one such object, and the comparisons between datasets are comparisons "
    "of three fixed objects.")

# --- Where the artefacts are. ----------------------------------------------------------------------------------

rep("The analysis scripts, the gate transcripts and the per-seed results from which every reported number is "
    "computed are provided.",
    "The analysis scripts, the gate transcripts, the pre-registered readings of every sweep and the per-seed "
    "results from which every reported number is computed are supplied as supplementary material, together with "
    "the script that recomputes each number in this manuscript from them and checks it against the text.")

# --- Two datasets of the three share a family. -----------------------------------------------------------------

rep("Three public datasets were used: OakInk-Image [3]",
    "Three public datasets were used, two of which, OakInk-Image and OakInk2, come from the same collection effort "
    "and share a capture protocol, so the three are not independent samples of the ways a hand-motion dataset can "
    "be built: OakInk-Image [3]")

P.write_text(s, encoding="utf-8")
print("referee round 1 (no-new-result items) applied")
