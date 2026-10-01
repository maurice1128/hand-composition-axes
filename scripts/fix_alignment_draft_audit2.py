"""Apply the second adversarial audit's corrections to docs/PAPER_ALIGNMENT_DRAFT.md (2026-09-20).

Matching ignores how the draft is line-wrapped: each `old` is turned into a regex whose whitespace runs match any
whitespace, and must match exactly once.
"""
import io
import re
import sys
from pathlib import Path

import numpy as np

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


la = np.load(ROOT / "data" / "bundles" / "oakink_pair_aligned.npz", allow_pickle=True)["lengths"]
lm = np.load(ROOT / "data" / "bundles" / "oakink_pair_misaligned.npz", allow_pickle=True)["lengths"]
LEN = (f"means {lm.mean():.0f} and {la.mean():.0f} frames, medians {np.median(lm):g} and {np.median(la):g}, "
       f"misaligned first")

# Scope of the dataset claim.
rep("Against it, all four axes of OakInk-Image carried a compositional penalty of +10.3 % to +17.9 %, whereas "
    "no axis of GRAB or OakInk2 did (+1.3 % to +3.0 %).",
    "Against it, the four OakInk-Image axes tested carried a compositional penalty of +10.3 % to +17.9 %, whereas "
    "the four GRAB and OakInk2 axes tested did not (+1.3 % to +3.0 %).")
rep("It followed whether a trajectory's label described the windows being scored.",
    "On OakInk-Image and OakInk2 it followed whether a trajectory's label described the windows being scored; "
    "GRAB's null was not explained.")
rep("and datasets that label long recordings as a whole return a null that reflects the labelling and not the prior.",
    "and a dataset that labels long recordings as a whole can return a null that reflects the labelling and not the "
    "prior.")
rep("All four OakInk-Image axes carried a penalty above the control, and no GRAB or OakInk2 axis did.",
    "The four OakInk-Image axes carried a penalty above the control, and the four GRAB and OakInk2 axes did not. "
    "[PENDING: a fifth OakInk2 axis, over annotated primitive transitions, is being re-run at the common stride.]")

# 2.2: window duration and the training details a reader needs.
rep("and validation was scored at the final value of beta.",
    "and validation was scored at the final value of beta. Beta was annealed to 1.0, the first-difference term was "
    "weighted 0.1, and optimisation used a learning rate of 0.001 and batches of 256 windows. A window spans 1.07 s "
    "on OakInk-Image and GRAB and 4.3 s on OakInk2. The error throughout is the mean-squared error on normalised "
    "pose over all 27 degrees of freedom.")

# 2.3: what the informed arm holds, how leaks were rejected, the worst coverage rate.
rep("and the informed arm's included the remaining trajectories of those cells.",
    "and the informed arm's included the remaining trajectories of those cells, up to half its budget.")
rep("the experiment refuses any split in which a target's exact fine composition appears in either training set.",
    "a split in which a target's exact fine composition appeared in either training set was rejected, by the "
    "experiment itself in the later sweeps and by a separate check in the earlier ones. A held-out cell had to span "
    "at least four distinct fine labels (five on GRAB's shape-by-fine-intent axis).")
rep("sweeps (at most 12 of 70); no seed was dropped,",
    "sweeps (at most 12 of 70 seeds, and 3 of 12 on functional class by intent); no seed was dropped,")

# 2.5: the zero cells come from removing the additive part, not from clamping.
rep("giving an interaction term matched in size to the synthetic planted offset; after clamping to the joint limits, "
    "which altered 9.8 % of values, 53 of 70 cells carried an offset.",
    "giving an interaction term matched in size to the synthetic planted offset. Removing the additive part left "
    "53 of 70 cells with an offset, and clamping to the joint limits altered 9.8 % of values.")

# 2.6: cross-fade, and what a whole-recording label is on OakInk2.
rep("labelled by its first clip, with no source clip used more than once.",
    "labelled by its first clip, with a four-frame linear cross-fade at the join and no source clip used more than "
    "once.")
rep("On OakInk2, each task recording was cut at the primitive boundaries",
    "On OakInk2, a whole recording had been labelled with its scene and the first primitive of its sequence. Each "
    "task recording was now cut at the primitive boundaries")

# 3.4: length, excluded seeds, informed-arm exposure, the segment sweeps' side effects, the repetition.
rep("The misaligned trajectories were the longer of the two (median 181 against 141.5 frames).",
    f"The two versions were of similar length ({LEN}).")
rep("Excluding the seeds that failed the coverage check left",
    "Excluding the seeds that failed the coverage check (2 aligned, 5 misaligned) left")
rep("in the aligned trajectories these were 0 %, 0 % and 100 %.",
    "in the aligned trajectories these were 0 %, 0 % and 100 %. The informed arm of the misaligned version was "
    "exposed likewise, in 36.5 % of target trajectories.")
rep("It did not differ from the repetition (p = 0.50).",
    "It was 1.8 points below the repetition, a difference that was not distinguishable (p = 0.50). Each arm of the "
    "segment sweeps trained on about a third of the frames of the whole-recording sweep, the naive error rose from "
    "0.018 to between 0.023 and 0.025, and the label grid had 79 cells against 62.")

# Discussion.
rep("removing that relation on OakInk-Image removed the penalty,",
    "removing that relation on OakInk-Image reduced the penalty to a level not distinguishable from the control,")
rep("and much of what is scored is not the held-out composition.",
    "and much of what is scored is not the held-out composition. In the constructed OakInk-Image trajectories this "
    "was measured directly (Section 3.4); for OakInk2 it is inferred from the effect of relabelling.")
rep("and the misaligned OakInk-Image trajectories were longer than the aligned ones.",
    "and the aligned and misaligned OakInk-Image trajectories were of similar length.")

# Limitations: GRAB unexplained; no planted control on OakInk2.
rep("Three datasets were examined.",
    "Three datasets were examined. GRAB's null was not explained: no manipulation produced a penalty on it, and "
    "keeping only its contact segments did not. No planted control was run on OakInk2, because most of its "
    "primitives occur in a single scene and an interaction term cannot be planted there; the penalty measured on its "
    "primitive segments is the only evidence that the instrument responds on its poses.")

# Acknowledgements.
rep("Four departures from plan occurred and are recorded here.", "Five departures from plan occurred and are recorded here.")

P.write_text(s, encoding="utf-8")
print("audit-2 corrections applied;", LEN)
