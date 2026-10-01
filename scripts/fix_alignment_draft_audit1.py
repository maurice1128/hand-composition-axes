"""Apply the first adversarial audit's corrections to docs/PAPER_ALIGNMENT_DRAFT.md (2026-09-20).

Each replacement is asserted to match exactly once. Table 2 and the aligned contrast are NOT touched here: the audit
found the aligned bundle's informed arm was exposed to the targets' constituent object->intent pairs, and a clean
aligned bundle (scripts/build_oakink_alignment_v4.py) is being built; that section is rewritten when it is swept.
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1] / "docs" / "PAPER_ALIGNMENT_DRAFT.md"
s = P.read_text(encoding="utf-8")


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, f"expected 1, found {n}: {old[:80]}"
    s = s.replace(old, new)


# Overstatements.
rep("It was explained by whether a trajectory's label\ndescribed the windows being scored.",
    "It followed whether a trajectory's label\ndescribed the windows being scored.")
rep("A dataset can therefore measure compositional generalisation only when\neach label describes",
    "A dataset can therefore measure compositional generalisation when\neach label describes")
rep("returned a null or a penalty depending only on how",
    "returned a null or a penalty depending on how")
rep("that dataset's null is a property of its labels and poses and not of the instrument.",
    "that dataset's null is not explained by insensitivity to an interaction of that size.")
rep("In conclusion, a hand-motion dataset measured compositional generalisation only when each label described",
    "In conclusion, a hand-motion dataset measured compositional generalisation when each label described")

# 2.1: inference is per dataset and digit group, not per joint.
rep("its sign and the abduction axis were inferred per joint from the data and were not assumed.",
    "its sign and the abduction axis were inferred from each dataset, separately for the fingers and the thumb, "
    "and were not assumed.")

# 2.2: stride is not 4 everywhere.
rep("for 120 epochs on windows cut every 4 frames, with mean-squared error on pose,",
    "for 120 epochs on windows cut every 4 frames (every 32 frames in the three GRAB sweeps of Section 3.3, which "
    "are compared with one another), with mean-squared error on pose,")

# 2.3: what the target set and the informed arm are; held counts; seed rule.
rep("For each seed, a fixed number of cells was held out, and the\ntrajectories in those cells formed the target set.",
    "For each seed, four to eight cells were held out (five on most axes), and about half of the\ntrajectories in "
    "those cells, taken as whole fine labels, formed the target set.")
rep("and the informed arm's included them. Both were scored on the",
    "and the informed arm's included the remaining trajectories of those cells. Both were scored on the")
rep("each seed drawing its own split.\n\nThree checks",
    "each seed drawing its own split. On sweeps that had also trained a second prior for a comparison not reported "
    "here, only seeds on which both had finished were used.\n\nThree checks")

# 2.3: the checks, as they actually were.
rep("Three checks were applied before any sweep was read. Held-out cells had to span several distinct fine labels, and\n"
    "whole fine labels were assigned to one side, so that no target's exact fine composition appeared in either training\n"
    "set. The informed arm had to reach more than 0.8 of the held-out cells with at least three examples each. No target\n"
    "frame was allowed to appear verbatim in either arm's training set; this was checked frame by frame, because a check\n"
    "on labels alone does not detect reused clips. Each derived dataset was required to pass all three before training.",
    "Three checks were applied. Held-out cells had to span several distinct fine labels, and whole fine labels were\n"
    "assigned to one side; the experiment refuses any split in which a target's exact fine composition appears in either\n"
    "training set. Coverage was checked by a separate script and required the informed arm to reach more than 0.8 of the\n"
    "held-out cells with a mean of at least three examples per covered cell. It failed on a minority of seeds in most\n"
    "sweeps (at most 12 of 70); no seed was dropped, and estimates excluding the failing seeds are given where they bear\n"
    "on a claim. Finally, no target frame was allowed to appear verbatim in either arm's training set. This frame-level\n"
    "check was introduced during the study after a fault (Acknowledgements), because a check on labels alone does not\n"
    "detect reused clips, and was applied to five seeds of each derived dataset.")

# 2.5: honest descriptions of the three manipulations.
rep("giving a pure interaction matched in size to the synthetic planted offset.",
    "giving an interaction term matched in size to the synthetic planted offset; after clamping to the joint limits, "
    "which altered 9.8 % of values, 53 of 70 cells carried an offset.")
rep("OakInk-Image poses were clamped so that the same six degrees of freedom were pinned or dead as in GRAB.",
    "OakInk-Image poses were clamped so that six degrees of freedom were pinned or dead, as in GRAB; one of GRAB's "
    "pinned degrees of freedom was left unchanged because OakInk-Image is already pinned more often there.")
rep("from 76 % to 49 %. A fourth variant",
    "from 76 % to 49 %, which also removed 8 categories and 243 of 770 trajectories. A fourth variant")
rep("kept only the contact segment of each GRAB trajectory (8.5 s to 4.9 s) without",
    "kept only the contact segment of each GRAB trajectory (8.5 s to 4.9 s; 58 of 1,048 trajectories dropped) without")

# 3.3: say which stride the GRAB comparisons ran at, so the two values for one axis are explained.
rep("(p = 3.1 × 10⁻⁹). With GRAB's pattern",
    "(p = 3.1 × 10⁻⁹); both ran at a stride of 32, at which the unmodified axis does not differ from its stride-4 "
    "value in Table 1 (p = 0.71). With GRAB's pattern")

# Acknowledgements: every fault on the record, not one.
rep("One fault occurred during the study and is recorded here. A first version",
    "Four departures from plan occurred and are recorded here. An interim value of the category-by-subject sweep was "
    "seen at 28 of 40 seeds through a scripting side effect; its seed list and readings were not changed. The "
    "affordance-by-intent sweep was extended from 70 to 130 seeds after its first 70 had been seen, and the confirmatory "
    "claim does not rest on it. Window stride was not uniform across the earliest sweeps, and the affected GRAB axis "
    "was re-run at the common stride. Finally, a first version")

P.write_text(s, encoding="utf-8")
print("audit-1 corrections applied")
