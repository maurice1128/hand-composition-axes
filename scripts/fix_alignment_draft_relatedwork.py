"""Add the related-work section the cold referee said was missing, and retire the novelty of the split itself.

A deep literature search found that held-out factor-pairing splits are established prior art in three communities,
and that the most dangerous case is TACO (CVPR 2024): it applies exactly this split to hand-object MOTION
FORECASTING. Its S3 is "The interaction triplet is novel, while the tool categories and geometries are included in
the training set" (arXiv:2401.08399v2), with N = 10 observed and M = 10 forecast frames. So no part of this paper's
contribution can rest on the split design, and the paper must cite and differentiate from TACO or a reviewer will
reject it on that ground alone.

What the same search did not find in any of those papers: a control whose true effect is zero, a planted positive
control on real data, any analysis of whether a label describes the scored window, or any window/stride sensitivity
check. TACO in particular reports single-run point estimates with no seeds, SDs, CIs or significance tests, and no
known-zero condition. That is where the contribution survives, and it is what this section says.

A second differentiation, and the cleaner one: in the prior art the compositional structure is designed or
generated, so a label is true of its episode by construction and the question this paper asks cannot arise. It
arises only for observational corpora with pre-existing annotations.

Every citation here was verified against the paper's own arXiv page for author list, title and venue.
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


# The split is adopted, not proposed. Say so where the split is introduced.
rep("Holding out a pairing of two factors while keeping each factor present in training gives a compositional split "
    "of the kind used to evaluate generalisation in other domains.",
    "Holding out a pairing of two factors while keeping each factor present in training gives a compositional split. "
    "This design is adopted here and is not proposed: it is already used to evaluate generalisation in hand-object "
    "interaction [8], in robot manipulation [9-11] and in language [12], as Section 1.1 sets out.")

rep("To our knowledge no study has asked whether a compositional penalty measured on one of these datasets can be "
    "compared with one measured on another, or whether a null result reflects the prior or the labelling.",
    "What has not been asked, to our knowledge, is whether a compositional penalty measured on one of these datasets "
    "can be compared with one measured on another, or whether a null result reflects the prior or the labelling.")

# The section itself, at the end of the Introduction so that no existing number changes.
RELATED = """
### 1.1 Relation to existing work

Held-out pairing splits are established, and this study adopts the design rather than proposing it. In hand-object
interaction, TACO [8] evaluates four splits of increasing difficulty, one of which holds out the tool-action-object
triplet while "the tool categories and geometries are included in the training set", and applies them to forecasting
hand and object motion from ten observed frames. In robot manipulation, policies have been evaluated on unseen
combinations of environmental factors in order to decide which data need be collected at all [9], on compositions of
atomic skills withheld from training [10], and with the resulting gap decomposed into marginal, compositional and
context terms [11].

What none of these reports is a calibration of the measurement. Their estimates come from one fixed partition and,
in the hand-motion case, from a single run per split, without variation over splits, without a condition whose true
effect is known to be zero, and without a demonstration that the measurement would register a difficulty of a given
size if one were present. TACO's finding that its triplet-level split is harder than its geometry-level split, for
example, rests on one run of each. The paired design used here was chosen for the same reason: it is the arrangement
under which a zero can be supplied.

A second difference concerns where the labels come from. In the work above the compositional structure is designed
or generated - a factor grid enumerated in simulation, a hierarchy of intentions synthesised - so a label is true of
its episode by construction, and whether the label describes the motion that is scored cannot become a question. The
datasets used here are observational: their annotations were made for other purposes, at a granularity chosen for
those purposes, and that question turns out to decide whether a measurement is possible at all.

Whether compositional evaluations measure what they intend to has been examined in language, where four datasets
under eight splitting strategies ranked the same models differently, and the authors concluded that establishing the
validity of an evaluation set remains open [12]. The present study is an instance in which a suspected cause of such
disagreement can be manipulated directly rather than only observed, by removing the relation between a label and the
scored motion from a dataset that shows a penalty and supplying it to one that does not.

"""
rep("## 2. Methods", RELATED.lstrip("\n") + "## 2. Methods")

# References. The existing list ends at 7.
rep("7. Higgins I, et al. beta-VAE: learning basic visual concepts with a constrained variational framework. "
    "ICLR 2017.",
    "7. Higgins I, et al. beta-VAE: learning basic visual concepts with a constrained variational framework. "
    "ICLR 2017.\n"
    "8. Liu Y, Yang H, Si X, Liu L, Li Z, Zhang Y, Liu Y, Yi L. TACO: benchmarking generalizable bimanual "
    "tool-action-object understanding. CVPR 2024. arXiv:2401.08399.\n"
    "9. Gao J, Xie A, Xiao T, Finn C, Sadigh D. Efficient data collection for robotic manipulation via "
    "compositional generalization. RSS 2024. arXiv:2403.05110.\n"
    "10. Chen Y, Chen Z, Chan NT, Chen J, Yin J, Shi J, Gao Y, Li Y-L, Huo J. RoboHiMan: a hierarchical evaluation "
    "paradigm for compositional generalization in long-horizon manipulation. arXiv:2510.13149, 2025.\n"
    "11. Wang Y, Wu C-E, Sun L, Wang P, Ji X, Liang B, Zhan G, Tomizuka M. Diagnosing compositional generalization "
    "in sequential robot tasks. arXiv:2607.29687, 2026.\n"
    "12. Sun K, Williams A, Hupkes D. The validity of evaluation results: assessing concurrence across "
    "compositionality benchmarks. CoNLL 2023. arXiv:2310.17514.")

# The Discussion should close the loop with the literature it now cites.
rep("For the collection of new data, the implication is that each recorded trajectory should contain the one action "
    "its label names, as OakInk-Image's single-grasp clips do.",
    "The disagreement between compositional benchmarks reported in language [12] has a candidate cause here that can "
    "be stated concretely: two datasets can be given the same split design and return different answers because "
    "their labels stand in different relations to the motion that is scored. For the collection of new data, the "
    "implication is that each recorded trajectory should contain the one action its label names, as OakInk-Image's "
    "single-grasp clips do.")

P.write_text(s, encoding="utf-8")
print("related work added; references now run to 12")
