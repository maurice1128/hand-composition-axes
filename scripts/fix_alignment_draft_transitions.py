"""Bring the OakInk2 transitions result (runs/DIAGNOSTICS_RESULTS.md section 17) into docs/PAPER_ALIGNMENT_DRAFT.md.

At the common stride that axis carries a penalty, so every sentence that said no GRAB or OakInk2 axis does must
change. Numbers are recomputed here from results.json. Matching ignores line wrapping.
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


def rel(d, both=False):
    by = {}
    for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]:
        if r["budget"] == 256:
            by.setdefault(r["seed"], {})[r["kind"]] = r
    return np.array([100 * v["perframe"]["penalty"] / v["perframe"]["naive"]["mse_target"]
                     for v in by.values() if "perframe" in v and (not both or "modular" in v)])


def welch_p(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


s4, s16, c = rel("oakink2_transitions_s4"), rel("oakink2_paired_v2", both=True), rel("pc_easy_rerun")
assert len(s4) == 12 and len(s16) == 12
p_c, p_16c, p_s = welch_p(s4, c), welch_p(s16, c), welch_p(s4, s16)

# Table 1: the fifth OakInk2/GRAB axis.
rep("| OakInk2 | scene × primitive | +2.3 | 0.82 | 40 |",
    f"| OakInk2 | scene × primitive | +2.3 | 0.82 | 40 |\n| OakInk2 | annotated transitions | +{s4.mean():.1f} | "
    f"{p_c:.4f} | {len(s4)} |")

# 3.2 text.
rep("The four OakInk-Image axes carried a penalty above the control, and the four GRAB and OakInk2 axes did not. "
    "[PENDING: a fifth OakInk2 axis, over annotated primitive transitions, is being re-run at the common stride.]",
    f"The four OakInk-Image axes carried a penalty above the control. Of the five GRAB and OakInk2 axes, four did "
    f"not, and OakInk2's axis over annotated primitive transitions did (+{s4.mean():.2f} %, {int((s4 > 0).sum())} of "
    f"{len(s4)} seeds positive). On that axis a recording is labelled with its whole sequence of primitives and a "
    f"composition is a transition between two consecutive primitives, whereas on the other two OakInk2 axes a "
    f"recording is labelled with its scene and a single item, the first primitive of its sequence or its verb. An "
    f"earlier sweep of the transitions axis at a stride of 16 had returned +{s16.mean():.2f} % (p = {p_16c:.2f} "
    f"against the control); the two strides differed (p = {p_s:.4f}).")
rep("reached at most +3.0 percentage points on the GRAB and OakInk2 axes.",
    "reached at most +3.0 percentage points on the four GRAB and OakInk2 axes without a penalty.")

# Abstract.
rep("whereas the four GRAB and OakInk2 axes tested did not (+1.3 % to +3.0 %).",
    f"whereas four GRAB and OakInk2 axes that label a whole recording by one pair of factors did not (+1.3 % to "
    f"+3.0 %); an OakInk2 axis whose label lists the primitives a recording contains carried +{s4.mean():.1f} %.")

# Discussion opening.
rep("a compositional penalty was present on every axis of OakInk-Image and on no axis of GRAB or OakInk2.",
    "a compositional penalty was present on every OakInk-Image axis, on one OakInk2 axis and on no GRAB axis.")
rep("for OakInk2 it is inferred from the effect of relabelling.",
    "for OakInk2 it is inferred from the effect of relabelling. The one OakInk2 axis that carried a penalty on whole "
    "recordings is consistent with this, because its label lists everything a recording contains; that reading is "
    "post hoc, since the axis was not designed to test it, and a held-out transition may also be close to a held-out "
    "task.")

# Acknowledgements: the stride departure now covers two axes, one of which changed.
rep("Window stride was not uniform across the earliest sweeps, and the affected GRAB axis was re-run at the common "
    "stride.",
    "Window stride was not uniform across the earliest sweeps; the two affected axes were re-run at the common stride, "
    "which left the GRAB axis unchanged and changed the OakInk2 transitions axis from a null to a penalty "
    "(Section 3.2).")

P.write_text(s, encoding="utf-8")
print(f"transitions stride 4 {s4.mean():+.2f}% (p vs control {p_c:.4f}); stride 16 {s16.mean():+.2f}% (p {p_16c:.2f}); "
      f"between strides p {p_s:.4f}")
