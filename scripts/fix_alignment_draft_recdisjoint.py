"""Fill the draft's three [PENDING] markers with the recording-disjoint OakInk2 result (runs/DIAGNOSTICS_RESULTS.md
section 15). Numbers are recomputed here from results.json so the inserted text cannot drift from the artifacts."""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "docs" / "PAPER_ALIGNMENT_DRAFT.md"
s = P.read_text(encoding="utf-8")


def rel(d):
    R = json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]
    return np.array([100 * r["penalty"] / r["naive"]["mse_target"] for r in R if r["kind"] == "perframe" and r["budget"] == 256])


def welch(a, b):
    t = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q = stats.t.ppf(0.975, df)
    d = a.mean() - b.mean()
    return t.pvalue, d, d - q * se, d + q * se


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, f"expected 1, found {n}: {old[:70]}"
    s = s.replace(old, new)


rd, c, whole, rp = rel("oakink2_primseg_recdisjoint"), rel("pc_easy_rerun"), rel("oakink2_scene_primitive"), rel("oakink2_primseg_rep")
assert len(rd) == 40
pc, dc, lc, hc = welch(rd, c)
pw, dw, lw, hw = welch(rd, whole)
pr = welch(rd, rp)[0]

rep("[PENDING: recording-disjoint confirmation.] A dataset",
    f"With every training segment that shared a recording with a target removed, the penalty was +{rd.mean():.1f} % "
    f"and still above the control. A dataset")
rep("[PENDING: in a third\nsweep, every training segment that shared a recording with a target segment was removed from both arms.]",
    "Because about\n45 % of target segments shared a recording with some training segment, a third sweep on 40 further "
    "seeds removed from\nboth arms every training segment whose recording supplied a target segment.")
rep("[PENDING: recording-disjoint sweep, seeds 200 to 239.]",
    f"With no recording shared between targets and training, primitive segments returned +{rd.mean():.2f} % over "
    f"{len(rd)} seeds ({int((rd > 0).sum())} positive), above the control (difference +{dc:.1f}, 95 % CI {lc:.1f} to "
    f"{hc:.1f}, p = {pc:.3f}) and above whole recordings (difference +{dw:.1f}, 95 % CI {lw:.1f} to {hw:.1f}, "
    f"p = {pw:.3f}). It did not differ from the repetition (p = {pr:.2f}).")
rep("items marked [PENDING] wait for a running sweep.*",
    "Table 2 and Section 3.4's OakInk-Image paragraph still describe the version-2 aligned bundle, which an audit "
    "found confounded; they are rewritten when the version-4 pair finishes.*")
P.write_text(s, encoding="utf-8")
print("filled: recording-disjoint %+.2f%%, vs control p %.3f, vs whole p %.3f, vs repetition p %.2f" % (rd.mean(), pc, pw, pr))
