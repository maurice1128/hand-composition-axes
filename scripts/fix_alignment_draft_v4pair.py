"""Rewrite the OakInk-Image alignment passages of docs/PAPER_ALIGNMENT_DRAFT.md around the version-4 pair
(runs/DIAGNOSTICS_RESULTS.md section 16), which has no constituent leak. Numbers inserted here are recomputed from
results.json; descriptive build facts come from runs/build_oakink_alignment_v4.json and runs/constituent_leak_pair.json
as reported in the pre-registration."""
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


def rows(d):
    R = json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]
    return {r["seed"]: 100 * r["penalty"] / r["naive"]["mse_target"] for r in R if r["kind"] == "perframe" and r["budget"] == 64}


def welch(a, b):
    a, b = np.array(a), np.array(b)
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


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, f"expected 1, found {n}: {old[:70]}"
    s = s.replace(old, new)


al, mis, base = rows("pc_oakink_pair_aligned"), rows("pc_oakink_pair_misaligned"), rows("pc_oakink_b64")
ctrl = list(rows("pc_easy_rerun").values())
A, M, B = list(al.values()), list(mis.values()), list(base.values())
assert len(A) == len(M) == len(B) == 40


def line(name, x):
    x = np.array(x)
    p = welch(x, ctrl)[0]
    return f"| {name} | +{x.mean():.2f} | {x.std(ddof=1):.2f} | {int((x > 0).sum())}/{len(x)} | {sci(p) if p < 1e-4 else f'{p:.2f}'} |"


p_ma, d_ma, lo_ma, hi_ma = welch(M, A)
p_ab, d_ab, lo_ab, hi_ab = welch(A, B)
p_mb, d_mb, lo_mb, hi_mb = welch(M, B)
# Gate-passing estimates (coverage-failing seeds from the gate transcripts).
fa, fm = {3, 29}, {8, 15, 18, 22, 25}
A2 = [v for k, v in al.items() if k not in fa]
M2 = [v for k, v in mis.items() if k not in fm]
p_mc2 = welch(M2, ctrl)[0]
p_ma2, d_ma2, _, _ = welch(M2, A2)

# Abstract.
rep("When OakInk-Image clips were concatenated so that the label still described every\n"
    "window, the penalty was +17.9 %; when the label described only the first third, it fell to +3.1 %, equal to the\n"
    "control.",
    f"When pairs of OakInk-Image clips were joined so that the label still described every\n"
    f"window, the penalty was +{np.mean(A):.1f} %; when the label described only the first clip, it fell to "
    f"+{np.mean(M):.1f} %, not\ndistinguishable from the control.")

# Methods 2.6.
rep("On OakInk-Image, three clips were concatenated into one trajectory\n"
    "labelled by its first clip, with no source clip used more than once. In the aligned version all three clips shared\n"
    "the label's category and intent, so the label described every window. In the misaligned version the second and third\n"
    "clips shared the category but had other intents, so the label described the first third only. Both versions were\n"
    "built from the same 14 object categories, with 210 trajectories each and per-category counts within two of each\n"
    "other, and were swept at a budget of 64 together with an unmodified baseline at that budget, because 770 clips cannot\n"
    "supply non-overlapping trajectories at 256.",
    "On OakInk-Image, two clips were joined into one trajectory\n"
    "labelled by its first clip, with no source clip used more than once. In the aligned version both clips had the same\n"
    "object and intent, so the label described every window and every clip carried the trajectory's own fine label. In\n"
    "the misaligned version the second clip had the same category but another intent, so the label described the first\n"
    "clip only. Both versions had 334 trajectories over the same 33 object categories, with per-category counts within\n"
    "one of each other, and were swept at a budget of 64 together with an unmodified baseline at that budget, because\n"
    "770 clips cannot supply non-overlapping trajectories at 256. Identical fine labels within a trajectory were\n"
    "required because the split keeps fine labels, and not individual clips, apart: in an earlier version that joined\n"
    "clips of different objects, 88 % of target trajectories contained a clip whose object and intent the informed arm\n"
    "had seen, and that version is not used here.")

# Results 3.4: Table 2 and its paragraph.
rep("| Concatenated, label describes every window | +17.86 | 8.77 | 40/40 | 1.7 × 10⁻¹³ |\n"
    "| Concatenated, label describes the first third | +3.05 | 4.51 | 32/40 | 0.92 |",
    line("Two clips joined, label describes every window", A) + "\n" +
    line("Two clips joined, label describes the first clip", M))
rep("The misaligned version was 14.8 percentage points below the aligned version (95 % CI 11.7 to 17.9, p = 1.9 × 10⁻¹³)\n"
    "and did not differ from the control. The misaligned trajectories were the longer of the two (median 280 against 206\n"
    "frames).",
    f"The misaligned version was {-d_ma:.1f} percentage points below the aligned version (95 % CI {-hi_ma:.1f} to "
    f"{-lo_ma:.1f}, p = {sci(p_ma)}) and {-d_mb:.1f} below the unmodified clips (95 % CI {-hi_mb:.1f} to {-lo_mb:.1f}, "
    f"p = {sci(p_mb)}), and did not differ from the control. The aligned version was above the unmodified clips "
    f"(difference +{d_ab:.1f}, 95 % CI {lo_ab:.1f} to {hi_ab:.1f}, p = {p_ab:.4f}), so joining clips did not by itself "
    f"reduce the penalty. The misaligned trajectories were the longer of the two (median 181 against 142 frames). "
    f"Excluding the seeds that failed the coverage check left the aligned version at +{np.mean(A2):.2f} % and the "
    f"misaligned version at +{np.mean(M2):.2f} %, still {-d_ma2:.1f} points apart (p = {sci(p_ma2)}), with the misaligned "
    f"version against the control at p = {p_mc2:.3f}.\n\n"
    "In the misaligned trajectories the split no longer separated the motion. Averaged over ten seeds, 8.4 % of the "
    "naive arm's training clips belonged to held-out cells, 27.9 % of target trajectories contained a clip whose object "
    "and intent the naive arm had seen, and 61 % of target clips belonged to a held-out cell; in the aligned "
    "trajectories these were 0 %, 0 % and 100 %.")

# Discussion: state the mechanism as it is.
rep("removing that\nrelation on OakInk-Image removed the penalty, and supplying it on OakInk2 produced one.",
    "removing that\nrelation on OakInk-Image removed the penalty, and supplying it on OakInk2 produced one. The reason "
    "is that a compositional split is made on labels. When a label does not describe what a trajectory contains, "
    "holding out the label does not hold out the motion: the naive arm sees the held-out motion under other labels, "
    "and much of what is scored is not the held-out composition.")

# Limitations: the misalignment sentence, reworded to match.
rep("In the misaligned OakInk-Image\n"
    "trajectories the label carried almost no information about intent over the whole trajectory, which is what\n"
    "misalignment means in this design, so a label that fails to describe the window and a label that is uninformative\n"
    "about it were not separated.",
    "In the misaligned OakInk-Image\n"
    "trajectories the label carried almost no information about intent over the whole trajectory, which is what\n"
    "misalignment means in this design, so a label that fails to describe the window and a label that is uninformative\n"
    "about it were not separated. The aligned version exceeded the unmodified clips, possibly because a budget counted\n"
    "in trajectories gives the informed arm more held-out clips when each trajectory holds two; this was not tested.")

# Acknowledgements: the second fault on the concatenated bundles.
rep("and the version reported here uses each clip once.",
    "and later versions use each clip once. A second version used each clip once but joined clips of different objects "
    "under one fine label, which exposed the informed arm to the targets' objects; it was replaced by the two-clip "
    "version reported here, and a clip-level check was added.")

# Header note.
rep("Table 2 and Section 3.4's OakInk-Image paragraph still describe the version-2 aligned bundle, which an audit "
    "found confounded; they are rewritten when the version-4 pair finishes.*",
    "Table 2 reports the version-4 pair (no constituent leak).*")

P.write_text(s, encoding="utf-8")
print(f"aligned {np.mean(A):+.2f}, misaligned {np.mean(M):+.2f}, baseline {np.mean(B):+.2f}")
print(f"mis vs aligned {d_ma:+.2f} p {p_ma:.3g}; aligned vs baseline {d_ab:+.2f} p {p_ab:.4f}; mis vs baseline {d_mb:+.2f} p {p_mb:.3g}")
print(f"gate-passing: aligned {np.mean(A2):+.2f}, misaligned {np.mean(M2):+.2f}, mis vs control p {p_mc2:.3f}")
