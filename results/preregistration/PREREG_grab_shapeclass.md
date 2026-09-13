# Pre-registration: GRAB shape x intent class

**Written before the sweep starts. Reported whatever it returns.**

## Why this axis, and why only this one

The necessity direction of the paper's relation ("no axis carries compositional
difficulty without positive interaction") rests on one informative observation.
Falsifying it needs an axis with non-positive interaction excess *and* a
measurable penalty, so only non-positive-excess axes are tests, and only ones
whose sweep is interpretable count.

Six axes in `runs/axis_screen.json` have non-positive excess. Five cannot serve:

| axis | excess | why not |
|---|---|---|
| DexYCB subject x object | −0.0899 | no transition appears in >= 2 distinct fine labels; a leak-free paired split cannot be built at any setting |
| DexYCB subject x shape | −0.0214 | same; this is the axis the paper withdraws |
| GRAB object x fine intent | −0.0134 | same |
| GRAB object x intent class | −0.0128 | buildable at `min_chains=2`, but the coverage gate fails 33 of 40 rows at held=5, 28 of 40 at held=3, and depth collapses to 1.00 example per held composition at held=2. No setting yields an interpretable null |
| OakInk2 annotated transitions | −0.0120 | this **is** the existing observation |

**That n = 1 is structural, not a sampling choice.** The paired split requires a
transition to appear in several distinct fine labels; on DexYCB and on GRAB's
object axes every transition appears in exactly one.

The sixth is the subject of this pre-registration:

**GRAB shape x intent class**, excess **−0.0005** (*z* = −0.4), 36 cells, pool of
30 eligible transitions at `min_chains=4`. Its coverage gate passes **39 of 40
rows** at 8.60 examples per covered composition, better than every OakInk axis
the paper reports. The one failing row reaches 4 of 5, scoring exactly 0.80
against a breadth rule of > 0.8.

Note the excess is indistinguishable from zero rather than clearly negative.
This is a weaker test than OakInk2's −0.0120 and will be reported as such: it
tests "difficulty without *positive* interaction", which is what necessity
claims, but it does not test a strongly negative case.

## The declared run

```
scripts/experiment_paired_composition.py
  --bundle data/bundles/grab.npz
  --granularity grab_shape_intentclass
  --out runs/grab_shapeclass
  --budgets 256
  --kinds perframe modular
  --held-compositions 5 --min-chains 4 --min-per-composition 5
  --epochs 120
  --seeds 0..39
```

Forty seeds, fixed now. **No interim looks, no extension, no second seed range**
whichever way it comes out. The gates run before the numbers are read:
`check_informed_coverage.py` (already run, 39/40) and
`check_composition_leak.py`.

## The declared analysis

**Difficulty** is the `perframe` arm's compositional penalty against zero, over
the 40 paired seeds, one-sample *t*-test, two-sided, alpha 0.05. This is the
same statistic and test the paper's Table II reports for every other axis.

## What each outcome means, declared now

- **Difficulty indistinguishable from zero.** The necessity direction gains a
  second informative observation. The paper's "only one of the six could have
  shown otherwise" becomes two of seven, and the accompanying sentence must
  still say that four of the remaining axes are structurally untestable rather
  than untested.

- **Difficulty distinguishable from zero and positive.** The necessity direction
  is **falsified**: an axis with non-positive interaction carries measurable
  compositional difficulty. The paper's claim that interaction is necessary must
  be withdrawn, and the screen's usefulness narrows to the sufficiency-free
  ranking it already admits it does badly. This outcome is publishable and will
  be reported as prominently as the other.

- **Difficulty distinguishable from zero and negative.** Reported as an
  anomaly; a naive arm beating an informed one on its own held-out compositions
  indicates a split defect, and the gates would be re-examined before anything
  is concluded.

The modular-versus-baseline paired contrast will also be computed, for the
survey figure. It is **not** what this sweep is for, and no claim rests on it.

---

## Outcome, recorded 2026-09-03 21:10 after all 40 seeds and the leak gate

Leak gate: PASS, 0.0% on both arms, all 40 seeds. Coverage: 39 of 40 rows.

**Declared test:** perframe penalty +0.00103, sd 0.00272, t = +2.40, **p = 0.021**,
n = 40. By the criterion declared above this is *difficulty distinguishable from
zero and positive* on an axis whose interaction excess is −0.0005 (z = −0.4), so
the necessity direction as the paper states it is **falsified on the declared
test**. That is reported first because it was declared first.

**What the declared test did not anticipate.** The paper's own zero-truth
control (`pc_easy`, Limitations) puts the instrument's floor at +0.00258 to
+0.00306, 2.9 to 3.3% of the naive error, at p < 0.01. The GRAB value is +2.4%
of naive error, *below* that floor, and of the same size as the two axes the
paper reports as null (OakInk2 +3.8% at n = 12, GRAB shape +1.2% at n = 40). The
test reaches significance because n = 40 has the power to detect the
instrument's own bias. Both readings go into the revision: the declared test
falsifies the sentence as written; the control says the effect is inside the
floor the paper already documents.

**Consequence for the manuscript.** "Carries difficulty" must be read against the
control floor, not against zero. On that reading the four OakInk-Image axes (13
to 16% of naive error) carry difficulty and the three others do not, and the
necessity direction survives, but the paper's current wording ("distinguishable
from zero") does not.

## Addendum for the OakInk2 scene x verb sweep, declared before its rows are read

Same declared test (perframe penalty vs zero, two-sided, alpha 0.05), unchanged.
In addition, and declared now: the penalty is reported against the `pc_easy`
floor of +0.00258 (budget 256, rate-matched) as a fraction of naive error, with
the OakInk-Image axes and the GRAB result on the same scale. No other statistic
is added, no seed is added, and the sweep is not stopped early.

---

## OakInk2 scene x verb: outcome, recorded 2026-09-12 after all 40 seeds and the leak gate

Leak gate: PASS, 0.0% on both arms, all 40 seeds. Coverage-thin rows: 2 of 40.
The sweep was interrupted three times (app closure 09-03, cuDNN internal error
under GPU contention 09-10 04:22, external termination 09-10 11:52) and resumed
each time from scored rows and per-model checkpoints; no seed was re-drawn.

**Declared test (vs zero):** perframe penalty +0.00051, sd 0.00126, t = +2.55,
**p = 0.015**, n = 40. Positive and distinguishable from zero, as GRAB was.

**Declared floor reading:** +3.1% of naive error against the declared point
value of +2.9% (pc_easy rate-matched, budget 256). Marginally above the point
value; inside the control's own range across its four readings (2.9 to 3.3%).

**Distribution comparison (not declared; the principled form of the floor
reading, added after both results were seen and labelled as such):** per-seed
relative penalty, Welch two-sample test against the twenty pc_easy control seeds
(mean +2.9%, sd 2.8):

| axis | excess | n | rel. penalty | vs control |
|---|---|---|---|---|
| OakInk2 scene x verb | −0.0001 | 40 | +3.0% | t = +0.05, p = 0.96 |
| GRAB shape x intent class | −0.0005 | 40 | +2.3% | t = −0.53, p = 0.60 |
| OakInk2 transitions | −0.0120 | 12 | +3.4% | t = +0.27, p = 0.79 |
| GRAB shape x fine intent (positive excess, null) | +0.0063 | 40 | +0.7% | t = −1.38, p = 0.17 |
| OakInk-Image category | +0.0238 | 70 | +15.2% | t = +10.6, p < 1e-12 |
| OakInk-Image affordance | +0.0309 | 130 | +12.2% | t = +9.8, p < 1e-12 |

**Reading.** Against zero, two of the three non-positive-excess axes are
"significant" at n = 40, and so is the zero-truth control itself at n = 20: the
paired penalty has a positive bias of about 3% of naive error, and a test
against zero at this n detects the bias, not composition. Against the control,
all three non-positive-excess axes are indistinguishable from it (p = 0.60 to
0.96) and every axis carrying difficulty is ten standard errors above it. The
necessity direction therefore holds on three axes across two datasets when
"carries difficulty" is read against the instrument's control, and fails on the
submitted wording ("distinguishable from zero"), which the revision must change.
Sufficiency remains false (GRAB shape x fine intent).

**Honesty note.** The Welch comparison was chosen after seeing that the point
threshold split 3.1% from 2.9%. It is reported alongside the declared readings,
not instead of them, and the revision text states which was declared.

## Table rows for the revision (computed 2026-09-13 from results.json, budget 256, perframe vs modular)

| axis | modular advantage (wins) | d | p | difficulty |
|---|---|---|---|---|
| GRAB shape x intent class | +0.00185 (25/40) | 0.25 | 0.129 | +0.00103 |
| OakInk2 scene x verb | −0.00097 (18/40) | −0.21 | 0.186 | +0.00051 |
| pc_easy rate-matched bundle, unmatched contrast (control, true zero) | +0.00154 (12/20) | 0.33 | 0.154 | +0.00258 |

Screen rows: GRAB shape x intent class left 0.028, right 0.009, excess −0.0005 (z −0.4);
OakInk2 scene x verb left 0.056, right 0.175, excess −0.0001 (z −0.1).
