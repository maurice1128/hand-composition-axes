# Pre-registration: an out-of-sample prediction on TACO

Written 2026-09-21 00:10, **while `Hand_Poses.zip` was still downloading and before any TACO file had been opened.**
Nothing about TACO's poses, clip lengths, label counts or grid is known to this document beyond what its paper and
README state.

## Why

The study's rule was found on three datasets, two of which share a collection effort, and a referee objected that
nothing validates it out of sample. TACO (Liu et al., CVPR 2024, arXiv:2401.08399) is a fourth, independent dataset:
2,317 bimanual sequences at 30 fps, MANO parameters per frame, each labelled with a `<tool, action, object>`
triplet. Its own authors evaluate a held-out-triplet split (S3) on motion forecasting, from single runs with no
calibration. It was not used in any way to form the rule.

## The prediction, stated before the data are seen

TACO's paper describes each sequence as one tool-use action named by its triplet. If that is so, a sequence's label
describes the motion in its scored windows, which is the condition this study found necessary.

**Prediction: TACO's confirmatory axis carries a compositional penalty above the zero-truth control.**

This prediction can fail, and the study already contains the way it can: GRAB satisfies the same condition and stays
at the control, so the condition is necessary and not sufficient. A null on TACO would therefore not refute the
rule as stated; it would be a second unexplained null, and **the paper would have to report that the rule has no
demonstrated predictive value out of sample.** A penalty would be the first out-of-sample confirmation. Both
outcomes are reported.

## What is measured before any training, and recorded here as an addendum

1. Clip duration: median and range in frames and seconds. **If the median sequence is longer than about 10 s or
   visibly contains several actions, the premise of the prediction is false and that is stated before the sweep is
   read**; the prediction is then "null expected, as for OakInk2 whole recordings", and it is recorded as changed
   *with the reason and the time*, before any model is trained.
2. DOF health and convention inference (flexion axis, sign, abduction axis), inferred from the data and logged, as
   for every other dataset. Right hand only, as elsewhere in the study.
3. The label grid for each candidate axis: cells, fine groups, candidate pool.

## The confirmatory axis is chosen by a rule fixed now, not by results

Candidate two-factor axes, in this order of preference: **action x tool**, action x object, tool x object. The fine
label is the full triplet, so a cell of `action x tool` spans the triplets that share that action and tool.

The confirmatory axis is the **first in that order** for which, at the largest number of held-out cells from
{5, 4, 3} that works: the label-leak gate passes, coverage fails on at most 10 of 40 seeds, the naive pool exceeds
the budget, and the frame-leak check shows at most 1% verbatim target frames per arm on five seeds. If none
qualifies, no sweep is run and the paper says TACO could not be tested and why. Axes later in the order are not run
as confirmatory tests; if run at all they are labelled exploratory.

## Settings

As every other Table 1 row: budget 256 (or the largest power of two the naive pool allows, stated if lower),
perframe prior, window 32, stride 4, 120 epochs, `--min-per-composition 5`, `--min-chains 4`, seeds 0-39.

## Readings

Statistic: per-seed penalty / naive mse_target x 100. Two-sided Welch test against the zero-truth control at the
same budget (`pc_easy_rerun`).

- **Prediction confirmed** if above the control at p < 0.05.
- **Prediction not confirmed** otherwise.
- If the permuted-grid control (`runs/PREREG_sham_grid.md`) shows that the synthetic control understates the bias on
  real data, TACO is read against the larger reference, as that document states. A permuted-grid control on TACO
  itself is run after the confirmatory sweep, whatever it shows.

No seeds are added or removed after results are seen.

## Addendum 1, 2026-09-21 00:45 — the data as found, recorded before any gate or model was run

From `runs/taco_build.json` (`scripts/build_taco_bundle.py`, right hand only).

- **Sequences:** 2,317 found, 2,317 kept, none dropped for any reason.
- **Duration: median 148 frames = 4.93 s** (min 2.27 s, max 16.57 s); 1.4% of sequences exceed 10 s. **The premise of
  the prediction holds** (median well under 10 s), so the prediction stands unchanged: a penalty above the control
  on the confirmatory axis. What the durations cannot show is whether a 5 s sequence holds one action or several;
  TACO's paper says one, and that is a description and not a measurement.
- **Conventions, inferred:** finger flexion about z, positive (PIP in range 100.0% against 16.7% for the other
  sign); finger abduction about y; thumb flexion about z, negative; thumb abduction about y, negative.
  **MANO's mean pose is not added:** 4.84% of values fall outside a limit without it and 14.54% with it on the
  smoke sample, which contradicts TACO's own loader and is logged in the build.
- **DOF health:** 1 of 27 unusable (`pinky_mcp_abd`, pinned at its upper limit in 87% of frames). The other bundles
  of this study have 1 to 6. 5.84% of values were outside a limit before clamping; projection residual is in the
  build report.
- **Vocabulary:** 15 actions, 17 tools, 9 target objects, 151 triplets.
- **Candidate axes** (cells; cells spanned by at least 4 triplets, i.e. the candidate pool):
  action x tool 42; 17. action x object 62; 10. tool x object 69; 8.
  The action-by-tool grid is 16% occupied, because most tools afford few actions. That is a property of the data
  and is stated here because it bears on how a result on that axis generalises.

The gates now run in the declared order, and the first axis and held count that pass are the confirmatory sweep.

## Addendum 2, 2026-09-21 01:05 — the axis the rule selected, recorded before any model was trained

Gate transcripts: `runs/gates/*taco_action_tool_h{5,4,3}.txt`, `runs/frame_leak_taco_action_tool.json`.

| axis | held | label leak | coverage fails | frame leak (5 seeds) |
|---|---|---|---|---|
| action x tool | 5 | exit 0 | 6 of 40 | 0.00% both arms |
| action x tool | 4 | exit 0 | 7 of 40 | not needed |
| action x tool | 3 | split cannot be drawn on most seeds | — | — |

**The rule selects action x tool at 5 held cells**, the first axis in the declared order at the largest held count
that works. `action x object` and `tool x object` were therefore not gated and are not confirmatory; if ever run they
are exploratory. The confirmatory sweep is `runs/taco_action_tool`, seeds 0-39, budget 256, queued at 01:05.

**The permuted-grid control cannot be run on this axis as written.** `scripts/experiment_paired_sham_grid.py`
refused: permuting the tool factor over the 151 triplets left at most 3 cells spanned by four triplets, against 17 in
the real grid, in all 200 draws. The real grid is concentrated (16% occupied) because a tool affords few actions, and
a factor permutation destroys that concentration, so it is not a matched control here. A control that preserves the
cell-size distribution exactly (reassigning whole triplets to cells) would be matched; it is a different control
from the one registered in `runs/PREREG_sham_grid.md` and will be declared separately before it is run. Until then
TACO is read against the synthetic zero-truth control, or against OakInk-Image's permuted-grid value if that turns
out to be the larger reference, as the Readings above state.

## Addendum 3, 2026-09-21 01:25 — TACO's own zero-truth control, declared before the confirmatory sweep had started

`runs/taco_action_tool` did not exist when this was written; the GPU was still on `sham_oakink_category`.

**The control.** `scripts/experiment_paired_sham_grid.py --sham-mode cells` reassigns whole triplets to cells by
shuffling the multiset of cell labels over the 151 triplets. The cells, the number of triplets in each, the candidate
pool (17) and each factor's marginal over triplets are **exactly** the real grid's; only which triplets share a cell
changes, so the trajectories the informed arm gains are no longer of the target's action and tool. It differs from
the factor permutation registered in `runs/PREREG_sham_grid.md`, which this grid cannot support (Addendum 2): here a
triplet's own action and tool no longer match its cell's in either factor, a stronger scrambling.

Dry run, 10 seeds, budget 256 (`runs/gates/sham_grid_dryrun_taco_action_tool.json`): label leak 0.000 in both
grids; coverage fails 0 of 10 (real) and 1 of 10 (sham); informed exposure 0.434 (real) and **0.493 (sham)**. The
sham arm is the *more* exposed, which works against this study's hypothesis, so no adjustment is made for it.

Settings: those of the confirmatory sweep with `--sham-mode cells --sham-seed 12345`, seeds 0-39, out
`runs/sham_taco_action_tool`.

**Readings.** Two-sided Welch tests on per-seed penalty / naive mse_target x 100.

- TACO's penalty is **compositional** if the sham grid is below the real grid (p < 0.05).
- It is **not** if the sham grid is not below the real grid: TACO's confirmatory result then shows a response to any
  structured holdout, and the out-of-sample prediction is reported as confirmed in size but not in kind.
- Whatever the outcome, the sham value becomes **TACO's reference in place of the synthetic control** if it is the
  larger of the two, and the confirmatory test is reported against both.

If the confirmatory sweep is at the control, this sweep is still run and reported, because a null against a
too-low reference and a null against the right one are different statements.
