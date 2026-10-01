# Pre-registration: a within-dataset zero-truth control (permuted label grid)

Written 2026-09-20, before any model of these sweeps is trained. Wrapper: `scripts/experiment_paired_sham_grid.py`.

## Why

Every penalty in this study is read against one synthetic zero-truth control, `pc_easy_rerun` (+2.59% at budget
256, SD 2.3). Nothing establishes that the paired design's bias is the same on a real dataset, where cell count,
group sizes, trajectory length, error scale and the informed arm's realised exposure all differ. If OakInk-Image's
own bias were +8%, most of Table 1 would evaporate. A referee raised this as the single most valuable missing
control and it is correct.

## The control

The second factor of the coarse label is permuted over **fine groups** (not trajectories, because whole fine groups
are assigned to one side of a split). Poses, lengths, trajectory count and each factor's marginal distribution are
untouched; the cell structure is close to the real grid's; the pairing carries no relationship.

Realised match, from the dry runs (3 seeds each, budget 256, `runs/gates/sham_grid_dryrun_*.json`):

| axis | grid | cells | candidate pool | cells covered | informed exposure | label leak |
|---|---|---|---|---|---|---|
| OakInk-Image category x intent | real | 100 | 25 | 5.00/5 | 0.182 | 0.000 |
| OakInk-Image category x intent | sham | 88 | 25 | 5.00/5 | 0.152 | 0.000 |
| GRAB shape x fine intent | real | 70 | 20 | 6.00/6 | 0.298 | 0.000 |
| GRAB shape x fine intent | sham | 76 | 20 | 5.67/6 | 0.405 | 0.000 |

**The exposure mismatch is declared now, before any result.** On OakInk-Image the sham arm is *less* exposed
(-0.030), which is a confound in the direction that favours this study's hypothesis: a lower sham penalty could in
principle be mechanical. On GRAB the sham arm is *more* exposed (+0.107), the opposite direction. Two things address
it, both declared here:

1. Exposure varies across seeds within each grid (0.121 to 0.223 on OakInk-Image's real grid). Per-seed penalty will
   be regressed on per-seed realised exposure **within the real grid**. If the slope is not distinguishable from
   zero, a 0.03 gap cannot explain a difference of the size Table 1 reports, and that is stated with the result.
2. The GRAB sham, with exposure 0.107 *above* its real grid, gives the opposite-signed test: if more exposure alone
   produced a penalty, the GRAB sham would show one.

## Settings

Identical to the sweeps they control, except for the permutation.

- `sham_oakink_category`: `--bundle data/bundles/oakink.npz --granularity oakink_category --held-compositions 5
  --min-chains 4 --min-per-composition 5 --budgets 256 --kinds perframe --window 32 --stride 4 --epochs 120`,
  seeds 0-39, `--sham-seed 12345`. Control for `oakink_official_category` + `oakink_category_rest` (+15.2%, n 69).
- `sham_grab_shape`: `--bundle data/bundles/grab.npz --granularity shape --held-compositions 6 --min-chains 5
  --min-per-composition 5 --budgets 256 --kinds perframe --window 32 --stride 4 --epochs 120`, seeds 0-39,
  `--sham-seed 12345`. Control for `grab_shape_s4` (+1.3%, n 40).

Before training: the label-leak gate must pass and the frame-leak check must show at most 1% of target frames
verbatim in each arm on five seeds. Coverage is recorded, failing seeds are kept, and an estimate excluding them is
reported beside the full one.

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100. Two-sided Welch tests on per-seed values.

**On OakInk-Image, the confirmatory test.**

- **The measurement is compositional** if the sham grid is below the real grid (p < 0.05). The real grid's +15.2%
  then reflects the factor pairing and not the act of holding out structured cells.
- **The measurement is not compositional** if the sham grid is not below the real grid. Table 1's OakInk-Image rows
  then measure a response to any structured cell holdout, the word "compositional" is withdrawn from the paper's
  claims, and the finding is reported as such. **This outcome would remove the paper's main result, and it is
  declared here as a real possibility.**
- Secondary, reported either way: the sham grid against the synthetic zero-truth control. If the sham sits at the
  synthetic control, the synthetic control is validated as a stand-in on real data and Table 1's readings stand as
  published. If the sham sits *above* it, the synthetic control understates the bias, and every axis in Table 1 must
  be re-read against the larger of the two. The sham value, not the synthetic one, is then the reference.

**On GRAB, a supporting test.** The sham grid is expected at the control. A sham penalty on GRAB despite its higher
exposure would show that exposure alone produces one, which would weaken the OakInk-Image reading; that is stated
with the result.

**What this cannot settle.** It does not test whether the prior composes, only what the measurement reads when
there is nothing to compose. It does not separate "unseen composition" from "unseen region of pose space" beyond
the extent to which a permuted grid holds out an equally structured region; a permuted cell is a union of fine
groups that need not be contiguous in pose space, and that is a difference from a real cell, stated as a limitation.

No seeds are added or removed after results are seen. Both sweeps are read at 40 seeds.

## Addendum, 2026-09-20 23:45 — written before either sham sweep had started

`runs/sham_oakink_category` did not exist and the GPU queue was still on `oakink2_transitions_rep` when this was
added. It revises point 1 above, because the condition written there turned out not to hold.

`scripts/axis_diagnostics.py` re-derived every seed's split for the nine Table 1 axes (1,760 checks over 423 seeds,
0 mismatches against stored records) and regressed per-seed penalty on per-seed realised exposure within each axis.
**The slope is not zero on OakInk-Image**, so point 1's "if the slope is not distinguishable from zero" does not
apply and the gap has to be priced instead:

| axis | r | p | slope (points per unit exposure) | what a 0.030 exposure gap predicts |
|---|---|---|---|---|
| OakInk-Image category x intent | +0.33 | 0.005 | +39.1 | **-1.2 points** |
| OakInk-Image affordance x intent | +0.39 | 4.5e-6 | +35.6 | -1.1 |
| OakInk-Image category x subject | +0.57 | 1.1e-4 | +49.6 | -1.5 |
| GRAB shape x fine intent | -0.10 | 0.53 | -7.5 | — |
| OakInk2 scene x primitive | -0.10 | 0.55 | -19.1 | — |

So on OakInk-Image the sham grid's lower exposure is expected to lower its penalty by about **1.2 points for a
mechanical reason**. The real grid reads +15.2% and the control +2.6%, a gap of 12.6. **The reading is therefore
revised, now and before any result:** the sham grid counts as below the real grid only if it is below it by more
than this 1.2 points, i.e. the Welch test is run on the sham values **plus 1.2**, and both the adjusted and the
unadjusted tests are reported. A sham penalty within 1.2 points of the real grid's is read as *not* below it.

Two observations that belong here because they were made before the result and bear on how it is read:

- The dose-response is itself what a real effect predicts: where a penalty exists, an informed arm given more
  examples of the held-out cells does better; where none exists (GRAB, OakInk2's two null axes) exposure has no
  relation to it.
- Exposure does not explain the pattern between datasets. GRAB's informed arm is the *most* exposed (0.30 to 0.32)
  and shows nothing; OakInk-Image's category axis (0.23) shows +15.2%; OakInk2's three axes share an exposure of
  0.11 to 0.12, and one of them reads +16.7% while two read the control. In absolute terms the pattern is sharper
  than in percentages, because the naive error is *smaller* on the null datasets (0.043 on GRAB, 0.017 on OakInk2,
  0.067 on OakInk-Image): the absolute penalty is 0.0104 on OakInk-Image category x intent against 0.0006 on GRAB
  and 0.0004 on OakInk2 scene x primitive.
