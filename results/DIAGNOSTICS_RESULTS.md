# Diagnostic results, read against their pre-registrations

Numbers from `scripts/analyze_diagnostics.py` (also `runs/diagnostics_summary.json`). The statistic is the per-seed
penalty divided by naive mse_target, times 100, at budget 256 on the perframe prior. The control is `runs/pc_easy_rerun`
(+2.59%, n 20). Welch tests are two-sample and two-sided. Each entry is written when its sweep completes.

## 1. GRAB planted interaction, stride 32 (H3, insensitivity) — complete 2026-09-16 06:24

Pre-registration: `runs/PREREG_grab_diagnostics.md`.

| sweep | % naive | sd | n | vs control | vs unmodified GRAB (stride 32) |
|---|---|---|---|---|---|
| GRAB shape x fine intent, unmodified (`grab_shape_v2`) | +0.71 | 9.16 | 40 | p 0.228 | — |
| GRAB planted (`grab_planted_s32`) | +13.15 | 7.27 | 40 | p 3.2e-11, diff +10.56 [+8.03, +13.08] | p 3.2e-9, diff +12.43 [+8.75, +16.11] |
| synthetic planted (`pc_hard_rerun`, reference) | +9.12 | 3.66 | 18 | p 4.7e-7 | — |

- **Reading, as declared:** above the control and above unmodified GRAB, both p < 0.05. The instrument can see a
  planted interaction on real GRAB poses, so H3 is rejected.
- **What that makes of GRAB's null:** it is evidence of no compositional difficulty of roughly this size, not an
  artefact of an insensitive instrument.
- **Magnitude (descriptive):** +13.2% on GRAB against +9.1% on the synthetic data at a matched frame-averaged
  offset RMS.
- **What this does not show:** the plant is constant over every frame of a trajectory. It therefore says nothing
  about dilution (H2), where real composition signal would be confined to part of a long trajectory.
- **Caveat:** 39 of 40 seeds had been seen (+12.95%) before the 40th finished. The reading did not change.

## 2. OakInk-Image category x subject (necessity) — complete 2026-09-16 08:39

Pre-registration: `runs/PREREG_oakink_category_subject.md`. Screen: excess -0.0005, z -0.11, i.e. no interaction.

| sweep | % naive | sd | n | vs control | vs category x intent |
|---|---|---|---|---|---|
| category x subject (`oakink_category_subject`) | +10.30 | 9.02 | 40 | p 5.9e-6, diff +7.71 [+4.67, +10.76] | p 0.006, diff -4.90 [-8.35, -1.44] |
| category x intent (69 seeds, both architectures) | +15.19 | — | 69 | p <1e-12 | — |
| control (`pc_easy_rerun`) | +2.59 | 2.3 | 20 | — | — |

- **Reading, as declared: necessity is refuted.** An OakInk-Image axis with no positive interaction carries
  difficulty well above the control. 37 of 40 seeds are positive; against zero p 1.1e-8.
- **Gate-failing seeds change nothing.** Excluding 12, 17 and 37 gives +10.82% (p 3.0e-6).
- **The dataset reading is supported.** Both OakInk-Image axes carry difficulty whatever their interaction; no GRAB
  or OakInk2 axis does, and the planted sweep shows the instrument would have seen it on GRAB.
- **What this does to the v7 argument.** "No axis without positive interaction carries difficulty above the control"
  is now false on this dataset, so the screen fails in both directions: sufficiency was already false (GRAB shape x
  fine intent, OakInk2 scene x primitive) and necessity is false here.
- **Caveats.** The fine label carries subject, so about 40% of target trajectories have the same object and intent
  recorded by another subject in both training sets; this is the left factor's main effect and reaches both arms.
  Subject is ambiguous in hand-over sequences. The magnitude is below category x intent's.

## 3. GRAB grasp-only (H2, dilution) — complete 2026-09-16 09:32

Pre-registration: `runs/PREREG_grab_diagnostics.md`. Bundle `grab_grasp.npz` keeps each trajectory's contact segment
only: 990 of 1048 trajectories, median 8.5 s down to 4.9 s, 64.2% of frames. Screen excess rose from +0.0063 (z 3.9)
to +0.0131 (z 6.8). Same stride (32) as the sweep it is compared with.

| sweep | % naive | sd | n | vs control | vs unmodified GRAB |
|---|---|---|---|---|---|
| unmodified (`grab_shape_v2`) | +0.71 | 9.16 | 40 | p 0.228 | — |
| grasp-only (`grab_grasp_s32`) | +2.22 | 7.69 | 40 | p 0.782, diff -0.37 [-3.02, +2.28] | p 0.428, diff +1.51 [-2.26, +5.27] |
| planted (`grab_planted_s32`) | +13.15 | 7.27 | 40 | p 3.2e-11 | p 3.2e-9 |

- **Reading, as declared: H2 is not supported at this resolution.** Grasp-only is indistinguishable from the
  unmodified sweep and from the control; 25 of 40 seeds are positive, p against zero 0.076.
- **The null is interpretable** because the planted sweep, at the same stride and bundle size, reads +13.15%.
- **What it does not rule out.** The kept segment is still 4.9 s against a 1.07 s window, so dilution *within* a
  grasp is untested. OakInk2's primitive-segment sweep (82 s down to 24.9 s) is queued and tests the same idea on the
  other dataset.
- **A rise in the screen did not predict a rise in difficulty**, which is consistent with the screen failing in both
  directions.

## 4. GRAB shape x fine intent re-run at stride 4 — complete 2026-09-16 11:09

Pre-registration: `runs/PREREG_stride4_reruns.md`. The v7 row was trained at stride 32, about one eighth of the
windows every other row had, and the manuscript stated no stride.

| sweep | % naive | sd | n | vs control | vs 0 |
|---|---|---|---|---|---|
| stride 32 (`grab_shape_v2`, the v7 row) | +0.71 | 9.16 | 40 | p 0.228, diff -1.87 [-4.96, +1.21] | 0.625 |
| stride 4 (`grab_shape_s4`) | +1.32 | 4.49 | 40 | p 0.155, diff -1.26 [-3.02, +0.49] | 0.070 |

Welch between the two strides: p 0.707. Both have 23 of 40 seeds positive.

- **Reading, as declared: the null holds at stride 4.** The table row is replaced by the stride-4 value, and the
  original stride is stated.
- **The setting was not the cause.** Eight times the windows changes neither the mean nor the sign count, so GRAB's
  shape x fine intent axis carries no difficulty despite positive interaction. The sufficiency counterexample stands.
- **The estimate is tighter.** Per-seed sd falls from 9.16 to 4.49 and the interval on the difference from the
  control narrows to [-3.02, +0.49], which excludes an OakInk-Image-sized effect (+12 to +18%) far more firmly.

## 5. OakInk2 primitive segments (dilution on the other dataset) — complete 2026-09-16 12:55

Pre-registration: `runs/PREREG_oakink2_primseg.md`, including the frame-budget note added before any seed finished.
`oakink2_primseg.npz` cuts each 82 s trajectory at OakInk2's own primitive spans: 2,128 segments, median 24.9 s,
79 cells, screen excess +0.0113 (z 16.4) against +0.0040 (z 2.9) whole.

| sweep | % naive | sd | n | vs control | vs whole trajectories |
|---|---|---|---|---|---|
| whole trajectories (`oakink2_scene_primitive`) | +2.30 | 7.29 | 40 | p 0.82 | — |
| primitive segments (`oakink2_primseg_s4`) | +8.97 | 12.93 | 20 | p 0.042, diff +6.38 [+0.26, +12.51] | p 0.042, diff +6.67 [+0.26, +13.08] |

- **Reading, as declared: dilution is supported on OakInk2.** Both comparisons clear p < 0.05, and against zero
  p 0.0059 with 15 of 20 seeds positive.
- **It is weak evidence, not strong.** Both p values sit just under 0.05 and both intervals reach down to +0.26,
  i.e. almost no difference. Per-seed sd is 12.93.
- **The frame budget works against this reading, not for it.** Each arm trains on roughly a third of the frames the
  whole-trajectory arms saw (naive error rises from 0.0180 to 0.0233), which makes difficulty harder to show.
  **[WRONG - superseded by sections 22 and 23: on OakInk2, fewer training frames RAISE the penalty (whole recordings
  +2.30% at 239k frames, +8.56% at 60k), so the reduced frame budget worked FOR this reading, not against it.]**
- **It disagrees with GRAB**, where grasp-only segments (8.5 s to 4.9 s) stayed null. The two differ in how far the
  cut goes: 82 s to 24.9 s still leaves ~6 windows per segment, while GRAB's cut removed only reach and release.
- **Open caveat from the build:** 42-46% of target segments share a recording with a training segment (naive 45.6%,
  informed 41.6%), which the leak gate cannot see. The asymmetry slightly favours the naive arm.

## 6. OakInk-Image ablation 2: GRAB-like DOF damage — complete 2026-09-16 14:25

Pre-registration: `runs/PREREG_oakink_ablations.md`. `oakink_dofdamage.npz` reproduces GRAB's damage on OakInk-Image
poses (thumb_cmc_flex pinned 63%, thumb_mcp_abd 27/13, thumb_cmc_abd 35%, pinky_dip_flex 28%, wrist_tz dead), so
`check_dof_health` counts 6 of 27 DOF unusable, matching GRAB. Labels, lengths and trajectory count are unchanged.

| sweep | % naive | sd | n | vs control | vs undamaged axis |
|---|---|---|---|---|---|
| undamaged (`oakink_official_category` + `_rest`) | +15.20 | 8.25 | 70 | — | — |
| DOF damaged (`pc_oakink_dofdamage`) | +13.70 | 8.86 | 40 | p 1.5e-9, diff +11.11 [+8.11, +14.11] | p 0.386, diff -1.49 [-4.91, +1.92] |

- **Reading, as declared: retargeting quality does not explain the difference.** The difficulty survives damage that
  matches GRAB's: 39 of 40 seeds positive, p against zero 4.9e-12, and the drop from the undamaged axis is within
  noise.
- **It strengthens the other datasets' nulls.** GRAB and OakInk2 cannot be null merely because 4 to 6 of their DOFs
  are pinned or dead, since the same damage leaves OakInk-Image's difficulty intact.
- **Side effect worth stating:** the clamp raises the naive error from 0.0673 to 0.0859, i.e. the damaged poses are
  harder to reconstruct, yet the penalty as a fraction of that error is unchanged.

## 7. OakInk-Image ablation 3: grid thinned toward GRAB's sparsity — complete 2026-09-16 15:36

Pre-registration: `runs/PREREG_oakink_ablations.md`. `oakink_sparse.npz` drops whole category x intent cells at
random: 49 of 100 cells kept, occupancy 76% to 49% (GRAB's 27% leaves too small a naive pool), 527 trajectories,
minimum naive pool 286.

| sweep | % naive | sd | n | vs control | vs full grid |
|---|---|---|---|---|---|
| full grid (`oakink_official_category` + `_rest`) | +15.20 | 8.25 | 70 | — | — |
| thinned (`pc_oakink_sparse`) | +19.64 | 6.50 | 40 | p 1.0e-20, diff +17.05 [+14.75, +19.35] | p 0.0024, diff +4.44 [+1.62, +7.27] |

- **Reading, as declared: sampling density does not explain OakInk-Image's difficulty.** The penalty survives
  thinning, with 40 of 40 seeds positive and p against zero 2.1e-21.
- **It moved the other way.** Thinning *raised* the penalty by 4.4 points, which is what one would expect if a
  sparser grid leaves the held-out pairing harder to infer from neighbouring cells. It is also consistent with the
  earlier finding that occupancy and interaction excess do not correlate (r = -0.095 over eleven axes).
- **Caveat that limits the rise, not the reading:** thinning removed 8 categories outright and a third of the
  trajectories, so part of the increase may be less data rather than sparser structure. Either way the difficulty
  did not disappear, which is what the ablation was asked to test.

## 8. OakInk-Image ablation 1: long clips, label misaligned — complete 2026-09-16 18:57 (half of a pair)

Pre-registration: `runs/PREREG_oakink_ablations.md`, which requires this to be read together with
`pc_oakink_longaligned` (started 18:57, due about 22:00) and which fixed 12 seeds for both.
`oakink_long.npz` concatenates 8 clips sharing a category but differing in intent, labelled by the first clip:
385 trajectories, median 743 frames, screen excess -0.0091 (z -4.6, `right_only` collapsed 0.045 to 0.003).

| sweep | % naive | sd | n | vs control | vs unmodified axis |
|---|---|---|---|---|---|
| unmodified (`oakink_official_category` + `_rest`) | +15.20 | 8.25 | 70 | — | — |
| long, misaligned (`pc_oakink_long`) | +8.74 | 18.70 | 12 | p 0.28, diff +6.15 [-5.75, +18.06] | p 0.263, diff -6.46 [-18.44, +5.53] |

- **No reading yet.** The pair is read together, and the aligned half is still running.
- **This half has no resolving power on its own.** It is indistinguishable from the control *and* from the
  undamaged axis, because the interval is 24 points wide. Per-seed sd is 18.70, driven by one seed at -47.5; the
  median is +12.84 and 11 of 12 seeds are positive, so the mean understates what most seeds did.
- **Do not read the drop from +15.20 to +8.74 as difficulty disappearing.** At twelve seeds this sweep could not
  have distinguished those two values either way.
- **Side effect:** concatenation lowers the naive error from 0.0673 to 0.0367, i.e. the long trajectories are easier
  to reconstruct.

## 26. GRAB at OakInk-Image's training volume - complete 2026-09-30 14:18, read 2026-10-01 10:00. VOLUME DOES NOT EXPLAIN GRAB'S NULL.

Pre-registration: `runs/PREREG_grab_volume_matched.md`, written before the sweep existed. `grab_shape_b64`: GRAB shape x
fine intent, budget 64 (about 4,400 training windows per arm, against OakInk-Image's 4,591 at 256), stride 4, seeds
0-39; coverage 40 of 40 passing. The runner and this sweep died at 2026-09-29 18:52 and resumed per row at 17 of 40 on
2026-09-30 09:18; no seed was added or removed.

| sweep | % naive | sd | > 0 | naive mse | vs control at 64 (+2.96 %) |
|---|---|---|---|---|---|
| GRAB shape x fine intent, budget 64 | +3.42 | 7.56 | 29/40 | 0.0476 | diff +0.46 [-2.16, +3.08], p 0.73 |

- **Reading, as declared: volume does not explain GRAB's null.** At OakInk-Image's training volume GRAB stayed at the
  control; the 95 % interval bounds the difference at +3.1 points.
- Secondary, descriptive: against GRAB at budget 256 (+1.32 %), diff +2.10 [-0.68, +4.88], p 0.14.
- Consequence for the paper: the `[PENDING: grab_shape_b64 ...]` sentence of Section 4.2 can be filled; the
  cross-dataset contrast is not explained by volume for GRAB (OakInk2's volume dependence, Appendix B, stands).

## 25. OakInk2 transitions, independent replication - complete 2026-09-30, read 2026-10-01 10:00. REPLICATES.

Pre-registration: `runs/PREREG_oakink2_transitions_replication.md`. Fresh seeds 100-139, settings identical to
`oakink2_transitions_s4`; coverage 0 of 40 failing.

| sweep | n | % naive | sd | > 0 | naive mse | vs control (+2.59 %) |
|---|---|---|---|---|---|---|
| replication | 40 | +12.20 | 10.03 | 36/40 | 0.0216 | diff +9.61 [+6.26, +12.96], p 6.2e-7 |
| original | 12 | +16.71 | 8.45 | 12/12 | 0.0229 | p 1.1e-4 |

- **Reading, as declared: replicates.** Descriptively the replication is 4.5 points below the original
  ([-10.55, +1.54], p 0.14). Pooled after the fact (52 seeds): +13.24 %, p 5.2e-10 against the control.
- Not settled, as declared: whether this is composition or task novelty.

## 24. OakInk-Image functional class x intent, independent replication - complete 2026-10-01, read 2026-10-01 10:00. REPLICATES.

Pre-registration: `runs/PREREG_oakink_class_replication.md`. Fresh seeds 100-139, settings identical to
`oakink_class_v1`; coverage fails 3 of 40 (seeds 112, 129, 138).

| sweep | n | % naive | sd | > 0 | naive mse | vs control (+2.59 %) |
|---|---|---|---|---|---|---|
| replication | 40 | +22.23 | 9.42 | 39/40 | 0.0801 | diff +19.64 [+16.47, +22.81], p 1.4e-16 |
| replication, coverage-passing | 37 | +22.87 | - | - | - | p 9.4e-16 |
| original | 12 | +17.92 | 14.0 | 11/12 | 0.0840 | p 0.0029 |

- **Reading, as declared: replicates.** Against OakInk-Image's permuted grid (+3.65 %) as well: diff +18.58
  [+14.91, +22.24], p 2.2e-15. Descriptively 4.3 points above the original ([-4.90, +13.52], p 0.33). Pooled after
  the fact (52 seeds): +21.24 %.
- Consequence for the paper: Table 2 can report the replication with its n and the original beside it, and the dagger
  on this row can be removed, as declared.

## 23. Rescue: primitive segments at the frame volume of whole recordings — complete 2026-09-29 15:18. NOT RESTORED; THE OAKINK2 DIRECTION IS WITHDRAWN.

Pre-registration: `runs/PREREG_oakink2_segments_matched_volume.md`, written before training. `oakink2_primseg_b896`:
segments at budget 896 per arm (about 239,000 frames, matching whole recordings at 256), fresh seeds 300-339, 40
seeds; label leak 0, coverage 0 of 40 failing, frame leak 0.00%.

| sweep | budget | frames per arm | % naive | naive mse |
|---|---|---|---|---|
| whole recordings | 256 | 239,225 | +2.30 | 0.0180 |
| **primitive segments** | **896** | **about 239,000** | **+5.19** (sd 8.08, 29/40 > 0) | 0.0196 |
| whole recordings | 64 | 59,806 | +8.56 | 0.0345 |
| primitive segments | 256 | 68,326 | +8.93 | 0.0233-0.0250 |

- **Confirmatory, as declared: segments@896 against whole@256, diff +2.89 [-0.54, +6.32], p 0.097.** Not above at
  p < 0.05, so **segmentation does not matter at matched volume, and the OakInk2 direction is withdrawn for good**, as
  the pre-registration states. No seeds will be added; that would be optional stopping.
- Secondary (approximate, no synthetic control at 896): against the budget-256 control, +2.60 [-0.17, +5.36],
  p 0.065.
- **Descriptive only, not a finding:** the point estimate lies in the predicted direction, and across the four rows the
  OakInk2 penalty falls as training volume rises whether or not the recordings are cut (about +8.6 to +8.9 near 60-70k
  frames, +2.3 to +5.2 near 239k). The simplest account of all four rows is data volume, with at most a small
  segmentation effect this design cannot resolve.
- **Consequence for the paper:** the causal evidence for the alignment condition is the OakInk-Image manipulation at
  matched budget (section 16), its window-class mechanism (section 20), the permuted-grid controls (sections 18, 21)
  and the TACO prediction (sections 19, 21). OakInk2 enters only descriptively (contamination 11.6%, purity 40%), and
  its whole-recording null is budget-dependent, not a demonstrated labelling artefact.

## 22. OakInk2 data-volume control — complete 2026-09-24 22:34. DATA VOLUME IS NOT EXCLUDED; THE SEGMENTATION RESULT IS CONFOUNDED.

Pre-registration: `runs/PREREG_oakink2_data_volume.md`. `oakink2_scene_primitive_b64` = the null whole-recording
sweep with the budget cut from 256 to 64 recordings, so each arm sees 59,806 frames against the segment sweeps'
68,326. Labels, grid and poses unchanged. 40 seeds; label leak 0; coverage fails 2 of 40 (seeds 0, 21).

| sweep | budget | frames per arm | % naive | vs its control |
|---|---|---|---|---|
| whole recordings (`oakink2_scene_primitive`) | 256 | 239,225 | +2.30 | p 0.82 |
| **whole recordings (`oakink2_scene_primitive_b64`)** | **64** | **59,806** | **+8.56** (sd 12.9, 26/40 > 0, naive mse 0.0345) | **+5.60 [+1.35, +9.85], p 0.011** (control b64 +2.96) |
| primitive segments, replication (`oakink2_primseg_rep`) | 256 | 68,326 | +8.93 | p 0.0018 |
| primitive segments, recording-disjoint | 256 | fewer | +7.14 | p 0.027 |

- **Reading, as declared: data volume is NOT excluded.** Whole recordings on fewer frames are above their control.
  Excluding the two coverage-failing seeds: +9.15%, p 0.007.
- **The declared remainder is nil.** Whole recordings at 64 against segments at 256 (matched frames): diff -0.36
  [-5.84, +5.11], p 0.90. At matched data volume, cutting recordings at primitive boundaries adds nothing measurable.
  Whole at 64 against recording-disjoint segments: +1.43 [-4.17, +7.02], p 0.61.
- **Consequence for the paper (declared): the OakInk2 "fix" direction is withdrawn as evidence for label alignment.**
  The rise from +2.30% to +8.93% is reproduced by shrinking the training set alone. What survives on OakInk2 is only
  descriptive: the motion-level contamination and purity of Table 3 (11.6%, 40%), which are measurements, not a
  causal test.
- **It also changes how OakInk2's whole-recording null reads.** Whole recordings show a penalty at 64 and none at 256
  (diff +6.27, p 0.0097). So their null at 256 is budget-dependent, and it can no longer be attributed to labelling
  from the data at hand.
- **What still stands:** the OakInk-Image manipulation (misaligned vs aligned at the SAME budget, same trajectory
  counts and similar lengths; section 16), its window-class mechanism (section 20), both permuted-grid controls
  (sections 18, 21) and the TACO prediction (section 19). The evidence for the alignment condition is now ONE
  controlled direction plus an out-of-sample prediction, not two directions.
- Declared limits of this control: it matches frames, not trajectory count (64 against 256), and the informed arm can
  swap at most 32 held-cell recordings; the grid differs from the segments' (62 against 79 cells).
- **A test that could restore the fix direction (not run, not pre-registered):** segments at a frame budget matched
  to whole@256 (about 896 segments of mean 267 frames; naive pools 1,842-1,991 allow it; informed-extra pools of
  69-143 cap exposure near whole@256's 0.12). A penalty there, above whole@256, would show segmentation matters at
  matched volume. Cost: roughly four times the frames of the segment sweeps.

## 21. TACO's own zero-truth control (cell-size-preserving permutation) — complete 2026-09-24 07:17. TACO'S PENALTY IS COMPOSITIONAL.

Pre-registration: addendum 3 of `runs/PREREG_taco_prediction.md`, written before `runs/taco_action_tool` existed.
Wrapper `scripts/experiment_paired_sham_grid.py --sham-mode cells`: the multiset of cell labels is shuffled over the
151 triplets, so the 42 cells, the triplets per cell and the candidate pool (17) are exactly the real grid's
(sidecar confirms). 40 seeds, budget 256.

| grid | % naive | sd | n | > 0 | naive mse | informed mse |
|---|---|---|---|---|---|---|
| TACO action x tool, REAL | +10.63 | 8.33 | 40 | 33/40 | 0.0321 | 0.0286 |
| TACO action x tool, PERMUTED CELLS | **-3.39** | 6.46 | 40 | 10/40 | 0.0293 | 0.0303 |
| synthetic zero-truth control | +2.59 | 2.30 | 20 | 17/20 | — | — |

- **Confirmatory, as declared: the permuted grid is below the real grid**, diff -14.01 [-17.34, -10.69], p 2.3e-12.
  TACO's penalty reflects the action-tool pairing, not the act of holding out structured cells. The out-of-sample
  prediction is therefore confirmed **in kind as well as in size**.
- **Reference, as declared: the larger of the two.** The sham reads *below* the synthetic control (diff -5.97,
  p 2.9e-6), so the synthetic control (+2.59%) stays TACO's reference and the confirmatory reading of section 19
  (+8.04 above it, p 6.9e-7) stands unchanged. Using the sham as reference would have enlarged the effect; the
  declared rule prevents that.
- **Why the sham reads below zero (observation, not tested).** In the sham arm the informed prior is *worse* than the
  naive one (0.0303 against 0.0293). The cell-mode permutation scrambles both factors, so the trajectories the
  informed arm swaps in are unrelated triplets, and they displace training trajectories that share an action or a
  tool with the targets; the sham arm is also the more exposed (0.493 against 0.434 in the dry run). This differs from
  the factor-mode sham on OakInk-Image (+3.65%, at the synthetic control), which permutes one factor only. **The
  cell-mode sham is therefore a conservative null, not a zero; do not report it as TACO's zero.**
## 20. Purity or contamination? The two-clip pair scored by window class — complete 2026-09-22 09:05. BOTH ACT; CONTAMINATION CARRIES MOST.

Pre-registration: `runs/PREREG_window_classes.md`. Wrapper: `scripts/experiment_paired_window_classes.py`. Same seeds
0-39 and settings as `pc_oakink_pair_{aligned,misaligned}` (budget 64); sidecars `runs/pc_oakink_pair_*_wc/window_classes.json`.

**Reproduction check passed:** totals +16.97% (original +17.03) and +3.92% (original +3.87); largest per-seed
difference 2.4 points (GPU non-determinism). No seed had fewer than 20 windows in any class.

| bundle | first clip (labelled) | second clip | straddle | total |
|---|---|---|---|---|
| aligned | +17.75 (sd 12.6, 36/40 > 0) | +16.07 (11.7, 38/40) | +16.51 | +16.97 |
| misaligned | **+5.64** (6.7, 31/40) | **+1.54** (7.6, 22/40) | +5.51 | +3.92 |

- **Sanity:** A1 and A2 do not differ (paired diff +1.68, p 0.079), so position inside a joined trajectory does not
  matter by itself.
- **M1 is below A1:** Welch diff -12.11 [-16.62, -7.60], p 1.3e-6. The windows the label *does* describe lost
  three quarters of the penalty.
- **M2 is below M1:** paired diff -4.10, p 0.0057. Windows the label does not describe carry less still.
- **Reading, as declared: both act.** Descriptively, (A1 - M1) = 12.1 points against (M1 - M2) = 4.1, so the fall is
  carried about three to one by what happened to the *training arms*, not by scoring impure windows. **The second
  referee's guess (an 8.4% dose is too small, so purity must do the work) is refuted by measurement**; the draft's
  original mechanism sentence ("the naive arm sees the held-out motion under other labels") is the better-supported
  one, with the caveat below.
- M1 against the budget-64 control: +2.68 [+0.31, +5.05], p 0.028, a small residual penalty on the labelled
  windows; M2 at the control (p 0.28).
- **What "contamination" cannot be separated from here.** M1 < A1 shows the arms' training data changed, not which
  arm. In the misaligned bundle the informed arm's swapped trajectories are held-cell motion in their first clip
  only, so the informed arm is *diluted* by construction as surely as the naive arm is contaminated (8.4% of its
  clips). Both are consequences of a label that describes only part of a trajectory; the draft should name both.

## 19. TACO, the out-of-sample prediction — complete 2026-09-22 07:09. PREDICTION CONFIRMED.

Pre-registration: `runs/PREREG_taco_prediction.md` (written before any TACO file was opened; three addenda before
any model trained). Axis chosen by its rule: action x tool, 5 held cells, budget 256, seeds 0-39.

| sweep | % naive | sd | n | > 0 | naive mse | vs synthetic control (+2.59%) |
|---|---|---|---|---|---|---|
| `taco_action_tool` | **+10.63** | 8.33 | 40 | 33/40 | 0.0321 | diff +8.04 [+5.20, +10.88], **p 6.9e-7** |
| excluding 6 coverage-failing seeds (17, 21, 22, 26, 32, 37) | +10.76 | — | 34 | — | — | diff +8.17 [+5.19, +11.15], p 1.9e-6 |

- **Reading, as declared: confirmed.** A dataset the rule was never fitted to, whose labels each name one tool-use
  action over a median 4.9 s clip, carries a penalty above the control. Against zero p 7.7e-10. This is the first
  out-of-sample test of the rule and it held.
- **In size** it sits between OakInk-Image's category axis (+15.2%) and category x subject (+10.3%), and above every
  GRAB and OakInk2 whole-recording axis (+1.3 to +3.0%).
- **Not yet settled:** whether TACO's penalty is compositional and what TACO's own zero is. `sham_taco_action_tool`
  (cell-size-preserving permutation, addendum 3) is running; it is read when it reaches 40 and may replace the
  synthetic control as TACO's reference, as declared.
- The prediction had a registered way to fail (a second GRAB-like null) and did not.

## 18. Within-dataset zero-truth control (permuted label grid) — complete 2026-09-21 17:53. THE QUANTITY IS COMPOSITIONAL.

Pre-registration: `runs/PREREG_sham_grid.md`, including its addendum written before either sweep started. Wrapper:
`scripts/experiment_paired_sham_grid.py`. Both sweeps 40 seeds, budget 256, perframe, label leak 0.000 on all 40
splits (`runs/gates/sham_grid_cover_*_full.json`).

| grid | % naive | sd | n | > 0 | naive mse | informed exposure (40 splits) |
|---|---|---|---|---|---|---|
| synthetic zero-truth control | +2.59 | 2.30 | 20 | 17/20 | 0.0878 | — |
| OakInk-Image category x intent, REAL | +15.19 | 8.31 | 69 | 66/69 | 0.0672 | 0.214 |
| OakInk-Image category x intent, PERMUTED | **+3.65** | 6.79 | 40 | 27/40 | 0.0625 | 0.208 |
| GRAB shape x fine intent, REAL | +1.32 | 4.49 | 40 | 23/40 | 0.0434 | 0.302 |
| GRAB shape x fine intent, PERMUTED | **+1.38** | 6.76 | 40 | 24/40 | 0.0442 | 0.377 |

- **Confirmatory test, OakInk-Image: the permuted grid is below the real grid**, diff -11.53 [-14.45, -8.62],
  p 5.8e-12; with the declared +1.2 exposure adjustment, diff -10.33 [-13.25, -7.42], p 2.9e-10. **As declared, the
  measurement is compositional:** the real grid's penalty reflects the factor pairing and not the act of holding out
  structured cells. The outcome that would have withdrawn the word from the paper did not occur.
- **Secondary, declared: the permuted grid sits at the synthetic control**, diff +1.07 [-1.32, +3.45], p 0.37. The
  synthetic control is therefore validated as a stand-in on real data, and Table 1's readings stand as they are. The
  interval allows a within-dataset bias up to 3.5 points above the synthetic one and no more.
- **GRAB, supporting test:** permuted +1.38 against real +1.32, diff +0.06 [-2.50, +2.62], p 0.96, and against the
  control p 0.31, with the permuted informed arm the *more* exposed (0.377 against 0.302). More exposure alone does
  not produce a penalty.
- **Coverage.** Permuted OakInk-Image fails 4 of 40 (seeds 6, 8, 28, 35); excluding them +3.84% (n 36), p 0.32
  against the control. Permuted GRAB fails 0 of 40.
- **The exposure gap was smaller than feared.** Over all 40 splits it is 0.214 (real) against 0.208 (permuted) on
  OakInk-Image; the 0.030 of the three-seed dry run was sampling noise. The declared 1.2-point adjustment therefore
  over-corrects, and both tests are reported.
- **What this changes for `category x subject`.** That axis (+10.30%) was described as showing that the measurement
  "responds to any structured cell holdout". The permuted grid is a structured cell holdout with no real pairing and
  reads at the control, so that description was **too strong**. Category x subject is a real pairing (how a given
  subject grasps a given category); what remains true is that it is not a composition of *action* factors.
- **Not settled by this:** a permuted cell is a union of fine groups that need not be contiguous in pose space, so
  "unseen composition" is not separated from "unseen contiguous region of pose space". Stated as declared.

## 17. OakInk2 transitions at stride 4 — complete 2026-09-20 22:11. OAKINK2 IS NOT UNIFORMLY NULL.

Pre-registration: `runs/PREREG_stride4_reruns.md`. Same settings as `oakink2_paired_v2` (granularity fine, held 6,
min_chains 6, budget 256, seeds 0-11) except stride 4 instead of 16, perframe only.

| sweep | % naive | sd | n | positive | vs control (+2.59%) |
|---|---|---|---|---|---|
| stride 16 (`oakink2_paired_v2`) | +3.44 | 6.98 | 12 | — | p 0.69 |
| stride 4 (`oakink2_transitions_s4`) | +16.71 | 8.45 | 12 | 12/12 | diff +14.12 [+8.69, +19.55], p 1.1e-4 |

Stride 4 against stride 16: diff +13.26 [+6.69, +19.83], p 4.0e-4. Against zero p 2.8e-5. Naive error 0.0229
against 0.0219, so the arms are not harder to fit; the informed arm gains more.

- **Reading, as declared: above the control.** The stride-16 null was an artefact of training on a quarter of the
  windows. The pre-registration's wording ("necessity is refuted") belongs to the abandoned interaction-screen
  argument; what matters now is below.
- **The dataset-level statement is false.** "No OakInk2 axis carries a penalty" cannot be written. OakInk2 carries
  one on this axis with whole 82 s recordings, no cutting.
- **How this axis differs, verified in code.** Here a recording's label is its whole primitive chain
  (`grip->rearrange->take_outside->...`) and a composition is a consecutive pair (`transitions_of`). On the two null
  OakInk2 axes a recording is labelled `<scene> x <the chain's FIRST primitive>` or `<scene> x <verb>`
  (`coarsen_labels`), i.e. one item standing for the whole recording. So on the transitions axis the label does
  describe what the recording contains, and a held-out transition really is absent from the naive arm's recordings.
- **That reading is post hoc.** The axis was not designed as a test of alignment, n is 12, and an alternative is
  open: a held-out transition may be close to a held-out *task*, so the penalty could be task novelty rather than
  composition. It is consistent with the alignment account; it does not test it.
- **Consequence for other rows.** Only two sweeps ever ran off stride 4: this one (changed) and GRAB shape x fine
  intent (re-run at stride 4, unchanged at +1.32%). Every other reported row is already at stride 4.

## 16. Alignment test v4, no constituent leak — complete 2026-09-20 18:45. THE PRIMARY OAKINK-IMAGE CONTRAST.

Pre-registration: `runs/PREREG_oakink_alignment_v4.md`. Two-clip trajectories, 334 each over the same 33 categories
(per-category counts within 1), budget 64, 40 seeds. In the aligned bundle both clips share one `object->intent`, so
the split's fine-label disjointness covers every constituent: constituent leak 0 in both arms, frame leak 0.00%.

| sweep | % naive | sd | positive | vs control (+2.96%) |
|---|---|---|---|---|
| unmodified baseline (`pc_oakink_b64`) | +10.15 | 6.87 | 36/40 | p 2.1e-7 |
| aligned pair (`pc_oakink_pair_aligned`) | +17.03 | 11.53 | 37/40 | (a) diff +14.06 [+10.24, +17.88], p 2.5e-9 |
| misaligned pair (`pc_oakink_pair_misaligned`) | +3.87 | 5.41 | 31/40 | (c) diff +0.91 [-1.10, +2.91], p 0.37 |

- (b) misaligned below aligned: diff -13.15 [-17.19, -9.12], p 2.1e-8. **Met.**
- (d) aligned not below the baseline: it is above it, diff +6.88 [+2.64, +11.12], p 0.0019. **Met.**
- misaligned against the baseline: diff -6.27 [-9.03, -3.52], p 2.2e-5.

**Reading, as declared: alignment is supported in its stronger form**; all four conditions hold.

Excluding the coverage-failing seeds (aligned 3, 29; misaligned 8, 15, 18, 22, 25; baseline 8, 13, 29): aligned +16.46
(n 38), misaligned +4.89 (n 35), baseline +9.72 (n 37). Condition (b) holds (diff -11.58, p 3.5e-7). Condition (c)
becomes marginal: misaligned against the control is +1.92 [-0.05, +3.89], p 0.056. So on gate-passing seeds the
misaligned penalty is close to the control but may sit slightly above it; the full-sweep statement "equal to the
control" should be written as "not distinguishable from the control".

What this settles about the audit's concern: removing the constituent leak changed the aligned value from +17.86%
(version 2) to +17.03%, so that leak was not what raised aligned above the baseline. A likelier reason, not tested,
is that a budget counted in trajectories gives the informed arm twice as many held-cell clips when each trajectory
holds two. The paper reports the numbers and does not assert this.

As pre-registered, the misaligned bundle's mechanism numbers go with the result: 8.4% of the naive arm's training
clips are held-cell motion, 27.9% of target trajectories contain a clip whose `object->intent` the naive arm has
seen, and 61% of target clips are held-cell motion (aligned: 0%, 0%, 100%).

## 15. Clean-up B: recording-disjoint OakInk2 primitive segments — complete 2026-09-20 15:34

Pre-registration: `runs/PREREG_cleanup_runs.md`, part B. `scripts/experiment_paired_recording_disjoint.py` removed
from both training pools every segment whose recording supplied a target segment, so target segments sharing a
recording with a training segment went from about 45% to 0.0 in both arms. Fresh seeds 200-239, budget 256.

| sweep | % naive | sd | n | positive | vs control (+2.59%) | vs whole recordings (+2.30%) |
|---|---|---|---|---|---|---|
| segments, replication (section 13) | +8.93 | 11.61 | 40 | 33/40 | p 0.0018 | p 0.0032 |
| segments, recording-disjoint | +7.14 | 12.18 | 40 | 29/40 | p 0.027, diff +4.55 [+0.53, +8.57] | p 0.035, diff +4.84 [+0.36, +9.33] |

- **Reading, as declared: the caveat is removed.** With no recording shared between targets and training, primitive
  segments are still above both the control and whole recordings (each p < 0.05), and against zero p 6.5e-4.
- **It is the weaker of the OakInk2 results and should be reported as such.** The penalty is 1.8 points below the
  replication, a difference that is not itself distinguishable (p 0.50), and both lower interval bounds are close to
  zero (+0.53 and +0.36). Shared recordings may have contributed a little; they did not produce the effect.
- Naive error is 0.0250 against 0.0238 for the replication, the expected cost of the smaller, filtered pools in both
  arms.

## 14. Clean-up A: category-matched misaligned clips — complete 2026-09-20 05:09

Pre-registration: `runs/PREREG_cleanup_runs.md`, part A. `oakink_long3_misaligned_cm.npz` repeats the misaligned
construction inside the aligned bundle's 14 categories: 210 trajectories, per-category counts within 2 of the aligned
bundle's, 0.00% frame leak. Budget 64, 40 seeds.

| sweep | % naive | sd | positive |
|---|---|---|---|
| aligned (`pc_oakink_long3_aligned`) | +17.86 | 8.77 | 40/40 |
| misaligned, category-matched (`pc_oakink_long3_misaligned_cm`) | +3.05 | 4.51 | 32/40 |
| misaligned, original 33 categories | +3.03 | 5.98 | 26/40 |
| zero-truth control at budget 64 | +2.96 | — | — |

- Against aligned: diff -14.81 [-17.93, -11.69], p 1.9e-13. Against the control: diff +0.09 [-1.68, +1.86], p 0.92.
- **Reading, as declared: the caveat is removed.** With the two bundles drawn from the same 14 categories in matched
  proportions, misaligned labels still put the penalty exactly at the control, and the result is indistinguishable
  from the original misaligned sweep (diff +0.02, p 0.99). The category mix played no part in section 11's contrast.
- The paper should report this pair (aligned against category-matched misaligned) as the primary contrast.

## 13. OakInk2 primitive-segment result REPLICATES on fresh seeds — complete 2026-09-18 19:21, read 2026-09-20

Pre-registration: `runs/PREREG_oakink2_primseg_replication.md`. Seeds 100-139, disjoint from the original sweep's
0-19; label-leak gate passed and 0.00% frame leak before training. Settings identical to section 5.

| sweep | % naive | sd | n | vs control (+2.59%) | vs whole trajectories (+2.30%) |
|---|---|---|---|---|---|
| original (`oakink2_primseg_s4`) | +8.97 | 12.93 | 20 | p 0.042 | p 0.042 |
| replication (`oakink2_primseg_rep`) | +8.93 | 11.61 | 40 | p 0.0018, diff +6.34 [+2.50, +10.18] | p 0.0032, diff +6.63 [+2.30, +10.96] |

- **Reading, as declared: it replicates.** Both comparisons clear p < 0.05 on the replication alone; 33 of 40 seeds
  positive, p against zero 1.9e-5. The two independent means agree to 0.04 points.
- Secondary, pooled after the fact (n 60): +8.94%, p 2.1e-4 against the control, p 8.3e-4 against whole trajectories.
- The direction "relabel a null dataset so each segment carries its own label, and difficulty becomes measurable" is
  no longer weak evidence. Unchanged caveats: each arm trains on about a third of the whole-trajectory frames, and
  about 45% of target segments share a recording (not frames) with a training segment.

## 12. Alignment repeated with a second prior (modular bank) — complete 2026-09-17 07:31, read 2026-09-18 11:10

Pre-registration: `runs/PREREG_alignment_v2_modular.md`. Same three bundles and settings as section 11, budget 64,
40 seeds, `--kinds modular`. Collapse check on all 240 models: final `val_primitives_used` minimum 10.0 of 12,
median 12.0, none below 2, so the bank is a genuine second architecture in every sweep.

| sweep (modular) | % naive | sd | positive | vs 0 | perframe value |
|---|---|---|---|---|---|
| baseline `pc_oakink_b64_modular` | +7.79 | 7.68 | 35/40 | p 1.4e-7 | +10.15 |
| aligned | +6.40 | 24.22 | 25/40 | p 0.103 | +17.86 |
| misaligned | -4.74 | 23.26 | 17/40 | p 0.205 | +3.03 |

Declared contrasts:
- (a) misaligned below aligned: diff -11.15 [-21.72, -0.57], p 0.039. **Met.**
- (b) aligned not below the baseline: diff -1.38 [-9.47, +6.70], p 0.732. **Met.**
- misaligned against the baseline: diff -12.53 [-20.32, -4.74], p 0.0022.

**Reading, as declared: it replicates.** The direction is the same as the perframe prior's, so the alignment effect
is not a property of one architecture.

**It is much weaker evidence than section 11, and must be reported that way.**
- Per-seed sd on the long bundles is 23 to 24 against the perframe prior's 8.8 and 6.0, so (a) only just clears 0.05.
- The aligned sweep is not itself distinguishable from zero (p 0.103), i.e. this prior barely measures difficulty on
  concatenated clips at all, against +17.86% for the perframe prior.
- The misaligned sweep's mean is negative (-4.74%), which the perframe prior never produced.
- ~~The modular zero-truth control was still running.~~ **Read 2026-09-18 11:40: the modular control makes the
  instrument unusable at this budget.** `pc_easy_rerun_modular` (20 seeds, bank not collapsed, minimum 12 of 12)
  reads -0.37% with a per-seed sd of **21.82**, against 2.3 for the perframe control. Against it, none of the three
  modular sweeps is distinguishable: baseline diff +8.15 [-2.29, +18.60] p 0.119; aligned +6.77 [-5.75, +19.29]
  p 0.281; misaligned -4.37 [-16.72, +7.97] p 0.478. Even the unmodified baseline cannot be shown to sit above a
  true zero for this prior.
- **Final status of section 12.** The within-prior contrast (a) holds, so the sign of the alignment effect is the
  same under a second architecture. But the modular prior at budget 64 cannot read any sweep against zero, so this
  section is a consistency note, not independent evidence. The paper's alignment claim rests on section 11 alone.
  Whether the bank would become a usable instrument at budget 256 is untested here.

## 11. ALIGNMENT CONFIRMED on OakInk-Image (test v2, no clip reuse) — complete 2026-09-17 00:39

Pre-registration: `runs/PREREG_oakink_alignment_v2.md` (design, 22:15 coverage amendment and 22:25 build caveats all
recorded before any v2 result). Bundles from `scripts/build_oakink_alignment_v2.py`, gated by
`scripts/gate_alignment_v2.py`; 0.00% of target frames verbatim in either arm. Budget 64, 40 seeds each.

| sweep | % naive | sd | positive | vs control at budget 64 (+2.96%) |
|---|---|---|---|---|
| `pc_oakink_b64`, unmodified baseline | +10.15 | 6.87 | 36/40 | p 2.1e-7, diff +7.18 [+4.76, +9.60] |
| `pc_oakink_long3_aligned` | +17.86 | 8.77 | 40/40 | p 1.7e-13, diff +14.90 [+11.92, +17.87] |
| `pc_oakink_long3_misaligned` | +3.03 | 5.98 | 26/40 | p 0.951, diff +0.07 [-2.10, +2.23] |

Declared contrasts:
- (a) misaligned below aligned: diff -14.83 [-18.18, -11.48], p 6.0e-13. **Met.**
- (b) aligned not below the baseline: aligned is above it, diff +7.72 [+4.21, +11.23], p 3.8e-5. **Met.**
- misaligned against the baseline: diff -7.11 [-9.98, -4.25], p 4.5e-6.

**Reading, as declared: alignment is the property.** When the label describes every window of a trajectory,
OakInk-Image's compositional difficulty is kept (and rises); when two thirds of each trajectory shows other intents
under the same label, the difficulty falls exactly to the zero-truth control. The misaligned trajectories are longer
(median 273 frames against 206), so length is not what removes it.

Together with section 5 this is evidence in both directions:
- OakInk2 whole tasks, one label per 82 s, are null; cut so each segment carries its own primitive label, they
  measure difficulty (+2.30% to +8.97%, weak).
- OakInk-Image clips measure difficulty; mislabelled by concatenation, they are null (+17.86% to +3.03%, strong).

Caveats, all stated in the pre-registration before the result:
- The misaligned bundle's label carries almost no intent information (screen `right_only` 0.006). Here
  "the label does not describe the window" and "the label is uninformative about the window" coincide, and this
  design cannot separate them.
- Category coverage differs, 14 against 33, as do cell counts (38 against 59), lengths, and source clips.
- The aligned rise above the baseline may partly reflect three same-composition clips per trajectory at a fixed
  trajectory budget; misaligned trajectories carry as many frames and lose the difficulty entirely, so frame count
  alone does not produce it.
- Budget 64, not the confirmatory 256.

## 10. The long-clip pair is VOID: both bundles leak frames — 2026-09-16 22:10

The aligned half read +57.24% of naive error over 11 of 12 seeds (sd 9.09, all positive), almost four times the
undamaged axis. That size was implausible, so before any reading a frame-level check was run
(`scripts/check_frame_leak.py`, `runs/frame_leak_long_ablations.json`). It rebuilds each split exactly as the sweep
does and counts the target frames that appear verbatim in each arm's training trajectories (seeds 0-4):

| bundle | target frames in naive training | in informed training | gap |
|---|---|---|---|
| unmodified `oakink.npz` | 0.0% | 0.0% | +0.0 |
| `oakink_long.npz` (misaligned) | 78.4% | 88.9% | +10.4 |
| `oakink_longaligned.npz` (aligned) | 0.0% | 81.2% | +81.2 |

- **Both halves are void.** The aligned sweep's informed arm had seen four fifths of the target frames and the naive
  arm none, so +57% measures memorisation, not composition. The misaligned sweep's two arms both saw most target
  frames, so its +8.74% is not interpretable either.
- **Cause: an error in the ablation design, not in the instrument.** The builders concatenated 8 reused source
  clips per trajectory (mean reuse about 4, and for the aligned version a median cell holds only 2 source clips), so
  the same clip appears inside many trajectories. `check_composition_leak.py` compares fine labels only and cannot
  see this; its PASS on both bundles was true and irrelevant. The build report had flagged frame recurrence for
  `oakink_long` and it was not followed up before sweeping.
- **Consequence: label-to-window alignment is still untested on OakInk-Image.** Nothing in this pair supports or
  refutes it. The earlier statement that the pair "may not resolve anything" understated the problem.
- **What a valid version needs:** no source clip may appear in more than one trajectory, and the frame-level check
  above must show near-equal rates for both arms before any seed is trained. At OakInk-Image's size that is not
  possible at a budget of 256: 770 clips give at most 96 non-overlapping 8-clip trajectories, and an aligned cell
  holds a median of 2 clips. A valid test needs a smaller budget or a smaller concatenation factor, declared anew.
- **Every other bundle used as evidence today is clean at the frame level** (same check, seeds 0-4, 0.0% of target
  frames in either arm's training set): unmodified OakInk-Image, OakInk2 whole trajectories, OakInk2 primitive
  segments, GRAB grasp-only, OakInk-Image DOF-damaged, OakInk-Image thinned, and OakInk-Image category x subject.
  So the OakInk2 primitive-segment result (+2.30% to +8.97%) stands as the only, and weak, evidence for alignment;
  its known caveat is shared recordings (about 45% of target segments), not shared frames.

## 9. A training-free surrogate for the whole experiment — FAILED, 2026-09-16 20:05

`scripts/surrogate_penalty.py`, results in `runs/surrogate_penalty.json` and `runs/gates/surrogate_penalty.txt`.
The idea was to skip training: predict each target window from training trajectories by nearest centroid, once
additively (left mean + right mean - global mean, the naive arm's best linear guess) and once from the held cell's
other trajectories (the informed arm's), and read the difference as a surrogate penalty. It was defined before
being compared with the later results, so stage 2 below is a genuine held-out test.

| stage | n | Pearson | Spearman | MAE |
|---|---|---|---|---|
| 1, the nine originally swept axes (descriptive) | 9 | +0.41 (p 0.27) | +0.60 (p 0.088) | 6.7 pp |
| 2, results measured after the definition (the real test) | 8 | +0.59 (p 0.12) | +0.52 (p 0.18) | 7.8 pp |

- **Verdict: the surrogate fails.** Not significant on the only stage that tests it, wrong by up to 17.7 points, and
  a 20-seed re-run moves it to r +0.63 without reaching significance, so it is not seed noise.
- **It fails exactly where it would have mattered.** It reproduces GRAB's null (-6.4, -1.8, -5.6) but manufactures an
  OakInk2 effect that does not exist (surrogate mean +10.3 against OakInk-Image's +9.0, while measured is 4.4 against
  14.8), which is the confound the argument turns on. It also misses the planted synthetic difficulty entirely
  (+1.1 surrogate against +9.1 measured) and over-predicts the planted GRAB one (+30.8 against +13.2).
- **It refutes the tidy mechanism I proposed.** Between-cell over within-cell dispersion, i.e. "compositions form
  tight clusters", correlates with the measured penalty at r = +0.00. OakInk2 has the highest cell separation
  (0.21-0.35) and no penalty; OakInk-Image has 0.06-0.16 and the largest. Only an interaction-shaped residual
  tracks at r +0.42, no better than the screen it was meant to replace.
- **What this adds to the paper.** Three cheap data-side statistics have now failed to predict measurable
  compositional difficulty: the interaction screen (both directions), this centroid surrogate, and cluster
  tightness. The honest conclusion is that difficulty is not readable from summary statistics of the poses; it
  depends on what the trained prior can represent, so the paired experiment with a zero-truth control and a planted
  positive control is the only way to know.
- **Caveats:** five seeds against the sweeps' 12 to 130, linear centroid prediction against a VAE, stride 16 against
  the sweeps' stride 4, and stage 1 is description rather than evidence.

### Split verification for category x subject, 09:05

`scripts/verify_category_subject_split.py` rebuilds the split with the sweep's own recorded arguments, calling
`build_paired_split`, `sample_pools` and `assert_split_sound` exactly as `main()` does, for seeds 0, 1, 7, 13, 25
and 39. It was written because the bundle uses a label workaround (`<fine>@<left>@-@<right>` read with the
`oakink2_scene_verb` parser), and a claim this size should not rest on an unchecked encoding.

- The packed factors agree with the parsed cell on all 770 trajectories, and the right factor is the subject in the
  fine label.
- 103 cells over 33 categories and 11 subjects. Held cells are real pairs, e.g. `mug->0000`, `knife->0010`.
- Every seed: targets lie only in held cells, the naive arm's 256 training trajectories contain none of them, the
  informed arm covers all 4, and both arms leak 0.0 of the targets' fine compositions.

So the refutation is not an artefact of the encoding or the split.

## Protocol note: an unintended interim look at category x subject, 08:05

`scripts/emit_table_rows.py` printed its whole table as a side effect of being imported for a lookup test, while
`runs/oakink_category_subject` was still running. The interim value, at 28 of 40 seeds, was therefore seen:
+10.8% of naive error, p against the control below 1e-4, with the axis's screen excess -0.0005 (z -0.11).

What was and was not done about it:
- The seed list was fixed at 0-39 in `runs/PREREG_oakink_category_subject.md` before the sweep started, and it has
  not been touched. The sweep runs to 40 seeds and the reading is taken there.
- No setting, test, or reading rule was changed after the look.
- The script now prints only when run directly, so importing it cannot repeat this.

It is recorded because this repository's own audit history includes an optional-stopping defect, and an interim look
at a pre-registered sweep is the same class of event even when nothing is changed in response.
