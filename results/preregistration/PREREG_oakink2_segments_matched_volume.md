# Pre-registration: do primitive segments carry a penalty at the SAME data volume as whole recordings?

Written 2026-09-25, before any model of this sweep is trained, at the author's request after the data-volume control
(`runs/PREREG_oakink2_data_volume.md`, `runs/DIAGNOSTICS_RESULTS.md` section 22) came back against us.

## Why

Cutting OakInk2's recordings at primitive boundaries raised the penalty from +2.30% (whole recordings, budget 256,
239,225 frames per arm) to +8.93% (segments, budget 256, 68,326 frames). The data-volume control showed that whole
recordings at budget 64 (59,806 frames) also read +8.56%, indistinguishable from the segments (p 0.90). So the rise
may be caused by training on less data, not by the labels describing the scored motion. That confound removed the
OakInk2 direction from the paper's evidence.

This sweep breaks the confound the other way: segments trained on the SAME number of frames as whole recordings at
256. If segmentation matters, segments should still carry a penalty where whole recordings at matched volume do not.

## Design

- Bundle `data/bundles/oakink2_primseg.npz` (2,128 segments, mean 267 frames), granularity `oakink2_scene_primitive`,
  held 5, min_chains 4, min_per_composition 5, window 32, stride 4, 120 epochs, perframe - identical to
  `oakink2_primseg_rep` except the budget.
- **Budget 896 segments per arm**, so each arm sees about 239,000 frames, matching whole recordings at 256.
- **Fresh seeds 300-339.** No result of this bundle at this budget has been seen.
- The informed arm swaps in at most budget/2 held-cell trajectories, but the informed-extra pools hold only about
  69-143 segments, so realised exposure is about 0.08-0.16, close to whole@256's 0.12. The realised value is
  recomputed and reported.
- Gates before training: label leak exit 0 on seeds 300-339; coverage recorded (failing seeds kept, an excluded
  estimate reported beside the full one); frame leak at most 1% per arm on seeds 300-304; naive pool above 896 on
  every seed.

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100. Two-sided Welch tests.

**Confirmatory contrast: segments at 896 against whole recordings at 256** (`oakink2_scene_primitive`, +2.30%, 40
seeds). Both are OakInk2, both see about 239,000 frames per arm, and both have exposure near 0.12, so this contrast
needs no synthetic reference.

- **Segmentation matters at matched volume** if segments@896 are above whole@256 at p < 0.05. The OakInk2 direction
  is then restored as evidence, stated with this sweep's size and interval, and the draft says the earlier
  budget-256 segment result was confounded but this one is not.
- **Segmentation does not matter at matched volume** if segments@896 are not above whole@256. The OakInk2 direction
  is then withdrawn for good; the paper's causal evidence for alignment is the OakInk-Image manipulation and the TACO
  prediction only, and the draft says so.

Secondary, reported either way, not confirmatory: segments@896 against the synthetic zero-truth control at budget
256 (+2.59%). No synthetic control exists at budget 896, and the estimator's bias depends on budget (b64 +2.96, b256
+2.59), so this comparison is flagged as approximate.

## What this cannot settle

It matches frames and exposure, not the number of trajectories (896 against 256) nor the grid (79 cells against 62).
A segment is shorter than a whole recording, so it offers fewer windows per trajectory, which the frame match
compensates for only on average. No seeds are added or removed after results are seen; the sweep is read at 40.
