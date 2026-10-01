# Pre-registration: OakInk2 scene x primitive on primitive segments (dilution test)

Written 2026-09-16 ~05:35, after the bundle was built and gated, before any model on it is trained.

## Question

OakInk2 scene x primitive returned +2.3% of naive error, at the zero-truth control, with positive excess (+0.0040,
z 2.9). Each OakInk2 trajectory is a whole task (median 82 s at 7.5 fps) carrying one label, so most 32-frame
windows (4.27 s) are unrelated to the labelled primitive. `data/bundles/oakink2_primseg.npz`
(`scripts/build_oakink2_diagnostics.py`) cuts every trajectory at OakInk2's own per-primitive frame spans
(`program_info` json; spans reproduce all 609 trajectory lengths) and labels each segment with its own
scene x primitive: 2,128 segments of at least 32 frames, median 187 frames (24.9 s), 79 cells. Screen excess rises
to +0.0113 (z 16.4). Gates at held 5, min_chains 4, min_per_composition 5, seeds 0-39: coverage 0 of 40 fail, leak
passes.

Known weakness, stated now: segments are split individually, so 42-46% of target segments share a recording with a
training segment (naive 45.6%, informed 41.6%). The leak gate does not see this. The two arms are affected about
equally.

## Settings

`--bundle data/bundles/oakink2_primseg.npz --granularity oakink2_scene_primitive --held-compositions 5
--min-chains 4 --min-per-composition 5 --budgets 256 --kinds perframe --window 32 --stride 4 --epochs 120`,
seeds 0-19 (twenty, not forty: OakInk2 costs ~30 min per seed and the manuscript deadline is 2026-09-17 14:59).

## Readings, declared now

Statistic and tests as in Table II: per-seed penalty / naive mse_target x 100; Welch against `runs/pc_easy_rerun`;
Welch against `runs/oakink2_scene_primitive` (whole trajectories, same axis); one-sample t against zero.

- Above the control and above the whole-trajectory sweep (both p < 0.05): cutting to the labelled primitive restores
  difficulty on OakInk2. Dilution is supported, and the OakInk-Image / OakInk2 difference is at least partly how
  clips are cut, not the dataset.
- Not distinguishable from the whole-trajectory sweep: dilution across primitives does not explain OakInk2's null.
  (Dilution within a primitive, segments still ~6 windows long, is not tested.)

If the GRAB planted sweep shows the instrument cannot see planted difficulty on real poses, this sweep's null is not
interpretable. No seeds are added or removed after results are seen.

## Added 11:12, before any seed of this sweep finished: the budget is matched in trajectories, not frames

The budget of 256 counts *trajectories*, and a primitive segment has a median of 187 frames against 613 for a whole
OakInk2 trajectory. So each arm here trains on roughly a third of the frames the whole-trajectory sweep's arms saw:
60 train batches per epoch against 214 to 247, and 1.4 s per epoch against about 5.5 s.

This was noticed from the first seed's timing, not from any result. Nothing above is changed in response. It is
recorded because it qualifies the declared comparison against `runs/oakink2_scene_primitive`:

- **If this sweep shows difficulty**, less training data makes that harder, not easier, so the reading stands.
- **If this sweep is null**, the null is ambiguous between "cutting to the labelled primitive does not help" and
  "one third of the frames is too little", and it must be reported that way. The honest follow-up would be a
  frame-matched budget, which there is no time to run before the manuscript deadline.
