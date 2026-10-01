# Pre-registration: GRAB at OakInk-Image's training volume

Written 2026-09-29, before `runs/grab_shape_b64` existed and before any model of it was trained.

## Why

The round-1 cold TMLR review (`docs/tmlr/review_round1.md`, weakness 1) points out that the trajectory budget is
not a volume budget. Expected training windows per arm at budget 256 (`runs/marginal_loss_volume.json`):
OakInk-Image 4,591, TACO 8,144, GRAB contact segment 11,368, GRAB 17,604, OakInk2 57,919. The two datasets with a
penalty are the two with the least data, so the cross-dataset contrast of Table 1 could be a volume effect.
OakInk2 already showed that a smaller budget raised its penalty (+2.30 % at 256, +8.56 % at 64).

## The sweep

GRAB shape x fine intent, exactly as `runs/grab_shape_s4` (held 6, min-chains 5, min-per-composition 5, window 32,
stride 4, 120 epochs, perframe, seeds 0-39), except **budget 64**: about 4,400 windows per arm, matching
OakInk-Image's 4,591 at 256.

Gates: the label-leak gate is budget-independent and passed for these seeds with `grab_shape_s4`; coverage at
budget 64 passes on 40 of 40 seeds (`runs/gates/cover_grab_shape_b64_full.txt`). The bundle is unmodified, so no
frame-leak check applies.

## Readings

Statistic: per-seed penalty / naive mse_target x 100. Reference: the zero-truth control at budget 64
(`pc_easy_rerun`, budget 64, +2.96 %), two-sided Welch.

- **Volume explains GRAB's null** if GRAB at budget 64 is above the budget-64 control at p < 0.05. The paper then
  states that the contrast between datasets in Table 1 is at least partly a matter of training volume, and the
  cross-dataset reading is withdrawn in favour of the within-dataset manipulation only.
- **Volume does not explain GRAB's null** otherwise. The paper then reports that GRAB stayed at the control with
  OakInk-Image's training volume, with the 95 % interval of the difference as the bound.
- Secondary, descriptive: GRAB at 64 against GRAB at 256 (+1.32 %).

No seeds are added or removed after results are seen. The sweep is read only at 40 seeds.
