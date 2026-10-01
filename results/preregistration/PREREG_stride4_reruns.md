# Pre-registration: stride-4 re-runs of GRAB shape x fine intent and OakInk2 transitions

Written 2026-09-16 ~05:00, before any model of these re-runs is trained.

## Why

Every sweep in Table II of the v7 manuscript trains on 32-frame windows at stride 4, except two:
`runs/grab_shape_v2` (GRAB shape x fine intent, stride 32) and `runs/oakink2_paired_v2` (OakInk2 annotated
transitions, stride 16). The manuscript does not state stride. These two axes carry load in the argument:
GRAB shape x fine intent is the positive-interaction axis outside OakInk-Image that returned nothing (+0.7% of naive
error), and the transitions axis is the one strongly negative necessity test (excess -0.0120, z = -2.6). At stride 32
a model sees about one eighth of the windows the other axes' models see, so their nulls are not comparable to the
other rows.

## Settings

Identical to the original sweeps' recorded `args` except `--stride 4` and `--kinds perframe` (the paper's
difficulty is the perframe prior's penalty), budget 256, 120 epochs, same seeds (GRAB 0-39; transitions the original
seed list). Output `runs/grab_shape_s4` and `runs/oakink2_transitions_s4`.

## Readings, declared now

Statistic and test as in the paper: per-seed penalty / naive mse_target x 100, Welch two-sample t-test against
`runs/pc_easy_rerun` (20 seeds, budget 256); one-sample t-test against zero also reported.

**GRAB shape x fine intent (positive excess +0.0063, z = 3.9).**
- p >= 0.05 against the control: the null holds at stride 4; the v7 reading stands and the table row is replaced by
  the stride-4 value.
- p < 0.05, above the control: the v7 null was produced by stride 32. Sufficiency is no longer contradicted by this
  axis, and **the dataset confound is broken** (a positive-interaction axis outside OakInk-Image carries difficulty).

**OakInk2 transitions (excess -0.0120, z = -2.6).**
- p >= 0.05: necessity survives at stride 4.
- p < 0.05, above the control: **necessity is refuted** on this axis.

Either way the manuscript reports the stride-4 values and states that the original runs used stride 32 and 16.
No seeds are added or removed after results are seen.
