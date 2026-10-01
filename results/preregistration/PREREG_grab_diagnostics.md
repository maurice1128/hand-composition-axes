# Pre-registration: GRAB diagnostic sweeps (grasp-only, planted interaction)

Written 2026-09-16 ~05:10, after the bundles were built and gated, before any model on them is trained.
Bundles: `scripts/build_grab_diagnostics.py`. Screen: `runs/axis_screen_grab_diagnostics.json`. Gates (held 6,
min_chains 5, min_per_composition 5, seeds 0-39): coverage 0 of 40 fail and leak 0 of 40 on both.

## Question

On GRAB shape x fine intent the instrument returned +0.7% of naive error, at the zero-truth control, although the
axis has positive interaction. Two explanations are tested, each by changing one thing:

- **H2, dilution.** `grab_grasp.npz` keeps only each trajectory's contact segment (first to last right-hand contact
  frame, padded 8 frames; 990 of 1048 trajectories, 64.2% of frames). Screen excess rises from +0.0063 (z 3.9) to
  +0.0131 (z 6.8).
- **H3, insensitivity.** `grab_planted.npz` adds a fixed per-cell pose offset with its additive part removed, i.e. a
  pure shape x intent interaction, RMS 0.166 normalised units (matched to synth_hard's frame-averaged via-point
  offset; 9.84% of values clamped). Screen excess +0.1512 (z 110.7), confirming only that the plant is present.

## Settings

Same as the GRAB shape x fine intent sweep except the bundle: granularity shape, held 6, min_chains 5,
min_per_composition 5, budget 256, 120 epochs, window 32, perframe, seeds 0-39. Each diagnostic runs twice:
at stride 32 (comparable with `runs/grab_shape_v2`, +0.7%) and at stride 4 (comparable with `runs/grab_shape_s4`).

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100. Test: Welch two-sample t-test against the zero-truth control
`runs/pc_easy_rerun` (budget 256), and against the unmodified GRAB sweep at the same stride.

**Planted (H3).**
- Above the control and above unmodified GRAB (both p < 0.05): the instrument can see a planted interaction on real
  GRAB poses. H3 is rejected; GRAB's null is evidence of no difficulty at that size.
- Not distinguishable from the control (p >= 0.05): the instrument cannot see even a planted interaction of
  synth_hard's size on GRAB. H3 is supported; GRAB's (and by extension OakInk2's) nulls are not evidence.
- Magnitude is compared descriptively with synth_hard's +9.1%; no threshold is set on it.

**Grasp-only (H2).**
- Above the control and above unmodified GRAB at the same stride (both p < 0.05): restricting windows to the grasp
  restores difficulty; H2 is supported.
- Not distinguishable from unmodified GRAB: H2 is not supported at this resolution.

A planted null makes the grasp-only reading uninterpretable (an insensitive instrument cannot show dilution), so the
planted result is read first. No seeds are added or removed after results are seen.
