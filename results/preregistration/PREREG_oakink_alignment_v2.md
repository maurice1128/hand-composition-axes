# Pre-registration: label-to-window alignment on OakInk-Image, version 2 (no clip reuse)

Written 2026-09-16 ~22:20, before the bundles are built and before any model is trained on them.

## Why a second version

The first long-clip pair (`pc_oakink_long`, `pc_oakink_longaligned`) is void. Its builders reused each source clip in
about four concatenated trajectories. A frame-level check (`scripts/check_frame_leak.py`) then found target frames
verbatim in training: the aligned version's informed arm had 81% and its naive arm 0%, and the misaligned version's
arms had 89% and 78%. See `runs/DIAGNOSTICS_RESULTS.md` section 10. The hypothesis it was meant to test remains
open: OakInk-Image measures compositional difficulty because its label describes the window being scored.

## Design

At a budget of 256, OakInk-Image's 770 clips cannot supply non-overlapping long trajectories. This version
therefore changes two things, both declared here before any result:

- **Concatenation factor 3**, with **no source clip used more than once** in a bundle. The 4-frame linear cross-fade
  at each join is kept.
  - `data/bundles/oakink_long3_aligned.npz`: all three clips share the category **and** the intent of the
    trajectory's label, which is the first clip's `object->intent`.
  - `data/bundles/oakink_long3_misaligned.npz`: all three share the category; the second and third have intents
    different from the label's.
- **Budget 64** for all sweeps in this test, including a baseline on the unmodified bundle, so every comparison is
  at the same training-set size.

Three sweeps, all at the confirmatory axis's other settings: `--granularity oakink_category --held-compositions 5
--min-chains 4 --min-per-composition 5 --budgets 64 --kinds perframe --window 32 --stride 4 --epochs 120`,
seeds 0-39.

| sweep | bundle |
|---|---|
| `pc_oakink_b64` (baseline) | `data/bundles/oakink.npz` |
| `pc_oakink_long3_aligned` | `data/bundles/oakink_long3_aligned.npz` |
| `pc_oakink_long3_misaligned` | `data/bundles/oakink_long3_misaligned.npz` |

## Gates that must pass before any of the two new bundles is trained

- `scripts/check_informed_coverage.py` and `scripts/check_composition_leak.py` at the settings above, seeds 0-39.
- **Frame-level leak:** `scripts/check_frame_leak.py` at budget 64, seeds 0-4. The share of target frames found
  verbatim in each arm's training set must be **at most 1%** for both arms of both bundles.
- The naive pool must exceed 64 on every seed.

A bundle that fails any of these is reported as unswept. It is not re-tuned after its sweep has been seen.

*Amended 22:15, before any build output existed:* the coverage gate as written above could not be met even by the
unmodified bundle, which fails 3 of 40 seeds on breadth at the confirmatory settings. The coverage requirement is
therefore **at most 10 of 40 seeds failing**, with the estimate excluding the failing seeds reported beside the full
one, as for the category axis. The label-leak gate (exit 0), the frame-leak gate (at most 1% per arm) and the naive
pool (above 64) stay strict. The check is automated by `scripts/gate_alignment_v2.py`, and
`scripts/wait_and_queue_alignment_v2.ps1` appends a bundle's sweep to the GPU queue only if it passes.

## Build report, recorded 22:25 before any v2 sweep has a result

From `scripts/build_oakink_alignment_v2.py` (`runs/build_oakink_alignment_v2.json`):
- **Bundles.** 210 trajectories each; the misaligned bundle was subsampled from 246 to match. Median length 206
  frames aligned against 273 misaligned. Coarse cells 38 (14 categories) against 59 (33 categories). 140 source
  clips unused in each. No clip or frame appears in two trajectories.
- **Frame leak.** 0.00% in both arms of both bundles, and of `oakink.npz`, at budget 64, seeds 0-4.
- **Gates.** Label leak passes for both. Coverage fails 3 of 40 aligned and 9 of 40 misaligned, both within the
  amended limit of 10. Two misaligned failures are also thin in depth (2.20 on seed 25). Unmodified `oakink.npz`
  fails 3 of 40 at budget 64. Minimum naive pool 130 aligned, 144 misaligned.
- **Screen**, as left_only / right_only / excess (z):

  | bundle | left_only | right_only | excess (z) |
  |---|---|---|---|
  | `oakink.npz` | 0.097 | 0.045 | +0.0238 (8.3) |
  | aligned | 0.086 | 0.070 | +0.0203 (3.6) |
  | misaligned | 0.112 | 0.006 | -0.0061 (-1.7) |

**Caveat that limits the reading, stated now.** In the misaligned bundle the label's intent explains almost no pose
variance (right_only 0.006). If that sweep is null, "the label does not describe the window" and "the axis has lost
its intent information" are not separable here. In this design they are close to the same thing, but a null must be
reported with that sentence. The pair also differs in category coverage (14 against 33), cell count and length, so
any difference between them is attributed to alignment only together with those differences.

## Comparators

- The zero-truth control at budget 64, `runs/pc_easy_rerun`: +2.96% of naive error, n 20.
- The baseline `pc_oakink_b64`.

## Readings, declared now

Statistic as in Table II: per-seed penalty divided by naive mse_target, times 100. Tests are two-sided Welch.

- **Alignment is the property** if both hold:
  - (a) misaligned is below aligned, p < 0.05;
  - (b) aligned is not below the baseline, p >= 0.05, or aligned is above it.
- **Length is the property** if both long versions are below the baseline (each p < 0.05) and do not differ from each
  other (p >= 0.05).
- **Neither** if both long versions are indistinguishable from the baseline (each p >= 0.05). In that case clip
  length and alignment, at this factor, do not explain OakInk-Image's difficulty.
- **Any other pattern** is reported as it falls, with no claim beyond the numbers.

The magnitude of each version against the control is reported as well. No seeds are added or removed after results
are seen.
