# Pre-registration: label-to-window alignment on OakInk-Image, version 4 (no constituent leak)

Written 2026-09-20, after the bundles were built and gated and before any model on them is trained.

## Why a fourth version

An adversarial audit of the draft found that version 2's aligned bundle is confounded. Each trajectory there is
three clips sharing a category and an intent, labelled by the first clip, and the split keeps only that first clip's
`object->intent` out of training. `scripts/check_constituent_leak.py` confirms it:

| measure, `oakink_long3_aligned` | informed arm | naive arm |
|---|---|---|
| target trajectories with a constituent clip whose exact `object->intent` occurs in the arm's training clips | 87.6% | 0% |

The unmodified experiment never lets the informed arm see a target's `object->intent`, so version 2's aligned value
(+17.86%, above the +10.15% baseline) is inflated and its contrast with the misaligned sweep is not clean. The
misaligned sweeps themselves, and their equality with the control, are unaffected.

## Design

Bundles come from `scripts/build_oakink_alignment_v4.py`, with statistics in `runs/build_oakink_alignment_v4.json`.

- **`oakink_pair_aligned.npz`.** Each trajectory is two clips with the **same `object->intent`**, so every
  constituent carries the trajectory's own fine label and the split's disjointness covers all of it. It has 334
  trajectories, median 141.5 frames, 33 categories and 93 cells, with no clip used twice. Constituent leak is 0 for
  both arms on every seed (asserted).
- **`oakink_pair_misaligned.npz`.** Each trajectory is two clips of the same category, the second with a
  **different intent**, labelled by the first clip. It has 334 trajectories, the same 33 categories (per-category
  counts within 1), median 181 frames and 83 cells.
- **Gates** (`scripts/gate_alignment_v2.py`, held 5, budget 64). Frame leak is 0.00% per arm and the label leak gate
  passes. Coverage fails 2 of 40 seeds (aligned: 3 and 29) and 5 of 40 (misaligned: 8, 15, 18, 22 and 25). The
  minimum naive pools are 236 and 240.

**Settings** are as in version 2: `--granularity oakink_category --held-compositions 5 --min-chains 4
--min-per-composition 5 --budgets 64 --kinds perframe --window 32 --stride 4 --epochs 120`, seeds 0-39, out
`runs/pc_oakink_pair_aligned` and `runs/pc_oakink_pair_misaligned`. **Comparators** are the unmodified baseline at
the same budget (`runs/pc_oakink_b64`, +10.15%) and the zero-truth control at budget 64 (+2.96%).

## What the misaligned manipulation does, stated before the result

Misalignment is not a nuisance to be controlled away; it is the mechanism under test. The split is made on labels,
so when a label does not describe a trajectory's content, holding out the label does not hold out the motion. In the
misaligned pair bundle, averaged over seeds 0-9:
- 8.4% of the naive arm's training clips are held-cell motion;
- 27.9% of target trajectories contain a clip whose `object->intent` the naive arm has seen;
- only 61% of target clips are held-cell motion.

In the aligned pair bundle these are 0%, 0% and 100%. The intent in a misaligned label explains almost no pose
variance (screen `right_only` 0.009 against 0.047). These numbers are reported with the result whatever it is.

## Readings, declared now

The statistic is the per-seed penalty / naive mse_target x 100, and the tests are two-sided Welch.

- **Alignment is supported** if both hold:
  - (a) aligned-pair is above the control, p < 0.05;
  - (b) misaligned-pair is below aligned-pair, p < 0.05.
- **Stronger form**, if also:
  - (c) misaligned-pair is not above the control (p >= 0.05);
  - (d) aligned-pair is not below the unmodified baseline (p >= 0.05), i.e. concatenation alone does not remove the
    penalty.
- **Not supported** if (b) fails. The draft's alignment claim on OakInk-Image is then reduced to the unmodified
  baseline against the version-2/3 misaligned sweeps, with that weakness stated.
- Estimates excluding the coverage-failing seeds are reported beside the full ones.

No seeds are added or removed after results are seen. Version 2's aligned sweep is kept in the record as confounded
and is not used as evidence.
