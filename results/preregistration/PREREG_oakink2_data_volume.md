# Pre-registration: is OakInk2's segmentation result an effect of data volume?

Written 2026-09-21 00:30, before any model of this sweep is trained.

## Why

Cutting OakInk2's recordings at primitive boundaries raised the penalty from +2.30% (whole recordings) to +8.93%
(segments, replicated) and +7.14% (recording-disjoint). The draft attributes this to the label describing the scored
motion. A cold referee pointed out that the manipulation changed other things as well: each arm of a segment sweep
trains on about a third of the frames of the whole-recording sweep, and the naive error rises from 0.018 to about
0.024. A smaller training set could by itself enlarge the gap between a naive and an informed arm, since the
informed arm's extra examples matter more when data are scarce.

## The control

Truncating whole recordings would not do: keeping the first third of a recording raises the share of frames its
first primitive covers, which changes label alignment, the very variable under test. Instead the recordings are left
untouched and the **budget is reduced from 256 to 64 whole recordings**, so labels, grid and poses are those of the
original null sweep and only the amount of training data changes.

Frames per arm, from the bundles' own lengths: whole recordings at 256: 239,225; **whole recordings at 64: 59,806**;
segments at 256: 68,326. The control therefore trains on slightly *fewer* frames than the segment sweeps did.

Settings: those of `oakink2_scene_primitive` with `--budgets 64`, seeds 0-39, out `runs/oakink2_scene_primitive_b64`.
Gates at this budget: label leak exit 0; coverage fails 2 of 40 seeds (`runs/gates/*oakink2_scene_primitive_b64.txt`).
The bundle's frame leak is 0.00% and is a property of the bundle, not of the budget.

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100, two-sided Welch tests, against the zero-truth control **at
budget 64** (`pc_easy_rerun`, +2.96%).

- **Data volume is excluded** if the budget-64 whole-recording sweep is not above its control (p >= 0.05). The same
  labels on even fewer frames then still return a null, so scarcity did not produce the segment sweeps' penalty.
- **Data volume is not excluded** if it is above the control. The segmentation result is then confounded, the draft
  must say so, and the defensible remainder of the OakInk2 direction is the difference, if any, between this sweep
  and the segment sweeps, reported with its interval.
- Secondary, reported either way: this sweep against the segment replication (+8.93%). The two budgets' controls
  differ by 0.4 points, which is stated beside the comparison.

## What this cannot settle

It matches frames, not the number of trajectories (64 against 256) nor the number of distinct labels seen, and a
budget of 64 leaves the informed arm at most 32 held-cell recordings. The grid also differs between whole recordings
(62 cells) and segments (79), which this sweep does not address. No seeds are added or removed after results are seen.
