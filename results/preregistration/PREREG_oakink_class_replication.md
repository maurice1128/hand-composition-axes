# Pre-registration: independent replication of OakInk-Image functional class x intent

Written 2026-09-20, before any model of this replication is trained.

## Why

Table 1's functional-class row is the largest OakInk-Image effect (+17.9%) and rests on 12 seeds, while the other
OakInk-Image rows have 40 to 130. A referee objected that the two most load-bearing surprises in the table are its
two n = 12 rows. Adding seeds to `oakink_class_v1` after seeing it would be optional stopping, which is already on
this project's record, so this is a fresh sweep on **new seeds, 100-139**, read on its own.

## Settings

Identical to `oakink_class_v1`: `--bundle data/bundles/oakink.npz --granularity oakink_class
--held-compositions 4 --min-chains 4 --min-per-composition 4 --budgets 256 --kinds perframe --window 32 --stride 4
--epochs 120`, seeds 100-139, out `runs/oakink_class_rep`.

Before training: the label-leak gate must pass on seeds 100-139 and the frame-leak check must show at most 1% of
target frames verbatim in each arm on seeds 100-104. Coverage is recorded, failing seeds are kept, and an estimate
excluding them is reported beside the full one. `oakink_class_v1` failed coverage on 3 of its 12 seeds, the worst
rate in the study, so the coverage-excluded estimate is reported for this sweep whatever it shows.

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100. Two-sided Welch tests.

- **Replicates** if the replication is above the zero-truth control at p < 0.05. Table 1 then reports the
  replication's value and its n, with the original given beside it.
- **Fails to replicate** if it is not. Both sweeps are then reported, and no claim rests on this axis.
- The two sweeps are compared descriptively. A pooled estimate is reported only as a secondary number, labelled as
  pooled after the fact.

If the sham-grid control (`runs/PREREG_sham_grid.md`) shows that OakInk-Image's within-dataset bias exceeds the
synthetic control's, this sweep is read against the larger reference, as that pre-registration states.

No seeds are added or removed after results are seen. The sweep is read at 40 seeds.
