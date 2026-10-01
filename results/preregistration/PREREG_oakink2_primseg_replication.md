# Pre-registration: independent replication of the OakInk2 primitive-segment result

Written 2026-09-17 ~01:30, before any model of this replication is trained.

## Why

The original sweep (`runs/oakink2_primseg_s4`, 20 seeds, readings in `runs/PREREG_oakink2_primseg.md`) moved OakInk2
scene x primitive from +2.30% to +8.97% of naive error: p 0.042 against the control and p 0.042 against the
whole-trajectory sweep, with both intervals reaching down to +0.26. That is weak. It is also the only evidence that
fixing label alignment turns a null dataset into one that measures compositional difficulty, which is half of the
paper's two-way argument.

Adding seeds to that sweep after seeing it would be optional stopping, which is already on this project's record.
This is instead a fresh sweep on **new seeds, 100-139**, whose seed list and reading are fixed here.

## Settings

Identical to the original: `--bundle data/bundles/oakink2_primseg.npz --granularity oakink2_scene_primitive
--held-compositions 5 --min-chains 4 --min-per-composition 5 --budgets 256 --kinds perframe --window 32 --stride 4
--epochs 120`, seeds 100-139, out `runs/oakink2_primseg_rep`.

Before training, the frame-leak check must show at most 1% of target frames verbatim in each arm for seeds 100-104,
and the label-leak gate must pass on seeds 100-139.

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100. Tests are two-sided Welch against the control
(`pc_easy_rerun`, budget 256) and against the whole-trajectory sweep (`oakink2_scene_primitive`).

- **Replicates** if the replication is above both the control and the whole-trajectory sweep (each p < 0.05).
- **Fails to replicate** if it is not above the whole-trajectory sweep (p >= 0.05).

The replication is read on its own. A pooled estimate over both sweeps (60 seeds) is reported **only** as a
secondary number, labelled as pooled after the fact.

The frame-budget caveat of the original applies unchanged: each arm trains on about a third of the frames the
whole-trajectory arms saw.
