# Pre-registration: independent replication of the OakInk2 transitions result at stride 4

Written 2026-09-20, before any model of this replication is trained.

## Why

`runs/oakink2_transitions_s4` (seeds 0-11) returned +16.71% of naive error at stride 4: 12 of 12 seeds positive,
p 1.1e-4 against the control, against +3.44% for the same axis at stride 16 (`runs/DIAGNOSTICS_RESULTS.md`
section 17). It is the only non-OakInk-Image axis that carries a penalty on whole recordings, and the draft now rests
a sentence on it. Twelve seeds is thin for that. Adding seeds to that sweep after seeing it would be optional
stopping, so this is a fresh sweep on **new seeds, 100-139**, read on its own.

## Settings

Identical to the original: `--bundle data/bundles/oakink2.npz --granularity fine --held-compositions 6
--min-chains 6 --min-per-composition 5 --budgets 256 --kinds perframe --window 32 --stride 4 --epochs 120`,
seeds 100-139, out `runs/oakink2_transitions_rep`.

Before training: the label-leak gate must pass on seeds 100-139, the frame-leak check must show at most 1% of target
frames verbatim in each arm on seeds 100-104, and the coverage result is recorded, with failing seeds kept and an
estimate excluding them reported beside the full one.

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100. Two-sided Welch tests.

- **Replicates** if the replication is above the zero-truth control (`pc_easy_rerun`, budget 256, +2.59%) at
  p < 0.05.
- **Fails to replicate** if it is not. The draft's sentence on this axis is then withdrawn, and Table 1 reports both
  sweeps.
- The replication's size is compared descriptively with the original +16.71%; a pooled estimate over both sweeps
  (52 seeds) is reported only as a secondary number, labelled as pooled after the fact.

## What this replication does not settle, stated now

Whether the penalty is compositional or reflects task novelty. On this axis a held-out composition is a transition
between two consecutive primitives, and recordings that contain a given transition may belong to the same task, so
holding out the transition may come close to holding out the task. This sweep tests only whether the penalty is
reproducible. No seeds are added or removed after results are seen.
