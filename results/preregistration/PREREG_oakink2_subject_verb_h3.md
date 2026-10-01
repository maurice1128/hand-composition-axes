# Pre-registration: OakInk2 subject x verb at three held compositions

Written 2026-09-16, before any model of this sweep is trained. This file is the declaration; its timestamp and the
first seed directory under `runs/oakink2_subject_verb_h3*` are the evidence of order.

## Why this sweep

The v7 audit found that OakInk2's subject x verb axis, reported as untestable, fails the informed-coverage gate on only
5 of 40 seeds at three held compositions (`runs/gates/cover_oakink2_subject_verb_h3.txt`), a smaller share than the
reported class axis (3 of 12). It is therefore a fourth necessity test that was not run. Its interaction excess is
-0.0360 (z = -4.7, `runs/oakink2_axis_screen.json`), the most negative of any swept or sweepable axis, so it is the
strongest test of necessity these data offer.

## Settings (fixed)

`scripts/experiment_paired_composition.py --bundle data/bundles/oakink2_subject_verb.npz --granularity
oakink2_subject_verb --budgets 256 --kinds perframe --held-compositions 3 --min-chains 4 --min-per-composition 5
--epochs 120`, seeds 0-39, run as three shards with disjoint seed lists and pooled afterwards. Identical to the
scene x primitive confound test except the axis and the held count (3, forced by coverage).

## Readings, declared now

Statistic: per-seed penalty / naive mse_target x 100 ("% naive"), budget 256, all 40 seeds.

1. **Primary (the paper's reading).** Welch two-sample t-test against `runs/pc_easy_rerun` (20 seeds, budget 256).
   - p >= 0.05: at the control. Necessity survives a fourth test, the first with strongly negative excess over forty
     seeds.
   - p < 0.05 with mean above the control: **necessity is refuted** on this axis, and the paper must say so.
   - p < 0.05 with mean below the control: reported as below the control; necessity not refuted.
2. **Also reported.** One-sample t-test of the per-seed penalty against zero (the pre-registered test of the earlier
   sweeps), and the 95% interval on the difference from the control.
3. **Gate.** Seeds failing the coverage gate are not dropped; the estimate excluding them is reported beside the full
   one, as for the category axis. The composition-leak gate must pass (0% informed leak) or the sweep is not reported.

No seeds are added or removed after results are seen. If the sweep cannot finish all 40 seeds before the manuscript
deadline, the seeds that finished are reported with their count and this sentence is quoted.
