# Pre-registration: two clean-up sweeps that remove caveats from the alignment argument

Written 2026-09-20, after both bundles were built and gated and before any model of either sweep is trained.
Builders and feasibility checks: `scripts/build_oakink_alignment_v3.py`,
`scripts/experiment_paired_recording_disjoint.py`, `runs/build_oakink_alignment_v3.json`,
`runs/gates/recording_disjoint_dryrun.json`.

## A. Category-matched misaligned clips on OakInk-Image

**Caveat being removed.** In the alignment test (`runs/DIAGNOSTICS_RESULTS.md` section 11) the aligned bundle covers
14 object categories and the misaligned one 33, so the pair differs in more than alignment.

**Design.** `data/bundles/oakink_long3_misaligned_cm.npz` repeats the misaligned construction using only the aligned
bundle's 14 categories: 210 trajectories, the same count as the aligned bundle; per-category counts within 2 of it;
39 coarse cells against 38; median 280 frames; no clip used twice. Gate (`scripts/gate_alignment_v2.py`): frame leak
0.00% in both arms, label leak passes, coverage fails 4 of 40 seeds (19, 20, 29, 38), minimum naive pool 132.
Screen: `right_only` 0.0041, excess -0.0080 (z -2.2).

**Settings.** As section 11: `--granularity oakink_category --held-compositions 5 --min-chains 4
--min-per-composition 5 --budgets 64 --kinds perframe --window 32 --stride 4 --epochs 120`, seeds 0-39, out
`runs/pc_oakink_long3_misaligned_cm`.

**Readings, declared now.** Per-seed penalty / naive mse_target x 100; two-sided Welch.
- **The caveat is removed** if the category-matched misaligned sweep is below the aligned sweep
  (`pc_oakink_long3_aligned`, +17.86%) at p < 0.05 **and** is not above the budget-64 control (+2.96%) at p < 0.05.
- If it is below aligned but above the control, alignment still matters but part of section 11's drop came from
  category mix; both numbers are reported.
- If it is not below aligned, section 11's contrast was a category artefact and the alignment claim on OakInk-Image
  is withdrawn.

## B. Recording-disjoint primitive segments on OakInk2

**Caveat being removed.** In the primitive-segment sweeps (sections 5 and 13) about 45% of target segments share a
recording with some training segment, in both arms.

**Design.** `scripts/experiment_paired_recording_disjoint.py` wraps `build_paired_split` and removes from both
training pools every segment whose recording supplies a target segment; everything else is the unmodified
experiment. Feasibility over seeds 200-239 at budget 256: smallest naive pool 1,368 after filtering, informed
coverage 5 of 5 on all 40 seeds, target segments sharing a recording 0.0 in both arms, label leak 0, frame leak
0.00% (seeds 200-204).

**Settings.** `--bundle data/bundles/oakink2_primseg.npz --granularity oakink2_scene_primitive
--held-compositions 5 --min-chains 4 --min-per-composition 5 --budgets 256 --kinds perframe --window 32 --stride 4
--epochs 120`, fresh seeds 200-239, out `runs/oakink2_primseg_recdisjoint`.

**Readings, declared now.** Two-sided Welch against the control (`pc_easy_rerun`, budget 256, +2.59%) and against
whole trajectories (`oakink2_scene_primitive`, +2.30%).
- **The caveat is removed** if the recording-disjoint sweep is above both (each p < 0.05).
- If it is above neither, the primitive-segment effect depended on shared recordings, and the OakInk2 direction of
  the argument is withdrawn.
- A mixed outcome is reported as it falls.

Known side effect, stated now: filtering shrinks both pools by up to a quarter in the worst seeds, equally in both
arms; a rise in absolute error in both arms is the filter, not composition.

No seeds are added or removed after results are seen, in either sweep.
