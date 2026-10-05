# Final pre-submission numbers audit, TMLR manuscript

Date: 2026-10-05. Method: an independent recomputation (own Welch/CI/bootstrap code in
`scratchpad/audit_numbers.py`, not `scripts/verify_tmlr_draft.py`; the verifier was read only to learn which
sweep directory and budget each number comes from). Text audited: the final anonymous PDF text
(`final_pages/anon_text.txt`), cross-checked against `docs/tmlr/PAPER_TMLR.md`.

Definitions used throughout: per-seed penalty = 100 (E_naive - E_informed) / E_naive on `mse_target`; two-sided
Welch t test with Welch-Satterthwaite df; 95 % CI of a difference = d +/- t_0.975(df) SE; 95 % CI of a mean =
t_0.975(n-1) SD/sqrt(n); seeds-positive = count of per-seed penalties > 0; Table 2 "both" rows (category x intent,
affordance x intent, shape x intent class, scene x verb) restricted to seeds where the modular bank also finished,
as the manuscript states for the 69-seed row (the other three lose no seeds). OakInk2 durations use 30 fps
although the bundle is labelled 7.5 fps. Bootstrap for the 82 % figure: percentile, 100,000 resamples,
independent resampling of the aligned first-clip, misaligned first-clip and control-64 seeds, three RNG seeds.

Preliminary data checks: no duplicate (kind, budget, seed) rows in any of the 57 results.json files read;
`penalty` equals `naive.mse_target - informed.mse_target` to 1e-12 in every row; the 10 Table-2 axes' seed sets in
`axis_diagnostics_tmlr.json` equal the results.json seed sets, and its stored `penalty_pct` equals the recomputed
value to 1e-12.

## 1. Summary

**250 rows checked (229 OK, 19 auditor probes, 2 mismatches).** Every statistic
(means, SDs, seeds positive, Welch p-values, CIs on differences and on means, sentence arithmetic, bootstrap CI,
Bonferroni, random-effects claim, counts, durations, windows, exposure, coverage failures, Appendix D rows)
reproduces to the printed precision, and every quantity stated in more than one place agrees across abstract, body,
tables, captions, appendices and conclusion. Rounding is correct at every printed value, including the borderline
ones (control-64 upper bound 2.9138 -> 2.9; shared fraction 6.69 % -> 7 %; sparse p 0.002498 -> 0.0025; bootstrap
lower bound 64.55-64.65 -> 65 %).

The two mismatches are both in method/appendix prose; neither changes a result:

1. **Section 3.5, "permuting one factor leaves at most 3 usable cells against 17."** The maximum over the 200
   permutations is **10**, not 3. The sham-grid script's refusal message reports the pool of the *last* draw
   (`best was {sham_pool}` is evaluated after the loop ends), and the pre-registration and manuscript copied that
   number. Distribution over the 200 draws: 1 (3 draws), 2 (13), 3 (39), 4 (62), 5 (40), 6 (33), 7 (6), 8 (3),
   10 (1). The conclusion (a factor permutation cannot match the real grid's 17) stands.
2. **Appendix A, "A held-out cell must span at least four distinct fine labels (five on GRAB's shape-by-fine-intent
   axis)."** The OakInk2 annotated-transitions sweeps (`oakink2_transitions_rep`, `oakink2_transitions_s4`,
   `oakink2_paired_v2`) use `min_chains = 6`, so on that axis a held transition must appear in at least **six**
   distinct recordings (fine labels). The GRAB planted and contact-segment sweeps use five, consistent with the
   stated GRAB rule. The sentence omits the transitions axis.

## 2. Table of every number checked

Status: OK = manuscript value equals the recomputed value at the printed precision; INFO = an auditor probe or
sanity check, not a manuscript number; MISMATCH = the manuscript value is wrong. Notes carry raw values for the
rounding check. Scientific notation is written as 4.7e-7 for 4.7 x 10^-7.

| # | Section | Quantity | Manuscript | Recomputed | Source | Status | Note |
|---|---|---|---|---|---|---|---|
| 1 | Abstract | control +2.6 % (abstract, discussion) | 2.6 | 2.6 | pc_easy_rerun@256 | OK |  |
| 2 | Abstract | aligned +17.0 % (abstract, Fig1, discussion) | 17.0 | 17.0 | pc_oakink_pair_aligned@64 | OK |  |
| 3 | Abstract | misaligned +3.9 % (abstract, Fig1) | 3.9 | 3.9 | pc_oakink_pair_misaligned@64 | OK |  |
| 4 | Abstract | at most 2.9 points above control (abstract, 4.3) | 2.9 | 2.9 | welch(mis, c64).hi | OK | raw 2.9138 |
| 5 | Abstract | TACO +10.6 % (abstract, Table 2) | 10.6 | 10.6 | taco_action_tool | OK |  |
| 6 | Abstract | Fig1 control at budget 64 +2.96 % | 2.96 | 2.96 | pc_easy_rerun@64 | OK |  |
| 7 | Abstract | Fig1 40 seeds | 40 | 40 | n rows | OK |  |
| 8 | Intro 3.1 datasets | oakink n trajectories | 770 | 770 | bundles/oakink.npz lengths | OK |  |
| 9 | Intro 3.1 datasets | oakink median frames | 72 | 72 | bundles/oakink.npz | OK |  |
| 10 | Intro 3.1 datasets | oakink median duration s | 2.4 | 2.4 | median/30.0 fps | OK | raw 2.400; bundle fps label 30.0 |
| 11 | Intro 3.1 datasets | grab n trajectories | 1,048 | 1,048 | bundles/grab.npz lengths | OK |  |
| 12 | Intro 3.1 datasets | grab median frames | 254 | 254 | bundles/grab.npz | OK |  |
| 13 | Intro 3.1 datasets | grab median duration s | 8.5 | 8.5 | median/30.0 fps | OK | raw 8.467; bundle fps label 30.0 |
| 14 | Intro 3.1 datasets | oakink2 n trajectories | 609 | 609 | bundles/oakink2.npz lengths | OK |  |
| 15 | Intro 3.1 datasets | oakink2 median frames | 613 | 613 | bundles/oakink2.npz | OK |  |
| 16 | Intro 3.1 datasets | oakink2 median duration s | 20 | 20 | median/30.0 fps | OK | raw 20.433; bundle fps label 7.5 |
| 17 | Intro 3.1 datasets | taco_action_tool n trajectories | 2,317 | 2,317 | bundles/taco_action_tool.npz lengths | OK |  |
| 18 | Intro 3.1 datasets | taco_action_tool median frames | 148 | 148 | bundles/taco_action_tool.npz | OK |  |
| 19 | Intro 3.1 datasets | taco_action_tool median duration s | 4.9 | 4.9 | median/30.0 fps | OK | raw 4.933; bundle fps label 30.0 |
| 20 | Intro 3.1 datasets | window 1.07 s | 1.07 | 1.07 | 32 frames / 30 fps | OK |  |
| 21 | Intro 3.1 datasets | OakInk2 350 of 609 multi-primitive | 350 of 609 | 350 of 609 | label_coverage_oakink2.json | OK |  |
| 22 | Intro 3.1 datasets | GRAB subjects s1-s7, s10 | s1..s7,s10 | s1..s7,s10 | grab meta | OK |  |
| 23 | Intro 3.1 datasets | GRAB 51 objects | 51 | 51 | caredex.data.grab.GRAB_SHAPE_CLASS | OK |  |
| 24 | Intro 3.1 datasets | GRAB 9 shape classes | 9 | 9 | GRAB_SHAPE_CLASS | OK |  |
| 25 | Intro 3.1 datasets | GRAB objects present in bundle (vs 51 stated) | 51 | 51 | grab.npz labels | OK | objects actually in the used trajectories |
| 26 | Intro 3.1 datasets | GRAB bundle fps 30 | 30 | 30 | grab.npz | OK |  |
| 27 | 3.2 prior | parameters 2,283,059 | 2,283,059 | 2,283,059 | results.json n_parameters | OK |  |
| 28 | 3.2 prior | epochs 120 | 120 | 120 | args | OK |  |
| 29 | 3.2 prior | KL warmup 12 epochs (= epochs//10) | 12 | 12 | experiment_paired_composition.py kl_warmup_epochs=epochs//10 | OK |  |
| 30 | 3.2 prior | stride 4 | 4 | 4 | args | OK |  |
| 31 | 3.2 prior | window 32 | 32 | 32 | args | OK |  |
| 32 | 3.2 prior | latent 12 | 12 | 12 | args | OK |  |
| 33 | 3.2 prior | hidden 256 | 256 | 256 | args | OK |  |
| 34 | 3.2 prior | lr 0.001 | 0.001 | 0.001 | args | OK |  |
| 35 | 3.2 prior | batch 256 | 256 | 256 | args | OK |  |
| 36 | 3.2 prior | validation 15 % | 0.15 | 0.15 | experiment_paired_composition.py | OK |  |
| 37 | 3.2 prior | target windows every 16 frames (window//2) | 16 | 16 | experiment_paired_composition.py target_ds stride | OK |  |
| 38 | 3.2 prior | config beta | 1.0 | 1.0 | configs/default.yaml | OK |  |
| 39 | 3.2 prior | config free_bits | 0.02 | 0.02 | configs/default.yaml | OK |  |
| 40 | 3.2 prior | config smoothness_weight | 0.1 | 0.1 | configs/default.yaml | OK |  |
| 41 | 3.2 prior | GRU n_layers 2 | 2 | 2 | latent_prior.py | OK |  |
| 42 | 3.2 prior | encoder bidirectional | True | True | latent_prior.py | OK |  |
| 43 | 3.2 prior | latent_prior.py default beta | 1.0 | 1.0 | latent_prior.py dataclass default | OK | the sweep script builds the model from these defaults, not the yaml |
| 44 | 3.2 prior | latent_prior.py default free_bits | 0.02 | 0.02 | latent_prior.py dataclass default | OK | the sweep script builds the model from these defaults, not the yaml |
| 45 | 3.2 prior | latent_prior.py default smoothness_weight | 0.1 | 0.1 | latent_prior.py dataclass default | OK | the sweep script builds the model from these defaults, not the yaml |
| 46 | 3.3-3.5 method | held cells four to six | 4-6 | 4-6 | axis_diagnostics n_held | OK |  |
| 47 | 3.3-3.5 method | arms share 7 % to 42 % | 7-42 | 7-42 | axis_diagnostics shared_training_fraction (10 Table-2 axes) | OK | raw 6.69 41.69 |
| 48 | 3.3-3.5 method | coverage > 0.8 and >= 3 per cell | 0.8 / 3 | 0.8 / 3 | verifier rule; see gate transcripts 'depth adequate (3.00 ...)' | OK | threshold read from gate transcript text |
| 49 | 3.3-3.5 method | 2,144 split checks, 0 mismatches | 2,144 / 0 | 2,144 / 0 | axis_diagnostics split_reproduction | OK |  |
| 50 | 3.3-3.5 method | synthetic 3,000 trajectories | 3,000 | 3,000 | synthetic_big.npz | OK |  |
| 51 | 3.3-3.5 method | synthetic mean 211 frames | 211 | 211 | synthetic_big.npz | OK | raw 211.43 |
| 52 | 3.3-3.5 method | synthetic fps 30 | 30 | 30 | synthetic_big.npz | OK |  |
| 53 | 3.3-3.5 method | nine grasp poses | 9 | 9 | meta n_primitives | OK |  |
| 54 | 3.3-3.5 method | 81 ordered pairs occur | 81 | 81 | synthetic_big labels | OK |  |
| 55 | 3.3-3.5 method | noise 7 deg rank 6 | 7.0 / 6 | 7.0 / 6 | meta | OK |  |
| 56 | 3.3-3.5 method | sensor noise 0.4 deg | 0.4 | 0.4 | meta | OK |  |
| 57 | 3.3-3.5 method | 3 to 7 segments | 3-7 | 3-7 | synthetic_big labels (segments = poses-1) | OK |  |
| 58 | 3.3-3.5 method | planted 18 deg | 18.0 | 18.0 | synth_hard meta | OK |  |
| 59 | 3.3-3.5 method | planted GRAB 53 of 70 cells | 53 of 70 | 53 of 70 | planted_cells.json | OK |  |
| 60 | 3.3-3.5 method | GRAB planted stride 32 | 32 | 32 | args | OK |  |
| 61 | 3.3-3.5 method | GRAB unmodified comparison stride 32 | 32 | 32 | args | OK |  |
| 62 | 3.3-3.5 method | TACO grid 16 % occupied | 16 | 16 | taco labels | OK | raw 16.47; taco_build occupancy 16.47 |
| 63 | 3.3-3.5 method | TACO candidate cells 17 | 17 | 17 | axis_diagnostics | OK |  |
| 64 | 3.3-3.5 method | TACO 151 triplets | 151 | 151 | taco_build.json | OK |  |
| 65 | 3.3-3.5 method | TACO permuted factor: at most 3 usable cells | at most 3 | max 10 over the 200 draws (last draw 3; distribution 1:3, 2:13, 3:39, 4:62, 5:40, 6:33, 7:6, 8:3, 10:1) | re-run of permute_grid's factor permutation (seed 12345, min_chains 4) on taco_action_tool.npz | MISMATCH | the script's refusal message prints the LAST draw's pool, not the maximum |
| 66 | 3.6 joined clips | 334 trajectories both | 334 / 334 | 334 / 334 | bundles | OK |  |
| 67 | 3.6 joined clips | 33 categories both | 33 / 33 | 33 / 33 | build_oakink_alignment_v4.json | OK |  |
| 68 | 3.6 joined clips | per-category counts within one | 1 | 1 | build json | OK |  |
| 69 | 3.6 joined clips | mean lengths 204 and 200 | 204 / 200 | 204 / 200 | bundles | OK | raw 203.67 199.99 |
| 70 | 3.6 joined clips | windows 2,787 and 2,728 | 2,787 / 2,728 | 2,787 / 2,728 | marginal_loss_volume.json extra | OK |  |
| 71 | 3.6 joined clips | 770 clips | 770 | 770 | build json | OK |  |
| 72 | 3.6 joined clips | four-frame cross-fade | 4 | 4 | window_classes.json | OK |  |
| 73 | 3.6 joined clips | budget 64 | 64 | 64 | args | OK |  |
| 74 | 3.7-3.8 | TACO held cells 5 | 5 | 5 | axis_diagnostics | OK |  |
| 75 | 3.7-3.8 | Bonferroni: all four < 1e-5 after x4 | True | True | welch | OK | max p x4 = 2.77e-06 |
| 76 | 3.7-3.8 | random effects: all four < 0.002 | True | True | cell_random_effects.json | OK | max p 0.0011 |
| 77 | 3.7-3.8 | min_chains statement: four (five on GRAB fine intent) | 4; GRAB-fine 5 | 4; GRAB-fine 5; transitions 6; (control 4) | results.json args.min_chains | MISMATCH | oakink2_transitions_rep/oakink2_transitions_s4 use min_chains 6; manuscript Appendix A omits it |
| 78 | 4.1 calibration | control +2.59 % over 20 seeds | 2.59 / 20 | 2.59 / 20 | pc_easy_rerun@256 | OK |  |
| 79 | 4.1 calibration | control SD 2.30 | 2.30 | 2.30 | pc_easy_rerun@256 | OK |  |
| 80 | 4.1 calibration | control64 +2.96 SD 2.34 20 seeds | 2.96 / 2.34 / 20 | 2.96 / 2.34 / 20 | pc_easy_rerun@64 | OK |  |
| 81 | 4.1 calibration | both above zero (one-sample p) | <0.05 | 7.2e-05 / 1.8e-05 | t-test vs 0 | INFO | ok if both < 0.05 |
| 82 | 4.1 calibration | planted synthetic +9.12 over 18 | 9.12 / 18 | 9.12 / 18 | pc_hard_rerun | OK |  |
| 83 | 4.1 calibration | planted +6.5 points | 6.5 | 6.5 | welch(hard,c256) | OK | raw 6.5278 |
| 84 | 4.1 calibration | planted CI 4.5 to 8.6 | 4.5 to 8.6 | 4.5 to 8.6 | welch | OK | raw 4.4736 8.5820 |
| 85 | 4.1 calibration | planted p 4.7e-7 | 4.7e-7 | 4.7e-7 | welch | OK | raw 4.672e-07 |
| 86 | 4.1 calibration | GRAB planted +13.15 % over 40 | 13.15 / 40 | 13.15 / 40 | grab_planted_s32 | OK |  |
| 87 | 4.1 calibration | GRAB unmodified s32 +0.71 % | 0.71 | 0.71 | grab_shape_v2 (both) | OK | all perframe seeds: 0.71 n=40 |
| 88 | 4.1 calibration | planted vs unmodified p 3.1e-9 | 3.1e-9 | 3.1e-9 | welch | OK | raw 3.147e-09 |
| 89 | 4.1 calibration | planted GRAB +10.6 points above control | 10.6 | 10.6 | welch | OK | raw 10.5574 |
| 90 | 4.1 calibration | planted GRAB vs control p 3.2e-11 | 3.2e-11 | 3.2e-11 | welch | OK | raw 3.178e-11 |
| 91 | Table 1 | zero-truth row | +2.59 ¦ 2.30 ¦ 17/20 | +2.59 ¦ 2.30 ¦ 17/20 | pc_easy_rerun@256 | OK |  |
| 92 | Table 1 | planted row | +9.12 ¦ 3.66 ¦ 18/18 ¦ +6.53 (4.47 to 8.58) ¦ 4.7e-7 | +9.12 ¦ 3.66 ¦ 18/18 ¦ +6.53 (4.47 to 8.58) ¦ 4.7e-7 | pc_hard_rerun | OK |  |
| 93 | Table 1 | OakInk permuted vs zero-truth | +3.65 ¦ 6.79 ¦ 27/40 ¦ +1.07 (-1.32 to 3.45) ¦ 0.37 | +3.65 ¦ 6.79 ¦ 27/40 ¦ +1.07 (-1.32 to 3.45) ¦ 0.37 | welch | OK | p raw 3.738e-01 |
| 94 | Table 1 | OakInk permuted vs real grid | -11.53 (-14.45 to -8.62) ¦ 5.8e-12 | -11.53 (-14.45 to -8.62) ¦ 5.8e-12 | welch | OK | p raw 5.814e-12; real mean 15.189 n=69 |
| 95 | Table 1 | GRAB permuted vs zero-truth | +1.38 ¦ 6.76 ¦ 24/40 ¦ -1.21 (-3.58 to 1.17) ¦ 0.31 | +1.38 ¦ 6.76 ¦ 24/40 ¦ -1.21 (-3.58 to 1.17) ¦ 0.31 | welch | OK | p raw 3.137e-01 |
| 96 | Table 1 | GRAB permuted vs real grid | +0.06 (-2.50 to 2.62) ¦ 0.96 | +0.06 (-2.50 to 2.62) ¦ 0.96 | welch | OK | p raw 9.640e-01; real mean 1.324 n=40 |
| 97 | Table 1 | TACO permuted cells vs zero-truth | -3.39 ¦ 6.46 ¦ 10/40 ¦ -5.97 (-8.27 to -3.68) ¦ 2.9e-6 | -3.39 ¦ 6.46 ¦ 10/40 ¦ -5.97 (-8.27 to -3.68) ¦ 2.9e-6 | welch | OK | p raw 2.866e-06 |
| 98 | Table 1 | TACO permuted cells vs real grid | -14.01 (-17.34 to -10.69) ¦ 2.3e-12 | -14.01 (-17.34 to -10.69) ¦ 2.3e-12 | welch | OK | p raw 2.329e-12; real mean 10.628 n=40 |
| 99 | Table 1 | caption: real grid has 69 seeds | 69 | 69 | oakink_official_category+rest both | OK | all perframe: 70 (mean 15.20) |
| 100 | Table 1 | caption: 40 seeds unless stated (permuted rows) | 40 | 40/40/40 | n | INFO |  |
| 101 | Table 1 | GRAB permuted exposure 0.38 against 0.30 | 0.38 / 0.30 | 0.38 / 0.30 | gates/sham_grid_cover_grab_shape_full.json | OK |  |
| 102 | Table 1 | GRAB real exposure (DIAG) 0.30 | 0.30 | 0.30 | axis_diagnostics | OK |  |
| 103 | Table 1 | TACO sham informed 0.0303 vs naive 0.0293 | 0.0303 / 0.0293 | 0.0303 / 0.0293 | sham_taco_action_tool | OK |  |
| 104 | Table 1 | real grids 11.5 and 14.0 above permuted | 11.5 / 14.0 | 11.5 / 14.0 | means | OK | raw 11.5350 14.0146 |
| 105 | Table 1 | declared reference: control > TACO sham | True | True | means | OK |  |
| 106 | Table 2 | row functional class x intent | 4,591 ¦ 4 ¦ 40 ¦ +22.2 ¦ 9.4 ¦ +19.2 to +25.2 ¦ 39/40 ¦ 0.0801 ¦ 0.40 ¦ < 1e-12 | 4,591 ¦ 4 ¦ 40 ¦ +22.2 ¦ 9.4 ¦ +19.2 to +25.2 ¦ 39/40 ¦ 0.0801 ¦ 0.40 ¦ < 1e-12 | ['oakink_class_rep'] + axis_diagnostics + marginal_loss_volume | OK | mean 22.2298 sd 9.4249 CI 19.2155..25.2440 p 1.430e-16 expo 0.3994 naive 0.080095 |
| 107 | Table 2 |   DIAG penalty_pct agrees (functional class x intent) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 108 | Table 2 | row category x intent | 4,591 ¦ 5 ¦ 69 ¦ +15.2 ¦ 8.3 ¦ +13.2 to +17.2 ¦ 66/69 ¦ 0.0672 ¦ 0.23 ¦ < 1e-12 | 4,591 ¦ 5 ¦ 69 ¦ +15.2 ¦ 8.3 ¦ +13.2 to +17.2 ¦ 66/69 ¦ 0.0672 ¦ 0.23 ¦ < 1e-12 | ['oakink_official_category', 'oakink_category_rest'] + axis_diagnostics + marginal_loss_volume | OK | mean 15.1893 sd 8.3098 CI 13.1930..17.1855 p 1.457e-18 expo 0.2262 naive 0.067199 |
| 109 | Table 2 |   DIAG penalty_pct agrees (category x intent) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 110 | Table 2 | row affordance x intent | 4,591 ¦ 5 ¦ 130 ¦ +12.2 ¦ 8.2 ¦ +10.8 to +13.6 ¦ 124/130 ¦ 0.0682 ¦ 0.24 ¦ < 1e-12 | 4,591 ¦ 5 ¦ 130 ¦ +12.2 ¦ 8.2 ¦ +10.8 to +13.6 ¦ 124/130 ¦ 0.0682 ¦ 0.24 ¦ < 1e-12 | ['oakink_official_attr', 'oakink_attr_more'] + axis_diagnostics + marginal_loss_volume | OK | mean 12.2012 sd 8.1789 CI 10.7819..13.6204 p 5.173e-19 expo 0.2391 naive 0.068165 |
| 111 | Table 2 |   DIAG penalty_pct agrees (affordance x intent) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 112 | Table 2 | row category x subject | 4,591 ¦ 4 ¦ 40 ¦ +10.3 ¦ 9.0 ¦ +7.4 to +13.2 ¦ 37/40 ¦ 0.0674 ¦ 0.16 ¦ 5.9e-6 | 4,591 ¦ 4 ¦ 40 ¦ +10.3 ¦ 9.0 ¦ +7.4 to +13.2 ¦ 37/40 ¦ 0.0674 ¦ 0.16 ¦ 5.9e-6 | ['oakink_category_subject'] + axis_diagnostics + marginal_loss_volume | OK | mean 10.2997 sd 9.0158 CI 7.4163..13.1831 p 5.898e-06 expo 0.1640 naive 0.067381 |
| 113 | Table 2 |   DIAG penalty_pct agrees (category x subject) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 114 | Table 2 | row action x tool | 8,144 ¦ 5 ¦ 40 ¦ +10.6 ¦ 8.3 ¦ +8.0 to +13.3 ¦ 33/40 ¦ 0.0321 ¦ 0.46 ¦ 6.9e-7 | 8,144 ¦ 5 ¦ 40 ¦ +10.6 ¦ 8.3 ¦ +8.0 to +13.3 ¦ 33/40 ¦ 0.0321 ¦ 0.46 ¦ 6.9e-7 | ['taco_action_tool'] + axis_diagnostics + marginal_loss_volume | OK | mean 10.6278 sd 8.3322 CI 7.9631..13.2926 p 6.923e-07 expo 0.4597 naive 0.032089 |
| 115 | Table 2 |   DIAG penalty_pct agrees (action x tool) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 116 | Table 2 | row shape x fine intent | 17,604 ¦ 6 ¦ 40 ¦ +1.3 ¦ 4.5 ¦ -0.1 to +2.8 ¦ 23/40 ¦ 0.0434 ¦ 0.30 ¦ 0.15 | 17,604 ¦ 6 ¦ 40 ¦ +1.3 ¦ 4.5 ¦ -0.1 to +2.8 ¦ 23/40 ¦ 0.0434 ¦ 0.30 ¦ 0.15 | ['grab_shape_s4'] + axis_diagnostics + marginal_loss_volume | OK | mean 1.3240 sd 4.4917 CI -0.1125..2.7605 p 1.546e-01 expo 0.3018 naive 0.043375 |
| 117 | Table 2 |   DIAG penalty_pct agrees (shape x fine intent) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 118 | Table 2 | row shape x intent class | 17,604 ¦ 5 ¦ 40 ¦ +2.3 ¦ 6.2 ¦ +0.3 to +4.2 ¦ 29/40 ¦ 0.0436 ¦ 0.31 ¦ 0.77 | 17,604 ¦ 5 ¦ 40 ¦ +2.3 ¦ 6.2 ¦ +0.3 to +4.2 ¦ 29/40 ¦ 0.0436 ¦ 0.31 ¦ 0.77 | ['grab_shapeclass'] + axis_diagnostics + marginal_loss_volume | OK | mean 2.2683 sd 6.1736 CI 0.2938..4.2427 p 7.732e-01 expo 0.3146 naive 0.043575 |
| 119 | Table 2 |   DIAG penalty_pct agrees (shape x intent class) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 120 | Table 2 | row scene x verb | 57,919 ¦ 5 ¦ 40 ¦ +3.0 ¦ 7.5 ¦ +0.5 to +5.4 ¦ 28/40 ¦ 0.0166 ¦ 0.11 ¦ 0.78 | 57,919 ¦ 5 ¦ 40 ¦ +3.0 ¦ 7.5 ¦ +0.5 to +5.4 ¦ 28/40 ¦ 0.0166 ¦ 0.11 ¦ 0.78 | ['oakink2_scene_verb'] + axis_diagnostics + marginal_loss_volume | OK | mean 2.9508 sd 7.5116 CI 0.5484..5.3531 p 7.801e-01 expo 0.1132 naive 0.016614 |
| 121 | Table 2 |   DIAG penalty_pct agrees (scene x verb) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 122 | Table 2 | row scene x primitive | 57,919 ¦ 5 ¦ 40 ¦ +2.3 ¦ 7.3 ¦ -0.0 to +4.6 ¦ 25/40 ¦ 0.0180 ¦ 0.12 ¦ 0.82 | 57,919 ¦ 5 ¦ 40 ¦ +2.3 ¦ 7.3 ¦ -0.0 to +4.6 ¦ 25/40 ¦ 0.0180 ¦ 0.12 ¦ 0.82 | ['oakink2_scene_primitive'] + axis_diagnostics + marginal_loss_volume | OK | mean 2.2967 sd 7.2917 CI -0.0353..4.6287 p 8.186e-01 expo 0.1232 naive 0.018009 |
| 123 | Table 2 |   DIAG penalty_pct agrees (scene x primitive) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 124 | Table 2 | row annotated transitions | 57,919 ¦ 6 ¦ 40 ¦ +12.2 ¦ 10.0 ¦ +9.0 to +15.4 ¦ 36/40 ¦ 0.0216 ¦ 0.13 ¦ 6.2e-7 | 57,919 ¦ 6 ¦ 40 ¦ +12.2 ¦ 10.0 ¦ +9.0 to +15.4 ¦ 36/40 ¦ 0.0216 ¦ 0.13 ¦ 6.2e-7 | ['oakink2_transitions_rep'] + axis_diagnostics + marginal_loss_volume | OK | mean 12.1985 sd 10.0286 CI 8.9912..15.4058 p 6.240e-07 expo 0.1294 naive 0.021609 |
| 125 | Table 2 |   DIAG penalty_pct agrees (annotated transitions) | 0 | 0 | axis_diagnostics per_seed.penalty_pct | OK |  |
| 126 | Table 2 | text: null axes CI reaches at most +3.0 points | 3.0 | 3.0 | welch(x,c256).hi over 4 null axes | OK | [('shape x fine intent', np.float64(0.49)), ('shape x intent class', np.float64(1.891)), ('scene x verb', np.float64(2.96)), ('scene x primitive', np.float64(2.242))] |
| 127 | Table 2 | category x intent if all perframe seeds (n=70) | n/a | n=70 mean +15.20 (vs 69: +15.19) | info | INFO | manuscript explains 69 = both architectures finished |
| 128 | Table 2 | shape x intent class both vs all | same | same | info | OK |  |
| 129 | Table 2 | scene x verb both vs all | same | same | info | OK |  |
| 130 | Table 2 | affordance both vs all | same | same | info | OK |  |
| 131 | Table 2 | caption originals +17.9 (class) and +16.7 (transitions), 12 seeds | 17.9 / 16.7 / 12 / 12 | 17.9 / 16.7 / 12 / 12 | oakink_class_v1 (both), oakink2_transitions_s4 | OK | class_v1 all perframe 17.92 n=12 |
| 132 | Table 2 | text: 4,591 / 8,144 / 17,604 / 57,919 | 4,591 8,144 17,604 57,919 | 4,591 8,144 17,604 57,919 | marginal_loss_volume | OK |  |
| 133 | Table 2 | all OakInk-Image rows same windows | 4,591 x4 | 4,591,4,591,4,591,4,591 | marginal_loss_volume | INFO |  |
| 134 | Table 2 | all OakInk2 rows same windows | 57,919 x3 | 57,919,57,919,57,919 | marginal_loss_volume | INFO |  |
| 135 | Table 2 | GRAB rows same windows | 17,604 x2 | 17,604,17,604 | marginal_loss_volume | INFO |  |
| 136 | Table 2 | text: OakInk2 prim +8.56 at 64 vs +2.30 at 256 | 8.56 / 2.30 | 8.56 / 2.30 | oakink2_scene_primitive(_b64) | OK |  |
| 137 | Table 2 | text: OakInk-Image +10.15 at 64 vs +15.19 at 256 | 10.15 / 15.19 | 10.15 / 15.19 | pc_oakink_b64, cat(69) | OK |  |
| 138 | Table 2 | text: transitions +12.2 | 12.2 | 12.2 | oakink2_transitions_rep | OK |  |
| 139 | Table 2 | text: GRAB at 64 about 4,400 windows | 4,400 | 4,401 | 17,604/4 | OK (stated as 'about') | expected windows scale with budget |
| 140 | Table 2 | text: GRAB b64 +3.42 (SD 7.56, 40 seeds) | 3.42 / 7.56 / 40 | 3.42 / 7.56 / 40 | grab_shape_b64@64 | OK |  |
| 141 | Table 2 | text: GRAB b64 diff +0.5, CI -2.2 to 3.1 | +0.5 / -2.2 to 3.1 | +0.5 / -2.2 to 3.1 | welch(gb64,c64) | OK | raw d 0.4575 lo -2.1612 hi 3.0762 p 0.727 |
| 142 | Table 2 | text: unmodified clips at 64 about 1,150 windows | 1,150 | 1,148 | marginal_loss_volume extra | OK (stated as 'about') | raw 1147.7; 4,591/4=1147.7 |
| 143 | Table 3 / 4.3 | T3 Unmodified | +10.15 ¦ 6.87 ¦ 36/40 ¦ 2.1e-7 | +10.15 ¦ 6.87 ¦ 36/40 ¦ 2.1e-7 | pc_oakink_b64 / pair_aligned / pair_misaligned @64 vs c64 | OK | p raw 2.070e-07 mean 10.1457 |
| 144 | Table 3 / 4.3 | T3 Aligned | +17.03 ¦ 11.53 ¦ 37/40 ¦ 2.5e-9 | +17.03 ¦ 11.53 ¦ 37/40 ¦ 2.5e-9 | pc_oakink_b64 / pair_aligned / pair_misaligned @64 vs c64 | OK | p raw 2.472e-09 mean 17.0256 |
| 145 | Table 3 / 4.3 | T3 Misaligned | +3.87 ¦ 5.41 ¦ 31/40 ¦ 0.37 | +3.87 ¦ 5.41 ¦ 31/40 ¦ 0.37 | pc_oakink_b64 / pair_aligned / pair_misaligned @64 vs c64 | OK | p raw 3.697e-01 mean 3.8710 |
| 146 | Table 3 / 4.3 | mis below al 13.2 (CI 9.1 to 17.2, p 2.1e-8) | 13.2 / 9.1 to 17.2 / 2.1e-8 | 13.2 / 9.1 to 17.2 / 2.1e-8 | welch(mis,al) | OK | raw 13.1546 9.1193 17.1900 p 2.139e-08 |
| 147 | Table 3 / 4.3 | mis below base 6.3 (CI 3.5 to 9.0, p 2.2e-5) | 6.3 / 3.5 to 9.0 / 2.2e-5 | 6.3 / 3.5 to 9.0 / 2.2e-5 | welch(mis,base) | OK | raw 6.2747 3.5198 9.0296 p 2.151e-05 |
| 148 | Table 3 / 4.3 | mis vs control diff +0.9, CI -1.1 to 2.9 | +0.9 / -1.1 to 2.9 | +0.9 / -1.1 to 2.9 | welch(mis,c64) | OK | raw 0.9064 -1.1010 2.9138 p 0.370 |
| 149 | Table 3 / 4.3 | al above base +6.9, CI 2.6 to 11.1, p 0.0019 | +6.9 / 2.6 to 11.1 / 0.0019 | +6.9 / 2.6 to 11.1 / 0.0019 | welch(al,base) | OK | raw 6.8799 2.6401 11.1197 p 0.00189 |
| 150 | Table 3 / 4.3 | coverage fails (2 aligned, 5 misaligned) | 2 / 5 | 2 / 5 | gates/cover_oakink_pair_*.txt status | OK | recomputed from columns: 2 / 5; seeds [3, 29] [8, 15, 18, 22, 25] |
| 151 | Table 3 / 4.3 | gate-passing +16.46 and +4.89, 11.6 apart, p 3.5e-7 | 16.46 / 4.89 / 11.6 / 3.5e-7 | 16.46 / 4.89 / 11.6 / 3.5e-7 | by seed minus gate failures | OK | n 38 35 raw diff 11.5751 p 3.541e-07 |
| 152 | Table 3 / 4.3 | gate-passing misaligned vs control p 0.056 | 0.056 | 0.056 | welch(m2,c64) | OK | raw 0.05574 |
| 153 | Table 3 / 4.3 | ten seeds averaged | 10 | 10 | constituent_leak_pair.json | OK |  |
| 154 | Table 3 / 4.3 | 8.4 % naive clips in held cells | 8.4 | 8.4 | constituent_leak_pair.json | OK | raw 8.359 |
| 155 | Table 3 / 4.3 | 27.9 % targets with clip seen by naive | 27.9 | 27.9 | constituent_leak_pair.json | OK | raw 27.926 |
| 156 | Table 3 / 4.3 | informed held share 41.9 -> 30.8 | 41.9 / 30.8 | 41.9 / 30.8 | constituent_leak_pair.json | OK |  |
| 157 | Table 3 / 4.3 | aligned naive figures 0 % | 0 / 0 | 0 / 0 | constituent_leak_pair.json | OK |  |
| 158 | Table 3 / 4.3 | 61 % target clips in held cell (mis), 100 % aligned | 61 / 100 | 61 / 100 | constituent_leak_pair.json | OK | raw 60.98 |
| 159 | Table 3 / 4.3 |   per_seed mean check naive.c (oakink_pair_aligned.npz) | 0.00 | 0.00 | constituent_leak_pair per_seed | OK |  |
| 160 | Table 3 / 4.3 |   per_seed mean check naive.c (oakink_pair_misaligned.npz) | 8.36 | 8.36 | constituent_leak_pair per_seed | OK |  |
| 161 | Table 4 / window classes | aligned wc seeds 40 | 40 | 40 | window_classes.json | OK |  |
| 162 | Table 4 / window classes |   aligned wc naive mse_target matches results.json | 0 | 0 | window_classes vs results.json | OK |  |
| 163 | Table 4 / window classes |   aligned class-weighted mse == total (max abs) | ~0 | 1.2e-08 | window_classes.json | INFO | sanity: classes partition the windows |
| 164 | Table 4 / window classes | misaligned wc seeds 40 | 40 | 40 | window_classes.json | OK |  |
| 165 | Table 4 / window classes |   misaligned wc naive mse_target matches results.json | 0 | 0 | window_classes vs results.json | OK |  |
| 166 | Table 4 / window classes |   misaligned class-weighted mse == total (max abs) | ~0 | 1.1e-08 | window_classes.json | INFO | sanity: classes partition the windows |
| 167 | Table 4 / window classes | re-run totals +16.97 and +3.92 | 16.97 / 3.92 | 16.97 / 3.92 | pc_oakink_pair_*_wc results.json | OK |  |
| 168 | Table 4 / window classes | T4 aligned | +17.75 (12.57) ¦ +16.51 (12.70) ¦ +16.07 (11.68) | +17.75 (12.57) ¦ +16.51 (12.70) ¦ +16.07 (11.68) | window_classes.json | OK |  |
| 169 | Table 4 / window classes | T4 misaligned | +5.64 (6.71) ¦ +5.51 (8.07) ¦ +1.54 (7.58) | +5.64 (6.71) ¦ +5.51 (8.07) ¦ +1.54 (7.58) | window_classes.json | OK |  |
| 170 | Table 4 / window classes | A1-A2 paired +1.7, p 0.079 | +1.7 / 0.079 | +1.7 / 0.079 | paired t | OK | raw 1.6795 p 0.07895 |
| 171 | Table 4 / window classes | M1 below A1 12.1 (CI 7.6 to 16.6, p 1.3e-6) | 12.1 / 7.6 to 16.6 / 1.3e-6 | 12.1 / 7.6 to 16.6 / 1.3e-6 | welch | OK | raw 12.1120 7.6049 16.6192 p 1.336e-06 |
| 172 | Table 4 / window classes | 82 % loss above control | 82 | 82 | (A1-M1)/(A1-c64) | OK | raw 81.906 |
| 173 | Table 4 / window classes | bootstrap CI 65 % to 97 % (seed 0) | 65 / 97 | 65 / 97 | percentile bootstrap, 100k, independent resampling of A1, M1, c64 | OK | raw 64.65 97.07 |
| 174 | Table 4 / window classes | bootstrap CI 65 % to 97 % (seed 1) | 65 / 97 | 65 / 97 | percentile bootstrap, 100k, independent resampling of A1, M1, c64 | OK | raw 64.65 97.22 |
| 175 | Table 4 / window classes | bootstrap CI 65 % to 97 % (seed 7) | 65 / 97 | 65 / 97 | percentile bootstrap, 100k, independent resampling of A1, M1, c64 | OK | raw 64.55 97.19 |
| 176 | Table 4 / window classes | M1 above control +2.7 (CI 0.3 to 5.0, p 0.028) | +2.7 / 0.3 to 5.0 / 0.028 | +2.7 / 0.3 to 5.0 / 0.028 | welch | OK | raw 2.6756 0.3057 5.0456 p 0.02765 |
| 177 | Table 4 / window classes | M2 further 4.1 lower (paired p 0.0057) | 4.1 / 0.0057 | 4.1 / 0.0057 | paired t | OK | raw 4.0984 p 0.00571 |
| 178 | 4.4 TACO | median 4.93 s (2.27 to 16.57; 1.4 % over 10 s) | 4.93 / 2.27 / 16.57 / 1.4 | 4.93 / 2.27 / 16.57 / 1.4 | taco_build.json frames | OK |  |
| 179 | 4.4 TACO |   recomputed from bundle | 4.93 / 2.27 / 16.57 / 1.4 | 4.93 / 2.27 / 16.57 / 1.4 | taco_action_tool.npz lengths/30 | OK | frac>10s raw 1.424; >=300: 1.467 |
| 180 | 4.4 TACO | 151 triplets, 15 actions, 17 tools, 9 objects | 151 / 15 / 17 / 9 | 151 / 15 / 17 / 9 | taco_build.json | OK |  |
| 181 | 4.4 TACO |   triplets from labels | 151 | 151 | taco_action_tool labels (fine part) | OK | example brush¦brush¦bowl@brush@-@brush |
| 182 | 4.4 TACO | TACO +10.63 % over 40, 8.0 above (CI 5.2 to 10.9, p 6.9e-7) | 10.63 / 40 / 8.0 / 5.2 to 10.9 / 6.9e-7 | 10.63 / 40 / 8.0 / 5.2 to 10.9 / 6.9e-7 | welch(taco,c256) | OK | raw d 8.0402 lo 5.1995 hi 10.8809 p 6.923e-07 |
| 183 | 4.4 TACO | +10.76 without 6 coverage-failing seeds | 10.76 / 6 | 10.76 / 6 | axis_diagnostics coverage rule | OK | n kept 34; failing seeds [17, 21, 22, 26, 32, 37] |
| 184 | 4.4 TACO |   coverage_thin seeds in results.json (covered<held) | info | 6 seeds [17, 21, 22, 26, 32, 37] | results.json soundness | INFO | gate = covered/held>0.8 AND >=3 per cell; thin = covered<held |
| 185 | 4.4 TACO | 14.0 above TACO permuted | 14.0 | 14.0 | means | OK |  |
| 186 | 4.4 TACO | range -3.76 to +27.43 | -3.76 / 27.43 | -3.76 / 27.43 | taco per seed | OK |  |
| 187 | 4.4 TACO | IQR +4.32 to +15.77 (numpy linear) | 4.32 / 15.77 | 4.32 / 15.77 | np.percentile linear | OK | other defs: hazen 4.09-16.18; weibull 3.86-16.59 |
| 188 | 4.4 TACO | 8 of 40 at or below control mean; 7 exceed +20 | 8 / 7 | 8 / 7 | taco vs c256.mean() | OK |  |
| 189 | 4.4 TACO | discussion range -3.8 to +27.4 | -3.8 / 27.4 | -3.8 / 27.4 | taco | OK |  |
| 190 | Discussion | GRAB contact +2.22 %, p 0.78 | 2.22 / 0.78 | 2.22 / 0.78 | grab_grasp_s32 vs c256 | OK | raw p 0.7821 |
| 191 | Discussion | GRAB intent class +2.3 %, p 0.77 vs control | 2.3 / 0.77 | 2.3 / 0.77 | grab_shapeclass vs c256 | OK | raw p 0.7732; one-sample vs 0 p 0.0254 |
| 192 | Appendix A | TACO 5.84 % / 15.99 % | 5.84 / 15.99 | 5.84 / 15.99 | taco_build.json | OK |  |
| 193 | Appendix A | TACO DOF unusable 1 of 27 | 1 / 27 | 1 / 27 | taco_build.json | OK |  |
| 194 | Appendix A | 12 to 60 eligible cells | 12-60 | 12-60 | axis_diagnostics candidate_pool_size | OK | [12, 17, 20, 25, 28, 30, 35, 47, 48, 60] |
| 195 | Appendix A | coverage fails 16/130 affordance, 5/69 category, 6/40 TACO | 16 / 5 / 6 | 16 / 5 / 6 | axis_diagnostics coverage rule | OK |  |
| 196 | Appendix A | at most 3 of 40 elsewhere | <=3 | 3 | axis_diagnostics | INFO | {'functional class x intent': 3, 'category x subject': 3, 'shape x fine intent': 0, 'shape x intent class': 1, 'scene x verb': 2, 'scene x primitive': 2, 'annotated transitions': 0} |
| 197 | Appendix A | A1 OakInk 770 / 33 categories | 770 / 33 | 770 / 33 | oakink.npz + coarsen | OK |  |
| 198 | Appendix A | A1 clips under 40 frames dropped (min length) | >=40 | 40 | oakink.npz min length | INFO |  |
| 199 | Appendix A | A1 OakInk2 min length >= 40 | >=40 | 81 | oakink2.npz | INFO |  |
| 200 | Appendix A | A1 GRAB 1,048 / 30 fps | 1,048 / 30 | 1,048 / 30 | grab.npz | OK |  |
| 201 | Appendix A | A1 OakInk2 609 | 609 | 609 | oakink2.npz | OK |  |
| 202 | Appendix A | A1 TACO 151 / 2,317 | 151 / 2,317 | 151 / 2,317 | taco_build.json | OK |  |
| 203 | Appendix A | DOF health other datasets 1 to 6 (from notes; not recomputed here) | 1-6 | OakInk 1, GRAB 6, OakInk2 4, TACO 1 | scripts/check_dof_health.py on the four bundles | OK |  |
| 204 | Appendix A |   meta dof info oakink | info | {} | oakink.npz meta | INFO |  |
| 205 | Appendix A |   meta dof info grab | info | {} | grab.npz meta | INFO |  |
| 206 | Appendix A |   meta dof info oakink2 | info | {} | oakink2.npz meta | INFO |  |
| 207 | Appendix A |   meta dof info taco_action_tool | info | {} | taco_action_tool.npz meta | INFO |  |
| 208 | Appendix B | occupancy 76 % to 49 % | 76 / 49 | 76 / 49 | oakink vs oakink_sparse labels | OK | raw 75.76 49.00 |
| 209 | Appendix B | removed 8 categories, 243 of 770 | 8 / 243 / 770 | 8 / 243 / 770 | bundles | OK |  |
| 210 | Appendix B | +13.70 and +19.64 vs +15.19; p 0.39 and 0.0025 | 13.70 / 19.64 / 15.19 / 0.39 / 0.0025 | 13.70 / 19.64 / 15.19 / 0.39 / 0.0025 | pc_oakink_dofdamage, pc_oakink_sparse vs cat(69) | OK | raw p 0.3904 0.00250; vs cat_all(70): 0.386 0.0024 |
| 211 | Appendix B | DOF-damage bundle: 6 DOF pinned or dead | 6 | 6 (5 pinned + wrist_tz dead) | check_dof_health.py: 6 of 27 unusable | OK |  |
| 212 | Appendix B | stride 32 vs 4 GRAB p 0.71 | 0.71 | 0.71 | grab_shape_s4 vs grab_shape_v2(both) | OK | raw 0.7066; vs all-perframe 0.7066 |
| 213 | Appendix B | contact 8.5 s to 4.9 s; 58 of 1,048 dropped | 8.5 / 4.9 / 58 | 8.5 / 4.9 / 58 | grab.npz, grab_grasp.npz | OK |  |
| 214 | Appendix B | contact about 11,400 windows | 11,400 | 11,400 | marginal_loss_volume extra | OK | raw 11367.7 |
| 215 | Appendix B | +2.22 %, p 0.78 control, p 0.43 unmodified | 2.22 / 0.78 / 0.43 | 2.22 / 0.78 / 0.43 | grab_grasp_s32 | OK | raw 0.4281 |
| 216 | Appendix B | abs diff 0.0104 / 0.0035 / 0.0006 / 0.0004 | 0.0104 / 0.0035 / 0.0006 / 0.0004 | 0.0104 / 0.0035 / 0.0006 / 0.0004 | axis_diagnostics absolute_mse_target | OK |  |
| 217 | Appendix B |   abs diff recomputed from results.json | 0.0104 / 0.0035 / 0.0006 / 0.0004 | 0.0104 / 0.0035 / 0.0006 / 0.0004 | results.json | OK |  |
| 218 | Appendix B | exposure TACO 0.46, GRAB 0.30 and 0.31 | 0.46 / 0.30 / 0.31 | 0.46 / 0.30 / 0.31 | axis_diagnostics | OK |  |
| 219 | Appendix B | TACO most exposed | True | True | axis_diagnostics | OK |  |
| 220 | Appendix B | marginal loss 31-51 OakInk, 53 TACO, 30-31 GRAB, 49-52 OakInk2 | 31-51 / 53 / 30-31 / 49-52 | 31-51 / 53 / 30-31 / 49-52 | marginal_loss_volume | OK | {'functional class x intent': 50.98, 'category x intent': 31.33, 'affordance x intent': 31.47, 'category x subject': 38.77, 'shape x fine intent': 30.92, 'shape x intent class': 30.01, 'scene x verb': 48.98, 'scene x primitive': 52.16, 'action x tool': 52.84} |
| 221 | Appendix B | r +0.11 to +0.34 OakInk; TACO r +0.25 p 0.12 | +0.11 to +0.34 / +0.25 / 0.12 | +0.11 to +0.34 / +0.25 / 0.12 | marginal_loss_volume | OK | raw TACO p 0.1151 |
| 222 | Appendix B | primitive covers median 57 %; 259 of 609 single | 57 / 259 / 609 | 57 / 259 / 609 | label_coverage_oakink2.json | OK | raw 57.37 |
| 223 | Appendix B | 40 splits; 11.6 % recordings; 4.9 % frames; 40 % target frames | 40 / 11.6 / 4.9 / 40 | 40 / 11.6 / 4.9 / 40 | motion_contamination_oakink2.json | OK | per_seed recomputed: 11.59 4.92 40.21 |
| 224 | Appendix B | OakInk2 +2.30 at 256, +8.56 at 64, diff +6.3 p 0.0097; vs control p 0.011 | 2.30 / 8.56 / +6.3 / 0.0097 / 0.011 | 2.30 / 8.56 / +6.3 / 0.0097 / 0.011 | oakink2_scene_primitive(_b64) | OK | raw diff 6.2681 p 0.00969; vs c64 p 0.01106 |
| 225 | Appendix B | segments +8.93 at 256; p 0.89 vs whole@64 | 8.93 / 0.89 | 8.93 / 0.89 | oakink2_primseg_rep | OK | raw p 0.8948 |
| 226 | Appendix B | +5.19 at 896; diff +2.9, CI -0.5 to 6.3, p 0.097 | 5.19 / +2.9 / -0.5 to 6.3 / 0.097 | 5.19 / +2.9 / -0.5 to 6.3 / 0.097 | oakink2_primseg_b896 vs oakink2_scene_primitive | OK | raw d 2.8895 lo -0.5366 hi 6.3157 p 0.09714 |
| 227 | Appendix B | whole@64 frames vs segments@256 frames (notes: 59,806 vs 68,326) | info | whole@256 frames/arm 239,225; whole@64 = 59,806; seg mean len 266.9 x256 = 68,326; x896 = 239,140 | bundles / marginal_loss_volume | INFO | manuscript says 896 segments match whole@256 frame volume, 256 segments match whole@64 |
| 228 | Appendix C | affordance 70 to 130 seeds | 70 / 130 | 70 / 130 | oakink_official_attr, oakink_attr_more | OK |  |
| 229 | Appendix C | transitions +3.44 % at stride 16 | 3.44 | 3.44 | oakink2_paired_v2 (both) | OK | all perframe 3.44 n=12 |
| 230 | Appendix C | stride 16 in oakink2_paired_v2 args | 16 | 16 | args | OK |  |
| 231 | Appendix C | two axes first run at 12 seeds | 12 / 12 | 12 / 12 | oakink_class_v1, oakink2_transitions_s4 | OK |  |
| 232 | Appendix D | second set of models ¦ 70 ¦ +14.78 | 70 / +14.78 | 70 / +14.78 | ratematch_a-d perframe | OK |  |
| 233 | Appendix D | prefix ¦ 18 to 70 ¦ +14.30 to +17.81 | 18-70 / +14.30 to +17.81 | 18-70 / +14.30 to +17.81 | oakink_n70, n70_v2, coarse, noleak | OK | [(70, np.float64(14.8)), (70, np.float64(14.3)), (18, np.float64(15.24)), (18, np.float64(17.81))] |
| 234 | Appendix D | object x intent ¦ 5 ¦ +5.56 and -2.22 | 5 / +5.56 / -2.22 | 5 / +5.56 / -2.22 | pc_oakink | OK | n64 5 |
| 235 | Appendix D | DexYCB ¦ 12 ¦ +0.09 and +1.12 | 12 / +0.09 / +1.12 | 12 / +0.09 / +1.12 | dexycb_paired(_v2) | OK | n2 12 |
| 236 | Appendix D | contact channels ¦ 20 ¦ -0.03 and +0.90 | 20 / -0.03 / +0.90 | 20 / -0.03 / +0.90 | contact_full, contact_pose | OK | n2 20 |
| 237 | Appendix D | GRAB first run ¦ 2 ¦ -10.12 | 2 / -10.12 | 2 / -10.12 | grab_shape | OK |  |
| 238 | Appendix D | contact segment s4 ¦ 7 ¦ -1.35 | 7 / -1.35 | 7 / -1.35 | grab_grasp_s4 | OK |  |
| 239 | Appendix D | transitions s16 ¦ 12 ¦ +9.06 and +3.44 | 12 / +9.06 / +3.44 | 12 / +9.06 / +3.44 | oakink2_paired(_v2) | OK | n2 12 |
| 240 | Appendix D | segments ¦ 20 and 40 ¦ +8.97 and +7.14 | 20 / 40 / +8.97 / +7.14 | 20 / 40 / +8.97 / +7.14 | oakink2_primseg_s4, _recdisjoint | OK |  |
| 241 | Appendix D | joined clips ¦ 11 to 40 ¦ +3.03 to +57.24 | 11-40 / +3.03 to +57.24 | 11-40 / +3.03 to +57.24 | five earlier joined-clip sweeps | OK | [('pc_oakink_long', 12, np.float64(8.74)), ('pc_oakink_longaligned', 11, np.float64(57.24)), ('pc_oakink_long3_aligned', 40, np.float64(17.86)), ('pc_oakink_long3_misaligned', 40, np.float64(3.03)), ('pc_oakink_long3_misaligned_cm', 40, np.float64(3.05))] |
| 242 | Appendix D | Appendix D: segments rep (+8.93) listed in App. B, b896 (+5.19) in App. B | info | primseg_rep +8.93 n40; b896 +5.19 n40 | info | INFO |  |
| 243 | Appendix A | DOF health: 1 of 27 on TACO, 1 to 6 elsewhere | 1 / 1-6 | TACO 1, OakInk 1, OakInk2 4, GRAB 6 | check_dof_health.py | OK |  |
| 244 | Appendix D | 2.2 trajectories per fine cell (OakInk-Image object x intent) | 2.2 | 2.23 (770 / 345 fine labels) | oakink.npz labels | OK |  |
| 245 | Table A1 | 627 = 363 primitive + 264 complex | 627 | 627 | arithmetic | OK | published counts themselves not recomputable here |
| 246 | Table 2 | held cells per row equal args.held_compositions | 4,5,5,4,5,6,5,5,5,6 | 4,5,5,4,5,6,5,5,5,6 | results.json args + axis_diagnostics | OK |  |
| 247 | Table 2 | W rows have a PREREG file | 7 rows | 7 files present (class_replication, category_subject, taco_prediction, stride4_reruns, grab_shapeclass, oakink2_scene_verb, oakink2_transitions_replication) | runs/PREREG_*.md | OK | existence only; file dates not audited |
| 248 | Statistics | Welch df/CI implementation | - | independent implementation agrees with every printed CI and p | own code vs manuscript | OK |  |
| 249 | Whole text | numbers in PDF text vs Markdown source | - | every numeric token of the MD body/appendices found in the PDF text (differences are citation markers and table reflow only) | anon_text.txt vs PAPER_TMLR.md | OK | one non-numeric wording difference noted in section 4 below |
| 250 | Appendix B | 'matches the frame volume': segments@256 vs whole@64; segments@896 vs whole@256 | matches / matches | 68,326 vs 59,806 frames (+14 %); 239,140 vs 239,225 (0.04 %) | bundle mean lengths x budget; marginal_loss_volume | INFO | no number printed; the first 'matches' is loose |

## 3. Mismatches, with the exact sentence and the correct value

### M1. Section 3.5 (Permuted-grid controls)

Sentence: "TACO's action-by-tool grid is 16 % occupied, because a tool affords few actions, and permuting one
factor leaves at most 3 usable cells against 17."

Correct value: permuting the tool factor over the 151 triplets (the script's own procedure, seed 12345, 200
draws, a cell counted as usable when spanned by at least four triplets) leaves **at most 10** usable cells
against 17; 3 is the pool of the 200th draw only. Suggested wording: "permuting one factor leaves at most 10
usable cells against 17 in 200 draws". Source of the error: `scripts/experiment_paired_sham_grid.py`,
`permute_grid`, whose `ValueError` message reports `sham_pool` of the final iteration; copied into
`runs/PREREG_taco_prediction.md` ("left at most 3 cells spanned by four triplets, against 17 in the real grid, in
all 200 draws") and from there into the manuscript. The same sentence's "16 %" (16.47 %) and "17" are correct.

### M2. Appendix A (Datasets and conventions)

Sentence: "A held-out cell must span at least four distinct fine labels (five on GRAB's shape-by-fine-intent axis),
and 12 to 60 cells per axis are eligible, so the spread over seeds understates the uncertainty over which cells are
held."

Correct value: four on every axis except five on GRAB shape by fine intent (also used by the GRAB planted and
contact-segment sweeps) and **six on OakInk2 annotated transitions** (`--min-chains 6` in
`runs/oakink2_transitions_rep/results.json` args; the same in the 12-seed original and the stride-16 versions).
Suggested wording: "(five on GRAB's shape-by-fine-intent axis, six on OakInk2's transitions axis)". The "12 to 60"
is correct (candidate pool sizes 12, 17, 20, 25, 28, 30, 35, 47, 48, 60).

## 4. Observations that are not number errors

- **Section 3.8 wording differs between the PDF and the current Markdown.** PDF: "The settings and readings of
  every sweep in Sections 4.1, 4.3 and 4.4, and of the rows of Table 2 marked as such, were written down before
  training". Markdown (modified 10:58, after the 10:47 PDF text extraction): "of the permuted-grid and planted-GRAB
  sweeps of Section 4.1, of every sweep in Sections 4.3 and 4.4, ...". The PDF version claims pre-registration of
  the two synthetic controls as well; the Markdown version is the narrower one. Not a numerical mismatch, but the
  PDF should be rebuilt from the current source if the narrower wording is intended.
- **Appendix B, "matches the frame volume".** Segments at budget 256 give about 68,326 training frames per arm
  against 59,806 for whole recordings at 64 (14 % more); segments at 896 give 239,140 against 239,225 (a true
  match). No number is printed, so this is not a mismatch; "matches" is loose for the first pair.
- **"about 4,400" and "about 1,150" windows** recompute as 4,401 and 1,148 (expected windows scale with budget);
  both are correctly described as approximate.
- **GRAB intent class, "a test against zero would declare ... a compositional penalty"**: one-sample p against
  zero is 0.025, so the claim holds.
- **The 82 % bootstrap lower bound** is 64.55-64.65 % across RNG seeds; "65 %" is the correct rounding but sits
  about 0.1 from the boundary.
- **Coverage threshold** (more than 0.8 of held cells, mean of at least 3.0 per covered cell) read from
  `scripts/check_informed_coverage.py` lines 40 and 104; the coverage-failure counts (16/130, 5/69, 6/40, 2 and 5
  for the pair, at most 3 elsewhere) were recomputed from the per-seed covered/held/per-cell numbers, not from the
  transcripts' status words.
- Quantities that cannot be recomputed from the artefacts and were not audited: the published dataset counts in
  Table A1 (32 categories, 100 objects, 12 subjects, 792 clips, 1,334 sequences, 10 subjects, 120 Hz, 627 =
  363 + 264, 131 triplets, 20 categories, 2.5K sequences), LAMP's architecture figures (8-step history,
  2-dimensional latent), BABEL's 20 %, the "28 of 40" interim reading in Appendix C, and the values drawn in
  Figures 1-3.
