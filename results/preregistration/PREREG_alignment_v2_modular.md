# Pre-registration: alignment test v2 repeated with a second prior architecture (modular primitive bank)

Written 2026-09-17 ~01:30, after the perframe alignment test v2 finished and before any modular model on these
bundles is trained.

## Why

The perframe prior showed alignment is what makes OakInk-Image's compositional difficulty measurable
(`runs/DIAGNOSTICS_RESULTS.md` section 11). The finding is:

| sweep (perframe prior, budget 64) | % naive |
|---|---|
| unmodified baseline | +10.15 |
| aligned | +17.86 |
| misaligned | +3.03, at the +2.96% control |

A reviewer's first question is whether this is a property of that one prior. The repository already has a
structurally different prior, the modular primitive bank (`models/modular_prior.py`, `--kinds modular`), trained by
the same script. This repeats the test with it.

Scope note: the bank-versus-latent *comparison* remains inconclusive in this project, because the rate-matched
control reproduces 90% of the bank's advantage. This test makes no comparison between the two priors. It asks only
whether the bank, read on its own, shows the same aligned-versus-misaligned pattern.

## Settings

The same three bundles and settings as `runs/PREREG_oakink_alignment_v2.md`: `--granularity oakink_category
--held-compositions 5 --min-chains 4 --min-per-composition 5 --budgets 64 --window 32 --stride 4 --epochs 120`,
seeds 0-39, with `--kinds modular` and the script's default bank size. Output directories:

| directory | bundle |
|---|---|
| `runs/pc_oakink_b64_modular` | `oakink.npz` |
| `runs/pc_oakink_long3_aligned_modular` | `oakink_long3_aligned.npz` |
| `runs/pc_oakink_long3_misaligned_modular` | `oakink_long3_misaligned.npz` |

The bundles already passed the frame-leak, label-leak and coverage gates; the splits do not depend on the prior.

**Collapse gate.** `scripts/report_primitive_collapse.py` (or the stored histories) must show the bank using more
than one primitive on these runs. A collapsed bank is a latent model in disguise, and its result is then reported as
not a second architecture.

*Collapse check, 02:20, on the finished baseline only.* Penalties were not read. The check covers all 80 models
(40 seeds, naive and informed). Final-epoch `val_primitives_used` has a minimum of 10.0 of 12 and a median of 12.0.
One model is below 11 and none is below 2, so the bank is not collapsed. The same check will be applied to the other
two sweeps when they finish.

## Comparator

The zero-truth control is a modular run of the same synthetic control under the current split builder:
`runs/pc_easy_rerun_modular`. It uses `synthetic_big.npz` with `--granularity fine --held-compositions 8
--min-chains 4 --min-per-composition 5 --budgets 64 --kinds modular --epochs 120`, seeds 0-19, the settings of
`pc_easy_rerun` except the prior. It is added here, before any modular result exists. The perframe budget-64 control
(+2.96%) is reported beside it for reference only.

## Readings, declared now

The modular penalty is used, as a percentage of the modular naive error, with the same tests as the perframe version.

- **Replicates** if both (a) and (b) hold:
  - (a) misaligned is below aligned, p < 0.05;
  - (b) aligned is not below the modular baseline, p >= 0.05, or aligned is above it.
- **Does not replicate** if (a) fails.
- Any other pattern is reported as it falls.

No seeds are added or removed after results are seen.
