# Which Hand-Motion Composition Axes Are Worth Measuring

Code, gate outputs and stored results for *Which Hand-Motion Composition Axes Are Worth Measuring* — a measurement for the premise underneath modular motion priors, run over six axes of three public hand datasets.

**Project page:** https://maurice1128.github.io/projects/hand-composition.html
**Paper:** [`hand_motion_composition_axes.pdf`](https://maurice1128.github.io/assets/papers/hand_motion_composition_axes.pdf) — manuscript under review, not posted to any preprint server.

---

## What is measured

A modular motion prior decomposes hand motion into a bank of reusable primitives. That family of models rests on one premise: novel recombinations of those primitives stay usable. The premise is almost never measured.

An *axis* pairs two factors, object category crossed with intent for instance. The instrument is a **paired split**: the same target trajectories are scored under two priors differing only in whether their training data contained the target's composition. The gap is the compositional penalty.

```
penalty = MSE_naive(T) - MSE_informed(T)      per seed, same targets in both arms
paired  = penalty_baseline - penalty_modular
```

## What the paper claims

**A measurement, with five gates.** Four of the five exist because an earlier design returned plausible numbers it could not have earned, one gate per way the authors had been fooled. See [`GATES.md`](GATES.md): two of the five still exit non-zero, and the paper reports them that way.

**Difficulty needs interaction, not informative factors.** Every axis carrying measurable compositional difficulty has positive interaction between its two factors; none carries difficulty without it; and the dataset whose factors explain the most pose variance *on their own* carries no difficulty at all. Only one of the six axes could have falsified this, so it is one-sided evidence, not a law. Sufficiency is demonstrably false: GRAB's shape axis has interaction and returns nothing.

A cheap statistic (`scripts/screen_axes.py`) finds the interaction without training anything. It diagnoses presence well and size badly. Over the five axes swept before it chose one, *r* = 0.843 (*p* = 0.073), already not significant. The axis it ranked fifth of six then came back the hardest measured, and the association fell to *r* = 0.463 (*p* = 0.355).

**A null on hardware, and why that null is weak.** Sixteen of seventeen defined comparisons on a Shadow Hand fail to separate novel compositions from familiar ones, and the seventeenth runs backwards. The retargeting has no collision term, the banks were trained on the one dataset the instrument itself finds null, and the equivalence test fails its own sensitivity check.

## What the paper does not claim

**No positive architecture result at matched information rate.** The two arms differ elevenfold in realized per-frame information, 0.207 against 2.275 on the confirmatory axis. Re-swept with the baseline's rate raised to match, the raw penalty loses most of the advantage. A scale-normalized penalty appears to keep it, and the negative control disposes of that: on `pc_easy`, a bundle whose compositional penalty is **zero by construction**, the normalized metric returns +0.099 against the real axis's +0.111, 18 of 20 wins, *p* = 0.0004. Normalization divides by each arm's own reconstruction error, and on the control the matched baseline reconstructs 4.04x better against 1.41x on real data, so it inverts the confound instead of removing it.

The paper therefore states: **we claim no modular advantage at matched rate.** What survives is that where difficulty exists, a modular bank pays a smaller compositional penalty than a parameter-matched monolithic latent. What does not survive is attributing that to modular structure rather than to channel bandwidth.

**The task-level question was never answered.** Sampling both priors unconditionally gives the architectures no channel through which to differ on grasp retention. The experiment that would answer it, each prior as the action space for a policy trying to grasp, was not run.

**No dataset design rule.** Every axis returning a penalty belongs to one dataset, and across eleven screened axes no measured collection property predicts which.

---

## Layout

```
src/caredex/
  hand_model.py        27-DOF spec, limits, normalization, DIP/PIP coupling.
                       Single source of truth for the action layout.
  kinematics.py        stick-figure FK + capsule self-intersection (cheap proxy)
  mano.py              chumpy-free MANO loader
  mesh_collision.py    LBS + triangle-triangle self-intersection (the real check)
  data/                oakink.py, oakink2.py, grab.py, dexycb.py, synthetic.py,
                       pipeline.py (windowing), base.py (bundles + registry)
  models/              eigengrasp.py (PCA floor), latent_prior.py (monolithic,
                       LAMP-style), modular_prior.py (the primitive bank)
  train/               checkpoint.py (RNG state included), trainer.py
  validate/            ergonomics.py

scripts/               one entry point per step; every check_*.py is a gate
  experiment_paired_composition.py    the experiment behind every reported penalty
  screen_axes.py                      the interaction screen, no training required
  check_*.py                          the five gates, plus the reporting gate
  task_confirm.sh, task_comparison.sh the invocations behind the two main sweeps

results/
  axis_screen.json          the eleven screened axes
  dexpilot_transfer.json    the retargeting-baseline comparison
  equivalence.json          TOST, with its failed sensitivity check
  identity_pooled.json      primitive identity pooled over 13 runs
  sweeps/<name>/results.json  every seed of every sweep the paper reports
  gates/                    the gate outputs, including the two that exit 1
  *.png                     the four figures in the paper
```

`experiment_data_efficiency.py` is superseded and kept only so pre-2026-08 runs stay reproducible: its unpaired gap metric measures test-set difficulty rather than composition. Use `experiment_paired_composition.py`.

## Running it

Windows, `uv`-managed venv. torch comes from the cu128 index, since the development GPU is Blackwell (sm_120) and default PyPI wheels do not build for it; `pyproject.toml` pins the index.

```bash
.venv/Scripts/python.exe scripts/screen_axes.py                    # minutes, no training
.venv/Scripts/python.exe scripts/experiment_paired_composition.py --help
```

**Dataset paths are hard-coded defaults pointing at `D:\datasets\`** in `data/grab.py`, `data/grab_contact.py` and `data/oakink_meta.py`. They are overridable arguments rather than requirements, but they are not configured for any other machine. OakInk annotations are read from inside `anno_v2.1.zip` without extracting it; extracting takes hours on Windows and buys nothing.

Before any sweep, run `scripts/check_primitive_collapse.py`. A result of `primitives_used == 1` means the modular model is monolithic in disguise and the comparison is void.

## Conventions that will bite you

- Angles are degrees in native units, radians only inside FK.
- Models consume values normalized by the joint-limit box, **not** by dataset statistics, so a prior trained on one dataset stays meaningful on another.
- Splits are by trajectory, never by frame. Consecutive frames at 30 fps are near-duplicates, and a frame-level split inflates validation.
- Validation is scored at the final beta, never the warmup value. Scoring at the current beta compares a different objective each epoch and pins the best checkpoint to epoch 0. This was a real bug.
- `bounded_output=True` makes joint-limit satisfaction architectural, so zero limit violations is not evidence the model learned anatomy. The checks with teeth are DIP/PIP coupling and smoothness.
- Dataset conventions are inferred from the data, never assumed. The first version of `data/oakink.py` hard-coded a flexion sign, got it backwards, and pinned 41% of DOF values at their limits.
- The `category` granularity keys on the first character of an object id, which records which sub-collection an object came from rather than what it is. It is deprecated and kept only for reproducibility. New evidence uses `--granularity oakink_category`, read from OakInk's own `metaV2.zip`.

## Citation

```bibtex
@unpublished{wang2026composition,
  author = {Wang, Mu-Hua},
  title  = {Which Hand-Motion Composition Axes Are Worth Measuring},
  note   = {Manuscript under review},
  year   = {2026}
}
```
