# Partial labels hide compositional difficulty in hand-motion data

Code, gate outputs, pre-registrations and every seed of every sweep behind the manuscript of this title (Mu-Hua Wang,
National Yang Ming Chiao Tung University; prepared for *Transactions on Machine Learning Research*, not yet submitted,
not posted to any preprint server).

**Project page:** https://maurice1128.github.io/projects/hand-composition.html

---

## The finding

Compositional generalisation is usually tested by holding out combinations of labels: every action and every tool
occurs in training, some of their pairings do not, and the model is scored on those pairings. The split is made on
labels. In public hand-motion datasets a correct label often describes only part of a recording, and then holding
out the label does not hold out the motion: it stays in training under other labels, and the difficulty is hidden.

- **Controlled manipulation (OakInk-Image).** Two clips are joined into one trajectory labelled by the first. When
  the label describes both clips the paired penalty is +17.0 % of the naive error; when it describes only the first,
  with trajectories, categories and training volume held fixed, it is +3.9 %, at most 2.9 points above a zero-truth
  control. The windows the label does describe lose 82 % of their penalty above the control (95 % CI 65 % to 97 %),
  so the change lies mostly in the training sets.
- **Calibration.** A synthetic dataset whose true penalty is zero returns +2.6 %; permuted label grids, planted
  interactions and per-seed distributions give each real dataset its own reference.
- **Out-of-sample prediction (TACO).** A penalty predicted before the data were opened appears: +10.6 % over 40
  seeds, while single seeds range from -3.8 % to +27.4 %.

The prior is a LAMP-style motion VAE ([arXiv:2607.06323](https://arxiv.org/abs/2607.06323)); its design is adopted,
not proposed. The error is a reconstruction error, not a generation error.

## Earlier version of this repository

Until 2026-09-13 this repository accompanied *Which Hand-Motion Composition Axes Are Worth Measuring*. Its central
claim, that compositional difficulty appears only where the two factors interact, was refuted by a later
pre-registered sweep (OakInk-Image category x subject: no interaction, +10.3 % penalty) and is withdrawn. The files of
that version are still here (`results/sweeps`, `results/*.png`, `GATES.md`, `scripts/screen_axes.py`) so its numbers
stay reproducible; nothing in the current manuscript rests on them.

## Layout

```
src/caredex/
  hand_model.py        27-DOF spec, limits, normalization, DIP/PIP coupling
  mano.py              chumpy-free MANO loader
  data/                oakink.py, oakink2.py, grab.py, taco.py, synthetic.py, pipeline.py (windowing),
                       base.py (bundles + registry), mano_retarget.py (convention inference)
  models/              latent_prior.py (the LAMP-style prior used throughout)
  train/               checkpoint.py (RNG state included), trainer.py

scripts/
  experiment_paired_composition.py      the paired split and penalty behind every reported number
  experiment_paired_sham_grid.py        permuted-grid controls (--sham-mode factor | cells)
  experiment_paired_window_classes.py   penalty by window class for the joined-clip bundles
  build_oakink_alignment_v4.py          the aligned / misaligned joined-clip bundles
  build_taco_bundle.py                  TACO loader and bundle (right hand, conventions inferred)
  build_grab_diagnostics.py             planted-interaction and contact-segment GRAB bundles
  check_composition_leak.py, check_informed_coverage.py, check_frame_leak*.py, check_constituent_leak.py
                                        the four gates
  axis_diagnostics.py                   re-derives every seed's split and checks it against the stored record
  marginal_loss_and_volume.py           training volume and factor loss per axis
  verify_tmlr_draft.py                  recomputes every number in the manuscript and checks it against the text
  plot_tmlr_figures.py, plot_tmlr_teaser.py, make_web_video_label_alignment.py   figures and video

results/
  sweeps/<name>/results.json            every seed of every sweep (79 sweeps); sham_grid.json and
                                        window_classes.json sidecars where they exist
  ALL_SWEEPS.md                         every sweep with its settings, mean penalty and where the manuscript uses it
  preregistration/PREREG_*.md           settings and readings written down before each sweep was trained
  gates/                                gate transcripts as they came out
  DIAGNOSTICS_RESULTS.md                the running record of each sweep read against its pre-registration
  axis_diagnostics_tmlr.json, marginal_loss_volume.json, constituent_leak_pair.json, planted_cells.json,
  motion_contamination_oakink2.json, label_coverage_oakink2.json, taco_build.json
                                        the analysis artefacts the manuscript's numbers come from
  figures_tmlr/                         the three manuscript figures
```

The scripts read `runs/<name>/results.json`; in this repository the same files are under `results/sweeps/<name>/`.

## Reproducing the numbers

```bash
.venv/Scripts/python.exe scripts/verify_tmlr_draft.py      # needs the manuscript and the bundles
```

Every statistic (means, SDs, Welch tests, intervals, seed counts) is recomputed from `results/sweeps/*/results.json`.
The descriptive checks (dataset sizes, medians, convention counts) read the trajectory bundles, which are not
distributed: they are rebuilt from the original datasets by the `build_*` scripts. Dataset paths default to
`D:\datasets\`; OakInk annotations are read from inside `anno_v2.1.zip` without extracting it.

Windows, `uv`-managed venv; torch comes from the cu128 index (the development GPU is Blackwell, sm_120).

## Conventions that will bite you

- Angles are degrees in native units, radians only inside FK.
- Models consume values normalized by the joint-limit box, not by dataset statistics.
- Splits are by trajectory, never by frame, and whole fine labels stay on one side.
- Validation is scored at the final beta, never the warmup value.
- Dataset conventions are inferred from the data, never assumed; on TACO the inference contradicts the dataset's own
  loader (MANO's mean pose is not added).
- Run the frame-leak and constituent-leak checks on any bundle built by joining or cutting recordings: a label-level
  check does not detect reused clips.

## Citation

```bibtex
@unpublished{wang2026labels,
  author = {Wang, Mu-Hua},
  title  = {Partial labels hide compositional difficulty in hand-motion data},
  note   = {Manuscript},
  year   = {2026}
}
```
