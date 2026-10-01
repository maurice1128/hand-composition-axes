# Pre-registration: OakInk-Image category x subject (interaction vs dataset)

Written 2026-09-16 ~05:50, after the axis search and gates, before any model on this axis is trained.

## Why this axis

Every OakInk-Image axis swept so far has positive interaction excess and carries difficulty; every GRAB and OakInk2
axis carries none. "Difficulty needs positive interaction" and "difficulty is an OakInk-Image property" predict the
same nine rows. They disagree on an OakInk-Image axis without interaction. `scripts/search_oakink_axes.py`
(`runs/oakink_axis_search.json`) screened every non-nested pair of OakInk-Image factors. Category x subject is the only
non-positive pair with a workable split:
- Excess -0.0005, z -0.1, 103 occupied cells.
- It shares the category factor with the confirmatory category x intent axis (+0.0238, z 8.3, +15.2% of naive error).
- The other non-positive pairs are object x intent (no leak-free split) and object x subject (coverage fails 40 of 40).
  Object x date fails coverage 6 to 12 of 40.

## Choices made after seeing gate output, stated now

- **Held compositions: 4.** Coverage fails 5, 3 and 5 of 40 seeds at held 5, 4 and 3 (min_chains 4,
  min_per_composition 5, budget 256; transcripts `runs/gates/cover_oakink_category_subject*`). Held 4 was picked as the
  fewest failures; the failing seeds there are recorded in the transcript. The leak gate passes.
- **Label encoding.** The bundle `data/bundles/oakink_category_subject.npz` (poses identical to `oakink.npz`) stores
  labels as `fine@left@-@right` and is read with the `oakink2_scene_verb` label parser, because
  `coarsen_labels` has no generic OakInk-Image mode. The name is wrong but the parsing is the same.
- **Overlap between arms.** About 40% of target trajectories have the same object and intent, recorded by another
  subject, in both training sets. This is the left factor's main effect, available to both arms by design.

## Settings

`--bundle data/bundles/oakink_category_subject.npz --granularity oakink2_scene_verb --held-compositions 4
--min-chains 4 --min-per-composition 5 --budgets 256 --kinds perframe --window 32 --stride 4 --epochs 120`,
seeds 0-39, out `runs/oakink_category_subject`.

## Readings, declared now

Statistic and tests as in Table II:
- per-seed penalty / naive mse_target x 100;
- Welch against `runs/pc_easy_rerun`;
- Welch against category x intent (`oakink_official_category` + `oakink_category_rest`, the 69 seeds with both
  architectures);
- one-sample t against zero;
- the estimate excluding coverage-failing seeds, reported beside the full one.

**At the control (Welch p >= 0.05).** On OakInk-Image itself, removing the interaction removes the difficulty.
- The interaction reading gains its first within-dataset support.
- "Difficulty is an OakInk-Image property" is contradicted.

**Above the control (p < 0.05).** An OakInk-Image axis without interaction carries difficulty.
- Necessity is refuted.
- The dataset reading is supported.

**Magnitude.** It is compared descriptively with category x intent (+15.2%). No threshold is set.

No seeds are added or removed after results are seen.

## Disclosure: an interim value was seen at 28 of 40 seeds, 2026-09-16 08:05

`scripts/emit_table_rows.py` printed its table as a side effect of being imported for an unrelated lookup test while
this sweep was running, so its value at 28 seeds was seen: +10.8% of naive error, p against the control below 1e-4.

Nothing above was changed in response. The seed list stays 0-39, the tests stay as declared, and the reading is taken
at 40 seeds. The script now prints only when run directly. This is recorded because an interim look at a
pre-registered sweep is the same class of event as the optional stopping this project's earlier audit found, even
when no decision follows from it.
