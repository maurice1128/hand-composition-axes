# Pre-registration: three OakInk-Image ablations, to name the property that produces difficulty

Written 2026-09-16 ~13:05, while the bundles are still being built and before any model is trained on any of them.

## Question

Compositional difficulty appears on all four OakInk-Image axes (+12 to +18% of naive error) and on no GRAB or OakInk2
axis. Ruled out so far: instrument insensitivity (a planted interaction on real GRAB poses reads +13.15%), window
stride (GRAB at stride 4 is still null), near-duplicate retrieval (`runs/nn_redundancy.json`), and, on GRAB, dilution
(grasp-only stays null). Supported, weakly: dilution on OakInk2 (cutting 82 s trajectories to 24.9 s primitive
segments moves +2.30% to +8.97%, p 0.042).

So the working hypothesis is that OakInk-Image measures difficulty because **its label describes the window being
scored**: 2.4 s clips, one object and one intent each. Rather than keep hunting for difficulty elsewhere, these three
ablations damage OakInk-Image one property at a time and re-run its confirmatory axis. Whichever damage removes the
difficulty names the property; if none does, the property is something we have not measured, which is also a result.

## Settings, identical for all three

The confirmatory axis's own settings, from `runs/oakink_official_category/results.json`:
`--granularity oakink_category --held-compositions 5 --min-chains 4 --min-per-composition 5 --budgets 256
--kinds perframe --window 32 --stride 4 --epochs 120`, seeds 0-39. Bundles from
`scripts/build_oakink_ablations.py`. Each bundle must pass the informed-coverage and composition-leak gates at those
settings before it is swept; a bundle that fails is reported as unswept, not quietly re-tuned.

## Amended 13:15, before any ablation result was read: seed counts and order

Concatenation multiplies the frames per trajectory about eighteenfold (200 training batches per epoch against 11 on
the unmodified bundle), so a long-clip seed costs about 36 minutes against about 2. Forty seeds would take a day and
consume every hour left before the manuscript deadline.

- `oakink_dofdamage` and `oakink_sparse` keep **40 seeds** and run first; each costs about 1.5 hours.
- `oakink_long` and `oakink_longaligned` use **12 seeds each**, the same count for both, so the pair stays
  comparable. Twelve seeds is what OakInk2's transitions axis used, and the OakInk2 primitive-segment sweep used 20.
- Nothing had been read from any ablation when this was decided; the only input was the first seed's batch count and
  epoch rate. The seed lists stay fixed from here.

At twelve seeds the interval on a difference is wide, so a null on a long variant will be reported with its interval
and not as evidence of no effect.

## Comparators

- The zero-truth control `runs/pc_easy_rerun`: +2.59% of naive error, n 20.
- The unmodified confirmatory axis: +15.19% over its 69 both-architecture seeds.

Statistic and tests as in Table II: per-seed penalty / naive mse_target x 100; Welch against the control; Welch
against the unmodified axis; one-sample t against zero.

## Readings, declared now

**1. `oakink_long` — clips concatenated into long recordings under one label.** Mimics OakInk2's structure: most
windows in a trajectory no longer show the labelled intent.

*Amended 13:10, before any sweep, after seeing the build report and no penalty:* concatenating clips of differing
intents also destroyed the intent's own marginal signal (screen `right_only` 0.045 -> 0.003, excess -0.0091), so a
null on this bundle alone cannot separate "the label no longer describes the window" from "the axis was degraded".
A paired control is therefore added and read first:

**1b. `oakink_longaligned` — the same concatenation, but only same-composition clips are joined.** Length grows
identically; the label still describes every window.
- **1b keeps the difficulty and 1 loses it:** label-to-window alignment is the property. This is the clean result
  and it matches the OakInk2 primitive-segment sweep.
- **Both lose it:** trajectory length alone is enough to hide difficulty, whatever the label says.
- **Both keep it:** neither length nor alignment explains OakInk-Image, and ablation 1's earlier reading is void.

The pair is read together. Ablation 1's reading below stands only if 1b keeps the difficulty.

*Build report for 1b, recorded 13:20 before any sweep:* the pair differs in more than alignment, so the reading above
is weaker than a clean control would give. 385 trajectories in both, but median length 535 frames against 743;
within-trajectory clip repetition 17.8% of slots against 4.8%; 75 coarse cells against 71; 48 unused source clips
against 3. The screen moved on both margins, not only the intended one: `right_only` 0.058 against the unmodified
0.045 and `oakink_long`'s 0.003, `left_only` 0.139 against 0.097, excess +0.0348 against +0.0238 unmodified and
-0.0091 for `oakink_long`. Coverage fails 7 of 40 seeds on breadth (depth fine, minimum 4.00; unmodified `oakink`
and `oakink_long` fail 3 of 40 at the same settings), and the leak gate passes 40 of 40 with a minimum naive pool of
268. Any conclusion drawn from the pair must state that 1b is longer-labelled but also shorter, more repetitive and
higher-interaction than 1.
- Falls to the control (Welch p >= 0.05 against it, and p < 0.05 against the unmodified axis): **label-to-window
  alignment is the property.** This is the result that would license a dataset-design rule, and it agrees with the
  OakInk2 primitive-segment sweep.
- Stays above the control and indistinguishable from the unmodified axis: length and alignment are not what
  OakInk-Image's difficulty rests on, and the OakInk2 result must be read as dataset-specific.

**2. `oakink_dofdamage` — GRAB's pinned and dead DOFs reproduced on OakInk-Image poses.**
- Falls to the control: retargeting quality is the property, and GRAB's and OakInk2's nulls are partly artefacts of
  their own damaged DOFs. That would also weaken every null in Table II.
- Stays above the control: retargeting quality does not explain the difference, and the nulls elsewhere stand.

**3. `oakink_sparse` — the category x intent grid thinned to GRAB-like occupancy (~27%).**
- Falls to the control: how densely the factor grid is sampled is the property.
- Stays above the control: sampling density does not explain it, consistent with the earlier occupancy correlation
  (r = -0.095 over eleven axes).

**Joint reading.** If exactly one ablation removes the difficulty, that property is the answer and the paper becomes a
dataset-design guideline. If more than one does, they are reported together as jointly sufficient, with no claim about
which matters more. If none does, the honest statement is that the difference between these datasets is not explained
by clip length, retargeting quality, or grid density, and the paper reports the dataset dependence as unexplained.

Each ablation changes more than its target property in small ways (concatenation changes trajectory count and
lengths; the DOF clamp shifts pose statistics; thinning removes data). Those are recorded with the build and must be
stated beside any conclusion. No seeds are added or removed after results are seen, and no ablation is re-tuned after
seeing its sweep.
