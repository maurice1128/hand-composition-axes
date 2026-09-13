# Pre-registration: OakInk2 scene x verb

**Written before the sweep starts. Reported whatever it returns.**

## Why this axis

OakInk2 is the only dataset in this study whose transitions recur across enough
distinct chains for a leak-free paired split to hold anything out (pool 108 at
`min_chains=4`, four times any other axis), and its one screened axis is the
one informative necessity test the paper has. Its bundle labels are primitive
chains and carry no crossed factor, so `screen_oakink2_axes.py` crosses the
scene and subject recorded in each sequence name with the chain's first
primitive and with the task sentence's leading verb, and scores each pairing
with the same statistic as `screen_axes.py`.

| axis | excess | z | cells | coverage gate, held 5 |
|---|---|---|---|---|
| scene x primitive | +0.0040 | +2.9 | 62 | not a necessity test (positive) |
| **scene x verb** | **−0.0001** | **−0.1** | 55 | **2 of 40 rows fail**, depth 3.40 |
| subject x primitive | −0.0333 | −4.1 | 240 | 13 of 40 fail, depth 2.40 |
| subject x verb | −0.0360 | −4.7 | 220 | 8 of 40 fail, depth 2.20 |

Scene x verb is the one whose sweep is interpretable at the paper's settings: 2
failing rows of 40 is inside the band of the reported OakInk-Image axes (4, 8
and 6 of 24). Its excess is indistinguishable from zero rather than negative,
so like GRAB's shape x intent class it tests "difficulty without *positive*
interaction", which is what necessity claims, and not a strongly negative case.
The two subject axes are the strongly negative cases; they are re-gated at
held 3 and pre-registered separately if they pass.

## The declared run

```
scripts/experiment_paired_composition.py
  --bundle data/bundles/oakink2_scene_verb.npz
  --granularity oakink2_scene_verb
  --out runs/oakink2_scene_verb
  --budgets 256
  --kinds perframe modular
  --held-compositions 5 --min-chains 4 --min-per-composition 5
  --epochs 120
  --seeds 0..39
```

Forty seeds, fixed now. No interim looks, no extension, no second seed range.
`check_composition_leak.py` runs on the same bundle and granularity before the
numbers are read.

## The declared analysis

Difficulty is the `perframe` arm's compositional penalty against zero over the
40 paired seeds, one-sample *t*-test, two-sided, alpha 0.05: Table II's own
statistic.

## What each outcome means, declared now

- **Indistinguishable from zero:** a further informative observation for the
  necessity direction. Counted with OakInk2's existing axis and GRAB's shape x
  intent class, that direction rests on three observations across two
  datasets, and the paper says how many others were structurally untestable.
- **Positive and distinguishable from zero:** the necessity direction is
  falsified on an axis with no positive interaction, and the claim is
  withdrawn. Reported as prominently as the other outcome.
- **Negative and distinguishable from zero:** a split defect until shown
  otherwise; the gates are re-examined before anything is concluded.

The modular-versus-baseline contrast is computed for the survey figure and no
claim rests on it.
