# Pre-registration: purity or contamination? Scoring the two-clip bundles by window class

Written 2026-09-21 00:35, before any model of these sweeps is trained. Wrapper:
`scripts/experiment_paired_window_classes.py`.

## Why

On the two-clip OakInk-Image bundles the penalty fell from +17.03% (label describes both clips) to +3.87% (label
describes the first clip only). Two properties differ between the bundles and vary together (Table 3 of the draft):

- **purity** — 61% of misaligned target clips belong to a held-out cell, against 100%;
- **contamination** — 8.4% of the naive arm's training clips belong to held-out cells, against 0%.

A cold referee pointed out that the draft asserts the second ("the naive arm sees the held-out motion under other
labels") while an 8.4% dose for a 13-point fall points at the first. The two make different predictions about the
target windows that lie wholly inside the FIRST clip, which the label names in both bundles.

## Design

Re-run `pc_oakink_pair_aligned` and `pc_oakink_pair_misaligned` with identical settings and the **same seeds 0-39**
(`--granularity oakink_category --held-compositions 5 --min-chains 4 --min-per-composition 5 --budgets 64 --kinds
perframe --window 32 --stride 4 --epochs 120`), recording the error per window class: `first` (wholly in the first
clip, clear of the 4-frame cross-fade), `second` (wholly in the second clip), `straddle` (discarded, counted). A dry
run on ten seeds gave about 94 first-clip and 103 second-clip windows per seed in the misaligned bundle, none with
fewer than 20. Both bundles already passed label-leak, frame-leak (0.00%) and constituent-leak checks.

Reusing the original seeds is deliberate. The quantities read here are new, and the **total** penalty must reproduce
the original sweeps (+17.03% and +3.87%); if it does not, within about a point, training is not reproducible and
the per-class numbers are not read until that is understood.

## Readings, declared now

Per seed and class, penalty = (naive - informed) / naive x 100 on that class's windows. Seeds with fewer than 20
windows in a class are dropped from that class and counted.

Let A1, A2 be the aligned bundle's first- and second-clip penalties and M1, M2 the misaligned bundle's.

- **Sanity, aligned:** A1 and A2 are both clearly positive and do not differ (paired t-test over seeds). If they
  differ, position within a joined trajectory matters by itself and M1 must be compared with A1 only.
- **Purity carries the fall** if M1 does not differ from A1 (Welch, p >= 0.05) **and** M2 is below M1 (paired,
  p < 0.05). The draft's mechanism sentence is then rewritten: the fall comes from scoring motion that is not the
  held-out composition, and contamination of the naive arm is not needed to explain it.
- **Contamination carries it** if M1 is below A1 (p < 0.05) and M2 does not differ from M1.
- **Both act** if M1 is below A1 and M2 is below M1. The share of the fall each carries is reported descriptively
  as (A1 - M1) against (M1 - M2), with no test.
- **Neither reading applies** otherwise, and that is reported as such.

## What this cannot settle

First-clip windows are pure in their *label*; whether the naive arm saw that motion elsewhere is the contamination
under test, not something removed. Class means rest on about a third of the windows, so they are noisier than the
sweeps' penalties, and a non-difference between M1 and A1 is reported with its interval, not as equality. The result
concerns a constructed bundle on one dataset at a budget of 64.

No seeds are added or removed after results are seen.
