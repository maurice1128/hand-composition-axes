# The five gates, and what they actually say

Four of the five exist because an earlier version of this work returned plausible numbers it could not have earned. There is one gate per way the authors were fooled. **Two of them still exit non-zero on the released data**, and the paper reports them that way rather than tuning them until they pass.

Every file named below is in [`results/gates/`](results/gates/) unless stated otherwise. Read the file, not this summary.

---

## 1. Informed coverage — **exits 1**

`check_informed_coverage.py`, files `cover_*.txt`

**Catches:** held-out compositions that are absent from the informed set as well. If the informed model never saw the composition either, the penalty is pinned at zero by construction and a null result says nothing about compositional generalization.

**Status:** fails on five of the seven bundles checked.

| File | Verdict |
|---|---|
| `cover_oakink_category.txt` | 4 of 40 rows fail; depth adequate at 5.60 examples per covered composition, breadth is not, some rows reaching only 4 of the held compositions |
| `cover_oakink_attr.txt` | 8 of 40 rows fail; depth 6.40, some rows reaching only 4 |
| `cover_oakink_class.txt` | 6 of 24 rows fail; depth 7.00, some rows reaching only 3 |
| `cover_oakink_idprefix.txt` | 4 of 40 rows fail; depth 6.40 |
| `cover_dexycb_shape_fine.txt` | **24 of 24 rows fail**; depth 5.00, some rows reaching 1 of the held compositions |
| `cover_dexycb_fine.txt` | informed model sees 2.00 examples per held composition on average, barely more informed than naive |
| `cover_grab_shape.txt` | passes; 6 of 6 covered, 15.00 examples per composition at budget 256 |
| `cover_oakink2_fine.txt` | passes; 6 of 6 covered, 4.17 examples |

The paper's §V-B gives the failing rows and an exclusion analysis. The gate is not a formality: it is why the DexYCB axes cannot be read as evidence of anything.

## 2. Composition leak — **passes**

`check_composition_leak.py`, files `composition_leak_*.txt`

**Catches:** target trajectories whose exact fine composition was in the informed training set. Before the fix, 48 to 69 percent of targets leaked, against 0 percent for the naive arm, which manufactured the entire effect.

**Status:** **0.0% on both arms, every seed, all three granularities.** This gate is now a required pre-condition of any sweep; results predating it are not reported.

## 3. DOF health — **exits 1**

`check_dof_health.py`, file `rerun_dof_health.txt` (in `results/`)

**Catches:** a retargeting convention that pins a joint at its limit, so a degree of freedom carries no information and everything downstream is measured on fewer DOF than claimed. An assumed thumb abduction axis once pinned `thumb_mcp_abd` in **100%** of GRAB and DexYCB frames.

**Status:** still fails, on every bundle. After the fix, `thumb_mcp_abd` is pinned in **40%** of GRAB frames and **33%** of DexYCB frames, against the gate's own 25% threshold. It is not the only offender: `pinky_mcp_abd` is pinned 47 to 58 percent across bundles, `thumb_cmc_flex` 63 percent on GRAB, and `wrist_tz` is dead on DexYCB.

```
grab.npz  (319,228 frames)
  thumb_cmc_flex   range used 74.2%   PINNED 63%
  pinky_mcp_abd    range used 100.0%  PINNED 47%
  thumb_mcp_abd    range used 100.0%  PINNED 40%
  wrist_tz         range used 32.3%   DEAD sd=0.0041
  FAIL: 6 of 27 DOF unusable. Anything measured on this bundle is measured on 21 DOF.
```

The instruction in the gate's own output is the right one and was followed: fix the retargeting, do not widen the limits. The limits are anatomy.

## 4. Primitive collapse — **passes**

`check_primitive_collapse.py` is the gate and has to pass *before* a sweep. It trains its own small bank, so it leaves nothing behind about the models the paper reports; the manuscript says no standalone artifact is released and gives the reading in prose.

`report_primitive_collapse.py` derives that reading from the stored training histories, so the claim now has a file behind it: `primitive_collapse.txt`.

**Catches:** `primitives_used = 1.0`. A modular model that uses one primitive is a monolithic model with extra steps, and every comparison against the monolithic baseline is void. This happened, in the first run of the data-efficiency experiment.

**Status:** over the 733 K=12 modular models in the sweeps Table II reports, the minimum final-epoch `val_primitives_used` is **9.732 of 12**, and exactly **16 sit below 11.0, all of them OakInk2's**. The K=16 banks of §V-C and §V-D are measured against a different ceiling and are outside this scope.

## 5. Split balance — **passes on what it tests**

`check_split_balance.py`, file `split_balance_oakink.txt`

**Catches:** the risk that an unpaired design measures how hard the test set is rather than how novel the composition is.

**Status:** passes, at **+1.4%** mean relative variance difference across five seeds. **The per-seed spread is −6.1% to +10.7%, and the gate does not test it.** This gate guards a risk the corrected data does not show, which is why the paired design in `experiment_paired_composition.py` supersedes the unpaired one rather than relying on this gate.

---

## What none of them catch

`check_paper_numbers.py` guards the reporting the way these five guard the experiments: it refuses a statistic taken while a shard is still writing, requires every load-bearing figure to appear in the manuscript, and fails any gate artifact whose rows and verdict disagree.

**It cannot check what a sentence claims, only that the numbers in it exist.** Two defects it could not have caught are on record: a gate run on a seed outside its sweep's range, and a table reporting a gate by its rows rather than its exit code.
