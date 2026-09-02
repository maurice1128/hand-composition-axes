#!/bin/sh
# The control Sec V-A does not have, on the dataset where its absence shows.
#
# Sec III-B measures a tenfold gap in realised per-frame rate between the two
# arms: the baseline's latent carries 0.207 to 0.223 nats per frame across three
# sweeps while the modular arm's assignment channel carries 2.230 to 2.399.
#
# Two intermediate versions of this script were wrong about why, and the second
# error is worth recording because it nearly discarded a working control. It
# claimed the baseline's KL penalty was inactive, on the grounds that its
# per-dimension KL (0.0172 to 0.0186) sits below the 0.02 free-bits floor. That
# divides a summed KL by 12 and then reasons about individual dimensions;
# ``clamp`` is applied per dimension. ``kl_clamped`` exceeds 12 x 0.02 = 0.24 in
# 102/140, 45/46 and 19/20 of the baselines at the final training epoch, so at
# least one dimension is above the floor and carrying gradient, and
# ``val_active_units`` is 0.996 to 1.000, so nothing is collapsed. The penalty
# is ACTIVE and binds at its floor on most dimensions.
#
# So both controls are legitimate, and this script runs BOTH, because they move
# the rate from opposite sides and a single arm cannot separate "the rate gap
# explains the effect" from "this intervention broke one model":
#
#   perframe_rate  raises the baseline's free-bits floor to 0.186/dim (2.23
#                  nats/frame, the modular arm's measured rate).
#   modular_rate   charges the assignment channel the categorical KL,
#                  ln K - H(q), at weight beta.
#
# A caution on the second, declared here rather than discovered afterwards: the
# loss already carries ``+ 0.02 * entropy``, so adding ``+ 1.0 * (ln K -
# entropy)`` takes the net per-frame-entropy coefficient to -0.98. That is a sign
# flip of one of the four bank-shaping terms, and it will push assignments toward
# uniform for reasons that have nothing to do with composition.
#
# Run on pc_easy first. That bundle has no ``transition_via_deg``, so by
# ``synthetic.py``'s own construction its compositional penalty is zero -- and
# the instrument currently reports +0.00270 at p = 0.007 there, 26% of the
# confirmatory axis's difficulty. Whatever else this sweep settles, it is also
# the only place the difficulty metric can be checked against a known truth.
#
# DECLARED BEFORE RUNNING, and reported whichever way it comes out:
#   * pc_easy, budgets 64 and 256, the same held compositions as runs/pc_easy.
#     Twenty seeds, not five: five cannot bound a false-positive rate, and the
#     seed count is fixed here before any p value is seen.
#   * Both budgets are reported. The existing sweep gives +0.00174 with 5 of 5
#     at 256 and -0.00785 with 1 of 5 at 64 under the same rate gap, so a
#     single-budget result settles nothing.
#   * Four numbers decide it, all pre-named:
#       1. the two arms' realised rates after the intervention. If they have not
#          converged the control did not bind and this sweep is void, whatever
#          the penalties say.
#       2. VOID CONDITION, declared because of the entropy sign flip above: the
#          modular arm's reconstruction error must stay within 20% of its
#          unmatched value. A modular arm degraded by the intervention would
#          mimic a controlled one, and check_primitive_collapse.py only tests
#          primitives_used == 1, so it would not catch this.
#       3. the baseline's penalty on data whose true penalty is zero, over 20
#          seeds. That is the instrument's false-positive rate and the paper
#          currently has no estimate of it.
#       4. the paired advantage under matched rate, in both control arms.
#   * Declared reading. If the advantage SURVIVES rate matching, Sec V-A's
#     effect is structural after all and the confirmatory axis is worth
#     re-running the same way. If it DISAPPEARS, Sec V-A was measuring channel
#     rate and the modular claim leaves the paper.
#
# It is the experiment most likely to change what the paper claims, and the
# direction it most likely changes it in is against us.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

$PY -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/synthetic_big.npz \
    --out runs/pc_easy_ratematched \
    --budgets 64 256 --kinds perframe perframe_rate modular modular_rate \
    --held-compositions 8 --min-per-composition 5 \
    --epochs 120 --window 32 --stride 4 --batch-size 256 \
    --lr 0.001 --latent-dim 12 --hidden 256 --n-primitives 12 \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --fresh \
    > runs/pc_easy_ratematched_log.txt 2>&1
echo "[rate] sweep done at $(date)"

$PY - <<'CHECK'
import collections, glob, json, os, statistics as st
import numpy as np
from scipy import stats


def last(f):
    h = json.load(open(f))
    r = h["epochs"] if isinstance(h, dict) and "epochs" in h else h
    return r[-1] if isinstance(r, list) else r


mod = [last(f) for f in glob.glob("runs/pc_easy_ratematched/*modular*/history.json")]
base = [last(f) for f in glob.glob("runs/pc_easy_ratematched/*perframe*/history.json")]
if mod and base:
    rate = st.mean(m["val_usage_entropy"] - m["val_assign_entropy"] for m in mod)
    bk = st.mean(b["val_kl"] for b in base)
    print(f"[1] modular assignment rate {rate:.3f} nats/frame vs baseline {bk:.3f} "
          f"(was 2.230 vs 0.219, ratio 10.2x; now {rate / bk:.2f}x)")
    print("[1] the control bound" if rate < 4 * bk
          else "[1] WARNING: rate still unmatched; this control is VOID")

res = json.load(open("runs/pc_easy_ratematched/results.json"))["results"]
by = collections.defaultdict(dict)
for r in res:
    by[(r["budget"], r["seed"])][r["kind"]] = r["penalty"]
for bud in sorted({b for b, _ in by}):
    pf = [v["perframe"] for (b, _), v in by.items() if b == bud and "perframe" in v]
    d = [v["perframe"] - v["modular_rate"] for (b, _), v in by.items()
         if b == bud and "perframe" in v and "modular_rate" in v]
    print(f"[2] budget {bud}: baseline penalty on a zero-truth bundle "
          f"{np.mean(pf):+.5f}, p vs 0 = {stats.ttest_1samp(pf, 0).pvalue:.4f} (n={len(pf)})")
    print(f"[3] budget {bud}: paired advantage under matched rate {np.mean(d):+.5f}, "
          f"{sum(1 for x in d if x > 0)}/{len(d)} positive, "
          f"p = {stats.ttest_1samp(d, 0).pvalue:.4f}")
CHECK
tail -12 runs/pc_easy_ratematched_log.txt
