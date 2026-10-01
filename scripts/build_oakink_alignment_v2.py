"""Label-to-window alignment on OakInk-Image, version 2: no clip reuse.

Specification: ``runs/PREREG_oakink_alignment_v2.md``. The first long-clip pair
(``build_oakink_ablations.py``: ``oakink_long``, ``oakink_longaligned``) reused
each source clip in about four trajectories, so target frames sat verbatim in
training and both sweeps are void (``runs/DIAGNOSTICS_RESULTS.md`` section 10).

This builds, from ``data/bundles/oakink.npz`` with numpy seed 0:

``oakink_long3_aligned``     3 clips per trajectory, all sharing the category
                             AND the intent of the label; label = first clip's
                             ``object->intent``.
``oakink_long3_misaligned``  3 clips per trajectory sharing the category; the
                             2nd and 3rd have intents different from the
                             label's; label = first clip's ``object->intent``.

Rules, asserted on the output:
- every source clip is used in at most one trajectory of a bundle;
- no frame (rounded to 1e-5, as ``check_frame_leak.py`` compares) occurs in two
  different trajectories of a bundle, which bounds any split's frame leak at 0;
- joins use the old builder's 4-frame linear cross-fade, unchanged.

Sizes. Aligned: each category x intent cell of n clips yields floor(n/3)
trajectories. Misaligned: a category of n clips whose largest intent holds
c_max yields min(floor(n/3), n - c_max) (a triple may not be single-intent);
a greedy construction reaches that bound and the builder asserts it. If the
misaligned bundle is larger, it is cut to the aligned size by uniform random
subsampling of whole trajectories, so the pair differs in alignment, not size.

    python scripts/build_oakink_alignment_v2.py build
    python scripts/build_oakink_alignment_v2.py screen

Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402

from experiment_paired_composition import build_paired_split, coarsen_labels  # noqa: E402

SRC = ROOT / "data" / "bundles" / "oakink.npz"
OUT_DIR = ROOT / "data" / "bundles"
OUT_ALIGNED = OUT_DIR / "oakink_long3_aligned.npz"
OUT_MISALIGNED = OUT_DIR / "oakink_long3_misaligned.npz"

GRANULARITY = "oakink_category"
HELD = 5
MIN_CHAINS = 4
BUDGET = 64
FACTOR = 3
CROSSFADE = 4  # identical to build_oakink_ablations.CROSSFADE
SEED = 0

#: Stored config of the category x intent row in runs/axis_screen.json.
SCREEN_CFG = dict(window=32, stride=16, max_per_traj=6, n_perm=20, seed=0)
REFERENCE_EXCESS = 0.023794310930612808
REFERENCE_Z = 8.32239280894553


def frame_keys(traj) -> set[bytes]:
    """Same identity as check_frame_leak.py."""
    return {np.round(row, 5).tobytes() for row in np.asarray(traj, dtype=np.float32)}


def concat(b: TrajectoryBundle, picks: list[int]) -> np.ndarray:
    """build_oakink_ablations' join, verbatim: k-frame linear cross-fade that
    consumes the incoming clip's first k frames."""
    parts = [np.asarray(b.trajectories[picks[0]], dtype=np.float32)]
    for p in picks[1:]:
        nxt = np.asarray(b.trajectories[p], dtype=np.float32)
        k = min(CROSSFADE, len(parts[-1]), len(nxt) - 1)
        if k > 0:
            w = np.linspace(0.0, 1.0, k + 2, dtype=np.float32)[1:-1][:, None]
            tail = parts[-1][-k:] * (1.0 - w) + nxt[:k] * w
            parts[-1] = np.concatenate([parts[-1][:-k], tail])
            nxt = nxt[k:]
        parts.append(nxt)
    return np.concatenate(parts).astype(np.float32)


def split_cat_intent(coarse: list[str]) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(coarse):
        cat, _, intent = c.partition("->")
        out[cat][intent].append(i)
    return out


def triples_aligned(coarse: list[str], rng: np.random.Generator) -> list[list[int]]:
    by = split_cat_intent(coarse)
    out = []
    for cat in sorted(by):
        for intent in sorted(by[cat]):
            clips = [int(x) for x in rng.permutation(by[cat][intent])]
            for j in range(len(clips) // FACTOR):
                out.append(clips[j * FACTOR:(j + 1) * FACTOR])
    return out


def triples_misaligned(coarse: list[str], rng: np.random.Generator) -> list[list[int]]:
    """Per category: one clip from the largest remaining intent, one from the
    second largest, one from the largest after decrement (ties broken at random).
    A triple is never single-intent. The label clip is the one whose intent
    differs from both others (random among the three if all differ)."""
    by = split_cat_intent(coarse)
    out = []
    for cat in sorted(by):
        queues = {i: [int(x) for x in rng.permutation(v)] for i, v in by[cat].items()}
        n = sum(len(v) for v in queues.values())
        bound = min(n // FACTOR, n - max(len(v) for v in queues.values()))
        made = 0
        while True:
            live = [i for i in queues if queues[i]]
            if len(live) < 2 or sum(len(queues[i]) for i in live) < FACTOR:
                break

            def ranked() -> list[str]:
                ks = [i for i in queues if queues[i]]
                tie = rng.permutation(len(ks))
                return [ks[j] for j in sorted(range(len(ks)),
                                              key=lambda j: (-len(queues[ks[j]]), tie[j]))]

            r = ranked()
            a, b2 = r[0], r[1]
            ca, cb = queues[a].pop(), queues[b2].pop()
            r = ranked()
            if not r:
                queues[a].append(ca); queues[b2].append(cb)
                break
            third_intent = r[0]
            cc = queues[third_intent].pop()
            trip = [(ca, a), (cb, b2), (cc, third_intent)]
            ints = Counter(x[1] for x in trip)
            odd = [x for x in trip if ints[x[1]] == 1]
            first = odd[int(rng.integers(len(odd)))]
            rest = [x for x in trip if x is not first]
            order = [int(rng.integers(2))]
            rest = rest if order[0] == 0 else rest[::-1]
            out.append([first[0]] + [x[0] for x in rest])
            made += 1
        assert made == bound, f"{cat}: greedy made {made}, bound {bound}"
    return out


def make_bundle(b: TrajectoryBundle, triples: list[list[int]], name: str,
                rule: str, extra: dict) -> TrajectoryBundle:
    trajs = [concat(b, t) for t in triples]
    labels = [b.labels[t[0]] for t in triples]
    meta = dict(b.meta)
    meta.pop("sequence_ids", None)
    meta.update({
        "source": name,
        "ablation": "label/window alignment v2 (runs/PREREG_oakink_alignment_v2.md)",
        "derived_from": "oakink.npz",
        "builder": "scripts/build_oakink_alignment_v2.py",
        "concat_factor": FACTOR,
        "crossfade_frames": CROSSFADE,
        "seed": SEED,
        "label_rule": rule,
        "source_clips": [list(map(int, t)) for t in triples],
        "clip_reuse_max": 1,
        **extra,
    })
    return TrajectoryBundle(trajectories=trajs, fps=b.fps, labels=labels, meta=meta)


def check_rules(b: TrajectoryBundle, coarse_src: list[str], out: TrajectoryBundle,
                triples: list[list[int]], aligned: bool) -> None:
    used = Counter(i for t in triples for i in t)
    assert max(used.values()) == 1, "a source clip is used more than once"
    assert all(len(t) == FACTOR for t in triples)
    for t, lab in zip(triples, out.labels):
        assert lab == b.labels[t[0]]
        cats = {coarse_src[i].partition("->")[0] for i in t}
        ints = [coarse_src[i].partition("->")[2] for i in t]
        assert len(cats) == 1, "clips of a trajectory must share the category"
        if aligned:
            assert len(set(ints)) == 1, "aligned: all intents equal the label's"
        else:
            assert ints[1] != ints[0] and ints[2] != ints[0], \
                "misaligned: 2nd and 3rd intents must differ from the label's"
    owner: dict[bytes, int] = {}
    for j, tr in enumerate(out.trajectories):
        for k in frame_keys(tr):
            prev = owner.setdefault(k, j)
            assert prev == j, f"frame shared by trajectories {prev} and {j}"


def min_naive_pool(bundle: TrajectoryBundle, seeds=range(40)) -> tuple[int, str]:
    fine = list(bundle.labels)
    view = SimpleNamespace(labels=coarsen_labels(fine, GRANULARITY))
    worst = 10**9
    for s in seeds:
        try:
            split = build_paired_split(view, HELD, s, fine_labels=fine,
                                       min_chains=MIN_CHAINS)
        except ValueError as exc:
            return -1, f"seed {s}: {exc}"
        worst = min(worst, len(split["naive_pool"]))
    return worst, ""


def stats(bundle: TrajectoryBundle, n_src: int, triples: list[list[int]]) -> dict:
    lengths = np.array([len(t) for t in bundle.trajectories])
    coarse = coarsen_labels(list(bundle.labels), GRANULARITY)
    pool, err = min_naive_pool(bundle)
    return {
        "n_traj": len(bundle.trajectories),
        "median_len": float(np.median(lengths)),
        "min_len": int(lengths.min()), "max_len": int(lengths.max()),
        "n_frames": int(lengths.sum()),
        "coarse_cells": len(set(coarse)),
        "fine_labels": len(set(bundle.labels)),
        "categories": len({c.partition("->")[0] for c in coarse}),
        "intents": len({c.partition("->")[2] for c in coarse}),
        "label_intent_counts": dict(Counter(c.partition("->")[2] for c in coarse)),
        "clips_used": len({i for t in triples for i in t}),
        "clips_unused": n_src - len({i for t in triples for i in t}),
        "min_naive_pool_seeds_0_39": pool,
        "split_error": err,
    }


def run_build() -> int:
    b = TrajectoryBundle.load(SRC)
    coarse = coarsen_labels(list(b.labels), GRANULARITY)
    n = len(b.trajectories)
    rng = np.random.default_rng(SEED)

    tri_a = triples_aligned(coarse, rng)
    tri_m = triples_misaligned(coarse, rng)
    uncapped = {"aligned": len(tri_a), "misaligned": len(tri_m)}
    cap_note = "none"
    if len(tri_m) > len(tri_a):
        keep = np.sort(rng.choice(len(tri_m), len(tri_a), replace=False))
        tri_m = [tri_m[i] for i in keep]
        cap_note = (f"misaligned subsampled uniformly at random from {uncapped['misaligned']} "
                    f"to {len(tri_a)} trajectories to match aligned")
    elif len(tri_a) > len(tri_m):
        keep = np.sort(rng.choice(len(tri_a), len(tri_m), replace=False))
        tri_a = [tri_a[i] for i in keep]
        cap_note = (f"aligned subsampled uniformly at random from {uncapped['aligned']} "
                    f"to {len(tri_m)} trajectories to match misaligned")
    # Random trajectory order, independent of the construction order.
    tri_a = [tri_a[i] for i in rng.permutation(len(tri_a))]
    tri_m = [tri_m[i] for i in rng.permutation(len(tri_m))]

    extra = {"uncapped_sizes": uncapped, "size_cap": cap_note}
    out_a = make_bundle(b, tri_a, "oakink_long3_aligned",
                        "first clip's object->intent; all three clips share its "
                        "category AND intent", extra)
    out_m = make_bundle(b, tri_m, "oakink_long3_misaligned",
                        "first clip's object->intent; all three share its category; "
                        "2nd and 3rd have intents different from the label's", extra)
    check_rules(b, coarse, out_a, tri_a, aligned=True)
    check_rules(b, coarse, out_m, tri_m, aligned=False)
    out_a.save(OUT_ALIGNED)
    out_m.save(OUT_MISALIGNED)

    report = {"source": {"n_traj": n, "min_naive_pool_seeds_0_39": min_naive_pool(b)[0]},
              "uncapped_sizes": uncapped, "size_cap": cap_note,
              "aligned": stats(out_a, n, tri_a),
              "misaligned": stats(out_m, n, tri_m)}
    print(json.dumps(report, indent=1))
    (ROOT / "runs" / "build_oakink_alignment_v2.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    return 0


def run_screen(out: Path) -> int:
    from screen_axes import screen

    rows = []
    for name in ("oakink", "oakink_long3_aligned", "oakink_long3_misaligned"):
        r = screen(OUT_DIR / f"{name}.npz", GRANULARITY, SCREEN_CFG["window"],
                   SCREEN_CFG["stride"], SCREEN_CFG["max_per_traj"],
                   SCREEN_CFG["n_perm"], SCREEN_CFG["seed"])
        if r is None or "error" in r:
            print(f"{name}: {r}")
            return 1
        r["bundle"] = name
        rows.append(r)
        print(f"{name:<24} traj {r['n_traj']:>4} cells {r['n_cells']:>4}  "
              f"left {r['left_only']:.4f}  right {r['right_only']:.4f}  "
              f"excess {r['excess']:+.4f}  z {r['z']:.2f}")
    ref = rows[0]
    ok = abs(ref["excess"] - REFERENCE_EXCESS) < 1e-6 and abs(ref["z"] - REFERENCE_Z) < 1e-3
    print(f"reference reproduction: {'MATCH' if ok else 'MISMATCH'}")
    out.write_text(json.dumps({
        "config": {**SCREEN_CFG, "granularity": GRANULARITY},
        "reference": {"excess": REFERENCE_EXCESS, "z": REFERENCE_Z, "reproduced": bool(ok)},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build", "screen"))
    ap.add_argument("--out", default="runs/axis_screen_alignment_v2.json")
    args = ap.parse_args()
    if args.mode == "screen":
        return run_screen(ROOT / args.out)
    return run_build()


if __name__ == "__main__":
    raise SystemExit(main())
