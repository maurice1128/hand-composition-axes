"""Constituent-level leak check for concatenated OakInk-Image bundles.

``build_paired_split`` keeps FINE labels disjoint between the target set and the
training sets, and ``check_composition_leak.py`` verifies exactly that. But a
concatenated trajectory's fine label is only its FIRST clip's ``object->intent``.
Its other clips can be other objects, and a training trajectory can contain a
clip carrying exactly a target clip's ``object->intent``. Neither the split nor
the label gate can see that. This script can.

For a bundle, it recovers which source clips of ``data/bundles/oakink.npz`` make
up each trajectory from ``meta["source_clips"]`` (saved by the v2/v3/v4
builders), and ASSERTS the record is true by re-concatenating those clips with
the builder's join and comparing with the bundle's frames exactly. A bundle with
no ``source_clips`` must be frame-identical to ``oakink.npz`` (one clip each).

Then, for each seed, it rebuilds the split and both arms exactly as
``experiment_paired_composition.py`` does (``build_paired_split`` +
``sample_pools`` with the seed) and reports per arm:

 (a) fraction of target TRAJECTORIES with >= 1 constituent clip whose
     ``object->intent`` occurs among the arm's training constituent clips;
 (b) fraction of target CLIPS so exposed;
 (c) fraction of the arm's training clips whose (category, intent) cell is one
     of the held cells;
 (d) fraction of target clips that belong to a held cell (arm-independent).

What this does NOT show
-----------------------
- "Exposed" is label identity of clips (same object, same intent), not frame
  identity; ``check_frame_leak.py`` covers frames. Two different clips of the
  same object->intent are near-duplicates, not copies.
- Cross-fade frames (4 per join) mix two clips and are attributed to neither.
- (c) counts clips, not frames or windows; clips differ in length.
- It reads the split, not any trained model: it says what an arm could have
  used, not what it did use.

    python scripts/check_constituent_leak.py            # the four existing bundles
    python scripts/check_constituent_leak.py --bundles data/bundles/x.npz --tag x

Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402

import build_oakink_alignment_v2 as V2  # noqa: E402
import experiment_paired_composition as E  # noqa: E402

SRC = ROOT / "data" / "bundles" / "oakink.npz"
DEFAULT = ["oakink_long3_aligned", "oakink_long3_misaligned",
           "oakink_long3_misaligned_cm", "oakink"]


def composition(bundle: TrajectoryBundle, src: TrajectoryBundle) -> tuple[list[list[int]], str]:
    """Source clips per trajectory, verified against the bundle's frames."""
    sc = bundle.meta.get("source_clips")
    if sc is None:
        assert len(bundle.trajectories) == len(src.trajectories), \
            "no source_clips in meta and not the same size as oakink.npz"
        for i, (x, y) in enumerate(zip(bundle.trajectories, src.trajectories)):
            assert np.array_equal(np.asarray(x), np.asarray(y)), f"trajectory {i} differs from oakink.npz"
        assert list(bundle.labels) == list(src.labels)
        return [[i] for i in range(len(src.trajectories))], "identity (frame-equal to oakink.npz)"
    sc = [[int(i) for i in t] for t in sc]
    assert len(sc) == len(bundle.trajectories)
    for j, t in enumerate(sc):
        rebuilt = V2.concat(src, t)
        got = np.asarray(bundle.trajectories[j], dtype=np.float32)
        assert rebuilt.shape == got.shape and np.array_equal(rebuilt, got), \
            f"trajectory {j}: meta source_clips {t} do not reproduce the stored frames"
        assert bundle.labels[j] == src.labels[t[0]], f"trajectory {j}: label is not the first clip's"
    return sc, "meta.source_clips, re-concatenated and frame-equal"


def measure(bundle_path: Path, seeds, granularity="oakink_category", held=5, min_chains=4,
            min_per_composition=5, budget=64) -> dict:
    src = TrajectoryBundle.load(SRC)
    bundle = TrajectoryBundle.load(bundle_path)
    clips, how = composition(bundle, src)
    src_fine = list(src.labels)
    src_coarse = E.coarsen_labels(src_fine, granularity)

    fine = list(bundle.labels)
    bundle.labels = E.coarsen_labels(fine, granularity)

    rows = []
    for seed in seeds:
        split = E.build_paired_split(bundle, held, seed, fine_labels=fine, min_chains=min_chains)
        naive, informed = E.sample_pools(split, budget, seed, bundle.labels, min_per_composition)
        sound = E.assert_split_sound(split, naive, informed, fine, bundle.labels)
        held_cells = set(split["held_compositions"])
        target = [int(i) for i in split["target"]]
        tclips = [c for i in target for c in clips[i]]
        row = {"seed": int(seed), "n_target_traj": len(target), "n_target_clips": len(tclips),
               "held": sorted(held_cells),
               "label_leak_naive": sound["leak_naive"], "label_leak_informed": sound["leak_informed"],
               "d_target_clips_in_held_cell":
                   sum(src_coarse[c] in held_cells for c in tclips) / len(tclips)}
        for arm, idx in (("naive", naive), ("informed", informed)):
            train = [c for i in idx for c in clips[int(i)]]
            assert not set(train) & set(tclips), "a source clip sits in both target and training"
            train_fine = {src_fine[c] for c in train}
            row[arm] = {
                "n_train_traj": len(idx), "n_train_clips": len(train),
                "a_target_traj_exposed":
                    sum(any(src_fine[c] in train_fine for c in clips[i]) for i in target) / len(target),
                "b_target_clips_exposed":
                    sum(src_fine[c] in train_fine for c in tclips) / len(tclips),
                "c_train_clips_in_held_cell":
                    sum(src_coarse[c] in held_cells for c in train) / len(train),
            }
        rows.append(row)

    def mean(f):
        return float(np.mean([f(r) for r in rows]))

    summary = {"d_target_clips_in_held_cell": mean(lambda r: r["d_target_clips_in_held_cell"])}
    for arm in ("naive", "informed"):
        for k in ("a_target_traj_exposed", "b_target_clips_exposed", "c_train_clips_in_held_cell"):
            summary[f"{arm}.{k}"] = mean(lambda r, a=arm, k=k: r[a][k])
            summary[f"{arm}.{k}.max"] = float(max(r[arm][k] for r in rows))
    return {"bundle": str(Path(bundle_path).as_posix()), "composition_source": how,
            "n_traj": len(bundle.trajectories),
            "clips_per_traj": sorted({len(c) for c in clips}),
            "settings": {"granularity": granularity, "held": held, "min_chains": min_chains,
                         "min_per_composition": min_per_composition, "budget": budget,
                         "seeds": [int(s) for s in seeds]},
            "summary_mean_over_seeds": summary, "per_seed": rows}


def table(results: list[dict]) -> str:
    lines = ["Constituent-level leak (scripts/check_constituent_leak.py)",
             "means over seeds; (max over seeds) in brackets for (a)",
             "(a) target trajectories with a constituent clip whose object->intent is among the arm's training clips",
             "(b) target clips so exposed   (c) arm's training clips lying in a held cell   (d) target clips in a held cell",
             ""]
    s0 = results[0]["settings"]
    lines.append(f"settings: {s0}")
    lines.append("")
    lines.append(f"{'bundle':<28} {'arm':<9} {'(a) traj':>16} {'(b) clips':>10} {'(c) train held':>15} {'(d) tgt held':>13}")
    for r in results:
        s = r["summary_mean_over_seeds"]
        name = Path(r["bundle"]).stem
        for arm in ("naive", "informed"):
            lines.append(f"{name:<28} {arm:<9} "
                         f"{100*s[f'{arm}.a_target_traj_exposed']:>7.2f}% ({100*s[f'{arm}.a_target_traj_exposed.max']:>5.1f}) "
                         f"{100*s[f'{arm}.b_target_clips_exposed']:>9.2f}% "
                         f"{100*s[f'{arm}.c_train_clips_in_held_cell']:>14.2f}% "
                         f"{100*s['d_target_clips_in_held_cell']:>12.2f}%")
        lines.append(f"{'':<28} composition: {r['composition_source']}; clips/traj {r['clips_per_traj']}; "
                     f"{r['n_traj']} trajectories")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", nargs="*", default=[f"data/bundles/{n}.npz" for n in DEFAULT])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--granularity", default="oakink_category")
    ap.add_argument("--held-compositions", type=int, default=5)
    ap.add_argument("--min-chains", type=int, default=4)
    ap.add_argument("--min-per-composition", type=int, default=5)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--tag", default="", help="suffix for the output files; empty = the default pair")
    ap.add_argument("--require-zero-a", action="store_true",
                    help="exit 1 unless measure (a) is exactly 0 for both arms on every seed")
    args = ap.parse_args()

    results = [measure(ROOT / p, args.seeds, args.granularity, args.held_compositions,
                       args.min_chains, args.min_per_composition, args.budget)
               for p in args.bundles]
    txt = table(results)
    print(txt)
    suffix = f"_{args.tag}" if args.tag else ""
    (ROOT / "runs" / f"constituent_leak{suffix}.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    (ROOT / "runs" / "gates" / f"constituent_leak{suffix}.txt").write_text(txt, encoding="utf-8")
    if args.require_zero_a:
        worst = max(r["summary_mean_over_seeds"][f"{arm}.a_target_traj_exposed.max"]
                    for r in results for arm in ("naive", "informed"))
        print(f"require-zero-a: worst (a) over bundles, arms, seeds = {worst}")
        return 0 if worst == 0.0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
