"""Screen candidate OakInk2 composition axes that its bundle labels do not carry.

The necessity direction of the paper's relation needs axes with non-positive
interaction excess whose paired split can actually be built. Of the eleven axes
in `runs/axis_screen.json`, four of the five non-positive ones are structurally
untestable: on DexYCB and on GRAB's object axes every transition falls in a
single fine label, so a leak-free split cannot hold anything out.

OakInk2 is the exception. Its 391 transitions include 108 that appear in four or
more distinct chains, four times the pool of any other axis here, and its one
screened axis is negative. So it is the only dataset in this study where more
necessity tests can be constructed at all.

Its bundle labels are primitive *chains*, not crossed factors. Scene and subject
live in `meta['segments']` instead, and this script crosses them with the chain's
own content to produce candidate axes, scores each with the same statistic
`screen_axes.py` uses, and reports the structural pool alongside so an axis that
cannot be swept is visible before anything is trained.

Nothing here is trained. Add `--build` to write the bundles for whichever axes
are worth sweeping.

    python scripts/screen_oakink2_axes.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402

from experiment_paired_composition import coarsen_labels, transitions_of  # noqa: E402
from screen_axes import decompose, features, interaction  # noqa: E402

BUNDLE = ROOT / "data" / "bundles" / "oakink2.npz"


def factors(bundle: TrajectoryBundle) -> dict[str, list[str]]:
    """The candidate factors, one value per trajectory."""
    segs = bundle.meta["segments"]
    if len(segs) != len(bundle.labels):
        raise SystemExit("segments and labels are not aligned; refusing to guess")
    scene, subject, verb = [], [], []
    for s in segs:
        m = re.match(r"(scene_\d+)", s["sequence"])
        scene.append(m.group(1) if m else "unknown")
        m = re.search(r"__([AO]\d+)\+\+", s["sequence"])
        subject.append(m.group(1) if m else "unknown")
        # The task sentence's leading verb, which is what the hand is doing
        # independently of which objects the scene happens to hold.
        verb.append((s["task"].strip().split() or ["unknown"])[0].rstrip(".").lower())
    head = [lab.partition("->")[0] for lab in bundle.labels]
    return {"scene": scene, "subject": subject, "verb": verb, "primitive": head}


def structural_pool(coarse: list[str], fine: list[str], min_chains: int) -> int:
    """How many transitions could be held out under a leak-free split."""
    chains: dict[tuple[str, str], set[str]] = {}
    for i, lab in enumerate(coarse):
        for t in transitions_of(lab):
            chains.setdefault(t, set()).add(fine[i])
    return sum(1 for c in chains.values() if len(c) >= min_chains)


def score(x: np.ndarray, owner: np.ndarray, coarse: list[str],
          n_perm: int, rng: np.random.Generator) -> dict:
    """The screen statistic, computed exactly as `screen_axes.screen` does."""
    def codes_of(values: list[str]) -> np.ndarray:
        vocab = {v: i for i, v in enumerate(sorted(set(values)))}
        return np.array([vocab[values[o]] for o in owner], dtype=np.intp)

    lefts = [c.partition("->")[0] for c in coarse]
    rights = [c.partition("->")[2] for c in coarse]
    parts = decompose(x, codes_of(lefts), codes_of(rights), codes_of(coarse))

    per_traj = np.array(coarse)
    tids = np.unique(owner)
    pos = {t: i for i, t in enumerate(tids)}
    idx = np.array([pos[o] for o in owner])
    null = []
    for _ in range(n_perm):
        sh = rng.permutation(per_traj)
        sl = [s.partition("->")[0] for s in sh]
        sr = [s.partition("->")[2] for s in sh]
        vl = {v: i for i, v in enumerate(sorted(set(sl)))}
        vr = {v: i for i, v in enumerate(sorted(set(sr)))}
        vc = {v: i for i, v in enumerate(sorted(set(sh)))}
        null.append(interaction(
            x,
            np.array([vl[sl[i]] for i in idx], dtype=np.intp),
            np.array([vr[sr[i]] for i in idx], dtype=np.intp),
            np.array([vc[sh[i]] for i in idx], dtype=np.intp))[0])
    null = np.array(null)
    sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    excess = parts["interaction_r2"] - float(null.mean())
    return {**parts, "null_mean": float(null.mean()), "excess": excess,
            "z": excess / sd if sd > 0 else 0.0,
            "n_cells": len(set(coarse)),
            "n_left": len(set(lefts)), "n_right": len(set(rights))}


#: (name, left factor, right factor). Crossing two content factors is not a
#: candidate: the paper's axes all pair a *provenance* factor with a *content*
#: one, and a content-by-content pairing measures something else.
CANDIDATES = [
    ("scene x primitive", "scene", "primitive"),
    ("subject x primitive", "subject", "primitive"),
    ("scene x verb", "scene", "verb"),
    ("subject x verb", "subject", "verb"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-per-traj", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--min-chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/oakink2_axis_screen.json")
    ap.add_argument("--build", action="store_true",
                    help="write a bundle per candidate so the sweep can run it")
    args = ap.parse_args()

    bundle = TrajectoryBundle.load(BUNDLE)
    fac = factors(bundle)
    fine = list(bundle.labels)

    rng = np.random.default_rng(args.seed)
    x, owner = features(bundle, args.window, args.stride, args.max_per_traj, rng)
    x = x - x.mean(axis=0)
    n_comp = min(48, x.shape[1], len(x) - 1)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    x = (x @ vt[:n_comp].T).astype(np.float32)
    print(f"{len(np.unique(owner))} trajectories, {len(x)} windows, {n_comp} components\n")

    hdr = f"{'axis':22s} {'cells':>6s} {'per cell':>9s} {'excess':>9s} {'z':>7s} {'pool':>6s} {'sweepable':>10s}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, lf, rf in CANDIDATES:
        coarse = [f"{a}->{b}" for a, b in zip(fac[lf], fac[rf])]
        r = score(x, owner, coarse, args.n_perm, rng)
        pool = structural_pool(coarse, fine, args.min_chains)
        r.update(dataset="OakInk2", axis=name, left=lf, right=rf,
                 structural_pool=pool, min_chains=args.min_chains,
                 n_traj=int(len(np.unique(owner))))
        rows.append(r)
        per = len(coarse) / r["n_cells"]
        ok = "yes" if pool >= 5 else "NO"
        print(f"{name:22s} {r['n_cells']:6d} {per:9.1f} {r['excess']:+9.4f} "
              f"{r['z']:+7.1f} {pool:6d} {ok:>10s}")

        # The sweep must see the crossed label as coarse and the chain-plus-
        # provenance as fine, so the bundle carries the provenance in a suffix
        # and the pool is re-measured through the sweep's own coarsening.
        suffixed = [f"{c}@{sc}@{su}@{vb}" for c, sc, su, vb in
                    zip(fine, fac["scene"], fac["subject"], fac["verb"])]
        mode = f"oakink2_{lf}_{rf}"
        pool_real = structural_pool(coarsen_labels(suffixed, mode), suffixed, args.min_chains)
        if pool_real < pool:
            raise SystemExit(f"{name}: pool through the sweep path ({pool_real}) is below "
                             f"the screen's ({pool}); the label encoding lost information")
        r["structural_pool_sweep_path"] = pool_real
        if args.build and pool >= 5:
            out = ROOT / "data" / "bundles" / f"oakink2_{lf}_{rf}.npz"
            b = TrajectoryBundle(trajectories=bundle.trajectories, labels=suffixed,
                                 fps=bundle.fps, meta={**bundle.meta,
                                                       "axis": name,
                                                       "granularity": mode,
                                                       "derived_from": "oakink2.npz"})
            b.save(out)
            print(f"    wrote {out}")

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=1),
                    encoding="utf-8")
    print(f"\nwrote {dest}")
    print("A non-positive excess with a pool >= 5 is a necessity test worth sweeping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
