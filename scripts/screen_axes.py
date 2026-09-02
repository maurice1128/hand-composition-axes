"""Which composition axes are worth spending GPU on?

Every dataset here was measured on exactly one axis, and four of the five came
back null. The obvious objection is that the axes were badly chosen rather than
the data being uncompositional -- DexYCB's ``subject -> object shape`` is the
clearest case, since two people grasping the same can have little reason to
differ, and "subject" is not a composition in any interesting sense.

That objection deserves evidence, not argument, but a full paired sweep costs
hours of GPU per axis. This is the cheap screen, and it measures the right
thing.

Why not eta^2
-------------
``scripts/analyse_composition_separability.py`` measures how much pose variance
the composition label explains. It is a bad screen: DexYCB scored the *highest*
excess eta^2 of the three real datasets (+0.123) and has no compositional
difficulty at all. The reason is that eta^2 counts everything the label
explains, including the parts explained by each factor **separately** -- and
subject identity alone is a strong pose signature.

The compositional penalty asks a narrower question: does knowing the *pairing*
help beyond knowing each factor on its own? That is an **interaction**, so this
fits two nested models per axis and reports what the interaction adds:

    additive   pose ~ left + right          (one-hot, least squares)
    full       pose ~ left + right + cell   (the pairing gets its own term)

    interaction R^2 = R^2(full) - R^2(additive)

A permutation null follows the same rule as the eta^2 diagnostic: labels are
shuffled over whole *trajectories*, because 30 fps windows within one
trajectory are near-duplicates and shuffling them independently would give an
absurdly low null.

This is a screen, not a substitute. A high interaction says an axis is worth
measuring properly; it does not say the modular prior will win on it.

    python scripts/screen_axes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import normalize  # noqa: E402

from experiment_paired_composition import coarsen_labels  # noqa: E402

#: (dataset, bundle, axis name, how to derive the coarse label)
#:
#: Each entry names a *candidate* axis. Several were never measured: GRAB's own
#: four documented intent classes, DexYCB's object shape without the subject,
#: and OakInk's three official groupings against each other.
AXES = [
    ("OakInk-Image", "oakink.npz", "official category x intent", "oakink_category"),
    ("OakInk-Image", "oakink.npz", "official class x intent", "oakink_class"),
    ("OakInk-Image", "oakink.npz", "affordance x intent", "oakink_attr"),
    ("OakInk-Image", "oakink.npz", "id-prefix x intent (provenance)", "category"),
    ("GRAB", "grab.npz", "shape x fine intent", "shape"),
    ("GRAB", "grab.npz", "shape x intent class", "grab_shape_intentclass"),
    ("GRAB", "grab.npz", "object x intent class", "grab_object_intentclass"),
    ("GRAB", "grab.npz", "object x fine intent", "fine"),
    ("DexYCB", "dexycb_shape.npz", "subject x shape", "fine"),
    ("DexYCB", "dexycb.npz", "subject x object", "fine"),
    ("OakInk2", "oakink2.npz", "annotated transitions", "fine"),
]


def derive(labels: list[str], mode: str) -> list[str]:
    """Coarse labels, including the GRAB intent-class axes the sweeps never used."""
    if mode.startswith("grab_"):
        from caredex.data.grab import GRAB_SHAPE_CLASS, intent_class

        out = []
        for lab in labels:
            left, _, right = lab.partition("->")
            group = GRAB_SHAPE_CLASS[left] if "shape" in mode else left
            out.append(f"{group}->{intent_class(right)}")
        return out
    return coarsen_labels(labels, mode)


def features(bundle: TrajectoryBundle, window: int, stride: int,
             max_per_traj: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One row per window, plus the trajectory index each row came from."""
    rows, owner = [], []
    for i, traj in enumerate(bundle.trajectories):
        t = np.asarray(traj)
        if len(t) < window:
            continue
        starts = np.arange(0, len(t) - window + 1, stride)
        if len(starts) > max_per_traj:
            starts = np.sort(rng.choice(starts, max_per_traj, replace=False))
        w = normalize(np.stack([t[s : s + window] for s in starts]))
        rows.append(w.reshape(len(w), -1).astype(np.float64))
        owner.append(np.full(len(w), i))
    return np.concatenate(rows), np.concatenate(owner)


def r_squared(x: np.ndarray, codes: list[np.ndarray]) -> float:
    """Variance explained by a set of one-hot factors, fitted jointly.

    The design matrix is built once per call and solved for all feature columns
    at once. Features are PCA-reduced upstream: a 32x27 window flattens to 864
    dimensions and OakInk2 contributes tens of thousands of windows, which made
    the first version of this screen too slow to finish.
    """
    n = len(x)
    blocks = [np.eye(int(c.max()) + 1, dtype=np.float32)[c] for c in codes]
    design = np.concatenate([np.ones((n, 1), dtype=np.float32)] + blocks, axis=1)
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    ss_res = float(((x - design @ coef) ** 2).sum())
    ss_tot = float(((x - x.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def decompose(x: np.ndarray, left: np.ndarray, right: np.ndarray,
              cell: np.ndarray) -> dict:
    """Split the explained variance into each factor and the pairing.

    The per-factor terms are what the trajectory-length hypothesis could not
    explain and what the standing one predicts: OakInk-Image's intents were
    *posed* -- subjects were asked to grasp the same object with a named intent,
    so intent is written into the finger configuration -- while GRAB's intents
    are natural, and change where the arm goes after the grasp rather than how
    the hand is shaped. If that is right, ``right_only`` should be large on
    OakInk-Image and small on GRAB.
    """
    left_only = r_squared(x, [left])
    right_only = r_squared(x, [right])
    additive = r_squared(x, [left, right])
    full = r_squared(x, [left, right, cell])
    return {
        "left_only": left_only,
        "right_only": right_only,
        "additive_r2": additive,
        "interaction_r2": full - additive,
    }


def interaction(x: np.ndarray, left: np.ndarray, right: np.ndarray,
                cell: np.ndarray) -> tuple[float, float]:
    d = decompose(x, left, right, cell)
    return d["interaction_r2"], d["additive_r2"]


def screen(path: Path, mode: str, window: int, stride: int, max_per_traj: int,
           n_perm: int, seed: int) -> dict | None:
    if not path.exists():
        return None
    b = TrajectoryBundle.load(path)
    try:
        coarse = derive(list(b.labels), mode)
    except Exception as exc:  # noqa: BLE001 - an axis that does not apply is not an error
        return {"error": str(exc)[:80]}

    rng = np.random.default_rng(seed)
    x, owner = features(b, window, stride, max_per_traj, rng)
    # Project onto the leading components. The interaction is a ratio of sums of
    # squares, and PCA is an orthogonal change of basis that keeps almost all of
    # it while turning an 864-column solve into a 48-column one.
    x = x - x.mean(axis=0)
    n_comp = min(48, x.shape[1], len(x) - 1)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    x = (x @ vt[:n_comp].T).astype(np.float32)

    def codes_of(values: list[str]) -> np.ndarray:
        vocab = {v: i for i, v in enumerate(sorted(set(values)))}
        return np.array([vocab[values[o]] for o in owner], dtype=np.intp)

    lefts = [c.partition("->")[0] for c in coarse]
    rights = [c.partition("->")[2] for c in coarse]
    parts = decompose(x, codes_of(lefts), codes_of(rights), codes_of(coarse))
    inter, additive = parts["interaction_r2"], parts["additive_r2"]

    # Null: reassign whole trajectories to cells, keeping the marginal factors.
    per_traj = np.array(coarse)
    tids = np.unique(owner)
    pos = {t: i for i, t in enumerate(tids)}
    idx = np.array([pos[o] for o in owner])
    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(per_traj)
        sl = [s.partition("->")[0] for s in shuffled]
        sr = [s.partition("->")[2] for s in shuffled]
        vl = {v: i for i, v in enumerate(sorted(set(sl)))}
        vr = {v: i for i, v in enumerate(sorted(set(sr)))}
        vc = {v: i for i, v in enumerate(sorted(set(shuffled)))}
        null.append(interaction(
            x,
            np.array([vl[sl[i]] for i in idx], dtype=np.intp),
            np.array([vr[sr[i]] for i in idx], dtype=np.intp),
            np.array([vc[shuffled[i]] for i in idx], dtype=np.intp),
        )[0])
    null = np.array(null)
    sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    return {
        **parts,
        "n_traj": int(len(tids)),
        "n_cells": len(set(coarse)),
        "n_left": len(set(lefts)),
        "n_right": len(set(rights)),
        "additive_r2": additive,
        "interaction_r2": inter,
        "null_mean": float(null.mean()),
        "excess": inter - float(null.mean()),
        "z": (inter - float(null.mean())) / sd if sd > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-per-traj", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/axis_screen.json")
    args = ap.parse_args()

    print(f"{'dataset':<14}{'axis':<32}{'cells':>6}{'left':>7}{'right':>7}"
          f"{'addit':>7}{'inter':>7}{'excess':>9}{'z':>7}")
    print("-" * 98)
    rows = []
    for dataset, bundle, name, mode in AXES:
        r = screen(ROOT / "data" / "bundles" / bundle, mode, args.window,
                   args.stride, args.max_per_traj, args.n_perm, args.seed)
        if r is None:
            print(f"{dataset:<14}{name:<34} bundle not built")
            continue
        if "error" in r:
            print(f"{dataset:<14}{name:<34} {r['error']}")
            continue
        r.update(dataset=dataset, axis=name, mode=mode)
        rows.append(r)
        print(f"{dataset:<14}{name:<32}{r['n_cells']:>6}{r['left_only']:>7.3f}"
              f"{r['right_only']:>7.3f}{r['additive_r2']:>7.3f}"
              f"{r['interaction_r2']:>7.3f}{r['excess']:>+9.4f}{r['z']:>7.1f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nExcess interaction is the screen. It is NOT the compositional penalty --"
          "\nit says an axis carries pairing-specific structure a linear additive model"
          "\nmisses, which is a precondition for the penalty being non-zero, not a"
          "\nguarantee. Axes scoring near zero are not worth a sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
