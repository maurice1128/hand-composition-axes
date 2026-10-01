"""Two quantities a TMLR referee asked for, per Table 1 axis, derived without training.

1. Training volume per arm. The budget counts trajectories, so the number of training windows and frames differs
   between datasets. Reported as the expectation over a random draw of `budget` trajectories from the whole bundle
   (the arms are random fills, so the realised value varies by seed around this), at window 32 and stride 4.

2. Marginal loss. Holding out a cell (a, b) removes all its trajectories from the naive arm's pool, so the naive arm
   also sees fewer trajectories of factor a and of factor b. For each held cell and each of its two factor values,
   the share of the bundle's trajectories carrying that value that lie in ANY held cell (and so are absent from the
   naive pool) is computed; the per-seed value is the mean over held cells and both factors. Its per-seed
   correlation with the penalty is reported within each axis. Not defined for the transitions axis, whose held unit
   is a consecutive pair inside a chain rather than a cell of a two-factor grid.

Splits are rebuilt with scripts/axis_diagnostics.py's derive_seed, which reproduces every stored split exactly.
Output: runs/marginal_loss_volume.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import axis_diagnostics as ad  # noqa: E402  (main-guarded; importing prints nothing)
from experiment_data_efficiency import transitions_of  # noqa: E402
from experiment_paired_composition import coarsen_labels  # noqa: E402

WINDOW, STRIDE, BUDGET = 32, 4, 256


def lengths_of(bundle: str) -> np.ndarray:
    with np.load(ROOT / bundle.replace("\\", "/"), allow_pickle=True) as z:
        if "lengths" in z.files:
            return np.asarray(z["lengths"])
        raise KeyError(f"{bundle}: no 'lengths' array ({z.files})")


out = []
for dataset, axis, dirs, both in ad.AXES:
    rows = ad.seed_rows(dirs, both)
    args = ad.read_run(dirs[0])["args"]
    L = lengths_of(args["bundle"])
    win = np.maximum(0, (L - WINDOW) // STRIDE + 1)
    fine = ad.load_labels(args["bundle"])
    gran = args["granularity"]
    coarse = fine if gran == "fine" else coarsen_labels(fine, gran)
    pairs = [transitions_of(c) for c in coarse]
    rec = {"dataset": dataset, "axis": axis, "bundle": args["bundle"], "n_seeds": len(rows),
           "expected_windows_per_arm": float(BUDGET * win.mean()),
           "expected_frames_per_arm": float(BUDGET * L.mean())}
    if axis != "annotated transitions":
        firsts = [{a for a, _ in p} for p in pairs]
        seconds = [{b for _, b in p} for p in pairs]
        loss, pen = [], []
        for d, seed, row in rows:
            held = [tuple(h.split("->")) for h in ad.derive_seed(args, seed)["held_compositions"]]
            held_set = set(held)
            in_held = [bool(set(p) & held_set) for p in pairs]
            vals = []
            for a, b in held:
                with_a = [i for i, f in enumerate(firsts) if a in f]
                with_b = [i for i, s in enumerate(seconds) if b in s]
                vals.append(np.mean([in_held[i] for i in with_a]))
                vals.append(np.mean([in_held[i] for i in with_b]))
            loss.append(float(np.mean(vals)))
            pen.append(100 * row["penalty"] / row["naive"]["mse_target"])
        r, p = stats.pearsonr(loss, pen)
        rec.update({"marginal_loss_mean": float(np.mean(loss)), "marginal_loss_min": float(np.min(loss)),
                    "marginal_loss_max": float(np.max(loss)), "loss_penalty_r": float(r), "loss_penalty_p": float(p)})
    out.append(rec)
    print(f"{dataset:13s} {axis:28s} windows/arm {rec['expected_windows_per_arm']:8.0f} frames/arm "
          f"{rec['expected_frames_per_arm']:8.0f}"
          + (f"  marginal loss {rec['marginal_loss_mean']:.3f} [{rec['marginal_loss_min']:.3f}, "
             f"{rec['marginal_loss_max']:.3f}]  r {rec['loss_penalty_r']:+.2f} p {rec['loss_penalty_p']:.3f}"
             if "marginal_loss_mean" in rec else ""))

extra = {}
for name, bundle in (("grab_contact_segment", "data/bundles/grab_grasp.npz"),
                     ("oakink_pair", "data/bundles/oakink_pair_aligned.npz"),
                     ("oakink_pair_misaligned", "data/bundles/oakink_pair_misaligned.npz")):
    L = lengths_of(bundle)
    b = 64 if name.startswith("oakink_pair") else 256
    extra[name] = {"budget": b, "expected_windows_per_arm": float(b * np.maximum(0, (L - WINDOW) // STRIDE + 1).mean()),
                   "expected_frames_per_arm": float(b * L.mean())}
    print(name, extra[name])
# Budget-64 OakInk-Image baseline for the volume-sign comparison.
L = lengths_of("data/bundles/oakink.npz")
extra["oakink_b64"] = {"budget": 64, "expected_windows_per_arm": float(64 * np.maximum(0, (L - WINDOW) // STRIDE + 1).mean())}
print("oakink_b64", extra["oakink_b64"])
(ROOT / "runs" / "marginal_loss_volume.json").write_text(json.dumps({"axes": out, "extra": extra}, indent=1),
                                                         encoding="utf-8")
