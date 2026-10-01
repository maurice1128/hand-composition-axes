"""How many GRAB shape x fine-intent cells received a non-zero planted offset. Compares grab_planted.npz with
grab.npz trajectory by trajectory (difference beyond clamping noise) and groups by cell. Output: runs/planted_cells.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from experiment_data_efficiency import transitions_of  # noqa: E402
from experiment_paired_composition import coarsen_labels  # noqa: E402

g = np.load(ROOT / "data/bundles/grab.npz", allow_pickle=True)
p = np.load(ROOT / "data/bundles/grab_planted.npz", allow_pickle=True)
fg, fp, L = g["frames"], p["frames"], g["lengths"]
assert fg.shape == fp.shape
off = np.r_[0, np.cumsum(L)]
cells = [transitions_of(c)[0] for c in coarsen_labels([str(x) for x in g["labels"]], "shape")]
per_traj = np.array([np.abs(fp[off[i]:off[i + 1]] - fg[off[i]:off[i + 1]]).mean() for i in range(len(L))])
by = {}
for c, d in zip(cells, per_traj):
    by[c] = max(by.get(c, 0.0), float(d))
out = {"cells": len(by), "cells_with_offset": int(sum(v > 1e-6 for v in by.values())),
       "note": "a cell counts as planted if any of its trajectories differs from the source bundle"}
(ROOT / "runs" / "planted_cells.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out)
