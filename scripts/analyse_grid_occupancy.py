"""How full is each dataset's object-intent grid?

Sec V-B used to say GRAB "records natural sequences in which each object-intent
combination occurs once as it naturally would". That is false in the direction
that matters: GRAB repeats each combination it has *more* than OakInk-Image
does (5.24 clips per occupied cell against 2.23). The real difference is
crossing, not repetition -- OakInk-Image defines the same four intents for
every object, so its grid is nearly full, whereas GRAB's intent vocabulary is
largely object-specific, so most of its grid does not exist at all.

A paired composition split needs cells, not clips. This measures both so the
claim in Sec V-B and Sec VII rests on an artefact rather than on an impression.

It also asks the obvious follow-up, and answers it against us: does crossing
predict compositional difficulty? Over the eleven axes in ``runs/axis_screen.json``
it does not (Pearson r = -0.10, p = 0.78). GRAB's object x intent-class grid is
92.6% crossed with a significantly *negative* interaction excess, and both of
DexYCB's axes are perfectly crossed with the most negative excess of all. So the
grid statistic supports a feasibility claim -- a paired split needs occupied
cells -- and not a causal one. An earlier draft of Sec V-B made the causal claim;
this script is why it was withdrawn.

    python scripts/analyse_grid_occupancy.py
"""

from __future__ import annotations

import collections
import glob
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402

GRAB_ZIPS = r"D:\datasets\grab\grab__s*.zip"
OAKINK_BUNDLE = "data/bundles/oakink.npz"
OUT = Path("runs/grid_occupancy.json")


def summarise(cells: collections.Counter, name: str) -> dict:
    objs = {o for o, _ in cells}
    ints = {i for _, i in cells}
    per_intent = collections.Counter(i for _, i in cells)
    grid = len(objs) * len(ints)
    return {
        "dataset": name,
        "n_objects": len(objs),
        "n_intents": len(ints),
        "grid_cells": grid,
        "occupied_cells": len(cells),
        "occupancy": len(cells) / grid,
        "clips": sum(cells.values()),
        "clips_per_occupied_cell": sum(cells.values()) / len(cells),
        "cells_with_more_than_one_clip": sum(1 for v in cells.values() if v > 1),
        "intents_on_ten_or_more_objects": sorted(
            i for i, c in per_intent.items() if c >= 10
        ),
        "intents_on_exactly_one_object": sum(1 for c in per_intent.values() if c == 1),
    }


def grab_cells() -> collections.Counter:
    cells: collections.Counter = collections.Counter()
    for z in sorted(glob.glob(GRAB_ZIPS)):
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                if not n.endswith(".npz"):
                    continue
                d = np.load(io.BytesIO(f.read(n)), allow_pickle=True)
                cells[(str(d["obj_name"]), str(d["motion_intent"]))] += 1
    return cells


def oakink_cells() -> collections.Counter:
    b = TrajectoryBundle.load(OAKINK_BUNDLE)
    fmt = b.meta.get("label_format")
    if fmt != "object->intent":
        raise SystemExit(f"expected object->intent labels, got {fmt!r}")
    cells: collections.Counter = collections.Counter()
    for lab in b.labels:
        o, _, i = lab.partition("->")
        cells[(o, i)] += 1
    return cells


def screened_axes() -> list[dict]:
    """Occupancy and interaction excess for every axis the screen scored."""
    src = json.loads(Path("runs/axis_screen.json").read_text(encoding="utf-8"))
    out = []
    for r in src["rows"]:
        grid = r["n_left"] * r["n_right"]
        out.append(
            {
                "dataset": r["dataset"],
                "axis": r["axis"],
                "occupied_cells": r["n_cells"],
                "grid_cells": grid,
                "occupancy": r["n_cells"] / grid,
                "interaction_excess": r["excess"],
                "z": r["z"],
            }
        )
    return out


def main() -> int:
    rows = [summarise(oakink_cells(), "oakink_image"), summarise(grab_cells(), "grab")]
    axes = screened_axes()
    occ = np.array([a["occupancy"] for a in axes])
    exc = np.array([a["interaction_excess"] for a in axes])
    from scipy import stats  # noqa: PLC0415

    pr, pp = stats.pearsonr(occ, exc)
    sr, sp = stats.spearmanr(occ, exc)
    crossing = {
        "n_axes": len(axes),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "axes": axes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"grids": rows, "crossing_vs_interaction": crossing}, indent=2),
        encoding="utf-8",
    )
    for r in rows:
        print(
            f"{r['dataset']:14s} {r['n_objects']:3d} obj x {r['n_intents']:3d} intent"
            f" -> {r['occupied_cells']:4d} / {r['grid_cells']:6d} cells"
            f" ({100 * r['occupancy']:.1f}%),"
            f" {r['clips_per_occupied_cell']:.2f} clips per occupied cell"
        )
        print(
            f"{'':14s} intents on >=10 objects: {r['intents_on_ten_or_more_objects']};"
            f" on exactly one: {r['intents_on_exactly_one_object']}"
        )
    print("")
    print(
        f"crossing vs interaction excess over {len(axes)} screened axes:"
        f" Pearson r = {pr:+.3f} (p = {pp:.3f}),"
        f" Spearman rho = {sr:+.3f} (p = {sp:.3f})"
    )
    for a in sorted(axes, key=lambda a: -a["occupancy"]):
        print(
            f"  {a['dataset']:13s} {a['axis'][:32]:32s}"
            f" {100 * a['occupancy']:5.1f}% crossed  excess {a['interaction_excess']:+.4f}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
