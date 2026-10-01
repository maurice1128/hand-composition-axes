"""Search OakInk-Image for composition axes with NON-POSITIVE interaction excess.

Every OakInk-Image axis swept so far has positive excess and carries difficulty;
every GRAB / OakInk2 axis carries none. "Difficulty needs positive interaction"
and "difficulty is an OakInk-Image property" are therefore confounded. An
OakInk-Image axis with non-positive excess separates them. This script finds
candidates on CPU only: no training, no sweeps.

Stages
------
1. Per-trajectory factors, from the sequence id (object, intent, subject,
   capture date) and OakInk's metaV2 (category, class, first / second affordance
   attribute, provenance ``from``) plus the id prefix.
2. ``screen_axes.screen`` on every pair of distinct, non-nested factors, with the
   stored screen's config (window 32, stride 16, max-per-traj 6, n-perm 20,
   seed 0). The stored OakInk-Image rows are reproduced first and asserted.
3. For non-positive pairs (and the two smallest positive ones) a relabelled
   bundle ``data/bundles/oakink_<pair>.npz`` is written and the coverage and
   leak gates are run on it.

How a new axis is expressed without editing the experiment
----------------------------------------------------------
``coarsen_labels`` has no generic OakInk-Image mode. Its ``oakink2_scene_verb``
mode, however, only parses ``chain@scene@subject@verb`` and emits
``scene->verb``, keeping the whole string as the fine label. The relabelled
bundles therefore store ``<fine>@<left>@-@<right>`` and are run with
``--granularity oakink2_scene_verb``. The mode's *name* is then wrong for these
bundles; the bundle's ``meta['label_format']`` says so. A clean alternative is a
generic ``prebuilt`` mode (``label.split('@')`` -> ``f"{parts[1]}->{parts[3]}"``)
added to ``coarsen_labels`` and the three ``--granularity`` choice lists.

The fine label (the near-duplicate unit the leak-free split keeps on one side)
is ``object->intent`` -- exactly what ``oakink_category`` uses -- refined by any
axis factor that ``(object, intent)`` does not determine (subject, date). It
must refine the coarse cell or one fine group would straddle cells. Caveat,
reported per axis: under that fine label a naive or informed training set can
still hold the *same object and intent performed by another subject*; the
stricter object->intent overlap is measured and stored as a diagnostic.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import screen_axes  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.oakink_meta import _tables, object_group  # noqa: E402

BUNDLE = ROOT / "data" / "bundles" / "oakink.npz"
CFG = dict(window=32, stride=16, max_per_traj=6, n_perm=20, seed=0)
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
GATE_MODE = "oakink2_scene_verb"


def factors_of(bundle: TrajectoryBundle) -> dict[str, list[str]]:
    objects, _ = _tables()
    out: dict[str, list[str]] = defaultdict(list)
    for sid in bundle.meta["sequence_ids"]:
        head, _, ts = sid.partition("__")
        f = head.split("_")
        obj = f[0]
        attrs = [a for a in (objects[obj].get("attr") or []) if str(a).strip()]
        out["object"].append(obj)
        out["category"].append(object_group(obj, "category"))
        out["class"].append(object_group(obj, "class"))
        out["attr"].append(object_group(obj, "attr"))
        out["attr2"].append(str(attrs[1]) if len(attrs) > 1 else "none")
        out["prefix"].append(obj[:1])
        out["from"].append(str(objects[obj].get("from")))
        out["intent"].append(f[1])
        out["subject"].append(f[2])
        out["date"].append(ts[:10])
    return dict(out)


def functional(a: list[str], b: list[str]) -> bool:
    """True when every level of ``a`` maps to exactly one level of ``b``."""
    m: dict[str, set] = defaultdict(set)
    for x, y in zip(a, b):
        m[x].add(y)
    return all(len(v) == 1 for v in m.values())


def run_screen(labels: list[str]) -> dict:
    """``screen_axes.screen`` with the coarse labels supplied directly."""
    original = screen_axes.derive
    screen_axes.derive = lambda _labels, _mode: labels
    try:
        return screen_axes.screen(BUNDLE, "injected", CFG["window"], CFG["stride"],
                                  CFG["max_per_traj"], CFG["n_perm"], CFG["seed"])
    finally:
        screen_axes.derive = original


def fine_label(fac: dict, i: int, a: str, b: str, determined: set[str]) -> str:
    extra = [f"{k}={fac[k][i]}" for k in (a, b) if k not in determined]
    return f"{fac['object'][i]}->{fac['intent'][i]}" + ("|" + "|".join(extra) if extra else "")


def gate(cmd: list[str], out: Path) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": "8", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    text = f"$ {' '.join(cmd)}\n" + p.stdout + (p.stderr if p.returncode not in (0, 1) or "Traceback" in p.stderr else "")
    out.write_text(text, encoding="utf-8")
    lines = p.stdout.splitlines()
    return {
        "exit": p.returncode,
        "transcript": str(out.relative_to(ROOT)),
        "rows_thin": sum("TOO THIN" in l for l in lines),
        "rows_leak": sum(l.rstrip().endswith("LEAK") for l in lines),
        "rows_ok": sum(l.rstrip().endswith(" ok") for l in lines),
        "summary": next((l for l in reversed(lines) if l.startswith(("ARTEFACT", "PASS", "FAIL", "Coverage"))), None),
        "error": p.stderr.strip().splitlines()[-1] if p.returncode not in (0, 1) and p.stderr.strip() else None,
    }


def strict_overlap(bundle_path: Path, held: int, seeds: list[int], fac: dict) -> dict:
    """Fraction of target trajectories whose object->intent is in a training set."""
    from experiment_paired_composition import build_paired_split, coarsen_labels, sample_pools

    b = TrajectoryBundle.load(bundle_path)
    fine = list(b.labels)
    b.labels = coarsen_labels(b.labels, GATE_MODE)
    oi = [f"{o}->{t}" for o, t in zip(fac["object"], fac["intent"])]
    naive_l, inf_l = [], []
    for s in seeds:
        try:
            sp = build_paired_split(b, held, s, fine_labels=fine, min_chains=4)
            n, inf = sample_pools(sp, 256, s, b.labels, 5)
        except ValueError:
            continue
        for store, idx in ((naive_l, n), (inf_l, inf)):
            train = {oi[i] for i in idx}
            store.append(sum(oi[i] in train for i in sp["target"]) / len(sp["target"]))
    return {"naive_mean": float(np.mean(naive_l)) if naive_l else None,
            "informed_mean": float(np.mean(inf_l)) if inf_l else None,
            "n_seeds": len(naive_l)}


def main() -> int:
    t0 = time.time()
    bundle = TrajectoryBundle.load(BUNDLE)
    fac = factors_of(bundle)
    names = list(fac)
    levels = {k: len(set(v)) for k, v in fac.items()}
    print("factors:", levels, flush=True)

    # (object, intent) determine every factor that is a function of object.
    determined = {k for k in names if functional(fac["object"], fac[k])} | {"intent"}

    # ---- reproduce stored rows ------------------------------------------------
    stored = {r["mode"]: r for r in json.loads((ROOT / "runs" / "axis_screen.json")
              .read_text(encoding="utf-8"))["rows"] if r["dataset"] == "OakInk-Image"}
    repro = {}
    for mode, left in (("oakink_category", "category"), ("oakink_attr", "attr"),
                       ("oakink_class", "class"), ("category", "prefix")):
        via_module = screen_axes.screen(BUNDLE, mode, **CFG)
        via_factors = run_screen([f"{l}->{r}" for l, r in zip(fac[left], fac["intent"])])
        for got in (via_module, via_factors):
            assert abs(got["excess"] - stored[mode]["excess"]) < 1e-6, (mode, got["excess"])
            assert abs(got["z"] - stored[mode]["z"]) < 1e-3, (mode, got["z"])
        repro[mode] = {"excess": via_factors["excess"], "z": via_factors["z"],
                       "stored_excess": stored[mode]["excess"], "stored_z": stored[mode]["z"]}
        print(f"reproduced {mode}: excess {via_factors['excess']:+.4f} z {via_factors['z']:.1f}", flush=True)

    # ---- screen every pair ----------------------------------------------------
    rows, skipped = [], []
    for a, b in itertools.combinations(names, 2):
        if functional(fac[a], fac[b]) or functional(fac[b], fac[a]):
            which = f"{a} determines {b}" if functional(fac[a], fac[b]) else f"{b} determines {a}"
            skipped.append({"pair": f"{a}x{b}", "reason": f"nested: {which}"})
            continue
        labels = [f"{x}->{y}" for x, y in zip(fac[a], fac[b])]
        cells = Counter(labels)
        r = run_screen(labels)
        r.update(pair=f"{a}x{b}", left=a, right=b,
                 cells_possible=levels[a] * levels[b],
                 cells_ge5=sum(v >= 5 for v in cells.values()),
                 median_traj_per_cell=float(np.median(list(cells.values()))))
        rows.append(r)
        print(f"{a:>9} x {b:<9} cells {r['n_cells']:>4}/{r['cells_possible']:<5} "
              f"excess {r['excess']:+.4f}  z {r['z']:6.1f}   ({time.time() - t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: r["excess"])
    nonpos = [r for r in rows if r["excess"] <= 0]
    smallest_pos = [r for r in rows if r["excess"] > 0][:2]

    # ---- gates ----------------------------------------------------------------
    seeds = [str(s) for s in range(40)]
    gates = {}
    for r in nonpos + smallest_pos:
        a, b = r["left"], r["right"]
        tag = f"{a}_{b}"
        path = ROOT / "data" / "bundles" / f"oakink_{tag}.npz"
        fines = [fine_label(fac, i, a, b, determined) for i in range(len(bundle.labels))]
        new = TrajectoryBundle(
            trajectories=bundle.trajectories, fps=bundle.fps,
            labels=[f"{fl}@{x}@-@{y}" for fl, x, y in zip(fines, fac[a], fac[b])],
            meta={**bundle.meta, "label_format": f"fine@{a}@-@{b}; run with --granularity "
                  f"{GATE_MODE} (mode name is a parser reuse, not OakInk2)",
                  "relabelled_by": "scripts/search_oakink_axes.py"},
        )
        new.save(path)
        g = {"bundle": str(path.relative_to(ROOT)), "n_fine": len(set(fines)), "held": {}}
        for held in (5, 4, 3):
            suffix = "" if held == 5 else f"_h{held}"
            cov = gate([PY, "scripts/check_informed_coverage.py", "--bundle", str(path.relative_to(ROOT)),
                        "--granularity", GATE_MODE, "--held-compositions", str(held),
                        "--min-chains", "4", "--min-per-composition", "5", "--budgets", "256",
                        "--seeds", *seeds],
                       ROOT / "runs" / "gates" / f"cover_oakink_{tag}{suffix}.txt")
            leak = gate([PY, "scripts/check_composition_leak.py", "--bundle", str(path.relative_to(ROOT)),
                         "--granularity", GATE_MODE, "--held-compositions", str(held),
                         "--min-chains", "4", "--min-per-composition", "5", "--budget", "256",
                         "--seeds", *seeds],
                        ROOT / "runs" / "gates" / f"composition_leak_oakink_{tag}{suffix}.txt")
            g["held"][held] = {"coverage": cov, "leak": leak,
                               "object_intent_overlap": strict_overlap(path, held, list(range(40)), fac)}
            print(f"gate {tag} held {held}: coverage exit {cov['exit']} thin {cov['rows_thin']} "
                  f"ok {cov['rows_ok']}; leak exit {leak['exit']} leak rows {leak['rows_leak']}"
                  f"{'  ERR ' + str(cov['error'] or leak['error']) if (cov['error'] or leak['error']) else ''}",
                  flush=True)
            if cov["exit"] == 0 and leak["exit"] == 0:
                break
        gates[tag] = g

    out = ROOT / "runs" / "oakink_axis_search.json"
    out.write_text(json.dumps({
        "config": CFG, "bundle": str(BUNDLE.relative_to(ROOT)),
        "factors": levels, "determined_by_object_intent": sorted(determined),
        "reproduction": repro, "skipped": skipped, "rows": rows,
        "non_positive": [r["pair"] for r in nonpos],
        "smallest_positive": [r["pair"] for r in smallest_pos],
        "gates": gates, "gate_mode": GATE_MODE,
        "elapsed_s": time.time() - t0,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
