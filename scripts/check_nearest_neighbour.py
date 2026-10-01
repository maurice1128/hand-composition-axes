"""Nearest-neighbour redundancy between the paired design's training sets and its targets.

The question
------------
The paired composition experiment measures ``MSE_naive(T) - MSE_informed(T)``.
On OakInk-Image axes that penalty is 12-18% of naive error; on GRAB and OakInk2
it sits at the synthetic zero-truth control. One mundane explanation is
retrieval: the informed training set may simply contain windows much closer to
the target windows than anything the naive set holds, so the "penalty" is a
lookup advantage rather than anything about composition.

This script measures that gap directly, with no model:

    d_naive  = min RMS distance from a target window to any naive-train window
    d_inf    = the same against the informed-train windows
    d_extra  = against windows of informed trajectories NOT in the naive set
    d_heldx  = against windows of the swapped-in trajectories (split["informed_extra"])

Splits are reproduced by *importing* ``build_paired_split``, ``sample_pools``,
``coarsen_labels`` and ``assert_split_sound`` from
``experiment_paired_composition.py`` with each sweep's stored args, not by
reimplementing them. Where a results row stores a ``soundness`` record, the
reproduced split's record must match it exactly or the axis is failed.

What this does NOT show
-----------------------
* Windows are cut at stride 16 (training uses stride 4), so the nearest
  training window can be up to 8 frames out of phase with a better one that
  exists in the data. Distances are upper bounds on the stride-4 distances.
* Any set above 20,000 windows is randomly subsampled (fixed seed), which
  inflates its nearest-neighbour distances relative to an unsubsampled set.
* RMS is in joint-limit-normalised units over all 27 DOF and 32 frames. It
  weighs a wrist translation and a DIP angle the same way the training loss
  does; it is not a perceptual or kinematic distance.
* A redundancy gap is compatible with the retrieval explanation; it does not
  prove the models use retrieval. A correlation over nine axes is weak evidence.
* Window length is fixed in frames, so an OakInk2 window (7.5 fps) spans four
  times the wall-clock time of an OakInk-Image window (30 fps), and datasets
  differ in trajectory count and length; see the per-axis set sizes recorded.

    .venv/Scripts/python.exe scripts/check_nearest_neighbour.py
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.pipeline import WindowedTrajectoryDataset  # noqa: E402
from experiment_paired_composition import (  # noqa: E402
    assert_split_sound,
    build_paired_split,
    coarsen_labels,
    sample_pools,
)

# sweep dir, dataset, measured penalty as % of naive error (supplied with the task)
AXES = [
    ("oakink_official_category", "OakInk-Image", 15.2, "real"),
    ("oakink_official_attr", "OakInk-Image", 12.2, "real"),
    ("oakink_n70_v2", "OakInk-Image", 14.3, "real"),
    ("oakink_class_v1", "OakInk-Image", 17.9, "real"),
    ("grab_shape_v2", "GRAB", 0.7, "real"),
    ("grab_shapeclass", "GRAB", 2.3, "real"),
    ("oakink2_scene_primitive", "OakInk2", 2.3, "real"),
    ("oakink2_scene_verb", "OakInk2", 3.0, "real"),
    ("oakink2_paired_v2", "OakInk2", 3.4, "real"),
    ("pc_easy_rerun", "synthetic", 2.6, "control"),
    ("pc_hard_rerun", "synthetic", 9.1, "control"),
]

WINDOW = 32
STRIDE = 16
MAX_WINDOWS = 20_000
BUDGET = 256
SEEDS = [0, 1, 2, 3, 4]
TIE_TOL = 1e-6
MIN_FREE_GB = 1.5


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def free_commit_gb() -> float:
    cmd = ["powershell", "-NoProfile", "-Command",
           "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        return float(out.strip().splitlines()[-1])
    except Exception:
        return float("nan")


def peak_commit_gb() -> float:
    """This process's peak commit (PeakPagefileUsage), Windows only."""
    try:
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float("nan")
        return pmc.PeakPagefileUsage / 1024**3
    except Exception:
        return float("nan")


def wait_for_memory(max_wait_s: int = 1200) -> tuple[bool, float]:
    t0 = time.time()
    while True:
        gb = free_commit_gb()
        if not (gb < MIN_FREE_GB):  # nan passes: cannot measure, do not block forever
            return True, gb
        if time.time() - t0 >= max_wait_s:
            return False, gb
        print(f"  free commit {gb:.2f} GB < {MIN_FREE_GB} GB, waiting 60 s")
        time.sleep(60)


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # average ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    return pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def window_table(trajs: list[np.ndarray], idx: list[int]):
    """Normalised stride-16 windows of bundle trajectories ``idx``.

    Uses the training pipeline's own dataset class, so normalisation and window
    placement are exactly those of the scored target loader.
    """
    ds = WindowedTrajectoryDataset([trajs[i] for i in idx], WINDOW, STRIDE)
    owner = np.asarray(idx)[ds.index[:, 0]]
    return ds, owner


def select(owner: np.ndarray, members: set[int], key: np.ndarray) -> np.ndarray:
    """Window positions belonging to ``members``, capped at MAX_WINDOWS by a shared key.

    The key is one random number per window, shared by every set, so a window
    present in two sets is kept in both or dropped from both whenever the cap
    allows -- subsampling alone cannot make the two sets' nearest neighbours differ.
    """
    pos = np.flatnonzero(np.isin(owner, list(members)))
    if len(pos) > MAX_WINDOWS:
        pos = pos[np.argsort(key[pos], kind="mergesort")[:MAX_WINDOWS]]
        pos.sort()
    return pos


def features(ds: WindowedTrajectoryDataset, pos: np.ndarray) -> np.ndarray:
    out = np.empty((len(pos), WINDOW * ds.n_pose), np.float32)
    for k, p in enumerate(pos):
        t, s = ds.index[p]
        out[k] = ds.trajectories[t][s:s + WINDOW, : ds.n_pose].reshape(-1)
    return out


@torch.no_grad()
def nn_dists(T: np.ndarray, U: np.ndarray, sets: dict[str, np.ndarray], chunk: int = 256):
    """Min RMS distance from each row of T to rows of U restricted to each set.

    Candidates come from the float32 expansion |a|^2+|b|^2-2ab; the winning
    neighbour's distance is then recomputed exactly, so near-duplicates are not
    lost to cancellation.
    """
    Tt, Ut = torch.from_numpy(T), torch.from_numpy(U)
    un = (Ut * Ut).sum(1)
    ix = {k: torch.from_numpy(v.astype(np.int64)) for k, v in sets.items()}
    out = {k: np.full(len(T), np.nan, np.float32) for k in sets}
    for s in range(0, len(T), chunk):
        a = Tt[s:s + chunk]
        d2 = (a * a).sum(1, keepdim=True) + un[None, :] - 2.0 * (a @ Ut.T)
        for k, cols in ix.items():
            if len(cols) == 0:
                continue
            j = d2.index_select(1, cols).argmin(1)
            nn = Ut.index_select(0, cols[j])
            out[k][s:s + chunk] = ((a - nn) ** 2).mean(1).sqrt().numpy()
        del d2
    return out


def run_axis(sweep: str, log) -> dict:
    res = json.loads((ROOT / "runs" / sweep / "results.json").read_text(encoding="utf-8"))
    a = res["args"]
    stored = {}
    for r in res["results"]:
        if r["budget"] == BUDGET and "soundness" in r:
            stored.setdefault(r["seed"], r["soundness"])

    bundle = TrajectoryBundle.load(ROOT / a["bundle"].replace("\\", "/"))
    fine_labels = list(bundle.labels)
    if a["granularity"] != "fine":
        bundle.labels = coarsen_labels(bundle.labels, a["granularity"])
    trajs = bundle.trajectories
    print(f"  bundle={a['bundle']} n_traj={len(trajs)} fps={bundle.fps} "
          f"granularity={a['granularity']} held={a['held_compositions']} "
          f"min_chains={a['min_chains']} min_per_composition={a['min_per_composition']}")

    per_seed = []
    for seed in SEEDS:
        split = build_paired_split(bundle, a["held_compositions"], seed,
                                   fine_labels=fine_labels, min_chains=a["min_chains"])
        naive_idx, inf_idx = sample_pools(split, BUDGET, seed, bundle.labels,
                                          a["min_per_composition"])
        naive_idx = [int(i) for i in naive_idx]
        inf_idx = [int(i) for i in inf_idx]
        sound = assert_split_sound(split, naive_idx, inf_idx, fine_labels, bundle.labels)
        target = [int(i) for i in split["target"]]

        assert len(naive_idx) == len(set(naive_idx)) == BUDGET, "naive set size"
        assert len(inf_idx) == len(set(inf_idx)) == BUDGET, "informed set size"
        assert not (set(target) & (set(naive_idx) | set(inf_idx))), "target overlaps training"
        if seed in stored:
            if stored[seed] != sound:
                raise AssertionError(
                    f"seed {seed}: reproduced soundness {sound} != stored {stored[seed]}")
            match = "matches stored soundness record"
        else:
            match = "no stored split record for this seed"

        naive_s, inf_s = set(naive_idx), set(inf_idx)
        extra_s = inf_s - naive_s
        heldx_s = inf_s & set(int(i) for i in split["informed_extra"])

        train_union = sorted(naive_s | inf_s)
        ds_tr, own_tr = window_table(trajs, train_union)
        ds_tg, own_tg = window_table(trajs, target)
        rng = np.random.default_rng(10_000 + seed)
        key_tr = rng.random(len(own_tr))
        key_tg = rng.random(len(own_tg))

        pos = {
            "naive": select(own_tr, naive_s, key_tr),
            "informed": select(own_tr, inf_s, key_tr),
            "extra": select(own_tr, extra_s, key_tr),
            "heldx": select(own_tr, heldx_s, key_tr),
        }
        n_full = {k: int(np.isin(own_tr, list(s)).sum())
                  for k, s in (("naive", naive_s), ("informed", inf_s),
                               ("extra", extra_s), ("heldx", heldx_s))}
        upos = np.unique(np.concatenate(list(pos.values())))
        remap = np.full(len(own_tr), -1, np.int64)
        remap[upos] = np.arange(len(upos))
        sets = {k: remap[v] for k, v in pos.items()}
        tpos = select(own_tg, set(target), key_tg)
        n_target_full = len(own_tg)

        U = features(ds_tr, upos)
        T = features(ds_tg, tpos)
        del ds_tr, ds_tg
        gc.collect()

        d = nn_dists(T, U, sets)
        del U, T
        gc.collect()

        dn, di = d["naive"], d["informed"]
        closer = di < dn - TIE_TOL
        row = {
            "seed": seed,
            "split_check": match,
            "soundness": sound,
            "held_compositions": split["held_compositions"],
            "n_target_traj": len(target),
            "n_target_windows": int(len(tpos)),
            "n_target_windows_full": int(n_target_full),
            "n_traj": {"naive": len(naive_s), "informed": len(inf_s),
                       "extra": len(extra_s), "heldx": len(heldx_s)},
            "n_windows_used": {k: int(len(v)) for k, v in pos.items()},
            "n_windows_full": n_full,
            "mean_len_frames": {
                "target": float(np.mean([len(trajs[i]) for i in target])),
                "naive": float(np.mean([len(trajs[i]) for i in naive_idx])),
                "informed": float(np.mean([len(trajs[i]) for i in inf_idx])),
            },
            "mean_d_naive": float(dn.mean()),
            "mean_d_inf": float(di.mean()),
            "mean_d_extra": float(np.nanmean(d["extra"])),
            "mean_d_heldx": float(np.nanmean(d["heldx"])),
            "median_d_naive": float(np.median(dn)),
            "median_d_inf": float(np.median(di)),
            "rel_gap": float((dn.mean() - di.mean()) / dn.mean()),
            "frac_closer_informed": float(closer.mean()),
            "frac_extra_closer_than_naive": float((d["extra"] < dn - TIE_TOL).mean()),
        }
        per_seed.append(row)
        print(f"  seed {seed}: target {len(target)} traj / {len(tpos)} win "
              f"(naive {n_full['naive']} win, informed {n_full['informed']}, "
              f"extra {len(extra_s)} traj) | d_naive {row['mean_d_naive']:.4f} "
              f"d_inf {row['mean_d_inf']:.4f} d_extra {row['mean_d_extra']:.4f} "
              f"d_heldx {row['mean_d_heldx']:.4f} | gap {row['rel_gap']:+.3f} "
              f"closer {row['frac_closer_informed']:.3f} | {match}")
        del d, dn, di
        gc.collect()

    keys = ["mean_d_naive", "mean_d_inf", "mean_d_extra", "mean_d_heldx",
            "rel_gap", "frac_closer_informed", "frac_extra_closer_than_naive"]
    summary = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
    summary.update({f"{k}_sd": float(np.std([r[k] for r in per_seed], ddof=1)) for k in keys})
    return {"args": {k: a[k] for k in ("bundle", "granularity", "held_compositions",
                                        "min_chains", "min_per_composition")},
            "fps": bundle.fps, "summary": summary, "per_seed": per_seed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/nn_redundancy.json")
    ap.add_argument("--log", default="runs/gates/nn_redundancy.txt")
    ap.add_argument("--axes", nargs="*", default=None, help="subset of sweep dirs")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    sys.stdout = Tee(ROOT / args.log)
    torch.set_num_threads(args.threads)
    print(f"nearest-neighbour redundancy | window {WINDOW} stride {STRIDE} | cap {MAX_WINDOWS} "
          f"windows/set | budget {BUDGET} | seeds {SEEDS} | feature: all 32 frames x 27 DOF, "
          f"joint-limit normalised, no temporal subsampling | CPU torch float32")

    out = {"config": {"window": WINDOW, "stride": STRIDE, "max_windows": MAX_WINDOWS,
                      "budget": BUDGET, "seeds": SEEDS, "tie_tol": TIE_TOL,
                      "feature": "normalised 32x27 window, flattened, every frame",
                      "distance": "RMS over 864 entries, normalised units"},
           "axes": {}, "skipped": [], "failed": []}
    for sweep, dataset, measured, role in AXES:
        if args.axes and sweep not in args.axes:
            continue
        print(f"\n== {sweep} ({dataset}, {role}, measured {measured}% of naive)")
        ok, gb = wait_for_memory()
        print(f"  free commit before axis: {gb:.2f} GB")
        if not ok:
            print("  SKIPPED: free commit stayed below threshold for 20 min")
            out["skipped"].append({"axis": sweep, "free_commit_gb": gb})
            continue
        t0 = time.time()
        try:
            r = run_axis(sweep, print)
        except AssertionError as e:
            print(f"  ASSERTION FAILED: {e}")
            out["failed"].append({"axis": sweep, "error": str(e)})
            continue
        r.update({"dataset": dataset, "measured_pct_naive": measured, "role": role,
                  "seconds": round(time.time() - t0, 1), "peak_commit_gb_so_far": peak_commit_gb()})
        out["axes"][sweep] = r
        s = r["summary"]
        print(f"  => rel_gap {s['rel_gap']:+.3f} (sd {s['rel_gap_sd']:.3f})  "
              f"closer {s['frac_closer_informed']:.3f}  "
              f"[{r['seconds']} s, process peak commit {r['peak_commit_gb_so_far']:.2f} GB]")
        gc.collect()
        (ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    real = [(k, v) for k, v in out["axes"].items() if v["role"] == "real"]
    if len(real) >= 3:
        y = [v["measured_pct_naive"] for _, v in real]
        corr = {"n_axes": len(real), "axes": [k for k, _ in real]}
        for m in ("rel_gap", "frac_closer_informed"):
            x = [v["summary"][m] for _, v in real]
            corr[m] = {"pearson": pearson(x, y), "spearman": spearman(x, y)}
        out["correlations_real_axes"] = corr

    print("\n" + "=" * 96)
    print(f"{'axis':<26}{'dataset':<14}{'d_naive':>9}{'d_inf':>9}{'d_extra':>9}"
          f"{'rel_gap':>10}{'closer':>9}{'meas%':>8}")
    print("-" * 96)
    for k, v in out["axes"].items():
        s = v["summary"]
        print(f"{k:<26}{v['dataset']:<14}{s['mean_d_naive']:>9.4f}{s['mean_d_inf']:>9.4f}"
              f"{s['mean_d_extra']:>9.4f}{s['rel_gap']:>+10.3f}{s['frac_closer_informed']:>9.3f}"
              f"{v['measured_pct_naive']:>8.1f}")
    if "correlations_real_axes" in out:
        c = out["correlations_real_axes"]
        print(f"\nover {c['n_axes']} real axes, vs measured % naive:")
        for m in ("rel_gap", "frac_closer_informed"):
            print(f"  {m:<22} pearson {c[m]['pearson']:+.3f}  spearman {c[m]['spearman']:+.3f}")
    print(f"skipped: {out['skipped'] or 'none'}   failed: {out['failed'] or 'none'}")
    print(f"process peak commit: {peak_commit_gb():.2f} GB")
    (ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
