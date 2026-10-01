"""experiment_paired_composition.py, additionally scoring target windows BY WHICH CLIP THEY LIE IN.

Why
---
On the two-clip OakInk-Image bundles (`scripts/build_oakink_alignment_v4.py`) the penalty fell from +17.0% when the
label described both clips to +3.9% when it described only the first. Two things differ between those bundles and
they vary together, so the sweep as run cannot separate them, and a referee said so:

  purity         only about 61% of the misaligned target clips belong to a held-out cell, so much of what is scored
                 is not the held-out composition and carries no penalty by construction;
  contamination  8.4% of the naive arm's training clips belong to held-out cells, so the naive arm has seen the
                 held-out motion under other labels and is no longer naive.

They make different predictions about the windows that lie WHOLLY INSIDE THE FIRST CLIP, which is the clip the label
names in both bundles. Purity alone predicts that those windows carry the full penalty in the misaligned bundle, as
they do in the aligned one, and that the fall comes entirely from second-clip windows. Contamination predicts that
the penalty is reduced on first-clip windows too. Scored models discard their checkpoints, so this is a re-run; it
uses the original seeds, so its TOTAL penalty must reproduce the original sweep, which is itself a check.

How
---
`recon_mse` is replaced by a version that returns the same number and also keeps the per-window error. Windows of
the target loader are in `dataset.index` order (`shuffle=False`), each `(trajectory, start)`. A trajectory's join is
recovered from `bundle.meta['source_clips']` and the source bundle's clip lengths, and ASSERTED against the
trajectory's own length, so a wrong cross-fade convention stops the run instead of mislabelling windows. With
`x = crossfade_frames`, a window of `w` frames starting at `s` in a trajectory whose first clip has `L1` frames is

  first     s + w <= L1 - x      wholly the labelled clip, untouched by the cross-fade
  second    s     >= L1          wholly the other clip
  straddle  otherwise            discarded from both class means, counted

Per-class errors go to the sidecar `<out>/window_classes.json`, one row per (seed, budget, kind, arm). results.json
is written by the module and is unchanged.

What this does NOT do
---------------------
- It does not separate the two accounts if both act; it measures how much of the fall each window class carries.
- A first-clip window is pure in its LABEL. Whether the naive arm saw that motion elsewhere is the contamination
  being tested, not something this script removes.
- Class means are over fewer windows than the total (about three first-clip windows per trajectory at a median
  first clip of 72 frames), so they are noisier than the sweep's penalty.
- It is meaningful only for bundles built by build_oakink_alignment_v4.py; any other bundle is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_STATE: dict = {"target": None, "joins": None, "xfade": None, "last": None, "sidecar": None, "rows": {}}


def joins_of(bundle) -> tuple[list[int], int]:
    """First-clip length of every trajectory, checked against the trajectory's own length."""
    import numpy as np

    meta = bundle.meta or {}
    if "source_clips" not in meta or meta.get("concat_factor") != 2:
        raise ValueError("window classes need a two-clip bundle from build_oakink_alignment_v4.py "
                         "(meta.source_clips, concat_factor == 2)")
    x = int(meta["crossfade_frames"])
    src = np.load(ROOT / "data" / "bundles" / meta["derived_from"], allow_pickle=True)["lengths"]
    joins = []
    for i, (a, b) in enumerate(meta["source_clips"]):
        n, la, lb = len(bundle.trajectories[i]), int(src[a]), int(src[b])
        if n not in (la + lb - x, la + lb):
            raise ValueError(f"trajectory {i}: length {n} is neither {la}+{lb}-{x} nor {la}+{lb}; "
                             "the cross-fade convention is not the one assumed, refusing to classify windows")
        joins.append(la)
    return joins, x


def classify(index, target, joins, x, w):
    import numpy as np

    out = np.empty(len(index), dtype="<U8")
    for k, (i, s) in enumerate(index):
        L1 = joins[target[int(i)]]
        out[k] = "first" if s + w <= L1 - x else ("second" if s >= L1 else "straddle")
    return out


def install(E) -> None:
    import numpy as np
    import torch

    original_split, original_tas = E.build_paired_split, E.train_and_score

    def build_paired_split_recording(bundle, *a, **kw):
        split = original_split(bundle, *a, **kw)
        if _STATE["joins"] is None:
            _STATE["joins"], _STATE["xfade"] = joins_of(bundle)
        _STATE["target"] = list(split["target"])
        return split

    def recon_mse_by_window(model, loader, device, channels=None):
        model.eval()
        per = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred, tgt = model(batch)["recon"], batch
                if channels is not None:
                    pred, tgt = pred[..., channels], tgt[..., channels]
                per.append(((pred - tgt) ** 2).flatten(1).mean(1).cpu().numpy())
        per = np.concatenate(per)
        ds = loader.dataset
        cls = classify(ds.index, _STATE["target"], _STATE["joins"], _STATE["xfade"], ds.window)
        _STATE["last"] = {c: {"n": int((cls == c).sum()),
                              "mse": float(per[cls == c].mean()) if (cls == c).any() else None}
                          for c in ("first", "second", "straddle")}
        # Every window has the same number of elements, so the mean of per-window means IS the
        # module's sum-over-elements / count. Returned so results.json is what it always was.
        return float(per.mean())

    def train_and_score_recording(kind, train_idx, bundle, target_loader, args, device, run_dir, *a, **kw):
        _STATE["last"] = None
        out = original_tas(kind, train_idx, bundle, target_loader, args, device, run_dir, *a, **kw)
        if _STATE["last"] is not None and _STATE["sidecar"] is not None:
            _STATE["rows"][Path(run_dir).name] = {"mse_target": out["mse_target"], **_STATE["last"]}
            p = Path(_STATE["sidecar"])
            p.parent.mkdir(parents=True, exist_ok=True)
            prior = {}
            if p.exists():
                try:
                    prior = json.loads(p.read_text(encoding="utf-8")).get("runs", {})
                except (OSError, ValueError):
                    prior = {}
            prior.update(_STATE["rows"])
            p.write_text(json.dumps({
                "wrapper": "scripts/experiment_paired_window_classes.py",
                "classes": "first: s+w <= L1-x; second: s >= L1; straddle: otherwise",
                "crossfade_frames": _STATE["xfade"], "runs": prior}, indent=1), encoding="utf-8")
        return out

    E.build_paired_split = build_paired_split_recording
    E.recon_mse = recon_mse_by_window
    E.train_and_score = train_and_score_recording


def dry_run(E, a) -> int:
    import numpy as np
    from caredex.data.base import TrajectoryBundle
    from caredex.data.pipeline import WindowedTrajectoryDataset

    bundle = TrajectoryBundle.load(a.bundle)
    fine = list(bundle.labels)
    joins, x = joins_of(bundle)
    print(f"{a.bundle}: {len(joins)} trajectories, cross-fade {x}, first clip median {np.median(joins):.0f} frames")
    if a.granularity != "fine":
        bundle.labels = E.coarsen_labels(bundle.labels, a.granularity)
    counts = []
    for seed in (a.seeds or [a.seed]):
        split = E.build_paired_split(bundle, a.held_compositions, seed, fine_labels=fine, min_chains=a.min_chains)
        ds = WindowedTrajectoryDataset([bundle.trajectories[i] for i in split["target"]], a.window, a.window // 2)
        cls = classify(ds.index, list(split["target"]), joins, x, a.window)
        counts.append({c: int((cls == c).sum()) for c in ("first", "second", "straddle")})
    for c in ("first", "second", "straddle"):
        v = [r[c] for r in counts]
        print(f"  {c:9s} windows per seed: mean {np.mean(v):6.1f}  min {min(v)}  max {max(v)}")
    thin = sum(r["first"] < 20 for r in counts)
    print(f"seeds with fewer than 20 first-clip windows: {thin} of {len(counts)}")
    print("DRY RUN: nothing was trained.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dry-run", action="store_true")
    mine, rest = pre.parse_known_args(argv)
    if mine.dry_run:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("OMP_NUM_THREADS", "2")

    import experiment_paired_composition as E

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--bundle", default="data/bundles/oakink.npz")
    peek.add_argument("--out", default="runs/paired")
    peek.add_argument("--held-compositions", type=int, default=8)
    peek.add_argument("--min-chains", type=int, default=4)
    peek.add_argument("--granularity", default="fine")
    peek.add_argument("--window", type=int, default=32)
    peek.add_argument("--seed", type=int, default=0)
    peek.add_argument("--seeds", type=int, nargs="*", default=None)
    a, _ = peek.parse_known_args(rest)

    if mine.dry_run and not ("-h" in argv or "--help" in argv):
        return dry_run(E, a)

    install(E)
    # main() resolves these names in its own module globals at call time; prove the patch is what it will find.
    for name in ("build_paired_split", "recon_mse", "train_and_score"):
        assert E.main.__globals__[name] is getattr(E, name), name
    _STATE["sidecar"] = str(Path(a.out) / "window_classes.json")
    sys.argv = [sys.argv[0], *rest]
    return E.main()


if __name__ == "__main__":
    raise SystemExit(main())
