"""Two diagnostic OakInk2 bundles for the OakInk2 compositional null, plus a screen.

    python scripts/build_oakink2_diagnostics.py

Mirror of ``scripts/build_grab_diagnostics.py`` for OakInk2's scene x primitive axis.

A (dilution) -> data/bundles/oakink2_primseg.npz
    Each 609-trajectory recording in ``oakink2_scene_primitive.npz`` is cut back
    into its annotated right-hand primitive segments. The span table is
    ``meta['segments'][i]['spans']``, copied by ``caredex.data.oakink2`` from
    ``program_info/<seq>.json`` keys ``(lh_span, rh_span)``. The bundle trajectory
    is the concatenation, in span order, of ``range(a, b, stride)`` per span, so
    segment k is rows ``[sum n_<k, sum n_<=k)``. That mapping is asserted to
    reproduce every trajectory length before anything is cut. Each segment is
    labelled ``<its primitive>@scene@subject@verb``, so
    ``coarsen_labels(..., "oakink2_scene_primitive")`` gives ``scene->primitive``
    for the segment's own primitive, where the full bundle used the chain's first.

B (insensitivity) -> data/bundles/oakink2_planted.npz
    ``build_grab_diagnostics.double_centre`` pure-interaction per-cell offset over
    scene x primitive cells, seed 0, RMS = synth_hard frame RMS (0.166 normalised),
    added in normalised space and clamped.

What this does NOT show
-----------------------
- Segments are still long: the median primitive segment is 182 frames (24 s at
  7.5 fps), against a 32-frame (4.27 s) window. This removes dilution *across*
  primitives, not dilution within one.
- A segment's fine label drops the rest of its chain. Segments of one recording
  sit in different cells and can land on different sides of the split. The leak
  gate checks fine labels, not recordings, so it cannot see that.
- The primitive label is the right-hand annotation only; ``_segments_for`` falls
  back to the left-hand span when the right span is missing.
- The planted offset is constant over an 82 s trajectory. Many primitives occur
  in one scene only; double-centring forces those cells to zero offset, so the
  plant lives in the few primitives seen in several scenes (counted below).
- The screen is a linear variance decomposition, not the penalty. A constant
  per-cell plant is what a cell one-hot captures, so the planted screen rising
  checks the plant, not the instrument. Only a sweep tests insensitivity.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import DOF_NAMES, LIMITS_HI, LIMITS_LO, N_DOF, denormalize, normalize  # noqa: E402
from experiment_paired_composition import coarsen_labels  # noqa: E402
from screen_axes import screen  # noqa: E402
from build_grab_diagnostics import double_centre, synthetic_via_magnitude  # noqa: E402

B = ROOT / "data" / "bundles"
MODE = "oakink2_scene_primitive"
SRC = B / "oakink2_scene_primitive.npz"
MIN_LEN = 32
MIN_LEN_ALT = 16


# --------------------------------------------------------------------- Task A
def build_primseg() -> dict:
    src = TrajectoryBundle.load(SRC)
    stride = int(src.meta["stride"])
    segs = src.meta["segments"]
    assert len(segs) == len(src.labels) == len(src.trajectories)

    # Verify the index mapping on every trajectory before cutting anything.
    for tr, row, lab in zip(src.trajectories, segs, src.labels):
        n = [len(range(a, max(b, a + 1), stride)) for a, b in row["spans"]]
        assert sum(n) == len(tr), (row["sequence"], sum(n), len(tr))
        assert len(row["spans"]) == len(row["primitives"])
        chain, _, suffix = lab.partition("@")
        assert chain.split("->") == list(row["primitives"]), (chain, row["primitives"])
    print(f"[A] span->row mapping reproduces all {len(segs)} trajectory lengths (stride {stride})")

    all_len, kept_tr, kept_lab, kept_rows = [], [], [], []
    alt_tr = 0
    for tr, row, lab in zip(src.trajectories, segs, src.labels):
        _, _, suffix = lab.partition("@")
        start = 0
        for (a, b), prim in zip(row["spans"], row["primitives"]):
            n = len(range(a, max(b, a + 1), stride))
            seg = tr[start:start + n]
            start += n
            all_len.append(n)
            if n >= MIN_LEN_ALT:
                alt_tr += 1
            if n < MIN_LEN:
                continue
            kept_tr.append(np.ascontiguousarray(seg))
            kept_lab.append(f"{prim}@{suffix}")
            kept_rows.append({"sequence": row["sequence"], "task": row["task"],
                              "primitives": [prim], "spans": [(a, b)]})

    all_len = np.array(all_len)
    kl = np.array([len(t) for t in kept_tr])
    meta = {k: v for k, v in src.meta.items() if k != "segments"}
    meta.update(source="oakink2_primseg", unit="segment",
                derived_from="data/bundles/oakink2_scene_primitive.npz",
                segment_rule=f"one trajectory per right-hand primitive span, drop if < {MIN_LEN} frames",
                label_format="primitive@scene@subject@verb", segments=kept_rows,
                granularity=MODE, axis="scene x primitive (segment's own primitive)")
    out = TrajectoryBundle(trajectories=kept_tr, fps=src.fps, labels=kept_lab, meta=meta)
    out.save(B / "oakink2_primseg.npz")
    back = TrajectoryBundle.load(B / "oakink2_primseg.npz")
    assert len(back.trajectories) == len(kept_tr) and list(back.labels) == kept_lab
    assert all(np.array_equal(x, y) for x, y in zip(back.trajectories, kept_tr))

    cells_full = Counter(coarsen_labels(list(src.labels), MODE))
    cells = Counter(coarsen_labels(kept_lab, MODE))
    print(f"[A] segments {len(all_len)}; kept >= {MIN_LEN}: {len(kept_tr)}; >= {MIN_LEN_ALT}: {alt_tr}")
    print(f"[A] all-segment length pct 0/10/25/50/75/90/100: "
          f"{np.percentile(all_len, [0, 10, 25, 50, 75, 90, 100]).tolist()}")
    print(f"[A] kept median {np.median(kl):.0f} frames ({np.median(kl) / src.fps:.1f} s); "
          f"full-trajectory median {np.median([len(t) for t in src.trajectories]):.0f}")
    print(f"[A] cells {len(cells_full)} (full) -> {len(cells)} (segments); "
          f"per-cell count median {np.median(list(cells.values())):.0f}, max {max(cells.values())}")
    return dict(
        n_segments=int(len(all_len)), kept=len(kept_tr), kept_ge16=alt_tr,
        len_percentiles_all={str(p): float(v) for p, v in
                             zip((0, 10, 25, 50, 75, 90, 100),
                                 np.percentile(all_len, [0, 10, 25, 50, 75, 90, 100]))},
        median_len_kept=float(np.median(kl)), median_len_kept_s=float(np.median(kl) / src.fps),
        n_cells_full=len(cells_full), n_cells=len(cells),
        cells_ge5=int(sum(v >= 5 for v in cells.values())),
        cell_counts=dict(sorted(cells.items(), key=lambda kv: -kv[1])),
    )


# --------------------------------------------------------------------- Task B
def build_planted(target_rms: float) -> dict:
    full = TrajectoryBundle.load(SRC)
    coarse = coarsen_labels(list(full.labels), MODE)
    lefts = sorted({c.partition("->")[0] for c in coarse})
    rights = sorted({c.partition("->")[2] for c in coarse})
    li = {v: i for i, v in enumerate(lefts)}
    ri = {v: i for i, v in enumerate(rights)}
    mask = np.zeros((len(lefts), len(rights)), bool)
    for c in coarse:
        a, _, b = c.partition("->")
        mask[li[a], ri[b]] = True

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((len(lefts), len(rights), N_DOF))
    inter, iters = double_centre(raw, mask)
    rms = float(np.sqrt((inter[mask] ** 2).mean()))
    inter *= target_rms / rms
    zero = np.abs(inter).max(axis=2) < 1e-9
    zero_cells = int((zero & mask).sum())
    traj_zero = sum(1 for c in coarse if zero[li[c.partition("->")[0]], ri[c.partition("->")[2]]])

    trajs, changed, total, sq_int, sq_real, n_frames = [], 0, 0, 0.0, 0.0, 0
    per_dof = np.zeros(N_DOF)
    for tr, c in zip(full.trajectories, coarse):
        a, _, b = c.partition("->")
        o = inter[li[a], ri[b]].astype(np.float32)
        q = denormalize(normalize(tr) + o)
        qc = np.clip(q, LIMITS_LO, LIMITS_HI).astype(np.float32)
        ch = ~np.isclose(q, qc, rtol=0, atol=1e-6)
        changed += int(ch.sum())
        per_dof += ch.sum(axis=0)
        total += ch.size
        realised = normalize(qc) - normalize(tr)
        sq_int += float((o.astype(np.float64) ** 2).sum()) * len(tr)
        sq_real += float((realised.astype(np.float64) ** 2).sum())
        n_frames += len(tr)
        trajs.append(qc)

    meta = dict(full.meta)
    meta.update(source="oakink2_planted", derived_from="data/bundles/oakink2_scene_primitive.npz",
                planted_axis=f"granularity={MODE}", planted_seed=0, planted_rms_norm=float(target_rms),
                planted_rule="build_grab_diagnostics.double_centre per-cell 27-D N(0,1) offset, "
                             "scaled to RMS over occupied cells, added in normalised space, clamped")
    out = TrajectoryBundle(trajectories=trajs, fps=full.fps, labels=list(full.labels), meta=meta)
    out.save(B / "oakink2_planted.npz")
    back = TrajectoryBundle.load(B / "oakink2_planted.npz")
    assert list(back.labels) == list(full.labels) and len(back.trajectories) == len(trajs)

    top = sorted(zip(per_dof / (total / N_DOF), DOF_NAMES), reverse=True)[:6]
    res = dict(
        n_left=len(lefts), n_right=len(rights), n_cells=int(mask.sum()),
        centring_iterations=iters, cells_with_zero_offset=zero_cells,
        trajectories_with_zero_offset=traj_zero, target_rms_norm=target_rms,
        frame_rms_before_clamp=float(np.sqrt(sq_int / (n_frames * N_DOF))),
        frame_rms_after_clamp=float(np.sqrt(sq_real / (n_frames * N_DOF))),
        clamp_fraction=changed / total,
        clamp_fraction_top_dofs={n: float(f) for f, n in top},
    )
    print(f"[B] {res['n_cells']} cells ({len(lefts)} scenes x {len(rights)} primitives); "
          f"{iters} centring passes; {zero_cells} cells ({traj_zero} trajectories) forced to zero")
    print(f"[B] target RMS {target_rms:.4f}; frame RMS before clamp "
          f"{res['frame_rms_before_clamp']:.4f}, after {res['frame_rms_after_clamp']:.4f}")
    print(f"[B] clamp changed {res['clamp_fraction']:.4%} of values; worst DOFs {top}")
    return res


def main() -> int:
    report: dict = {"primseg": build_primseg()}
    via = synthetic_via_magnitude()
    report["synthetic_via"] = via
    target = via["frame_rms_norm_27dof"]
    assert abs(target - 0.166) < 1e-3, target
    report["planted"] = build_planted(target)

    ref_doc = json.loads((ROOT / "runs" / "oakink2_axis_screen.json").read_text(encoding="utf-8"))
    cfg = ref_doc["config"]
    ref = next(r for r in ref_doc["rows"] if r["axis"] == "scene x primitive")
    rows = {}
    for name in ("oakink2_scene_primitive", "oakink2_primseg", "oakink2_planted"):
        r = screen(B / f"{name}.npz", MODE, cfg["window"], cfg["stride"],
                   cfg["max_per_traj"], cfg["n_perm"], cfg["seed"])
        rows[name] = r
        print(f"[C] {name:<24} traj {r['n_traj']:>5} cells {r['n_cells']:>3} "
              f"additive {r['additive_r2']:.4f} inter {r['interaction_r2']:.4f} "
              f"excess {r['excess']:+.4f} z {r['z']:.2f}")
    base = rows["oakink2_scene_primitive"]
    assert abs(base["excess"] - ref["excess"]) < 5e-4, (base["excess"], ref["excess"])
    assert abs(base["z"] - ref["z"]) < 0.2, (base["z"], ref["z"])
    print(f"[C] reproduces runs/oakink2_axis_screen.json (excess {ref['excess']:+.4f}, z {ref['z']:.2f})")

    out = ROOT / "runs" / "axis_screen_oakink2_diagnostics.json"
    out.write_text(json.dumps({
        "config": {**cfg, "mode": MODE, "min_len": MIN_LEN},
        "reference": ref, "rows": rows, "build": report,
        "caveat": "Screen is linear variance decomposition, not the penalty. A constant per-cell "
                  "plant is what a cell one-hot captures, so oakink2_planted rising validates the "
                  "plant, not the instrument. Primitive segments median 24 s vs a 4.27 s window.",
    }, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
