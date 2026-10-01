"""Two diagnostic GRAB bundles for the GRAB/OakInk2 compositional null, plus a screen.

    python scripts/build_grab_diagnostics.py

H2 (dilution) -> data/bundles/grab_grasp.npz
    Each trajectory cut to its first-to-last right-hand contact frame, padded by
    PAD frames each side, dropped if shorter than MIN_LEN. The label is kept. If
    the null was the label being diluted over long reach/release motion, the
    grasp-only bundle should show interaction the full bundle does not.

H3 (insensitivity) -> data/bundles/grab_planted.npz
    A known, pure-interaction offset added per (shape, fine intent) cell. If the
    instrument cannot recover *planted* difficulty on real GRAB motion, a GRAB
    null says nothing about GRAB.

What this does NOT show
-----------------------
- The screen is a linear variance decomposition, not the compositional penalty.
  A planted offset that is constant per cell is exactly what a cell one-hot
  captures, so the planted screen rising is close to guaranteed; it checks the
  plant, not the instrument. Only the sweep tests H3.
- The planted offset is constant over the trajectory. The synthetic via-point is
  a mid-transition bump. Matching RMS does not make them equally hard to learn.
- Clamping to joint limits removes part of the plant where GRAB sits at a limit
  (6 of 27 DOF are pinned or dead), so the realised offset is not pure
  interaction any more; both magnitudes are reported.
- Contact is "any of 16 right-hand region fractions > 0", i.e. any touching,
  including incidental touches before or after the manipulation.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.synthetic import minimum_jerk  # noqa: E402
from caredex.hand_model import (  # noqa: E402
    ARTICULATED_SLICE, LIMITS_HI, LIMITS_LO, N_ARTICULATED, N_DOF, denormalize, normalize,
)
from experiment_paired_composition import coarsen_labels  # noqa: E402
from screen_axes import screen  # noqa: E402

B = ROOT / "data" / "bundles"
PAD = 8
MIN_LEN = 40
MODE = "shape"


def cell_counts(labels: list[str]) -> Counter:
    return Counter(coarsen_labels(list(labels), MODE))


# --------------------------------------------------------------------- Task A
def build_grasp() -> dict:
    full = TrajectoryBundle.load(B / "grab.npz")
    con = TrajectoryBundle.load(B / "grab_contact.npz")
    assert len(full.trajectories) == len(con.trajectories)
    assert list(full.labels) == list(con.labels), "labels differ from grab.npz"
    for a, b in zip(full.trajectories, con.trajectories):
        assert a.shape == b.shape and np.array_equal(a, b), "frames differ from grab.npz"
    assert con.aux is not None and np.isclose(con.fps, full.fps)
    print(f"[A] grab_contact frames/labels identical to grab.npz "
          f"({len(full.trajectories)} traj, {full.n_frames} frames)")

    kept_tr, kept_lab, seg_len, dropped = [], [], [], []
    no_contact = 0
    for i, (tr, aux) in enumerate(zip(con.trajectories, con.aux)):
        touch = np.flatnonzero((aux > 0).any(axis=1))
        if len(touch) == 0:
            no_contact += 1
            dropped.append(i)
            continue
        lo = max(int(touch[0]) - PAD, 0)
        hi = min(int(touch[-1]) + PAD + 1, len(tr))
        if hi - lo < MIN_LEN:
            dropped.append(i)
            continue
        kept_tr.append(np.ascontiguousarray(tr[lo:hi]))
        kept_lab.append(con.labels[i])
        seg_len.append(hi - lo)

    before = np.array([len(t) for t in con.trajectories])
    after = np.array(seg_len)
    meta = dict(full.meta)
    meta.update(
        source="grab_grasp",
        derived_from="data/bundles/grab_contact.npz",
        segment_rule=f"first-to-last frame with any right-hand aux>0, +/-{PAD} frames, "
                     f"drop if < {MIN_LEN} frames",
        n_dropped=len(dropped),
    )
    out = TrajectoryBundle(trajectories=kept_tr, fps=con.fps, labels=kept_lab, meta=meta)
    out.save(B / "grab_grasp.npz")
    back = TrajectoryBundle.load(B / "grab_grasp.npz")
    assert back.aux is None and len(back.trajectories) == len(kept_tr)
    assert all(np.array_equal(a, b) for a, b in zip(back.trajectories, kept_tr))
    assert list(back.labels) == kept_lab
    coarsen_labels(list(back.labels), MODE)  # raises if the shape axis cannot apply

    cells_before, cells_after = cell_counts(con.labels), cell_counts(kept_lab)
    frac = after.sum() / before.sum()
    print(f"[A] kept {len(kept_tr)}  dropped {len(dropped)} "
          f"({no_contact} with no contact at all)")
    print(f"[A] median length {np.median(before):.0f} -> {np.median(after):.0f} frames "
          f"({np.median(before) / con.fps:.2f} s -> {np.median(after) / con.fps:.2f} s); "
          f"fraction of frames kept {frac:.3f}")
    print(f"[A] shape cells {len(cells_before)} -> {len(cells_after)}; per-cell counts (after/before):")
    for c in sorted(cells_before):
        print(f"      {c:<32} {cells_after.get(c, 0):>3} / {cells_before[c]}")
    return dict(
        kept=len(kept_tr), dropped=len(dropped), no_contact=no_contact,
        median_len_before=float(np.median(before)), median_len_after=float(np.median(after)),
        frac_frames_kept=float(frac), n_cells_before=len(cells_before),
        n_cells_after=len(cells_after),
        cell_counts_after=dict(sorted(cells_after.items())),
        cell_counts_before=dict(sorted(cells_before.items())),
    )


# --------------------------------------------------------------------- Task B
def synthetic_via_magnitude() -> dict:
    """Reproduce synth_hard.npz's via-points exactly and measure them normalised.

    SyntheticSource.load draws ``basis`` then ``coeff`` as the first two calls on
    ``default_rng(seed)``, so the 9x9 via table is recoverable without
    regenerating the bundle. Checked against the bundle's own meta.
    """
    d = np.load(B / "synth_hard.npz", allow_pickle=True)
    import ast
    meta = ast.literal_eval(str(d["meta"]))
    deg, rank, n_prims = meta["transition_via_deg"], meta["noise_rank"], meta["n_primitives"]
    rng = np.random.default_rng(meta["seed"])
    basis = rng.standard_normal((rank, N_ARTICULATED)).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coeff = rng.standard_normal((n_prims, n_prims, rank)).astype(np.float32)
    via = deg * (coeff @ basis)                                   # (9, 9, 21) degrees
    half_span = 0.5 * (LIMITS_HI - LIMITS_LO)[ARTICULATED_SLICE]
    via27 = np.zeros((n_prims, n_prims, N_DOF))
    via27[..., ARTICULATED_SLICE] = via / half_span               # normalised units
    peak_rms27 = float(np.sqrt((via27 ** 2).mean()))
    peak_rms21 = float(np.sqrt((via27[..., ARTICULATED_SLICE] ** 2).mean()))
    # Frame-weighted: the bump sin(pi*w) over a minimum-jerk w, seg_frames 25..60.
    ms = [float((np.sin(np.pi * minimum_jerk(n)) ** 2).mean()) for n in range(25, 61)]
    bump_ms = float(np.mean(ms))
    # Interaction-only share of the via table (double-centred over the 9x9 pairs).
    inter = via27 - via27.mean(0, keepdims=True) - via27.mean(1, keepdims=True) \
        + via27.mean((0, 1), keepdims=True)
    return dict(
        transition_via_deg=deg, noise_rank=rank, n_primitives=n_prims,
        peak_rms_norm_27dof=peak_rms27, peak_rms_norm_21dof=peak_rms21,
        peak_rms_deg_21dof=float(np.sqrt((via ** 2).mean())),
        bump_mean_square=bump_ms,
        frame_rms_norm_27dof=float(peak_rms27 * np.sqrt(bump_ms)),
        interaction_rms_norm_27dof=float(np.sqrt((inter ** 2).mean())),
    )


def double_centre(off: np.ndarray, mask: np.ndarray, iters: int = 500) -> tuple[np.ndarray, int]:
    """Row/column centring over occupied cells, repeated to convergence.

    One pass of (x - row mean - col mean + grand mean) is exact only for a fully
    occupied table; GRAB's shape x intent table is sparse, so the pass is
    repeated (alternating projections), which converges to the least-squares
    residual of an additive fit over occupied cells.
    """
    x = np.where(mask[..., None], off, 0.0)
    m = mask[..., None].astype(float)
    for it in range(iters):
        r = (x * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)
        x = (x - r) * m
        c = (x * m).sum(0, keepdims=True) / np.maximum(m.sum(0, keepdims=True), 1)
        x = (x - c) * m
        r2 = (x * m).sum(1) / np.maximum(m.sum(1), 1)
        if np.abs(r2).max() < 1e-12:
            return x, it + 1
    return x, iters


def build_planted(target_rms: float) -> dict:
    full = TrajectoryBundle.load(B / "grab.npz")
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
    # One-pass formula, reported so the sparse-table correction is visible.
    mm = mask[..., None].astype(float)
    rw = np.where(mask[..., None], raw, 0.0)
    rmean = (rw * mm).sum(1, keepdims=True) / np.maximum(mm.sum(1, keepdims=True), 1)
    cmean = (rw * mm).sum(0, keepdims=True) / np.maximum(mm.sum(0, keepdims=True), 1)
    gmean = (rw * mm).sum((0, 1), keepdims=True) / mm.sum()
    one_pass = (rw - rmean - cmean + gmean) * mm
    resid_row_one = float(np.abs((one_pass * mm).sum(1) / np.maximum(mm.sum(1), 1)).max())
    resid_col_one = float(np.abs((one_pass * mm).sum(0) / np.maximum(mm.sum(0), 1)).max())

    inter, iters = double_centre(raw, mask)
    occ = inter[mask]                                             # (n_cells, 27)
    rms = float(np.sqrt((occ ** 2).mean()))
    inter *= target_rms / rms
    zero_cells = int((np.abs(inter[mask]).max(axis=1) < 1e-9).sum())

    trajs, changed, total, sq_int, sq_real, n_frames = [], 0, 0, 0.0, 0.0, 0
    per_dof_changed = np.zeros(N_DOF)
    for tr, c in zip(full.trajectories, coarse):
        a, _, b = c.partition("->")
        o = inter[li[a], ri[b]].astype(np.float32)
        x = normalize(tr) + o
        q = denormalize(x)
        qc = np.clip(q, LIMITS_LO, LIMITS_HI).astype(np.float32)
        ch = ~np.isclose(q, qc, rtol=0, atol=1e-6)
        changed += int(ch.sum())
        per_dof_changed += ch.sum(axis=0)
        total += ch.size
        realised = normalize(qc) - normalize(tr)
        sq_int += float((o.astype(np.float64) ** 2).sum()) * len(tr)
        sq_real += float((realised.astype(np.float64) ** 2).sum())
        n_frames += len(tr)
        trajs.append(qc)

    meta = dict(full.meta)
    meta.update(
        source="grab_planted", derived_from="data/bundles/grab.npz",
        planted_axis=f"granularity={MODE} (shape x fine intent)",
        planted_seed=0, planted_rms_norm=float(target_rms),
        planted_rule="per-cell 27-D N(0,1) offset, double-centred over occupied cells "
                     "to convergence, scaled to RMS, added in normalised space, clamped",
    )
    out = TrajectoryBundle(trajectories=trajs, fps=full.fps, labels=list(full.labels), meta=meta)
    out.save(B / "grab_planted.npz")
    back = TrajectoryBundle.load(B / "grab_planted.npz")
    assert back.aux is None and list(back.labels) == list(full.labels)

    from caredex.hand_model import DOF_NAMES
    top = sorted(zip(per_dof_changed / (total / N_DOF), DOF_NAMES), reverse=True)[:6]
    res = dict(
        n_left=len(lefts), n_right=len(rights), n_cells=int(mask.sum()),
        one_pass_max_row_mean=resid_row_one, one_pass_max_col_mean=resid_col_one,
        centring_iterations=iters, cells_with_zero_offset=zero_cells,
        target_rms_norm=target_rms,
        frame_rms_before_clamp=float(np.sqrt(sq_int / (n_frames * N_DOF))),
        frame_rms_after_clamp=float(np.sqrt(sq_real / (n_frames * N_DOF))),
        clamp_fraction=changed / total,
        clamp_fraction_top_dofs={n: float(f) for f, n in top},
    )
    print(f"[B] {res['n_cells']} occupied cells ({len(lefts)} shapes x {len(rights)} intents); "
          f"one-pass centring leaves max |row mean| {resid_row_one:.3f}, |col mean| "
          f"{resid_col_one:.3f}; iterated {iters} passes; {zero_cells} cells forced to zero")
    print(f"[B] target RMS {target_rms:.4f} norm units; frame RMS before clamp "
          f"{res['frame_rms_before_clamp']:.4f}, after {res['frame_rms_after_clamp']:.4f}")
    print(f"[B] clamp changed {res['clamp_fraction']:.4%} of values; worst DOFs {top}")
    return res


def main() -> int:
    report: dict = {}
    report["grasp"] = build_grasp()

    via = synthetic_via_magnitude()
    report["synthetic_via"] = via
    print("[B] synth_hard via-point (reconstructed from its generator RNG):")
    for k, v in via.items():
        print(f"      {k:<28} {v}")
    target = via["frame_rms_norm_27dof"]
    report["planted"] = build_planted(target)

    cfg = json.loads((ROOT / "runs" / "axis_screen.json").read_text(encoding="utf-8"))["config"]
    ref = next(r for r in json.loads((ROOT / "runs" / "axis_screen.json").read_text(
        encoding="utf-8"))["rows"] if r["dataset"] == "GRAB" and r["mode"] == MODE)
    rows = {}
    for name in ("grab", "grab_grasp", "grab_planted"):
        r = screen(B / f"{name}.npz", MODE, cfg["window"], cfg["stride"],
                   cfg["max_per_traj"], cfg["n_perm"], cfg["seed"])
        rows[name] = r
        print(f"[C] {name:<13} traj {r['n_traj']:>5} cells {r['n_cells']:>3} "
              f"additive {r['additive_r2']:.4f} inter {r['interaction_r2']:.4f} "
              f"excess {r['excess']:+.4f} z {r['z']:.2f}")
    assert abs(rows["grab"]["excess"] - ref["excess"]) < 5e-4, (rows["grab"]["excess"], ref["excess"])
    assert abs(rows["grab"]["z"] - ref["z"]) < 0.2, (rows["grab"]["z"], ref["z"])
    print(f"[C] grab.npz reproduces runs/axis_screen.json "
          f"(excess {ref['excess']:+.4f}, z {ref['z']:.2f})")

    out = ROOT / "runs" / "axis_screen_grab_diagnostics.json"
    out.write_text(json.dumps({
        "config": {**cfg, "mode": MODE, "pad": PAD, "min_len": MIN_LEN},
        "reference": ref, "rows": rows, "build": report,
        "caveat": "Screen is linear variance decomposition, not the penalty. A constant "
                  "per-cell plant is what a cell one-hot captures, so grab_planted rising "
                  "validates the plant, not the instrument.",
    }, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
