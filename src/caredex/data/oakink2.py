"""OakInk2 adapter: the first dataset here with author-annotated composition.

Why this dataset replaces the others for compositional work
-----------------------------------------------------------
Every earlier compositional axis in this project was chosen from whatever
metadata happened to exist, and each turned out too thin or too easy:

* DexYCB, ``subject x object-shape``: 200 cells over 499 trajectories (2.5
  each), and the measured paired penalty was **+0.00006** -- no compositional
  difficulty at all, because every clip is one grasp of one object.
* OakInk-Image, ``object x intent``: 345 cells over 770 trajectories (2.23
  each), too thin to split into a target and an informed set. The coarser
  "category" axis used instead was the *first character of an object id*, a
  grouping invented here and traceable to no OakInk documentation.

OakInk2 annotates **named primitives with explicit frame ranges**, per hand,
authored by the dataset's creators. A survey of the 627 sequences found 67
primitives, 2,185 segments, 391 distinct ordered transitions out of 4,489
possible (8.7% density), and 66 transitions occurring in at least 6 sequences.
The transitions carry real sequential dependency -- ``uncap_alcohol_lamp ->
ignite_alcohol_lamp``, ``open_gate -> take_outside`` -- and the vocabulary
contains inverse pairs (``screw``/``unscrew``, ``remove_lid``/``put_on_lid``),
which is what compositional structure looks like when it is genuine rather than
constructed.

Layout
------
    anno_preview/<seq>.pkl        raw_mano[frame] -> {rh__pose_coeffs (1,16,4)
                                  quaternions, rh__tsl, rh__betas, and lh__*}
    program/program/program_info/<seq>.json
                                  "(lh_span, rh_span)" -> {primitive, obj_list,
                                  interaction_mode, primitive_lh, primitive_rh}
    program/program/task_target.json
                                  sequence -> the complex task in English

Hand pose is in the same quaternion form as OakInk-Image, so the convention
inference in :mod:`caredex.data.mano_retarget` applies unchanged -- the layout,
flexion axis and flexion sign are inferred from the data here too, never
assumed. The reading used here was checked against manotorch's source and the
dataset maintainer's own answer: ``pose_coeffs`` is ``(16, 4)`` real-part-first
quaternions, parent-relative for joints 1-15 and global for joint 0, consumed
by ``ManoLayer(rot_mode="quat", use_pca=False)`` without reinterpretation.

Rotation fidelity is lower here than in OakInk-Image, and this is a property of
the dataset, not of the conversion
--------------------------------------------------------------------------
The same conversion code yields a flexion-axis dominance of 0.66 and a 5.9
degree projection residual on OakInk-Image, but 0.455 and 23.7 degrees here:
the PIP joints do not behave like hinges under any reading. The cause is
documented in the OakInk2 paper (Sec. 3.2.2): the subject's body is fitted as
SMPL-X from mocap markers, and "other body representations like MANO are
derived from this result". The hand parameters are therefore a byproduct of a
body-level marker optimisation rather than an independent hand fit, and the
derivation code is confirmed unreleased by the maintainer.

The paper's cross-dataset validation (PA-MPJPE ~11-12 mm) does **not** rebut
this: keypoint position error is insensitive to twist error at a joint and only
weakly sensitive to flexion/abduction share being swapped between DOF, so good
keypoints are fully compatible with a poor rotation decomposition.

Consequences to respect:

* the paired compositional comparison remains valid -- both models see the same
  data under the same split, so representation noise is symmetric;
* absolute reconstruction errors from this source must **not** be tabulated
  beside OakInk-Image's;
* any anatomical claim (coupling residuals, joint-limit statistics) is weaker
  here than there, and should be reported from OakInk-Image instead.

Two things this makes possible that no earlier dataset did
----------------------------------------------------------
1. **Held-out transitions are the dataset's own.** ``label`` is the primitive
   sequence (``grip->rearrange->take_outside``), which the existing
   compositional split consumes directly.
2. **Segmentation can be scored against ground truth.** Earlier the only check
   on a model's learned assignments was whether switches coincided with fast
   motion. Here the true primitive boundaries are known.
"""

from __future__ import annotations

import ast
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from caredex.data.base import TrajectoryBundle, TrajectorySource, register_source
from caredex.data.mano_retarget import (
    RetargetConventions,
    center_wrist_translation,
    infer_quat_layout,
    limit_clip_fraction,
    matrix_to_euler_xyz,
    quat_to_matrix,
)
from caredex.hand_model import apply_dip_pip_coupling, clamp_to_limits

HF_REPO = "kelvin34501/OakInk-v2"

_HELP = f"""\
OakInk2 not found.

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="{HF_REPO}", repo_type="dataset",
        local_dir=r"D:\\datasets\\oakink2",
        allow_patterns=["anno_preview/**", "program.tar", "program_extension.tar"],
    )
    tar -xf program.tar -C program     # inside the local_dir

That is ~34 GB of annotations with no imagery (the full multi-view release is
~2 TB). Licence CC-BY-SA-4.0.

Then set:  data.source=oakink2  data.root=D:\\datasets\\oakink2
"""


def _load_frames_worker(job: tuple[str, str, list[int]]) -> tuple[str, np.ndarray, np.ndarray, list[int]] | None:
    """Load one sequence's hand pose. Module-level so it can be sent to a pool.

    The pickle is the cost: each ``anno_preview`` file is ~55 MB holding
    per-frame camera intrinsics for four views, per-frame poses for every object
    in the scene, and a full SMPL-X body track. ``raw_mano`` cannot be reached
    without deserialising all of it, which is millions of small arrays per file
    and takes tens of seconds. Subsampling frames does not help, because the
    cost is paid before any frame is selected -- the only lever is doing several
    files at once, since the work is CPU-bound and independent per file.
    """
    path_str, hand, wanted = job
    try:
        with open(path_str, "rb") as fh:
            rec = pickle.load(fh)
    except Exception:  # noqa: BLE001 - a corrupt file should not kill the build
        return None

    mano = rec.get("raw_mano")
    if not isinstance(mano, dict) or not mano:
        return None

    frames = [f for f in wanted if f in mano]
    if not frames:
        return None

    pose_key, tsl_key = f"{hand}__pose_coeffs", f"{hand}__tsl"
    quats = np.empty((len(frames), 16, 4), dtype=np.float32)
    tsl = np.zeros((len(frames), 3), dtype=np.float32)
    for i, fid in enumerate(frames):
        entry = mano[fid]
        if pose_key not in entry:
            return None
        quats[i] = _to_numpy(entry[pose_key]).reshape(16, 4)
        if tsl_key in entry:
            tsl[i] = _to_numpy(entry[tsl_key]).reshape(3)
    return Path(path_str).stem, quats, tsl, frames


def parse_span_key(key: str) -> tuple[tuple | None, tuple | None]:
    """``"(None, (966, 5954))"`` -> (left-hand span, right-hand span)."""
    try:
        val = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return None, None
    if isinstance(val, tuple) and len(val) == 2:
        return val[0], val[1]
    return None, None


@register_source("oakink2")
class OakInk2Source(TrajectorySource):
    """Load OakInk2 primitive segments as labelled hand trajectories.

    Parameters
    ----------
    root:
        Directory holding ``anno_preview/`` and the extracted ``program/``.
    hand:
        ``"rh"`` or ``"lh"``. One hand at a time: the 27-DOF model describes a
        single hand, and mixing chiralities would put two anatomies in one
        action space.
    unit:
        ``"sequence"`` yields one trajectory per recording, labelled with its
        whole primitive chain (``a->b->c``) -- this is what the compositional
        split consumes. ``"segment"`` yields one trajectory per primitive
        instead, for training a prior on clean single-primitive motion.
    stride:
        Mocap runs far faster than the 30 fps of the earlier datasets; segments
        have a median length of 725 frames. Subsampling keeps window lengths
        comparable across sources.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        hand: str = "rh",
        unit: str = "sequence",
        stride: int = 4,
        fps: float = 30.0,
        min_frames: int = 40,
        max_sequences: int | None = None,
        enforce_coupling: bool = False,
        n_workers: int = 6,
        verbose: bool = True,
    ) -> None:
        #: Parallel pickle loads. Each worker expands a 55 MB file into a
        #: gigabyte or two of small arrays, so this trades RAM for wall clock;
        #: 6 is chosen against ~9 GB free, not against the core count.
        self.n_workers = max(1, n_workers)
        if hand not in ("rh", "lh"):
            raise ValueError(f"hand must be rh or lh, got {hand!r}")
        if unit not in ("sequence", "segment"):
            raise ValueError(f"unit must be sequence or segment, got {unit!r}")
        self.root = Path(root) if root else None
        self.hand = hand
        self.unit = unit
        self.stride = stride
        self.fps = fps
        self.min_frames = min_frames
        self.max_sequences = max_sequences
        self.enforce_coupling = enforce_coupling
        self.verbose = verbose

    # -- entry point --------------------------------------------------------

    def load(self, **kwargs: Any) -> TrajectoryBundle:
        root, prog_dir = self._resolve()
        anno_dir = root / "anno_preview"
        program_dir = prog_dir / "program_info"
        targets = self._task_targets(prog_dir)

        stems = sorted(p.stem for p in anno_dir.glob("*.pkl"))
        if self.max_sequences is not None:
            stems = stems[: self.max_sequences]

        # Segment annotations are small JSON reads, so they happen up front in
        # the parent; only the expensive pickle loads are farmed out.
        jobs: list[tuple[str, str, list[int]]] = []
        seg_of: dict[str, list[tuple[int, int, str]]] = {}
        skipped_no_program = 0
        for stem in stems:
            pj = program_dir / f"{stem}.json"
            if not pj.exists():
                skipped_no_program += 1
                continue
            segs = self._segments_for(pj)
            if not segs:
                continue
            wanted = sorted({
                f for a, b, _ in segs
                for f in range(int(a), max(int(b), int(a) + 1), self.stride)
            })
            seg_of[stem] = segs
            jobs.append((str(anno_dir / f"{stem}.pkl"), self.hand, wanted))

        if self.verbose:
            print(f"[oakink2] {len(jobs)} sequences to read, {self.n_workers} workers", flush=True)

        raw: list[dict] = []
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            for result in pool.map(_load_frames_worker, jobs, chunksize=1):
                done += 1
                if result is None:
                    continue
                stem, quats, tsl, frames = result
                raw.extend(self._cut(stem, seg_of[stem], quats, tsl, frames, targets))
                if self.verbose and (done % 25 == 0 or done == len(jobs)):
                    rate = done / max(time.time() - t0, 1e-9)
                    eta = (len(jobs) - done) / max(rate, 1e-9)
                    print(f"[oakink2]   {done}/{len(jobs)}  {len(raw)} trajectories  "
                          f"{rate * 60:.1f}/min  eta {eta / 60:.1f} min", flush=True)

        if not raw:
            raise RuntimeError(f"no usable trajectories under {root}")

        # One convention inference over the pooled data, so every sequence is
        # converted the same way. Inferring per sequence could silently mix two.
        pooled = np.concatenate([r["quats"] for r in raw], axis=0)
        w_first, quat_stats = infer_quat_layout(pooled)
        euler_pool = matrix_to_euler_xyz(quat_to_matrix(pooled, w_first))
        conv = RetargetConventions.infer(euler_pool)

        if self.verbose:
            print(conv.describe("oakink2"), flush=True)

        trajectories: list[np.ndarray] = []
        raw_q: list[np.ndarray] = []
        labels: list[str] = []
        meta_rows: list[dict] = []
        residuals: list[dict] = []
        for r in raw:
            euler = matrix_to_euler_xyz(quat_to_matrix(r["quats"], w_first))
            q, res = conv.apply(euler, r["tsl"])
            if self.enforce_coupling:
                q = apply_dip_pip_coupling(q)
            q = center_wrist_translation(q)
            raw_q.append(q)
            trajectories.append(clamp_to_limits(q).astype(np.float32))
            labels.append(r["label"])
            residuals.append(res)
            meta_rows.append({k: r[k] for k in ("sequence", "task", "primitives", "spans")})

        clipped = limit_clip_fraction(np.concatenate(raw_q, axis=0))
        residual = {k: float(np.mean([x[k] for x in residuals])) for k in ("mean", "p95", "max")}

        if self.verbose:
            n_multi = sum(1 for lab in labels if "->" in lab)
            print(
                f"[oakink2] {len(trajectories)} trajectories "
                f"({sum(len(t) for t in trajectories):,} frames, stride {self.stride}); "
                f"{n_multi} carry a transition; projection residual "
                f"{residual['mean']:.2f} deg mean / {residual['p95']:.2f} p95; "
                f"{clipped:.2%} of DOF values hit a limit"
            )
            if skipped_no_program:
                print(f"[oakink2] {skipped_no_program} sequences had no program annotation")

        return TrajectoryBundle(
            trajectories=trajectories,
            fps=self.fps / max(self.stride, 1),
            labels=labels,
            meta={
                "source": "oakink2",
                "hf_repo": HF_REPO,
                "hand": self.hand,
                "unit": self.unit,
                "stride": self.stride,
                "quat_w_first": bool(w_first),
                "enforce_coupling": self.enforce_coupling,
                "projection_residual_deg": residual,
                "limit_clip_fraction": clipped,
                "label_format": "primitive->primitive->...",
                # Ground-truth segment boundaries, so a model's learned
                # assignments can be scored against them rather than against a
                # proxy such as "does the switch land on fast motion".
                "segments": meta_rows,
                **quat_stats,
                **conv.stats,
            },
        )

    # -- internals ----------------------------------------------------------

    def _resolve(self) -> tuple[Path, Path]:
        if self.root is None or not self.root.exists():
            raise FileNotFoundError(_HELP)
        root = self.root
        for candidate in (root / "program" / "program", root / "program"):
            if (candidate / "program_info").exists():
                return root, candidate
        found = next((p.parent for p in root.rglob("task_target.json")), None)
        if found is None:
            raise FileNotFoundError(_HELP)
        return root, found

    @staticmethod
    def _task_targets(prog_dir: Path) -> dict[str, str]:
        f = prog_dir / "task_target.json"
        if not f.exists():
            return {}
        data = json.loads(f.read_text(encoding="utf-8"))
        # Keys use '/' where filenames use '++'.
        return {k.replace("/", "++"): v for k, v in data.items()}

    def _segments_for(self, path: Path) -> list[tuple[int, int, str]]:
        """Ordered ``(start, end, primitive)`` for the chosen hand."""
        data = json.loads(path.read_text(encoding="utf-8"))
        field = f"primitive_{self.hand}"
        idx = 1 if self.hand == "rh" else 0
        out = []
        for key, val in data.items():
            prim = val.get(field)
            if not prim:
                continue
            spans = parse_span_key(key)
            span = spans[idx] or spans[1] or spans[0]
            if not span:
                continue
            out.append((int(span[0]), int(span[1]), str(prim)))
        out.sort()
        return out

    def _read_hand(
        self, path: Path, wanted: list[int] | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
        """Read one hand's quaternions and translations for ``wanted`` frames.

        Converting every frame and selecting afterwards was the original
        implementation and it did four to ten times more work than needed: the
        adapter keeps only frames inside annotated segments, subsampled by
        ``stride``. On 627 sequences of ~10,400 frames that difference was the
        gap between a build that finishes and one that ran for three hours
        without completing.
        """
        with path.open("rb") as fh:
            rec = pickle.load(fh)
        mano = rec.get("raw_mano")
        if not isinstance(mano, dict) or not mano:
            return None, None, []

        frames = sorted(mano) if wanted is None else [f for f in wanted if f in mano]
        if not frames:
            return None, None, []

        pose_key, tsl_key = f"{self.hand}__pose_coeffs", f"{self.hand}__tsl"
        quats = np.empty((len(frames), 16, 4), dtype=np.float32)
        tsl = np.zeros((len(frames), 3), dtype=np.float32)
        for i, fid in enumerate(frames):
            entry = mano[fid]
            quats[i] = _to_numpy(entry[pose_key]).reshape(16, 4)
            if tsl_key in entry:
                tsl[i] = _to_numpy(entry[tsl_key]).reshape(3)
        return quats, tsl, frames

    def _cut(
        self,
        stem: str,
        segs: list[tuple[int, int, str]],
        quats: np.ndarray,
        tsl: np.ndarray,
        frames: list[int],
        targets: dict[str, str],
    ) -> list[dict]:
        # `frames` holds the mocap ids actually read, already subsampled, so a
        # span maps to a contiguous slice of the arrays rather than to a strided
        # range over the full recording.
        idx_of = {fid: i for i, fid in enumerate(frames)}
        task = targets.get(stem, "")

        def slice_of(a: int, b: int) -> np.ndarray:
            want = range(int(a), max(int(b), int(a) + 1), self.stride)
            return np.array([idx_of[f] for f in want if f in idx_of], dtype=np.int64)

        if self.unit == "segment":
            out = []
            for a, b, prim in segs:
                sel = slice_of(a, b)
                if len(sel) < self.min_frames:
                    continue
                out.append({
                    "quats": quats[sel], "tsl": tsl[sel], "label": prim,
                    "sequence": stem, "task": task, "primitives": [prim],
                    "spans": [(a, b)],
                })
            return out

        # One trajectory per recording, covering its annotated segments only --
        # the gaps between them are unlabelled and would pollute the primitive
        # chain the label claims.
        sel = np.concatenate([slice_of(a, b) for a, b, _ in segs]) if segs else np.empty(0, int)
        if len(sel) < self.min_frames:
            return []
        prims = [p for _, _, p in segs]
        return [{
            "quats": quats[sel], "tsl": tsl[sel],
            "label": "->".join(prims),
            "sequence": stem, "task": task, "primitives": prims,
            "spans": [(a, b) for a, b, _ in segs],
        }]


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
