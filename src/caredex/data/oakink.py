"""OakInk adapter: MANO quaternions -> the 27-DOF anatomical hand model.

Reads directly from ``anno_v2.1.zip``. The archive holds ~1.57M small files;
extracting it on Windows takes hours and buys nothing, since zip members are
randomly accessible.

What the archive contains
-------------------------
    anno/general_info/<stem>.pkl   dict with hand_anno.{hand_pose, hand_tsl, hand_shape}
    anno/hand_j/<stem>.pkl         (21, 3) 3D joint positions
    anno/hand_v/<stem>.pkl         (778, 3) hand mesh vertices
    anno/obj_transf/<stem>.pkl     (4, 4) object pose -- not used, this is the
                                   "drop the object point cloud" step of Phase 1

``hand_pose`` is ``(16, 4)``: one quaternion per MANO joint, expressed relative
to its parent. 16 = wrist + 15 finger joints. Filenames are
``<seq>__<timestamp>__<f2>__<frame>__<view>``; the four camera views record the
same hand configuration, so only one view is kept.

The retargeting problem
-----------------------
MANO gives every joint a full 3-DOF rotation. The anatomical model gives a PIP
or DIP one flexion DOF. Projecting the former onto the latter therefore has a
residual, and that residual has to be reported rather than hidden -- see
:meth:`OakInkSource.load`, which returns it in ``bundle.meta``.

Two conventions could be guessed wrong, so neither is guessed:

* **Quaternion order.** Determined empirically by :func:`infer_quat_layout`:
  under the correct reading, distal joints are small rotations. Reading
  ``(w,x,y,z)`` as ``(x,y,z,w)`` turns a 13-degree DIP flexion into 129
  degrees, which the check detects.
* **Which Euler axis is flexion.** Determined empirically by
  :func:`infer_flexion_axis`: PIP joints are anatomically near-1-DOF, so the
  axis carrying their variance is the flexion axis. Assuming it instead would
  manufacture abduction that is not in the data.

Both inferences are logged. If they disagree with the ergonomic validator's
DIP/PIP coupling check, the conversion is wrong -- do not trust the numbers.
"""

from __future__ import annotations

import io
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from caredex.data.base import TrajectoryBundle, TrajectorySource, register_source
from caredex.data.mano_retarget import (
    DISTAL_JOINTS_16,
    MANO_JOINT_MAP,
    MANO_PARENTS,
    PIP_JOINTS_16,
    RetargetConventions,
    infer_quat_layout,
    limit_clip_fraction,
    matrix_to_euler_xyz,
    quat_angle,
    quat_to_matrix,
)
from caredex.hand_model import apply_dip_pip_coupling, clamp_to_limits

HUGGINGFACE_URL = "https://huggingface.co/datasets/oakink/OakInk-v1"
REQUEST_FORM_URL = "https://forms.gle/g6QEmmCeZYLGaVe29"

_HELP = f"""\
OakInk annotations not found.

  1. Bulk archives (public):  {HUGGINGFACE_URL}
  2. anno_v2.1.zip by request:  {REQUEST_FORM_URL}
     Requires an ACADEMIC email -- gmail/yahoo/hotmail/163 are rejected.

Then point the config at the zip itself:
  data.source=oakink  data.root=<path to anno_v2.1.zip>
"""

_STEM_RE = re.compile(r"^(?P<seq>[^_].*?)__(?P<ts>[\d\-]+)__(?P<f2>\d+)__(?P<frame>\d+)__(?P<view>\d+)$")


def composition_label(sequence_id: str) -> str:
    """Turn an OakInk sequence id into an ``object->intent`` composition label.

    OakInk sequence ids look like ``A15015_0004_0008_0007``: object id, intent
    id, then subject id(s). OakInk-Image records several intent-oriented
    interactions (use, hold, lift up, hand over, receive) per object, so the
    ``(object, intent)`` pair is a real compositional axis in the data —
    unlike the synthetic bundle's primitive sequences, nobody designed it to
    make this experiment work.

    That matters: a compositional-generalisation result on synthetic data only
    shows the model recovers structure this repo planted. Held-out
    ``(object, intent)`` pairs are structure the OakInk authors created for
    unrelated reasons, so the same experiment there is evidence.
    """
    head = sequence_id.split("__", 1)[0]
    fields = head.split("_")
    obj = fields[0] if fields else head
    intent = fields[1] if len(fields) > 1 else "unknown"
    return f"{obj}->{intent}"


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


@register_source("oakink")
class OakInkSource(TrajectorySource):
    """Load OakInk-Image hand trajectories from ``anno_v2.1.zip``.

    Parameters
    ----------
    root:
        Path to ``anno_v2.1.zip`` (or a directory containing it).
    view:
        Which camera view to keep. All views hold the same hand configuration,
        so keeping more than one would duplicate poses and inflate the dataset.
    max_sequences:
        Cap for quick iteration; ``None`` loads all 792.
    fps:
        OakInk-Image was captured at 30 fps.
    enforce_coupling:
        Overwrite DIP with ``2/3 * PIP``. Off by default -- for real data the
        coupling residual is a *measurement*, and forcing it would destroy the
        one biomechanical check that has teeth.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        view: int = 0,
        max_sequences: int | None = None,
        fps: float = 30.0,
        enforce_coupling: bool = False,
        min_frames: int = 40,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root) if root else None
        self.view = view
        self.max_sequences = max_sequences
        self.fps = fps
        self.enforce_coupling = enforce_coupling
        self.min_frames = min_frames
        self.verbose = verbose

    # -- entry point --------------------------------------------------------

    def load(self, **kwargs: Any) -> TrajectoryBundle:
        archive = self._resolve_archive()
        with zipfile.ZipFile(archive) as zf:
            members = self._select_members(zf)
            if not members:
                raise RuntimeError(
                    f"no general_info entries for view {self.view} in {archive}"
                )
            poses, translations, seq_of_frame = self._read_frames(zf, members)

        w_first, quat_stats = infer_quat_layout(poses)
        matrices = quat_to_matrix(poses, w_first)
        euler = matrix_to_euler_xyz(matrices)
        conv = RetargetConventions.infer(euler)

        if self.verbose:
            print(conv.describe("oakink"), flush=True)

        q, residual = conv.apply(euler, translations)

        trajectories: list[np.ndarray] = []
        labels: list[str] = []
        seq_ids: list[str] = []
        for seq, idx in _group_runs(seq_of_frame):
            if len(idx) < self.min_frames:
                continue
            traj = q[idx]
            if self.enforce_coupling:
                traj = apply_dip_pip_coupling(traj)
            trajectories.append(clamp_to_limits(traj).astype(np.float32))
            labels.append(composition_label(seq))
            seq_ids.append(seq)

        if not trajectories:
            raise RuntimeError(
                f"every sequence was shorter than min_frames={self.min_frames}"
            )

        clipped = limit_clip_fraction(q)
        if self.verbose:
            print(
                f"[oakink] {len(trajectories)} sequences, "
                f"{sum(len(t) for t in trajectories):,} frames; "
                f"projection residual {residual['mean']:.2f} deg mean / "
                f"{residual['p95']:.2f} p95; {clipped:.2%} of DOF values hit a limit"
            )

        return TrajectoryBundle(
            trajectories=trajectories,
            fps=self.fps,
            labels=labels,
            meta={
                "source": "oakink",
                "archive": str(archive),
                "view": self.view,
                "quat_w_first": bool(w_first),
                "enforce_coupling": self.enforce_coupling,
                # Everything below is a caveat, not a result. The projection
                # residual is how much 3-DOF MANO rotation the 1-DOF anatomical
                # model could not represent.
                "projection_residual_deg": residual,
                "limit_clip_fraction": clipped,
                "sequence_ids": seq_ids,
                "label_format": "object->intent",
                **quat_stats,
                **conv.stats,
            },
        )

    # -- internals ----------------------------------------------------------

    def _resolve_archive(self) -> Path:
        if self.root is None:
            raise FileNotFoundError(_HELP)
        p = self.root
        if p.is_dir():
            p = p / "anno_v2.1.zip"
        if not p.exists():
            raise FileNotFoundError(_HELP)
        return p

    def _select_members(self, zf: zipfile.ZipFile) -> list[tuple[str, str, int]]:
        """Return ``(member, sequence_id, frame_index)`` for the chosen view."""
        rows: list[tuple[str, str, int]] = []
        for name in zf.namelist():
            if "/general_info/" not in name or not name.endswith(".pkl"):
                continue
            stem = name.rsplit("/", 1)[1][: -len(".pkl")]
            m = _STEM_RE.match(stem)
            if not m or int(m["view"]) != self.view:
                continue
            rows.append((name, f"{m['seq']}__{m['ts']}", int(m["frame"])))

        if self.max_sequences is not None:
            keep = sorted({s for _, s, _ in rows})[: self.max_sequences]
            wanted = set(keep)
            rows = [r for r in rows if r[1] in wanted]

        # Sort by sequence then frame so that grouping yields ordered trajectories.
        rows.sort(key=lambda r: (r[1], r[2]))
        return rows

    def _read_frames(
        self, zf: zipfile.ZipFile, members: list[tuple[str, str, int]]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        poses = np.empty((len(members), 16, 4), dtype=np.float32)
        tsl = np.zeros((len(members), 3), dtype=np.float32)
        seq_of_frame: list[str] = []

        for i, (name, seq, _frame) in enumerate(members):
            with zf.open(name) as fh:
                info = pickle.load(io.BytesIO(fh.read()))
            hand = info["hand_anno"]
            poses[i] = _as_numpy(hand["hand_pose"]).reshape(16, 4)
            if "hand_tsl" in hand:
                tsl[i] = _as_numpy(hand["hand_tsl"]).reshape(3)
            seq_of_frame.append(seq)

            if self.verbose and i and i % 20000 == 0:
                print(f"[oakink]   read {i:,}/{len(members):,} frames")

        return poses, tsl, seq_of_frame

def _as_numpy(x: Any) -> np.ndarray:
    """Accept torch tensors or numpy arrays without importing torch eagerly."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _group_runs(seq_of_frame: list[str]) -> list[tuple[str, np.ndarray]]:
    """Contiguous index runs per sequence (members were pre-sorted)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(seq_of_frame):
        groups[s].append(i)
    return [(s, np.array(idx, dtype=np.int64)) for s, idx in groups.items()]
