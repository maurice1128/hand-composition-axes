"""DexYCB adapter, via the annotation-only VITRA mirror.

Provenance — state this in any paper that uses it
--------------------------------------------------
The official DexYCB release is hosted on Google Drive and returns a download
quota error ("Too many users have viewed or downloaded this file recently"),
and its per-subject archives are ~12 GB of which roughly 99% is RGB-D imagery
this project discards. There is no annotation-only official download.

This adapter therefore reads **MIT-Media-Lab/dexycb-vitra-streaming-v2** on
Hugging Face (CC-BY-NC-4.0), a third-party conversion describing itself as an
"audited VITRA Stage-1 conversion of all 1,000 physical DexYCB sequences" built
on an "audited HandNeRF-derived world shard". It is ~1 GB of pose annotations
with no imagery.

**It is not the official release.** The poses may have been re-anchored or
filtered relative to NVIDIA's original. Cite the mirror, not DexYCB alone, and
do not present numbers from it as reproducing the official benchmark.

What each ``.npy`` holds
------------------------
One dict per (sequence, camera), with a ``left`` and a ``right`` entry::

    hand_pose                 (T, 15, 3, 3)  MANO joint rotations, parent-relative
    global_orient_worldspace  (T, 3, 3)      wrist orientation
    transl_worldspace         (T, 3)         wrist translation (MANO joint 0)
    joints_worldspace         (T, 21, 3)     3D joint positions
    beta                      (10,)          shape parameters
    annotation_valid          (T,)           per-frame validity
    visible / task_active     (T,)           further per-frame masks

Rotation *matrices* rather than quaternions, so unlike OakInk there is no
layout to infer -- but the flexion axis and sign are still inferred rather than
assumed, by the same code in :mod:`caredex.data.mano_retarget`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from caredex.data.base import TrajectoryBundle, TrajectorySource, register_source
from caredex.data.mano_retarget import (
    RetargetConventions,
    center_wrist_translation,
    limit_clip_fraction,
    matrix_to_euler_xyz,
)
from caredex.hand_model import apply_dip_pip_coupling, clamp_to_limits

MIRROR_REPO = "MIT-Media-Lab/dexycb-vitra-streaming-v2"
OFFICIAL_URL = "https://dex-ycb.github.io/"

_HELP = f"""\
DexYCB annotations not found.

The official release ({OFFICIAL_URL}) is on Google Drive and currently returns a
download quota error; its subject archives are also ~12 GB each of mostly imagery.

Download the annotation-only mirror instead (~1 GB, no images):

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="{MIRROR_REPO}",
        repo_type="dataset",
        local_dir=r"E:\\datasets\\dexycb_vitra",
        allow_patterns=["Annotation/**"],
    )

Then set:  data.source=dexycb  data.root=E:\\datasets\\dexycb_vitra
"""

#: DexYCB_<date>-subject-NN_<seq timestamp>_<camera serial>
_STEM_RE = re.compile(
    r"^DexYCB_(?P<date>[\d\-]+)-subject-(?P<subject>\d+)_(?P<seq>[\d\-]+)_(?P<camera>\d+)$"
)

#: YCB object -> grasp-relevant shape class.
#:
#: With 10 subjects and 20 objects the ``subject->object`` grid has 200 cells
#: over ~500 trajectories: 2.5 each, too thin for a paired compositional split
#: (the same limitation OakInk has at ``object->intent``). Grouping by shape
#: gives ~6 classes and a workable density.
#:
#: The grouping is by what the hand has to *do*, not by semantics: a soup can
#: and a tuna can afford the same cylindrical grasp, while a mug and a bowl
#: both afford a rim grasp. A can and a marker are both "cylinders"
#: geometrically but demand different apertures, so they are separated.
YCB_SHAPE_CLASS: dict[str, str] = {
    "master_chef_can": "can", "tomato_soup_can": "can",
    "tuna_fish_can": "can", "potted_meat_can": "can",
    "cracker_box": "box", "sugar_box": "box", "pudding_box": "box",
    "gelatin_box": "box", "wood_block": "box", "foam_brick": "box",
    "mustard_bottle": "bottle", "bleach_cleanser": "bottle", "pitcher_base": "bottle",
    "bowl": "opencontainer", "mug": "opencontainer",
    "power_drill": "tool", "scissors": "tool",
    "large_clamp": "tool", "extra_large_clamp": "tool",
    "banana": "elongated", "large_marker": "elongated",
}


def shape_class(object_id: str) -> str:
    """``002_master_chef_can`` -> ``can``; unknown ids fall back to the id."""
    name = object_id.split("_", 1)[1] if "_" in object_id else object_id
    return YCB_SHAPE_CLASS.get(name, name)


@register_source("dexycb")
class DexYcbSource(TrajectorySource):
    """Load DexYCB hand trajectories from the annotation-only mirror.

    Parameters
    ----------
    root:
        Directory containing ``Annotation/``.
    hand:
        ``"right"``, ``"left"`` or ``"both"``. Left-hand sequences are mirrored
        into right-hand convention when ``"both"`` is used, because the 27-DOF
        model describes one hand and mixing chiralities would put two different
        anatomies in the same action space.
    one_camera_per_sequence:
        DexYCB records each sequence from 8 cameras. The hand configuration is
        identical across them, so keeping all 8 would octuple the dataset with
        duplicates and make any split leak.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        hand: str = "right",
        one_camera_per_sequence: bool = True,
        min_frames: int = 40,
        max_sequences: int | None = None,
        fps: float = 30.0,
        enforce_coupling: bool = False,
        label_granularity: str = "object",
        verbose: bool = True,
    ) -> None:
        if hand not in ("right", "left", "both"):
            raise ValueError(f"hand must be right/left/both, got {hand!r}")
        if label_granularity not in ("object", "shape"):
            raise ValueError(f"label_granularity must be object/shape, got {label_granularity!r}")
        self.label_granularity = label_granularity
        self.root = Path(root) if root else None
        self.hand = hand
        self.one_camera_per_sequence = one_camera_per_sequence
        self.min_frames = min_frames
        self.max_sequences = max_sequences
        self.fps = fps
        self.enforce_coupling = enforce_coupling
        self.verbose = verbose

    # -- entry point --------------------------------------------------------

    def load(self, **kwargs: Any) -> TrajectoryBundle:
        files = self._select_files()
        segments, labels, stats = self._read_segments(files)
        if not segments:
            raise RuntimeError(
                f"no valid segments of >= {self.min_frames} frames found under {self.root}"
            )

        # Inference runs on the pooled data so every sequence is converted under
        # one convention -- inferring per sequence could silently mix two.
        pooled = np.concatenate([s["euler"] for s in segments], axis=0)
        conv = RetargetConventions.infer(pooled)

        if self.verbose:
            print(conv.describe("dexycb"), flush=True)

        trajectories: list[np.ndarray] = []
        raw_q: list[np.ndarray] = []
        residuals: list[dict[str, float]] = []
        for seg in segments:
            q, res = conv.apply(seg["euler"], seg["transl"])
            if self.enforce_coupling:
                q = apply_dip_pip_coupling(q)
            q = center_wrist_translation(q)
            raw_q.append(q)
            trajectories.append(clamp_to_limits(q).astype(np.float32))
            residuals.append(res)

        clipped = limit_clip_fraction(np.concatenate(raw_q, axis=0))
        residual = {
            k: float(np.mean([r[k] for r in residuals])) for k in ("mean", "p95", "max")
        }

        if self.verbose:
            print(
                f"[dexycb] {len(trajectories)} segments, "
                f"{sum(len(t) for t in trajectories):,} frames; "
                f"projection residual {residual['mean']:.2f} deg mean / {residual['p95']:.2f} p95; "
                f"{clipped:.2%} of DOF values hit a limit"
            )
            print(f"[dexycb] dropped {stats['dropped_invalid']:,} frames failing validity masks")

        return TrajectoryBundle(
            trajectories=trajectories,
            fps=self.fps,
            labels=labels,
            meta={
                "source": "dexycb",
                # Provenance is part of the data, not a footnote.
                "mirror": MIRROR_REPO,
                "official_release": False,
                "provenance_note": (
                    "third-party VITRA conversion of DexYCB; not the NVIDIA release"
                ),
                "hand": self.hand,
                "enforce_coupling": self.enforce_coupling,
                "projection_residual_deg": residual,
                "limit_clip_fraction": clipped,
                "label_format": f"subject->{self.label_granularity}",
                **conv.stats,
                **stats,
            },
        )

    # -- internals ----------------------------------------------------------

    def _select_files(self) -> list[Path]:
        if self.root is None:
            raise FileNotFoundError(_HELP)
        root = self.root
        anno = root / "Annotation" if (root / "Annotation").exists() else root
        files = sorted(anno.rglob("*.npy"))
        files = [f for f in files if _STEM_RE.match(f.stem)]
        if not files:
            raise FileNotFoundError(_HELP)

        if self.one_camera_per_sequence:
            best: dict[tuple[str, str], Path] = {}
            for f in files:
                m = _STEM_RE.match(f.stem)
                key = (m["subject"], m["seq"])
                # Deterministic pick: lowest camera serial.
                if key not in best or m["camera"] < _STEM_RE.match(best[key].stem)["camera"]:
                    best[key] = f
            files = [best[k] for k in sorted(best)]

        if self.max_sequences is not None:
            files = files[: self.max_sequences]
        return files

    def _read_segments(self, files: list[Path]) -> tuple[list[dict], list[str], dict]:
        segments: list[dict] = []
        labels: list[str] = []
        dropped = 0
        per_hand = defaultdict(int)

        sides = ("right", "left") if self.hand == "both" else (self.hand,)

        for path in files:
            m = _STEM_RE.match(path.stem)
            rec = np.load(path, allow_pickle=True).item()
            obj = _object_label(rec)

            # DexYCB annotates one hand per sequence; the other side's arrays
            # exist but are not real annotations. Reading them would inject
            # fabricated motion into the dataset.
            annotated = _annotated_hand(rec)
            for side in sides:
                if annotated is not None and side != annotated:
                    continue
                hand = rec.get(side)
                hand = hand.item() if hasattr(hand, "item") and getattr(hand, "shape", None) == () else hand
                if not isinstance(hand, dict) or "hand_pose" not in hand:
                    continue

                pose = np.asarray(hand["hand_pose"], dtype=np.float64)  # (T, 15, 3, 3)
                if pose.ndim != 4 or pose.shape[1:] != (15, 3, 3):
                    continue
                glob = np.asarray(hand.get("global_orient_worldspace"), dtype=np.float64)
                transl = np.asarray(hand.get("transl_worldspace"), dtype=np.float64)

                valid = np.ones(len(pose), dtype=bool)
                for key in ("annotation_valid", "kept_frames", "visible"):
                    if key in hand:
                        valid &= np.asarray(hand[key], dtype=bool)
                dropped += int((~valid).sum())

                # Prepend the wrist so joint indices match MANO_JOINT_MAP's
                # 16-joint convention.
                full = np.concatenate([glob[:, None], pose], axis=1)  # (T, 16, 3, 3)
                euler = matrix_to_euler_xyz(full)
                if side == "left":
                    # Mirror across the sagittal plane: abduction and the third
                    # axis flip sign, flexion does not.
                    euler[..., 0] *= -1.0
                    euler[..., 1] *= -1.0
                    transl = transl.copy()
                    transl[:, 0] *= -1.0

                right = shape_class(obj) if self.label_granularity == "shape" else obj
                for run in _contiguous_runs(valid, self.min_frames):
                    segments.append({"euler": euler[run], "transl": transl[run]})
                    labels.append(f"subject{m['subject']}->{right}")
                    per_hand[side] += 1

        return segments, labels, {
            "dropped_invalid": dropped,
            "segments_per_hand": dict(per_hand),
            "files_read": len(files),
        }


def _object_label(rec: dict) -> str:
    """The grasped YCB object, for the compositional split label.

    ``objects`` maps YCB ids (``002_master_chef_can``) to per-object dicts, with
    ``is_target`` marking the one the subject picks up. Several objects sit on
    the table in every sequence, so the target flag is what identifies the
    interaction rather than the scene.

    With 10 subjects and 20 YCB objects this gives a ``subject->object``
    compositional axis of up to 200 cells -- structure DexYCB's authors created
    for their own reasons, which is what makes it usable as evidence rather
    than as a construct of this repo.
    """
    objects = rec.get("objects")
    objects = (
        objects.item()
        if hasattr(objects, "item") and getattr(objects, "shape", None) == ()
        else objects
    )
    if isinstance(objects, dict) and objects:
        for key, val in objects.items():
            if isinstance(val, dict) and val.get("is_target"):
                return str(key)[:40].replace("->", "_")
        # No target flag: fall back to the single object, or the first by name.
        return str(sorted(objects)[0])[:40].replace("->", "_")
    return "unknown"


def _annotated_hand(rec: dict) -> str | None:
    """Which hand this file actually annotates, per ``anno_type``."""
    v = rec.get("anno_type")
    v = v.item() if hasattr(v, "item") and getattr(v, "shape", None) == () else v
    return str(v) if isinstance(v, str) and v in ("left", "right") else None


def _contiguous_runs(valid: np.ndarray, min_len: int) -> list[np.ndarray]:
    """Index arrays for maximal runs of True at least ``min_len`` long.

    Splitting on invalid frames rather than interpolating across them keeps
    fabricated motion out of the velocity and smoothness statistics.
    """
    runs, start = [], None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                runs.append(np.arange(start, i))
            start = None
    if start is not None and len(valid) - start >= min_len:
        runs.append(np.arange(start, len(valid)))
    return runs
