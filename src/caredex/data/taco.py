"""TACO adapter: a fourth, independent dataset for an out-of-sample prediction.

Why this dataset specifically
-----------------------------
The study's rule -- a label must describe the motion in the scored window before
compositional generalisation can be measured -- was found on three datasets, two
of which share a collection effort. TACO (Liu et al., CVPR 2024,
arXiv:2401.08399) was not used to form it. Each sequence is labelled with a
``(action, tool, object)`` triplet, which gives three candidate two-factor axes
with the full triplet as the fine label. The prediction and the rule that picks
the confirmatory axis are fixed in ``runs/PREREG_taco_prediction.md``, written
before any TACO file was opened.

Format, as documented -- and what is verified rather than assumed
-----------------------------------------------------------------
``Hand_Poses.zip`` is read in place (14 GB; never extract it). Per TACO's README
and its ``dataset_utils/hand_pose_loader.py``::

    Hand_Poses/(action, tool, object)/<YYYYMMDD_NNN>/right_hand.pkl
                                                     right_hand_shape.pkl
                                                     left_hand.pkl  ...

    right_hand.pkl   {frame_key: {"hand_pose": (48,), "hand_trans": (3,)}}
                     48 = 3 global + 15 joints x 3, MANO axis-angle, no PCA

None of that is trusted. The loader records what it actually finds -- member
layout, frame-key type, value type (torch tensor or ndarray), pose width,
sequences with no right hand, non-finite frames -- in ``bundle.meta["format"]``
so the build report can state every difference from the documentation.

Three things differ from the other sources and all three matter:

* **The values are pickled torch tensors**, possibly saved from a GPU. A plain
  ``pickle.load`` would try to restore them to CUDA, which either fails or
  touches a GPU that other jobs hold. ``_CpuUnpickler`` maps every storage to
  the CPU.
* **The pose may or may not include MANO's mean pose.** TACO's loader builds
  ``ManoLayer(use_pca=False)`` with manopth's default ``flat_hand_mean=False``,
  under which the stored 45 numbers are *offsets from* ``hands_mean`` and the
  articulation is ``hand_pose[3:] + hands_mean``. GRAB's ``fullpose`` is the
  opposite. Getting this wrong shifts every flexion angle by a constant
  (MANO's mean is 4 to 49 degrees per joint), which still looks plausible in aggregate
  -- the same failure shape as the flipped OakInk sign. So it is **inferred**:
  both readings are converted under their own inferred conventions and the one
  the anatomical limits admit more of is kept, with both numbers logged and the
  agreement with TACO's documented loader stated. ``pose_mean="add"`` or
  ``"flat"`` overrides, for reproducing a specific reading.
* **Bimanual, and the tool hand is not guaranteed to be the right one.** This
  study uses the right hand everywhere, so this does too, and the action label
  may then describe what the *other* hand is doing. Nothing here can detect
  that from 27 joint angles; see "What this does NOT do".

What this does NOT do
---------------------
* It does not verify that the right hand is the hand performing the labelled
  action. TACO's paper describes the right hand as holding the tool, but a
  left-handed subject or a swapped take would be loaded without complaint.
  Right-hand motion energy relative to the left is recorded per sequence in
  ``meta["right_vs_left_motion"]`` as a weak flag, not as a check.
* It does not verify one-action-per-sequence. Duration is reported; whether a
  long sequence contains several actions needs the video.
* The mean-pose inference compares two clip fractions. If they are close the
  choice is weakly determined and the report says so; it is not a proof.
* Zero joint-limit violations after ``clamp_to_limits`` say nothing. The clip
  fraction reported is measured **before** clamping.
"""

from __future__ import annotations

import io
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from caredex.data.base import TrajectoryBundle, TrajectorySource, register_source
from caredex.data.grab import _rodrigues
from caredex.data.mano_retarget import (
    RetargetConventions,
    center_wrist_translation,
    limit_clip_fraction,
    matrix_to_euler_xyz,
)
from caredex.hand_model import apply_dip_pip_coupling, clamp_to_limits

PROJECT_URL = "https://taco2024.github.io/"

_HELP = f"""\
TACO hand poses not found.

  1. Follow the dataset link on {PROJECT_URL} (Dropbox, "Version 1").
  2. Download Hand_Poses.zip only (14.2 GB). The RGB-D videos are not used.
  3. Do NOT extract it. Point the config at the zip:
       data.source=taco  data.root=D:\\datasets\\taco\\Hand_Poses.zip
"""

#: .../(action, tool, object)/<sequence>/<side>_hand.pkl   (integrated file)
#: .../(action, tool, object)/<sequence>/<side>_hand/<frame>.pkl  (per-frame)
_MEMBER_RE = re.compile(
    r"^(?:.*/)?(?P<triplet>\([^/()]+\))/(?P<seq>[^/]+)/"
    r"(?P<side>left|right)_hand(?P<kind>\.pkl|/[^/]+\.pkl)$"
)

#: Characters the experiment's label parsers give meaning to.
_RESERVED = ("@", "->", "|")


#: What the unpickler actually met, for the build report: storage classes and
#: the device each tensor was saved from.
FORMAT_SEEN: dict[str, set] = {"storage_types": set(), "saved_from_device": set(), "decoder": set()}

_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32,
}


class _StoragePidUnpickler(pickle.Unpickler):
    """Reads the one persistent id inside a legacy ``torch.save`` of a storage."""

    def find_class(self, module: str, name: str):
        if module == "torch" and name.endswith("Storage"):
            return name
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return pid


def _storage_from_bytes(b: bytes):
    """Decode ``torch.storage._load_from_bytes`` payloads without torch.

    Every TACO tensor pickles its **whole backing storage** -- a 48-value pose is
    a view into a ``(T, 48)`` buffer, stored again for each frame, which is why a
    221-frame sequence is a 10 MB pickle. ``torch.load`` on each costs about 2 ms,
    over an hour for the dataset. The legacy format is three header pickles, the
    storage's persistent id, the key list, then ``int64 numel`` and raw
    little-endian data, so it is read directly. Anything else (the zip format, an
    unknown storage class) falls back to ``torch.load`` mapped to the CPU, never
    to the device it was saved from: other jobs hold the GPU.
    """
    f = io.BytesIO(b)
    try:
        if pickle.load(f) != _LEGACY_MAGIC:
            raise ValueError("not legacy")
        pickle.load(f)  # protocol version
        pickle.load(f)  # sys info
        pid = _StoragePidUnpickler(f).load()
        pickle.load(f)  # storage keys
        kind, storage_type, _key, location, numel = pid[:5]
        dtype = _STORAGE_DTYPES[storage_type]
        (n,) = np.frombuffer(f.read(8), dtype="<i8")
        if kind != "storage" or n != numel:
            raise ValueError("unexpected layout")
        data = np.frombuffer(f.read(int(n) * np.dtype(dtype).itemsize), dtype=dtype)
        if len(data) != n:
            raise ValueError("truncated storage")
    except Exception:  # noqa: BLE001 - any surprise goes to the slow, general path
        import torch

        FORMAT_SEEN["decoder"].add("torch.load(map_location=cpu)")
        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    FORMAT_SEEN["storage_types"].add(storage_type)
    FORMAT_SEEN["saved_from_device"].add(str(location))
    FORMAT_SEEN["decoder"].add("legacy torch.save storage, read directly")
    return data


def _rebuild_tensor(storage, offset, size, stride, *rest):
    if isinstance(storage, np.ndarray):
        item = storage.itemsize
        return np.lib.stride_tricks.as_strided(
            storage[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)
        ).copy()
    import torch._utils

    return torch._utils._rebuild_tensor_v2(storage, offset, size, stride, *rest)


class _CpuUnpickler(pickle.Unpickler):
    """Unpickle TACO's torch tensors as numpy arrays, never touching a GPU."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return _storage_from_bytes
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        return super().find_class(module, name)


def _loads(data: bytes) -> Any:
    return _CpuUnpickler(io.BytesIO(data)).load()


def _to_numpy(x: Any) -> tuple[np.ndarray, str]:
    kind = type(x).__module__.split(".")[0] + "." + type(x).__name__
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1), kind


def parse_triplet(dirname: str) -> tuple[str, str, str]:
    """``"(stir, spoon, bowl)"`` -> ``("stir", "spoon", "bowl")``.

    The order of the three fields is **not** decided here; the caller names
    them. TACO's README writes ``<tool, action, object>`` while its directory
    names read as (action, tool, object), so the build report prints each
    position's vocabulary for a human to confirm.
    """
    parts = [p.strip().replace(" ", "_") for p in dirname.strip()[1:-1].split(",")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"not a triplet directory: {dirname!r}")
    for p in parts:
        if any(r in p for r in _RESERVED):
            raise ValueError(f"triplet field {p!r} contains a reserved label character")
    return parts[0], parts[1], parts[2]


def fine_label(triplet: tuple[str, str, str]) -> str:
    return "|".join(triplet)


@register_source("taco")
class TacoSource(TrajectorySource):
    """Load TACO right-hand trajectories, labelled ``action|tool|object``.

    Parameters
    ----------
    root:
        ``Hand_Poses.zip`` itself, or the folder containing it.
    pose_mean:
        ``"infer"`` (default), ``"add"`` (pose is an offset from MANO's
        ``hands_mean``, as TACO's own loader implies) or ``"flat"``.
    stride:
        Frame subsampling. TACO is 30 fps like OakInk-Image, so 1.
    """

    def __init__(
        self,
        root: str | Path | None = r"D:\datasets\taco\Hand_Poses.zip",
        hand: str = "right",
        stride: int = 1,
        min_frames: int = 40,
        max_sequences: int | None = None,
        pose_mean: str = "infer",
        mano_path: str | Path = r"D:\datasets\mano\MANO_RIGHT.pkl",
        enforce_coupling: bool = False,
        compare_hands: bool = True,
        verbose: bool = True,
    ) -> None:
        if hand not in ("right", "left"):
            raise ValueError(f"hand must be right/left, got {hand!r}")
        if pose_mean not in ("infer", "add", "flat"):
            raise ValueError(f"pose_mean must be infer/add/flat, got {pose_mean!r}")
        self.root = Path(root) if root else None
        self.hand = hand
        self.stride = max(1, stride)
        self.min_frames = min_frames
        self.max_sequences = max_sequences
        self.pose_mean = pose_mean
        self.mano_path = Path(mano_path)
        self.enforce_coupling = enforce_coupling
        #: Reading the other hand doubles the I/O; it buys only the weak flag.
        self.compare_hands = compare_hands
        self.verbose = verbose

    # -- entry point --------------------------------------------------------

    def load(self, **kwargs: Any) -> TrajectoryBundle:
        zp = self._find_zip()
        fmt: dict[str, Any] = {}
        dropped: dict[str, list[str]] = {
            "no_hand_file": [], "unreadable": [], "bad_shape": [],
            "non_finite": [], "too_short": [],
        }
        raw: list[dict] = []

        with zipfile.ZipFile(zp) as z:
            index = self._index(z, fmt)
            seq_keys = sorted(index)
            fmt["n_sequences_found"] = len(seq_keys)
            fmt["n_triplet_dirs"] = len({k[0] for k in seq_keys})
            for n, key in enumerate(seq_keys):
                if self.max_sequences is not None and len(raw) >= self.max_sequences:
                    break
                entry = index[key]
                if self.hand not in entry:
                    dropped["no_hand_file"].append("/".join(key))
                    continue
                rec = self._read_sequence(z, key, entry, fmt, dropped)
                if rec is not None:
                    raw.append(rec)
                if self.verbose and (n + 1) % 200 == 0:
                    print(f"[taco] read {n + 1}/{len(seq_keys)} sequences, "
                          f"{len(raw)} usable", flush=True)

        if not raw:
            raise RuntimeError(f"no usable sequences found in {zp}")

        # "value_type" is what arrives after decoding (ndarray on the fast path);
        # this is what the pickles themselves declared.
        fmt["pickled_as"] = {"class": "torch.Tensor via torch._utils._rebuild_tensor_v2",
                             **self._seen()}
        mean_stats, use_mean = self._choose_pose_mean(raw)
        eulers = [self._euler(r["pose"], use_mean) for r in raw]

        # One convention inferred over the pooled data, so every sequence is
        # converted identically -- inferring per file could silently mix two.
        conv = RetargetConventions.infer(np.concatenate(eulers, axis=0))
        if self.verbose:
            print(conv.describe("taco"), flush=True)

        trajectories, labels, seq_ids, residuals, raw_q, motion = [], [], [], [], [], []
        for r, e in zip(raw, eulers):
            q, res = conv.apply(e, r["transl"])
            if self.enforce_coupling:
                q = apply_dip_pip_coupling(q)
            if len(q) < self.min_frames:
                dropped["too_short"].append(r["key"])
                continue
            q = center_wrist_translation(q)
            raw_q.append(q)
            trajectories.append(clamp_to_limits(q).astype(np.float32))
            labels.append(r["label"])
            seq_ids.append(r["key"])
            residuals.append(res)
            motion.append(r["right_vs_left_motion"])

        if not trajectories:
            raise RuntimeError(f"every sequence was shorter than min_frames={self.min_frames}")

        # Measured BEFORE clamping. Three adapters once fed this post-clamp
        # values and it returned 0.00% for a DOF pinned in every frame.
        clipped = limit_clip_fraction(np.concatenate(raw_q, axis=0))
        residual = {k: float(np.mean([r[k] for r in residuals])) for k in ("mean", "p95", "max")}
        fps = 30.0 / self.stride

        if self.verbose:
            print(
                f"[taco] {len(trajectories)} sequences, "
                f"{sum(len(t) for t in trajectories):,} frames at {fps:g} fps; "
                f"projection residual {residual['mean']:.2f} deg mean / {residual['p95']:.2f} p95; "
                f"{clipped:.2%} of DOF values outside a limit before clamping"
            )

        return TrajectoryBundle(
            trajectories=trajectories,
            fps=fps,
            labels=labels,
            meta={
                "source": "taco",
                "hand": self.hand,
                "stride": self.stride,
                "enforce_coupling": self.enforce_coupling,
                "projection_residual_deg": residual,
                "limit_clip_fraction": clipped,
                "label_format": "action|tool|object",
                "sequence_ids": seq_ids,
                "right_vs_left_motion": motion,
                "dropped": {k: v for k, v in dropped.items()},
                "format": fmt,
                "pose_mean": mean_stats,
                **conv.stats,
            },
        )

    # -- internals ----------------------------------------------------------

    def _find_zip(self) -> Path:
        if self.root is None or not self.root.exists():
            raise FileNotFoundError(_HELP)
        zp = self.root / "Hand_Poses.zip" if self.root.is_dir() else self.root
        if not zp.exists():
            raise FileNotFoundError(_HELP)
        return zp

    def _index(self, z: zipfile.ZipFile, fmt: dict) -> dict[tuple[str, str], dict]:
        """``(triplet_dir, sequence) -> {side: [member names]}``."""
        index: dict[tuple[str, str], dict[str, list[str]]] = {}
        names = z.namelist()
        unmatched = []
        kinds = set()
        for name in names:
            if name.endswith("/"):
                continue
            m = _MEMBER_RE.match(name)
            if m is None:
                if not name.endswith("_shape.pkl"):
                    unmatched.append(name)
                continue
            kinds.add("integrated" if m["kind"] == ".pkl" else "per_frame")
            index.setdefault((m["triplet"], m["seq"]), {}).setdefault(m["side"], []).append(name)
        fmt["n_zip_members"] = len(names)
        fmt["member_kinds"] = sorted(kinds)
        fmt["n_unmatched_members"] = len(unmatched)
        fmt["unmatched_examples"] = unmatched[:5]
        fmt["example_members"] = [n for n in names if not n.endswith("/")][:4]
        return index

    @staticmethod
    def _seen() -> dict:
        return {k: sorted(v) for k, v in FORMAT_SEEN.items()}

    def _read_side(self, z: zipfile.ZipFile, members: list[str], fmt: dict):
        """Returns ``(pose (T, W), trans (T, 3))`` in frame order."""
        members = sorted(members)
        if len(members) == 1 and _MEMBER_RE.match(members[0])["kind"] == ".pkl":
            data = _loads(z.read(members[0]))
            if not isinstance(data, dict):
                raise TypeError(f"expected a dict keyed by frame, got {type(data).__name__}")
            keys = sorted(data)
            fmt.setdefault("frame_key_type", type(keys[0]).__name__ if keys else "none")
            fmt.setdefault("frame_key_examples", [repr(k) for k in keys[:3]])
            fmt.setdefault("frame_value_keys", sorted(map(str, data[keys[0]])) if keys else [])
            frames = [data[k] for k in keys]
        else:
            frames = [_loads(z.read(m)) for m in members]
        poses, trans = [], []
        for f in frames:
            p, kind = _to_numpy(f["hand_pose"])
            t, _ = _to_numpy(f["hand_trans"])
            fmt.setdefault("value_type", kind)
            poses.append(p)
            trans.append(t)
        return np.stack(poses), np.stack(trans)

    def _read_sequence(self, z, key, entry, fmt, dropped) -> dict | None:
        name = "/".join(key)
        try:
            pose, transl = self._read_side(z, entry[self.hand], fmt)
        except Exception as exc:  # noqa: BLE001 - one bad member should not stop the load
            dropped["unreadable"].append(f"{name}: {type(exc).__name__}: {exc}"[:200])
            return None
        fmt.setdefault("pose_width", int(pose.shape[1]))
        if pose.ndim != 2 or pose.shape[1] != 48 or transl.shape != (len(pose), 3):
            dropped["bad_shape"].append(f"{name}: pose {pose.shape} trans {transl.shape}")
            return None
        if not (np.isfinite(pose).all() and np.isfinite(transl).all()):
            dropped["non_finite"].append(name)
            return None

        # A weak flag for "which hand is doing the work", never a filter.
        # None, not NaN: bundle meta round-trips through repr/literal_eval.
        ratio: float | None = None
        other = "left" if self.hand == "right" else "right"
        if self.compare_hands and other in entry:
            try:
                o_pose, _ = self._read_side(z, entry[other], {})
                if o_pose.shape[1] == 48 and len(o_pose) > 1 and len(pose) > 1:
                    mine = float(np.abs(np.diff(pose[:, 3:], axis=0)).mean())
                    theirs = float(np.abs(np.diff(o_pose[:, 3:], axis=0)).mean())
                    ratio = mine / theirs if theirs > 0 else None
            except Exception:  # noqa: BLE001
                pass

        sel = np.arange(0, len(pose), self.stride)
        return {
            "pose": pose[sel],
            "transl": transl[sel],
            "label": fine_label(parse_triplet(key[0])),
            "key": name,
            "right_vs_left_motion": ratio,
        }

    def _hands_mean(self) -> np.ndarray:
        if getattr(self, "_mean_cache", None) is None:
            from caredex.mano import load_mano

            mean = load_mano(self.mano_path).hands_mean
            if mean is None:
                raise RuntimeError(f"{self.mano_path} carries no hands_mean")
            self._mean_cache = np.asarray(mean, dtype=np.float64).reshape(45)
        return self._mean_cache

    def _euler(self, pose: np.ndarray, add_mean: bool) -> np.ndarray:
        aa = pose.reshape(len(pose), 16, 3).copy()
        if add_mean:
            # manopth adds the mean in axis-angle space, before Rodrigues.
            aa[:, 1:, :] += self._hands_mean().reshape(15, 3)
        return matrix_to_euler_xyz(_rodrigues(aa))

    def _choose_pose_mean(self, raw: list[dict]) -> tuple[dict, bool]:
        """Decide whether ``hands_mean`` belongs in the pose, from the data."""
        # A subsample is enough to separate the two readings and keeps the
        # double conversion cheap.
        step = max(1, sum(len(r["pose"]) for r in raw) // 200_000)
        pooled = np.concatenate([r["pose"][::step] for r in raw], axis=0)
        transl = np.zeros((len(pooled), 3))
        stats: dict[str, Any] = {"documented_by_taco_loader": "add (manopth flat_hand_mean=False)"}
        for tag, add in (("flat", False), ("add", True)):
            e = self._euler(pooled, add)
            conv = RetargetConventions.infer(e)
            q, _ = conv.apply(e, transl)
            stats[f"clip_fraction_{tag}"] = limit_clip_fraction(q)
            stats[f"conventions_{tag}"] = (
                f"finger {conv.stats['flexion_axis']}{conv.finger[1]:+.0f} "
                f"abd {conv.stats['abduction_axis']}{conv.finger[3]:+.0f}; "
                f"thumb {conv.stats['thumb_flexion_axis']}{conv.thumb[1]:+.0f} "
                f"abd {conv.stats['thumb_abduction_axis']}{conv.thumb[3]:+.0f}"
            )
        inferred = "add" if stats["clip_fraction_add"] < stats["clip_fraction_flat"] else "flat"
        stats["inferred"] = inferred
        stats["chosen"] = inferred if self.pose_mean == "infer" else self.pose_mean
        stats["mode"] = self.pose_mean
        stats["agrees_with_documentation"] = stats["chosen"] == "add"
        if self.verbose:
            print(
                f"[taco] mean pose: clip fraction {stats['clip_fraction_flat']:.2%} without "
                f"hands_mean, {stats['clip_fraction_add']:.2%} with it -> {stats['chosen']} "
                f"({'agrees with' if stats['agrees_with_documentation'] else 'CONTRADICTS'} "
                f"TACO's loader)", flush=True,
            )
        return stats, stats["chosen"] == "add"
