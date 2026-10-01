"""GRAB adapter: the replication test for the one axis that showed difficulty.

Why this dataset specifically
-----------------------------
Across five datasets the compositional penalty spans 200x, and exactly one axis
produced measurable difficulty: OakInk-Image's ``object category -> intent``.
That axis was constructed here from object-id prefixes, so a single positive
result on it is a candidate artefact rather than a finding.

GRAB (Taheri et al., ECCV 2020, arXiv:2008.11200) records the **same axis type**
from a different lab, a different capture rig (VICON mocap rather than
multi-view fitting), and a different object set. If difficulty reproduces here,
the axis type is a property of hand manipulation; if it does not, OakInk-Image
is the outlier and the survey has to say so.

Numbers, with their sources kept separate:

* **10 subjects, 51 objects** -- stated in the paper's abstract, verified.
* **1048 sequences, 319,228 frames after stride 4, 29 distinct
  ``motion_intent`` values, 120 fps** -- measured from the archives here, over
  the **8 of 10** subject archives downloaded. Not paper figures.

An earlier version of this docstring said "51 objects x 4 motion intents,
10 subjects, 1334 sequences". The intent count and the sequence count were
assumed rather than read, and the data contradicts the first: ``motion_intent``
takes 29 values (``drink``, ``staple``, ``toast``, ...), not four. Do not
restate either number without a source.

Format
------
Sequences arrive as ``sX/<object>_<intent>[_n].npz`` inside per-subject zips,
read in place -- extracting buys nothing and costs disk. Each file carries::

    obj_name        e.g. 'stapler'
    motion_intent   e.g. 'lift', 'pass', 'use', 'offhand'
    framerate       120.0
    rhand.params.fullpose      (T, 45)  axis-angle, 15 joints x 3
    rhand.params.global_orient (T, 3)
    rhand.params.transl        (T, 3)
    contact                    per-frame object-vertex contact labels

Two things differ from the other sources and both matter:

* **Axis-angle, not quaternions or rotation matrices.** The layout inference in
  :mod:`caredex.data.mano_retarget` is quaternion-specific and is skipped here;
  the flexion axis and sign are still inferred, never assumed.
* **120 fps, not 30.** Sub-sampled by ``stride`` so window lengths cover
  comparable real time to the other datasets; a 32-frame window at 120 fps
  would span a quarter of the motion the same window covers on OakInk.

``keep_contact=True`` reduces GRAB's per-frame contact annotation to 16 hand
regions and returns it as ``bundle.aux`` (see :mod:`caredex.data.grab_contact`).
This is the one hand dataset here carrying contact, which makes it the only
place the untested hypothesis -- that compositional information lives in what
the hand touches rather than in joint angles -- can be checked without a
simulator. The check is controlled: same trajectories, same split, same models,
27 channels against 27 + 16.

Note that GRAB's ``contact`` arrays are **part labels 0-55**, not a binary
mask, and the right-hand label range is 41-54 rather than the 40-54 that the
obvious reading suggests. That correction came from GRAB's own SMPL-X
correspondence file, not from guesswork; the empirical check available without
it was inconclusive.
"""

from __future__ import annotations

import io
import re
import zipfile
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
from caredex.data.grab_contact import (
    REGION_NAMES,
    load_hand_vertex_ids,
    region_contact,
    vertex_regions,
)
from caredex.hand_model import apply_dip_pip_coupling, clamp_to_limits

PROJECT_URL = "https://grab.is.tue.mpg.de/"

_HELP = f"""\
GRAB not found.

  1. Register and accept the licence at {PROJECT_URL}
  2. Download the ten per-subject "GRAB parameters" zips (~3.5 GB total).
     Do NOT download "Raw VICON MoCap" -- those are marker trajectories this
     project does not use, and they add 2.3 GB.
  3. Put the zips in one folder and point the config at it:
       data.source=grab  data.root=<folder containing grab__s*.zip>
"""

#: sX/<object>_<intent>[_<take>][_Retake].npz
_NAME_RE = re.compile(r"^(?P<subject>s\d+)/(?P<stem>[^/]+)\.npz$")

#: Object -> grasp-relevant shape class, for the ``shape->intent`` axis.
#:
#: This grouping is **constructed here**, not documented by GRAB, and any paper
#: text must say so -- the same caveat that applies to OakInk-Image's
#: ``category``. It is better grounded than that one in exactly one respect:
#: GRAB deliberately includes named geometric primitives at three sizes each
#: (cube/cylinder/sphere/pyramid/torus), so 15 of the 51 objects carry the
#: dataset's own shape label and the rest are grouped to match them. OakInk's
#: ``category`` is the first character of an object id.
#:
#: The criterion is what the *hand* has to do, not what the object is for: a
#: lightbulb is grasped like a sphere, a doorknob like a short cylinder.
GRAB_SHAPE_CLASS: dict[str, str] = {
    **{o: "sphere" for o in ("apple", "spherelarge", "spheremedium", "spheresmall", "lightbulb")},
    **{o: "box" for o in ("cubelarge", "cubemedium", "cubesmall", "alarmclock", "stamp", "toothpaste")},
    **{o: "cylinder" for o in ("cylinderlarge", "cylindermedium", "cylindersmall", "banana",
                               "waterbottle", "flute", "doorknob")},
    **{o: "pyramid" for o in ("pyramidlarge", "pyramidmedium", "pyramidsmall")},
    **{o: "ring" for o in ("toruslarge", "torusmedium", "torussmall", "watch", "eyeglasses",
                           "headphones")},
    **{o: "handle_tool" for o in ("hammer", "knife", "toothbrush", "scissors", "stapler",
                                  "flashlight", "fryingpan")},
    **{o: "vessel" for o in ("mug", "cup", "teapot", "wineglass", "bowl")},
    **{o: "flat_device" for o in ("phone", "mouse", "gamecontroller", "camera", "binoculars")},
    **{o: "figurine" for o in ("airplane", "duck", "elephant", "train", "stanfordbunny",
                               "piggybank", "hand")},
}

#: ``motion_intent`` takes 29 values in the archives. Grouping the 26
#: object-specific verbs (``drink``, ``staple``, ``toast``, ...) under ``use``
#: leaves four classes.
#:
#: **This grouping is constructed here.** An earlier comment claimed it
#: "collapses back to the documented four", which was an assumption: the GRAB
#: abstract names no intent taxonomy, and nothing read so far documents one.
#: Treat it like ``GRAB_SHAPE_CLASS`` -- a partition this repo invented, useful
#: for screening axes, and never to be described as the dataset's own.
_INTENT_KEEP = ("pass", "lift", "offhand")


def intent_class(intent: str) -> str:
    """Group GRAB's object-specific verbs under ``use``. Constructed here."""
    return intent if intent in _INTENT_KEEP else "use"


def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """``(..., 3)`` axis-angle -> ``(..., 3, 3)`` rotation matrices."""
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = axis_angle / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [
            np.stack([zero, -kz, ky], -1),
            np.stack([kz, zero, -kx], -1),
            np.stack([-ky, kx, zero], -1),
        ],
        axis=-2,
    )
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    return np.where(theta[..., None] < 1e-8, eye, eye + s * K + (1 - c) * (K @ K))


@register_source("grab")
class GrabSource(TrajectorySource):
    """Load GRAB right-hand trajectories, labelled ``object->intent``.

    Parameters
    ----------
    root:
        Folder holding ``grab__s*.zip``. Partial sets are fine; the loader
        reports how many subjects it actually found, because a survey claim
        about "GRAB" made from three subjects is not the same claim.
    stride:
        Frame subsampling. GRAB is 120 fps against 30 fps elsewhere, so the
        default of 4 puts window lengths on a comparable timescale.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        hand: str = "rhand",
        stride: int = 4,
        min_frames: int = 40,
        max_sequences: int | None = None,
        enforce_coupling: bool = False,
        keep_contact: bool = False,
        mano_path: str | None = None,
        verbose: bool = True,
    ) -> None:
        if hand not in ("rhand", "lhand"):
            raise ValueError(f"hand must be rhand/lhand, got {hand!r}")
        self.root = Path(root) if root else None
        self.hand = hand
        self.stride = max(1, stride)
        self.min_frames = min_frames
        self.max_sequences = max_sequences
        self.enforce_coupling = enforce_coupling
        #: Contact masks are large; reduced to 16 hand regions on read, and only
        #: when something will use them.
        self.keep_contact = keep_contact
        if keep_contact:
            from caredex.mano import load_mano

            self._hand_ids = load_hand_vertex_ids(side="right" if hand == "rhand" else "left")
            self._regions = vertex_regions(
                np.asarray(load_mano(mano_path or r"D:\datasets\mano\MANO_RIGHT.pkl").weights)
            )
        self.verbose = verbose

    # -- entry point --------------------------------------------------------

    def load(self, **kwargs: Any) -> TrajectoryBundle:
        zips = self._find_zips()
        raw: list[dict] = []
        contacts: dict[str, np.ndarray] = {}

        for zp in zips:
            if self.max_sequences is not None and len(raw) >= self.max_sequences:
                break
            with zipfile.ZipFile(zp) as z:
                members = [n for n in z.namelist() if _NAME_RE.match(n)]
                for name in sorted(members):
                    if self.max_sequences is not None and len(raw) >= self.max_sequences:
                        break
                    rec = self._read_sequence(z, name)
                    if rec is None:
                        continue
                    if self.keep_contact:
                        contacts[rec["key"]] = rec.pop("contact")
                    raw.append(rec)
            if self.verbose:
                print(f"[grab] {zp.name}: {len(raw)} sequences so far", flush=True)

        if not raw:
            raise RuntimeError(f"no usable sequences found in {self.root}")

        # One convention inferred over the pooled data, so every sequence is
        # converted identically -- inferring per file could silently mix two.
        pooled = np.concatenate([r["euler"] for r in raw], axis=0)
        conv = RetargetConventions.infer(pooled)
        if self.verbose:
            print(conv.describe("grab"), flush=True)

        trajectories, labels, residuals, raw_q, aux = [], [], [], [], []
        for r in raw:
            q, res = conv.apply(r["euler"], r["transl"])
            if self.enforce_coupling:
                q = apply_dip_pip_coupling(q)
            if len(q) < self.min_frames:
                continue
            q = center_wrist_translation(q)
            raw_q.append(q)
            trajectories.append(clamp_to_limits(q).astype(np.float32))
            labels.append(r["label"])
            residuals.append(res)
            if self.keep_contact:
                c = contacts[r["key"]]
                # Contact and pose come from the same frames, but a sequence can
                # be one frame shorter after projection; trim rather than pad so
                # a misalignment cannot be silently filled with zeros.
                aux.append(c[: len(q)].astype(np.float32))

        if not trajectories:
            raise RuntimeError(f"every sequence was shorter than min_frames={self.min_frames}")

        clipped = limit_clip_fraction(np.concatenate(raw_q, axis=0))
        residual = {k: float(np.mean([r[k] for r in residuals])) for k in ("mean", "p95", "max")}

        if self.verbose:
            print(
                f"[grab] {len(trajectories)} sequences, "
                f"{sum(len(t) for t in trajectories):,} frames at {120 / self.stride:g} fps; "
                f"projection residual {residual['mean']:.2f} deg mean / {residual['p95']:.2f} p95; "
                f"{clipped:.2%} of DOF values hit a limit"
            )

        return TrajectoryBundle(
            aux=aux if self.keep_contact else None,
            aux_names=list(REGION_NAMES) if self.keep_contact else [],
            trajectories=trajectories,
            fps=120.0 / self.stride,
            labels=labels,
            meta={
                "source": "grab",
                "hand": self.hand,
                "n_subject_archives": len(zips),
                "subjects": sorted({p.stem.split("__")[-1] for p in zips}),
                "stride": self.stride,
                "enforce_coupling": self.enforce_coupling,
                "projection_residual_deg": residual,
                "limit_clip_fraction": clipped,
                "label_format": "object->intent",
                "has_contact": bool(contacts),
                **conv.stats,
            },
        )

    # -- internals ----------------------------------------------------------

    def _find_zips(self) -> list[Path]:
        if self.root is None or not self.root.exists():
            raise FileNotFoundError(_HELP)
        zips = sorted(self.root.glob("grab__s*.zip"))
        if not zips:
            raise FileNotFoundError(_HELP)
        if self.verbose and len(zips) < 10:
            print(
                f"[grab] WARNING: {len(zips)} of 10 subject archives present. "
                "Any claim about GRAB must state the subset actually used."
            )
        return zips

    def _read_sequence(self, z: zipfile.ZipFile, name: str) -> dict | None:
        m = _NAME_RE.match(name)
        if m is None:
            return None
        try:
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
        except Exception:  # noqa: BLE001 - one bad member should not stop the load
            return None

        hand = d.get(self.hand)
        if hand is None:
            return None
        params = hand.item()["params"] if hand.dtype == object else hand["params"]

        # fullpose is the 15 joints as axis-angle; hand_pose is a 24-component
        # PCA projection of the same thing, and reconstructing from it would
        # throw away detail for no reason.
        pose = np.asarray(params["fullpose"], dtype=np.float64)
        if pose.ndim != 2 or pose.shape[1] != 45:
            return None
        glob = np.asarray(params["global_orient"], dtype=np.float64)
        transl = np.asarray(params["transl"], dtype=np.float64)

        sel = np.arange(0, len(pose), self.stride)
        pose, glob, transl = pose[sel], glob[sel], transl[sel]

        aa = np.concatenate([glob[:, None, :], pose.reshape(len(pose), 15, 3)], axis=1)
        euler = matrix_to_euler_xyz(_rodrigues(aa))

        obj = str(d["obj_name"]) if "obj_name" in d else "unknown"
        intent = str(d["motion_intent"]) if "motion_intent" in d else "unknown"
        rec = {
            "euler": euler,
            "transl": transl,
            "label": f"{obj}->{intent}".replace(" ", "_"),
            "key": f"{m['subject']}/{m['stem']}",
        }
        if self.keep_contact and "contact" in d:
            c = d["contact"]
            c = c.item() if c.dtype == object else c
            # Subsample to match the pose stride, then reduce to 16 hand regions
            # here rather than carrying (T, 10475) masks through the loader --
            # 1048 sequences of those would not fit in memory.
            body = np.asarray(c["body"] if isinstance(c, dict) else c)[sel]
            rec["contact"] = region_contact(body, self._hand_ids, self._regions)
        return rec
