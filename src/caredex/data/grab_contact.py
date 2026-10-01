"""Per-region hand contact from GRAB, as a 16-channel signal.

Why
---
Four of five datasets in this project show no compositional difficulty in joint
angles, including OakInk2, which annotates 391 primitive transitions and whose
retargeting fidelity is now the best of the four. The working explanation is
that the primitives differ in *what the hand touches and why*, not in *how the
fingers are configured* -- posture is dominated by generic grasp synergies, so
predicting 27 joint angles does not require knowing whether the subject is
igniting a lamp or opening a gate.

That explanation has never been tested, because none of the other datasets
carries contact. GRAB does: a per-frame binary contact map over body and object
vertices. So the experiment is a controlled one -- same trajectories, same
split, same models, one thing changed:

    27 joint angles                    -> compositional penalty ~0 (measured)
    27 joint angles + 16 contact regions -> ?

If difficulty appears only when contact is present, the four nulls stop being
"we found nothing" and become "we found where the signal is not".

Aggregation
-----------
GRAB's ``contact['body']`` is a mask over SMPL-X's 10475 body vertices. Two
steps bring it to something a model can consume:

1. **SMPL-X body vertex -> MANO hand vertex.** GRAB ships the correspondence
   itself, in a 77.8 KB ``tools__smplx_correspondence.zip`` on its download
   page: ``rhand_smplx_ids.npy`` lists the 778 body vertices that are MANO's
   right hand, in MANO order. No separate SMPL-X registration is needed.

   Getting this file mattered more than its size suggests. GRAB's contact
   arrays are **part labels 0-55**, not a binary mask, and guessing which
   labels are the right hand was off by one: the correspondence shows right
   hand = 41-54 and left hand = 26-40, so treating 40 as right -- the obvious
   reading of "left 25-39, right 40-54" -- would have mixed the left wrist into
   every right-hand contact feature. An empirical check using GRAB's
   ``offhand`` sequences was inconclusive (right-hand share 0.89 vs 0.72, no
   clean flip), because ``offhand`` means passing the object to the other hand
   rather than performing the whole sequence with it.
2. **778 vertices -> 16 regions.** No extra asset needed: MANO's own skinning
   weights already assign every vertex a dominant joint, which *is* an
   anatomical partition -- 199 vertices on the palm, 20-54 on each phalanx. A
   hand-drawn segmentation would be a second invented grouping, and this
   project has already been bitten once by inventing a partition (OakInk's
   "category" turned out to encode object provenance).

The feature per region is the **fraction of that region's vertices in contact**,
which is bounded in [0, 1], comparable across regions of different sizes, and
degrades gracefully -- unlike a raw count, which would make the palm dominate
purely by having more vertices.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

#: MANO joint order, which is also the region order of the emitted features.
#: Joint 0 is the wrist, and its skinning basin is the palm.
REGION_NAMES: tuple[str, ...] = (
    "palm",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "thumb_cmc", "thumb_mcp", "thumb_ip",
)

N_REGIONS = len(REGION_NAMES)

#: Searched in order; ``CAREDEX_GRAB_CORRESPONDENCE`` overrides.
_CANDIDATES = (
    r"D:\datasets\grab\tools__smplx_correspondence.zip",
)

_HELP = """\
GRAB SMPL-X correspondence not found.

Needed file: tools__smplx_correspondence.zip (77.8 KB)

On the GRAB download page, row "SMPL-X to MANO and FLAME mapping (optional)".
GRAB's contact arrays index SMPL-X body vertices, so without it there is no way
to say which contacts belong to the right hand -- and guessing the part-label
ranges is off by one, which would fold the left wrist into every right-hand
feature.

Put it beside the grab__s*.zip archives, or point at it with:
    set CAREDEX_GRAB_CORRESPONDENCE=<path to tools__smplx_correspondence.zip>
"""


def load_hand_vertex_ids(path: str | Path | None = None, side: str = "right") -> np.ndarray:
    """SMPL-X body-vertex indices for one MANO hand, in MANO vertex order."""
    import os
    import zipfile

    for cand in ([path] if path else []) + [os.environ.get("CAREDEX_GRAB_CORRESPONDENCE")] + list(_CANDIDATES):
        if cand and Path(cand).exists():
            with zipfile.ZipFile(cand) as z:
                member = f"smplx_correspondence/{side[0]}hand_smplx_ids.npy"
                out = np.load(io.BytesIO(z.read(member))).astype(np.int64)
            if out.shape != (778,):
                raise ValueError(f"expected 778 MANO vertices, got {out.shape}")
            return out
    raise FileNotFoundError(_HELP)


def vertex_regions(mano_weights: np.ndarray) -> np.ndarray:
    """``(778, 16)`` skinning weights -> ``(778,)`` dominant-joint region index."""
    w = np.asarray(mano_weights)
    if w.ndim != 2 or w.shape[0] != 778 or w.shape[1] != N_REGIONS:
        raise ValueError(f"expected (778, {N_REGIONS}) skinning weights, got {w.shape}")
    return w.argmax(axis=1).astype(np.int64)


def region_contact(
    body_contact: np.ndarray, hand_vertex_ids: np.ndarray, regions: np.ndarray
) -> np.ndarray:
    """``(T, 10475)`` SMPL-X contact mask -> ``(T, 16)`` per-region contact fraction.

    A fraction rather than a count: the palm holds 199 of the 778 vertices and a
    raw count would let it dominate the feature purely by size.
    """
    body_contact = np.asarray(body_contact)
    if body_contact.ndim != 2:
        raise ValueError(f"expected (T, n_vertices), got {body_contact.shape}")
    hand = (body_contact[:, hand_vertex_ids] > 0).astype(np.float32)  # (T, 778)

    sizes = np.bincount(regions, minlength=N_REGIONS).astype(np.float32)
    sums = np.zeros((len(hand), N_REGIONS), dtype=np.float32)
    np.add.at(sums.T, regions, hand.T)
    return sums / np.maximum(sizes, 1.0)


def summarise(features: np.ndarray) -> dict:
    """Diagnostics, so a silently all-zero contact channel cannot pass."""
    f = np.asarray(features)
    per_region = f.mean(axis=0)
    return {
        "n_frames": int(len(f)),
        "frames_with_any_contact": float((f.sum(axis=1) > 0).mean()),
        "mean_contact_fraction": float(f.mean()),
        "dead_regions": [REGION_NAMES[i] for i in range(N_REGIONS) if per_region[i] == 0.0],
        "busiest": REGION_NAMES[int(per_region.argmax())],
    }
