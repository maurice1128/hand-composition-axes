"""OakInk's own object taxonomy, read from ``metaV2.zip``.

Why this exists
---------------
The one dataset in this project that shows compositional difficulty was
measured on a ``category -> intent`` axis, where ``category`` was **the first
character of the object id** -- a grouping this repo invented. That is the
single weakest point in the evidence, and the obvious reviewer question is
whether the effect is a property of hand manipulation or an artefact of an
arbitrary partition.

It turns out to have been worse than "arbitrary": the leading character takes
five values (A, C, O, S, Y) that track **where the object came from**, not what
it is. OakInk ships its own taxonomy in a 15 KB file that was sitting in the
download the whole time, and every one of the 100 object ids in the bundle
resolves in it.

Three groupings are available, all of them the dataset's own:

``category``
    ``yodaobject_cat.json`` maps the two digits after the leading letter to a
    named object category -- ``mug``, ``cup``, ``bowl``, ``teapot``,
    ``wineglass``, ``cylinder_bottle``, ``trigger_sprayer``, ... 33 groups over
    the 100 objects here.
``class``
    Coarse functional class from ``object_id.json``: ``container``,
    ``maniptools``, ``wearable``, ``misc``, ``geometry``. 5 groups.
``attr``
    Affordance attributes, also from ``object_id.json``: ``handled``,
    ``pourable``, ``squeezable``, ``lotion_pump``, ... The first attribute is
    used when an object carries several. 12 groups, and the most directly
    grasp-relevant of the three -- what the hand has to do with an object is
    closer to its affordance than to its name.

None of these is a substitute for the others; running all three and reporting
all three is the honest response to "did you pick the partition that worked".
"""

from __future__ import annotations

import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path

#: Searched in order. ``CAREDEX_OAKINK_META`` overrides.
_CANDIDATES = (
    r"D:\datasets\oakink\zipped\shape\metaV2.zip",
    r"D:\datasets\oakink\metaV2.zip",
)

_HELP = """\
OakInk metaV2.zip not found.

It is a 15 KB file in the OakInk-v1 "shape" download, alongside
OakInkObjectsV2.zip. Point at it with:

    set CAREDEX_OAKINK_META=<path to metaV2.zip>
"""


def _locate() -> Path:
    env = os.environ.get("CAREDEX_OAKINK_META")
    for c in ([env] if env else []) + list(_CANDIDATES):
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(_HELP)


@lru_cache(maxsize=1)
def _tables() -> tuple[dict, dict]:
    with zipfile.ZipFile(_locate()) as z:
        objects = json.loads(z.read("metaV2/object_id.json").decode("utf-8"))
        categories = json.loads(z.read("metaV2/yodaobject_cat.json").decode("utf-8"))
    return objects, categories


#: OakInk's own attribute list spells one affordance two ways. Left unmerged it
#: splits a single group in two, which changes the axis rather than the data.
#: This is the only normalisation applied to the dataset's taxonomy, and it is
#: listed here rather than done silently.
_ATTR_ALIASES = {"squeezeable": "squeezable"}


def object_group(object_id: str, mode: str) -> str:
    """Map an OakInk object id to one of the dataset's own groupings.

    Unknown ids fall back to the id itself rather than to a shared "unknown"
    bucket: silently merging unmapped objects into one cell would invent a
    composition that the dataset does not contain.
    """
    objects, categories = _tables()
    rec = objects.get(object_id)
    if rec is None:
        return object_id
    if mode == "category":
        # 'A01001' -> '01' -> 'trigger_sprayer'
        return categories.get(object_id[1:3], object_id[1:3])
    if mode == "class":
        return str(rec.get("class", "unknown"))
    if mode == "attr":
        # Some records carry attr [""] rather than [], which produced cells
        # named "->0001" -- a group with no name is a group nobody can check.
        attrs = [a for a in (rec.get("attr") or []) if str(a).strip()]
        return _ATTR_ALIASES.get(str(attrs[0]), str(attrs[0])) if attrs else "no_attr"
    raise ValueError(f"unknown OakInk grouping {mode!r}; expected category/class/attr")


def coverage(object_ids: list[str]) -> dict:
    """How many ids resolve, so a partial mapping cannot pass unnoticed."""
    objects, _ = _tables()
    known = [o for o in object_ids if o in objects]
    return {
        "n_ids": len(set(object_ids)),
        "n_resolved": len(set(known)),
        "fraction": len(set(known)) / max(len(set(object_ids)), 1),
    }
