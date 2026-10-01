"""How much of an OakInk2 recording does its label describe? A per-axis operationalisation of alignment.

Why. The referee asked for the paper's explanatory variable to be measured across the Table 1 axes and not only on
constructed bundles. The three diagnostics of Section 3.4 cannot do it: `scripts/axis_diagnostics.py` showed they are
0.000, 0.000 and 1.000 on all nine axes and all 423 seeds BY CONSTRUCTION, because `build_paired_split` defines the
naive pool as the trajectories carrying no held cell. They discriminate only where one trajectory holds several
clips.

What is measurable instead, from OakInk2's own annotation: on the `scene x primitive` axis a whole recording is
labelled by the FIRST primitive of its sequence. OakInk2 gives each primitive's frame span, so the share of a
recording's annotated frames that belong to the labelled primitive can be read directly. That share is the fraction
of scored motion the label actually names.

What this does NOT do
---------------------
- It covers OakInk2 only. OakInk-Image and GRAB have no within-recording annotation, so no comparable number exists
  for them; OakInk-Image's clips are single grasps by the dataset's own description, which is a description and not
  a measurement. **The paper must not present this as a measurement on all three datasets.**
- Spans are in the dataset's native frame index, not the 7.5 fps bundle's, so shares are of annotated native frames.
- It says nothing about whether the labelled primitive's motion differs from the unlabelled primitives' motion.
- Primitives of the same NAME later in a recording are counted separately as "same primitive name", because the
  label names a primitive type and not an occurrence.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from caredex.data.base import TrajectoryBundle

    b = TrajectoryBundle.load(str(ROOT / "data" / "bundles" / "oakink2_primseg.npz"))
    by_rec = defaultdict(list)
    for row in b.meta["segments"]:
        for prim, (lo, hi) in zip(row["primitives"], row["spans"]):
            by_rec[row["sequence"]].append((int(lo), int(hi), prim))

    first_share, name_share, n_prims, n_distinct = [], [], [], []
    for rec, spans in by_rec.items():
        spans.sort()
        total = sum(hi - lo for lo, hi, _ in spans)
        if total <= 0:
            continue
        first = spans[0]
        first_share.append((first[1] - first[0]) / total)
        name_share.append(sum(hi - lo for lo, hi, p in spans if p == first[2]) / total)
        n_prims.append(len(spans))
        n_distinct.append(len({p for _, _, p in spans}))

    def q(x):
        x = np.asarray(x, float)
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75))}

    out = {
        "what": "share of a recording's annotated frames that belong to the primitive its scene x primitive label names",
        "recordings": len(first_share),
        "first_primitive_share": q(first_share),
        "same_primitive_name_share": q(name_share),
        "primitives_per_recording": q(n_prims),
        "distinct_primitives_per_recording": q(n_distinct),
        "recordings_with_one_primitive": int(sum(n == 1 for n in n_prims)),
        "after_segmentation": "1.0 by construction: each segment is one primitive and carries that primitive's label",
    }
    (ROOT / "runs" / "label_coverage_oakink2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
