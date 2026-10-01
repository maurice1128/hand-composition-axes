"""Motion-level contamination on a Table 1 axis: does holding out a LABEL hold out the MOTION on OakInk2?

Why. The paper's mechanism is that a compositional split is made on labels, so it holds out motion only when the
label describes what a trajectory contains. Section 3.4 measures that directly, but only on constructed OakInk-Image
bundles. The label-level diagnostics are 0 / 0 / 1 on every Table 1 axis by construction
(`scripts/axis_diagnostics.py`), so they cannot show it on a real axis. A referee asked for exactly that.

OakInk2 permits it. On `scene x primitive` a whole recording is labelled `<scene> x <FIRST primitive>`, but OakInk2
annotates every primitive's frame span, and the bundle is the concatenation of those spans. So for each seed's real
split one can ask, at the level of motion and not of labels:

  contaminated_recordings  share of the naive arm's training recordings that CONTAIN a held-out (scene, primitive)
                           somewhere after their first primitive - motion the split meant to withhold, present
                           under another label.
  contaminated_frames      the same, as a share of the naive arm's training frames.
  target_frame_purity      share of the target recordings' frames that belong to their own held-out primitive -
                           how much of what is scored is the held-out composition.

The splits are re-derived exactly as the sweep made them (same args, same seeds), as `axis_diagnostics.py` verified
is deterministic.

What this does NOT do
---------------------
- It does not show that the contaminating motion is what removes the penalty; it shows the contamination exists and
  how large it is. The causal test is the segmentation experiment of Section 3.4.
- "Contains the held-out primitive" is by primitive NAME within the same scene. Two occurrences of `grip` may differ;
  the label does not distinguish them, and neither does this.
- OakInk-Image and GRAB have no within-recording annotation, so no comparable number exists for them.
- Frames are counted in native span units (hi - lo), the same units the bundle's stride subsamples uniformly, so
  shares are unaffected by the stride.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = "oakink2_scene_primitive"


def main() -> int:
    import experiment_paired_composition as E
    from caredex.data.base import TrajectoryBundle

    res = json.loads((ROOT / "runs" / RUN / "results.json").read_text(encoding="utf-8"))
    args = res["args"]
    seeds = sorted({r["seed"] for r in res["results"] if r["kind"] == "perframe" and r["budget"] == 256})
    bundle = TrajectoryBundle.load(str(ROOT / args["bundle"].replace("\\", "/")))
    fine = list(bundle.labels)
    rows = bundle.meta["segments"]
    assert len(rows) == len(fine), "one span table per recording is required"
    # Per recording: its scene, and frames per primitive name, first primitive kept apart.
    scene = [lab.split("@")[1] for lab in fine]
    per = []
    for row in rows:
        spans = sorted(zip([int(a) for a, _ in row["spans"]], [int(b) for _, b in row["spans"]], row["primitives"]))
        d, first = {}, spans[0][2]
        for k, (a, b, p) in enumerate(spans):
            d.setdefault(p, [0, 0])[0 if k == 0 else 1] += max(b - a, 1)   # [as first, as later]
        per.append({"first": first, "frames": d, "total": sum(max(b - a, 1) for a, b, _ in spans)})

    bundle.labels = E.coarsen_labels(bundle.labels, args["granularity"])
    out_rows = []
    for seed in seeds:
        split = E.build_paired_split(bundle, args["held_compositions"], seed, fine_labels=fine,
                                     min_chains=args["min_chains"])
        naive, informed = E.sample_pools(split, 256, seed, bundle.labels, args["min_per_composition"])
        held = {tuple(h.split("->")) for h in split["held_compositions"]}      # (scene, primitive)
        # Sanity: the label-level rule really does exclude held cells from the naive arm.
        assert not any((scene[i], per[i]["first"]) in held for i in naive)

        def held_frames(i, later_only):
            return sum((v[1] if later_only else v[0] + v[1])
                       for p, v in per[i]["frames"].items() if (scene[i], p) in held)

        n_frames = sum(per[i]["total"] for i in naive)
        n_cont_rec = sum(held_frames(i, True) > 0 for i in naive)
        n_cont_frm = sum(held_frames(i, True) for i in naive)
        t_frames = sum(per[i]["total"] for i in split["target"])
        t_pure = sum(per[i]["frames"][per[i]["first"]][0] for i in split["target"])
        t_any = sum(held_frames(i, False) for i in split["target"])
        out_rows.append({"seed": seed,
                         "contaminated_recordings": n_cont_rec / len(naive),
                         "contaminated_frames": n_cont_frm / n_frames,
                         "target_frame_purity_first_span": t_pure / t_frames,
                         "target_frames_any_held_primitive": t_any / t_frames})

    def m(k):
        x = np.array([r[k] for r in out_rows])
        return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}

    out = {"run": RUN, "seeds": len(seeds), "budget": 256,
           "naive_label_contamination": 0.0,
           "naive_motion_contaminated_recordings": m("contaminated_recordings"),
           "naive_motion_contaminated_frames": m("contaminated_frames"),
           "target_frame_purity_first_span": m("target_frame_purity_first_span"),
           "target_frames_any_held_primitive": m("target_frames_any_held_primitive"),
           "after_segmentation": "0 and 1 by construction: a segment is one primitive under its own label",
           "per_seed": out_rows}
    (ROOT / "runs" / "motion_contamination_oakink2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
