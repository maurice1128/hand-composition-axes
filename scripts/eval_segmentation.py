"""Do the learned primitives agree with the dataset's annotated primitives?

Until OakInk2 there was no ground truth to check against. The only available
test was whether a model's assignment switches coincided with fast motion --
a proxy that a model could satisfy by segmenting on velocity alone, learning
nothing about primitives. OakInk2 annotates named primitives with explicit
frame ranges, so the question becomes answerable directly.

Three measures, in increasing strength:

1. **Boundary F1** at a tolerance. Do switches land near true boundaries? A
   model that switches constantly scores high on recall and low on precision,
   which is why F1 rather than either alone.
2. **Label purity.** Within one true primitive segment, how concentrated is the
   model's assignment on a single index? A learned primitive that fires across
   half a segment and hands over mid-action is not tracking the primitive.
3. **Mutual information** between the true primitive label and the model's
   assignment, normalised. This is the one that cannot be gamed by
   over-segmenting: it asks whether knowing the model's index tells you which
   primitive is happening, independent of how many switches it makes.

None of these are trained for. The model never sees a primitive label.

    python scripts/eval_segmentation.py --run-dir runs/prior_oakink2_modular
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import normalize  # noqa: E402
from caredex.models.field_prior import FieldConfig, FieldPrimitivePrior  # noqa: E402
from caredex.models.modular_prior import ModularConfig, ModularPrimitivePrior  # noqa: E402
from caredex.train.checkpoint import CheckpointManager  # noqa: E402


def load_model(run_dir: Path, checkpoint: str, device: torch.device):
    mgr = CheckpointManager(run_dir)
    path = run_dir / checkpoint
    cfg = mgr.peek(path).get("config", {}).get("model", {})
    if "field_hidden" in cfg or "blend" in cfg:
        model = FieldPrimitivePrior(FieldConfig(**{
            k: v for k, v in cfg.items() if k in FieldConfig.__dataclass_fields__
        }))
    else:
        model = ModularPrimitivePrior(ModularConfig(**{
            k: v for k, v in cfg.items() if k in ModularConfig.__dataclass_fields__
        }))
    mgr.load(model, path=path, restore_rng=False)
    return model.to(device).eval()


def true_labels_for(bundle: TrajectoryBundle, i: int) -> np.ndarray | None:
    """Per-frame ground-truth primitive index for trajectory ``i``.

    The bundle records each trajectory's primitive chain and the mocap spans it
    was cut from; frames were taken from those spans in order, so the label
    sequence is reconstructible by repeating each primitive for the number of
    frames its span contributed.
    """
    segs = bundle.meta.get("segments")
    if not segs or i >= len(segs):
        return None
    row = segs[i]
    prims, spans = row.get("primitives"), row.get("spans")
    if not prims or not spans or len(prims) != len(spans):
        return None

    stride = max(int(bundle.meta.get("stride", 1)), 1)
    counts = [max(len(range(int(a), max(int(b), int(a) + 1), stride)), 0) for a, b in spans]
    total = sum(counts)
    if total != len(bundle.trajectories[i]):
        # Frame ids were not dense, so the reconstruction is not exact; refuse
        # rather than score against a misaligned label track.
        return None

    vocab = {p: k for k, p in enumerate(sorted(set(prims)))}
    out = np.concatenate([np.full(c, vocab[p], dtype=np.int64) for p, c in zip(prims, counts)])
    return out


def boundary_f1(pred_sw: np.ndarray, true_sw: np.ndarray, tol: int) -> tuple[float, float, float]:
    p_idx = np.flatnonzero(pred_sw)
    t_idx = np.flatnonzero(true_sw)
    if len(p_idx) == 0 or len(t_idx) == 0:
        return 0.0, 0.0, 0.0
    matched_p = sum(1 for p in p_idx if np.any(np.abs(t_idx - p) <= tol))
    matched_t = sum(1 for t in t_idx if np.any(np.abs(p_idx - t) <= tol))
    prec = matched_p / len(p_idx)
    rec = matched_t / len(t_idx)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def normalised_mi(true: np.ndarray, pred: np.ndarray) -> float:
    """Normalised mutual information, no sklearn dependency."""
    tu, ti = np.unique(true, return_inverse=True)
    pu, pi = np.unique(pred, return_inverse=True)
    if len(tu) < 2 or len(pu) < 2:
        return 0.0
    joint = np.zeros((len(tu), len(pu)))
    np.add.at(joint, (ti, pi), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return mi / max((hx * hy) ** 0.5, 1e-12)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--tolerance", type=int, default=4, help="boundary match tolerance in frames")
    ap.add_argument("--max-trajectories", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    model = load_model(run_dir, args.checkpoint, device)
    bundle = TrajectoryBundle.load(args.bundle)
    window = model.cfg.window

    scored, skipped = 0, 0
    precs, recs, f1s, purities, nmis = [], [], [], [], []
    rand_f1s = []
    rng = np.random.default_rng(0)

    for i, traj in enumerate(bundle.trajectories[: args.max_trajectories]):
        true = true_labels_for(bundle, i)
        if true is None or len(traj) < window + 1 or len(np.unique(true)) < 2:
            skipped += 1
            continue

        x = torch.from_numpy(normalize(traj)).unsqueeze(0).to(device)
        logits = model.encode(x)["logits"][0]
        pred = logits.argmax(-1).cpu().numpy()
        n = min(len(pred), len(true))
        pred, t = pred[:n], true[:n]

        p, r, f = boundary_f1(pred[1:] != pred[:-1], t[1:] != t[:-1], args.tolerance)
        precs.append(p); recs.append(r); f1s.append(f)

        # Chance baseline: same number of switches, placed at random. Without
        # it a high F1 could simply reflect switching often.
        n_sw = int((pred[1:] != pred[:-1]).sum())
        fake = np.zeros(n - 1, dtype=bool)
        if n_sw:
            fake[rng.choice(n - 1, min(n_sw, n - 1), replace=False)] = True
        rand_f1s.append(boundary_f1(fake, t[1:] != t[:-1], args.tolerance)[2])

        for lab in np.unique(t):
            seg = pred[t == lab]
            if len(seg):
                purities.append(np.bincount(seg).max() / len(seg))
        nmis.append(normalised_mi(t, pred))
        scored += 1

    if not scored:
        print("no trajectory could be scored: ground-truth label tracks did not align")
        return 1

    report = {
        "trajectories_scored": scored,
        "trajectories_skipped": skipped,
        "boundary_precision": float(np.mean(precs)),
        "boundary_recall": float(np.mean(recs)),
        "boundary_f1": float(np.mean(f1s)),
        "boundary_f1_chance": float(np.mean(rand_f1s)),
        "label_purity": float(np.mean(purities)),
        "normalised_mi": float(np.mean(nmis)),
        "tolerance_frames": args.tolerance,
    }

    print(f"scored {scored} trajectories ({skipped} skipped)\n")
    print(f"  boundary precision {report['boundary_precision']:.3f}")
    print(f"  boundary recall    {report['boundary_recall']:.3f}")
    print(f"  boundary F1        {report['boundary_f1']:.3f}  "
          f"(chance {report['boundary_f1_chance']:.3f})")
    print(f"  label purity       {report['label_purity']:.3f}")
    print(f"  normalised MI      {report['normalised_mi']:.3f}")
    print()
    lift = report["boundary_f1"] - report["boundary_f1_chance"]
    print("  " + (
        f"Boundaries beat chance by {lift:+.3f} F1."
        if lift > 0.05 else
        "Boundaries are at chance: the model's switches carry no information about "
        "where primitives actually change, whatever the F1 looks like in isolation."
    ))
    print("  " + (
        f"Assignments track primitive identity (NMI {report['normalised_mi']:.3f})."
        if report["normalised_mi"] > 0.15 else
        "Assignments are near-independent of primitive identity -- the bank has "
        "learned something, but not the annotated primitives."
    ))

    out = Path(args.out or run_dir / "segmentation.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
