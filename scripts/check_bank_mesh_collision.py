"""Does a bank's generated motion really self-intersect, or is it the proxy?

`composition.json` reports an interpenetration rate from `kinematics.py`'s
capsule model, which `ergonomics.py` labels a proxy in its own docstring. On the
OakInk-Image banks that proxy reads 0.89 against 0.014 on the OakInk2 banks --
a sixtyfold gap that would change what can be claimed about the newer banks, if
it is real. Capsules are generous about adjacent fingers touching, so it may not
be.

This runs the real check: skin the MANO mesh at each generated pose and test
triangles between digits. Both dataset families are measured the same way, so
the comparison stands whatever the absolute rate turns out to be.

    python scripts/check_bank_mesh_collision.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.hand_model import N_DOF, denormalize  # noqa: E402
from caredex.mano import load_mano  # noqa: E402
from caredex.mesh_collision import (MeshSelfIntersection,  # noqa: E402
                                    dof_to_mano_rotations, pose_mesh)

from demo_composition import load_model  # noqa: E402


@torch.no_grad()
def sample(run_dir: Path, n: int, device, seed: int) -> np.ndarray:
    model, _ = load_model(run_dir, "best.pt", device)
    torch.manual_seed(seed)
    out = [model.sample(1, temperature=1.0, device=device)[0].cpu().numpy()[:, :N_DOF]
           for _ in range(n)]
    return denormalize(np.stack(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["runs/oakbricks_s1", "runs/oakbricks_s2",
                                                  "runs/bricks_s1", "runs/bricks_s2"])
    ap.add_argument("--mano", default=r"D:\datasets\mano\MANO_RIGHT.pkl")
    ap.add_argument("--clips", type=int, default=6,
                    help="per run; each is a 32-frame window, so 6 is ~190 poses")
    ap.add_argument("--stride", type=int, default=4,
                    help="test every Nth frame -- the mesh test is not cheap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/bank_mesh_collision.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mano(args.mano)
    checker = MeshSelfIntersection(model)

    print(f"{'run':<20}{'poses':>7}{'mesh':>9}{'capsule':>10}")
    print("-" * 46)
    report: dict = {}

    for name in args.runs:
        d = ROOT / name
        if not (d / "best.pt").exists():
            print(f"{name:<20}  no checkpoint")
            continue
        clips = sample(d, args.clips, device, args.seed)
        poses = clips.reshape(-1, clips.shape[-1])[:: args.stride]

        hits = 0
        for q in poses:
            verts = pose_mesh(model, dot_rot(q))
            hits += int(checker.check(verts).intersects)
        mesh_rate = hits / len(poses)

        cap = None
        comp = d / "composition.json"
        if comp.exists():
            blob = json.loads(comp.read_text(encoding="utf-8"))
            cap = (blob.get("validity_unseen") or {}).get("interpenetration_rate")

        print(f"{name.split('/')[-1]:<20}{len(poses):>7}{mesh_rate:>8.1%}"
              + (f"{cap:>10.1%}" if cap is not None else f"{'-':>10}"))
        report[name] = {"n_poses": len(poses), "mesh_rate": mesh_rate,
                        "capsule_rate": cap}

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    print("The capsule column is the number `composition.json` carries. Where the "
          "\ntwo disagree, the mesh column is the one to quote.")
    return 0


def dot_rot(q_deg: np.ndarray) -> np.ndarray:
    """27 anatomical DOF -> the (16, 3, 3) MANO rotations `pose_mesh` wants."""
    return dof_to_mano_rotations(q_deg)


if __name__ == "__main__":
    raise SystemExit(main())
