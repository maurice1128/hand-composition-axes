"""Does modularity help the *task*, not just the reconstruction?

Every modularity result in this project is measured in reconstruction error.
The grasp experiment then showed that a modular prior's unseen compositions are
retained as well as its seen ones -- but that comparison lives entirely inside
the modular prior. It never asked whether modularity buys anything a
bandwidth-matched monolithic prior does not, on a task.

That is the question a reviewer asks first, and leaving it unanswered would let
the paper claim modularity matters while only ever having tested it on MSE.

Design. Both priors are trained identically and sampled identically; the only
difference is the architecture. Their generated motions are executed on the same
MuJoCo scene at the same difficulty, with the same positive and negative
controls in front. The controls come first for the same reason as before -- a
setting whose full fist cannot hold the object measures nothing, and at the
"hard" level both arms floor at ~0%, where a comparison is meaningless.

Sampling is unconditional for both: the monolithic prior has no primitive index
to condition on, so asking it for "pair (a, b)" is not defined. Both are asked
for the same number of samples at the same temperature, which is the only
comparison the two architectures both admit.

    python scripts/experiment_grasp_modular_vs_mono.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.grasp_task import GraspTask  # noqa: E402
from caredex.hand_model import DOF_NAMES, LIMITS_HI, N_DOF, denormalize  # noqa: E402

from experiment_data_efficiency import build_model  # noqa: E402
from experiment_grasp_success import scripted_fist  # noqa: E402


@torch.no_grad()
def sample_clips(model, n: int, device, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    out = []
    for _ in range(n):
        z = model.sample(1, temperature=1.0, device=device)
        out.append(z[0].cpu().numpy()[:, :N_DOF])
    return denormalize(np.stack(out))


def load_trained(run_dir: Path, kind: str, device):
    """Load a checkpoint into the architecture it was trained as."""
    from caredex.train.checkpoint import CheckpointManager

    mgr = CheckpointManager(run_dir)
    path = run_dir / "best.pt"
    cfg = mgr.peek(path).get("config", {}).get("model", {})
    model = build_model(kind, cfg.get("window", 32), cfg.get("latent_dim", 12),
                        cfg.get("hidden_dim", 256), cfg.get("n_primitives", 16))
    mgr.load(model, path=path, restore_rng=False)
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modular", nargs="*",
                    default=[f"runs/bricks_s{i}" for i in (1, 2, 3, 4)])
    ap.add_argument("--mono", nargs="*", default=None,
                    help="perframe run dirs. Defaults to the paired sweep's "
                         "baseline arms, which were trained on the same data.")
    ap.add_argument("--difficulty", default="medium",
                    help="'medium' is the only level that discriminates: easy "
                         "ceilings near 87%% and hard floors near 0%%.")
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/grasp_modular_vs_mono.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = GraspTask("box", args.difficulty)

    fist = task.run(scripted_fist(32))
    openh = task.run(np.zeros((32, len(DOF_NAMES))))
    print(f"controls  fist held={fist.held} drop={fist.drop_m:.3f}  |  "
          f"open held={openh.held} drop={openh.drop_m:.3f}")
    if not (fist.held and not openh.held):
        print("STOP: controls do not separate at this difficulty.")
        return 1

    report: dict = {"difficulty": args.difficulty, "n_clips": args.n_clips,
                    "controls": {"fist_held": fist.held, "open_held": openh.held}}
    arms: dict[str, list[float]] = {"modular": [], "monolithic": []}

    mono_dirs = args.mono or sorted(
        str(p.parent.relative_to(ROOT))
        for p in (ROOT / "runs").glob("*/s*_perframe_b256_naive/best.pt"))

    for label, dirs, kind in (("modular", args.modular, "modular"),
                              ("monolithic", mono_dirs, "perframe")):
        for i, d in enumerate(dirs):
            p = ROOT / d
            if not (p / "best.pt").exists():
                continue
            model = load_trained(p, kind, device)
            clips = sample_clips(model, args.n_clips, device, seed=args.seed + i)
            held = [task.run(c).held for c in clips]
            arms[label].extend(float(h) for h in held)
            print(f"  {label:<11} {p.name:<16} {np.mean(held):.1%}")

    if not arms["monolithic"]:
        print("\nno monolithic checkpoints found -- the paired sweeps delete their "
              "\nweights after scoring. Train one perframe model on the same bundle "
              "\nand pass it with --mono.")
        return 1

    m, o = np.array(arms["modular"]), np.array(arms["monolithic"])
    p = stats.fisher_exact([[int(m.sum()), len(m) - int(m.sum())],
                            [int(o.sum()), len(o) - int(o.sum())]])[1]
    report.update(modular_rate=float(m.mean()), monolithic_rate=float(o.mean()),
                  n_modular=len(m), n_monolithic=len(o), p_fisher=float(p))
    print(f"\nmodular {m.mean():.1%} (n={len(m)})   monolithic {o.mean():.1%} "
          f"(n={len(o)})   Fisher p = {p:.4f}")
    print("\nThis is the first place modularity is tested on a task rather than on "
          "\nreconstruction error. A null means modularity's measured advantage "
          "\ndoes not reach grasp retention, which is worth reporting as plainly "
          "\nas a positive would be.")

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
