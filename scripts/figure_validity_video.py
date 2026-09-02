"""What invalid hand motion looks like, next to generated and real motion.

Every other clip in this project shows things that work, and three working
hands side by side teach a viewer nothing: without seeing what invalid motion
looks like, "valid" carries no information. This puts the paper's own reference
arm on screen -- joint angles drawn uniformly inside the limit box -- beside the
prior's output and a real held-out human trajectory.

The number in each panel is the DIP/PIP coupling residual, the check with teeth
in this project. Joint limits are satisfied by construction in every arm here,
uniform noise included, so limit satisfaction proves nothing; the distal joints
of a real finger move together, and whether a motion respects that is what
separates a hand from a set of angles.

    uniform noise  29.5 deg      generated  5.9 deg      real  8.4 deg

Read the comparison honestly. Per-frame uniform noise is a deliberately weak
reference: it is not a competing method, and beating it is a floor rather than
an achievement. It earns its place by making the floor visible, not by being
hard to beat -- which is why the paper quotes the effect size against it and
draws its real comparisons against the PCA and monolithic baselines instead.

    python scripts/figure_validity_video.py --run-dir runs/bricks_s4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import (LIMITS_HI, LIMITS_LO, N_DOF,  # noqa: E402
                                coupling_residual, denormalize)

from demo_composition import load_model  # noqa: E402
from figure_composition_video import SCENE, assets  # noqa: E402
from figure_rollout_video import qpos_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/bricks_s4")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--hand-dir", default=r"D:\datasets\mujoco_menagerie\shadow_hand")
    ap.add_argument("--n-draws", type=int, default=3,
                    help="clips played in sequence, so one lucky draw cannot "
                         "stand in for the distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="runs/validity_random_vs_generated.gif")
    args = ap.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior, _ = load_model(ROOT / args.run_dir, "best.pt", device)
    bundle = TrajectoryBundle.load(ROOT / args.bundle)
    win = prior.cfg.window
    rng = np.random.default_rng(args.seed)

    hand_dir = Path(args.hand_dir)
    m = mujoco.MjModel.from_xml_string(SCENE, assets(hand_dir))
    d = mujoco.MjData(m)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    mujoco.mj_forward(m, d)
    cam.distance *= 0.6
    cam.azimuth, cam.elevation = 135.0, -20.0
    cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")]
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    def render(clip):
        qpos = qpos_of(mujoco, m, clip)
        frames = []
        with mujoco.Renderer(m, args.height, args.width) as r:
            for frame in qpos:
                d.qpos[:] = frame
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam, opt)
                frames.append(Image.fromarray(r.render()))
        return frames

    columns: list[list] = [[], [], []]
    residuals: list[list[float]] = [[], [], []]

    for draw in range(args.n_draws):
        rand = rng.uniform(LIMITS_LO, LIMITS_HI, size=(win, len(LIMITS_LO)))

        torch.manual_seed(args.seed + draw)
        with torch.no_grad():
            z = prior.sample(1, temperature=1.0, device=device)
        gen = denormalize(z[0].cpu().numpy()[:, :N_DOF])

        ti = int(rng.integers(len(bundle.trajectories)))
        traj = bundle.trajectories[ti]
        s = int(rng.integers(0, max(1, len(traj) - win)))
        real = traj[s:s + win]
        if len(real) < win:
            real = np.vstack([real, np.repeat(real[-1:], win - len(real), axis=0)])

        spec = [("UNIFORM NOISE", "angles drawn inside the joint limits",
                 (220, 120, 130), rand),
                ("GENERATED", "sampled from the primitive prior",
                 (120, 200, 255), gen),
                ("REAL", "held-out human motion", (190, 195, 205), real)]
        print(f"draw {draw + 1}/{args.n_draws}  real traj {ti} [{s}:{s + win}]")
        for col, (title, sub, colour, clip) in enumerate(spec):
            res = float(np.mean(coupling_residual(clip)))
            residuals[col].append(res)
            print(f"  {title:<15} coupling residual {res:6.2f} deg")
            for f in render(clip):
                dr = ImageDraw.Draw(f)
                dr.rectangle([0, 0, f.width, 50], fill=(15, 17, 21))
                dr.text((14, 8), title, fill=colour)
                dr.text((14, 28), sub, fill=(150, 155, 165))
                dr.rectangle([12, f.height - 46, 214, f.height - 12],
                             fill=(28, 32, 40))
                dr.text((22, f.height - 39), "DIP/PIP coupling error",
                        fill=(150, 155, 165))
                dr.text((22, f.height - 25), f"{res:.1f} deg", fill=colour)
                columns[col].append(f)

    n = min(len(c) for c in columns)
    gap = 6
    W = args.width * 3 + gap * 2
    out_frames = []
    for i in range(n):
        canvas = Image.new("RGB", (W, args.height), (15, 17, 21))
        for j, c in enumerate(columns):
            canvas.paste(c[i], (j * (args.width + gap), 0))
        out_frames.append(canvas)

    out = ROOT / args.out
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    print(f"\nmean coupling residual over {args.n_draws} draws: "
          + "   ".join(f"{t} {np.mean(r):.1f}"
                       for t, r in zip(("noise", "generated", "real"), residuals)))
    print(f"wrote {out}  {len(out_frames)} frames  {out_frames[0].size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
