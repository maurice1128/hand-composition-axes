"""One long generated motion, with every unrehearsed junction marked as it passes.

The side-by-side composition clip shows one pair at a time and asks the viewer
to notice that two things look alike, which is a weak thing to ask. This asks
for something stronger and easier: watch a single continuous motion and see
whether it ever stumbles. A banner marks each transition the prior never saw in
training, and 168 of the 240 possible transitions are of that kind, so most of
what plays has no precedent in the data.

If the motion runs through those junctions without a hitch, the compositional
claim has been shown rather than tabulated. If it stutters exactly there, that
is the more interesting outcome and this is how it would become visible.

A real held-out human trajectory runs beside it at the same length and speed.
Generated motion is *smoother* than real motion here -- 5.94 degrees of coupling
residual against 8.40 -- and without the reference an audience has no way to
judge whether the smoothness reads as natural or as synthetic.

Sampling is at temperature 0, so the motion is the prior's mean path for that
sequence rather than a lucky draw; the sequence itself is drawn by seed and
printed.

    python scripts/figure_rollout_video.py --run-dir runs/bricks_s4
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
from caredex.hand_model import DOF_INDEX  # noqa: E402
from caredex.robot_hand import SHADOW_MAP  # noqa: E402

from demo_composition import load_model, observed_transitions  # noqa: E402
from figure_composition_video import SCENE, assets  # noqa: E402
from figure_primitive_library import rollout  # noqa: E402


def qpos_of(mujoco, m, clip: np.ndarray) -> np.ndarray:
    """Anatomical degrees to the hand's configuration vector, clipped to range."""
    qpos = np.zeros((len(clip), m.nq))
    rad = np.radians(clip)
    range_of = {int(m.jnt_qposadr[i]): m.jnt_range[i] for i in range(m.njnt)}
    for dof, joint in SHADOW_MAP.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            continue
        a = int(m.jnt_qposadr[jid])
        lo, hi = range_of[a]
        qpos[:, a] = np.clip(rad[:, DOF_INDEX[dof]], lo, hi)
    return qpos


def strip(dr, x0, y0, width, height, seq, unseen_at, pos):
    """A progress bar of the primitive sequence, unrehearsed junctions in amber."""
    n = len(seq)
    seg = width / n
    for i in range(n):
        x = x0 + i * seg
        done = pos >= (i + 1) / n
        dr.rectangle([x + 1, y0, x + seg - 1, y0 + height],
                     fill=(70, 78, 92) if done else (38, 42, 50))
        if i in unseen_at:
            dr.rectangle([x + 1, y0, x + 3, y0 + height], fill=(255, 176, 102))
    dr.rectangle([x0 + width * pos - 1, y0 - 3, x0 + width * pos + 1, y0 + height + 3],
                 fill=(235, 240, 245))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/bricks_s4")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--hand-dir", default=r"D:\datasets\mujoco_menagerie\shadow_hand")
    ap.add_argument("--n-primitives", type=int, default=10,
                    help="how many primitives to chain into one motion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=620)
    ap.add_argument("--height", type=int, default=620)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="runs/rollout_generated_vs_real.gif")
    args = ap.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior, _ = load_model(ROOT / args.run_dir, "best.pt", device)
    bundle = TrajectoryBundle.load(ROOT / args.bundle)
    k, win = prior.cfg.n_primitives, prior.cfg.window

    seen, _ = observed_transitions(prior, bundle, win, device)
    rng = np.random.default_rng(args.seed)

    # Drawn uniformly, not chosen to maximise unseen junctions: with 168 of 240
    # transitions absent from training, a uniform draw is already mostly
    # unrehearsed, and picking for it would make the banner count meaningless.
    seq = [int(rng.integers(k))]
    while len(seq) < args.n_primitives:
        nxt = int(rng.integers(k))
        if nxt != seq[-1]:
            seq.append(nxt)
    unseen_at = {i for i in range(1, len(seq)) if (seq[i - 1], seq[i]) not in seen}
    print(f"{k} primitives | sequence {seq}")
    print(f"  {len(unseen_at)} of {len(seq) - 1} junctions never seen in training: "
          f"{sorted((seq[i - 1], seq[i]) for i in unseen_at)}")

    gen = rollout(prior, seq, device)
    n_frames = len(gen)
    seg = n_frames / len(seq)

    ti = int(rng.integers(len(bundle.trajectories)))
    traj = bundle.trajectories[ti]
    s = int(rng.integers(0, max(1, len(traj) - n_frames)))
    real = traj[s:s + n_frames]
    if len(real) < n_frames:                      # short trajectory: hold the end
        real = np.vstack([real, np.repeat(real[-1:], n_frames - len(real), axis=0)])
    print(f"  generated {n_frames} frames | real traj {ti} [{s}:{s + n_frames}]")

    hand_dir = Path(args.hand_dir)
    m = mujoco.MjModel.from_xml_string(SCENE, assets(hand_dir))
    d = mujoco.MjData(m)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    mujoco.mj_forward(m, d)
    cam.distance *= 0.62
    cam.azimuth, cam.elevation = 135.0, -20.0
    cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")]
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    panels = []
    for clip in (gen, real):
        qpos = qpos_of(mujoco, m, clip)
        frames = []
        with mujoco.Renderer(m, args.height, args.width) as r:
            for frame in qpos:
                d.qpos[:] = frame
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam, opt)
                frames.append(Image.fromarray(r.render()))
        panels.append(frames)

    gap, W = 6, args.width * 2 + 6
    out_frames = []
    for i in range(n_frames):
        canvas = Image.new("RGB", (W, args.height + 46), (15, 17, 21))
        canvas.paste(panels[0][i], (0, 0))
        canvas.paste(panels[1][i], (args.width + gap, 0))
        dr = ImageDraw.Draw(canvas)

        dr.rectangle([0, 0, W, 46], fill=(15, 17, 21))
        which = min(len(seq) - 1, int(i / seg))
        dr.text((14, 8), f"GENERATED   primitive {seq[which]}", fill=(120, 200, 255))
        dr.text((14, 26), f"chained from {len(seq)} primitives, temperature 0",
                fill=(150, 155, 165))
        dr.text((args.width + gap + 14, 8), "REAL", fill=(190, 195, 205))
        dr.text((args.width + gap + 14, 26), "held-out human motion, same length",
                fill=(150, 155, 165))

        # The banner runs for the first third of a segment entered by a junction
        # the prior never trained on -- long enough to read, short enough that
        # the viewer sees the motion continue past it.
        if which in unseen_at and (i - which * seg) < seg / 3:
            dr.rectangle([12, args.height - 52, 320, args.height - 16],
                         fill=(150, 96, 24))
            dr.text((26, args.height - 42),
                    f"UNSEEN JUNCTION  {seq[which - 1]} -> {seq[which]}",
                    fill=(255, 240, 220))

        strip(dr, 14, args.height + 18, W - 28, 10, seq, unseen_at,
              min(1.0, (i + 1) / n_frames))
        out_frames.append(canvas)

    out = ROOT / args.out
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    print(f"wrote {out}  {len(out_frames)} frames  {out_frames[0].size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
