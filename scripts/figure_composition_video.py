"""Side-by-side video: a composition the prior saw, and one it never did.

Every other asset in this project shows a number. This shows the claim. Two
clips play together -- left, a primitive pair that occurs in training; right, a
pair that never does -- on the same robot hand, from the same camera, at the
same speed. If the two look equally like hand motion, the paper's central
result needs no statistics to be understood, and if they do not, that is worth
seeing too.

Rendered on the Shadow Hand rather than the stick figure because the stick
figure cannot show whether a pose is *plausible*, only whether it is legal.
The scene adds a floor and two lights to the Menagerie model: the bare hand XML
renders against nothing, and without a ground plane there is no shadow and no
depth cue.

Nothing here is cherry-picked beyond the pair indices, which are drawn by seed
from the seen and unseen pools and printed so a different draw can be checked.

    python scripts/figure_composition_video.py --run-dir runs/bricks_s4
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
from caredex.robot_hand import SHADOW_MAP, ShadowHand  # noqa: E402

from demo_composition import load_model, observed_transitions  # noqa: E402
from experiment_robot_transfer import decode_pairs  # noqa: E402

#: Hand plus a ground plane and two lights. The Menagerie hand ships neither,
#: and a render with no floor has no shadow, which flattens the fingers into an
#: unreadable silhouette.
SCENE = """
<mujoco model="composition_view">
  <include file="right_hand.xml"/>
  <visual>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096" offsamples="8"/>
    <!-- MuJoCo's offscreen buffer defaults to 640x480 and refuses any larger
         render, so the panel size is declared here rather than passed in. -->
    <global offwidth="1280" offheight="1280"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.23 0.26"
             rgb2="0.26 0.27 0.30" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.3 0.8" dir="-0.3 0.3 -1" directional="true"
           diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="-0.2 0.4 0.6" dir="0.2 -0.4 -1" directional="true"
           diffuse="0.25 0.25 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.05" pos="0 0 -0.12"
          material="grid"/>
  </worldbody>
</mujoco>
"""


def assets(hand_dir: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for p in list(hand_dir.glob("*.xml")) + list((hand_dir / "assets").rglob("*")):
        if p.is_file():
            out[p.name if p.parent == hand_dir else f"assets/{p.name}"] = p.read_bytes()
    return out


def render_clip(mujoco, model, data, qpos, width, height, cam, opt) -> list:
    from PIL import Image

    frames = []
    with mujoco.Renderer(model, height, width) as r:
        for frame in qpos:
            data.qpos[: len(frame)] = frame
            mujoco.mj_forward(model, data)
            r.update_scene(data, cam, opt)
            frames.append(Image.fromarray(r.render()))
    return frames


def label(img, text: str, sub: str, colour: tuple[int, int, int]):
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 46], fill=(17, 19, 24))
    d.text((14, 8), text, fill=colour)
    d.text((14, 26), sub, fill=(150, 155, 165))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/bricks_s4")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--hand-dir", default=r"D:\datasets\mujoco_menagerie\shadow_hand")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="runs/composition_seen_vs_unseen.gif")
    args = ap.parse_args()

    import mujoco
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prior, _ = load_model(ROOT / args.run_dir, "best.pt", device)
    bundle = TrajectoryBundle.load(ROOT / args.bundle)
    k = model_prior.cfg.n_primitives

    seen, _ = observed_transitions(model_prior, bundle, model_prior.cfg.window, device)
    seen_list = sorted(seen)
    unseen = [(a, b) for a in range(k) for b in range(k)
              if a != b and (a, b) not in seen]
    if not seen_list or not unseen:
        print(f"cannot split: {len(seen_list)} seen, {len(unseen)} unseen")
        return 1

    rng = np.random.default_rng(args.seed)
    pair_seen = seen_list[int(rng.integers(len(seen_list)))]
    pair_unseen = unseen[int(rng.integers(len(unseen)))]
    print(f"{k} primitives | {len(seen_list)} seen, {len(unseen)} unseen")
    print(f"  seen   pair {pair_seen}")
    print(f"  unseen pair {pair_unseen}")

    clips = decode_pairs(model_prior, [pair_seen, pair_unseen], device)

    hand_dir = Path(args.hand_dir)
    m = mujoco.MjModel.from_xml_string(SCENE, assets(hand_dir))
    d = mujoco.MjData(m)

    # Joint angles are written straight into qpos: this is a kinematic view, not
    # a simulation, so the fingers show exactly what the prior produced rather
    # than what a controller could track.
    robot = ShadowHand()
    addr = {dof: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for dof, j in SHADOW_MAP.items()
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0}

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance *= 0.62
    cam.azimuth, cam.elevation = 135, -22
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    mujoco.mj_forward(m, d)
    cam.lookat[:] = d.xpos[palm]

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    panels = []
    for clip, (title, sub, colour) in zip(
        clips,
        [(f"SEEN  {pair_seen[0]} -> {pair_seen[1]}",
          "this pair occurs in training", (120, 200, 255)),
         (f"UNSEEN  {pair_unseen[0]} -> {pair_unseen[1]}",
          "this pair never occurs in training", (255, 176, 102))],
    ):
        qpos = np.zeros((len(clip), m.nq))
        rad = np.radians(clip)
        from caredex.hand_model import DOF_INDEX
        for dof, a in addr.items():
            lo, hi = m.jnt_range[[i for i in range(m.njnt)
                                 if m.jnt_qposadr[i] == a][0]]
            qpos[:, a] = np.clip(rad[:, DOF_INDEX[dof]], lo, hi)
        fr = render_clip(mujoco, m, d, qpos, args.width, args.height, cam, opt)
        panels.append([label(f, title, sub, colour) for f in fr])

    n = min(len(panels[0]), len(panels[1]))
    out_frames = []
    for i in range(n):
        canvas = Image.new("RGB", (args.width * 2 + 6, args.height), (17, 19, 24))
        canvas.paste(panels[0][i], (0, 0))
        canvas.paste(panels[1][i], (args.width + 6, 0))
        out_frames.append(canvas)

    out = ROOT / args.out
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    print(f"wrote {out}  {len(out_frames)} frames  {out_frames[0].size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
