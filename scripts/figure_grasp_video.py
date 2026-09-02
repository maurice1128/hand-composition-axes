"""Three hands, one object: real motion, a seen composition, an unseen one.

``figure_composition_video.py`` puts a seen and an unseen composition side by
side and they look alike -- which is the result, but it is a result an audience
cannot check, because "these two are indistinguishable" is a non-event to
watch. Adding the object gives a criterion anyone can apply without reading a
statistic: the object is still in the hand, or it is on the floor.

The first panel is a real held-out human trajectory driven through the same
retargeting onto the same hand. It is the reference the other two are asked to
match, and including it means the viewer is not asked to take the generated
motion's plausibility on trust.

What this does NOT show, and must not be captioned as showing: the prior is not
*trying* to grasp. It generates hand motion unconditionally; a plausible hand
motion happens to close the fingers, and an object resting in the palm is
retained as a consequence. There is no goal, no planner and no reward here.

Physics is on -- ``mj_step``, not ``mj_forward`` -- because retention is a
question about forces. The difficulty level is printed, and a single rendered
clip is not a success rate: quote the measured one beside it.

    python scripts/figure_grasp_video.py --run-dir runs/bricks_s4
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
from caredex.grasp_task import (DIFFICULTY, OBJECTS, _asset_dict,  # noqa: E402
                                _closed_fingertip_centre)
from caredex.hand_model import DOF_INDEX, LIMITS_HI, LIMITS_LO  # noqa: E402
from caredex.robot_hand import SHADOW_MAP  # noqa: E402

from demo_composition import load_model, observed_transitions  # noqa: E402
from experiment_robot_transfer import decode_pairs  # noqa: E402

#: The Menagerie hand ships no floor and no lights, so a render against it has
#: no shadow and the fingers flatten into an unreadable silhouette. MuJoCo's
#: offscreen buffer also defaults to 640x480 and refuses anything larger, which
#: is why the panel size is declared in the model rather than passed in.
SCENE = """
<mujoco model="grasp_view">
  <include file="right_hand.xml"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.33 0.33 0.36" diffuse="0.45 0.45 0.45"/>
    <quality shadowsize="4096" offsamples="8"/>
    <global offwidth="1280" offheight="1280"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.21 0.24"
             rgb2="0.24 0.25 0.28" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.45 -0.35 0.9" dir="-0.3 0.3 -1" directional="true"
           diffuse="0.75 0.75 0.75" specular="0.25 0.25 0.25" castshadow="true"/>
    <light pos="-0.25 0.45 0.7" dir="0.25 -0.45 -1" directional="true"
           diffuse="0.22 0.22 0.28" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.05" pos="0 0 -0.28" material="grid"/>
    <body name="obj" pos="{cx:.4f} {cy:.4f} {cz:.4f}">
      <freejoint name="obj_free"/>
      <geom name="obj_geom" type="{shape}" size="{dims}" mass="{mass}"
            rgba="0.85 0.42 0.25 1" friction="1.0 0.02 0.001"
            solref="0.004 1" condim="4"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(mujoco, hand_dir: Path, difficulty: str, shape: str):
    geom_type, size = OBJECTS[shape]
    cfg = DIFFICULTY[difficulty]
    assets = _asset_dict(hand_dir)
    cx, cy, cz = _closed_fingertip_centre(mujoco, hand_dir, assets)
    xml = SCENE.format(cx=cx, cy=cy, cz=cz, shape=geom_type,
                       dims=" ".join(str(s) for s in size), mass=cfg["mass"])
    m = mujoco.MjModel.from_xml_string(xml, assets)
    return m, mujoco.MjData(m), cfg


def run_and_render(mujoco, m, d, act_of_dof, q_deg, cfg, cam, opt, w, h):
    """Drive, hold, shake -- capturing a frame every few simulated milliseconds."""
    from PIL import Image

    obj_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
    obj_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    start = d.xpos[obj_b].copy()

    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    spf = max(1, int(round((1 / 30) / m.opt.timestep)))
    frames = []

    with mujoco.Renderer(m, h, w) as r:
        def shoot():
            r.update_scene(d, cam, opt)
            frames.append(Image.fromarray(r.render()))

        for frame in np.radians(np.asarray(q_deg, dtype=np.float64)):
            for dof, aid in act_of_dof.items():
                d.ctrl[aid] = np.clip(frame[DOF_INDEX[dof]], lo[aid], hi[aid])
            for _ in range(spf):
                mujoco.mj_step(m, d)
            shoot()

        for _ in range(8):
            for _ in range(int(0.04 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            shoot()

        g0 = m.opt.gravity.copy()
        for k in range(4):
            m.opt.gravity[:] = [cfg["shake_g"] * 9.81 * (1 if k % 2 == 0 else -1),
                                0.0, g0[2]]
            for _ in range(5):
                for _ in range(int(0.02 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                shoot()
        m.opt.gravity[:] = g0
        for _ in range(8):
            for _ in range(int(0.025 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            shoot()

    drop = float(np.linalg.norm(d.xpos[obj_b] - start))
    contacts = sum(1 for i in range(int(d.ncon))
                   if obj_g in (d.contact[i].geom1, d.contact[i].geom2))
    return frames, bool(drop < 0.10 and contacts > 0), drop


def label(img, title, sub, colour, verdict, tally, draw_text):
    from PIL import ImageDraw

    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, img.width, 50], fill=(15, 17, 21))
    dr.text((14, 8), title, fill=colour)
    dr.text((14, 28), sub, fill=(150, 155, 165))
    dr.text((img.width - 68, 8), draw_text, fill=(150, 155, 165))
    ok = verdict == "HELD"
    dr.rectangle([img.width - 96, img.height - 40, img.width - 12,
                  img.height - 12], fill=(24, 92, 54) if ok else (120, 38, 38))
    dr.text((img.width - 84, img.height - 32), verdict, fill=(235, 240, 245))
    # The running tally is the point of playing several draws: a viewer who sees
    # only the current clip cannot tell a lucky draw from a rate.
    dr.text((16, img.height - 30), f"held {tally}", fill=(150, 155, 165))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/bricks_s4")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--hand-dir", default=r"D:\datasets\mujoco_menagerie\shadow_hand")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-draws", type=int, default=5,
                    help="pairs played in sequence. One draw is not a result: "
                         "at easy difficulty roughly one clip in six fails in "
                         "both arms, so a single rendered success invites the "
                         "reader to mistake it for the rate. Several draws with "
                         "a running tally show the failure rate instead of "
                         "hiding it, and cost nothing but render time.")
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="runs/grasp_real_seen_unseen.gif")
    args = ap.parse_args()

    import mujoco
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior, _ = load_model(ROOT / args.run_dir, "best.pt", device)
    bundle = TrajectoryBundle.load(ROOT / args.bundle)
    k, win = prior.cfg.n_primitives, prior.cfg.window

    seen, _ = observed_transitions(prior, bundle, win, device)
    seen_list = sorted(seen)
    unseen = [(a, b) for a in range(k) for b in range(k)
              if a != b and (a, b) not in seen]
    if not seen_list or not unseen:
        print(f"cannot split: {len(seen_list)} seen, {len(unseen)} unseen")
        return 1

    rng = np.random.default_rng(args.seed)
    print(f"{k} primitives | {len(seen_list)} seen, {len(unseen)} unseen")

    hand_dir = Path(args.hand_dir)
    m, d, cfg = build(mujoco, hand_dir, args.difficulty, "box")

    # The Shadow Hand couples each finger's two distal joints onto one actuator,
    # so PIP flexion drives J0 and DIP is dropped rather than mapped to nothing.
    act_of_dof = {}
    for dof, joint in SHADOW_MAP.items():
        if dof.endswith("dip_flex"):
            continue
        name = joint.replace("rh_", "rh_A_")
        if dof.endswith("pip_flex"):
            name = name[:-1] + "0"
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            act_of_dof[dof] = aid

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    mujoco.mj_forward(m, d)
    cam.distance *= 0.5
    cam.azimuth, cam.elevation = 138.0, -18.0
    cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")]
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    # Each column accumulates every draw's frames, so the three play in step and
    # the tally in the corner grows as the viewer watches.
    columns: list[list] = [[], [], [], []]
    tally = {"RANDOM": [0, 0], "REAL": [0, 0], "SEEN": [0, 0], "UNSEEN": [0, 0]}

    for draw in range(args.n_draws):
        p_seen = seen_list[int(rng.integers(len(seen_list)))]
        p_unseen = unseen[int(rng.integers(len(unseen)))]
        clips = decode_pairs(prior, [p_seen, p_unseen], device)
        n_frames = clips.shape[1]

        # A real window sampled away from the trajectory start, so the reference
        # is mid-motion rather than a rest pose no grasp could come from.
        ti = int(rng.integers(len(bundle.trajectories)))
        traj = bundle.trajectories[ti]
        s = int(rng.integers(0, max(1, len(traj) - n_frames)))

        # Uniform inside the limit box, exactly the reference arm of
        # ``experiment_motion_validity.py`` (29.5 deg of coupling residual
        # against the generated motion's 5.9). It is here because a video of
        # three things that all work teaches a viewer nothing: without seeing
        # what invalid motion looks like, "valid" carries no information. Every
        # frame is drawn independently, so it also satisfies the joint limits --
        # which is the point of that experiment, and worth seeing.
        rand = rng.uniform(LIMITS_LO, LIMITS_HI, size=(n_frames, len(LIMITS_LO)))

        spec = [
            (rand, "RANDOM", "RANDOM",
             "uniform inside the joint limits", (200, 120, 130)),
            (traj[s:s + n_frames], "REAL", "REAL",
             "held-out human motion, retargeted", (190, 195, 205)),
            (clips[0], "SEEN", f"SEEN  {p_seen[0]} -> {p_seen[1]}",
             "primitive pair present in training", (120, 200, 255)),
            (clips[1], "UNSEEN", f"UNSEEN  {p_unseen[0]} -> {p_unseen[1]}",
             "primitive pair never in training", (255, 176, 102)),
        ]
        print(f"draw {draw + 1}/{args.n_draws}  seen {p_seen}  unseen {p_unseen}  "
              f"real traj {ti} [{s}:{s + n_frames}]")
        for col, (clip, key, title, sub, colour) in enumerate(spec):
            fr, held, drop = run_and_render(mujoco, m, d, act_of_dof, clip, cfg,
                                            cam, opt, args.width, args.height)
            tally[key][0] += int(held)
            tally[key][1] += 1
            verdict = "HELD" if held else "DROPPED"
            text = f"{tally[key][0]}/{tally[key][1]}"
            print(f"  {title:<24} {verdict:<8} drop={drop:.3f} m")
            columns[col].extend(
                label(f, title, sub, colour, verdict, text,
                      f"{draw + 1}/{args.n_draws}") for f in fr)

    n = min(len(c) for c in columns)
    gap = 6
    width_total = args.width * 4 + gap * 3
    out_frames = []
    for i in range(n):
        canvas = Image.new("RGB", (width_total, args.height), (15, 17, 21))
        for j, c in enumerate(columns):
            canvas.paste(c[i], (j * (args.width + gap), 0))
        out_frames.append(canvas)

    print("\ntally  " + "   ".join(f"{k} {v[0]}/{v[1]}" for k, v in tally.items()))

    out = ROOT / args.out
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    print(f"wrote {out}  {len(out_frames)} frames  {out_frames[0].size}")
    print(f"difficulty '{args.difficulty}' -- one rendered clip is not a success "
          f"rate. Quote the measured rate beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
