"""Side-by-side video of one stage-F held-out episode: keyframe vs continuous.

Stage F (``docs/RL_RESULTS.md`` section 6, ``docs/PREREG_rl_action_space.md``
"Stage F"): naive PPO policies trained on ``Hx-R, Hy-R, Hx-Hy, Hy-Hx`` for
1.5 M frames, evaluated deterministically on the held-out triples
``Hx-Hy-R, Hy-Hx-R``, 100 episodes, eval seed 10 000. Held-out all-stage
success over 20 RL seeds: keyframe 0.59, continuous 0.22.

This script replays ONE of those evaluation episodes for both arms of ONE RL
seed and renders them side by side. The episode is reproduced exactly: the
environment is rebuilt with the arguments ``scripts/rl_stage_breakdown.py``
uses for stage F (the ones the training script used, as recorded in each run's
``eval.json``), and the evaluation's reset draws are regenerated from the eval
seed for 100 environments and the chosen episode's draws handed to a
single-environment copy of the task. Physics and decoders are per-environment,
so episode ``i`` of the 100-environment evaluation and this one-environment
replay face the same initial state, sequence and tilt sign. The replayed
outcome is checked against the run's ``eval.json`` record and printed.

Selection rule (so a caption can say how the clip was chosen):
  * seed: the RL seed whose keyframe and continuous held-out success are
    jointly closest to the 20-seed arm means (``--seed`` overrides);
  * episode: the first evaluation episode (lowest index, among the first 50)
    in which keyframe succeeds and continuous fails; if none, episode 0
    (``--episode`` overrides).

The environment has no render mode, so each frame's physics state is copied
into ``MjData`` of a render-only copy of the same scene (identical bodies and
joints, plus lights and a nicer floor texture) and drawn with
``mujoco.Renderer``. After a policy's episode ends (the cube was lost during a
hold) its panel freezes on the terminal frame; nothing is simulated beyond
what the evaluation simulated.

    OMP_NUM_THREADS=2 python scripts/render_stageF_video.py
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

from caredex.grasp_task import OBJECTS, _asset_dict, _closed_fingertip_centre, scene_xml  # noqa: E402
from caredex.rl.staged_env import SKILLS, StagedRetainVecEnv  # noqa: E402
from caredex.rl.vec_env import DEFAULT_HAND_DIR, _sensor_block, parse_difficulty  # noqa: E402

SWEEP = ROOT / "runs/rl/stageF_staged"
PRIORS = {"keyframe": "runs/oakkeyframe_s1", "continuous": "runs/oakperframe_s1"}
TITLES = {"keyframe": "keyframe (modular), 1.5 M frames",
          "continuous": "continuous latent, 1.5 M frames"}
SKILL_TEXT = {"Hx": "hold, tilt x", "Hy": "hold, tilt y", "R": "release"}
N_EVAL = 100            # evaluation episodes = environments in the stage-F eval


class _ReplayRng:
    """Stand-in for the env's rng: serves precomputed draws, then a real rng."""

    def __init__(self, draws: list[np.ndarray], fallback_seed: int):
        self.q = list(draws)
        self.fallback = np.random.default_rng(fallback_seed)

    def _next(self, name, *a, **k):
        if self.q:
            return self.q.pop(0)
        return getattr(self.fallback, name)(*a, **k)

    def integers(self, *a, **k):
        return self._next("integers", *a, **k)

    def uniform(self, *a, **k):
        return self._next("uniform", *a, **k)

    def choice(self, *a, **k):
        return self._next("choice", *a, **k)


class _Replay(StagedRetainVecEnv):
    """Keep the terminal physics state, which ``step_wait`` overwrites on reset."""

    def _reset_env(self, idx):
        self.pre_reset_state = self.states.copy()
        super()._reset_env(idx)


def make_env(arm: str, eval_sequences: str, eval_seed: int) -> _Replay:
    # Stage-F arguments, as in scripts/rl_stage_breakdown.py and the runs' eval.json.
    return _Replay(arm, ROOT / PRIORS[arm], num_envs=1, nthread=1, seed=eval_seed,
                   init_poses=ROOT / "data/bundles/oakink_first_frames.npy",
                   shape="box", difficulty="m0.20g1.5", rolling_friction=None,
                   sequences=eval_sequences, stage_frames=40, max_stages=3,
                   pose_residual_deg=15.0, macro_every=16, residual_mode="rate",
                   residual_rate_deg=3.0, perturb_mode="tilt", reward_mode="sparse",
                   release_reward="graded",
                   stepper_kw={"untrained": False, "untrained_seed": 0})


def eval_draws(n_bank: int, n_seq: int, eval_seed: int, episode: int) -> list[np.ndarray]:
    """The reset draws of the 100-env evaluation, restricted to one episode.

    Same calls, same order, same sizes as RetainVecEnv._reset_env followed by
    StagedRetainVecEnv._reset_env on ``np.arange(100)``.
    """
    rng = np.random.default_rng(eval_seed)
    k = rng.integers(0, n_bank, size=N_EVAL)
    ang = rng.uniform(0, 2 * np.pi, N_EVAL)
    seq = rng.integers(0, n_seq, size=N_EVAL)
    sign = rng.choice([-1.0, 1.0], size=N_EVAL)
    e = [episode]
    return [k[e], ang[e], seq[e], sign[e]]


def rollout(arm: str, seed: int, episode: int, eval_sequences: str, eval_seed: int) -> dict:
    from stable_baselines3 import PPO

    env = make_env(arm, eval_sequences, eval_seed)
    env.rng = _ReplayRng(eval_draws(len(env.bank_states), len(env.sequences), eval_seed, episode),
                         fallback_seed=eval_seed)
    model = PPO.load(str(SWEEP / f"{arm}_naive_s{seed}" / "policy.zip"), device="cpu")
    obs = env.reset()
    states = [env.states[0].copy()]
    skills = [-1]
    held = [True]
    while True:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(act)
        info = infos[0]
        if dones[0]:
            states.append(env.pre_reset_state[0].copy())
        else:
            states.append(env.states[0].copy())
        skills.append(info["skill"])
        held.append(info["held"])
        if dones[0]:
            out = {"states": np.array(states), "skills": skills, "held": held,
                   "success": bool(info["success"]), "frames": int(info["frames"]),
                   "stages_ok": int(info["stages_ok"]), "sequence": info["sequence"],
                   "terminated": not bool(info["TimeLimit.truncated"]),
                   "warmup": env.warmup_frames, "stage_frames": env.stage_frames,
                   "state_spec": env.spec_state}
            break
    env.close()
    del model
    return out


def render_model(mujoco):
    """The environment's scene, rebuilt identically, plus lights and a floor texture."""
    hand_dir = Path(DEFAULT_HAND_DIR)
    diff = parse_difficulty("m0.20g1.5")
    geom_type, size = OBJECTS["box"]
    size = tuple(v * diff["size_scale"] for v in size)
    assets = _asset_dict(hand_dir)
    centre = _closed_fingertip_centre(mujoco, hand_dir, assets)
    xml = scene_xml("right_hand.xml", geom_type, size, mass=diff["mass"], centre=centre)
    tail, site = _sensor_block(geom_type, size)
    xml = xml.replace('<freejoint name="obj_free"/>', '<freejoint name="obj_free"/>\n      ' + site)
    xml = xml.replace("</mujoco>", tail)
    # Visual-only additions (no bodies, joints or actuators change).
    vis = """
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
  </asset>"""
    xml = xml.replace('integrator="implicitfast"/>', 'integrator="implicitfast"/>' + vis, 1)
    lights = """<worldbody>
    <light pos="0.45 -0.35 0.9" dir="-0.3 0.3 -1" directional="true"
           diffuse="0.75 0.75 0.75" specular="0.25 0.25 0.25" castshadow="true"/>
    <light pos="-0.25 0.45 0.7" dir="0.25 -0.45 -1" directional="true"
           diffuse="0.22 0.22 0.28" castshadow="false"/>"""
    xml = xml.replace("<worldbody>", lights, 1)
    xml = xml.replace('pos="0 0 -0.5"\n          rgba="0.3 0.3 0.35 1"', 'pos="0 0 -0.5" material="grid"')
    return mujoco.MjModel.from_xml_string(xml, assets)


def choose(seed_arg: int | None, ep_arg: int | None) -> tuple[int, int, str]:
    ev = {a: {s: json.loads((SWEEP / f"{a}_naive_s{s}" / "eval.json").read_text()) for s in range(20)}
          for a in ("keyframe", "continuous")}
    means = {a: np.mean([ev[a][s]["success"] for s in range(20)]) for a in ev}
    if seed_arg is None:
        dist = {s: abs(ev["keyframe"][s]["success"] - means["keyframe"])
                + abs(ev["continuous"][s]["success"] - means["continuous"]) for s in range(20)}
        seed = min(dist, key=dist.get)
        why = (f"seed {seed}: held-out success keyframe {ev['keyframe'][seed]['success']:.2f}, "
               f"continuous {ev['continuous'][seed]['success']:.2f}, closest to the 20-seed means "
               f"{means['keyframe']:.3f} / {means['continuous']:.3f}")
    else:
        seed, why = seed_arg, f"seed {seed_arg} (given)"
    kr, cr = ev["keyframe"][seed]["records"], ev["continuous"][seed]["records"]
    if ep_arg is None:
        cand = [i for i in range(50) if kr[i]["success"] and not cr[i]["success"]]
        ep = cand[0] if cand else 0
        why += (f"; episode {ep}: first of the first 50 eval episodes with keyframe success and "
                f"continuous failure ({len(cand)} such of 50)" if cand else
                "; no episode in the first 50 with keyframe success and continuous failure, using 0")
    else:
        ep, why = ep_arg, why + f"; episode {ep_arg} (given)"
    return seed, ep, why


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="RL seed (default: selection rule)")
    ap.add_argument("--episode", type=int, default=None, help="eval episode (default: selection rule)")
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--eval-sequences", default="Hx-Hy-R,Hy-Hx-R")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=400, help="panel height including the two bars")
    ap.add_argument("--fps", type=int, default=30, help="30 = the simulation's frame rate (real time)")
    ap.add_argument("--hold-s", type=float, default=1.5, help="hold of the last frame")
    ap.add_argument("--azimuth", type=float, default=138.0)
    ap.add_argument("--elevation", type=float, default=-35.0)
    ap.add_argument("--distance", type=float, default=0.34)
    ap.add_argument("--out", default="runs/rl/video/motion_level.mp4")
    ap.add_argument("--peek", default="runs/rl/video/motion_level_peek.png")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    torch.set_num_threads(1)
    seed, ep, why = choose(args.seed, args.episode)
    print(why, flush=True)

    runs = {}
    for arm in ("keyframe", "continuous"):
        r = rollout(arm, seed, ep, args.eval_sequences, args.eval_seed)
        rec = json.loads((SWEEP / f"{arm}_naive_s{seed}" / "eval.json").read_text())["records"][ep]
        match = all(r[k] == rec[k] for k in ("success", "frames", "stages_ok", "sequence"))
        print(f"{arm:10s} replay: {r['sequence']} success={r['success']} frames={r['frames']} "
              f"stages_ok={r['stages_ok']} | eval.json: success={rec['success']} frames={rec['frames']} "
              f"stages_ok={rec['stages_ok']} | {'MATCH' if match else 'MISMATCH'}", flush=True)
        r["match"] = match
        runs[arm] = r
    seq = runs["keyframe"]["sequence"]
    assert seq == runs["continuous"]["sequence"]

    m = render_model(mujoco)
    d = mujoco.MjData(m)
    spec = runs["keyframe"]["state_spec"]
    assert mujoco.mj_stateSize(m, spec) == runs["keyframe"]["states"].shape[1], "render model differs"
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
    mujoco.mj_setState(m, d, runs["keyframe"]["states"][0], spec)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation
    cam.lookat[:] = d.xpos[obj] + np.array([0.0, 0.0, -0.01])
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    bar = 36
    rh = args.height - 2 * bar
    fb = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 18)
    fr = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    fs = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    colour = {"keyframe": (120, 200, 255), "continuous": (255, 176, 102)}
    n_total = max(len(r["states"]) for r in runs.values())
    warm, sf = runs["keyframe"]["warmup"], runs["keyframe"]["stage_frames"]

    def panel(arm: str, t: int, img: np.ndarray) -> Image.Image:
        r = runs[arm]
        last = len(r["states"]) - 1
        p = Image.new("RGB", (args.width, args.height), (15, 17, 21))
        p.paste(Image.fromarray(img), (0, bar))
        dr = ImageDraw.Draw(p)
        dr.text((12, bar // 2), TITLES[arm], font=fb, fill=colour[arm], anchor="lm")
        tt = min(t, last)
        if t >= last:
            ok = r["success"]
            if ok:
                txt, fill = "succeeded", (24, 92, 54)
            else:
                sk = r["skills"][last]
                where = SKILL_TEXT[SKILLS[sk]] if sk >= 0 else "warm-up"
                txt, fill = f"failed: cube lost during {where}", (120, 38, 38)
            dr.rectangle([0, args.height - bar, args.width, args.height], fill=fill)
            dr.text((12, args.height - bar // 2), txt, font=fr, fill=(235, 240, 245), anchor="lm")
        else:
            sk = r["skills"][tt] if tt > 0 else -1
            txt = SKILL_TEXT[SKILLS[sk]] if sk >= 0 else "settle (no tilt)"
            dr.text((12, args.height - bar // 2), txt, font=fr, fill=(225, 228, 235), anchor="lm")
        dr.text((args.width - 12, args.height - bar // 2), f"t = {tt / 30:.1f} s",
                font=fs, fill=(150, 155, 165), anchor="rm")
        return p

    frames = []
    with mujoco.Renderer(m, rh, args.width) as ren:
        imgs = {}
        for arm, r in runs.items():
            out = []
            for st in r["states"]:
                mujoco.mj_setState(m, d, st, spec)
                mujoco.mj_forward(m, d)
                ren.update_scene(d, cam, opt)
                out.append(ren.render().copy())
            imgs[arm] = out
    gap = 8
    for t in range(n_total):
        canvas = Image.new("RGB", (2 * args.width + gap, args.height), (40, 42, 48))
        for j, arm in enumerate(("keyframe", "continuous")):
            im = imgs[arm][min(t, len(imgs[arm]) - 1)]
            canvas.paste(panel(arm, t, im), (j * (args.width + gap), 0))
        frames.append(np.asarray(canvas))
    frames += [frames[-1]] * int(round(args.hold_s * args.fps))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                           pixelformat="yuv420p", macro_block_size=8)
    for f in frames:
        w.append_data(f)
    w.close()
    dur = len(frames) / args.fps
    print(f"wrote {out}  {len(frames)} frames at {args.fps} fps = {dur:.2f} s  size {frames[0].shape[1]}x{frames[0].shape[0]}")

    # Contact sheet: settle, first hold, second hold, end.
    picks = [0, warm + sf // 2, warm + sf + sf // 2, n_total - 1]
    tiles = [Image.fromarray(frames[i]) for i in picks]
    sc = 0.5
    tw, th = int(tiles[0].width * sc), int(tiles[0].height * sc)
    sheet = Image.new("RGB", (2 * tw, 2 * th), (15, 17, 21))
    for k, tile in enumerate(tiles):
        sheet.paste(tile.resize((tw, th), Image.LANCZOS), ((k % 2) * tw, (k // 2) * th))
    sheet.save(ROOT / args.peek)
    print(f"wrote {ROOT / args.peek}  frames {picks}")
    print(json.dumps({"seed": seed, "episode": ep, "sequence": seq, "eval_seed": args.eval_seed,
                      "why": why, **{f"{a}_success": runs[a]["success"] for a in runs},
                      **{f"{a}_frames": runs[a]["frames"] for a in runs},
                      **{f"{a}_match_eval_json": runs[a]["match"] for a in runs},
                      "duration_s": dur}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
