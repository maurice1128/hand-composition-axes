"""N retention environments in one process, batched decoder, threaded physics.

``hand_env.ShadowRetainEnv`` is the reference implementation, one env per
process. It is also slow in the way that matters on a laptop: every process
pays a full GRU step for one frame and a full Python loop over 17 physics
substeps. This class keeps N environments in one process, decodes all N
latents in one batched GRU step, and hands the N physics rollouts to
``mujoco.rollout`` with a thread pool. It implements the Stable-Baselines3
``VecEnv`` interface directly, so PPO does not know the difference.

No per-environment ``MjData`` survives between frames. The rollout takes the
N physics states in and returns them, and everything the observation and the
reward need is read from sensors the scene declares (object pose and
velocity, palm position, and a touch sensor on the object), so the per-env
``mj_forward`` that would otherwise cost as much as the physics is gone.

The task is the same as the single env's: object rests against the palm,
random lateral pulses after a warm-up, +1 per held frame, done when lost.
The pulse is an external force on the object, ``mass * a``, which is what
tilting gravity did to the object in the offline task; gravity itself lives
on the shared model and cannot differ per environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from caredex.grasp_task import DIFFICULTY, OBJECTS, _asset_dict, _closed_fingertip_centre, scene_xml
from caredex.hand_model import DOF_INDEX, N_ARTICULATED, N_DOF, denormalize, normalize
from caredex.robot_hand import SHADOW_MAP
from caredex.rl.stepper import load_stepper

FPS = 30
DEFAULT_HAND_DIR = Path(r"D:\datasets\mujoco_menagerie\shadow_hand")


def parse_difficulty(spec: str) -> dict:
    """``"medium"`` from the offline table, or ``"m0.20g2.0"`` for mass and shake, with an optional size scale, ``"m0.20g1.5s1.2"``."""
    if spec in DIFFICULTY:
        return DIFFICULTY[spec]
    import re
    m = re.fullmatch(r"m([0-9.]+)g([0-9.]+)(?:s([0-9.]+))?", spec)
    if not m:
        raise ValueError(f"unknown difficulty {spec!r}")
    return {"mass": float(m.group(1)), "shake_g": float(m.group(2)),
            "size_scale": float(m.group(3)) if m.group(3) else 1.0}


def _sensor_block(shape: str, size: tuple[float, ...]) -> str:
    dims = " ".join(f"{s * 1.05}" for s in size)
    return f"""
  <sensor>
    <framepos name="obj_pos" objtype="body" objname="obj"/>
    <framequat name="obj_quat" objtype="body" objname="obj"/>
    <framelinvel name="obj_vel" objtype="body" objname="obj"/>
    <framepos name="palm_pos" objtype="body" objname="rh_palm"/>
    <touch name="obj_touch" site="obj_site"/>
  </sensor>
</mujoco>""", f'<site name="obj_site" type="{shape}" size="{dims}" rgba="0 0 0 0"/>'


class _BatchedDecoder:
    """Step a modular or continuous decoder for B environments at once."""

    def __init__(self, stepper, batch: int, rewindow: int = 32):
        self.kind = stepper.spec.kind
        self.model = stepper.model
        self.cfg = getattr(stepper, "cfg", None)
        self.n_channels = stepper.n_channels
        self.B = batch
        self.rewindow = rewindow
        self.style_mode = getattr(stepper, "style_mode", "window")
        self.logit_scale = getattr(stepper, "logit_scale", 1.0)
        L = self.cfg.n_layers if self.cfg is not None else 1
        H = self.cfg.hidden_dim if self.cfg is not None else 1
        self.h = torch.zeros(L, batch, H)
        self.anchor = torch.zeros(batch, self.n_channels)
        self.last = torch.zeros(batch, self.n_channels)
        self.style = torch.zeros(batch, getattr(self.cfg, "style_dim", 0) if self.cfg is not None else 0)
        self.t = np.zeros(batch, dtype=np.int64)
        self.fresh = np.ones(batch, dtype=bool)
        # keyframe / linear interfaces: latched target and frames since latch
        self.horizon = getattr(stepper, "horizon", rewindow)
        self.target = torch.zeros(batch, self.n_channels)
        self.t_in = np.zeros(batch, dtype=np.int64)
        # brick interface: K logits select one of K fixed keyframes (arg-max)
        self.bricks = getattr(stepper, "bricks", None)

    def reset(self, idx: np.ndarray, initial_pose: np.ndarray) -> None:
        p = torch.as_tensor(initial_pose, dtype=torch.float32)
        self.anchor[idx] = p
        self.last[idx] = p
        self.t[idx] = 0
        self.fresh[idx] = True

    @torch.no_grad()
    def step(self, actions: np.ndarray) -> np.ndarray:
        a = torch.as_tensor(actions, dtype=torch.float32)
        wrap = (self.t > 0) & (self.t % self.rewindow == 0)
        if wrap.any():
            w = torch.as_tensor(wrap)
            self.anchor[w] = self.last[w]
            self.fresh |= wrap
        fresh = torch.as_tensor(self.fresh)
        m, c = self.model, self.cfg
        if self.kind in ("keyframe", "linear"):
            if self.bricks is not None:
                a = self.bricks[a.argmax(dim=1)]
            if fresh.any():
                tgt = self.anchor[fresh].clone()
                tgt[:, :N_ARTICULATED] = a[fresh][:, :N_ARTICULATED]
                self.target[fresh] = tgt
                self.t_in[self.fresh] = 0
            if self.kind == "linear":
                self.t_in += 1
                w = torch.as_tensor(np.minimum(self.t_in / self.horizon, 1.0), dtype=torch.float32).unsqueeze(1)
                pose = (1 - w) * self.anchor + w * self.target
            else:
                mask = torch.zeros(self.B, self.n_channels); mask[:, :N_ARTICULATED] = 1.0
                remaining = torch.as_tensor(np.maximum(self.horizon - self.t_in, 0) / c.window, dtype=torch.float32).unsqueeze(1)
                x = torch.cat([self.anchor, self.target * mask, mask, remaining], dim=-1)
                if fresh.any():
                    h0 = m.decoder_init(x[fresh]).view(-1, c.n_layers, c.hidden_dim).permute(1, 0, 2)
                    self.h[:, fresh] = h0
                out, self.h = m.decoder_rnn(x.unsqueeze(1), self.h.contiguous())
                self.t_in += 1
                pose = m.to_pose(out[:, 0])
                pose = torch.tanh(pose) if c.bounded_output else pose
            self.last = pose
            self.fresh[:] = False
            self.t += 1
            return pose.numpy()
        if self.kind == "modular":
            K = c.n_primitives
            logits, style_in = a[:, :K], a[:, K:]
            if self.style_mode == "frame":
                self.style = style_in
            else:
                self.style[fresh] = style_in[fresh]
            ctx = torch.cat([self.style, self.anchor], dim=-1)
            if fresh.any():
                h0 = m.decoder_init(ctx[fresh]).view(-1, c.n_layers, c.hidden_dim).permute(1, 0, 2)
                self.h[:, fresh] = h0
            prim = F.softmax(self.logit_scale * logits, dim=-1) @ m.primitives
            x = torch.cat([prim, ctx], dim=-1).unsqueeze(1)
        else:
            z = a
            if fresh.any():
                ctx0 = torch.cat([z[fresh], self.anchor[fresh]], dim=-1)
                h0 = m.decoder_init(ctx0).view(-1, c.n_layers, c.hidden_dim).permute(1, 0, 2)
                self.h[:, fresh] = h0
            x = torch.cat([z, self.anchor], dim=-1).unsqueeze(1)
        out, self.h = m.decoder_rnn(x, self.h.contiguous())
        pose = m.to_pose(out[:, 0])
        pose = torch.tanh(pose) if c.bounded_output else pose
        self.last = pose
        self.fresh[:] = False
        self.t += 1
        return pose.numpy()


class RetainVecEnv(VecEnv):
    def __init__(self, arm: str, run_dir: str | Path, num_envs: int = 32,
                 shape: str = "box", difficulty: str = "medium",
                 episode_frames: int = 90, warmup_frames: int = 30,
                 perturb: bool = True, hold_tol_m: float = 0.10,
                 hand_dir: str | Path | None = None, stepper_kw: dict | None = None,
                 nthread: int = 8, seed: int = 0,
                 init_poses: str | Path | None = None, settle_frames: int = 10,
                 place_per_pose: bool = True, perturb_mode: str = "pulse",
                 ramp_frames: int = 45, reward_mode: str = "sparse",
                 pose_residual_deg: float = 0.0, macro_every: int = 1,
                 residual_mode: str = "rate", residual_rate_deg: float = 3.0,
                 rolling_friction: float | None = None):
        import mujoco
        from mujoco import rollout

        self.mj = mujoco
        skw = dict(stepper_kw or {})
        if arm in ("keyframe", "linear", "brick", "brickkf"):
            skw.setdefault("horizon", max(1, int(macro_every)))
        stepper = load_stepper(run_dir, arm, device="cpu", **skw)
        self.dec = _BatchedDecoder(stepper, num_envs, rewindow=getattr(stepper, "horizon", 32))
        spec = stepper.spec
        # The policy acts in [-1, 1]^d; the env scales to the decoder's bounds
        # (5 on logits, 3 on style and latent). A unit Gaussian in [-1, 1]
        # then explores the whole range from the first rollout, where a unit
        # Gaussian on raw logits never left the near-uniform mixture.
        self.act_scale = spec.high.astype(np.float32)
        # Optional pose residual: the policy adds up to +-pose_residual_deg to
        # each of the 21 articulated targets the decoder produced, the same
        # correction channel VQ-ACE and LAMP give their policies. Offered to
        # both arms identically; the decoder's re-anchor sees the corrected pose.
        self.n_dec = int(spec.action_dim)
        self.pose_residual_deg = float(pose_residual_deg)
        if residual_mode not in ("abs", "rate"):
            raise ValueError(residual_mode)
        # "abs":  the action *is* the residual, +-pose_residual_deg per frame.
        #         A stochastic policy then swings every joint by tens of
        #         degrees frame to frame (jitter ~2e4) and no grasp survives.
        # "rate": the action is a residual *velocity*, +-residual_rate_deg per
        #         frame, integrated within the episode and clipped to
        #         +-pose_residual_deg. The correction channel stays fine and
        #         smooth, which is what a residual is for.
        self.residual_mode = residual_mode
        if self.pose_residual_deg > 0:
            from caredex.hand_model import LIMITS_HI, LIMITS_LO
            span = (LIMITS_HI - LIMITS_LO)[:N_ARTICULATED].astype(np.float32)
            self.res_scale = (2.0 * self.pose_residual_deg / span).astype(np.float32)
            self.rate_scale = (2.0 * float(residual_rate_deg) / span).astype(np.float32)
            self.res_state = np.zeros((num_envs, N_ARTICULATED), dtype=np.float32)
            d = self.n_dec + N_ARTICULATED
        else:
            self.res_scale = None
            d = self.n_dec
        action_space = spaces.Box(-np.ones(d, np.float32), np.ones(d, np.float32), dtype=np.float32)
        # Hierarchical interface (the MotionBricks split of *what* from *how*):
        # the decoder part of the action is read only every ``macro_every``
        # frames and held in between, so a primitive, once chosen, plays out;
        # the residual part, if any, stays per frame. Applied to both arms:
        # for the continuous arm the held latent is a low-rate target.
        self.macro_every = max(1, int(macro_every))
        self._macro = np.zeros((num_envs, self.n_dec), dtype=np.float32)

        # Scene: the offline task's, plus a site on the object and sensors.
        hand_dir = Path(hand_dir or DEFAULT_HAND_DIR)
        diff = parse_difficulty(difficulty)
        geom_type, size = OBJECTS[shape]
        size = tuple(v * diff["size_scale"] for v in size)
        assets = _asset_dict(hand_dir)
        centre = _closed_fingertip_centre(mujoco, hand_dir, assets)
        xml = scene_xml("right_hand.xml", geom_type, size, mass=diff["mass"], centre=centre)
        if rolling_friction is not None:
            # Opt-in: round objects roll off the palm without rolling friction
            # (the offline scene uses condim 4). Default leaves the scene as is.
            assert 'friction="1.0 0.02 0.001"' in xml and 'condim="4"' in xml
            xml = xml.replace('friction="1.0 0.02 0.001"', f'friction="1.0 0.02 {float(rolling_friction)}"')
            xml = xml.replace('condim="4"', 'condim="6"')
        tail, site = _sensor_block(geom_type, size)
        xml = xml.replace('<freejoint name="obj_free"/>', '<freejoint name="obj_free"/>\n      ' + site)
        xml = xml.replace("</mujoco>", tail)
        self.m = mujoco.MjModel.from_xml_string(xml, assets)
        self.d0 = mujoco.MjData(self.m)
        self.pool = [mujoco.MjData(self.m) for _ in range(max(1, nthread))]
        self.roll = rollout.Rollout(nthread=max(1, nthread))

        # Actuator map, as in GraspTask: PIP drives the coupled J0, DIP is dropped.
        self.dof_aid: list[tuple[int, int]] = []
        for dof, joint in SHADOW_MAP.items():
            if dof.endswith("dip_flex"):
                continue
            name = joint.replace("rh_", "rh_A_")
            if dof.endswith("pip_flex"):
                name = name[:-1] + "0"
            aid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self.dof_aid.append((DOF_INDEX[dof], aid))

        self.obj_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_mass = float(self.m.body_mass[self.obj_body])
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        self.obj_qadr, self.obj_vadr = int(self.m.jnt_qposadr[jid]), int(self.m.jnt_dofadr[jid])
        self.lo, self.hi = self.m.actuator_ctrlrange[:, 0].copy(), self.m.actuator_ctrlrange[:, 1].copy()
        self.steps_per_frame = max(1, int(round((1 / FPS) / self.m.opt.timestep)))
        self.spec_state = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.spec_ctrl = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED
        self.nstate = mujoco.mj_stateSize(self.m, self.spec_state)
        self.nu, self.nq, self.nv = self.m.nu, self.m.nq, self.m.nv
        # FULLPHYSICS layout is [time, qpos, qvel, act]; hand indices exclude the free joint.
        qidx = np.setdiff1d(np.arange(self.nq), np.arange(self.obj_qadr, self.obj_qadr + 7))
        vidx = np.setdiff1d(np.arange(self.nv), np.arange(self.obj_vadr, self.obj_vadr + 6))
        self.hand_q_cols = 1 + qidx
        self.hand_v_cols = 1 + self.nq + vidx

        def sens(name):
            sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            a, n = int(self.m.sensor_adr[sid]), int(self.m.sensor_dim[sid])
            return slice(a, a + n)
        self.s_obj_pos, self.s_obj_quat = sens("obj_pos"), sens("obj_quat")
        self.s_obj_vel, self.s_palm, self.s_touch = sens("obj_vel"), sens("palm_pos"), sens("obj_touch")

        self.episode_frames, self.warmup_frames = int(episode_frames), int(warmup_frames)
        self.perturb, self.hold_tol_m = bool(perturb), float(hold_tol_m)
        self.shake_g = diff["shake_g"]
        if perturb_mode not in ("pulse", "tilt"):
            raise ValueError(perturb_mode)
        if reward_mode not in ("sparse", "dense"):
            raise ValueError(reward_mode)
        # "pulse": 0.1 s lateral kicks at random times, the offline task's shake
        #          made stochastic. Too fast for any decoder to react to, so the
        #          only strategy is to pre-tighten.
        # "tilt":  one lateral acceleration in a random direction per episode,
        #          ramped from 0 to shake_g over ``ramp_frames`` after warm-up
        #          and then held: the palm tilting until the object slides. The
        #          object starts to move before it is lost, so a policy can
        #          react, and the reward carries signal before the loss.
        self.perturb_mode = perturb_mode
        self.ramp_frames = int(ramp_frames)
        self.reward_mode = reward_mode
        self.tilt_dir = np.zeros((num_envs, 2))

        obs_dim = len(qidx) + len(vidx) + 3 + 3 + 4 + N_DOF + 3 + 1
        observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)
        self.render_mode = None

        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((num_envs, self.nstate))
        self.sensors = np.zeros((num_envs, self.m.nsensordata))
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.accel = np.zeros((num_envs, 3))
        self.pulse_left = np.zeros(num_envs, dtype=np.int64)
        self.start = np.zeros((num_envs, 3))
        self.last_pose = np.zeros((num_envs, N_DOF), dtype=np.float32)
        self.cmd_prev = np.zeros((num_envs, 2, N_ARTICULATED))
        self.jit_sum = np.zeros(num_envs)
        self.ep_frames = np.zeros(num_envs, dtype=np.int64)
        self._actions = None
        self.open_pose = normalize(np.zeros(N_DOF, dtype=np.float32))

        # Reset bank. The decoders were trained with real first frames as the
        # window anchor, and a flat zero-degree hand is a pose no OakInk clip
        # starts from: from it a held primitive flexes the fingers by 6 to 20
        # degrees in 30 frames where a fist needs 104, so no policy can close
        # in time. Each episode therefore starts from a first frame drawn from
        # the training data (``init_poses``), settled for ``settle_frames``
        # with that pose held, so the object is genuinely resting in the hand
        # when the policy takes over. States are settled once here, in the
        # threaded rollout, and sampled at reset.
        self.init_poses = None if init_poses is None else np.load(init_poses).astype(np.float32)
        self.settle_frames = int(settle_frames)
        self.place_per_pose = bool(place_per_pose)
        self._build_reset_bank()

    # -- reset bank -----------------------------------------------------------------

    def _ctrl_for(self, q_deg: np.ndarray) -> np.ndarray:
        """``(N, 27)`` degrees -> ``(N, nu)`` clipped actuator targets."""
        q_rad = np.radians(q_deg.astype(np.float64))
        ctrl = np.zeros((q_deg.shape[0], self.nu))
        for di, aid in self.dof_aid:
            ctrl[:, aid] = np.clip(q_rad[:, di], self.lo[aid], self.hi[aid])
        return ctrl

    def _build_reset_bank(self) -> None:
        mj = self.mj
        poses = (np.zeros((1, N_DOF), dtype=np.float32) if self.init_poses is None
                 else self.init_poses)
        n = poses.shape[0]
        # Start every candidate from the scene's rest state with the hand's
        # joints already at the pose, so the first frame is not a snap.
        mj.mj_resetData(self.m, self.d0)
        mj.mj_forward(self.m, self.d0)
        base = np.empty(self.nstate)
        mj.mj_getState(self.m, self.d0, base, self.spec_state)
        init = np.tile(base, (n, 1))
        q_rad = np.radians(poses.astype(np.float64))
        for dof, joint in SHADOW_MAP.items():
            jid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0:
                continue
            lo, hi = self.m.jnt_range[jid]
            init[:, 1 + int(self.m.jnt_qposadr[jid])] = np.clip(q_rad[:, DOF_INDEX[dof]], lo, hi)
        # Place the object for *this* hand shape, not for a closed fist: midway
        # between the palm and the fingertip centroid of the initial pose, as
        # the offline scene did once for its fist. Placing every pose's object
        # at the fist's spot kept only the most open hands (the closed ones
        # ejected it in settling), which biased the anchors toward poses the
        # priors then could not close from.
        if self.place_per_pose:
            tips = [mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, b)
                    for b in ("rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal")]
            palm = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "rh_palm")
            qa = 1 + self.obj_qadr
            for i in range(n):
                mj.mj_setState(self.m, self.d0, init[i], self.spec_state)
                mj.mj_forward(self.m, self.d0)
                centre = (self.d0.xpos[palm] + np.mean([self.d0.xpos[t] for t in tips], axis=0)) / 2
                init[i, qa:qa + 3] = centre
        ctrl = self._ctrl_for(poses)
        control = np.concatenate([ctrl, np.zeros((n, self.m.nbody * 6))], axis=-1)[:, None, :]
        nstep = max(1, self.settle_frames) * self.steps_per_frame
        state, sens = self.roll.rollout(self.m, self.pool, init, control,
                                        control_spec=self.spec_ctrl, nstep=nstep)
        st, se = state[:, -1], sens[:, -1]
        start0 = self.d0.sensordata[self.s_obj_pos]
        drop = np.linalg.norm(se[:, self.s_obj_pos] - start0, axis=1)
        ok = (se[:, self.s_touch][:, 0] > 0) & (drop < self.hold_tol_m)
        self.bank_ok_fraction = float(ok.mean())
        if not ok.any():
            raise RuntimeError("no initial pose keeps the object in the hand after settling")
        self.bank_states, self.bank_sensors = st[ok], se[ok]
        self.bank_poses = normalize(poses[ok])
        self.bank_ctrl = ctrl[ok]

    # -- per-env helpers ----------------------------------------------------------

    def _reset_env(self, idx: np.ndarray) -> None:
        k = self.rng.integers(0, len(self.bank_states), size=len(idx))
        self.states[idx] = self.bank_states[k]
        self.sensors[idx] = self.bank_sensors[k]
        self.start[idx] = self.bank_sensors[k][:, self.s_obj_pos]
        self.t[idx] = 0
        self.accel[idx] = 0.0
        self.pulse_left[idx] = 0
        ang = self.rng.uniform(0, 2 * np.pi, len(idx))
        self.tilt_dir[idx] = np.stack([np.cos(ang), np.sin(ang)], -1)
        self.last_pose[idx] = self.bank_poses[k]
        if self.res_scale is not None:
            self.res_state[idx] = 0.0
        self.cmd_prev[idx] = 0.0
        self.jit_sum[idx] = 0.0
        self.ep_frames[idx] = 0
        self.dec.reset(idx, self.bank_poses[k])

    def _obs(self) -> np.ndarray:
        s = self.sensors
        rel = s[:, self.s_obj_pos] - s[:, self.s_palm]
        return np.concatenate([
            self.states[:, self.hand_q_cols], self.states[:, self.hand_v_cols],
            rel, s[:, self.s_obj_vel], s[:, self.s_obj_quat],
            self.last_pose, self.accel, (self.t / self.episode_frames)[:, None],
        ], axis=1).astype(np.float32)

    def _held(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        drop = np.linalg.norm(self.sensors[:, self.s_obj_pos] - self.start, axis=1)
        touch = self.sensors[:, self.s_touch][:, 0]
        held = (drop < self.hold_tol_m) & (touch > 0)
        return held, drop, touch

    def _schedule(self) -> None:
        if not self.perturb:
            return
        if self.perturb_mode == "tilt":
            s = np.clip((self.t - self.warmup_frames) / max(self.ramp_frames, 1), 0.0, 1.0)
            self.accel[:, :2] = self.tilt_dir * (s * self.shake_g)[:, None]
            self.accel[:, 2] = 0.0
            return
        active = self.t >= self.warmup_frames
        self.accel[~active] = 0.0
        dec = active & (self.pulse_left > 0)
        self.pulse_left[dec] -= 1
        ended = dec & (self.pulse_left == 0)
        self.accel[ended] = 0.0
        idle = active & (self.pulse_left == 0) & ~ended
        fire = idle & (self.rng.random(self.num_envs) < 1 / 15)
        n = int(fire.sum())
        if n:
            ang = self.rng.uniform(0, 2 * np.pi, n)
            mag = self.rng.uniform(0.5, 1.0, n) * self.shake_g
            self.accel[fire] = np.stack([mag * np.cos(ang), mag * np.sin(ang), np.zeros(n)], -1)
            self.pulse_left[fire] = 3

    # -- VecEnv API -----------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._reset_env(np.arange(self.num_envs))
        return self._obs()

    def step_async(self, actions: np.ndarray) -> None:
        a = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        dec = a[:, :self.n_dec] * self.act_scale
        if self.macro_every > 1:
            upd = (self.t % self.macro_every) == 0
            self._macro[upd] = dec[upd]
            dec = self._macro
        self._actions = dec
        if self.res_scale is None:
            self._residual = None
        elif self.residual_mode == "abs":
            self._residual = a[:, self.n_dec:] * self.res_scale
        else:
            self.res_state = np.clip(self.res_state + a[:, self.n_dec:] * self.rate_scale,
                                     -self.res_scale, self.res_scale)
            self._residual = self.res_state

    def step_wait(self):
        pose = self.dec.step(self._actions)
        if self._residual is not None:
            pose = pose.copy()
            pose[:, :N_ARTICULATED] = np.clip(pose[:, :N_ARTICULATED] + self._residual, -1.0, 1.0)
            self.dec.last = torch.as_tensor(pose, dtype=torch.float32)
        self.last_pose = pose.astype(np.float32)
        q_deg = denormalize(pose)
        q_rad = np.radians(q_deg.astype(np.float64))
        cmd = q_deg[:, :N_ARTICULATED]

        have2 = self.ep_frames >= 2
        dd = cmd - 2 * self.cmd_prev[:, 1] + self.cmd_prev[:, 0]
        self.jit_sum[have2] += np.sum(dd[have2] ** 2, axis=-1)
        self.cmd_prev[:, 0] = self.cmd_prev[:, 1]
        self.cmd_prev[:, 1] = cmd

        self._schedule()

        ctrl = np.zeros((self.num_envs, self.nu))
        for di, aid in self.dof_aid:
            ctrl[:, aid] = np.clip(q_rad[:, di], self.lo[aid], self.hi[aid])
        xfrc = np.zeros((self.num_envs, self.m.nbody * 6))
        xfrc[:, self.obj_body * 6:self.obj_body * 6 + 3] = self.accel * 9.81 * self.obj_mass
        control = np.concatenate([ctrl, xfrc], axis=-1)[:, None, :]  # tiled over nstep
        state, sensordata = self.roll.rollout(self.m, self.pool, self.states, control,
                                              control_spec=self.spec_ctrl,
                                              nstep=self.steps_per_frame)
        self.states = state[:, -1]
        self.sensors = sensordata[:, -1]

        self.t += 1
        self.ep_frames += 1
        held, drop, touch = self._held()
        lost = drop >= self.hold_tol_m
        rewards = held.astype(np.float32)
        if self.reward_mode == "dense":
            # Same optimum as sparse (held every frame, object where it started),
            # with a gradient before the loss event: up to -0.5 per frame at the
            # loss threshold. Identical for both arms.
            rewards = rewards - 5.0 * np.minimum(drop, self.hold_tol_m).astype(np.float32)
        term = lost
        trunc = (self.t >= self.episode_frames) & ~term
        dones = term | trunc
        infos: list[dict] = [{"held": bool(held[i]), "drop_m": float(drop[i]),
                              "touch": float(touch[i])} for i in range(self.num_envs)]
        obs = self._obs()
        for i in np.flatnonzero(dones):
            nfr = int(self.ep_frames[i])
            infos[i]["success"] = bool(held[i] and not lost[i])
            infos[i]["frames"] = nfr
            infos[i]["jitter"] = float(self.jit_sum[i] / max(nfr - 2, 1))
            infos[i]["TimeLimit.truncated"] = bool(trunc[i])
            infos[i]["terminal_observation"] = obs[i].copy()
        if dones.any():
            idx = np.flatnonzero(dones)
            self._reset_env(idx)
            obs[idx] = self._obs()[idx]
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.roll.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self._n(indices)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kw):
        return [getattr(self, method_name)(*args, **kw)] * self._n(indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._n(indices)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    def _n(self, indices) -> int:
        if indices is None:
            return self.num_envs
        return len(indices) if hasattr(indices, "__len__") else 1
