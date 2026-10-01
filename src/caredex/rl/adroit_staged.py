"""Long-horizon, sparse-reward relocate: grasp-and-lift, carry, place-and-release, repeat.

Line C's task (PREREG, "Line C"). Built on ``AdroitHandRelocateSparse-v1`` so the
hand, the ball and the scene are the benchmark's; only the goal structure and
the reward are ours.

Stages cycle ``grasp -> carry -> place`` up to ``max_stages``:

* **grasp**: the ball is lifted (z above ``lift_z``) with the palm within
  ``near`` of it. The target site is parked above the ball so the native
  observation's goal vectors point at the ball.
* **carry**: the ball is within ``goal_tol`` of an air target drawn as the
  benchmark draws its targets.
* **place**: the ball rests within ``goal_tol`` of a table target (z at the
  table) and the palm has withdrawn beyond ``release`` from it.

Reward is +1 at each stage completion and 0 otherwise; nothing shapes it. The
episode ends when ``max_stages`` are done or at the time limit. ``info``
carries ``stage`` (stages completed), ``success`` (all stages done) and the
per-stage flags. The observation is the native 39-D vector plus a 3-D stage
one-hot and the fraction of stages completed.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

STAGES = ("grasp", "carry", "place")


class StagedRelocate(gym.Wrapper):
    def __init__(self, env, max_stages: int = 3, lift_z: float = 0.10, near: float = 0.08,
                 goal_tol: float = 0.10, release: float = 0.12, table_z: float = 0.035,
                 hold_steps: int = 1, strict: bool = False, rest_speed: float = 0.1, plan=None,
                 progress_obs: bool = True):
        super().__init__(env)
        # plan: the stage sequence, e.g. ("grasp", "carry", "carry"). Default: STAGES cycled up to
        # max_stages (the original grasp -> carry -> place). A plan fixes max_stages to its length.
        if plan is None:
            plan = tuple(STAGES[i % len(STAGES)] for i in range(int(max_stages)))
        self.plan = tuple(plan)
        # progress_obs=False zeroes the "fraction of stages done" feature: it depends on the plan's
        # length, so a skill trained in one plan would see an unseen value when chained in a longer one.
        self.progress_obs = bool(progress_obs)
        assert all(s in STAGES for s in self.plan), self.plan
        max_stages = len(self.plan)
        # A stage completes when its condition has held for hold_steps consecutive steps.
        # hold_steps = 1 with strict = False is the original, instantaneous criterion, which a
        # policy can satisfy by flinging the ball through the target (found 2026-09-25: carry
        # "successes" at a median ball speed of 3.7 m/s). strict adds: carry needs the ball
        # still near the palm, place needs the ball nearly at rest (speed < rest_speed).
        self.hold_steps, self.strict, self.rest_speed = int(hold_steps), bool(strict), float(rest_speed)
        self._held = 0
        self.max_stages = int(max_stages)
        self.lift_z, self.near, self.goal_tol, self.release, self.table_z = lift_z, near, goal_tol, release, table_z
        u = env.unwrapped
        self.u, self.m, self.d = u, u.model, u.data
        n = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (n + len(STAGES) + 1,), np.float64)
        self.k = 0
        m = self.m
        self._ball_dadr = np.asarray([m.jnt_dofadr[j] for j in range(m.njnt)
                                      if m.joint(j).name in ("OBJTx", "OBJTy", "OBJTz")])

    # -- helpers ------------------------------------------------------------------
    @property
    def _ball(self):
        return self.d.xpos[self.u.obj_body_id]

    @property
    def _palm(self):
        return self.d.site_xpos[self.u.S_grasp_site_id]

    def _set_target(self, pos):
        self.m.site_pos[self.u.target_obj_site_id] = pos
        mujoco.mj_forward(self.m, self.d)

    def _stage_name(self):
        return self.plan[min(self.k, len(self.plan) - 1)]

    def _new_subgoal(self):
        self._held = 0
        rng = self.u.np_random
        s = self._stage_name()
        if s == "grasp":
            self._set_target(self._ball + np.array([0.0, 0.0, 0.15]))
        elif s == "carry":
            self._set_target(np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.15, 0.35)]))
        else:
            self._set_target(np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.3), self.table_z]))

    def _stage_cond(self) -> bool:
        s = self._stage_name()
        ball, palm = self._ball, self._palm
        tgt = self.d.site_xpos[self.u.target_obj_site_id]
        if s == "grasp":
            return ball[2] > self.lift_z and np.linalg.norm(ball - palm) < self.near
        if s == "carry":
            ok = np.linalg.norm(ball - tgt) < self.goal_tol
            return ok and (not self.strict or np.linalg.norm(ball - palm) < self.near)
        ok = (np.linalg.norm(ball - tgt) < self.goal_tol and ball[2] < self.table_z + 0.03
              and np.linalg.norm(ball - palm) > self.release)
        return ok and (not self.strict or np.linalg.norm(self.d.qvel[self._ball_dadr]) < self.rest_speed)

    def _stage_done(self) -> bool:
        """Call once per step: counts consecutive steps of the stage condition."""
        self._held = self._held + 1 if self._stage_cond() else 0
        return self._held >= self.hold_steps

    def _obs(self, obs):
        oh = np.zeros(len(STAGES)); oh[STAGES.index(self._stage_name())] = 1.0
        return np.concatenate([obs, oh, [self.k / self.max_stages if self.progress_obs else 0.0]])

    # -- gym ----------------------------------------------------------------------
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.k = 0
        self._new_subgoal()
        obs = self.u._get_obs()
        info.update(stage=0, success=False)
        return self._obs(obs), info

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        rew = 0.0
        if self.k < self.max_stages and self._stage_done():
            rew = 1.0
            self.k += 1
            if self.k < self.max_stages:
                self._new_subgoal()
                obs = self.u._get_obs()
        done_all = self.k >= self.max_stages
        info = dict(info, stage=self.k, success=done_all, stage_name=self._stage_name() if not done_all else "done")
        return self._obs(obs), rew, term or done_all, trunc, info


def make_staged_relocate(max_stages: int = 3, max_episode_steps: int = 600, plan=None, **kw):
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make("AdroitHandRelocateSparse-v1", max_episode_steps=max_episode_steps)
    return StagedRelocate(env, max_stages=max_stages, plan=plan, **kw)


# -- single-stage skill environments (line C, question 2) ------------------------

#: scripted "ball in hand" recipe found 2026-09-23: rotate the palm up, open, teleport the
#: ball 2 cm along the palm normal, close. Actuator names -> targets in [-1, 1].
HELD_ARM = {"A_ARRx": -1.0, "A_ARRy": -0.75, "A_ARTz": 0.3}


class SkillRelocate(StagedRelocate):
    """One stage of :class:`StagedRelocate`, reset into that stage's start state.

    ``skill``: ``"carry"`` or ``"place"`` start with the ball held (scripted grasp);
    ``"grasp"`` starts with the palm turned up and open and the ball ``grasp_dist``
    metres from the palm centre along the palm plane (0 = resting on the palm), the
    curriculum knob. The episode ends when the stage completes (+1) or at the time
    limit; ``info["success"]`` is the stage flag. Stage counter and observation are
    those of the full task, so a skill policy sees what it will see when composed.
    """

    def __init__(self, env, skill: str, grasp_dist: float = 0.0, grasp_start: str = "palm", start_bank=None,
                 reward: str = "sparse", **kw):
        super().__init__(env, max_stages=3, **kw)
        # reward "dense": the benchmark's own relocate shaping, per stage: bring the palm to the
        # ball, +1 per step while the ball is off the table, minus distances to the goal, +10 on
        # completion. "sparse": +1 on completion only (default).
        if reward not in ("sparse", "dense"):
            raise ValueError(reward)
        self.reward_mode = reward
        # start_bank: an .npz of states where the previous skill succeeded (scripts/lineC_chain.py collect);
        # when given, every reset draws one of them, which is how chained skills are trained.
        self.start_bank = None
        if start_bank is not None:
            z = np.load(start_bank)
            self.start_bank = {k: z[k] for k in ("qpos", "qvel", "obj_body_pos")}
        if grasp_start not in ("palm", "table"):
            raise ValueError(grasp_start)
        self.grasp_start = grasp_start
        if skill not in STAGES:
            raise ValueError(skill)
        self.skill, self.grasp_dist = skill, float(grasp_dist)
        m = self.m
        self._act_idx = {m.actuator(i).name: i for i in range(m.nu)}
        # The benchmark's reset re-draws the ball's x, y but leaves its body z and its six
        # joint displacements to whatever the last episode set, so every reset after the
        # first started the ball off the hand (bug found 2026-09-24). Keep the model's
        # original body position and zero the object joints on every scripted reset.
        self._obj_body_pos0 = m.body_pos[self.u.obj_body_id].copy()
        obj_j = [j for j in range(m.njnt) if m.joint(j).name.startswith("OBJ")]
        self._obj_qadr = np.asarray([m.jnt_qposadr[j] for j in obj_j])
        self._obj_dadr = np.asarray([m.jnt_dofadr[j] for j in obj_j])

    def _scripted_reset(self, hold: bool):
        u, d = self.u, self.d
        self.m.body_pos[u.obj_body_id, 2] = self._obj_body_pos0[2]
        d.qpos[self._obj_qadr] = 0.0
        d.qvel[self._obj_dadr] = 0.0
        mujoco.mj_forward(self.m, d)
        a = np.full(self.m.nu, -1.0, dtype=np.float32)            # fingers open
        a[:6] = 0.0
        for k, v in HELD_ARM.items():
            a[self._act_idx[k]] = v
        for _ in range(60):
            u.do_simulation(u.act_mean + a * u.act_rng, u.frame_skip)
        R = d.site_xmat[u.S_grasp_site_id].reshape(3, 3)
        normal, side, along = R[:, 2], R[:, 0], R[:, 1]
        rng = u.np_random
        offset = 0.02 * normal
        if not hold and self.grasp_dist > 0:
            ang = rng.uniform(0, 2 * np.pi)
            offset = offset + self.grasp_dist * (np.cos(ang) * side + np.sin(ang) * along)
        s = u.get_env_state()
        s2 = {k: s[k].copy() for k in ("obj_pos", "qpos", "qvel", "target_pos")}
        s2["obj_pos"] = d.site_xpos[u.S_grasp_site_id] + offset
        s2["qpos"][self._obj_qadr] = 0.0
        s2["qvel"][self._obj_dadr] = 0.0
        u.set_env_state(s2)
        if hold:
            a[6:] = 1.0
            for _ in range(40):
                u.do_simulation(u.act_mean + a * u.act_rng, u.frame_skip)

    def _bank_reset(self):
        b = self.start_bank
        i = int(self.u.np_random.integers(len(b["qpos"])))
        self.m.body_pos[self.u.obj_body_id] = b["obj_body_pos"][i]
        self.d.qpos[:] = b["qpos"][i]
        self.d.qvel[:] = b["qvel"][i]
        mujoco.mj_forward(self.m, self.d)

    def snapshot(self) -> dict:
        """The simulator state a later skill needs to start from here."""
        return {"qpos": self.d.qpos.copy(), "qvel": self.d.qvel.copy(),
                "obj_body_pos": self.m.body_pos[self.u.obj_body_id].copy()}

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.k = STAGES.index(self.skill)
        if self.start_bank is not None:
            self._bank_reset()
        elif not (self.skill == "grasp" and self.grasp_start == "table"):
            # "table": the benchmark's own reset, ball on the table, as in the composed task
            self._scripted_reset(hold=self.skill != "grasp")
        self._new_subgoal()
        info.update(stage=self.k, success=False)
        return self._obs(self.u._get_obs()), info

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        done = self._stage_done()
        rew = 1.0 if done else 0.0
        if self.reward_mode == "dense":
            ball, palm = self._ball, self._palm
            tgt = self.d.site_xpos[self.u.target_obj_site_id]
            rew = -0.1 * float(np.linalg.norm(palm - ball))
            if ball[2] > 0.04:
                rew += 1.0 - 0.5 * float(np.linalg.norm(palm - tgt)) - 0.5 * float(np.linalg.norm(ball - tgt))
            if done:
                rew += 10.0
        info = dict(info, stage=self.k + int(done), success=done, stage_name=self.skill)
        return self._obs(obs), rew, term or done, trunc, info


def make_skill_relocate(skill: str, grasp_dist: float = 0.0, max_episode_steps: int = 200, grasp_start: str = "palm",
                        start_bank=None, reward: str = "sparse", **kw):
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make("AdroitHandRelocateSparse-v1", max_episode_steps=max_episode_steps)
    return SkillRelocate(env, skill=skill, grasp_dist=grasp_dist, grasp_start=grasp_start, start_bank=start_bank,
                         reward=reward, **kw)


# -- flat policy, same reset distribution as the skills (line C control) ----------

class MixedStartRelocate(SkillRelocate):
    """The full staged task for ONE flat policy, reset into a random stage's start state.

    Each episode draws a start stage with probabilities ``start_probs`` (grasp, carry,
    place): grasp starts palm-up with the ball resting on the open palm, carry and place
    start with the ball held, exactly as :class:`SkillRelocate` does for the skills. The
    episode then runs the remaining stages of the full task (reward +1 per stage) to the
    end, so the flat policy sees the same start states the modular skills are trained
    from while still having to chain the stages itself. This separates "the skills had
    better start states" from "the task was decomposed" (Nachum et al. 2019; OmniReset).
    """

    def __init__(self, env, start_probs=(1 / 3, 1 / 3, 1 / 3), **kw):
        super().__init__(env, skill="grasp", **kw)
        p = np.asarray(start_probs, dtype=float)
        self.start_probs = p / p.sum()

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        k0 = int(self.u.np_random.choice(len(STAGES), p=self.start_probs))
        self.skill = STAGES[k0]
        self.k = k0
        self._scripted_reset(hold=k0 > 0)
        self._new_subgoal()
        info.update(stage=self.k, success=False, start_stage=k0)
        return self._obs(self.u._get_obs()), info

    def step(self, action):
        # the full task's step: advance through the remaining stages, +1 each
        return StagedRelocate.step(self, action)


def make_mixed_start_relocate(start_probs=(1 / 3, 1 / 3, 1 / 3), max_episode_steps: int = 600, **kw):
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make("AdroitHandRelocateSparse-v1", max_episode_steps=max_episode_steps)
    return MixedStartRelocate(env, start_probs=start_probs, **kw)
