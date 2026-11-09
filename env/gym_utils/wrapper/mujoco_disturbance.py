"""
Training-condition (disturbance) wrapper for MuJoCo locomotion tasks, used by
Failure-Aware Sampling (FAS) and the uniform domain-randomization baseline.

Place it between `mujoco_locomotion_lowdim` and `multi_step` in the wrapper list
(make_async always applies `multi_step` last).

At every reset one condition bin c is chosen, in priority order:
  1. options["fas_condition"] passed to reset()
  2. the next entry of a per-env schedule (set_condition_schedule), used for
     evaluation so that every bin is covered
  3. a sample from the probability vector set by set_condition_probs (FAS / uniform)

Supported condition parameters (any subset, missing ones stay nominal):
  mass_scale      -- multiplies the mass of `body_name` (torso)
  friction_scale  -- multiplies the sliding friction of every geom
  impulse_force   -- magnitude [N] of random horizontal pushes on `body_name`

Per-step info:      fas_cond
At episode end:     fas_ep_cond, fas_ep_return, fas_ep_len, fas_ep_terminated
(Episode outcomes are tracked here, not in the agent, so episodes spanning a
PPO iteration boundary are still counted.)
"""

import gym
import numpy as np

from util.fas import ConditionSpace

SUPPORTED_PARAMS = ("mass_scale", "friction_scale", "impulse_force")


def _find_mujoco_env(env):
    """Walk the `.env` chain down to the mujoco_py env (has `.sim`)."""
    e = env
    for _ in range(20):
        if hasattr(e, "sim") and hasattr(e, "model"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e and hasattr(e.unwrapped, "sim"):
            return e.unwrapped
        if not hasattr(e, "env"):
            break
        e = e.env
    raise RuntimeError("Could not find a mujoco_py env with `.sim` inside the wrapper stack")


class MujocoDisturbanceWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        condition_space,
        body_name="torso",
        impulse_prob=0.01,
        impulse_duration=5,
        jitter=0.0,
        max_episode_steps=None,
        **kwargs,
    ):
        super().__init__(env)
        self.space = ConditionSpace(dict(condition_space))
        unknown = set(self.space.names) - set(SUPPORTED_PARAMS)
        if unknown:
            raise ValueError(f"Unsupported condition parameters {unknown}; use {SUPPORTED_PARAMS}")
        self.n_conditions = len(self.space)
        self.impulse_prob = float(impulse_prob)
        self.impulse_duration = int(impulse_duration)
        self.jitter = float(jitter)
        self.max_episode_steps = max_episode_steps

        self.mj_env = _find_mujoco_env(env)
        model = self.mj_env.model
        self.body_id = model.body_name2id(body_name)
        self.nominal_body_mass = np.array(model.body_mass, copy=True)
        self.nominal_geom_friction = np.array(model.geom_friction, copy=True)

        self.rng = np.random.default_rng()
        self.probs = np.full(self.n_conditions, 1.0 / self.n_conditions)
        self.schedule = []
        self.condition = 0
        self.params = self.space.params(0)
        self._reset_episode_stats()

    # ----------------------------------------------------- control from agent
    def set_condition_probs(self, probs):
        probs = np.asarray(probs, dtype=float)
        assert probs.shape == (self.n_conditions,), f"{probs.shape} vs {self.n_conditions}"
        self.probs = probs / probs.sum()

    def set_condition_schedule(self, schedule):
        self.schedule = [int(c) for c in schedule]

    def get_num_conditions(self):
        return self.n_conditions

    def seed(self, seed=None):
        self.rng = np.random.default_rng(None if seed is None else seed + 100003)
        return self.env.seed(seed)

    # ----------------------------------------------------------- gym api
    def reset(self, **kwargs):
        options = kwargs.get("options") or {}
        if "fas_condition" in options:
            cond = int(options["fas_condition"])
        elif len(self.schedule) > 0:
            cond = self.schedule.pop(0)
        else:
            cond = int(self.rng.choice(self.n_conditions, p=self.probs))
        self._apply_condition(cond)
        self._reset_episode_stats()
        return self.env.reset(**kwargs)

    def step(self, action):
        self._maybe_push()
        obs, reward, done, info = self.env.step(action)
        self.ep_return += reward
        self.ep_len += 1
        info["fas_cond"] = self.condition

        hit_limit = self.max_episode_steps is not None and self.ep_len >= self.max_episode_steps
        if (done or hit_limit) and not self.ep_reported:
            truncated = info.get("TimeLimit.truncated", False) or (hit_limit and not done)
            info["fas_ep_cond"] = self.condition
            info["fas_ep_return"] = self.ep_return
            info["fas_ep_len"] = self.ep_len
            info["fas_ep_terminated"] = bool(done and not truncated)
            self.ep_reported = True
        return obs, reward, done, info

    # ----------------------------------------------------------- internals
    def _reset_episode_stats(self):
        self.ep_return = 0.0
        self.ep_len = 0
        self.ep_reported = False
        self.push_steps_left = 0

    def _jit(self, x):
        if self.jitter <= 0:
            return x
        return x * (1.0 + self.rng.uniform(-self.jitter, self.jitter))

    def _apply_condition(self, cond):
        self.condition = int(cond)
        self.params = self.space.params(cond)
        model = self.mj_env.model
        model.body_mass[:] = self.nominal_body_mass
        model.geom_friction[:] = self.nominal_geom_friction
        if "mass_scale" in self.params:
            model.body_mass[self.body_id] = self.nominal_body_mass[self.body_id] * self._jit(
                self.params["mass_scale"]
            )
        if "friction_scale" in self.params:
            model.geom_friction[:, 0] = self.nominal_geom_friction[:, 0] * self._jit(
                self.params["friction_scale"]
            )
        self.impulse_force = self._jit(self.params.get("impulse_force", 0.0))
        self.mj_env.sim.data.xfrc_applied[:] = 0.0

    def _maybe_push(self):
        """Random horizontal push on the torso; held for impulse_duration env steps."""
        xfrc = self.mj_env.sim.data.xfrc_applied
        if self.push_steps_left > 0:
            self.push_steps_left -= 1
            if self.push_steps_left == 0:
                xfrc[self.body_id, :] = 0.0
            return
        if self.impulse_force > 0 and self.rng.random() < self.impulse_prob:
            xfrc[self.body_id, :] = 0.0
            xfrc[self.body_id, 0] = self.impulse_force * self.rng.choice([-1.0, 1.0])
            self.push_steps_left = self.impulse_duration
