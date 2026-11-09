"""
Unit tests for Failure-Aware Sampling. No MuJoCo / GPU needed.

    cd dppo && python -m pytest tests -q
"""

import os
import pickle
import sys

import gym
import numpy as np
import pytest
from gym.wrappers import TimeLimit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from util.fas import ConditionSpace, FailureAwareSampler, episode_success  # noqa: E402

TASKS = ["hopper-v2", "walker2d-v2", "halfcheetah-v2"]


# --------------------------------------------------------------------- util.fas
def test_condition_space_roundtrip_and_marginal():
    cs = ConditionSpace({"mass_scale": [0.5, 1.0, 1.5], "friction_scale": [0.5, 1.0]})
    assert len(cs) == 6
    for i in range(len(cs)):
        assert cs.index(**cs.params(i)) == i
    per_bin = np.array([0.0, 1.0, 0.2, 0.4, np.nan, 1.0])  # (mass, friction) grid
    m = cs.marginal(per_bin, "mass_scale")
    assert m[0.5] == pytest.approx(0.5)
    assert m[1.0] == pytest.approx(0.3)
    assert m[1.5] == pytest.approx(1.0)  # nan bin skipped


def test_sampler_beta0_is_uniform():
    s = FailureAwareSampler(5, beta=0.0)
    s.update([0, 0, 1], [False, False, True])
    np.testing.assert_allclose(s.probs(), np.full(5, 0.2))


def test_sampler_matches_eq2():
    alpha, beta, delta = 2.0, 0.7, 0.1
    s = FailureAwareSampler(3, alpha=alpha, beta=beta, delta=delta, ema_rate=1.0)
    s.update([0, 1, 1, 2, 2], [True, False, True, False, False])  # fail rates 0, .5, 1
    w = (np.array([0.0, 0.5, 1.0]) + delta) ** alpha
    expected = beta * w / w.sum() + (1 - beta) / 3
    np.testing.assert_allclose(s.probs(), expected)
    assert s.probs().sum() == pytest.approx(1.0)
    assert s.probs()[2] > s.probs()[1] > s.probs()[0]


def test_sampler_ema_and_first_observation():
    s = FailureAwareSampler(2, ema_rate=0.25, init_score=0.5)
    batch = s.update([0], [False])
    assert s.scores[0] == 1.0  # first observation replaces prior
    assert s.scores[1] == 0.5 and np.isnan(batch[1])  # unobserved bin untouched
    s.update([0, 0], [True, True])
    assert s.scores[0] == pytest.approx(0.75 * 1.0 + 0.25 * 0.0)
    assert list(s.n_observed) == [3, 0]


def test_sampler_state_dict_roundtrip():
    s = FailureAwareSampler(4, alpha=1.5, beta=0.3)
    s.update([1, 2], [False, True])
    s2 = FailureAwareSampler(4)
    s2.load_state_dict(s.state_dict())
    np.testing.assert_allclose(s2.probs(), s.probs())


def test_episode_success():
    ret, term = np.array([100.0, 5000.0, 4000.0]), np.array([True, False, False])
    assert list(episode_success(ret, term, "survive")) == [False, True, True]
    assert list(episode_success(ret, term, "return", 4500)) == [False, True, False]
    assert list(episode_success(ret, term, "survive_and_return", 3000)) == [False, True, True]
    with pytest.raises(ValueError):
        episode_success(ret, term, "return")


# --------------------------------------------------------------------- wrapper
class _Model:
    def __init__(self):
        self.body_mass = np.array([0.0, 3.5, 2.0])
        self.geom_friction = np.tile([0.9, 0.005, 0.0001], (4, 1))

    def body_name2id(self, name):
        return {"world": 0, "torso": 1, "foot": 2}[name]


class _Data:
    def __init__(self):
        self.xfrc_applied = np.zeros((3, 6))


class _Sim:
    def __init__(self, model):
        self.model, self.data = model, _Data()


class FakeMujocoEnv(gym.Env):
    """Falls after 5 steps iff torso mass >= 1.5x nominal; otherwise survives."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float64)
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def __init__(self):
        self.model = _Model()
        self.sim = _Sim(self.model)
        self.t = 0
        self.pushes = []

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(3)

    def step(self, a):
        self.t += 1
        self.pushes.append(self.sim.data.xfrc_applied[1, 0])
        done = self.model.body_mass[1] >= 1.5 * 3.5 - 1e-9 and self.t >= 5
        return np.full(3, self.t, dtype=float), 1.0, bool(done), {}


def make_chain(tmp_path, condition_space, impulse_prob=0.0, max_steps=20):
    from env.gym_utils.wrapper.mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper
    from env.gym_utils.wrapper.mujoco_disturbance import MujocoDisturbanceWrapper
    from env.gym_utils.wrapper.multi_step import MultiStep

    norm = tmp_path / "normalization.npz"
    np.savez(norm, obs_min=-np.ones(3), obs_max=np.ones(3) * 100, action_min=-np.ones(2), action_max=np.ones(2))
    raw = FakeMujocoEnv()
    env = TimeLimit(raw, max_episode_steps=max_steps)
    env = MujocoLocomotionLowdimWrapper(env, normalization_path=str(norm))
    env = MujocoDisturbanceWrapper(
        env, condition_space=condition_space, impulse_prob=impulse_prob, impulse_duration=3,
        max_episode_steps=max_steps,
    )
    env = MultiStep(env, n_obs_steps=1, n_action_steps=4, max_episode_steps=max_steps, reset_within_step=True)
    env.seed(0)
    return env, raw


def run_episodes(env, n_steps):
    outcomes = []
    env.reset()
    for _ in range(n_steps):
        _, _, term, trunc, info = env.step(np.zeros((4, 2)))
        if "fas_ep_cond" in info:
            outcomes.append(
                (int(info["fas_ep_cond"][-1]), float(info["fas_ep_return"][-1]),
                 int(info["fas_ep_len"][-1]), bool(info["fas_ep_terminated"][-1]), term, trunc)
            )
    return outcomes


def test_wrapper_applies_condition_and_reports_outcome(tmp_path):
    space = {"mass_scale": [1.0, 1.5], "friction_scale": [0.5, 1.0]}
    env, raw = make_chain(tmp_path, space)
    # call through MultiStep (gym.Wrapper forwards attributes), like venv.call_sync does
    assert env.get_num_conditions() == 4
    env.set_condition_schedule([3, 0, 0])  # (1.5, 1.0) then (1.0, 0.5) twice
    env.reset()
    assert raw.model.body_mass[1] == pytest.approx(3.5 * 1.5)
    assert raw.model.geom_friction[0, 0] == pytest.approx(0.9)
    outcomes = []
    for _ in range(8):
        _, _, term, trunc, info = env.step(np.zeros((4, 2)))
        if "fas_ep_cond" in info:
            outcomes.append((int(info["fas_ep_cond"][-1]), bool(info["fas_ep_terminated"][-1]),
                             int(info["fas_ep_len"][-1]), term, trunc))
    # episode 1: heavy torso -> falls at t=5; episode 2: nominal mass -> time limit at t=20.
    # Note upstream MultiStep reports terminated=True on time-limit too, which is why
    # the wrapper tracks fas_ep_terminated itself.
    assert outcomes[0] == (3, True, 5, True, False)
    assert outcomes[1] == (0, False, 20, True, True)
    assert raw.model.geom_friction[0, 0] == pytest.approx(0.45)  # cond 0 friction applied
    assert raw.model.body_mass[1] == pytest.approx(3.5)  # mass restored to nominal


def test_wrapper_samples_from_probs(tmp_path):
    env, _ = make_chain(tmp_path, {"mass_scale": [1.0, 1.25, 1.5]}, max_steps=4)
    env.set_condition_probs([0.0, 0.0, 1.0])
    out = run_episodes(env, 20)
    assert len(out) > 0 and all(o[0] == 2 for o in out)
    env.set_condition_probs([1.0, 0.0, 0.0])
    out = run_episodes(env, 20)
    assert all(o[0] == 0 for o in out)


def test_wrapper_impulse_is_applied_and_cleared(tmp_path):
    env, raw = make_chain(tmp_path, {"impulse_force": [200.0]}, impulse_prob=1.0, max_steps=40)
    env.reset()
    for _ in range(3):
        env.step(np.zeros((4, 2)))
    pushes = np.abs(np.array(raw.pushes))
    assert pushes.max() == pytest.approx(200.0)
    assert (pushes == 0).any()  # gaps between pushes: force is cleared after duration
    env.set_condition_schedule([0])
    env.reset()
    assert np.all(raw.sim.data.xfrc_applied == 0)


# --------------------------------------------------------------------- configs
@pytest.fixture
def hydra_env(monkeypatch):
    import math
    from omegaconf import OmegaConf

    monkeypatch.setenv("DPPO_LOG_DIR", "/tmp/log")
    monkeypatch.setenv("DPPO_DATA_DIR", "/tmp/data")
    monkeypatch.setenv("DPPO_WANDB_ENTITY", "test")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def compose(config_dir, name):
    from hydra import compose as hcompose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hcompose(config_name=name, return_hydra_config=True)
    from hydra.core.hydra_config import HydraConfig

    HydraConfig.instance().set_config(cfg)  # needed for ${now:}
    OmegaConf.set_readonly(cfg, False)
    with open_dict(cfg):
        del cfg["hydra"]  # read-only node; not needed by the agents
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize("task", TASKS)
def test_finetune_configs_compose(hydra_env, task):
    d = os.path.join(REPO, "cfg/gym/finetune", task)
    base = compose(d, "ft_ppo_diffusion_mlp")
    fas = compose(d, "ft_ppo_diffusion_mlp_fas")
    uni = compose(d, "ft_ppo_diffusion_mlp_uniform")
    for cfg in (fas, uni):
        assert cfg._target_.endswith("TrainPPODiffusionFASAgent")
        assert "mujoco_disturbance" in cfg.env.wrappers
        # identical DPPO hyperparameters to the upstream config
        assert cfg.train == base.train and cfg.model == base.model
        assert cfg.base_policy_path == base.base_policy_path
    assert fas.fas.beta == 0.5 and uni.fas.beta == 0.0
    assert fas.env.wrappers == uni.env.wrappers
    assert fas.name != uni.name
    assert len(ConditionSpace(fas.env.wrappers.mujoco_disturbance.condition_space)) == 36


@pytest.mark.parametrize("task", TASKS)
def test_eval_configs_compose(hydra_env, task):
    d = os.path.join(REPO, "cfg/gym/eval", task)
    ft = compose(os.path.join(REPO, "cfg/gym/finetune", task), "ft_ppo_diffusion_mlp")
    ev = compose(d, "eval_diffusion_mlp_disturbance")
    ood = compose(d, "eval_diffusion_mlp_disturbance_ood")
    assert ev.base_policy_path == ft.base_policy_path  # fixes upstream eval path bugs
    assert ev.env_name == ft.env_name
    assert ev._target_.endswith("EvalDiffusionDisturbanceAgent")
    assert len(ConditionSpace(ev.env.wrappers.mujoco_disturbance.condition_space)) == 36
    assert len(ConditionSpace(ood.env.wrappers.mujoco_disturbance.condition_space)) == 8
    assert "ood" in ood.name and "train" in ev.name


# --------------------------------------------------------------------- agent hooks
class FakeVenv:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.schedules = None
        self.probs = None

    def call_sync_arg(self, method, arg, values):
        assert method == "set_condition_schedule"
        self.schedules = values

    def call_sync(self, method, **kw):
        if method == "set_condition_probs":
            self.probs = np.asarray(kw["probs"])
        return [(None, True)] * self.n_envs


def test_fas_agent_hooks(tmp_path):
    pytest.importorskip("torch")
    from agent.finetune.train_ppo_diffusion_fas_agent import TrainPPODiffusionFASAgent

    a = object.__new__(TrainPPODiffusionFASAgent)  # skip env / model construction
    a.cond_space = ConditionSpace({"mass_scale": [1.0, 1.5], "impulse_force": [0.0, 100.0, 200.0]})
    a.n_conditions = 6
    a.n_envs = 4
    a.venv = FakeVenv(4)
    a.sampler = FailureAwareSampler(6, alpha=1.0, beta=1.0, delta=0.01, ema_rate=1.0)
    a.fas_update_from = {"eval", "train"}
    a.success_criterion, a.success_return_threshold = "survive", None
    a.eval_schedule_len = 3
    a._eval_offset = 0
    a._episodes, a._probs, a.fas_history = [], a.sampler.probs(), []
    a.fas_history_path = str(tmp_path / "fas_history.pkl")
    a.use_wandb, a.itr = False, 0

    a.on_iteration_start(eval_mode=True)
    covered = {c for s in a.venv.schedules for c in s}
    assert covered == set(range(6))  # every bin is evaluated
    np.testing.assert_allclose(a.venv.probs, np.full(6, 1 / 6))

    # bin 5 always fails, every other bin succeeds
    def info(c, term):
        return {"fas_ep_cond": np.array([c]), "fas_ep_return": np.array([1.0]),
                "fas_ep_len": np.array([10]), "fas_ep_terminated": np.array([term])}
    a.on_env_step(0, [info(c, c == 5) for c in range(4)], None, True)
    a.on_env_step(1, [info(4, False), info(5, True), {}, {}], None, True)
    a.on_rollout_end(eval_mode=True)
    assert a.sampler.scores[5] == 1.0 and a.sampler.scores[0] == 0.0

    a.itr = 1
    a.on_iteration_start(eval_mode=False)
    assert all(len(s) == 0 for s in a.venv.schedules)  # eval schedule cleared
    assert a.venv.probs.argmax() == 5 and a.venv.probs[5] > 0.9

    with open(a.fas_history_path, "rb") as f:
        hist = pickle.load(f)
    assert hist["history"][0]["source"] == "eval" and len(hist["history"][0]["cond"]) == 6


def test_gym_config_targets_exist():
    """Every _target_ in cfg/gym must point to an existing module file (upstream had stale ones)."""
    import glob
    import re

    missing = set()
    for f in glob.glob(os.path.join(REPO, "cfg/gym/**/*.yaml"), recursive=True):
        for target in re.findall(r"_target_:\s*([\w.]+)", open(f).read()):
            module = target.rsplit(".", 1)[0]
            if not os.path.exists(os.path.join(REPO, *module.split(".")) + ".py"):
                missing.add((os.path.relpath(f, REPO), target))
    assert not missing, sorted(missing)
