"""
DPPO fine-tuning with Failure-Aware Sampling (FAS) over training conditions.

The same class runs the uniform domain-randomization baseline (fas.beta = 0),
so DPPO-uniform and DPPO-FAS use identical code paths and differ only in p(c).

Per iteration:
  * train iteration -> envs sample conditions from p(c) (Eq. 2) at every reset
  * eval iteration  -> envs follow a round-robin schedule so all bins are covered,
                       using the deterministic (DDIM) policy, as in upstream DPPO eval
  * finished episodes (cond, return, terminated) are turned into binary success and
    used to update the EMA failure score s_c (Eq. 1) from the sources in fas.update_from

Requires the `mujoco_disturbance` env wrapper.
"""

import os
import pickle
import logging

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from util.fas import ConditionSpace, FailureAwareSampler, episode_success

log = logging.getLogger(__name__)


class TrainPPODiffusionFASAgent(TrainPPODiffusionAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert "mujoco_disturbance" in cfg.env.wrappers, "FAS needs env.wrappers.mujoco_disturbance"
        dist_cfg = OmegaConf.to_container(cfg.env.wrappers.mujoco_disturbance, resolve=True)
        self.cond_space = ConditionSpace(dist_cfg["condition_space"])
        self.n_conditions = len(self.cond_space)
        n_env_side = [r[0] for r in self.venv.call_sync("get_num_conditions")]
        assert all(n == self.n_conditions for n in n_env_side), (n_env_side, self.n_conditions)

        fas_cfg = cfg.fas
        self.sampler = FailureAwareSampler(
            n_bins=self.n_conditions,
            alpha=fas_cfg.alpha,
            beta=fas_cfg.beta,
            delta=fas_cfg.delta,
            ema_rate=fas_cfg.ema_rate,
            init_score=fas_cfg.get("init_score", 0.5),
            score=fas_cfg.get("score", "failure"),
            target=fas_cfg.get("target", 0.66),
            kappa=fas_cfg.get("kappa", 5.0),
            temperature=fas_cfg.get("temperature", 2.0),
            eps=fas_cfg.get("eps", 1e-8),
            ema_slow=fas_cfg.get("ema_slow", 0.1),
        )
        self.fas_update_from = set(fas_cfg.get("update_from", ["eval", "train"]))
        assert self.fas_update_from <= {"eval", "train"}, self.fas_update_from
        self.success_criterion = fas_cfg.success.criterion
        self.success_return_threshold = fas_cfg.success.get("return_threshold", None)
        self.success_min_steps = fas_cfg.success.get("min_steps", None)
        self.eval_schedule_len = fas_cfg.get("eval_schedule_len", 16)
        self._eval_offset = 0
        self._episodes = []
        self._probs = self.sampler.probs()
        self.fas_history = []
        self.fas_history_path = os.path.join(self.logdir, "fas_history.pkl")

        # RQ3 (RFT): keep the successful training episodes as (policy input, executed chunk) pairs
        self.save_success_trajs = bool(fas_cfg.get("save_success_trajs", False))
        self.traj_drop_last = int(fas_cfg.get("traj_drop_last", 25))
        self.traj_dir = os.path.join(self.logdir, "success_trajs")
        self._traj_buf = [None] * self.n_envs  # None = episode started outside a clean train rollout
        self._traj_out = []
        self._n_saved_eps = 0
        self._prev_eval = False
        if self.save_success_trajs:
            assert self.act_steps == self.horizon_steps, "RFT needs the full executed chunk as BC target"
            os.makedirs(self.traj_dir, exist_ok=True)
        log.info(
            f"FAS: {self.n_conditions} condition bins over {self.cond_space.names}, "
            f"score={self.sampler.score} alpha={self.sampler.alpha} beta={self.sampler.beta} delta={self.sampler.delta} "
            f"ema_rate={self.sampler.ema_rate} update_from={sorted(self.fas_update_from)}"
        )

    # ------------------------------------------------------------------ hooks
    def on_iteration_start(self, eval_mode):
        self._episodes = []
        if self.save_success_trajs:
            self._traj_out = []
            # mirrors the reset rule in TrainPPODiffusionAgent.run (its last_itr_eval == eval_mode there)
            if eval_mode:
                self._traj_buf = [None] * self.n_envs  # eval (deterministic) episodes are not saved
            elif self.reset_at_iteration:
                self._traj_buf = [[] for _ in range(self.n_envs)]
            elif self._prev_eval:
                # envs continue the episodes the eval iteration started: drop them until they end
                self._traj_buf = [None] * self.n_envs
        self._prev_eval = eval_mode
        if eval_mode:
            # round-robin over bins so every bin is evaluated; rotate between evals
            n, E, K = self.n_conditions, self.n_envs, self.eval_schedule_len
            schedules = [
                [(self._eval_offset + i + k * E) % n for k in range(K)] for i in range(E)
            ]
            self._eval_offset = (self._eval_offset + E * K) % n
            self._probs = np.full(n, 1.0 / n)
        else:
            schedules = [[] for _ in range(self.n_envs)]
            self._probs = self.sampler.probs()
        self.venv.call_sync_arg("set_condition_schedule", "schedule", schedules)
        self.venv.call_sync("set_condition_probs", probs=self._probs)

    def on_env_step(self, step, info_venv, done_venv, eval_mode, obs=None, action=None):
        if self.save_success_trajs and not eval_mode and obs is not None:
            self._record_traj_step(info_venv, done_venv, obs, action)
        for info in info_venv:
            if "fas_ep_cond" in info:
                self._episodes.append(
                    (
                        int(np.asarray(info["fas_ep_cond"]).reshape(-1)[-1]),
                        float(np.asarray(info["fas_ep_return"]).reshape(-1)[-1]),
                        int(np.asarray(info["fas_ep_len"]).reshape(-1)[-1]),
                        bool(np.asarray(info["fas_ep_terminated"]).reshape(-1)[-1]),
                    )
                )

    def _record_traj_step(self, info_venv, done_venv, obs, action):
        done_venv = np.asarray(done_venv).reshape(-1)
        for i in range(self.n_envs):
            if self._traj_buf[i] is not None:
                self._traj_buf[i].append((np.array(obs[i], np.float32), np.array(action[i], np.float32)))
            if done_venv[i]:
                info = info_venv[i]
                if self._traj_buf[i] and "fas_ep_cond" in info:
                    last = lambda k: np.asarray(info[k]).reshape(-1)[-1]
                    cond, ret = int(last("fas_ep_cond")), float(last("fas_ep_return"))
                    length, term = int(last("fas_ep_len")), bool(last("fas_ep_terminated"))
                    ok = episode_success(
                        np.array([ret]), np.array([term]), self.success_criterion,
                        self.success_return_threshold, ep_len=np.array([length]),
                        min_steps=self.success_min_steps,
                    )[0]
                    steps = self._traj_buf[i]
                    if term and self.traj_drop_last > 0:
                        steps = steps[: -self.traj_drop_last]  # do not imitate the fall itself
                    if ok and len(steps) > 0:
                        self._traj_out.append((cond, ret, length, term, steps))
                self._traj_buf[i] = []  # the env auto-reset: next step starts a clean episode

    def _save_success_trajs(self):
        if not self._traj_out:
            return
        obs, act, cond, ep, meta = [], [], [], [], []
        for cond_id, ret, length, term, steps in self._traj_out:
            obs.append(np.stack([o for o, _ in steps]))
            act.append(np.stack([a for _, a in steps]))
            cond.append(np.full(len(steps), cond_id))
            ep.append(np.full(len(steps), self._n_saved_eps))
            meta.append((self._n_saved_eps, cond_id, ret, length, term))
            self._n_saved_eps += 1
        np.savez_compressed(
            os.path.join(self.traj_dir, f"itr_{self.itr:04d}.npz"),
            obs=np.concatenate(obs), act=np.concatenate(act), cond=np.concatenate(cond).astype(np.int16),
            ep=np.concatenate(ep).astype(np.int32), itr=np.int32(self.itr),
            episodes=np.array(meta, dtype=np.float64),  # ep id, cond, return, env-step length, terminated
        )
        log.info(f"RFT: saved {len(meta)} successful episodes ({sum(len(o) for o in obs)} chunks)")

    def on_rollout_end(self, eval_mode):
        source = "eval" if eval_mode else "train"
        if self.save_success_trajs and not eval_mode:
            self._save_success_trajs()
        if len(self._episodes) == 0:
            log.info(f"FAS [{source}]: no finished episodes this iteration")
            return
        cond, ret, length, term = map(np.array, zip(*self._episodes))
        success = episode_success(
            ret, term, self.success_criterion, self.success_return_threshold,
            ep_len=length, min_steps=self.success_min_steps,
        ).astype(float)

        if source in self.fas_update_from:
            batch_fail = self.sampler.update(cond, success)
        else:
            batch_fail = FailureAwareSampler(self.n_conditions).update(cond, success)
        batch_succ = 1.0 - batch_fail  # nan for bins without data
        covered = ~np.isnan(batch_succ)
        p = self._probs
        entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(self.n_conditions)

        stats = {
            f"fas/{source}/num_episodes": len(cond),
            f"fas/{source}/success_rate": float(success.mean()),
            f"fas/{source}/balanced_success": float(np.nanmean(batch_succ)),
            f"fas/{source}/worst_bin_success": float(np.nanmin(batch_succ)),
            f"fas/{source}/bins_covered": int(covered.sum()),
            f"fas/{source}/avg_return": float(ret.mean()),
            f"fas/{source}/avg_len": float(length.mean()),
            "fas/score_mean": float(self.sampler.scores.mean()),
            "fas/score_max": float(self.sampler.scores.max()),
            "fas/prob_max": float(p.max()),
            "fas/prob_min": float(p.min()),
            "fas/prob_entropy_normalized": float(entropy),
        }
        for name in self.cond_space.names:
            for v, s in self.cond_space.marginal(batch_succ, name).items():
                stats[f"fas/{source}/success_by_{name}/{v:g}"] = s
        if self.use_wandb:
            wandb.log(stats, step=self.itr, commit=False)
        log.info(
            f"FAS [{source}] itr {self.itr}: episodes {len(cond)} | success {success.mean():.3f} "
            f"| balanced {np.nanmean(batch_succ):.3f} | worst bin {np.nanmin(batch_succ):.3f} "
            f"| bins covered {covered.sum()}/{self.n_conditions} | p max {p.max():.3f} "
            f"| entropy {entropy:.3f}"
        )

        self.fas_history.append(
            {
                "itr": self.itr,
                "source": source,
                "probs": p.copy(),
                "scores": self.sampler.scores.copy(),
                "cond": cond,
                "success": success,
                "return": ret,
                "len": length,
                "terminated": term,
            }
        )
        with open(self.fas_history_path, "wb") as f:
            pickle.dump(
                {
                    "condition_names": self.cond_space.names,
                    "condition_values": self.cond_space.values,
                    "history": self.fas_history,
                },
                f,
            )

    # ------------------------------------------------------------- checkpoint
    def save_model(self):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            # tensors/floats only, so torch.load(weights_only=True) still works
            "fas_sampler": {
                k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in self.sampler.state_dict().items()
            },
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")
