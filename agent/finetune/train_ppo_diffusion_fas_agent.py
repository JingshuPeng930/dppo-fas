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
        log.info(
            f"FAS: {self.n_conditions} condition bins over {self.cond_space.names}, "
            f"score={self.sampler.score} alpha={self.sampler.alpha} beta={self.sampler.beta} delta={self.sampler.delta} "
            f"ema_rate={self.sampler.ema_rate} update_from={sorted(self.fas_update_from)}"
        )

    # ------------------------------------------------------------------ hooks
    def on_iteration_start(self, eval_mode):
        self._episodes = []
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

    def on_env_step(self, step, info_venv, done_venv, eval_mode):
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

    def on_rollout_end(self, eval_mode):
        source = "eval" if eval_mode else "train"
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
