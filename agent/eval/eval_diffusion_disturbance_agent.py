"""
Evaluate a pre-trained / fine-tuned diffusion policy on a fixed grid of test
conditions (mujoco_disturbance wrapper), n_episodes_per_condition episodes per bin.

Every method (pretrained, DPPO, DPPO-uniform-DR, DPPO-FAS) should be evaluated
with this agent and the same eval config, so test conditions are identical.

Outputs in logdir:
  result.npz   -- per-condition arrays + summary scalars
  result.json  -- human-readable summary incl. per-parameter marginals
"""

import json
import logging
import math

import numpy as np
import torch
from omegaconf import OmegaConf

from agent.eval.eval_agent import EvalAgent
from util.fas import ConditionSpace, episode_success
from util.timer import Timer

log = logging.getLogger(__name__)


class EvalDiffusionDisturbanceAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        dist_cfg = OmegaConf.to_container(cfg.env.wrappers.mujoco_disturbance, resolve=True)
        self.cond_space = ConditionSpace(dist_cfg["condition_space"])
        self.n_conditions = len(self.cond_space)
        self.n_per_cond = cfg.n_episodes_per_condition
        self.success_criterion = cfg.success.criterion
        self.success_return_threshold = cfg.success.get("return_threshold", None)
        self.success_min_steps = cfg.success.get("min_steps", None)
        self.result_json_path = self.result_path.replace(".npz", ".json")

    def run(self):
        timer = Timer()
        E, C, N = self.n_envs, self.n_conditions, self.n_per_cond
        queue = [c for c in range(C) for _ in range(N)]
        schedules = [queue[i::E] for i in range(E)]
        n_target = np.array([len(s) for s in schedules])
        self.venv.call_sync_arg("set_condition_schedule", "schedule", schedules)

        self.model.eval()
        prev_obs_venv = self.reset_env_all()
        n_done = np.zeros(E, dtype=int)
        episodes = []  # (cond, return, len, terminated)
        steps_per_ep = math.ceil(self.max_episode_steps / self.act_steps)
        max_steps = int(n_target.max() * steps_per_ep * 1.1) + 10

        for step in range(max_steps):
            if np.all(n_done >= n_target):
                break
            if step % 50 == 0:
                log.info(f"step {step}/{max_steps} | episodes {len(episodes)}/{len(queue)}")
            with torch.no_grad():
                cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                samples = self.model(cond=cond, deterministic=True)
                action_venv = samples.trajectories.cpu().numpy()[:, : self.act_steps]
            prev_obs_venv, _, _, _, info_venv = self.venv.step(action_venv)
            for i, info in enumerate(info_venv):
                if "fas_ep_cond" in info and n_done[i] < n_target[i]:
                    c = int(np.asarray(info["fas_ep_cond"]).reshape(-1)[-1])
                    assert c == schedules[i][n_done[i]], "schedule / outcome mismatch"
                    episodes.append(
                        (
                            c,
                            float(np.asarray(info["fas_ep_return"]).reshape(-1)[-1]),
                            int(np.asarray(info["fas_ep_len"]).reshape(-1)[-1]),
                            bool(np.asarray(info["fas_ep_terminated"]).reshape(-1)[-1]),
                        )
                    )
                    n_done[i] += 1
        if len(episodes) < len(queue):
            log.warning(f"Only {len(episodes)}/{len(queue)} episodes finished")

        cond, ret, length, term = map(np.array, zip(*episodes))
        success = episode_success(
            ret, term, self.success_criterion, self.success_return_threshold,
            ep_len=length, min_steps=self.success_min_steps,
        )
        count = np.bincount(cond, minlength=C)
        with np.errstate(invalid="ignore", divide="ignore"):
            succ_c = np.bincount(cond, weights=success.astype(float), minlength=C) / count
            ret_c = np.bincount(cond, weights=ret, minlength=C) / count

        summary = {
            "num_episodes": int(len(cond)),
            "success_rate": float(success.mean()),
            "balanced_success": float(np.nanmean(succ_c)),
            "worst_bin_success": float(np.nanmin(succ_c)),
            "avg_return": float(ret.mean()),
            "balanced_return": float(np.nanmean(ret_c)),
            "worst_bin_return": float(np.nanmin(ret_c)),
            "time": float(timer()),
        }
        marginals = {
            name: {
                "success": self.cond_space.marginal(succ_c, name),
                "return": self.cond_space.marginal(ret_c, name),
            }
            for name in self.cond_space.names
        }
        per_condition = [
            {
                "cond": c,
                **self.cond_space.params(c),
                "n": int(count[c]),
                "success": float(succ_c[c]),
                "return": float(ret_c[c]),
            }
            for c in range(C)
        ]
        log.info("eval: " + " | ".join(f"{k} {v:.4f}" for k, v in summary.items()))
        for name, m in marginals.items():
            log.info(f"  success by {name}: " + ", ".join(f"{v:g}->{s:.3f}" for v, s in m["success"].items()))

        np.savez(
            self.result_path,
            cond=cond,
            ret=ret,
            length=length,
            terminated=term,
            success=success,
            success_per_condition=succ_c,
            return_per_condition=ret_c,
            count_per_condition=count,
            condition_names=np.array(self.cond_space.names),
            **summary,
        )
        with open(self.result_json_path, "w") as f:
            json.dump(
                {
                    "base_policy_path": self.cfg.base_policy_path,
                    "summary": summary,
                    "marginals": {
                        n: {k: {str(v): s for v, s in d.items()} for k, d in m.items()}
                        for n, m in marginals.items()
                    },
                    "per_condition": per_condition,
                },
                f,
                indent=2,
            )
        log.info(f"Saved results to {self.result_path} and {self.result_json_path}")
