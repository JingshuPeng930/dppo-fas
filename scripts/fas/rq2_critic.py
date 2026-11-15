"""
RQ2: per-condition critic error of a fine-tuned DPPO(-FAS) checkpoint.

For every condition bin, roll out the *training-mode* (stochastic) policy and compare the critic V(s_t)
with the Monte-Carlo discounted return G_t = sum_k gamma^k r_{t+k} (decision-level, gamma = train.gamma).

  * G_t runs to the end of the episode with no bootstrap, at a fall *and* at the time limit: that is the
    target DPPO's GAE regresses the critic toward, because upstream MultiStep reports gym's TimeLimit
    `done` as terminated=True (nonterminal = 1 - terminated in the GAE). The critic sees no time feature,
    so in bins where the policy survives to the limit part of the error is irreducible (time ambiguity);
    compare the same bin across methods rather than bins against each other.
  * DPPO trains the critic on rewards divided by a running std that is not checkpointed, so V is only
    known up to a scale: a single least-squares scale a (V ~ a G, pooled over bins) is fitted per checkpoint
    and errors are reported as |V/a - G| / mean|G|, i.e. relative to the global return scale.

    python scripts/fas/rq2_critic.py <run_dir> <itr> [--episodes 2] [--out <json>]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", lambda x: int(np.ceil(x)), replace=True)
OmegaConf.register_new_resolver("round_down", lambda x: int(np.floor(x)), replace=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("itr", type=int)
    ap.add_argument("--episodes", type=int, default=2, help="episodes per bin")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(a.run_dir, ".hydra", "config.yaml"))
    n_bins = int(np.prod([len(v) for v in cfg.env.wrappers.mujoco_disturbance.condition_space.values()]))
    T_lim = int(cfg.env.max_episode_steps)
    tmp = os.path.join(os.environ["DPPO_LOG_DIR"], "rq2_tmp", f"{os.path.basename(os.path.dirname(a.run_dir))}_{os.path.basename(a.run_dir)}_{a.itr}")
    OmegaConf.update(cfg, "logdir", tmp)
    OmegaConf.update(cfg, "wandb", None)
    OmegaConf.update(cfg, "env.n_envs", n_bins)
    OmegaConf.update(cfg, "train.logprob_batch_size", n_bins * 250)  # PPO asserts divisibility by n_envs
    OmegaConf.update(cfg, "fas.save_success_trajs", False)
    OmegaConf.update(cfg, "_target_", "agent.finetune.train_ppo_diffusion_fas_agent.TrainPPODiffusionFASAgent")
    OmegaConf.resolve(cfg)

    from agent.finetune.train_ppo_diffusion_fas_agent import TrainPPODiffusionFASAgent

    agent = TrainPPODiffusionFASAgent(cfg)
    data = torch.load(os.path.join(a.run_dir, "checkpoint", f"state_{a.itr}.pt"), weights_only=True)
    agent.model.load_state_dict(data["model"])
    agent.model.eval()
    gamma, act_steps = float(cfg.train.gamma), int(cfg.act_steps)
    max_dec = T_lim // act_steps

    # env i runs bin i for every episode
    agent.venv.call_sync_arg("set_condition_schedule", "schedule", [[i] * 10000 for i in range(n_bins)])
    obs = agent.reset_env_all()
    n_steps = a.episodes * max_dec + 1  # >= `episodes` complete episodes per bin
    R, D, V = [], [], []
    falls = np.zeros(n_bins, int)
    n_eps = np.zeros(n_bins, int)
    for step in range(n_steps):
        with torch.no_grad():
            cond = {"state": torch.from_numpy(obs["state"]).float().to(agent.device)}
            V.append(agent.model.critic(cond).cpu().numpy().flatten())
            act = agent.model(cond=cond, deterministic=False, return_chain=True).trajectories.cpu().numpy()[:, :act_steps]
        obs, rew, term, trunc, info = agent.venv.step(act)
        for j, inf in enumerate(info):  # every episode must have run the bin its env was assigned
            if "fas_ep_cond" in inf:
                assert int(np.asarray(inf["fas_ep_cond"]).reshape(-1)[-1]) == j, (j, inf["fas_ep_cond"])
                n_eps[j] += 1
                falls[j] += bool(np.asarray(inf["fas_ep_terminated"]).reshape(-1)[-1])
        R.append(rew)
        D.append(term | trunc)  # episode end (fall or time limit), both terminal for the GAE
    V, R, D = np.array(V), np.array(R), np.array(D)  # (T, n_bins)
    agent.venv.close()

    per_bin = []
    allV, allG, allB = [], [], []
    for i in range(n_bins):
        start = 0
        for t in range(n_steps):
            if D[t, i]:  # only complete episodes (an unfinished tail has no well-defined target)
                end = t + 1
                r, v = R[start:end, i], V[start:end, i]
                G = np.zeros(len(r))
                acc = 0.0
                for k in range(len(r) - 1, -1, -1):
                    acc = r[k] + gamma * acc
                    G[k] = acc
                allV.append(v); allG.append(G); allB.append(np.full(len(r), i))
                start = end
        per_bin.append(dict(bin=i, episodes=int(n_eps[i]), falls=int(falls[i])))
    allV, allG, allB = map(np.concatenate, (allV, allG, allB))
    scale = float((allV * allG).sum() / (allG * allG).sum())  # V ~ scale * G
    err = allV / scale - allG
    gnorm = float(np.abs(allG).mean())
    for d in per_bin:
        m = allB == d["bin"]
        d.update(n_states=int(m.sum()),
                 rel_abs_err=float(np.abs(err[m]).mean() / gnorm) if m.any() else None,
                 rel_bias=float(err[m].mean() / gnorm) if m.any() else None,
                 mean_return=float(allG[m].mean()) if m.any() else None,
                 corr=float(np.corrcoef(allV[m], allG[m])[0, 1]) if m.sum() > 2 else None,
                 fall_rate=d["falls"] / max(d["episodes"], 1))
    res = dict(run_dir=a.run_dir, itr=a.itr, scale=scale, return_norm=gnorm,
               overall_rel_abs_err=float(np.abs(err).mean() / gnorm),
               overall_corr=float(np.corrcoef(allV, allG)[0, 1]), per_bin=per_bin)
    out = a.out or os.path.join(a.run_dir, f"rq2_critic_{a.itr}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"{out}: scale {scale:.4f} overall rel err {res['overall_rel_abs_err']:.3f} corr {res['overall_corr']:.3f}")


if __name__ == "__main__":
    main()
