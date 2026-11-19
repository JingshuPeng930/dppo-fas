"""
RQ3 follow-up: how stale are the saved RFT successes, per task?

For each source run of Table 2, load the final RL policy (state_199, deterministic sampling as in eval) and
the pretrained policy, and query both on the states of the saved successes (success_trajs/itr_*.npz). For
each saved chunk, report the squared distance (mean over action entries, normalized units) between
  - the executed action (clipped to +-1, as RFT trains on it) and the final policy's action  -> d_final
  - the pretrained policy's action and the final policy's action                           -> d_pre_final
binned by the RL iteration the chunk was collected at. If walker2d's early successes are much closer to its
final policy (relative to the late-iteration noise floor) than hopper's / halfcheetah's, staleness explains
why RFT recovers most of the gain on walker2d only.

  usage: python scripts/fas/rft_staleness.py   (on gpu3, after source ~/dppo_fas/activate.sh)
  output: ~/dppo_fas/figs/rft_staleness.json + table on stdout
"""

import glob
import json
import math
import os

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = os.path.join(os.environ["DPPO_LOG_DIR"], "gym-finetune")
PER_FILE = 300  # chunks sampled per saved iteration
BIN = 25  # iterations per row
SOURCES = {  # task: (env, {method: rl_key}, seeds), as in Table 2
    "walker2d-v2": ("walker2d-medium-v2", {"fas": "rq3fas_a3.0_b0.9_ta4", "uniform": "rq3uniform_ta4"}, [0, 1, 2]),
    "hopper-v2": ("hopper-medium-v2", {"fas": "rq3fas_gv2_a3.0_b0.9_ta4", "uniform": "rq3uniform_gv2_ta4"}, [3, 4, 5]),
    "halfcheetah-v2": ("halfcheetah-medium-v2", {"fas": "rq3fas_a3.0_b0.9_ta4", "uniform": "rq3uniform_ta4"}, [5, 6, 7]),
}


def load_policy(task, ckpt, ft):
    with initialize_config_dir(config_dir=os.path.join(REPO, "cfg/gym/eval", task), version_base=None):
        ov = [f"ft_denoising_steps={ft}"] + ([f"base_policy_path={ckpt}"] if ckpt else [])  # None = the pretrained policy
        cfg = compose("eval_diffusion_mlp_disturbance", overrides=ov)
    return instantiate(cfg.model).eval(), cfg


@torch.no_grad()
def act(model, obs, dev):
    out = []
    for i in range(0, len(obs), 4096):
        s = torch.from_numpy(obs[i : i + 4096]).float().to(dev)
        out.append(model(cond={"state": s}, deterministic=True).trajectories.cpu().numpy())
    return np.concatenate(out)


def main():
    rng = np.random.default_rng(0)
    res = {}
    for task, (env, keys, seeds) in SOURCES.items():
        pre = None
        for m, k in keys.items():
            for s in seeds:
                d = sorted(glob.glob(f"{L}/{env}_ppo_diffusion_mlp_{k}_*/*_{s}"))[-1]
                final, cfg = load_policy(task, f"{d}/checkpoint/state_199.pt", 10)
                if pre is None:
                    pre, _ = load_policy(task, None, 0)
                    rl_pre = OmegaConf.load(f"{d}/.hydra/config.yaml").base_policy_path
                    assert os.path.basename(os.path.dirname(os.path.dirname(rl_pre))) in str(_.base_policy_path), rl_pre
                dev = cfg.device
                rows = []
                for f in sorted(glob.glob(f"{d}/success_trajs/itr_*.npz")):
                    z = np.load(f)
                    n = len(z["act"])
                    idx = rng.choice(n, min(n, PER_FILE), replace=False)
                    obs, a = z["obs"][idx], np.clip(z["act"][idx], -1, 1)
                    af, ap = act(final, obs, dev), act(pre, obs, dev)
                    rows.append((int(z["itr"]), n, float(((a - af) ** 2).mean()), float(((ap - af) ** 2).mean())))
                res[f"{task}/{m}/{s}"] = rows
                print(task, m, s, len(rows), "files", flush=True)
                del final
                torch.cuda.empty_cache()
    out = os.path.expanduser("~/dppo_fas/figs/rft_staleness.json")
    json.dump(res, open(out, "w"))
    report(res)


def report(res):
    for task in SOURCES:
        print(f"\n== {task}: chunk-weighted mean per {BIN}-iteration bin (over methods x seeds) ==")
        print(f"{'itr':>8} {'chunks%':>8} {'d_final':>8} {'d_pre_fin':>9} {'d_final/late':>12}")
        rows = np.array([r for k, v in res.items() if k.startswith(task + "/") for r in v])
        late = rows[rows[:, 0] >= 150]
        floor = np.average(late[:, 2], weights=late[:, 1])
        for b in range(0, 200, BIN):
            r = rows[(rows[:, 0] >= b) & (rows[:, 0] < b + BIN)]
            if len(r) == 0:
                continue
            w = r[:, 1]
            df = np.average(r[:, 2], weights=w)
            print(f"{b:>3}-{b + BIN - 1:<4} {w.sum() / rows[:, 1].sum():>8.1%} {df:>8.4f} "
                  f"{np.average(r[:, 3], weights=w):>9.4f} {df / floor:>12.2f}")
        allw = np.average(rows[:, 2], weights=rows[:, 1])
        print(f"{'all':>8} {'':>8} {allw:>8.4f} {np.average(rows[:, 3], weights=rows[:, 1]):>9.4f} {allw / floor:>12.2f}")


if __name__ == "__main__":
    main()
