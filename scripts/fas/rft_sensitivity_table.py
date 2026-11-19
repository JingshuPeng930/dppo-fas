"""Summarize scripts/fas/run_rft_sensitivity.sh: train-grid balanced success of the final RL policies
(FAS and uniform sources pooled) and of the pretrained policy under N(0, sigma^2) action noise, and the
loss as a share of each task's RL gain (RL(0) - pretrained(0))."""
import glob
import json
import os

import numpy as np

O = os.path.join(os.environ["DPPO_LOG_DIR"], "gym-sensitivity")
SIGS = ["0.0", "0.1", "0.2", "0.3"]


def bs(pattern):
    v = [json.load(open(f))["summary"]["balanced_success"] if "summary" in json.load(open(f))
         else json.load(open(f))["balanced_success"] for f in glob.glob(f"{O}/{pattern}/result.json")]
    return np.array(v)


for env in ["walker2d-medium-v2", "hopper-medium-v2", "halfcheetah-medium-v2"]:
    pre0 = bs(f"{env}_pretrained_*_sig0.0").mean()
    rl0 = bs(f"{env}_rq3*_sig0.0").mean()
    print(f"\n{env}: pretrained {pre0:.3f}, RL {rl0:.3f}, gain {rl0 - pre0:.3f}")
    print(f"{'sigma':>6} {'RL':>12} {'pre':>12} {'RL loss / gain':>15}")
    for s in SIGS:
        rl, pre = bs(f"{env}_rq3*_sig{s}"), bs(f"{env}_pretrained_*_sig{s}")
        print(f"{s:>6} {rl.mean():.3f}±{rl.std(ddof=1) / np.sqrt(len(rl)):.3f} (n={len(rl)}) "
              f"{pre.mean():.3f} (n={len(pre)}) {(rl0 - rl.mean()) / (rl0 - pre0):>8.2f}")
