"""
Dataset for rejection-sampling fine-tuning (RFT, RQ3): the successful training episodes that a
DPPO / DPPO-FAS run saved with fas.save_success_trajs=True (<run>/success_trajs/itr_*.npz).

Each sample is one decision step: the policy input it saw and the action chunk it executed,
already in the policy's normalized space, i.e. the same format the pretraining dataset yields.
"""

import glob
import logging
import os

import numpy as np
import torch

from agent.dataset.sequence import Batch

log = logging.getLogger(__name__)


class SuccessChunkDataset(torch.utils.data.Dataset):
    def __init__(self, run_dirs, horizon_steps, cond_steps=1, max_itr=None, device="cuda:0"):
        """run_dirs: one fine-tuning run dir, a glob, or a list of either (e.g. one per seed)."""
        if isinstance(run_dirs, str):
            run_dirs = [run_dirs]
        files = []
        for pattern in run_dirs:
            for d in sorted(glob.glob(pattern)):
                files += sorted(glob.glob(os.path.join(d, "success_trajs", "itr_*.npz")))
        assert files, f"no success_trajs/itr_*.npz under {run_dirs}"
        obs, act, cond, n_eps = [], [], [], 0
        for f in files:
            z = np.load(f)
            if max_itr is not None and int(z["itr"]) > max_itr:
                continue
            obs.append(z["obs"])
            act.append(z["act"])
            cond.append(z["cond"])
            n_eps += len(z["episodes"])
        obs, act = np.concatenate(obs), np.concatenate(act)
        assert obs.shape[1] == cond_steps, (obs.shape, cond_steps)
        assert act.shape[1] == horizon_steps, (act.shape, horizon_steps)
        self.cond = np.concatenate(cond).astype(np.int64)
        self.device = device
        self.states = torch.from_numpy(obs).float().to(device)  # (N, cond_steps, obs_dim)
        self.actions = torch.from_numpy(act).float().to(device)  # (N, horizon_steps, action_dim)
        log.info(
            f"RFT dataset: {n_eps} successful episodes, {len(self)} chunks from {len(files)} files; "
            f"bins covered {len(np.unique(self.cond))}"
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return Batch(self.actions[idx], {"state": self.states[idx]})
