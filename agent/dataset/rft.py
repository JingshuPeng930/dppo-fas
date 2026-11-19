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
    def __init__(self, run_dirs, horizon_steps, cond_steps=1, max_itr=None, action_clip=1.0,
                 demo_path=None, demo_weight=0.0, min_ep_len=None, min_return=None, match_full_steps=True,
                 min_itr=None, draws_per_epoch=None, device="cuda:0"):
        """run_dirs: one fine-tuning run dir, a glob, or a list of either (e.g. one per seed).

        demo_path / demo_weight: optionally mix the original pretraining demos back in, as openpi-comet's
        RFT does (demo:rollout = 0.6:0.4). A fraction demo_weight of every epoch is demo chunks drawn
        uniformly at random; the rollout chunks are all visited once per epoch. Epochs get longer by
        1 / (1 - demo_weight), so scale n_epochs down to keep the gradient-step budget.

        min_ep_len / min_return: keep only successes with a margin over the success criterion, i.e. episodes
        of at least min_ep_len env steps / with return >= min_return (RQ3 margin test). With
        match_full_steps, an epoch still has as many rollout draws as the unfiltered data (the kept chunks
        are cycled), so the gradient-step budget does not shrink with the filter.

        min_itr: only use episodes collected at or after this RL iteration (RQ3 staleness test); like the
        episode filter, it does not shrink the step budget when match_full_steps is set.

        draws_per_epoch: fix the number of rollout draws per epoch (overrides match_full_steps), e.g. to
        give a distillation dataset the same gradient-step budget as an RFT run.

        action_clip: the saved chunks are the sampled actions *before* the env clipped them (DPPO's
        exploration noise reaches ~1.3 in normalized units, and normalized +-1 is the MuJoCo control
        range for these tasks). Clipping recovers the executed action and matches the pretraining
        data range / the sampler's denoised_clip_value. null = keep raw samples.
        """
        if isinstance(run_dirs, str):
            run_dirs = [run_dirs]
        files = []
        for pattern in run_dirs:
            for d in sorted(glob.glob(pattern)):
                files += sorted(glob.glob(os.path.join(d, "success_trajs", "itr_*.npz")))
        assert files, f"no success_trajs/itr_*.npz under {run_dirs}"
        obs, act, cond, n_eps, n_full = [], [], [], 0, 0
        for f in files:
            z = np.load(f)
            n_full += len(z["act"])
            if max_itr is not None and int(z["itr"]) > max_itr:
                continue
            if min_itr is not None and int(z["itr"]) < min_itr:
                continue
            eps = z["episodes"]  # ep id, cond, return, env-step length, terminated
            keep = np.ones(len(eps), bool)
            if min_ep_len is not None:
                keep &= eps[:, 3] >= min_ep_len
            if min_return is not None:
                keep &= eps[:, 2] >= min_return
            m = np.isin(z["ep"], eps[keep, 0].astype(np.int64))
            obs.append(z["obs"][m])
            act.append(z["act"][m])
            cond.append(z["cond"][m])
            n_eps += int(keep.sum())
        obs, act = np.concatenate(obs), np.concatenate(act)
        assert len(act) > 0, "the episode filter removed every chunk"
        if action_clip is not None:
            log.info(f"RFT dataset: clipping {(np.abs(act) > action_clip).mean():.1%} of action entries to +-{action_clip}")
            act = np.clip(act, -action_clip, action_clip)
        assert obs.shape[1] == cond_steps, (obs.shape, cond_steps)
        assert act.shape[1] == horizon_steps, (act.shape, horizon_steps)
        self.cond = np.concatenate(cond).astype(np.int64)
        self.device = device
        self.states = torch.from_numpy(obs).float().to(device)  # (N, cond_steps, obs_dim)
        self.actions = torch.from_numpy(act).float().to(device)  # (N, horizon_steps, action_dim)
        self.n_kept = len(self.actions)
        self.n_rollout = n_full if match_full_steps else self.n_kept
        if draws_per_epoch is not None:
            self.n_rollout = int(draws_per_epoch)
        if self.n_kept != self.n_rollout:
            log.info(f"RFT dataset: filter kept {self.n_kept}/{n_full} chunks; {self.n_rollout} rollout draws per epoch")
        assert 0.0 <= demo_weight < 1.0, demo_weight
        self.demo = None
        self.n_demo = 0
        if demo_path is not None and demo_weight > 0:
            from agent.dataset.sequence import StitchedSequenceDataset

            self.demo = StitchedSequenceDataset(demo_path, horizon_steps=horizon_steps,
                                                cond_steps=cond_steps, device=device)
            self.n_demo = int(round(self.n_rollout * demo_weight / (1.0 - demo_weight)))
        log.info(
            f"RFT dataset: {n_eps} successful episodes, {self.n_rollout} chunks from {len(files)} files; "
            f"bins covered {len(np.unique(self.cond))}; + {self.n_demo} demo draws per epoch "
            f"(demo_weight {demo_weight if self.demo is not None else 0})"
        )

    def __len__(self):
        return self.n_rollout + self.n_demo

    def __getitem__(self, idx):
        if idx < self.n_rollout:
            idx = idx % self.n_kept
            return Batch(self.actions[idx], {"state": self.states[idx]})
        return self.demo[int(torch.randint(len(self.demo), ()))]
