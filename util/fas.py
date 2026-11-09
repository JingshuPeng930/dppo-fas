"""
Failure-Aware Sampling (FAS) utilities.

Pure numpy, no simulator dependency, so it can be imported both by the env
wrapper (inside env subprocesses) and by the training agent (main process).

  * ConditionSpace      -- discrete grid of training-condition bins C
  * FailureAwareSampler -- EMA failure score s_c and sampling distribution p(c)

    p(c) = beta * (s_c + delta)^alpha / sum_j (s_j + delta)^alpha + (1 - beta) / |C|

beta = 0 recovers uniform domain randomization (the DPPO baseline).
"""

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np


class ConditionSpace:
    """Cartesian product of per-parameter value lists.

    Example:
        ConditionSpace({"mass_scale": [0.75, 1.0, 1.5], "friction_scale": [0.5, 1.0]})
        -> 6 bins, bin i <-> dict(mass_scale=..., friction_scale=...)
    """

    def __init__(self, params: Dict[str, Sequence[float]]):
        if len(params) == 0:
            raise ValueError("ConditionSpace needs at least one parameter")
        self.names: List[str] = list(params.keys())
        self.values: List[List[float]] = [list(map(float, params[k])) for k in self.names]
        for name, vals in zip(self.names, self.values):
            if len(vals) == 0:
                raise ValueError(f"Parameter '{name}' has no values")
        self.shape = tuple(len(v) for v in self.values)
        self._grid = list(itertools.product(*self.values))

    def __len__(self):
        return len(self._grid)

    def params(self, idx: int) -> Dict[str, float]:
        return dict(zip(self.names, self._grid[int(idx)]))

    def index(self, **kwargs) -> int:
        """Inverse of params(); every parameter must be given exactly."""
        multi = tuple(self.values[i].index(float(kwargs[n])) for i, n in enumerate(self.names))
        return int(np.ravel_multi_index(multi, self.shape))

    def label(self, idx: int) -> str:
        return ",".join(f"{k}={v:g}" for k, v in self.params(idx).items())

    def marginal(self, per_bin: np.ndarray, name: str, weights: Optional[np.ndarray] = None):
        """Average a per-bin quantity over all other parameters.

        Returns {value: mean}. Bins with zero weight (e.g. no data) are skipped.
        """
        per_bin = np.asarray(per_bin, dtype=float).reshape(self.shape)
        w = np.ones_like(per_bin) if weights is None else np.asarray(weights, float).reshape(self.shape)
        axis = self.names.index(name)
        other = tuple(i for i in range(len(self.shape)) if i != axis)
        num = np.nansum(per_bin * w, axis=other)
        den = np.sum(w * ~np.isnan(per_bin), axis=other)
        out = {}
        for v, n, d in zip(self.values[axis], num, den):
            out[v] = float(n / d) if d > 0 else float("nan")
        return out


class FailureAwareSampler:
    """Tracks an EMA failure score per bin and produces p(c) (Eq. 1-2)."""

    def __init__(
        self,
        n_bins: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        delta: float = 0.05,
        ema_rate: float = 0.3,
        init_score: float = 0.5,
    ):
        assert n_bins > 0
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        assert delta > 0, "delta must be > 0"
        assert 0.0 < ema_rate <= 1.0, "ema_rate must be in (0, 1]"
        self.n_bins = n_bins
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.ema_rate = float(ema_rate)  # weight on the newest estimate
        self.scores = np.full(n_bins, float(init_score))
        self.n_observed = np.zeros(n_bins, dtype=np.int64)  # cumulative episodes per bin

    # ------------------------------------------------------------------ update
    def update(self, cond_ids: Sequence[int], successes: Sequence[bool]) -> np.ndarray:
        """Update EMA scores from a batch of finished episodes.

        Returns the batch failure rate per bin (nan for bins without data).
        """
        cond_ids = np.asarray(cond_ids, dtype=np.int64).reshape(-1)
        successes = np.asarray(successes, dtype=float).reshape(-1)
        assert cond_ids.shape == successes.shape
        batch_fail = np.full(self.n_bins, np.nan)
        if cond_ids.size == 0:
            return batch_fail
        counts = np.bincount(cond_ids, minlength=self.n_bins)
        fails = np.bincount(cond_ids, weights=1.0 - successes, minlength=self.n_bins)
        seen = counts > 0
        batch_fail[seen] = fails[seen] / counts[seen]
        first = seen & (self.n_observed == 0)
        later = seen & (self.n_observed > 0)
        # first observation of a bin replaces the prior instead of being averaged in
        self.scores[first] = batch_fail[first]
        self.scores[later] = (1 - self.ema_rate) * self.scores[later] + self.ema_rate * batch_fail[later]
        self.n_observed += counts
        return batch_fail

    # ------------------------------------------------------------------ probs
    def probs(self) -> np.ndarray:
        w = np.power(self.scores + self.delta, self.alpha)
        p = self.beta * w / w.sum() + (1.0 - self.beta) / self.n_bins
        return p / p.sum()

    # ------------------------------------------------------------------ io
    def state_dict(self):
        return {
            "scores": self.scores.copy(),
            "n_observed": self.n_observed.copy(),
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
            "ema_rate": self.ema_rate,
        }

    def load_state_dict(self, d):
        self.scores = np.asarray(d["scores"], dtype=float).copy()
        self.n_observed = np.asarray(d["n_observed"], dtype=np.int64).copy()
        for k in ("alpha", "beta", "delta", "ema_rate"):
            setattr(self, k, float(d[k]))


def episode_success(ep_return, terminated, criterion="survive", return_threshold=None):
    """Binary success for locomotion tasks (no native success signal).

    survive            -- reached the time limit without falling (hopper / walker2d)
    return             -- episode return >= return_threshold (halfcheetah never terminates)
    survive_and_return -- both
    """
    ep_return = np.asarray(ep_return, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    if criterion == "survive":
        return ~terminated
    if return_threshold is None:
        raise ValueError(f"criterion '{criterion}' needs return_threshold")
    if criterion == "return":
        return ep_return >= return_threshold
    if criterion == "survive_and_return":
        return (~terminated) & (ep_return >= return_threshold)
    raise ValueError(f"Unknown success criterion '{criterion}'")
