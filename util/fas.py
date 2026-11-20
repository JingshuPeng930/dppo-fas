"""
Failure-Aware Sampling (FAS) utilities.

Pure numpy, no simulator dependency, so it can be imported both by the env
wrapper (inside env subprocesses) and by the training agent (main process).

  * ConditionSpace      -- discrete grid of training-condition bins C
  * FailureAwareSampler -- EMA failure score s_c and sampling distribution p(c)

    p(c) = beta * w_c / sum_j w_j + (1 - beta) / |C|

with the per-bin weight w_c chosen by `score` (all share the EMA estimates and the beta-mix):
  failure  : (s_c + delta)^alpha                      -- FAS, s_c = EMA failure rate
  frontier : (K(q_c) + eps)^(1/T),  K(q) = (q+eps)^(kappa t) (1-q+eps)^(kappa (1-t)),  q_c = 1 - s_c
             Success Guided Sampling (SGS, Zhang et al. CoRL 2026, Eq. 1-2): a Beta kernel peaked
             at target success t. softmax(log(K+eps)/T) == (K+eps)^(1/T) normalized; beta=1 is SGS.
  progress : (LP_c / max LP + delta)^alpha,  LP_c = |s_c - s_c^slow|  -- absolute learning
             progress (TSCL / ALP-GMM), fast EMA (ema_rate) minus slow EMA (ema_slow).
  failure_progress : (s_c + delta)^alpha * (LP_c / max LP + lp_floor)  -- FAS gated by learnability:
             bins that keep failing without any change in their failure EMA (likely unlearnable) get
             lp_floor / (1 + lp_floor) of the weight of an equally failing bin that is improving.
  failure_stall : (s_c + delta)^alpha * g_c,  g_c = clip((s_c^init - s_c^slow) / stall_margin, stall_floor, 1)
             after stall_warmup updates (g = 1 before): FAS gated by *cumulative* progress, s_c^init = mean
             failure over the first stall_init updates. Bins whose failure has not dropped by stall_margin since
             the start of training (likely unlearnable within the budget) are down-weighted (post hoc, RQ2).
  value    : (v_c / max v + delta)^alpha,  v_c = EMA of the positive value loss mean_t max(A_t, 0) of the
             bin's train steps (PLR score, Jiang et al. ICML 2021), updated by update_value(); same power
             form and beta-mix as FAS so only the signal (value error vs failure) differs.

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


SCORES = ("failure", "frontier", "progress", "value", "failure_progress", "failure_stall")
_FLOAT_PARAMS = ("alpha", "beta", "delta", "ema_rate", "target", "kappa", "temperature", "eps", "ema_slow", "lp_floor",
                 "stall_margin", "stall_floor", "stall_warmup", "stall_init")


class FailureAwareSampler:
    """Tracks EMA failure scores per bin and produces p(c) (Eq. 1-2; see module doc for `score`)."""

    def __init__(
        self,
        n_bins: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        delta: float = 0.05,
        ema_rate: float = 0.3,
        init_score: float = 0.5,
        score: str = "failure",
        target: float = 0.66,
        kappa: float = 5.0,
        temperature: float = 2.0,
        eps: float = 1e-8,
        ema_slow: float = 0.1,
        lp_floor: float = 0.25,
        stall_margin: float = 0.1,
        stall_floor: float = 0.25,
        stall_warmup: int = 40,
        stall_init: int = 10,
    ):
        assert n_bins > 0
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        assert delta > 0, "delta must be > 0"
        assert 0.0 < ema_rate <= 1.0, "ema_rate must be in (0, 1]"
        assert 0.0 < ema_slow <= 1.0, "ema_slow must be in (0, 1]"
        assert score in SCORES, f"score must be one of {SCORES}"
        assert 0.0 <= target <= 1.0 and kappa >= 0 and temperature > 0 and eps > 0
        self.n_bins = n_bins
        self.score = score
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.ema_rate = float(ema_rate)  # weight on the newest estimate
        self.target = float(target)
        self.kappa = float(kappa)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.ema_slow = float(ema_slow)
        self.lp_floor = float(lp_floor)
        self.stall_margin = float(stall_margin)
        self.stall_floor = float(stall_floor)
        self.stall_warmup = float(stall_warmup)
        self.stall_init = float(stall_init)
        self.n_updates = 0
        self.init_fail = np.zeros(n_bins)  # failures / episodes over the first stall_init updates
        self.init_n = np.zeros(n_bins)
        self.scores = np.full(n_bins, float(init_score))
        self.scores_slow = self.scores.copy()  # slow EMA, only used by score="progress"
        self.values = np.zeros(n_bins)  # EMA positive value loss, only used by score="value"
        self.n_value_obs = np.zeros(n_bins, dtype=np.int64)
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
        self.scores_slow[first] = batch_fail[first]
        self.scores[later] = (1 - self.ema_rate) * self.scores[later] + self.ema_rate * batch_fail[later]
        self.scores_slow[later] = (1 - self.ema_slow) * self.scores_slow[later] + self.ema_slow * batch_fail[later]
        self.n_observed += counts
        if self.n_updates < self.stall_init:
            self.init_fail += fails
            self.init_n += counts
        self.n_updates += 1
        return batch_fail

    def stall_gate(self) -> np.ndarray:
        if self.n_updates < self.stall_warmup:
            return np.ones(self.n_bins)
        init = np.where(self.init_n > 0, self.init_fail / np.maximum(self.init_n, 1), self.scores_slow)
        return np.clip((init - self.scores_slow) / self.stall_margin, self.stall_floor, 1.0)

    def update_value(self, cond_ids: Sequence[int], pos_adv: Sequence[float]) -> np.ndarray:
        """PLR score: per-bin mean of max(A_t, 0) over a batch of train steps, EMA-smoothed (ema_rate).
        Bins never observed keep the current max (optimistic), so they are still sampled."""
        cond_ids = np.asarray(cond_ids, dtype=np.int64).reshape(-1)
        pos_adv = np.asarray(pos_adv, dtype=float).reshape(-1)
        counts = np.bincount(cond_ids, minlength=self.n_bins)
        sums = np.bincount(cond_ids, weights=pos_adv, minlength=self.n_bins)
        seen = counts > 0
        batch = np.full(self.n_bins, np.nan)
        batch[seen] = sums[seen] / counts[seen]
        first = seen & (self.n_value_obs == 0)
        later = seen & (self.n_value_obs > 0)
        self.values[first] = batch[first]
        self.values[later] = (1 - self.ema_rate) * self.values[later] + self.ema_rate * batch[later]
        self.n_value_obs += counts
        unseen = self.n_value_obs == 0
        if unseen.any() and (~unseen).any():
            self.values[unseen] = self.values[~unseen].max()
        return batch

    # ------------------------------------------------------------------ probs
    def weights(self) -> np.ndarray:
        """Unnormalized priority w_c of each bin under the chosen score."""
        if self.score == "failure":
            return np.power(self.scores + self.delta, self.alpha)
        if self.score == "frontier":
            q = 1.0 - self.scores  # estimated success rate
            k = np.power(q + self.eps, self.kappa * self.target) * np.power(
                1.0 - q + self.eps, self.kappa * (1.0 - self.target))
            return np.power(k + self.eps, 1.0 / self.temperature)
        if self.score == "failure_progress":
            lp = np.abs(self.scores - self.scores_slow)
            lp = lp / lp.max() if lp.max() > 0 else np.zeros_like(lp)
            return np.power(self.scores + self.delta, self.alpha) * (lp + self.lp_floor)
        if self.score == "failure_stall":
            return np.power(self.scores + self.delta, self.alpha) * self.stall_gate()
        if self.score == "value":
            v = self.values / self.values.max() if self.values.max() > 0 else np.zeros_like(self.values)
            return np.power(v + self.delta, self.alpha)
        lp = np.abs(self.scores - self.scores_slow)  # "progress"
        lp = lp / lp.max() if lp.max() > 0 else np.zeros_like(lp)
        return np.power(lp + self.delta, self.alpha)

    def probs(self) -> np.ndarray:
        w = self.weights()
        p = self.beta * w / w.sum() + (1.0 - self.beta) / self.n_bins
        return p / p.sum()

    # ------------------------------------------------------------------ io
    def state_dict(self):
        d = {k: getattr(self, k) for k in _FLOAT_PARAMS}
        d.update(score=self.score, scores=self.scores.copy(), scores_slow=self.scores_slow.copy(),
                 values=self.values.copy(), n_value_obs=self.n_value_obs.copy(), n_observed=self.n_observed.copy(),
                 n_updates=self.n_updates, init_fail=self.init_fail.copy(), init_n=self.init_n.copy())
        return d

    def load_state_dict(self, d):
        self.scores = np.asarray(d["scores"], dtype=float).copy()
        self.scores_slow = np.asarray(d.get("scores_slow", d["scores"]), dtype=float).copy()
        self.n_observed = np.asarray(d["n_observed"], dtype=np.int64).copy()
        self.score = d.get("score", "failure")
        self.values = np.asarray(d.get("values", np.zeros(self.n_bins)), dtype=float).copy()
        self.n_value_obs = np.asarray(d.get("n_value_obs", np.zeros(self.n_bins)), dtype=np.int64).copy()
        self.n_updates = int(d.get("n_updates", 0))
        self.init_fail = np.asarray(d.get("init_fail", np.zeros(self.n_bins)), dtype=float).copy()
        self.init_n = np.asarray(d.get("init_n", np.zeros(self.n_bins)), dtype=float).copy()
        for k in _FLOAT_PARAMS:
            if k in d:  # checkpoints from before the score option lack the new keys
                setattr(self, k, float(d[k]))


def episode_success(ep_return, terminated, criterion="survive", return_threshold=None,
                    ep_len=None, min_steps=None):
    """Binary success for locomotion tasks (no native success signal).

    survive            -- reached the time limit without falling (hopper / walker2d)
    survive_steps      -- did not fall within the first `min_steps` env steps
                          (ep_len >= min_steps, or reached the time limit)
    return             -- episode return >= return_threshold (halfcheetah never terminates)
    survive_and_return -- both
    """
    ep_return = np.asarray(ep_return, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    if criterion == "survive":
        return ~terminated
    if criterion == "survive_steps":
        if ep_len is None or min_steps is None:
            raise ValueError("criterion 'survive_steps' needs ep_len and min_steps")
        return (~terminated) | (np.asarray(ep_len) >= min_steps)
    if return_threshold is None:
        raise ValueError(f"criterion '{criterion}' needs return_threshold")
    if criterion == "return":
        return ep_return >= return_threshold
    if criterion == "survive_and_return":
        return (~terminated) & (ep_return >= return_threshold)
    raise ValueError(f"Unknown success criterion '{criterion}'")
