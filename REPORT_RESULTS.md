# Results section: outline and key numbers (MuJoCo part)

This is the skeleton of the report's Experiments / Results section for the three MuJoCo tasks.
- Every number below comes from `figs/` (regenerate with `python scripts/fas/report_figs.py --out figs`) or from `figs/rq2_analysis.txt`.
- "t" is Welch's t against uniform DR; with 3–5 seeds, |t| > ~2.5 is a reasonably clear difference.
- Success is *balanced success*: the mean over the 36 condition bins of per-bin success, from 10 evaluation episodes per bin with the deterministic policy.
- TODO marks things that are still running or are not part of this MuJoCo study.

## 1. Setup (one paragraph plus one table)

- **Tasks.** D4RL-medium hopper, walker2d and halfcheetah. Each starts from the DPPO pretrained diffusion policy; fine-tuning starts from its EMA weights.
- **Condition space.** 36 bins = torso mass × friction × push force.
  - walker2d / halfcheetah: mass {0.75, 1, 1.25, 1.5}, friction {0.5, 1, 1.5}, push {0, 150, 300} N.
  - hopper uses grid v2.
  - OOD grid: 8 held-out bins, mass {0.6, 1.75} × friction {0.35, 2.0} × push {0, 450} N (hopper: friction {0.35, 1.2}, push {0, 375} N).
- **Success.**
  - walker2d: the agent does not fall within the first 500 steps; hopper: within the first 300 steps (calibrated on the pretrained policies).
  - halfcheetah: return ≥ 4000.
- **Budget.** 200 PPO iterations × 80k env steps = 16M steps, identical for all methods.
- **Methods.** All except DPPO use the same sampler form, p(c) = β·w_c/Σw + (1−β)/|C|:

  | Method | Sampling rule |
  |---|---|
  | DPPO | nominal env only |
  | Uniform DR | β = 0 |
  | FAS | w = (failure EMA + δ)^α, α = 3, β = 0.9 |
  | SGS | Beta kernel at a target success of 0.66, β = 1 |
  | Learning progress (LP) | \|fast − slow failure EMA\| |
  | PLR | positive value loss (mean max(GAE advantage, 0) per bin), same α, β as FAS |

- **Pretrained (EMA) reference, train / OOD balanced success:**

  | Task | Train | OOD |
  |---|---|---|
  | walker2d | 0.464 | 0.288 |
  | hopper | 0.394 | 0.250 |
  | halfcheetah | 0.511 | 0.263 |

  - Note in a footnote that upstream eval loads the raw weights, not EMA (README, "Known upstream quirks").

## 2. RQ1: robustness and sample efficiency

**Figures:** `fig_curves` (learning curves), `fig_final` (final success, train and OOD; main methods only, `fig_final_all` with every baseline for the appendix), `fig_bins` (by difficulty).

**Table 1: final balanced success** (state_199, mean over seeds)

| | walker2d train | walker2d OOD | hopper train | hopper OOD | halfcheetah train | halfcheetah OOD |
|---|---|---|---|---|---|---|
| Pretrained (EMA) | 0.464 | 0.288 | 0.394 | 0.250 | 0.511 | 0.263 |
| DPPO (nominal) | 0.695 | 0.325 | 0.463 | 0.278 | 0.636 | 0.312 |
| Uniform DR | 0.836 | 0.395 | 0.561 | 0.279 | 0.761 | 0.459 |
| FAS | 0.874 | 0.435 | 0.557 | 0.279 | 0.790 | 0.481 |
| SGS | 0.866 | 0.418 | 0.496 | 0.265 | 0.765 | 0.460 |
| LP | 0.854 | 0.418 | **0.568** | **0.287** | 0.757 | 0.450 |
| PLR | 0.790 | 0.370 | 0.493 | 0.272 | 0.712 | 0.402 |
| FAS + LP gate (§3e) | **0.896** | **0.457** | 0.557 | 0.278 | **0.795** | **0.498** |

(Seeds: at least 5 per cell; uniform and FAS 6 on hopper and 8 on halfcheetah, where the RQ3 source runs add seeds. Extra seeds came from `scripts/fas/run_table1_seeds.sh` and `run_rq3_more.sh` with unchanged configs. The hopper FAS + LP seeds 3–4 were first evaluated on the wrong checkpoint by a glob bug in `run_fasp.sh`, since fixed and re-evaluated.)

**Uncertainty.** `figs/comparisons.csv` gives, for every method vs uniform, the difference, Welch t and a 95% percentile bootstrap CI over seeds.
- Walker2d FAS: the CIs for train, worst 25%, OOD and AUC all exclude 0.
- Halfcheetah FAS (8 seeds): train (+2.9, t = 2.0) and worst 25% (+5.1, t = 2.0) CIs exclude 0; OOD and AUC do not.
- Hopper FAS: no CI excludes 0.
- With 5–8 seeds a bootstrap CI is still too narrow. Say this in a footnote and treat |t| as the main guide.

**Claims, in order of strength:**

1. **Training under disturbances matters most.** Uniform DR beats nominal DPPO by 10–14 points on train and up to 14 on OOD.
2. **FAS adds a smaller, task-dependent gain on top of uniform:**
   - walker2d: +3.9 points train (t = 3.8), +4.0 OOD (t = 1.9);
   - halfcheetah (8 seeds): +2.9 train (t = 2.0), +5.1 worst 25% (t = 2.0), +2.2 OOD (t = 1.0), AUC t = 1.4;
   - hopper (6 seeds): within noise (|t| < 1).
   - Section 3d shows *why* hopper shows no gain, and that FAS does help on hopper once its unlearnable bins are removed.
3. **The gain concentrates in the hard bins** (Table 2 below).
   - walker2d worst 25% of bins: 0.576 vs 0.498 for uniform.
   - `fig_bins`: the "solved" group is identical across methods; the differences are all in the frontier and unsolved groups.
4. **Sample efficiency** (`sample_efficiency.csv`). Report per-seed AUC of the eval curve, plus iterations to a fixed threshold (walker2d 0.8, hopper 0.5, halfcheetah 0.7).

   | Task | AUC (FAS vs uniform) | Iterations to threshold (median, FAS vs uniform) |
   |---|---|---|
   | walker2d | 0.780 vs 0.754, t = 2.75 | 80 vs 100 |
   | hopper | 0.453 vs 0.456, t = −0.8 | 135 vs 115 (FAS slower early) |
   | halfcheetah (8 seeds) | 0.687 vs 0.676, t = 1.4 | 80 vs 80 |
   | hopper, learnable grid (§3d, 5 seeds) | 0.601 vs 0.584, t = 2.6 | – |

   - SGS is fastest on walker2d (median 60) but slowest on hopper.
   - FAS + LP gate on walker2d: AUC 0.793 (t = 7.6), median 70 iterations.
   - Write this honestly: there is no consistent speed-up across tasks.

**Why the gain is modest (discussion):**
- **On the original grid, 24 of 36 bins are already solved by uniform DR.** On walker2d, picking the best method per bin gives only 0.888, against 0.836 for uniform.
- **halfcheetah's gain is in the frontier bins.** On the 12 bins where uniform has 0.2–0.8 success: uniform 0.442, FAS 0.492, FAS + LP 0.520; the 23 solved bins are ~0.95–0.97 for all. (With 5 seeds this looked like no gain; at 8 seeds it is small but significant.)

**Harder grid (walker2d, `fig_hard`).** Grid: mass 1–1.75 × friction 0.35–1 × push 150–450 N; the pretrained policy scores 0.22.

| Method | Final success | Worst 25% | AUC |
|---|---|---|---|
| Uniform DR | 0.618 | 0.207 | 0.489 |
| FAS | 0.635 (t = 1.8) | 0.252 (t = 1.7) | 0.480 |
| SGS | 0.623 | 0.230 | 0.472 (t = −2.5) |

- **Conclusion:** a harder space does not widen the gap. Most bins look equally failed, so FAS's distribution flattens: it samples as evenly as about 21 of 36 bins, vs 12 on the original grid.
- **Side result:** policies trained on the hard grid transfer well back to the original grid (about 0.86).

## 3. RQ2: exploration vs stability (`figs/rq2_analysis.txt`, `fig_sparse`, `fig_hopper_v3`)

**The argument, in three steps:**
1. Focusing on failures does **not** destabilize the critic (3b).
2. Critic / value error is therefore a poor signal for *where* to sample; PLR, which uses it, does worse than uniform (§5).
3. The real risk is spending samples on conditions that **cannot be learned**. This explains the hopper results (3a, 3c). We tested it directly: we removed hopper's unlearnable condition, and FAS then beats uniform (3d).

**3a. Sampling strength α (walker2d)**

| Sampler | Equivalent number of bins sampled evenly (of 36) | Final success | Worst 25% |
|---|---|---|---|
| α = 1, β = 0.5 | 30 | 0.840 | 0.489 |
| α = 3, β = 0.9 | 12 | 0.874 | 0.576 |
| Uniform DR | 36 | 0.836 | 0.498 |

- FAS only helps when it actually concentrates its samples.
- The cost of concentrating shows on hopper: FAS puts 46% of its samples on bins it currently fails, vs 26% for uniform. That gives no final gain and slightly slower early learning: the exploration side of the trade-off.

**3b. Critic error per condition**
- **Method.** `scripts/fas/rq2_critic.py`: the stochastic policy is rolled out in each bin and V(s) is compared with the Monte-Carlo return-to-episode-end (the same target as DPPO's GAE). Because of the running reward scaler, V is fitted up to one scale factor.
- **Result.** Relative error is about 0.32–0.36 for every method. Most of it is irreducible time ambiguity (see the caveat below).
- **FAS does not destabilize the critic:**
  - walker2d at iteration 199: FAS 0.319 vs uniform 0.332 overall, and 0.311 vs 0.337 on frontier bins;
  - bins sampled more have slightly lower error (corr −0.1 to −0.36).
- **Caveat for the report.** Upstream DPPO treats the 1000-step time limit as a terminal state, and the critic has no time input.
- **Under sparse reward** (`figs/rq2_sparse_critic.txt`; K = 25, 100; walker2d and hopper; 3 seeds): FAS's critic error is never above uniform's, at either checkpoint.

  | Iteration 199 | K = 1 | K = 25 | K = 100 |
  |---|---|---|---|
  | walker2d, uniform → FAS | 0.332 → 0.319 | 0.312 → 0.307 | 0.279 → 0.276 |
  | hopper, uniform → FAS | 0.360 → 0.350 | 0.344 → 0.337 | 0.299 → 0.293 |

  - This includes hopper at K = 100, where FAS's *success* is worse than uniform's (3c). The error on the bins FAS keeps failing (unsolved group) is also equal: 0.312 vs 0.312.
  - **So the hopper degradation is not a critic-stability effect.** It comes from where the samples go (unlearnable bins), which 3d confirms.
  - Do not compare error across K: delayed rewards change the shape of the Monte-Carlo target, so the absolute level is not comparable.

**3c. Reward sparsity (walker2d and hopper, all available seeds, `fig_sparse`).** Seeds for K = 1 / 25 / 100: walker2d 5 / 5 / 5, hopper 6 / 5 / 5 (seeds 3–4 at K = 25 / walker2d K = 100 added 2026-11-19, `scripts/fas/run_stall.sh`). The dense reward is paid out every K env steps; the episode return is unchanged.

| Task | K | Uniform final | FAS final | Δ final | Δ worst 25% | Δ AUC |
|---|---|---|---|---|---|---|
| walker2d | 1 | 0.836 | 0.874 | +0.038 (t = 3.8) | +0.078 (t = 2.3) | +0.026 (t = 2.8) |
| walker2d | 25 | 0.833 | 0.879 | +0.046 (t = 4.6) | +0.098 (t = 4.7) | +0.032 (t = 3.4) |
| walker2d | 100 | 0.827 | 0.870 | +0.043 (t = 3.6) | +0.106 (t = 5.2) | +0.031 (t = 4.1) |
| hopper | 1 | 0.561 | 0.557 | −0.004 (t = −0.3) | +0.037 (t = 1.8) | −0.003 (t = −0.8) |
| hopper | 25 | 0.564 | 0.539 | −0.025 (t = −1.85, CI [−0.049, −0.002]) | −0.020 (t = −1.5) | −0.006 (t = −0.7) |
| hopper | 100 | 0.538 | 0.517 | −0.021 (t = −1.4) | −0.011 (t = −0.4) | −0.027 (t = −3.3, CI [−0.040, −0.012]) |

- **Claim (weakened at 5 seeds): FAS's advantage persists under sparse reward where the hard bins are learnable, and FAS falls behind where they are not.**
  - walker2d: the final-success gain is flat in K (+3.8 / +4.6 / +4.3); only the worst-25% gain grows modestly (+7.8 / +9.8 / +10.6). At 3 seeds K = 25 looked larger (+5.4 / +11.5); do not claim the gain grows with sparsity.
  - hopper: FAS spends 46% of its samples on bins it keeps failing (300 N pushes, see 3a), and it falls behind uniform as K grows. The AUC at K = 100 (CI excludes 0) and the final success at K = 25 (t = −1.85, CI barely excludes 0) are the significant gaps; the rest are trends. With seeds 3–4 added, the K = 100 final-success gap shrank from −0.034 to −0.021 while the AUC gap held.
- **Interpretation (ties RQ2 together):** concentrating on failures is a bet that the failed conditions are learnable. With a weaker reward signal, wasted samples on unlearnable conditions cost more, and samples on learnable ones are worth more.
- **Not a claim:** do not say "FAS is better under sparse reward" in general, or "sparser is always better" (walker2d is not monotonic).

**3d. Hypothesis test: hopper with only learnable hard conditions (`fig_hopper_v3`, 5 seeds).**
- **Why this test.** On the main hopper grid, the 300 N push is essentially unlearnable: uniform DR leaves those bins at about 0.2 success, the same as the pretrained policy. FAS keeps sampling them.
- **Grid.** We kept friction {0.5, 1.0, 1.15} and replaced the pushes with {0, 100, 200} N. This grid was chosen *before* training, by evaluating existing policies on candidate grids:
  - the pretrained policy scores 0.43;
  - the uniform policy trained on the main grid leaves only 2 of 36 bins on the new grid unsolved.
- **Prediction, written down before the runs:** FAS > uniform.

| Method | Final success | Worst 25% | AUC | OOD (hopper OOD grid) |
|---|---|---|---|---|
| Uniform DR | 0.684 | 0.282 | 0.584 | 0.307 |
| FAS | **0.723** (+0.039, t = 3.5, CI [0.018, 0.058]) | **0.358** (+0.076, t = 2.7, CI [0.029, 0.124]) | 0.601 (+0.017, t = 2.6, CI [0.006, 0.029]) | 0.303 |
| FAS + LP gate | 0.715 (+0.031, t = 1.9, CI [0.001, 0.058]) | 0.316 (t = 1.0) | **0.610** (+0.026, t = 3.4) | 0.305 |

- **Result.** The prediction holds. On the same robot where FAS gave nothing on the main grid, it now improves final success, worst-case bins and speed. With 5 seeds the final and worst-25% gaps became *more* significant (t 2.1 → 3.5 and 1.7 → 2.7), and all three CIs exclude 0. FAS's AUC is higher than uniform's for every seed index.
- **Where the gain comes from.** It is all in the 21 frontier bins (0.49 → 0.56). The 15 solved bins are unchanged (0.96).
- **OOD is unchanged.** The hopper OOD grid contains 375 N pushes, far beyond the 200 N maximum seen in training on this grid.
- **Takeaway for RQ2:** the exploration / stability trade-off of failure-focused sampling is governed by *learnability* of the failed conditions, not by critic stability.

**3e. Exploratory: gating FAS by learning progress (FAS + LP gate).**
- **Rule.** w = (failure + δ)^α · (LP / max LP + 0.25). This down-weights bins whose failure rate has stopped changing.
- **walker2d (5 seeds):** best method overall.
  - Final success 0.896 vs 0.836 for uniform (t = 8.2) and 0.874 for FAS.
  - Worst 25%: 0.638 vs 0.498.
  - OOD: 0.457 vs 0.395.
  - AUC: 0.793 vs 0.754.
  - All four bootstrap CIs exclude 0.
- **halfcheetah (5 seeds):** best method: 0.795 vs 0.761 (t = 2.7), worst 25% 0.407 vs 0.333 (t = 3.2), AUC 0.705 vs 0.676 (t = 3.8); all three CIs exclude 0. OOD 0.498 vs 0.459 (t = 1.4).
- **hopper main grid (5 seeds):** no help (0.557; AUC lower, t = −3.9). The gate does not identify the unlearnable bins: their failure rate fluctuates as much as frontier bins' (|Δ failure| 0.053 vs 0.088).
- **How to present it:** a promising variant, not the main method. It was designed after seeing the main results, so it should be labeled post hoc.

**Cumulative-progress gate (`fas.score=failure_stall`, post hoc, `scripts/fas/run_stall.sh`, 5 seeds).** Weight (s_c+δ)^α · g_c with g_c = clip((failure over the first 10 updates − slow failure EMA) / 0.1, 0.25, 1), active from update 40. Designed after the LP gate failed on hopper; checked offline first on the hopper FAS/uniform histories (300 N bins vs other hard bins, per-run AUC of cumulative drop ≈ 0.7 at itr 40–60, 0.8 at itr 80–100).

| Task | uniform | FAS | FAS+LP | FAS+stall gate | stall vs uniform | stall vs FAS |
|---|---|---|---|---|---|---|
| walker2d train / worst / AUC / OOD | .836 / .498 / .754 / .395 | .874 / .576 / .780 / .435 | .896 / .638 / .793 / .457 | .890 / .622 / .793 / .450 | final t = 7.5, worst t = 5.0, AUC t = 5.8, OOD t = 2.3 | final t = 1.5, worst 1.6, AUC 1.3 (n.s.) |
| hopper train / worst / AUC / OOD | .561 / .141 / .456 / .279 | .557 / .178 / .453 / .279 | .557 / .164 / .439 / .278 | .564 / .189 / .455 / .297 | final t = 0.2, worst t = 1.7, OOD t = 1.2 | n.s. |

- **Negative on hopper.** The gate barely moves the samples: mass on the 300 N bins in iterations 100–199 is 0.53 vs 0.57 for FAS. At the end of training the gate on those bins averages 0.7 (8–25% at the floor) vs 0.9 for the other hard bins, because the 300 N bins *also* improve slowly (failure −0.07 to −0.09 over training; success 0.24 → 0.27). Hopper's hardest bins are slowly learnable, not unlearnable, so cumulative progress separates them only weakly. A larger margin would be tuned to hopper and was not tried.
- **walker2d:** as expected the gate rarely fires (mass on 300 N bins 0.91 vs 0.90 for FAS); results match FAS+LP and are not significantly above FAS.

## 4. RQ3: RL vs RFT on discovered successes (`fig_rq3`, all three tasks)

- **RFT setup.** Diffusion behavior cloning from the same EMA initialization, trained on all successful *training* episodes that the corresponding RL run collected. Training runs 10 epochs over each source run's successes, and actions are clipped to ±1, i.e. the actions actually executed.
- **Results** (train / OOD balanced success):

  | Source of successes | RL policy | RFT | RFT + 60% D4RL demos |
  |---|---|---|---|
  | FAS run | 0.882 / 0.450 | 0.820 / 0.400 | 0.654 / 0.367 |
  | Uniform run | 0.831 / 0.400 | 0.752 / 0.383 | 0.635 / 0.350 |
  | Frozen EMA (no RL) | 0.464 / 0.288 | 0.454 / 0.275 | 0.488 / 0.325 |

- **Claims:**
  - Most of RL's gain can be recovered by imitating the successes RL discovered.
  - The policy-gradient update itself adds 6–8 points.
  - FAS's successes make a better RFT dataset than uniform's (0.820 vs 0.752).
  - Successes from the frozen policy give nothing: exploration is what matters.
  - Mixing in nominal-only D4RL demos hurts.
- **hopper (source seeds 3–5, train / OOD):**

  | Source of successes | RL policy | RFT | RFT + 60% D4RL demos |
  |---|---|---|---|
  | FAS run | 0.550 / 0.296 | 0.468 / 0.258 | 0.420 / 0.283 |
  | Uniform run | 0.568 / 0.283 | 0.469 / 0.225 | 0.424 / 0.233 |

  - RFT recovers only about 45% of RL's gain over the pretrained policy (0.394), vs about 85% on walker2d.
  - RFT on FAS's successes generalizes better OOD (0.258 vs 0.225).
  - Demos cost about 5 points of train success; the OOD change is within noise.
- **halfcheetah (source seeds 5–7, train / OOD; these RL runs also count as extra Table 1 seeds):**

  | Source of successes | RL policy | RFT | RFT + 60% D4RL demos |
  |---|---|---|---|
  | FAS run | 0.797 / 0.471 | 0.665 / 0.383 | 0.620 / 0.312 |
  | Uniform run | 0.733 / 0.433 | 0.628 / 0.371 | 0.612 / 0.342 |

  - RFT recovers 53–54% of RL's gain over the pretrained policy (0.511).
  - FAS's successes give a slightly better RFT policy (0.665 vs 0.628, within one sd at n=3).
  - Demos lower train success again (2–5 points) and do not help OOD consistently.
- **Conclusion:** imitation is not the bottleneck: BC on the final RL policy's successes matches RL on all three tasks. RFT on the successes accumulated during training recovers 78–85% of the gain on walker2d but only about half on hopper and halfcheetah, because those successes come largely from earlier policies. Across all three tasks, FAS's successes are an equal or better RFT dataset than uniform's, and nominal-only demos never help train success.
- **Why RFT falls short on hopper / halfcheetah: margin test** (`scripts/fas/run_rft_margin.sh`, same source runs, same gradient steps via `match_full_steps`).
  - Diagnosis: most saved successes barely pass the criterion. On hopper 88–98% of the successes fall after step 300 (median length 374–422); on halfcheetah they cluster just above 4000 (10th pct ≈ 4050, median ≈ 4300). On walker2d the median success runs ~1000 steps.
  - Test: keep only successes with a margin (hopper ≥ 600 steps, halfcheetah return ≥ 4500), train/OOD:

    | Task, source | RFT (all successes) | RFT (margin only) | Welch t (train) |
    |---|---|---|---|
    | hopper, FAS | 0.468 / 0.258 | 0.463 / 0.233 | −0.5 |
    | hopper, uniform | 0.469 / 0.225 | 0.474 / 0.225 | 1.0 |
    | halfcheetah, FAS | 0.665 / 0.383 | 0.681 / 0.400 | 0.5 |
    | halfcheetah, uniform | 0.628 / 0.371 | 0.651 / 0.383 | 0.5 |

  - Result: no significant change, so near-threshold labels do not explain the gap. Also ruled out: bin coverage (≥ 140 successes in every bin; count vs RL−RFT gap has no consistent sign) and early-training data (~45% from the first 100 iterations on every task, walker2d included).
- **Why RFT falls short: staleness, not imitation** (`scripts/fas/run_rft_whygap.sh`, same source runs, same gradient steps). Recovered share of the RL gain on the train grid (FAS / uniform source):

  | Task | RFT (all successes) | RFT (last 50 iterations only) | BC on the final RL policy's successes |
  |---|---|---|---|
  | walker2d | 0.85 / 0.79 | 0.94 / 0.90 | 1.00 / 1.02 |
  | hopper | 0.47 / 0.43 | 0.71 / 0.77 | 1.02 / 0.96 |
  | halfcheetah | 0.54 / 0.52 | 0.70 / 0.88 | 0.98 / 1.15 |

  - Distillation: 100 deterministic rollouts per bin of the final RL policy on the train grid (seed 2000+s, not the eval episodes), successes only (2–3.3k episodes), BC with the same steps as the full RFT. Train success matches RL on every task (e.g. hopper 0.553 / 0.560 vs RL 0.550 / 0.568; t vs RFT ≈ 5), and OOD as well.
  - So diffusion BC can reproduce the RL policy; what limits RFT is that it imitates successes accumulated over training, most from earlier, noisier (stochastic exploration) policies. The last-50-iteration RFT closes most of the gap; the rest (late vs distillation) is consistent with exploration noise in the training rollouts.
  - Why walker2d is the exception: staleness costs it least (+9–11 points of recovery from late-only data vs +24–34 on hopper and +16–36 on halfcheetah).
  - Why staleness costs walker2d least is *open*. It is not that walker2d's early successes are closer to its final policy (`scripts/fas/rft_staleness.py`: final RL policy queried deterministically on 300 saved chunks per iteration, all 18 source runs). Squared distance between the executed (clipped) action and the final policy's action, relative to the last-50-iteration level:

    | Task | itr 0–24 | 50–74 | 100–124 | all data |
    |---|---|---|---|---|
    | walker2d | 1.38 | 1.26 | 1.12 | 1.15 |
    | hopper | 1.23 | 1.12 | 1.07 | 1.08 |
    | halfcheetah | 1.30 | 1.21 | 1.12 | 1.12 |

    By this measure walker2d's data are the *most* stale. The late-iteration level itself (0.10–0.13) is about as large as the pretrained-vs-final distance on the same states (0.13–0.16), i.e. most of the mismatch is exploration noise. So equal action-space staleness costs hopper and halfcheetah more.
  - **Sensitivity test** (`scripts/fas/run_rft_sensitivity.sh`, eval agent `++action_noise_std`): the 18 final RL policies (FAS + uniform pooled, n=6 per task) and the pretrained policy (3 eval seeds) on the train grid with N(0, σ²) noise on every executed action. Loss as a share of the RL gain (RL(0) − pretrained(0)):

    | Task | RL gain | σ=0.1 | σ=0.2 | σ=0.3 |
    |---|---|---|---|---|
    | walker2d | .353 (.506→.859) | 0.07 (.836) | 0.18 (.796) | 0.66 (.626) |
    | hopper | .176 (.386→.562) | 0.14 (.538) | 0.20 (.528) | 0.59 (.458) |
    | halfcheetah | .220 (.552→.772) | 0.48 (.666) | 2.18 (.292) | 3.33 (.039) |

    At small perturbations walker2d loses the smallest share of its gain. On hopper the *absolute* loss at σ=0.1 equals walker2d's (−2.4 points) and the gain is half as large; halfcheetah's 4000-return threshold makes it far more sensitive in absolute terms. This separates walker2d from the other two tasks, which is the question. It does not order hopper vs halfcheetah: by sensitivity halfcheetah should recover less, but it recovers slightly more. Isotropic noise is only a proxy for RFT's imitation error, and there are only three tasks, so this is consistent evidence, not proof.
- **Relation to openpi-comet-style RFT:** ours is the same recipe (successes only, flow/diffusion BC loss, optional demo mix) applied per run.

## 5. Related baselines (answer to the TA)

- **SGS and learning progress:** implemented, see Tables 1 and 2.
  - SGS ties FAS on walker2d, is worse on hopper, matches uniform on halfcheetah, and is fastest on walker2d.
  - LP is between uniform and FAS on walker2d and within noise of uniform elsewhere.
- **PLR** (Jiang et al. 2021, positive-value-loss score, 5 seeds per task): *worse than uniform on all three tasks*.

  | | train | AUC | worst 25% |
  |---|---|---|---|
  | walker2d | 0.790 vs 0.836, t = −3.4 | 0.743 vs 0.754, t = −2.1 | 0.389 vs 0.498 |
  | hopper | 0.493 vs 0.561, t = −3.9 | 0.426 vs 0.456, t = −6.3 | 0.102 vs 0.141 |
  | halfcheetah | 0.712 vs 0.761, t = −2.9 | 0.647 vs 0.676, t = −3.0 | 0.258 vs 0.333 |

  - **Why:** PLR puts its sampling mass on the *easy* bins (sampling statistics from the first 3 seeds).
    - walker2d: 87% on bins uniform already solves.
    - hopper: 66% on solved bins, 7% on unsolved ones.
    - Correlation between PLR's sampling probability and uniform's per-bin success: +0.73 / +0.60.
  - **Mechanism:** in dense-reward locomotion, the positive value loss grows with the return scale and episode length. A policy that survives longer collects larger and more time-ambiguous returns (cf. 3b), so value error mostly tracks how long the policy survives, not how much there is left to learn.
  - **Takeaway:** failure rate is a more direct difficulty signal than value error for this setting. This supports FAS's design choice.
- **ADR:** not run. ADR *expands* the parameter ranges online. Our study fixes the condition space so every method trains and is evaluated on the same 36 bins, so ADR is discussed as complementary: FAS decides where to sample *within* a space, ADR decides how large the space is.

## 6. Limitations

- 3 seeds for most comparisons; 5–6 for the main uniform vs FAS comparison. Bootstrap CIs with so few seeds are optimistic.
- The hopper "learnable grid" result (3d) is a single 5-seed test of a prediction written down before training (not a formal pre-registration); the learnable grid is also easier overall. It supports the learnability explanation but does not prove it in general.
- FAS + LP gate was designed after seeing the main results.
- 36 discrete bins.
- Success criteria were calibrated per task.
- The DPPO time-limit and critic-scale quirks limit how precisely critic error can be measured.
- The LEGO task: TODO (teammates).
