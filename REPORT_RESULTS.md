# Results section: outline and key numbers (MuJoCo part)

This is the skeleton of the report's Experiments / Results section for the three MuJoCo tasks.
- Every number below comes from `figs/` (regenerate with `python scripts/fas/report_figs.py --out figs`) or from `figs/rq2_analysis.txt`.
- "t" is Welch's t against uniform DR; with 3–5 seeds, |t| > ~2.5 is a reasonably clear difference.
- Success is *balanced success*: the mean over the 36 condition bins of per-bin success, from 20 evaluation episodes per bin with the deterministic policy.
- TODO marks things that are still running or are not part of this MuJoCo study.

## 1. Setup (one paragraph plus one table)

- **Tasks.** D4RL-medium hopper, walker2d and halfcheetah. Each starts from the DPPO pretrained diffusion policy; fine-tuning starts from its EMA weights.
- **Condition space.** 36 bins = torso mass × friction × push force.
  - walker2d / halfcheetah: mass {0.75, 1, 1.25, 1.5}, friction {0.5, 1, 1.5}, push {0, 150, 300} N.
  - hopper uses grid v2.
  - OOD grid: 8 bins at the extremes, e.g. mass {0.6, 1.75}, friction {0.35, 2.0}, push {0, 450}.
- **Success.**
  - hopper / walker2d: the agent does not fall within the first 500 steps.
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

  - Note in a footnote that upstream eval loads the raw weights, not EMA (FAS_README, "Known upstream quirks").

## 2. RQ1: robustness and sample efficiency

**Figures:** `fig_curves` (learning curves), `fig_final` (final success, train and OOD), `fig_bins` (by difficulty).

**Table 1: final balanced success** (state_199, mean over seeds)

| | walker2d train | walker2d OOD | hopper train | hopper OOD | halfcheetah train | halfcheetah OOD |
|---|---|---|---|---|---|---|
| Pretrained (EMA) | 0.464 | 0.288 | 0.394 | 0.250 | 0.511 | 0.263 |
| DPPO (nominal) | 0.700 | 0.325 | 0.451 | 0.275 | 0.645 | 0.317 |
| Uniform DR | 0.836 | 0.395 | 0.554 | 0.275 | 0.769 | 0.458 |
| FAS | **0.874** | **0.435** | 0.565 | 0.263 | **0.777** | **0.483** |
| SGS | 0.871 | 0.429 | 0.498 | 0.271 | 0.771 | 0.475 |
| LP | 0.849 | 0.404 | **0.576** | **0.283** | 0.769 | 0.471 |
| PLR | 0.808 | 0.388 | 0.500 | 0.267 | – | – |

(walker2d uniform / FAS: 5 seeds; everything else: 3 seeds)

**Claims, in order of strength:**

1. **Training under disturbances matters most.** Uniform DR beats nominal DPPO by 10–14 points on train and up to 14 on OOD.
2. **FAS adds a smaller, task-dependent gain on top of uniform:**
   - walker2d: +3.9 points train (t = 3.8), +4.0 OOD (t = 1.9);
   - hopper and halfcheetah: within noise.
3. **The gain concentrates in the hard bins** (Table 2 below).
   - walker2d worst 25% of bins: 0.576 vs 0.498 for uniform.
   - `fig_bins`: the "solved" group is identical across methods; the differences are all in the frontier and unsolved groups.
4. **Sample efficiency** (`sample_efficiency.csv`). Report per-seed AUC of the eval curve, plus iterations to a fixed threshold (walker2d 0.8, hopper 0.5, halfcheetah 0.7).

   | Task | AUC (FAS vs uniform) | Iterations to threshold (median, FAS vs uniform) |
   |---|---|---|
   | walker2d | 0.780 vs 0.754, t = 2.75 | 80 vs 100 |
   | hopper | 0.449 vs 0.459, t = −3.1 | 150 vs 120 (FAS slower early) |
   | halfcheetah | 0.689 vs 0.679, t = 1.3 | 100 vs 80 |

   - SGS is fastest on walker2d (median 50) but slowest on hopper.
   - Write this honestly: there is no consistent speed-up across tasks.

**Why the gain is modest (discussion):**
- **On the original grid, 24 of 36 bins are already solved by uniform DR.** On walker2d, picking the best method per bin gives only 0.888, against 0.836 for uniform.
- **halfcheetah's hard bins are not limited by data.** All methods sit at about 0.42 there, even though FAS spends 65% of its samples on them.

**Harder grid (walker2d, `fig_hard`).** Grid: mass 1–1.75 × friction 0.35–1 × push 150–450 N; the pretrained policy scores 0.22.

| Method | Final success | Worst 25% | AUC |
|---|---|---|---|
| Uniform DR | 0.618 | 0.207 | 0.489 |
| FAS | 0.635 (t = 1.8) | 0.252 (t = 1.7) | 0.480 |
| SGS | 0.623 | 0.230 | 0.472 (t = −2.5) |

- **Conclusion:** a harder space does not widen the gap. Most bins look equally failed, so FAS's distribution flattens: it samples as evenly as about 21 of 36 bins, vs 12 on the original grid.
- **Side result:** policies trained on the hard grid transfer well back to the original grid (about 0.86).

## 3. RQ2: exploration vs stability (`figs/rq2_analysis.txt`, `fig_sparse`)

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

**3c. Reward sparsity (walker2d and hopper, seeds 0–2, `fig_sparse`).** The dense reward is paid out every K env steps; the episode return is unchanged.

| Task | K | Uniform final | FAS final | Δ final | Δ worst 25% | Δ AUC |
|---|---|---|---|---|---|---|
| walker2d | 1 | 0.836 | 0.866 | +0.030 | +0.059 | +0.019 |
| walker2d | 25 | 0.831 | 0.884 | +0.054 (t = 4.5) | +0.115 (t = 7.8) | +0.039 (t = 4.2) |
| walker2d | 100 | 0.828 | 0.862 | +0.034 (t = 2.5) | +0.093 (t = 3.7) | +0.023 (t = 2.1) |
| hopper | 1 | 0.554 | 0.565 | +0.011 | +0.022 | −0.010 (t = −3.1) |
| hopper | 25 | 0.563 | 0.541 | −0.022 (t = −1.2) | −0.030 | −0.014 |
| hopper | 100 | 0.544 | 0.510 | −0.034 (t = −1.5) | −0.045 | −0.028 (t = −2.2) |

- **Claim: sparsity amplifies whatever FAS's concentration does on that task.**
  - walker2d: the hard bins are learnable, and FAS's advantage holds or grows.
  - hopper: FAS spends 46% of its samples on bins it keeps failing (300 N pushes, see 3a), and it falls behind uniform as K grows. Only the AUC at K = 100 is clearly significant; the other hopper gaps are trends (|t| ≈ 1.2–1.5).
- **Interpretation (ties RQ2 together):** concentrating on failures is a bet that the failed conditions are learnable. With a weaker reward signal, wasted samples on unlearnable conditions cost more, and samples on learnable ones are worth more.
- **Not a claim:** do not say "FAS is better under sparse reward" in general, or "sparser is always better" (walker2d is not monotonic).

## 4. RQ3: RL vs RFT on discovered successes (`fig_rq3`, walker2d)

- **RFT setup.** Diffusion behavior cloning from the same EMA initialization, trained on all successful *training* episodes that the corresponding RL run collected. The training-step budget is matched, and actions are clipped to ±1, i.e. the actions actually executed.
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
- **Relation to openpi-comet-style RFT:** ours is the same recipe (successes only, flow/diffusion BC loss, optional demo mix) applied per run.

## 5. Related baselines (answer to the TA)

- **SGS and learning progress:** implemented, see Tables 1 and 2.
  - SGS ties FAS on walker2d, is worse on hopper, and is fastest on walker2d.
  - LP is between uniform and FAS.
- **PLR** (Jiang et al. 2021, positive-value-loss score, walker2d and hopper, 3 seeds): *worse than uniform*.

  | | train | AUC | worst 25% |
  |---|---|---|---|
  | walker2d | 0.808 vs 0.836 | 0.747 vs 0.754 | 0.433 vs 0.498 |
  | hopper | 0.500 vs 0.554 | 0.423 vs 0.459, t = −5.0 | — |

  - **Why:** PLR puts its sampling mass on the *easy* bins.
    - walker2d: 87% on bins uniform already solves.
    - hopper: 66% on solved bins, 7% on unsolved ones.
    - Correlation between PLR's sampling probability and uniform's per-bin success: +0.73 / +0.60.
  - **Mechanism:** in dense-reward locomotion, the positive value loss grows with the return scale and episode length. A policy that survives longer collects larger and more time-ambiguous returns (cf. 3b), so value error mostly tracks how long the policy survives, not how much there is left to learn.
  - **Takeaway:** failure rate is a more direct difficulty signal than value error for this setting. This supports FAS's design choice.
- **ADR:** not run. ADR *expands* the parameter ranges online. Our study fixes the condition space so every method trains and is evaluated on the same 36 bins, so ADR is discussed as complementary: FAS decides where to sample *within* a space, ADR decides how large the space is.

## 6. Limitations

- 3 seeds for most comparisons, 5 for walker2d main.
- 36 discrete bins.
- Success criteria were calibrated per task.
- The DPPO time-limit and critic-scale quirks limit how precisely critic error can be measured.
- The LEGO task: TODO (teammates).
