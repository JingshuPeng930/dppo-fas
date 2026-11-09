# DPPO + Failure-Aware Sampling (16-831 course project)

This repo is a fork of [irom-princeton/dppo](https://github.com/irom-princeton/dppo) with **Failure-Aware Sampling (FAS)** added.
FAS changes only *which training conditions* DPPO collects episodes in.
The PPO update is untouched.

## What was added

| File | Purpose |
|---|---|
| `util/fas.py` | `ConditionSpace` (grid of bins C), `FailureAwareSampler` (EMA score Eq. 1, p(c) Eq. 2), `episode_success` |
| `env/gym_utils/wrapper/mujoco_disturbance.py` | Env wrapper. On each reset it picks a bin (schedule → probs), sets torso mass / ground friction / random torso pushes, and reports the episode outcome in `info` |
| `agent/finetune/train_ppo_diffusion_fas_agent.py` | DPPO agent + FAS. Pushes p(c) to envs each iteration, updates s_c from finished episodes, logs `fas/*` |
| `agent/eval/eval_diffusion_disturbance_agent.py` | Evaluates any checkpoint on a fixed test grid, N episodes per bin, and writes `result.json` |
| `cfg/gym/finetune/<task>/ft_ppo_diffusion_mlp_{fas,uniform}.yaml` | Inherit **all** DPPO hyperparameters from the upstream `ft_ppo_diffusion_mlp.yaml` |
| `cfg/gym/eval/<task>/eval_diffusion_mlp_disturbance{,_ood}.yaml` | Test grids: the training grid, plus held-out values for generalization |
| `scripts/fas/*` | Setup, launch, evaluation, plotting, summary |
| `tests/test_fas.py` | Tests for the sampler math, the wrapper (against a fake MuJoCo env), config composition and agent hooks |

Small changes to upstream code:

1. `train_ppo_diffusion_agent.py` gets three no-op hooks: `on_iteration_start`, `on_env_step` and `on_rollout_end`.
2. `make_async` always applies `multi_step` as the outermost wrapper.
3. `.gitignore` no longer ignores `scripts/fas/*.sh`.

**Fair comparison by construction:** the *uniform* baseline is the same agent with `fas.beta=0`.
The two arms share the wrappers, bins, seeds, budget and hyperparameters.

## Methods

| method | config | training conditions |
|---|---|---|
| `dppo` | `ft_ppo_diffusion_mlp` (upstream) | nominal env only. This is the DPPO reproduction |
| `uniform` | `ft_ppo_diffusion_mlp_uniform` | 36 disturbance bins, p(c) = 1/36 |
| `fas` | `ft_ppo_diffusion_mlp_fas` | 36 bins, p(c) from Eq. 2 (default α=1, β=0.5, δ=0.05) |

Bins: torso mass ×{0.75, 1.0, 1.25, 1.5} × friction ×{0.5, 1.0, 1.5} × push force {0, 150, 300} N.
Pushes happen with probability 0.01 per step and last 5 steps.

**Success:**
- hopper and walker2d: the episode reaches the 1000-step limit without falling.
- halfcheetah: return ≥ `return_threshold`. This is a placeholder of 3000; **calibrate it** from the pretrained policy's nominal return.

**FAS score updates:**
- At eval iterations (every `val_freq`=10), the deterministic policy runs a round-robin over all bins.
- At train iterations, the stochastic policy samples bins from p(c).
- Both kinds of episode update s_c by default (`fas.update_from`).

## Setup (Linux + NVIDIA GPU; MuJoCo/mujoco_py does not work on Apple Silicon)

```bash
bash scripts/fas/setup_linux.sh dppo        # conda env, mujoco210, pip install -e .[gym], runs tests
conda activate dppo && source scripts/fas/env.sh
export DPPO_WANDB_ENTITY=<team>             # optional; without it wandb=null
```

The pretrained checkpoints and normalization files download automatically on the first run.

## Step 1: reproduce DPPO

```bash
# pretrained policy, nominal env (upstream eval)
python script/run.py --config-dir=cfg/gym/eval/hopper-v2 --config-name=eval_diffusion_mlp
# DPPO fine-tuning, nominal env
bash scripts/fas/finetune.sh hopper-v2 dppo 0
# quick smoke test (a few minutes)
N_ITR=5 bash scripts/fas/finetune.sh hopper-v2 fas 0 env.n_envs=4 train.n_steps=50 train.val_freq=2
```

Check the eval episode return against the DPPO paper and `cfg/finetuning.md`, Gym section.
Upstream configs run 1000 iterations (80M env steps).
Our scripts default to `N_ITR=200` (16M steps), which matches the proposal's *limited budget*.

## Step 2: RQ1, DPPO vs DPPO-uniform vs DPPO-FAS

```bash
bash scripts/fas/sweep_rq1.sh hopper-v2 "0 1 2"     # 3 methods x 3 seeds, same budget
python scripts/fas/plot_compare.py $DPPO_LOG_DIR/gym-finetune --task hopper-medium-v2 --out hopper.png

# final test on identical conditions (train grid + held-out grid)
for G in train ood; do
  bash scripts/fas/evaluate.sh hopper-v2 pretrained $G
  for CK in $DPPO_LOG_DIR/gym-finetune/hopper-medium-v2_*/*/checkpoint/state_199.pt; do
    bash scripts/fas/evaluate.sh hopper-v2 $CK $G
  done
done
python scripts/fas/summarize_eval.py $DPPO_LOG_DIR/gym-eval-disturbance --csv rq1.csv
```

Main metrics:
- `balanced_success`: mean of per-bin success
- `worst_bin_success`
- the same two for return
- sample efficiency: the `fas/eval/balanced_success` curve against env steps

## Step 3 (later): RQ2 and RQ3

- **RQ2:** sweep `fas.alpha`, `fas.beta` and `fas.ema_rate`, e.g. `bash scripts/fas/finetune.sh hopper-v2 fas 0 fas.alpha=2 fas.beta=0.8`. For "critic error", watch the `explained variance` and `value loss` curves in wandb.
- **RQ3 (RFT):** not implemented yet. Set `train.save_trajs=True` to keep the rollouts; the successful ones become the imitation dataset.
- **LEGO:** a new env and wrapper that implement the same `set_condition_probs` / `set_condition_schedule` / `fas_ep_*` interface. The agent itself is env-agnostic apart from the wrapper-name assert.

## Things to calibrate before the real runs

1. **Disturbance strength.** First evaluate the pretrained policy on the train grid. Hard bins should fail often and easy bins should mostly succeed. If every bin is 0% or 100%, FAS has nothing to prioritize. Then adjust `condition_space`, `impulse_prob` and `impulse_duration`.
2. The halfcheetah `return_threshold`.
3. Coverage. Each eval iteration gives about 1–3 episodes per bin with 40 envs, which is noisy. The EMA (`ema_rate`) smooths it, and `train.val_freq` controls how often eval happens.

## Known upstream quirks (left unchanged)

- `MultiStep` reports `terminated=True` on time-limit truncation as well. The disturbance wrapper therefore tracks falls itself (`fas_ep_terminated`).
- The upstream gym eval configs for walker2d and halfcheetah have a wrong or empty `base_policy_path`. Our `eval_diffusion_mlp_disturbance` configs set the correct one.
