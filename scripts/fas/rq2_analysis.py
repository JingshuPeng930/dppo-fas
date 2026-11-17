"""
RQ2 / RQ1-supplement analyses from existing runs (no new training):

  1. sampling concentration over training: normalized entropy, effective #bins (1/sum p^2), p_max,
     mass on bins the sampler currently estimates as unsolved (EMA success < 0.2)
  2. alpha sweep: FAS alpha=1 (beta 0.5) vs alpha=3 (beta 0.9) next to uniform -- final success
     (train / OOD), worst-25% bins, eval-curve AUC
  3. tail robustness (RQ1): mean success over the worst 25% bins of each policy (per seed)
  4. critic error per bin (from scripts/fas/rq2_critic.py JSONs in --critic-dir), grouped by the uniform
     policy's per-bin difficulty, and its relation to how much each method sampled the bin

    python scripts/fas/rq2_analysis.py --critic-dir ~/dppo_fas/rq2 --out figs
"""

import argparse
import collections
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_figs as R  # noqa: E402

LABEL = {"uniform": "uniform", "fas": "FAS a3", "fas_a1": "FAS a1", "sgs": "SGS", "lp": "LP", "plr": "PLR"}
RUNS = {
    "walker2d": dict(R.TASKS["walker2d"]["runs"], fas_a1="fas_a1.0_b0.5_ta4"),
    "hopper": dict(R.TASKS["hopper"]["runs"], fas_a1="fas_gv2_a1.0_b0.5_ta4"),
    "halfcheetah": dict(R.TASKS["halfcheetah"]["runs"]),
}


def mean_std(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std():.3f}" if len(x) else "  -  "


def sampling_stats(env, key):
    out = collections.defaultdict(list)
    for d in R.run_dirs(env, key):
        h = pickle.load(open(f"{d}/fas_history.pkl", "rb"))["history"]
        P = np.array([e["probs"] for e in h if e["source"] == "train"])
        S = np.array([e["scores"] for e in h if e["source"] == "train"])  # EMA failure (before update)
        n = P.shape[1]
        out["entropy"].append(np.mean(-(P * np.log(P + 1e-12)).sum(1) / np.log(n)))
        out["eff_bins"].append(np.mean(1.0 / (P ** 2).sum(1)))
        out["p_max"].append(np.mean(P.max(1)))
        out["mass_unsolved"].append(np.mean((P * (S > 0.8)).sum(1)))
    return out


def final_metrics(env, grid, key):
    res = {}
    for split in ["train", "ood"]:
        ev = R.load_evals(env, split + grid)
        rs = [r for r in R.pick(ev, f"{env}_ppo_diffusion_mlp_{key}", "state_199.pt") if "L400" not in r["base_policy_path"]]
        per = np.array([[c["success"] for c in r["per_condition"]] for r in rs]) if rs else np.zeros((0, 1))
        k = max(1, per.shape[1] // 4)
        res[split] = per.mean(1) if len(per) else []
        res[split + "_worst25"] = np.sort(per, 1)[:, :k].mean(1) if len(per) else []
    cs = [R.eval_curve(d) for d in R.run_dirs(env, key)]
    res["auc"] = [c[1].mean() for c in cs]
    return res


def critic_table(critic_dir, task, env, grid):
    files = glob.glob(f"{critic_dir}/{env}_*.json")
    if not files:
        return
    ev = R.load_evals(env, "train" + grid)
    uni = [r for r in R.pick(ev, f"{env}_ppo_diffusion_mlp_{RUNS[task]['uniform']}", "state_199.pt")]
    u = np.array([[c["success"] for c in r["per_condition"]] for r in uni]).mean(0)
    groups = [("unsolved<0.2", u < 0.2), ("frontier", (u >= 0.2) & (u <= 0.8)), ("solved>0.8", u > 0.8)]
    print(f"\n[4] critic |V/a - G| / mean|G| per bin group  ({task}; groups by uniform's final per-bin success)")
    print(f"    {'method':8s} {'itr':>4s} {'n':>2s} {'overall':>15s} " + " ".join(f"{g:>16s}" for g, m in groups)
          + f" {'corr(err, p_bin)':>17s}")
    for m, key in RUNS[task].items():
        for itr in (100, 199):
            js = [json.load(open(f)) for f in files if f"_{key}" in os.path.basename(f).split("__")[0] + "_"
                  and os.path.basename(f).endswith(f"__{itr}.json")]
            js = [j for j in js if f"ppo_diffusion_mlp_{key}" in j["run_dir"]]
            if not js:
                continue
            E = np.array([[b["rel_abs_err"] for b in j["per_bin"]] for j in js])  # (seeds, bins)
            row = f"    {LABEL[m]:8s} {itr:4d} {len(js):2d} {mean_std([j['overall_rel_abs_err'] for j in js]):>15s} "
            row += " ".join(f"{mean_std(E[:, msk].mean(1)) if msk.any() else '-':>16s}" for g, msk in groups)
            # does a method's critic get better where it sampled more? (mean train-time p per bin)
            Ps = []
            for j in js:
                h = pickle.load(open(os.path.join(j["run_dir"], "fas_history.pkl"), "rb"))["history"]
                Ps.append(np.mean([e["probs"] for e in h if e["source"] == "train" and e["itr"] <= itr], 0))
            c = np.mean([np.corrcoef(e, p)[0, 1] if np.std(p) > 1e-9 else np.nan for e, p in zip(E, Ps)])
            row += f" {c:17.2f}"
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic-dir", default=None)
    a = ap.parse_args()
    for task, runs in RUNS.items():
        env, grid = R.TASKS[task]["env"], R.TASKS[task]["grid"]
        print(f"\n===================== {task}")
        print("[1] sampling concentration over training (mean over train iterations; seeds mean ± std)")
        print(f"    {'method':8s} {'entropy':>15s} {'eff. #bins/36':>15s} {'p_max':>15s} {'mass on EMA-unsolved':>21s}")
        for m, key in runs.items():
            s = sampling_stats(env, key)
            if s:
                print(f"    {LABEL[m]:8s} {mean_std(s['entropy']):>15s} {mean_std(s['eff_bins']):>15s} "
                      f"{mean_std(s['p_max']):>15s} {mean_std(s['mass_unsolved']):>21s}")
        print("[2,3] final success (state_199) and eval-curve AUC, per-seed mean ± std")
        print(f"    {'method':8s} {'train':>15s} {'train worst25%':>15s} {'OOD':>15s} {'OOD worst25%':>15s} {'AUC':>15s}")
        for m, key in runs.items():
            f = final_metrics(env, grid, key)
            print(f"    {LABEL[m]:8s} {mean_std(f['train']):>15s} {mean_std(f['train_worst25']):>15s} "
                  f"{mean_std(f['ood']):>15s} {mean_std(f['ood_worst25']):>15s} {mean_std(f['auc']):>15s}")
        if a.critic_dir and task != "halfcheetah":
            critic_table(os.path.expanduser(a.critic_dir), task, env, grid)


if __name__ == "__main__":
    main()
