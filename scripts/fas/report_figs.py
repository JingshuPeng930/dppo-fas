"""
Report figures (run on the machine holding $DPPO_LOG_DIR):

  fig_curves.png     balanced eval success vs env steps, uniform / FAS / SGS / LP, one panel per task
                     (in-training deterministic evals from fas_history.pkl, mean +- std over seeds)
  fig_final.png      final disturbance-eval success (train grid and OOD grid) per task and method
  fig_rq3.png        RQ3: source RL policy vs RFT on its successful episodes (+ demo-mix ablation)
  fig_bins.png       per-bin success grouped by difficulty for uniform (why FAS gains little)
  sample_efficiency.csv  per-seed AUC of the eval curve (+ Welch t vs uniform) and itr to a fixed threshold
  final.csv

    python scripts/fas/report_figs.py --out figs
"""

import argparse
import collections
import csv
import glob
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = os.environ["DPPO_LOG_DIR"]
FT, EV = f"{LOG}/gym-finetune", f"{LOG}/gym-eval-disturbance"
STEPS_PER_ITR = 40 * 500 * 4  # n_envs * n_steps * act_steps

# fixed categorical order: color follows the method everywhere
METHODS = collections.OrderedDict([
    ("uniform", ("Uniform DR", "#2a78d6")),
    ("fas", ("FAS (α=3)", "#eb6834")),
    ("sgs", ("SGS", "#1baf7a")),
    ("lp", ("Learning progress", "#9b59b6")),
])
REF = {"ema": ("Pretrained", "#b8b8b8"), "dppo": ("DPPO (nominal)", "#6e6e6e")}

TASKS = collections.OrderedDict([
    ("walker2d", dict(env="walker2d-medium-v2", grid="", runs={
        "uniform": "uniform_ta4", "fas": "fas_a3.0_b0.9_ta4", "sgs": "sgs_a1.0", "lp": "lp_a1.0"})),
    ("hopper", dict(env="hopper-medium-v2", grid="_gv2", runs={
        "uniform": "uniform_gv2_ta4", "fas": "fas_gv2_a3.0_b0.9_ta4", "sgs": "sgs_gv2", "lp": "lp_gv2"})),
    ("halfcheetah", dict(env="halfcheetah-medium-v2", grid="", runs={
        "uniform": "uniform_ta4", "fas": "fas_a3.0_b0.9_ta4", "sgs": "sgs_a1.0", "lp": "lp_a1.0"})),
])


# fixed a priori per task (roughly uniform's level at mid-training; hopper/halfcheetah never reach 0.8)
THRESHOLD = {"walker2d": 0.8, "hopper": 0.5, "halfcheetah": 0.7}


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def style(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#555555", labelsize=8)


# ----------------------------------------------------------------- learning curves
def run_dirs(env, key):
    ds = sorted(glob.glob(f"{FT}/{env}_ppo_diffusion_mlp_{key}*/*"))
    return [d for d in ds if os.path.exists(f"{d}/fas_history.pkl") and "L400" not in d]


def eval_curve(run_dir):
    h = pickle.load(open(f"{run_dir}/fas_history.pkl", "rb"))
    n = len(np.array(np.meshgrid(*h["condition_values"])).reshape(len(h["condition_values"]), -1).T)
    itr, bal = [], []
    for e in h["history"]:
        if e["source"] != "eval":
            continue
        c, s = np.asarray(e["cond"]), np.asarray(e["success"], float)
        cnt = np.bincount(c, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            bal.append(np.nanmean(np.bincount(c, weights=s, minlength=n) / cnt))
        itr.append(e["itr"])
    return np.array(itr), np.array(bal)


def fig_curves(out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
    rows = []
    for ax, (task, t) in zip(axes, TASKS.items()):
        curves = {}
        for m, key in t["runs"].items():
            cs = [eval_curve(d) for d in run_dirs(t["env"], key)]
            if not cs:
                continue
            itr = sorted(set.intersection(*[set(c[0].tolist()) for c in cs]))
            Y = np.array([[c[1][c[0] == i][0] for i in itr] for c in cs])
            curves[m] = (np.array(itr), Y)
            lab, col = METHODS[m]
            x = (np.array(itr) + 1) * STEPS_PER_ITR / 1e6
            ax.plot(x, Y.mean(0), color=col, lw=2, label=f"{lab} (n={len(cs)})")
            ax.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=col, alpha=0.15, lw=0)
        # sample efficiency, all per seed (no threshold picking on the seed-mean curve):
        #  * AUC = mean eval success over the whole run (primary; one number per seed -> mean, std, Welch t vs uniform)
        #  * itr to a fixed per-task threshold (secondary): first eval with success >= th whose average with
        #    the next eval is also >= th (damps single-eval spikes); median and range over seeds
        th = THRESHOLD[task]
        auc_u = curves["uniform"][1].mean(1)
        for m, (itr, Y) in curves.items():
            auc = Y.mean(1)
            t_stat = welch_t(auc, auc_u) if m != "uniform" else float("nan")
            hits = []
            for y in Y:
                ok = [k for k in range(len(y)) if y[k] >= th and (k + 1 >= len(y) or (y[k] + y[k + 1]) / 2 >= th)]
                hits.append(int(itr[ok[0]]) if ok else None)
            reached = [h for h in hits if h is not None]
            rows.append(dict(
                task=task, method=m, seeds=len(Y),
                auc_mean=round(float(auc.mean()), 3), auc_std=round(float(auc.std()), 3),
                auc_welch_t_vs_uniform=round(t_stat, 2), auc_per_seed=" ".join(f"{a:.3f}" for a in auc),
                threshold=th, reached=f"{len(reached)}/{len(hits)}",
                itr_median=float(np.median(reached)) if reached else None,
                itr_range=f"{min(reached)}-{max(reached)}" if reached else None,
                steps_median_M=round((np.median(reached) + 1) * STEPS_PER_ITR / 1e6, 1) if reached else None,
                itr_per_seed=" ".join("-" if h is None else str(h) for h in hits)))
        ax.axhline(th, color="#bbbbbb", lw=1, ls=":")
        ax.set_title(task, fontsize=10, color="#222222")
        ax.set_xlabel("env steps (M)", fontsize=9, color="#444444")
        style(ax)
    axes[0].set_ylabel("balanced success (train grid)", fontsize=9, color="#444444")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.text(0.99, 0.01, "dotted: fixed threshold for steps-to-threshold (sample_efficiency.csv)", ha="right",
             fontsize=7, color="#777777")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_curves.png", dpi=200)
    fig.savefig(f"{out}/fig_curves.pdf")
    with open(f"{out}/sample_efficiency.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ----------------------------------------------------------------- final evals
def load_evals(env, split):
    latest = {}
    for d in glob.glob(f"{EV}/{env}_eval_disturbance_{split}_ta4_td20_tdf*"):
        for f in sorted(glob.glob(f"{d}/*/result.json")):
            r = json.load(open(f))
            latest[r["base_policy_path"]] = r
    return latest


def pick(evals, pattern, ckpt):
    return [r for p, r in evals.items() if f"/{pattern}" in p and p.endswith(ckpt)]


def per_seed(evals, env, key, ckpt="state_199.pt"):
    return np.array([r["summary"]["balanced_success"] for r in pick(evals, f"{env}_ppo_diffusion_mlp_{key}", ckpt)
                     if "L400" not in r["base_policy_path"]])


def fig_final(out):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    rows = []
    for j, (task, t) in enumerate(TASKS.items()):
        for i, split in enumerate(["train", "ood"]):
            ev = load_evals(t["env"], split + t["grid"])
            ax = axes[i, j]
            bars = [("ema", np.array([r["summary"]["balanced_success"] for r in
                                      pick(ev, f"{t['env']}_pre_diffusion_mlp_ema", "state_3000.pt")]))]
            bars.append(("dppo", np.array([r["summary"]["balanced_success"] for p, r in ev.items()
                                           if f"/{t['env']}_ppo_diffusion_mlp_ta4" in p and p.endswith("state_199.pt")])))
            bars += [(m, per_seed(ev, t["env"], key)) for m, key in t["runs"].items()]
            for k, (m, v) in enumerate(bars):
                if len(v) == 0:
                    continue
                lab, col = REF.get(m) or METHODS[m]
                ax.bar(k, v.mean(), width=0.72, color=col, label=lab, yerr=v.std() if len(v) > 1 else None,
                       error_kw=dict(lw=1, capsize=2, ecolor="#444444"))
                ax.scatter(np.full(len(v), k) + np.linspace(-0.15, 0.15, len(v)), v, s=6, color="#222222", zorder=3)
                ax.text(k, v.mean() + (v.std() if len(v) > 1 else 0) + 0.015, f"{v.mean():.2f}",
                        ha="center", fontsize=7, color="#333333")
                rows.append(dict(task=task, split=split, method=m, n=len(v), mean=round(v.mean(), 4),
                                 std=round(v.std(), 4), per_seed=" ".join(f"{x:.3f}" for x in v)))
            ax.set_xticks([])
            ax.set_ylim(0, 1.0)
            style(ax)
            if i == 0:
                ax.set_title(task, fontsize=10, color="#222222")
            if j == 0:
                ax.set_ylabel(f"balanced success\n({'train' if i == 0 else 'OOD'} grid)", fontsize=9, color="#444444")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out}/fig_final.png", dpi=200)
    fig.savefig(f"{out}/fig_final.pdf")
    with open(f"{out}/final.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ----------------------------------------------------------------- RQ3
def fig_rq3(out):
    env = "walker2d-medium-v2"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    variants = [("RL policy", "#2a78d6"), ("RFT", "#eb6834"), ("RFT + demos (0.6)", "#f2b38f")]
    sources = [("fas", "FAS rollouts", "fas_a3.0_b0.9_ta4", "rq3fas"), ("uniform", "Uniform rollouts", "uniform_ta4", "rq3uniform"),
               ("frozen", "Pretrained rollouts\n(frozen)", None, "rq3frozen")]
    for ax, split in zip(axes, ["train", "ood"]):
        ev = load_evals(env, split)
        ema = [r["summary"]["balanced_success"] for r in pick(ev, f"{env}_pre_diffusion_mlp_ema", "state_3000.pt")]
        for k, (src, lab, rl_key, rq3) in enumerate(sources):
            # RL: the source run itself (rq3 runs = same config with trajectory saving), frozen = pretrained
            rl = per_seed(ev, env, rq3 + ("_a3.0" if src == "fas" else "_ta4"))
            if src == "frozen":
                rl = np.array(ema)
            vals = [rl,
                    np.array([r["summary"]["balanced_success"] for r in pick(ev, f"{env}_rft_{src}_clip_ta4", "state_10.pt")]),
                    np.array([r["summary"]["balanced_success"] for r in pick(ev, f"{env}_rft_{src}_mix_ta4", "state_4.pt")])]
            for v_i, ((vlab, col), v) in enumerate(zip(variants, vals)):
                x = k + (v_i - 1) * 0.26
                if len(v) == 0:
                    continue
                ax.bar(x, v.mean(), width=0.24, color=col, label=vlab,
                       yerr=v.std() if len(v) > 1 else None, error_kw=dict(lw=1, capsize=2, ecolor="#444444"))
                ax.text(x, v.mean() + (v.std() if len(v) > 1 else 0) + 0.015, f"{v.mean():.2f}", ha="center",
                        fontsize=6.5, color="#333333")
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([s[1] for s in sources], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"walker2d, {'train' if split == 'train' else 'OOD'} grid", fontsize=10, color="#222222")
        style(ax)
    axes[0].set_ylabel("balanced success", fontsize=9, color="#444444")
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.99, 0.01, "RL policy for pretrained rollouts = the frozen pretrained (EMA) policy", ha="right",
             fontsize=7, color="#777777")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(f"{out}/fig_rq3.png", dpi=200)
    fig.savefig(f"{out}/fig_rq3.pdf")


# ----------------------------------------------------------------- per-bin difficulty groups
def fig_bins(out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    groups = ["unsolved\n(<0.2)", "frontier\n(0.2–0.8)", "solved\n(>0.8)"]
    for ax, (task, t) in zip(axes, TASKS.items()):
        ev = load_evals(t["env"], "train" + t["grid"])
        P = {}
        for m, key in t["runs"].items():
            rs = [r for r in pick(ev, f"{t['env']}_ppo_diffusion_mlp_{key}", "state_199.pt") if "L400" not in r["base_policy_path"]]
            if rs:
                P[m] = np.array([[c["success"] for c in r["per_condition"]] for r in rs]).mean(0)
        u = P["uniform"]
        masks = [u < 0.2, (u >= 0.2) & (u <= 0.8), u > 0.8]
        for k, msk in enumerate(masks):
            for mi, (m, p) in enumerate(P.items()):
                x = k + (mi - (len(P) - 1) / 2) * 0.2
                if msk.any():
                    ax.bar(x, p[msk].mean(), width=0.18, color=METHODS[m][1], label=METHODS[m][0] if k == 1 else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\n{m.sum()} bins" for g, m in zip(groups, masks)], fontsize=7.5)
        ax.set_title(task, fontsize=10, color="#222222")
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("mean per-bin success (train grid)", fontsize=9, color="#444444")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.text(0.99, 0.01, "bins grouped by the uniform policy's final success", ha="right", fontsize=7, color="#777777")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_bins.png", dpi=200)
    fig.savefig(f"{out}/fig_bins.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for r in fig_curves(a.out):
        print(r)
    fig_final(a.out)
    fig_rq3(a.out)
    fig_bins(a.out)
    print("wrote", sorted(os.listdir(a.out)))
