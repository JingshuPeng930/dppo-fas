"""
Report figures (run on the machine holding $DPPO_LOG_DIR):

  fig_curves.png     balanced eval success vs env steps, uniform / FAS / SGS / LP, one panel per task
                     (in-training deterministic evals from fas_history.pkl, mean +- std over seeds)
  fig_final.png      final disturbance-eval success (train grid and OOD grid): pretrained, DPPO, uniform, FAS, FAS + LP gate
  fig_final_all.png  the same with every baseline (SGS, LP, PLR) for the appendix
  fig_rq3.png        RQ3: source RL policy vs RFT on its successful episodes (+ demo-mix ablation)
  fig_bins.png       per-bin success grouped by difficulty for uniform (why FAS gains little)
  fig_hard.png       walker2d harder grid;  fig_hopper_v3.png  hopper without the unlearnable 300 N push
  fig_sparse.png     reward sparsity (reward delay K) on walker2d and hopper;  comparisons.csv  vs uniform with bootstrap CIs
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
# colour hierarchy: our samplers warm and saturated, uniform DR (main comparison) steel blue,
# the reference curriculum baselines light and desaturated, pretrained / nominal DPPO grey
METHODS = collections.OrderedDict([
    ("uniform", ("Uniform DR", "#3F6E9A")),
    ("fas", ("FAS (α=3)", "#E8853D")),
    ("sgs", ("SGS", "#6FA585")),
    ("lp", ("Learning progress", "#958AC4")),
    ("plr", ("PLR (value loss)", "#C9A75A")),
    ("fasp", ("FAS + LP gate", "#B23A2E")),
])
BASELINES = {"sgs", "lp", "plr"}
REF = {"ema": ("Pretrained", "#D9D9D9"), "dppo": ("DPPO (nominal)", "#9A9A9A")}

TASKS = collections.OrderedDict([
    ("walker2d", dict(env="walker2d-medium-v2", grid="", runs={
        "uniform": "uniform_ta4", "fas": "fas_a3.0_b0.9_ta4", "sgs": "sgs_a1.0", "lp": "lp_a1.0", "plr": "plr_a3.0_b0.9_ta4", "fasp": "fasp1_a3.0"})),
    ("hopper", dict(env="hopper-medium-v2", grid="_gv2", runs={
        "uniform": "uniform_gv2_ta4|rq3uniform_gv2_ta4", "fas": "fas_gv2_a3.0_b0.9_ta4|rq3fas_gv2_a3.0_b0.9_ta4", "sgs": "sgs_gv2", "lp": "lp_gv2", "plr": "plr_gv2_a3.0_b0.9_ta4", "fasp": "fasp1_gv2"})),
    ("halfcheetah", dict(env="halfcheetah-medium-v2", grid="", runs={
        "uniform": "uniform_ta4", "fas": "fas_a3.0_b0.9_ta4", "sgs": "sgs_a1.0", "lp": "lp_a1.0", "fasp": "fasp1_a3.0"})),
])


# fixed a priori per task (roughly uniform's level at mid-training; hopper/halfcheetah never reach 0.8)
THRESHOLD = {"walker2d": 0.8, "hopper": 0.5, "halfcheetah": 0.7}


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def boot_ci(a, b, n=20000, seed=0):
    """95% percentile-bootstrap CI of mean(a) - mean(b), resampling seeds within each group"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def comparisons(out):
    """every method vs uniform per task: final train / OOD success, worst-25% bins, AUC; diff, Welch t, bootstrap CI"""
    rows = []
    for task, t in TASKS.items():
        grid = t["grid"]
        ev = {sp: load_evals(t["env"], sp + grid) for sp in ["train", "ood"]}

        def metrics(key):
            m = {}
            for sp in ["train", "ood"]:
                rs = [r for r in pick(ev[sp], f"{t['env']}_ppo_diffusion_mlp_{key}", "state_199.pt")
                      if "L400" not in r["base_policy_path"]]
                per = np.array([[c["success"] for c in r["per_condition"]] for r in rs])
                if len(per):
                    m[sp] = per.mean(1)
                    m[sp + "_worst25"] = np.sort(per, 1)[:, :max(1, per.shape[1] // 4)].mean(1)
            m["auc"] = np.array([eval_curve(d)[1].mean() for d in run_dirs(t["env"], key)])
            return m
        M = {m: metrics(k) for m, k in t["runs"].items()}
        for m in M:
            if m == "uniform":
                continue
            for metric in ["train", "train_worst25", "ood", "auc"]:
                a, b = M[m].get(metric, []), M["uniform"].get(metric, [])
                if len(a) < 2 or len(b) < 2:
                    continue
                lo, hi = boot_ci(a, b)
                rows.append(dict(task=task, method=m, metric=metric, n=len(a), n_uniform=len(b),
                                 mean=round(float(np.mean(a)), 3), uniform=round(float(np.mean(b)), 3),
                                 diff=round(float(np.mean(a) - np.mean(b)), 3), welch_t=round(welch_t(a, b), 2),
                                 ci95_lo=round(lo, 3), ci95_hi=round(hi, 3), ci_excludes_0=bool(lo > 0 or hi < 0)))
    with open(f"{out}/comparisons.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


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
    """key may be "a|b": run groups with the identical training config (e.g. RQ3 source runs = extra seeds)"""
    ds = sorted(d for k in key.split("|") for d in glob.glob(f"{FT}/{env}_ppo_diffusion_mlp_{k}*/*"))
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
            base = m in BASELINES
            ax.plot(x, Y.mean(0), color=col, lw=1.6 if base else 2.2, label=f"{lab} (n={len(cs)})", zorder=2 if base else 3)
            if not base:
                ax.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=col, alpha=0.13, lw=0)
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


def _alts(pattern):
    head, sep, tail = pattern.rpartition("_mlp_")
    return [f"{head}{sep}{t}" for t in tail.split("|")] if sep and "|" in tail else [pattern]


def pick(evals, pattern, ckpt):
    alts = _alts(pattern)
    return [r for p, r in evals.items() if any(f"/{q}" in p for q in alts) and p.endswith(ckpt)]


def per_seed(evals, env, key, ckpt="state_199.pt"):
    return np.array([r["summary"]["balanced_success"] for r in pick(evals, f"{env}_ppo_diffusion_mlp_{key}", ckpt)
                     if "L400" not in r["base_policy_path"]])


MAIN_METHODS = ("uniform", "fas", "fasp")


def fig_final(out, methods=None, name="fig_final"):
    """methods=None: every method (appendix); otherwise only these (main text). Pretrained and DPPO always shown."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2) if methods is None else (9, 5.0))
    rows = []
    for j, (task, t) in enumerate(TASKS.items()):
        for i, split in enumerate(["train", "ood"]):
            ev = load_evals(t["env"], split + t["grid"])
            ax = axes[i, j]
            bars = [("ema", np.array([r["summary"]["balanced_success"] for r in
                                      pick(ev, f"{t['env']}_pre_diffusion_mlp_ema", "state_3000.pt")]))]
            bars.append(("dppo", np.array([r["summary"]["balanced_success"] for p, r in ev.items()
                                           if f"/{t['env']}_ppo_diffusion_mlp_ta4" in p and p.endswith("state_199.pt")])))
            bars += [(m, per_seed(ev, t["env"], key)) for m, key in t["runs"].items() if methods is None or m in methods]
            for k, (m, v) in enumerate(bars):
                if len(v) == 0:
                    continue
                lab, col = REF.get(m) or METHODS[m]
                ax.bar(k, v.mean(), width=0.72, color=col, label=lab, yerr=v.std() if len(v) > 1 else None,
                       error_kw=dict(lw=1, capsize=2, ecolor="#444444"))
                ax.scatter(np.full(len(v), k) + np.linspace(-0.15, 0.15, len(v)), v, s=6, color="#222222", zorder=3)
                ax.text(k, max(v.mean() + (v.std() if len(v) > 1 else 0), v.max()) + 0.025, f"{v.mean():.2f}",
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
    fig.savefig(f"{out}/{name}.png", dpi=200)
    fig.savefig(f"{out}/{name}.pdf")
    if methods is not None:
        return rows
    with open(f"{out}/final.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ----------------------------------------------------------------- RQ3
def fig_rq3(out):
    """rows: walker2d (seeds 0-2, + frozen source and demo-mix variant), hopper (source runs seeds 3-5)"""
    variants = [("RL policy", "#2a78d6"), ("RFT", "#eb6834"), ("RFT + demos (0.6)", "#f2b38f")]
    setups = [
        ("walker2d", "walker2d-medium-v2", "", "_clip_ta4", [
            ("fas", "FAS rollouts", "rq3fas_a3.0"), ("uniform", "Uniform rollouts", "rq3uniform_ta4"),
            ("frozen", "Pretrained rollouts\n(frozen)", None)]),
        ("hopper", "hopper-medium-v2", "_gv2", "_ta4", [
            ("fas", "FAS rollouts", "rq3fas_gv2_a3.0"), ("uniform", "Uniform rollouts", "rq3uniform_gv2_ta4")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.0))
    for row, (task, env, grid, rft_suffix, sources) in enumerate(setups):
        for ax, split in zip(axes[row], ["train", "ood"]):
            ev = load_evals(env, split + grid)
            ema = [r["summary"]["balanced_success"] for r in pick(ev, f"{env}_pre_diffusion_mlp_ema", "state_3000.pt")]
            for k, (src, lab, rl_key) in enumerate(sources):
                # RL: the source run itself (rq3 runs = same config with trajectory saving), frozen = pretrained
                rl = np.array(ema) if rl_key is None else per_seed(ev, env, rl_key)
                vals = [rl,
                        np.array([r["summary"]["balanced_success"] for r in pick(ev, f"{env}_rft_{src}{rft_suffix}", "state_10.pt")]),
                        np.array([r["summary"]["balanced_success"] for r in pick(ev, f"{env}_rft_{src}_mix_ta4", "state_4.pt")])]
                for v_i, ((vlab, col), v) in enumerate(zip(variants, vals)):
                    x = k + (v_i - 1) * 0.26
                    if len(v) == 0:
                        continue
                    ax.bar(x, v.mean(), width=0.24, color=col, label=vlab,
                           yerr=v.std() if len(v) > 1 else None, error_kw=dict(lw=1, capsize=2, ecolor="#444444"))
                    ax.text(x, v.mean() + (v.std() if len(v) > 1 else 0) + 0.015, f"{v.mean():.2f}", ha="center",
                            fontsize=6.5, color="#333333")
            if ema:
                ax.axhline(ema[0], color="#b8b8b8", lw=1, ls="--")
            ax.set_xticks(range(len(sources)))
            ax.set_xticklabels([s[1] for s in sources], fontsize=8)
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{task}, {'train' if split == 'train' else 'OOD'} grid", fontsize=10, color="#222222")
            style(ax)
        axes[row, 0].set_ylabel("balanced success", fontsize=9, color="#444444")
    h, l = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.99, 0.005, "dashed: pretrained (EMA) policy; RL policy for pretrained rollouts = the frozen pretrained policy",
             ha="right", fontsize=7, color="#777777")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
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
        gm = [(g, msk) for g, msk in zip(groups, [u < 0.2, (u >= 0.2) & (u <= 0.8), u > 0.8]) if msk.any()]
        for k, (_, msk) in enumerate(gm):
            for mi, (m, p) in enumerate(P.items()):
                bw = 0.84 / len(P)
                x = k + (mi - (len(P) - 1) / 2) * bw
                if msk.any():
                    ax.bar(x, p[msk].mean(), width=bw * 0.9, color=METHODS[m][1], label=METHODS[m][0] if k == 1 else None)
        ax.set_xticks(range(len(gm)))
        ax.set_xticklabels([f"{g}\n{m.sum()} bins" for g, m in gm], fontsize=7.5)
        ax.set_title(task, fontsize=10, color="#222222")
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("mean per-bin success (train grid)", fontsize=9, color="#444444")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.text(0.99, 0.01, "bins grouped by the uniform policy's final success", ha="right", fontsize=7, color="#777777")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_bins.png", dpi=200)
    fig.savefig(f"{out}/fig_bins.pdf")


# ----------------------------------------------------------------- walker2d: harder grid & reward sparsity
def seed_of(path):
    return int(path.split("/checkpoint/")[0].rsplit("_", 1)[1]) if "/checkpoint/" in path else int(path.rsplit("_", 1)[1])


def seed_metrics(env, split, key, seeds=None):
    """per-seed final balanced success, worst-25%-bins success (state_199 eval on `split`) and eval-curve AUC.
    seeds=None: every seed that exists"""
    ok = lambda s: seeds is None or s in seeds
    rs = [r for r in pick(load_evals(env, split), f"{env}_ppo_diffusion_mlp_{key}", "state_199.pt")
          if "L400" not in r["base_policy_path"] and ok(seed_of(r["base_policy_path"]))]
    per = np.array([[c["success"] for c in r["per_condition"]] for r in rs])
    k = max(1, per.shape[1] // 4)
    auc = [eval_curve(d)[1].mean() for d in run_dirs(env, key) if ok(seed_of(d))]
    return dict(final=per.mean(1), worst=np.sort(per, 1)[:, :k].mean(1), auc=np.array(auc), per_bin=per.mean(0))


def _bars(ax, groups, series, ylabel):
    """groups: x labels; series: {method: [per-seed array per group]} -> bars (mean) + seed dots"""
    n = len(series)
    for mi, (m, vals) in enumerate(series.items()):
        for g, v in enumerate(vals):
            x = g + (mi - (n - 1) / 2) * 0.8 / n
            ax.bar(x, v.mean(), width=0.8 / n * 0.9, color=METHODS[m][1], label=METHODS[m][0] if g == 0 else None)
            ax.scatter(np.full(len(v), x), v, s=8, color="#333333", zorder=3, linewidths=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_title(ylabel, fontsize=9.5, color="#222222")
    style(ax)


def _summary_fig(out, name, env, split, keys, title, note, ood_split=None):
    """left: final / worst-25% / AUC (/ OOD) per seed; right: per-bin success grouped by uniform's difficulty"""
    M = {m: seed_metrics(env, split, k) for m, k in keys.items()}
    groups, ser = ["final success", "worst-25% bins", "AUC (speed)"], {m: [v["final"], v["worst"], v["auc"]] for m, v in M.items()}
    if ood_split:
        groups.append("final success (OOD)")
        for m, k in keys.items():
            ser[m].append(seed_metrics(env, ood_split, k)["final"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ns = sorted({len(v["final"]) for v in M.values()})
    _bars(axes[0], groups, ser, f"{title}: summary ({'/'.join(map(str, ns))} seeds)")
    axes[0].set_ylim(0, 0.9)
    u = M["uniform"]["per_bin"]
    masks = [u < 0.2, (u >= 0.2) & (u <= 0.8), u > 0.8]
    names = ["unsolved (<0.2)", "frontier", "solved (>0.8)"]
    names, masks = zip(*[(g, msk) for g, msk in zip(names, masks) if msk.any()])
    _bars(axes[1], [f"{g}\n{m.sum()} bins" for g, m in zip(names, masks)],
          {m: [np.array([v["per_bin"][msk].mean()]) for msk in masks] for m, v in M.items()},
          f"{title}: per-bin success by difficulty")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.text(0.99, 0.01, note, ha="right", fontsize=7, color="#777777")
    fig.tight_layout()
    fig.savefig(f"{out}/{name}.png", dpi=200)
    fig.savefig(f"{out}/{name}.pdf")
    for m in M:
        if m == "uniform":
            continue
        for metric in ["final", "worst", "auc"]:
            a_, b_ = M[m][metric], M["uniform"][metric]
            print(f"  {name} {m} {metric}: {b_.mean():.3f}->{a_.mean():.3f} (n={len(a_)}/{len(b_)}) "
                  f"t={welch_t(a_, b_):.2f} CI={tuple(round(x, 3) for x in boot_ci(a_, b_))}")
        if ood_split:
            a_, b_ = ser[m][3], ser["uniform"][3]
            print(f"  {name} {m} ood: {b_.mean():.3f}->{a_.mean():.3f} t={welch_t(a_, b_):.2f} "
                  f"CI={tuple(round(x, 3) for x in boot_ci(a_, b_))}")
    for gi, (g, msk) in enumerate(zip(names, masks)):
        print(f"  {name} group {g} ({msk.sum()} bins): " + " ".join(f"{m} {v['per_bin'][msk].mean():.3f}" for m, v in M.items()))
    return M


def fig_hard(out):
    return _summary_fig(out, "fig_hard", "walker2d-medium-v2", "hard",
                        {"uniform": "hard_uniform", "fas": "hard_fas", "sgs": "hard_sgs"}, "harder grid",
                        "walker2d, grid mass 1–1.75 × friction 0.35–1 × push 150–450 N; bins grouped by uniform")


def fig_v3(out):
    """hopper with the unlearnable 300 N push removed (impulse 0/100/200): hypothesis test for 'learnable hard bins'"""
    return _summary_fig(out, "fig_hopper_v3", "hopper-medium-v2", "v3_gv2",
                        {"uniform": "uniform_gv3", "fas": "fas_gv3", "fasp": "fasp1_gv3"}, "hopper, learnable grid",
                        "hopper, friction 0.5/1.0/1.15 × push 0/100/200 N (no 300 N); OOD = standard hopper OOD grid; bins grouped by uniform",
                        ood_split="ood_gv2")


def _points(ax, groups, series, title):
    """per group: mean as a marker joined across groups by a line, individual seeds as faint dots"""
    n = len(series)
    for mi, (m, vals) in enumerate(series.items()):
        lab, col = METHODS[m]
        xs = np.arange(len(groups)) + (mi - (n - 1) / 2) * 0.14
        ax.plot(xs, [v.mean() for v in vals], "-o", color=col, lw=1.8, ms=6, label=lab, zorder=3)
        for x, v in zip(xs, vals):
            ax.scatter(np.full(len(v), x), v, s=16, color=col, alpha=0.35, linewidths=0, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_xlim(-0.4, len(groups) - 0.6)
    ax.set_title(title, fontsize=9.5, color="#222222")
    style(ax)


SPARSE_SEEDS = None  # every available seed (K=1 runs have 5-6 seeds, sparse runs 3-5)


def fig_sparse(out):
    """rows: walker2d, hopper; columns: train final, worst-25%, AUC, OOD final; x: reward delay K"""
    setups = {
        "walker2d": ("walker2d-medium-v2", {1: {"uniform": "uniform_ta4", "fas": "fas_a3.0_b0.9_ta4"},
                                            25: {"uniform": "sparse25_uniform", "fas": "sparse25_fas"},
                                            100: {"uniform": "sparse100_uniform", "fas": "sparse100_fas"}}),
        "hopper": ("hopper-medium-v2", {1: {m: TASKS["hopper"]["runs"][m] for m in ("uniform", "fas")},
                                        25: {"uniform": "sparse25_uniform_gv2", "fas": "sparse25_fas_gv2"},
                                        100: {"uniform": "sparse100_uniform_gv2", "fas": "sparse100_fas_gv2"}}),
    }
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    labels = ["K=1 (dense)", "K=25", "K=100"]
    rows = []
    for r, (task, (env, keys)) in enumerate(setups.items()):
        grid = TASKS[task]["grid"]
        M = {K: {m: seed_metrics(env, "train" + grid, k, SPARSE_SEEDS) for m, k in ks.items()} for K, ks in keys.items()}
        O = {K: {m: seed_metrics(env, "ood" + grid, k, SPARSE_SEEDS)["final"] for m, k in ks.items()} for K, ks in keys.items()}
        cols = [("final", "final success (train grid)"), ("worst", "worst-25% bins"), ("auc", "AUC (speed)")]
        for ax, (metric, title) in zip(axes[r], cols):
            _points(ax, labels, {m: [M[K][m][metric] for K in keys] for m in ["uniform", "fas"]}, f"{task}: {title}")
        _points(axes[r, 3], labels, {m: [O[K][m] for K in keys] for m in ["uniform", "fas"]}, f"{task}: final success (OOD)")
        for ax in axes[r]:  # zoom each panel to its data (points, not bars, so a non-zero baseline is fine)
            ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
            pad = 0.15 * (ys.max() - ys.min())
            ax.set_ylim(ys.min() - pad, ys.max() + pad)
        for K in keys:
            u, f = M[K]["uniform"], M[K]["fas"]
            rows.append(f"{task} K={K} n={len(u['final'])}/{len(f['final'])}: " + "  ".join(
                f"{k} {u[k].mean():.3f}->{f[k].mean():.3f} (t={welch_t(f[k], u[k]):.2f}, CI={tuple(round(x, 3) for x in boot_ci(f[k], u[k]))})" for k in ["final", "worst", "auc"])
                + f"  ood {O[K]['uniform'].mean():.3f}->{O[K]['fas'].mean():.3f} (t={welch_t(O[K]['fas'], O[K]['uniform']):.2f})")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.text(0.99, 0.01, "line: mean over all available seeds; faint dots: individual seeds. Dense reward paid every K env steps (episode return unchanged)",
             ha="right", fontsize=7, color="#777777")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_sparse.png", dpi=200)
    fig.savefig(f"{out}/fig_sparse.pdf")
    return rows

def fig_sparse_compact(out):
    """paper version of fig_sparse: one row, final train success and worst-25% bins for walker2d and hopper"""
    setups = {
        "walker2d": ("walker2d-medium-v2", {1: {"uniform": "uniform_ta4", "fas": "fas_a3.0_b0.9_ta4"},
                                            25: {"uniform": "sparse25_uniform", "fas": "sparse25_fas"},
                                            100: {"uniform": "sparse100_uniform", "fas": "sparse100_fas"}}),
        "hopper": ("hopper-medium-v2", {1: {m: TASKS["hopper"]["runs"][m] for m in ("uniform", "fas")},
                                        25: {"uniform": "sparse25_uniform_gv2", "fas": "sparse25_fas_gv2"},
                                        100: {"uniform": "sparse100_uniform_gv2", "fas": "sparse100_fas_gv2"}}),
    }
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    labels = ["K=1", "K=25", "K=100"]
    k = 0
    for task, (env, keys) in setups.items():
        grid = TASKS[task]["grid"]
        M = {K: {m: seed_metrics(env, "train" + grid, kk, SPARSE_SEEDS) for m, kk in ks.items()} for K, ks in keys.items()}
        for metric, title in [("final", "final success"), ("worst", "worst-25% bins")]:
            ax = axes[k]; k += 1
            _points(ax, labels, {m: [M[K][m][metric] for K in keys] for m in ["uniform", "fas"]}, f"{task}: {title}")
            ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
            pad = 0.15 * (ys.max() - ys.min())
            ax.set_ylim(ys.min() - pad, ys.max() + pad)
    axes[0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_sparse_compact.png", dpi=200)
    fig.savefig(f"{out}/fig_sparse_compact.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for r in fig_curves(a.out):
        print(r)
    fig_final(a.out, name="fig_final_all")
    fig_final(a.out, MAIN_METHODS)
    fig_rq3(a.out)
    fig_bins(a.out)
    for m, v in fig_hard(a.out).items():
        print("hard", m, {k: round(float(v[k].mean()), 3) for k in ["final", "worst", "auc"]})
    for m, v in fig_v3(a.out).items():
        print("v3", m, {k: round(float(v[k].mean()), 3) for k in ["final", "worst", "auc"]})
    fig_sparse_compact(a.out)
    for r in fig_sparse(a.out):
        print("sparse", r)
    for r in comparisons(a.out):
        print("cmp", r)
    print("wrote", sorted(os.listdir(a.out)))
