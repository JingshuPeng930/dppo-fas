"""
Compare fine-tuning runs (DPPO / uniform / FAS) across seeds.

    python scripts/fas/plot_compare.py $DPPO_LOG_DIR/gym-finetune --task hopper-medium-v2 --out hopper.png

Finds every run dir with a result.pkl under the root whose name contains --task, infers the
method from the run name (_fas_ / _uniform_ / otherwise dppo), and plots mean +- std over seeds:
  (left)  eval avg episode reward on the env each method trains in, vs env steps
  (right) eval balanced success over all disturbance bins (FAS / uniform only), vs env steps
"""

import argparse
import glob
import os
import pickle
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical slots 1-3 of the reference palette, fixed per method
COLORS = {"dppo": "#2a78d6", "uniform": "#eb6834", "fas": "#1baf7a"}
LABELS = {"dppo": "DPPO (nominal)", "uniform": "DPPO + uniform DR", "fas": "DPPO + FAS"}


def method_of(run_dir):
    name = os.path.basename(os.path.dirname(run_dir))
    if "_fas_" in name:
        return "fas"
    if "_uniform_" in name:
        return "uniform"
    return "dppo"


def load_run(run_dir):
    with open(os.path.join(run_dir, "result.pkl"), "rb") as f:
        res = pickle.load(f)
    itr2step = {r["itr"]: r["step"] for r in res}
    ev = [(r["step"], r["eval_episode_reward"]) for r in res if "eval_episode_reward" in r]
    out = {"reward": np.array(ev) if ev else np.zeros((0, 2))}
    fas_path = os.path.join(run_dir, "fas_history.pkl")
    if os.path.exists(fas_path):
        with open(fas_path, "rb") as f:
            hist = pickle.load(f)["history"]
        pts = []
        for h in hist:
            if h["source"] != "eval":
                continue
            succ = np.bincount(h["cond"], weights=h["success"], minlength=len(h["probs"]))
            cnt = np.bincount(h["cond"], minlength=len(h["probs"]))
            pts.append((itr2step.get(h["itr"], np.nan), np.mean(succ[cnt > 0] / cnt[cnt > 0])))
        out["balanced_success"] = np.array(pts) if pts else np.zeros((0, 2))
    return out


def aggregate(curves):
    """Align seeds on the shared step grid (truncate to shortest run)."""
    n = min(len(c) for c in curves)
    if n == 0:
        return None
    arr = np.stack([c[:n] for c in curves])
    return arr[0, :, 0], arr[:, :, 1].mean(0), arr[:, :, 1].std(0), len(curves)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--task", default="hopper-medium-v2")
    p.add_argument("--out", default="compare.png")
    args = p.parse_args()

    runs = defaultdict(list)
    for res in glob.glob(os.path.join(args.root, "**", "result.pkl"), recursive=True):
        run_dir = os.path.dirname(res)
        if args.task in run_dir:
            runs[method_of(run_dir)].append(load_run(run_dir))
    if not runs:
        raise SystemExit(f"No runs for {args.task} under {args.root}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [("reward", "Eval episode return"), ("balanced_success", "Eval balanced success (all bins)")]
    for ax, (key, title) in zip(axes, panels):
        for m in ("dppo", "uniform", "fas"):
            curves = [r[key] for r in runs.get(m, []) if key in r]
            agg = aggregate(curves) if curves else None
            if agg is None:
                continue
            x, mu, sd, n = agg
            ax.plot(x, mu, color=COLORS[m], lw=2, label=f"{LABELS[m]} (n={n})")
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, lw=0)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle(args.task, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
