"""
Learning curves of DPPO-uniform vs DPPO-FAS from the eval iterations stored in fas_history.pkl
(every val_freq iterations the deterministic policy runs a round-robin over all bins).

Two metrics per eval iteration, mean +- std over seeds:
  balanced success  -- mean over bins of per-bin success
  hard-bin success  -- same, restricted to the k bins where a reference policy (the pretrained
                       one, via its disturbance-eval result.json) has the lowest success
plus the area under each curve (mean over eval iterations) as a sample-efficiency number.

    python scripts/fas/learning_curves.py \
        --run uniform "$DPPO_LOG_DIR/gym-finetune/hopper-medium-v2_*uniform_gv2*/*" \
        --run FAS     "$DPPO_LOG_DIR/gym-finetune/hopper-medium-v2_*fas_gv2*/*" \
        --hard-ref <pretrained eval dir or result.json> --hard-k 12 --out hopper_gv2_curves.png
"""

import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order


def load_curve(run_dir, n_bins, hard):
    h = pickle.load(open(os.path.join(run_dir, "fas_history.pkl"), "rb"))["history"]
    itrs, bal, hrd = [], [], []
    for e in h:
        if e["source"] != "eval":
            continue
        cond, succ = np.asarray(e["cond"]), np.asarray(e["success"], dtype=float)
        cnt = np.bincount(cond, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_bin = np.bincount(cond, weights=succ, minlength=n_bins) / cnt
        itrs.append(e["itr"])
        bal.append(np.nanmean(per_bin))
        hrd.append(np.nanmean(per_bin[hard]) if hard is not None else np.nan)
    return np.array(itrs), np.array(bal), np.array(hrd)


def aggregate(curves):
    """Align seeds on eval iterations; use whatever seeds have reached each iteration."""
    all_itr = sorted(set().union(*[set(c[0].tolist()) for c in curves]))
    out = {"itr": [], "n": [], "bal_m": [], "bal_s": [], "hard_m": [], "hard_s": []}
    for it in all_itr:
        b = [c[1][c[0] == it][0] for c in curves if it in c[0]]
        hd = [c[2][c[0] == it][0] for c in curves if it in c[0]]
        out["itr"].append(it)
        out["n"].append(len(b))
        out["bal_m"].append(np.mean(b))
        out["bal_s"].append(np.std(b))
        out["hard_m"].append(np.mean(hd))
        out["hard_s"].append(np.std(hd))
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "GLOB"), required=True)
    ap.add_argument("--hard-ref", default=None)
    ap.add_argument("--hard-k", type=int, default=12)
    ap.add_argument("--common-itr", action="store_true",
                    help="AUC only over iterations every seed of every method has reached")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="learning_curves.png")
    args = ap.parse_args()

    hard, n_bins = None, None
    if args.hard_ref:
        p = args.hard_ref if args.hard_ref.endswith(".json") else os.path.join(args.hard_ref, "result.json")
        ref = json.load(open(p))
        ref_succ = np.array([c["success"] for c in ref["per_condition"]])
        n_bins = len(ref_succ)
        hard = np.argsort(ref_succ, kind="stable")[: args.hard_k]
        print(f"hard bins (reference success): mean {ref_succ[hard].mean():.3f} over {args.hard_k} bins")

    results = {}
    for label, pattern in args.run:
        dirs = sorted(d for d in glob.glob(pattern) if os.path.exists(os.path.join(d, "fas_history.pkl")))
        assert dirs, f"no runs with fas_history.pkl match {pattern}"
        if n_bins is None:
            n_bins = len(pickle.load(open(os.path.join(dirs[0], "fas_history.pkl"), "rb"))["history"][0]["probs"])
        curves = [load_curve(d, n_bins, hard) for d in dirs]
        results[label] = (aggregate(curves), curves, dirs)
        print(f"{label}: {len(dirs)} seeds, last eval itr per seed {[int(c[0][-1]) for c in curves]}")

    max_common = min(int(c[0][-1]) for _, cs, _ in results.values() for c in cs)
    print(f"\n{'method':10s} {'AUC balanced':>16s} {'AUC hard':>16s} {'final balanced':>16s} {'final hard':>14s}"
          + (f"   (AUC and final at itr <= {max_common})" if args.common_itr else ""))
    rows = []
    for label, (agg, curves, _) in results.items():
        def per_seed(idx):
            v = []
            for c in curves:
                m = c[0] <= max_common if args.common_itr else np.ones_like(c[0], bool)
                v.append(np.nanmean(c[idx][m]))
            return np.mean(v), np.std(v)
        ab, ah = per_seed(1), per_seed(2)
        last = (lambda c: c[0] == max_common) if args.common_itr else (lambda c: c[0] == c[0][-1])
        fb = [c[1][last(c)][0] for c in curves]
        fh = [c[2][last(c)][0] for c in curves]
        print(f"{label:10s} {ab[0]:8.3f} ± {ab[1]:.3f} {ah[0]:8.3f} ± {ah[1]:.3f} "
              f"{np.mean(fb):8.3f} ± {np.std(fb):.3f} {np.nanmean(fh):7.3f} ± {np.nanstd(fh):.3f}")
        for i, it in enumerate(agg["itr"]):
            rows.append([label, int(it), int(agg["n"][i]), agg["bal_m"][i], agg["bal_s"][i],
                         agg["hard_m"][i], agg["hard_s"][i]])

    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "itr", "n_seeds", "balanced_mean", "balanced_std", "hard_mean", "hard_std"])
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("bal", "Balanced success (all bins)")]
    if hard is not None:
        panels.append(("hard", f"Success on the {args.hard_k} hardest bins"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels):
        for color, (label, (agg, curves, _)) in zip(COLORS, results.items()):
            m, s = agg[f"{key}_m"], agg[f"{key}_s"]
            ax.plot(agg["itr"], m, color=color, lw=2, label=f"{label} (n={agg['n'].min()}-{agg['n'].max()} seeds)" if agg["n"].min() != agg["n"].max()
                    else f"{label} (n={len(curves)} seeds)")
            ax.fill_between(agg["itr"], m - s, m + s, color=color, alpha=0.15, lw=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("fine-tuning iteration (80k env steps each)")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#9c9a93")
    axes[0].set_ylabel("success rate (mean ± std over seeds)")
    axes[0].legend(frameon=False, loc="upper left")
    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nsaved {args.out} and {os.path.splitext(args.out)[0]}.csv")


if __name__ == "__main__":
    main()
