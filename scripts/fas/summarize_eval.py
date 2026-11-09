"""
Collect disturbance-grid evaluation results into one table (mean +- std over seeds).

    python scripts/fas/summarize_eval.py $DPPO_LOG_DIR/gym-eval-disturbance [--csv out.csv]

Method is inferred from base_policy_path (gym-pretrain -> pretrained, _fas_, _uniform_, else dppo);
grid (train / ood) from the eval run name.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = ["balanced_success", "worst_bin_success", "success_rate", "balanced_return", "worst_bin_return"]


def method_of(path):
    if "gym-pretrain" in path:
        return "pretrained"
    if "_fas_" in path:
        return "fas"
    if "_uniform_" in path:
        return "uniform"
    return "dppo"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    groups = defaultdict(list)
    for f in glob.glob(os.path.join(args.root, "**", "result.json"), recursive=True):
        with open(f) as fh:
            r = json.load(fh)
        run_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        env = run_name.split("_eval_")[0]
        grid = "ood" if "_ood_" in run_name else "train"
        groups[(env, grid, method_of(r["base_policy_path"]))].append(r["summary"])

    rows = []
    for (env, grid, method), summaries in sorted(groups.items()):
        row = {"env": env, "grid": grid, "method": method, "n": len(summaries)}
        for m in METRICS:
            v = np.array([s[m] for s in summaries])
            row[m] = f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"
        rows.append(row)

    cols = ["env", "grid", "method", "n"] + METRICS
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols} if rows else {}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
