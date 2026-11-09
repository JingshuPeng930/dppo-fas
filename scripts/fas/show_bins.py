"""
Print per-condition-bin results of one disturbance evaluation (result.json).
Used to calibrate disturbance strength: FAS needs bins that differ in failure rate.

    python scripts/fas/show_bins.py <eval_logdir_or_result.json>
    python scripts/fas/show_bins.py latest          # newest eval under $DPPO_LOG_DIR
"""

import glob
import json
import os
import sys


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "latest"
    if arg == "latest":
        files = glob.glob(os.path.join(os.environ["DPPO_LOG_DIR"], "gym-eval-disturbance/*/*/result.json"))
        path = max(files, key=os.path.getmtime)
    else:
        path = arg if arg.endswith(".json") else os.path.join(arg, "result.json")
    r = json.load(open(path))
    print(path)
    print("policy:", r["base_policy_path"])
    rows = r["per_condition"]
    names = [k for k in rows[0] if k not in ("cond", "n", "success", "return")]
    print("  ".join(f"{n:>14}" for n in names) + "  |  success   return    n")
    for c in rows:
        bar = "#" * int(round(c["success"] * 10))
        print(
            "  ".join(f"{c[n]:>14g}" for n in names)
            + f"  |  {c['success']:7.2f}  {c['return']:7.0f}  {c['n']:3d}  {bar}"
        )
    print("\nsummary:", ", ".join(f"{k}={v:.3f}" for k, v in r["summary"].items()))
    for name, m in r["marginals"].items():
        print(f"success by {name}: " + ", ".join(f"{v}->{s:.2f}" for v, s in m["success"].items()))
    succ = [c["success"] for c in rows]
    print(
        f"\nbins with success == 1: {sum(s == 1 for s in succ)}/{len(succ)}   "
        f"== 0: {sum(s == 0 for s in succ)}/{len(succ)}   "
        f"in between: {sum(0 < s < 1 for s in succ)}/{len(succ)}"
    )


if __name__ == "__main__":
    main()
