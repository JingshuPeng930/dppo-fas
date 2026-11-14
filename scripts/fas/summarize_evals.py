# usage: python summarize_evals.py <env prefix e.g. walker2d-medium-v2>
# For each (split, run, checkpoint) keep the latest eval per checkpoint; print mean +- std over seeds + Welch t.
import json, glob, os, sys, collections
import numpy as np
env = sys.argv[1]
base = os.path.expanduser("~/dppo_fas/log/gym-eval-disturbance")
splits = collections.defaultdict(list)  # RL checkpoints are evaluated with tdf10, pretrained / RFT with tdf0
for d in glob.glob(f"{base}/{env}_eval_disturbance_*_tdf*"):
    splits[d.split("_eval_disturbance_")[1].rsplit("_ta4_td20_tdf", 1)[0]].append(d)
for split in sorted(splits):
    latest = {}
    files = [f for d in splits[split] for f in glob.glob(f"{d}/*/result.json")]
    for f in sorted(files, key=os.path.getmtime):
        r = json.load(open(f)); latest[r["base_policy_path"]] = r["summary"]
    groups = collections.defaultdict(dict)
    for p, s in latest.items():
        parts = p.split("/")
        run, seed, ck = parts[-4], parts[-3].split("_")[-1], parts[-1]
        name = run.replace(env + "_", "").replace("_ta4_td20_tdf10", "") + ":" + ck.replace(".pt", "")
        groups[name][seed] = s
    print(f"== {env} {split}")
    stats = {}
    for name in sorted(groups):
        g = groups[name]; seeds = sorted(g)
        b = np.array([g[k]["balanced_success"] for k in seeds]); w = np.array([g[k]["worst_bin_success"] for k in seeds])
        ret = np.array([g[k]["balanced_return"] for k in seeds]); stats[name] = b
        sd = b.std(ddof=1) if len(b) > 1 else 0.0
        print(f"  {name:48s} n={len(b)} succ {b.mean():.3f} ± {sd:.3f}  worst {w.mean():.2f}  ret {ret.mean():6.0f}  per-seed {np.round(b, 3).tolist()}")
    for a, c in [("ppo_diffusion_mlp_fas_a3.0_b0.9:state_199", "ppo_diffusion_mlp_uniform:state_199"),
                 ("ppo_diffusion_mlp_fas_gv2_a3.0_b0.9:state_199", "ppo_diffusion_mlp_uniform_gv2:state_199")]:
        if a in stats and c in stats and len(stats[a]) > 1 and len(stats[c]) > 1:
            x, y = stats[a], stats[c]
            t = (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            print(f"  Welch t ({a.split(':')[0]} - uniform) = {t:.2f}")
