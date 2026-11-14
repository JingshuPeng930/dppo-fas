"""
Write the EMA weights of each pretrained checkpoint as a checkpoint the eval model reads.

DPPO fine-tuning starts from the pretrained checkpoint's "ema" weights (DiffusionModel loads "ema"
when present), but the eval model (model/diffusion/diffusion_eval.py) reads "model", the raw
weights. So `evaluate.sh <task> pretrained` scores a different policy than the one DPPO starts
from. The exported file stores the EMA weights under both keys:

    python scripts/fas/export_pretrained_ema.py
    bash scripts/fas/evaluate.sh walker2d-v2 \
        $DPPO_LOG_DIR/gym-pretrain-ema/walker2d-medium-v2_pre_diffusion_mlp_ema/ckpt_0/checkpoint/state_3000.pt train
"""
import glob
import os

import torch

root = os.environ["DPPO_LOG_DIR"]
for env in ("hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"):
    src = glob.glob(f"{root}/gym-pretrain/{env}_pre_diffusion_mlp_ta4_td20/*/checkpoint/state_3000.pt")
    assert len(src) == 1, src
    ck = torch.load(src[0], map_location="cpu", weights_only=True)
    out = f"{root}/gym-pretrain-ema/{env}_pre_diffusion_mlp_ema/ckpt_0/checkpoint/state_3000.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"epoch": ck["epoch"], "model": ck["ema"], "ema": ck["ema"]}, out)
    print(out)
