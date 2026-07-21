"""
python -m tools.probe_batch_size

Intended to only be run on a single GPU
"""

import torch
from torch import amp
import gc
from typing import List

from models import VLMWrapper, CLIP_MODELS, SIGLIP_MODELS
from utils.config import get_config_train, load_train_config_dict
from utils.loss import Criterion
from utils.utils import load_split

import pdb


DATASET = "bryo"


def simulate_batch_train(
    modelw,
    B:         int,
    device:    torch.device,
    n_classes: int,
) -> int:

    modelw.model.train()

    class_enc_b = torch.randint(low=0, high=n_classes, size=(B,), device=device, dtype=torch.long)
    imgs_b      = torch.randn(B, 3, modelw.img_res, modelw.img_res, device=device, requires_grad=True)
    txts_b      = ["a photo of a butterfly"] * B

    modelw.model.zero_grad(set_to_none=True)
    with amp.autocast(device_type="cuda"):
        loss, _, _, _, _, _, _ = modelw.batch_step(imgs_b, txts_b, class_enc_b, None)

    opt = torch.optim.SGD((p for p in modelw.model.parameters() if p.requires_grad), lr=1e-5)
    loss.backward(); opt.step()

    torch.cuda.synchronize(device)
    peak_vram = int(torch.cuda.max_memory_allocated(device))

    return peak_vram

def probe_model(
    model_id:  str,
    device:    torch.device,
    sizes:     List[int],
    loss_type: str,
):
    print(f"\n=== {model_id} ===")

    cfg_dict = load_train_config_dict()
    cfg_dict.update({"campaign": "tool", "setting": "tool", "seed": None, "dataset": DATASET})
    config_train = get_config_train(cfg_dict)
    # bandaid ~ override target types to be aligned
    config_train.loss["targ"]  = "iw"
    config_train.loss2["targ"] = "iw"

    config_train.model_type = model_id
    config_train.loss_type  = loss_type

    modelw = VLMWrapper.build(config_train)
    modelw.crit1 = Criterion.build(config_train.loss, DATASET, config_train.split, config_train.train_pt, device)
    modelw.crit2 = Criterion.build(config_train.loss2, DATASET, config_train.split, config_train.train_pt, device) if config_train.loss2["mix"] != 0.0 else None

    n_classes = len(load_split(DATASET, config_train.split).class_counts[config_train.train_pt])

    bs_ok   = 0
    vram_ok = 0
    for bs in sizes:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            vram_ok = simulate_batch_train(modelw, bs, device, n_classes)
            bs_ok   = bs
            print(f"  ✔ BS = {bs:<5} Peak VRAM = {vram_ok / 1024**3:7.1f} GiB")
        except torch.cuda.OutOfMemoryError:
            print(f"  ✖ OOM at BS = {bs}")
            break

    del modelw
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Max BS = {bs_ok} (Peak VRAM ≈ {vram_ok / 1024**3:.1f} GiB)")

def main():
    
    EXP2_MIN = 6
    EXP2_MAX = 15

    device = torch.device("cuda")

    sizes = [2 ** p for p in range(EXP2_MIN, EXP2_MAX + 1)]
    print(f"Probing batch sizes: {sizes}")

    models_bce     = list(SIGLIP_MODELS.keys())
    models_infonce = list(CLIP_MODELS.keys())

    for model_id in models_bce:
        probe_model(model_id, device, sizes, "bce")
    for model_id in models_infonce:
        probe_model(model_id, device, sizes, "infonce1")

if __name__ == "__main__":
    main()