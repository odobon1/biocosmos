"""
python -m tools.probe_batch_size
"""

import torch  # type: ignore[import]
from torch import amp  # type: ignore[import]
import gc
from typing import List

from models import VLMWrapper, CLIP_MODELS, SIGLIP_MODELS, VITAMIN_MODELS
from utils_config import get_config_train

import pdb


def simulate_batch_train(
    modelw,
    B:      int,
    device: torch.device,
) -> int:

    # dynamic per-batch weights so indexing stays in-bounds
    modelw.class_wts      = torch.ones(B,   device=device)
    modelw.class_pair_wts = torch.ones((B, B), device=device)

    modelw.model.train()

    labels = list(range(B))
    imgs   = torch.randn(B, 3, modelw.img_res, modelw.img_res, device=device, requires_grad=True)
    texts  = ["a photo of a butterfly"] * B

    modelw.model.zero_grad(set_to_none=True)
    with amp.autocast(device_type="cuda"):
        loss, _, _, _ = modelw.batch_step(imgs, texts, labels, None)

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

    config_train = get_config_train(verbose=False)

    config_train.model_type = model_id
    config_train.loss_type  = loss_type

    modelw = VLMWrapper.build(config_train)
    modelw.set_targ_type(config_train.targ_type)

    bs_ok   = 0
    vram_ok = 0
    for bs in sizes:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            vram_ok = simulate_batch_train(modelw, bs, device)
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
    models_infonce = list(CLIP_MODELS.keys()) + list(VITAMIN_MODELS.keys())

    for model_id in models_bce:
        probe_model(model_id, device, sizes, "bce")
    for model_id in models_infonce:
        probe_model(model_id, device, sizes, "infonce1")

if __name__ == "__main__":
    main()
