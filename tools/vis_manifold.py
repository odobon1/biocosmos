"""
torchrun --standalone --nproc-per-node=auto -m tools.vis_manifold
"""

import torch.distributed as dist

from models import VLMWrapper
from utils.config import get_config_manifold_viz
from utils.utils import paths
from utils.ddp import setup_ddp, cleanup_ddp
from utils.manifold_viz import get_dataloader, generate_manifold_viz


def main():

    setup_ddp()

    rank0 = dist.get_rank() == 0

    # component of plot title that appears in parentheses, set to None for no tag
    # TAG = "base"
    TAG = None

    cfg = get_config_manifold_viz(verbose=rank0)

    dpath_vis = paths["root"] / cfg.rdpath_model / "viz"
    if dpath_vis.exists():
        if rank0:
            print(f"viz directory already exists ({dpath_vis}); skipping t-SNE/PCA generation.")
        cleanup_ddp()
        return

    modelw = VLMWrapper.build(cfg, verbose=rank0)

    dataloader_id = get_dataloader(cfg, "id", modelw)
    dataloader_ood = get_dataloader(cfg, "ood", modelw)

    generate_manifold_viz(cfg, modelw, dataloader_id, dataloader_ood, dpath_vis, tag=TAG, tsne_cfg=cfg.tsne)

    cleanup_ddp()

if __name__ == "__main__":
    main()
