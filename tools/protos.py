"""
torchrun --standalone --nproc-per-node=auto -m tools.protos

Note: only tested with 1 GPU
"""

print("Importing modules...")

import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
from PIL import Image  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from models import VLMWrapper
from utils.config import get_config_eval
from utils.ddp import setup_ddp, cleanup_ddp
from utils.utils import load_pickle, save_pickle, paths

import pdb


split_p = load_pickle(paths["metadata"]["nymph"] / "splits/P38-42/split.pkl")

partition = ["id"] * len(split_p.data_indexes["validation"]["id"]["sids"])
sids = split_p.data_indexes["validation"]["id"]["sids"]
rfpaths = split_p.data_indexes["validation"]["id"]["rfpaths"]

partition += ["ood"] * len(split_p.data_indexes["validation"]["ood"]["sids"])
sids += split_p.data_indexes["validation"]["ood"]["sids"]
rfpaths += split_p.data_indexes["validation"]["ood"]["rfpaths"]

gpu_rank, _, _, device = setup_ddp()
config_eval = get_config_eval(verbose=(gpu_rank==0))

modelw = VLMWrapper.build(config_eval, verbose=(gpu_rank==0))
modelw.model = modelw.model.to(device).eval()

fpath_imgs = paths["imgs"]["nymph"]

protos = {"id": {}, "ood": {}}

for i in tqdm(range(len(sids) // config_eval.batch_size + 1)):
    start = i * config_eval.batch_size
    end = min((i + 1) * config_eval.batch_size, len(sids))
    if start >= end:
        break

    partition_b = partition[start:end]
    sids_b = sids[start:end]
    rfpaths_b = rfpaths[start:end]

    imgs = [modelw.img_pp_val(Image.open(fpath_imgs / rfpath).convert("RGB")) for rfpath in rfpaths_b]
    n_imgs = len(imgs)
    imgs = torch.stack(imgs).to(device)

    with torch.no_grad():
        img_embs = modelw.model.encode_image(imgs)  # pt[B, D]
        img_embs = F.normalize(img_embs, p=2, dim=1)  # normalized to unit length

    for j in range(n_imgs):
        partition_j = partition_b[j]
        sid_j = sids_b[j]
        emb_j = img_embs[j].cpu()

        if sid_j not in protos[partition_j]:
            protos[partition_j][sid_j] = {"embs": [], "count": 0}

        protos[partition_j][sid_j]["embs"].append(emb_j)
        protos[partition_j][sid_j]["count"] += 1

for partition_k in tqdm(protos.keys()):
    for sid_k in protos[partition_k].keys():

        n_samps = protos[partition_k][sid_k]["count"]
        embs_k = torch.stack(protos[partition_k][sid_k]["embs"])  # pt[N, D]
        proto_k = torch.mean(embs_k, dim=0)  # pt[D]
        proto_k = F.normalize(proto_k, p=2, dim=0)  # normalized to unit length

        protos[partition_k][sid_k] = {"prototype": proto_k, "n_samples": n_samps}

save_pickle(protos, "prototypes_cos-cos_1-0.pkl")

cleanup_ddp()