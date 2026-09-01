"""
torchrun --standalone --nproc-per-node=auto -m tools.protos --dataset <dataset>

Note: only tested with 1 GPU
"""

print("Importing modules...")

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from models import VLMWrapper
from utils.config import get_config_train, load_train_config_dict
from utils.ddp import setup_ddp, cleanup_ddp
from utils.utils import load_split, save_pickle, paths

import pdb


parser = ArgumentParser()
parser.add_argument("--dataset", required=True, choices=("bryo", "cub", "lepid", "nymph"))
args = parser.parse_args()

_, device = setup_ddp()

# base model under the train.yaml base config (split, batch_size, arch); campaign/phase/arm/coord/seed are placeholders
cfg_dict = load_train_config_dict()
cfg_dict.update({"campaign": "protos", "phase": "_screen", "arm": "protos", "coord": "protos", "seed": None, "dataset": args.dataset})
cfg = get_config_train(cfg_dict)
cfg.device = device  # set local device

split_p = load_split(cfg.dataset, cfg.split)
enc2cid = split_p.enc2cid

di_val_id = split_p.get_data("val_id")
di_val_ood = split_p.get_data("val_ood")

partition = ["id"] * len(di_val_id) + ["ood"] * len(di_val_ood)
cids = [enc2cid[d["class_enc"]] for d in di_val_id] + [enc2cid[d["class_enc"]] for d in di_val_ood]
rfpaths = [d["rfpath"] for d in di_val_id] + [d["rfpath"] for d in di_val_ood]

modelw = VLMWrapper.build(cfg, verbose=(dist.get_rank() == 0))
modelw.model = modelw.model.to(device).eval()

fpath_imgs = paths["imgs"][cfg.dataset]

protos = {"id": {}, "ood": {}}

for i in tqdm(range(len(cids) // cfg.batch_size + 1)):
    start = i * cfg.batch_size
    end = min((i + 1) * cfg.batch_size, len(cids))
    if start >= end:
        break

    partition_b = partition[start:end]
    cids_b = cids[start:end]
    rfpaths_b = rfpaths[start:end]

    imgs = [modelw.img_pp_inf(Image.open(fpath_imgs / rfpath).convert("RGB")) for rfpath in rfpaths_b]
    n_imgs = len(imgs)
    imgs = torch.stack(imgs)  # uint8; prep (device, fp32, normalize) happens in embed_images

    with torch.no_grad():
        img_embs = modelw.embed_images(imgs)  # pt[B, D], unit length

    for j in range(n_imgs):
        partition_j = partition_b[j]
        cid_j = cids_b[j]
        emb_j = img_embs[j].cpu()

        if cid_j not in protos[partition_j]:
            protos[partition_j][cid_j] = {"embs": [], "count": 0}

        protos[partition_j][cid_j]["embs"].append(emb_j)
        protos[partition_j][cid_j]["count"] += 1

for partition_k in tqdm(protos.keys()):
    for cid_k in protos[partition_k].keys():

        n_samps = protos[partition_k][cid_k]["count"]
        embs_k = torch.stack(protos[partition_k][cid_k]["embs"])  # pt[N, D]
        proto_k = torch.mean(embs_k, dim=0)  # pt[D]
        proto_k = F.normalize(proto_k, p=2, dim=0)  # normalized to unit length

        protos[partition_k][cid_k] = {"prototype": proto_k, "n_samples": n_samps}

save_pickle(protos, f"prototypes_{cfg.dataset}_cos-cos_1-0.pkl")

cleanup_ddp()
