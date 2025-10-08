import pickle
from pathlib import Path
import random
import os
import numpy as np
import torch  # type: ignore[import]
import json
import subprocess
import re


""" CONFIG PARAMS """

# CLUSTER = "pace"
CLUSTER = "hpg"


if CLUSTER == "pace":
    paths = {
        "repo_o": Path("/home/hice1/odobon3/Documents/biocosmos"),
        "vlm4bio": Path("VLM4Bio/datasets")
    }
elif CLUSTER == "hpg":

    dpath_biocosmos        = Path("/blue/arthur.porto-biocosmos")
    dpath_repo_o           = dpath_biocosmos / "odobon3.gatech/biocosmos"
    dpath_data             = dpath_biocosmos / "data"
    dpath_nymph            = dpath_data / "datasets/nymphalidae_whole_specimen-v250613"
    fpath_nymph_metadata   = dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv"
    dpath_vlm4bio          = dpath_data / "datasets/VLM4Bio"

    paths = {
        "hf_cache":         dpath_data / "cache/huggingface/hub",
        "biocosmos":        dpath_biocosmos,
        "repo_o":           dpath_repo_o,
        "metadata_o":       dpath_repo_o / "metadata_o",
        "artifacts":        dpath_repo_o / "artifacts",
        "nymph":            dpath_nymph,
        "nymph_imgs":       dpath_nymph / "images",
        "nymph_metadata":   fpath_nymph_metadata,
        "vlm4bio":          dpath_vlm4bio,
    }

def seed_libs(seed):
    if seed is not None:
        random.seed(seed)
        os.putenv("PYTHONHASHSEED", str(seed))
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True  # (True) trades speed for reproducibility (default is False)
        torch.backends.cudnn.benchmark     = False  # (False) trades speed for reproducibility (default is False)

def save_json(data, fpath):
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
    return data

def save_pickle(obj, picklepath):
    with open(picklepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(picklepath):
    with open(picklepath, "rb") as f:
        obj = pickle.load(f)
    return obj

def load_split(split_name):
    split = load_pickle(paths["metadata_o"] / f"splits/{split_name}/split.pkl")
    return split

def get_text_preps(text_preps_type):

    TEXT_PREPS_MIXED = [
        [
            "",
            "a photo of ",  # BioCLIP-style prepending
            "a photo of a ",  # OpenAI CLIP-style prepending
        ],
    ]

    TEXT_PREPS_BIOCLIP_SCI = [["a photo of $SCI$"]]  # scientific name, BioCLIP-style prepending

    COMBO_TEMPS_TRAIN = [
        [
            "",
            "a photo of ",
        ],
        [
            "",
            "$AAN$ ",
        ],
        [
            "",
            "$SEX$",
        ],
        [
            "$SCI$",
            "$TAX$",
            "$COM$",
        ],
        [
            "",
            " butterfly",
        ],
        [
            "",
            "$POS$",
        ],
    ]


    if text_preps_type == "combo_temps":
        return COMBO_TEMPS_TRAIN
    if text_preps_type == "mixed":
        return TEXT_PREPS_MIXED
    elif text_preps_type == "bioclip_sci":
        return TEXT_PREPS_BIOCLIP_SCI

def get_slurm_alloc():
    job_id = os.getenv("SLURM_JOB_ID")
    out    = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
    tres   = re.search(r"TRES=([^ ]+)", out)

    info = {}
    for pair in tres.group(1).split(","):
        key, val = pair.split("=")
        info[key] = val

    slurm_alloc = {
        "n_gpus": int(info.get("gres/gpu", "0")),
        "n_cpus": int(info.get("cpu", "0")),
        "ram":    int(info.get("mem", "0").rstrip("G")),
    }

    return slurm_alloc

def compute_dataloader_workers_prefetch():
    slurm_alloc     = get_slurm_alloc()
    n_workers       = slurm_alloc["n_cpus"]
    prefetch_factor = min(n_workers, 8)

    return n_workers, prefetch_factor, slurm_alloc
