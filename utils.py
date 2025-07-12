import pickle
from pathlib import Path
import random
import os
import numpy as np
import torch
import subprocess
import re


""" CONFIG PARAMS """

# MACHINE = "pace"
MACHINE = "hpg"


if MACHINE == "pace":
    paths = {
        "repo_o" : Path("/home/hice1/odobon3/Documents/biocosmos"),
        "vlm4bio" : Path("VLM4Bio/datasets")
    }
elif MACHINE == "hpg":

    dpath_biocosmos        = Path("/blue/arthur.porto-biocosmos")
    dpath_repo_o           = dpath_biocosmos / "odobon3.gatech/biocosmos"
    # dpath_nymph            = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v240606"
    # fpath_nymph_metadata   = dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v240606.csv"
    dpath_nymph            = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v250613"
    fpath_nymph_metadata   = dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv"
    dpath_nymph_1          = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v240606"
    fpath_nymph_1_metadata = dpath_nymph_1 / "metadata/data_meta-nymphalidae_whole_specimen-v240606.csv"
    dpath_nymph_2          = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v250613"
    fpath_nymph_2_metadata = dpath_nymph_2 / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv"
    dpath_vlm4bio          = dpath_biocosmos / "data/datasets/VLM4Bio"

    paths = {
        "biocosmos" :        dpath_biocosmos,
        "repo_o" :           dpath_repo_o,
        "metadata_o" :       dpath_repo_o / "metadata_o",
        "artifacts" :        dpath_repo_o / "artifacts",
        "nymph" :            dpath_nymph,
        "nymph_imgs" :       dpath_nymph / "images",
        "nymph_metadata" :   fpath_nymph_metadata,
        "nymph_1" :          dpath_nymph_1,
        "nymph_1_metadata" : fpath_nymph_1_metadata,
        "nymph_2" :          dpath_nymph_2,
        "nymph_2_metadata" : fpath_nymph_2_metadata,
        "vlm4bio" :          dpath_vlm4bio,
    }

def get_slurm_alloc():
    job_id = os.getenv("SLURM_JOB_ID")
    out = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
    tres = re.search(r"TRES=([^ ]+)", out)

    info = {}
    for pair in tres.group(1).split(","):
        key, val = pair.split("=")
        info[key] = val

    alloc = {
        "gpus": int(info.get("gres/gpu", "0")),
        "cpus": int(info.get("cpu", "0")),
        "ram":  int(info.get("mem", "0").rstrip("G")),
    }

    return alloc

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

def write_pickle(obj, picklepath):
    with open(picklepath, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(picklepath):
    with open(picklepath, "rb") as f:
        obj = pickle.load(f)
    return obj
