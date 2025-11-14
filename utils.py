import pickle
from pathlib import Path
import random
import os
import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
import json

import pdb


# CLUSTER = "pace"
CLUSTER = "hpg"


if CLUSTER == "pace":

    dpath_root = Path(os.getcwd())

    paths = {
        "root":    dpath_root,
        "vlm4bio": dpath_root / "VLM4Bio/datasets"
    }
elif CLUSTER == "hpg":

    dpath_group   = Path("/blue/arthur.porto-biocosmos")
    dpath_root    = Path(os.getcwd())
    dpath_data    = dpath_group / "data"
    dpath_nymph   = dpath_data / "datasets/nymphalidae_whole_specimen-v250613"
    dpath_vlm4bio = dpath_data / "datasets/VLM4Bio"

    paths = {
        "hf_cache":         dpath_data / "cache/huggingface/hub",
        "group":            dpath_group,
        "root":             dpath_root,
        "config":           dpath_root / "config",
        "metadata":         dpath_root / "metadata",
        "artifacts":        dpath_root / "artifacts",
        "nymph_imgs":       dpath_nymph / "images",
        "nymph_metadata":   dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv",
        "nymph_phylo_tree": dpath_root / "tree_nymphalidae_chazot2021_all.tree",
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
    split = load_pickle(paths["metadata"] / f"splits/{split_name}/split.pkl")
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
    
class RunningMean:
    """
    Track running mean via Welford's algorithm
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n

    def value(self):
        return self.mean