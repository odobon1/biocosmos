import pickle
from pathlib import Path
import random
import os
import numpy as np
import torch
import json
from typing import List, Any
from datetime import datetime, timezone

from utils.paths import paths

import pdb


def seed_libs(seed, seed_torch=True):
    random.seed(seed)
    os.putenv("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # (True) trades speed for reproducibility (default is False)

def utc_now():
    """The UTC stamp the metadata files carry as datetime_start / datetime_last_seen."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def save_json(data, fpath):
    # written to a sibling temp file and renamed over fpath, so a concurrent reader (the detached render worker,
    # reading phase_metadata.json while the runner applies a replan) sees the old file or the new one, never a torn one
    fpath_tmp = Path(f"{fpath}.{os.getpid()}.tmp")
    with open(fpath_tmp, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(fpath_tmp, fpath)

def _json_flat_leaves(obj, indent):
    # indent=4 nested dicts, but each leaf (a list of per-trial values) stays on a single line
    if isinstance(obj, dict) and obj:
        pad = " " * (indent + 4)
        body = ",\n".join(
            f"{pad}{json.dumps(k, ensure_ascii=False)}: {_json_flat_leaves(v, indent + 4)}"
            for k, v in obj.items()
        )
        return "{\n" + body + "\n" + " " * indent + "}"
    return json.dumps(obj, ensure_ascii=False)

def save_json_listview(data, fpath):
    with open(fpath, "w") as f:
        f.write(_json_flat_leaves(data, 0))

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

def save_npz(fpath, **arrays):
    # written to a sibling temp file and renamed over fpath (save_json's scheme), so a process killed
    # mid-write (the render worker, terminated by the runner or the job's cancellation) can never leave
    # a torn npz behind for the resume probes to crash on. np.savez appends ".npz" to any filename not
    # already carrying it, so the tmp file is written through an open handle to keep its exact name.
    fpath_tmp = Path(f"{fpath}.{os.getpid()}.tmp")
    with open(fpath_tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(fpath_tmp, fpath)

def load_split(dataset, split):
    fpath_split = paths["metadata"][dataset] / f"splits/{split}/split.pkl"
    split = load_pickle(fpath_split)
    return split

def model_grad_l2_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum()  # accumulate on-device; single host sync at the end
    return total.sqrt().item()
    
def shuffle_list(input: List[Any], seed: int) -> List[int]:
    rng = random.Random(seed)
    input_shuf = input.copy()
    rng.shuffle(input_shuf)
    return input_shuf

def get_subdirectory_names(dir_path):
    return [p.name for p in Path(dir_path).iterdir() if p.is_dir()]
