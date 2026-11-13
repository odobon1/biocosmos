from contextlib import contextmanager
import pickle
from pathlib import Path
import platform
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

def tmp_fpath(fpath):
    """The sibling temp path an atomic write of `fpath` goes through (atomic_write): named by node + pid, so no two
    writers of one target share it -- the ranks of a trial differ in pid, and processes on different nodes of a
    shared filesystem, which can share a pid, differ in node."""
    return Path(f"{fpath}.{platform.node()}.{os.getpid()}.tmp")

@contextmanager
def atomic_write(fpath, mode="wb", encoding=None):
    """Open a sibling temp file (tmp_fpath) for the with-block to write and, once it has, publish it as `fpath` by
    an atomic rename: a concurrent reader -- or a resume after a kill mid-write -- sees the old file or the new one,
    never a torn one. The data is fsynced before the rename, so a write error the filesystem reports late (NFS can
    surface disk-full / over-quota only at fsync or close) raises here, with `fpath` still the old file. A write
    that fails takes its temp file with it."""
    fpath_tmp = tmp_fpath(fpath)
    try:
        with open(fpath_tmp, mode, encoding=encoding) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(fpath_tmp, fpath)
    except BaseException:
        try:
            os.unlink(fpath_tmp)
        except OSError:
            pass
        raise

@contextmanager
def dir_lock(dpath):
    """Exclusive flock on `dpath`/.lock, held for the with-block: the one locking convention of the artifact tree
    (utils.train.arm_lock, update_json). Not reentrant: never nest two holds of one dir in one process. Released
    by the OS if the holder dies."""
    import fcntl  # Unix-only; imported here so the module stays importable on Windows
    with open(dpath / ".lock", "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        yield

def save_json(data, fpath):
    # published atomically (atomic_write), so a concurrent reader (the detached render worker, reading
    # phase_metadata.json while the runner applies a replan) sees the old file or the new one, never a torn one
    with atomic_write(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

@contextmanager
def update_json(fpath):
    """Read-modify-write of a shared metadata file as ONE critical section: yields `fpath`'s content for the
    with-block to mutate and writes it back on a clean exit, all under the flock of the dir that holds the file
    (dir_lock). Every writer that updates a campaign's shared metadata and counters in place (campaign_metadata.json,
    phase_metadata.json, coord_metadata.json, trial_metadata.json) goes through this, so two updates landing
    together -- trial processes at their checkpoints, the runner counting a crash, a reselection -- are serialized
    instead of one overwriting the other from a stale read. A leaf lock: the with-block does nothing slow and takes
    no other lock (an arm_lock holder may take this one, never the reverse)."""
    with dir_lock(fpath.parent):
        data = load_json(fpath)
        yield data
        save_json(data, fpath)

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
    with atomic_write(fpath, "w") as f:
        f.write(_json_flat_leaves(data, 0))

def load_json(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
    return data

def save_pickle(obj, picklepath):
    with atomic_write(picklepath) as f:
        pickle.dump(obj, f)

def load_pickle(picklepath):
    with open(picklepath, "rb") as f:
        obj = pickle.load(f)
    return obj

def save_npz(fpath, **arrays):
    # published atomically (atomic_write, save_json's scheme), so a process killed
    # mid-write (the render worker, terminated by the runner or the job's cancellation) can never leave
    # a torn npz behind for the resume probes to crash on. np.savez appends ".npz" to any filename not
    # already carrying it, so the tmp file is written through an open handle to keep its exact name.
    with atomic_write(fpath) as f:
        np.savez(f, **arrays)

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
