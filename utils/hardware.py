import os
import subprocess
import re
from typing import Optional


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

def compute_dataloader_workers_prefetch(
    max_n_workers_gpu: Optional[int] = None,
    prefetch_factor: int = 2,
):
    slurm_alloc = get_slurm_alloc()

    n_gpus = max(1, slurm_alloc["n_gpus"])
    n_workers_auto = max(1, (slurm_alloc["n_cpus"] // n_gpus) - 1)

    if max_n_workers_gpu is None:
        n_workers = n_workers_auto
    else:
        n_workers = max(1, min(max_n_workers_gpu, n_workers_auto))

    prefetch_factor = max(1, int(prefetch_factor))

    return n_workers, prefetch_factor, slurm_alloc