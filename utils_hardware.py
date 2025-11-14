import os
import subprocess
import re


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