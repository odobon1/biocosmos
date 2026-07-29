import os
import subprocess
import re
import threading
import time
from pathlib import Path
from typing import Optional

import torch


def apply_backend_flags(hw):
    # torch backend wall-clock switches from hardware.yaml (semantics-neutral up to cudnn.benchmark's
    # algo choice). cudnn.benchmark is owned here, not by seed_libs. cuDNN conv TF32
    # (torch.backends.cudnn.allow_tf32) is left at torch's default (true).
    #
    # Hardcoded, deliberately not a config knob: the eval retrieval metrics run their similarity
    # matmuls in fp32, so TF32 here would truncate the mantissa, perturb the sort order, and move
    # i2i/i2t/t2i mAP -- a silent metric shift from a "speed" flag. Training matmuls run under
    # autocast (mixed_prec), which is unaffected by this.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = hw.cudnn_benchmark


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

def _bounded_memcg():
    """(dir, limit_bytes) of the enclosing memory-limited cgroup. The leaf cgroup (SLURM task) is
    unbounded; the OOM-kill limit sits at the job level, so walk up to the first ancestor with a
    bounded memory.max -- its accounting covers every process in the job (all ranks, DataLoader
    workers, the render worker)."""
    root = Path("/sys/fs/cgroup")
    rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    dpath = root / rel.lstrip("/")
    while dpath != root:
        fpath_limit = dpath / "memory.max"
        if fpath_limit.exists():
            limit = fpath_limit.read_text().strip()
            if limit != "max":
                return dpath, int(limit)
        dpath = dpath.parent
    raise RuntimeError("no memory-limited cgroup found (not running under a SLURM job?)")


class _RamPeakTracker(threading.Thread):
    """Daemon thread polling the job cgroup's memory.current, keeping the running max so RAM
    readings are peaks over the tracker's lifetime (the trial process) rather than instantaneous
    values at snapshot time. The kernel's own watermark (memory.peak) can't serve here: the
    cgroup -- and so its watermark -- spans the whole SLURM job (every trial in the campaign),
    and resetting it needs kernel >= 6.12."""

    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.fpath_current = _bounded_memcg()[0] / "memory.current"
        self.peak = 0

    def run(self):
        while True:
            self.peak = max(self.peak, int(self.fpath_current.read_text()))
            time.sleep(self.interval)


_ram_peak_tracker = None

def start_ram_peak_tracker(interval):
    """Arm per-process RAM peak polling; read_cgroup_ram folds the tracked peak into its reading
    from then on. Idempotent."""
    global _ram_peak_tracker
    if _ram_peak_tracker is None:
        _ram_peak_tracker = _RamPeakTracker(interval)
        _ram_peak_tracker.start()

def read_cgroup_ram():
    """(used_bytes, limit_bytes) of host RAM for the enclosing memory-limited cgroup (see
    _bounded_memcg). used is the peak since start_ram_peak_tracker() when armed, else the
    instantaneous usage."""
    dpath, limit = _bounded_memcg()
    used = int((dpath / "memory.current").read_text())
    if _ram_peak_tracker is not None:
        used = max(used, _ram_peak_tracker.peak)
    return used, limit


def compute_dataloader_workers_prefetch(
    max_n_workers_gpu: Optional[int] = None,
    prefetch_factor: int = 2,
):
    slurm_alloc = get_slurm_alloc()

    n_gpus = max(1, slurm_alloc["n_gpus"])
    # One rank per GPU. Reserve 2 cores per rank for non-decode load -- the main/DDP process and the
    # DataLoader pin_memory thread -- so decode workers don't oversubscribe the cores. (Reserving only 1
    # left no slack for the pin thread: 16 CPUs / 2 GPUs -> 7 workers/rank -> 14 workers + 2 main = 16,
    # fully packed.) Dividing the live SLURM core count by n_gpus makes this scale across 1/2/4 GPUs
    # without retuning, for whatever (n_cpus, n_gpus) the alloc actually grants.
    n_workers_auto = max(1, (slurm_alloc["n_cpus"] // n_gpus) - 2)

    if max_n_workers_gpu is None:
        n_workers = n_workers_auto
    else:
        n_workers = max(1, min(max_n_workers_gpu, n_workers_auto))

    prefetch_factor = max(1, int(prefetch_factor))

    return n_workers, prefetch_factor, slurm_alloc
