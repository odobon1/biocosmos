import torch
import torch.distributed as dist
import os
from datetime import timedelta


def setup_ddp(pg_timeout=None):
    assert torch.cuda.is_available(), "CUDA is not available!"
    assert dist.is_available(), "torch.distributed is not available!"

    local_gpu_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_gpu_rank)
    device = torch.device("cuda", local_gpu_rank)

    # pg_timeout is the NCCL collective watchdog timeout in seconds (see hardware.yaml). None -> PyTorch
    # default (10 min); training passes cfg.hw.pg_timeout, short-lived eval/tool entrypoints leave it default.
    pg_kwargs = {"device_id": device}
    if pg_timeout is not None:
        pg_kwargs["timeout"] = timedelta(seconds=pg_timeout)
    dist.init_process_group("nccl", **pg_kwargs)
    assert dist.is_initialized(), "torch.distributed failed to initialize!"

    return local_gpu_rank, device

def cleanup_ddp():
    if dist.get_rank() == 0:
        print("Cleaning up DDP...\n")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def rank0(fn):
    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
    return wrapper