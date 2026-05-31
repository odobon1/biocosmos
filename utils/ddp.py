import torch
import torch.distributed as dist
import os


def setup_ddp():
    assert torch.cuda.is_available(), "CUDA is not available!"
    assert dist.is_available(), "torch.distributed is not available!"

    local_gpu_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_gpu_rank)
    device = torch.device("cuda", local_gpu_rank)

    dist.init_process_group("nccl", device_id=device)
    assert dist.is_initialized(), "torch.distributed failed to initialize!"

    return local_gpu_rank, device

def cleanup_ddp():
    if dist.get_rank() == 0:
        print("Cleaning up DDP...")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def rank0(fn):
    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
    return wrapper