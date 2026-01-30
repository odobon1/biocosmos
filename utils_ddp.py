import torch  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import os


def setup_ddp():
    assert torch.cuda.is_available(), "CUDA is not available!"
    assert dist.is_available(), "torch.distributed is not available!"

    dist.init_process_group("nccl")
    assert dist.is_initialized(), "torch.distributed failed to initialize!"

    gpu_rank       = dist.get_rank()
    gpu_world_size = dist.get_world_size()
    local_gpu_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_gpu_rank)
    device = torch.device("cuda", local_gpu_rank)

    return gpu_rank, gpu_world_size, local_gpu_rank, device

def cleanup_ddp():
    print("Cleaning up DDP...")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()