import torch  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import os


def setup_ddp():
    assert torch.cuda.is_available(), "CUDA is not available!"
    assert dist.is_available(), "torch.distributed is not available!"

    dist.init_process_group("nccl")
    assert dist.is_initialized(), "torch.distributed failed to initialize!"

    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return rank, world_size, local_rank, device