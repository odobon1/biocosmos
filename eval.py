import torch.distributed as dist  # type: ignore[import]

from models import VLMWrapper
from utils.eval import ValidationPipeline
from utils.utils import get_text_preps, PrintLog
from utils.config import get_config_eval
from utils.ddp import setup_ddp, cleanup_ddp

import pdb


def main():
    gpu_rank, _, _, device = setup_ddp()

    config_eval = get_config_eval(verbose=(gpu_rank==0))
    config_eval.device = device  # set local device

    modelw = VLMWrapper.build(config_eval, verbose=(gpu_rank==0))
    modelw.set_class_wts(config_eval)
    if config_eval.loss2["mix"] != 0.0:
        modelw.set_class_wts(config_eval, secondary=True)

    text_preps = get_text_preps(config_eval.text_preps)
    val_pipe   = ValidationPipeline(config_eval, text_preps, modelw.img_pp_val)

    scores_val, _, _, _ = val_pipe.run_validation(modelw)

    if gpu_rank == 0:
        PrintLog.eval(scores_val)

    cleanup_ddp()

if __name__ == "__main__":
    main()