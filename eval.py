import torch.distributed as dist  # type: ignore[import]

from models import VLMWrapper
from utils_eval import ValidationPipeline
from utils import get_text_preps
from utils_config import get_config_eval
from utils_ddp import setup_ddp

import pdb


def main():
    rank, _, _, device = setup_ddp()

    config_eval = get_config_eval(verbose=(rank==0))
    config_eval.device = device  # set local device

    modelw = VLMWrapper.build(config_eval, verbose=(rank==0))
    modelw.set_class_wts(config_eval)
    if config_eval.loss2["mix"] != 0.0:
        modelw.set_class_wts(config_eval, secondary=True)

    text_preps = get_text_preps(config_eval.text_preps)
    val_pipe   = ValidationPipeline(config_eval, text_preps, modelw.img_pp_val)

    val_pipe.run_validation(modelw, verbose_batch_loss=config_eval.dev['verbose_batch_loss'])

    dist.destroy_process_group()  # DDP cleanup

if __name__ == "__main__":
    main()