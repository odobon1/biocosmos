from torch import dist

from models import VLMWrapper
from utils.eval import EvaluationPipeline
from utils.utils import get_text_template, PrintLog
from utils.config import get_config_eval
from utils.ddp import setup_ddp, cleanup_ddp

import pdb


def main():
    _, device = setup_ddp()

    config_eval = get_config_eval(verbose=(dist.get_rank() == 0))
    config_eval.device = device  # set local device

    modelw = VLMWrapper.build(config_eval, verbose=(dist.get_rank() == 0))
    modelw.set_class_wts(config_eval)
    if config_eval.loss2["mix"] != 0.0:
        modelw.set_class_wts(config_eval, secondary=True)

    text_template = get_text_template(config_eval.text_template, dataset=config_eval.dataset)
    eval_pipe = EvaluationPipeline(
        config_eval,
        text_template,
        modelw.img_pp_inf,
    )

    eval_metrics, _, _, _ = eval_pipe.evaluate(modelw, loss_flag=False)

    PrintLog.eval(eval_metrics, eval_pipe)

    cleanup_ddp()


if __name__ == "__main__":
    main()