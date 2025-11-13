from models import VLMWrapper
from utils_eval import ValidationPipeline
from utils import get_text_preps
from utils_imb import compute_class_wts
from utils_config import get_config_eval

import pdb


def main():

    config_eval = get_config_eval()

    modelw = VLMWrapper.build(config_eval)
    class_wts, class_pair_wts = compute_class_wts(config_eval)
    modelw.set_class_wts(class_wts, class_pair_wts)
    modelw.set_targ_type(config_eval.loss['targ'])

    text_preps = get_text_preps(config_eval.text_preps)
    val_pipe   = ValidationPipeline(config_eval, text_preps, modelw.img_pp_val)

    val_pipe.run_validation(modelw, verbose_batch_loss=config_eval.dev['verbose_batch_loss'])

if __name__ == "__main__":
    main()
