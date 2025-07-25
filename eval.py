import torch

from models import VLMWrapper
from utils_eval import ValidationPipeline
from utils import compute_dataloader_workers_prefetch

import pdb


""" CONFIG PARAMS """

MODEL_TYPE = "siglip_vitb16"

TRIAL_NAME  = None  # which trial to load from (set None for original model)
# TRIAL_NAME  = "dev/dev/42"
SAVE_CRIT   = "comp"  # "comp" / "img2img" --- model save criterion (only applicable if TRIAL_NAME != None)
CACHED_IMGS = False  # preload, preprocess, cache all images into memory
BATCH_SIZE  = 1024
SPLIT_NAME  = "S29-0"
TEXT_PREPS  = [["a photo of "]]  # scientific name, BioCLIP-style prepending

N_WORKERS, PREFETCH_FACTOR, _ = compute_dataloader_workers_prefetch()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelw = VLMWrapper.build(MODEL_TYPE, device, trial_name=TRIAL_NAME, save_crit=SAVE_CRIT)

    print(
        f"device: {device}",
        f"",
        f"Split -------- {SPLIT_NAME}",
        f"Model Type --- {modelw.type}",
        f"Trial -------- {TRIAL_NAME}{'' if TRIAL_NAME is None else ' (' + SAVE_CRIT + ')'}",
        f"",
        sep="\n"
    )

    val_pipe = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_preps     =TEXT_PREPS,
        batch_size     =BATCH_SIZE,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        n_workers      =N_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    val_pipe.run_validation(modelw)

if __name__ == "__main__":
    main()
