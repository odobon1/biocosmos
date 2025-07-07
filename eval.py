import torch

from models import CLIPWrapper
from utils_eval import ValidationPipeline

import pdb


# config params
CLIP_TYPE      = "bioclip"  # "openai" / "bioclip"
CHECKPOINT     = "clip_o1"  # which checkpoint to load from (set None for original model)
CRITERION      = "comp"  # "comp" / "img2img" (only applicable if CHECKPOINT != None)
CACHED_IMGS    = False  # preload, preprocess, cache all images into memory
BATCH_SIZE_VAL = 512
NUM_WORKERS    = 4  # adjust to CPU cores
SPLIT_NAME     = "C"
TEXT_PREPS     = [["a photo of "]]  # scientific name, BioCLIP-style prepending


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"",
        f"device: {device}",
        f"",
        f"Split ------------ {SPLIT_NAME}",
        f"CLIP-type -------- {CLIP_TYPE}",
        f"",
        sep="\n"
    )

    modelw = CLIPWrapper(CLIP_TYPE, device, checkpoint=CHECKPOINT, criterion=CRITERION)

    val_pipe = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_preps     =TEXT_PREPS,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
    )

    val_pipe.evaluate(modelw)

if __name__ == "__main__":
    main()
