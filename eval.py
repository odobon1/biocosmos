import torch
import pandas as pd

from models import CLIPWrapper
from utils_eval import EvaluationPipeline, ValidationPipeline

import pdb


torch.set_printoptions(profile="full")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


# config params
CLIP_TYPE      = "bioclip"  # "openai" / "bioclip"
CACHED_IMGS    = False  # preload, preprocess, cache all images into memory
BATCH_SIZE_VAL = 512
NUM_WORKERS    = 4  # adjust to CPU cores
SPLIT_NAME     = "D"
TEXT_PREP_TYPE = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
TEXT_BASE_TYPE = "tax"  # "tax" / "sci"


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"",
        f"device: {device}",
        f"",
        f"Split ------------ {SPLIT_NAME}",
        f"CLIP-type -------- {CLIP_TYPE}",
        f"Text Type -------- {TEXT_PREP_TYPE}",
        f"Text Base Type --- {TEXT_BASE_TYPE}",
        f"",
        sep="\n"
    )

    modelw = CLIPWrapper(CLIP_TYPE, device)

    val_pipe = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_base_type =TEXT_BASE_TYPE,
        text_prep_type =TEXT_PREP_TYPE,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
    )

    _ = val_pipe.evaluate(modelw)

if __name__ == "__main__":
    main()
