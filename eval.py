import torch
import pandas as pd

from models import CLIPWrapper
from utils_eval import EvaluationPipeline

import pdb


torch.set_printoptions(profile="full")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


# config params
CLIP_TYPE      = "bioclip"  # "openai" / "bioclip"
CACHED_IMGS    = False  # preload, preprocess, cache all images into memory
BATCH_SIZE     = 512
NUM_WORKERS    = 4  # adjust to CPU cores
SPLIT_NAME     = "D"
TEXT_PREP_TYPE = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
TEXT_BASE_TYPE = "tax"  # "tax" / "sci"


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"({device})",
        f"Split ------------ {SPLIT_NAME}",
        f"CLIP-type -------- {CLIP_TYPE}",
        f"Text Type -------- {TEXT_PREP_TYPE}",
        f"Text Base Type --- {TEXT_BASE_TYPE}",
        sep="\n"
    )

    modelw = CLIPWrapper(CLIP_TYPE, device)

    id_val_pipe = EvaluationPipeline(
        split_type     ="id_val", 
        split_name     =SPLIT_NAME, 
        text_base_type =TEXT_BASE_TYPE, 
        text_prep_type =TEXT_PREP_TYPE,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
        modes          =["img2txt", "img2img", "txt2img"],
    )

    eval_scores, time_elapsed_val = id_val_pipe.evaluate(modelw)

    print(
        f"",
        f"img2txt Prec@1 --- {eval_scores['img2txt_prec1']:.2%}",
        f"img2img mAP ------ {eval_scores['img2img_map']:.4f}",
        f"txt2img mAP ------ {eval_scores['txt2img_map']:.4f}",
        f"",
        f"Elapsed Time: {time_elapsed_val:.2f} (s)",
        sep="\n"
    )

if __name__ == "__main__":
    main()
