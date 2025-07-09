import torch

from models import CLIPWrapper
from utils_eval import ValidationPipeline

import pdb


""" CONFIG PARAMS """

# CLIP_TYPE = "openai_vitb32_hf"
# CLIP_TYPE = "bioclip"
# CLIP_TYPE = "bioclip2"
CLIP_TYPE = "openai_vitb32"
# CLIP_TYPE = "openai_vitb16"
# CLIP_TYPE = "openai_vitl14"
# CLIP_TYPE = "openai_rn50"
# CLIP_TYPE = "openai_rn101"
# CLIP_TYPE = "openai_rn101_yfcc15m"
# CLIP_TYPE = "openai_rn50x4"
# CLIP_TYPE = "openai_rn50x16"
# CLIP_TYPE = "openai_rn50x64"

RUN_NAME       = "test_run_42"  # which train-run to load from (set None to baseline original model)
# RUN_NAME       = None
CHKPT_CRIT     = "comp"  # "comp" / "img2img" --- checkpoint criterion (only applicable if RUN_NAME != None)
CACHED_IMGS    = False  # preload, preprocess, cache all images into memory
BATCH_SIZE_VAL = 512
NUM_WORKERS    = 4  # adjust to CPU cores
SPLIT_NAME     = "D"
TEXT_PREPS     = [["a photo of "]]  # scientific name, BioCLIP-style prepending


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"",
        f"device: {device}",
        f"",
        f"Split -------- {SPLIT_NAME}",
        f"CLIP-type ---- {CLIP_TYPE}",
        f"Checkpoint --- {RUN_NAME} ({CHKPT_CRIT})"
        f"",
        sep="\n"
    )

    modelw = CLIPWrapper(CLIP_TYPE, device, run_name=RUN_NAME, chkpt_crit=CHKPT_CRIT)

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
