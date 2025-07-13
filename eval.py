import torch

from models import VisionLanguageModelWrapper
from utils_eval import ValidationPipeline

import pdb


""" CONFIG PARAMS """

# MODEL_TYPE = "bioclip"
# MODEL_TYPE = "bioclip2"
# MODEL_TYPE = "clip_vitb16"
MODEL_TYPE = "clip_vitb32"
# MODEL_TYPE = "clip_vitl14"
# MODEL_TYPE = "clip_rn50"
# MODEL_TYPE = "clip_rn101"
# MODEL_TYPE = "clip_rn101_yfcc15m"
# MODEL_TYPE = "clip_rn50x4"
# MODEL_TYPE = "clip_rn50x16"
# MODEL_TYPE = "clip_rn50x64"
# MODEL_TYPE = "siglip_vitb16"
# MODEL_TYPE = "siglip_vitb16_384"
# MODEL_TYPE = "siglip_vitl16_384"
# MODEL_TYPE = "siglip_vitso400m14"
# MODEL_TYPE = "siglip2_vitb16"
# MODEL_TYPE = "siglip2_vitb16_384"
# MODEL_TYPE = "siglip2_vitl16_384"
# MODEL_TYPE = "siglip2_vitso400m14"
# MODEL_TYPE = "siglip2_vitgopt16_384"
# MODEL_TYPE = "vitamin_s"
# MODEL_TYPE = "vitamin_s_ltt"
# MODEL_TYPE = "vitamin_b"
# MODEL_TYPE = "vitamin_b_ltt"
# MODEL_TYPE = "vitamin_l"
# MODEL_TYPE = "vitamin_l_256"
# MODEL_TYPE = "vitamin_l_336"
# MODEL_TYPE = "vitamin_l_384"
# MODEL_TYPE = "vitamin_l2"
# MODEL_TYPE = "vitamin_l2_384"
# MODEL_TYPE = "vitamin_xl_384"

# RUN_NAME       = "test_run_42"  # which train-run to load from (set None to baseline original model)
RUN_NAME       = None
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
        f"CLIP-type ---- {MODEL_TYPE}",
        f"Checkpoint --- {RUN_NAME}{'' if RUN_NAME is None else ' (' + CHKPT_CRIT + ')'}"
        f"",
        sep="\n"
    )

    modelw = VisionLanguageModelWrapper(MODEL_TYPE, device, run_name=RUN_NAME, chkpt_crit=CHKPT_CRIT)

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
