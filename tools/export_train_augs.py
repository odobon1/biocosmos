"""
Export stochastic training augmentations for a shuffled subset of the train split.

python -m tools.export_train_augs
"""

from __future__ import annotations

import random
from pathlib import Path

import open_clip
from PIL import Image

from models import CLIP_MODELS, SIGLIP_MODELS
from utils.data import MaybeConvertMode, build_train_augmentation_transforms
from utils.config import get_config_train
from utils.utils import load_split, paths


NUM_IMAGES = 10
NUM_AUGS = 8
OUTPUT_DIR = Path("tools/image_aug")


def resolve_img_res_from_model(model_type: str) -> int:
    if model_type in CLIP_MODELS:
        model_name, _, quick_gelu = CLIP_MODELS[model_type]
    elif model_type in SIGLIP_MODELS:
        model_name, _, quick_gelu = SIGLIP_MODELS[model_type]
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    _, _, img_pp_inf = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,
        force_quick_gelu=quick_gelu,
    )

    resize_size = img_pp_inf.transforms[0].size
    if isinstance(resize_size, int):
        return resize_size
    if isinstance(resize_size, tuple):
        return resize_size[0]
    return int(resize_size[0])

def main() -> None:
    if NUM_IMAGES <= 0:
        raise ValueError("NUM_IMAGES must be greater than 0")
    if NUM_AUGS <= 0:
        raise ValueError("NUM_AUGS must be greater than 0")

    cfg = get_config_train()

    split = load_split(cfg.dataset, cfg.split)
    train_rows = list(split.get_data("train"))
    random.shuffle(train_rows)

    n_imgs = min(NUM_IMAGES, len(train_rows))
    if n_imgs < NUM_IMAGES:
        print(f"WARNING: requested {NUM_IMAGES} images but train split only has {len(train_rows)}; exporting {n_imgs}.")

    output_root = OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    img_res = resolve_img_res_from_model(cfg.arch["model_type"])
    augmenter = build_train_augmentation_transforms(img_res, aug_cfg=cfg.aug)
    convert_mode = MaybeConvertMode()
    imgs_root = paths["imgs"][cfg.dataset]

    for img_idx, row in enumerate(train_rows[:n_imgs], start=1):
        source_path = imgs_root / row["rfpath"]
        with Image.open(source_path) as opened_image:
            image = opened_image.convert("RGB")

        image_dir = output_root / f"image{img_idx}"
        image_dir.mkdir(parents=True, exist_ok=True)

        for aug_idx in range(1, NUM_AUGS + 1):
            aug_image = augmenter(image)
            aug_image = convert_mode(aug_image)
            aug_path = image_dir / f"aug{aug_idx}.png"
            aug_image.save(aug_path, format="PNG")

    print(f"Exported {n_imgs} images x {NUM_AUGS} augmentations to {output_root}")


if __name__ == "__main__":
    main()