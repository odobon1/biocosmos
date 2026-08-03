"""
The preprocessors emit uint8 and the fp32 cast + normalization happen on-device
(normalize_imgs_u8, via VLMWrapper.prep_imgs). These tests pin the split pipeline to the old
in-worker arithmetic (to_tensor + Normalize): bit-identical, not just close.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
from torchvision.transforms.functional import to_tensor

from utils.data import (
    MaybeConvertMode,
    MaybePILToTensor,
    make_image_preprocessor_inference,
    normalize_imgs_u8,
)

NORM_MEAN = (0.5, 0.26, 0.71)
NORM_STD = (0.27, 0.31, 0.29)


def _norm_tensors():
    mean = torch.as_tensor(NORM_MEAN, dtype=torch.float32).view(-1, 1, 1)
    std = torch.as_tensor(NORM_STD, dtype=torch.float32).view(-1, 1, 1)
    return mean, std


def _random_pil(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), mode="RGB")


def test_normalize_imgs_u8_bit_identical_to_worker_pipeline():
    img = _random_pil(224, 224)
    ref = Normalize(mean=NORM_MEAN, std=NORM_STD)(to_tensor(img))

    mean, std = _norm_tensors()
    out = normalize_imgs_u8(MaybePILToTensor()(img), mean, std)

    assert out.dtype == torch.float32
    assert torch.equal(out, ref)


def test_inference_preprocessor_plus_normalize_matches_old_compose():
    img = _random_pil(320, 260, seed=1)  # non-square, larger than crop -> exercises Resize + CenterCrop
    res = 224

    old_pp = Compose([
        Resize(size=res, interpolation=Image.BICUBIC),
        CenterCrop(size=(res, res)),
        MaybeConvertMode(),
        to_tensor,
        Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    ref = old_pp(img)

    mean, std = _norm_tensors()
    u8 = make_image_preprocessor_inference(res)(img)
    assert u8.dtype == torch.uint8
    assert u8.shape == (3, res, res)
    out = normalize_imgs_u8(u8, mean, std)

    assert torch.equal(out, ref)


def test_normalize_imgs_u8_batched_broadcast():
    imgs = [_random_pil(224, 224, seed=s) for s in range(3)]
    mean, std = _norm_tensors()

    batch = torch.stack([MaybePILToTensor()(img) for img in imgs], dim=0)
    out = normalize_imgs_u8(batch, mean, std)

    for i, img in enumerate(imgs):
        ref = Normalize(mean=NORM_MEAN, std=NORM_STD)(to_tensor(img))
        assert torch.equal(out[i], ref)


def test_normalize_imgs_u8_rejects_non_uint8():
    mean, std = _norm_tensors()
    with pytest.raises(TypeError, match="uint8"):
        normalize_imgs_u8(torch.rand(2, 3, 8, 8), mean, std)
