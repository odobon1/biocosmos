"""
Contract tests for VLMWrapper.freeze (config freeze.text / freeze.image): the flags freeze the towers only,
so the logit scalars (temperature / bias) stay trainable -- with both towers frozen they are all that trains.
"""
import pytest
import torch
import torch.nn as nn

from models import CLIPWrapper, SigLIPWrapper

SCALARS = ["logit_scale", "logit_bias"]
# open_clip parameter names, one per name pattern the freeze loops key on
CLIP_TOWERS = [
    "visual.conv1.weight", "visual.proj",
    "token_embedding.weight", "positional_embedding", "transformer.resblocks.0.attn.in_proj_weight",
    "ln_final.weight", "text_projection",
]
SIGLIP_TOWERS = [
    "visual.trunk.blocks.0.attn.qkv.weight", "visual.head.proj.weight",
    "text.transformer.resblocks.0.mlp.c_fc.weight", "text.text_projection.weight",
]


def _model(names):
    """nn.Module whose parameters carry the given dotted names."""
    root = nn.Module()
    for name in names:
        *path, leaf = name.split(".")
        mod = root
        for part in path:
            if not hasattr(mod, part):
                mod.add_module(part, nn.Module())
            mod = getattr(mod, part)
        mod.register_parameter(leaf, nn.Parameter(torch.zeros(2)))
    return root


@pytest.mark.parametrize("cls, towers", [(CLIPWrapper, CLIP_TOWERS), (SigLIPWrapper, SIGLIP_TOWERS)], ids=["clip", "siglip"])
def test_freeze_both_towers_keeps_logit_scalars_trainable(cls, towers):
    wrapper = cls.__new__(cls)  # skip __init__ (pretrained load); freeze only touches .model
    wrapper.model = _model(towers + SCALARS)
    wrapper.freeze(True, True)
    frozen = {name for name, p in wrapper.model.named_parameters() if not p.requires_grad}
    assert frozen == set(towers)
