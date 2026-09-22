"""
Where the logit bias lives (VLMWrapper.__init__): it goes with the CRITERION, not the model family.

- InfoNCE carries none, whichever the family -- logit_bias = None (as open_clip spells a bias-free model), so
  compute_logits adds nothing. A shared scalar bias cannot move a row softmax, and it is dropped outright rather
  than left in unused: a SigLIP model's pretrained bias is a trainable parameter, and DDP (no
  find_unused_parameters) would wait on a gradient that never comes.
- a BCE-family loss always carries one: a SigLIP model's own, and for a CLIP model (none of its own) a fixed
  0.0 buffer under loss.logits.bce.bias.init: null, a learnable parameter under a set one.

Either family trains under either criterion, so a pretrained model can be held fixed across a
softmax-vs-sigmoid comparison. The real constructor runs here against a stand-in open_clip model.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import models
from models import CLIPWrapper, SigLIPWrapper

SIGLIP_BIAS = -12.9324


class _FakeOpenClipModel(nn.Module):
    """The logit scalars as open_clip ships them: a CLIP model's logit_bias is None, a SigLIP model's a Parameter."""

    def __init__(self, siglip):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(4.6))
        if siglip:
            self.logit_bias = nn.Parameter(torch.tensor(SIGLIP_BIAS))
        else:
            self.logit_bias = None

    def set_grad_checkpointing(self, enable):
        pass


def _build(monkeypatch, model_type, crit, bias_init=None, bias_freeze=False, sep=False):
    siglip = model_type.startswith("siglip")
    # the transforms each wrapper reads: the trailing Normalize's stats, and a Resize's size for img_res
    # (CLIP's at [1], SigLIP's at [0])
    resize, norm = SimpleNamespace(size=(224, 224)), SimpleNamespace(mean=(0.5,) * 3, std=(0.5,) * 3)
    pp = SimpleNamespace(transforms=[resize, resize, norm])
    monkeypatch.setattr(models.open_clip, "create_model_and_transforms", lambda *a, **k: (_FakeOpenClipModel(siglip), pp, pp))
    monkeypatch.setattr(models.open_clip, "get_tokenizer", lambda name: (lambda txts: torch.zeros(1)))
    monkeypatch.setattr(models.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(models.dist, "get_world_size", lambda: 1)
    loss = {
        "crit": crit, "blend": {"lambda": 0.3 if sep else 0.0, "type": "loss"},
        "logits": {"shared": not sep, "scale": {"init": None, "freeze": False},
                   "bce": {"bias": {"init": bias_init, "freeze": bias_freeze}}},
    }
    config = SimpleNamespace(
        model={"arch": {"model_type": model_type, "siglip": {"vis_proj_head": None}}}, device=torch.device("cpu"),
        hw=SimpleNamespace(act_chkpt=False), loss=loss,
    )
    return (SigLIPWrapper if siglip else CLIPWrapper)(config).model


@pytest.mark.parametrize("model_type", ["clip_vitb16", "siglip_vitb16"])
def test_infonce_carries_no_bias_whichever_the_family(monkeypatch, model_type):
    model = _build(monkeypatch, model_type, "infonce", bias_init=-10.0)  # bias.init is never read here
    assert model.logit_bias is None
    # and nothing of it is left for DDP to wait on, or for the optimizer / a checkpoint to carry
    assert [name for name, _ in model.named_parameters()] == ["logit_scale"]
    assert "logit_bias" not in model.state_dict()


def test_infonce_separate_scalars_copy_the_scale_alone(monkeypatch):
    model = _build(monkeypatch, "siglip_vitb16", "infonce", sep=True)
    assert model.logit_bias is None and model.logit_bias2 is None
    assert sorted(name for name, _ in model.named_parameters()) == ["logit_scale", "logit_scale2"]


@pytest.mark.parametrize("crit", ["bce", "bif_bce"])
def test_a_clip_model_gets_a_bias_under_a_bce_family_loss(monkeypatch, crit):
    # bias.init: null -> a fixed 0.0 buffer (nothing to learn from: logits = sim * scale.exp() + 0)
    model = _build(monkeypatch, "clip_vitb16", crit)
    assert not isinstance(model.logit_bias, nn.Parameter) and model.logit_bias.item() == 0.0
    assert "logit_bias" in dict(model.named_buffers())
    # a set init -> a learnable parameter at that value, which bias.freeze then holds
    model = _build(monkeypatch, "clip_vitb16", crit, bias_init=-10.0)
    assert isinstance(model.logit_bias, nn.Parameter) and model.logit_bias.requires_grad
    assert model.logit_bias.item() == pytest.approx(-10.0)
    assert not _build(monkeypatch, "clip_vitb16", crit, bias_init=-10.0, bias_freeze=True).logit_bias.requires_grad


def test_a_siglip_model_keeps_its_own_bias_under_bce(monkeypatch):
    model = _build(monkeypatch, "siglip_vitb16", "bce")
    assert isinstance(model.logit_bias, nn.Parameter) and model.logit_bias.item() == pytest.approx(SIGLIP_BIAS)
    assert _build(monkeypatch, "siglip_vitb16", "bce", bias_init=-5.0).logit_bias.item() == pytest.approx(-5.0)
    # separate scalars: the second pair starts as a copy of the first, bias included
    model = _build(monkeypatch, "siglip_vitb16", "bce", sep=True)
    assert isinstance(model.logit_bias2, nn.Parameter) and model.logit_bias2.item() == pytest.approx(SIGLIP_BIAS)
