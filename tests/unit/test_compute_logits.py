"""
Contract tests for VLMWrapper.compute_logits's `center` modes (loss.logits.bce.center).

- None --------- plain scale + bias.
- "sim" -------- forward centering of the scaled sims (changes the operating point); dL/dsim zero-sum.
- "grad_proj" -- forward identical to None; backward-only zero-sum projection of dL/dsim at the sim
  node, leaving BOTH logit scale and bias grads equal to the raw (None) ones.
- "grad_proj2" - forward identical to None; projection at the scaled sims: bias grad raw, scale grad
  projected. Encoder gradient identical to grad_proj.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models import VLMWrapper

B = 64


def _make_stub():
    model = SimpleNamespace(
        logit_scale=torch.nn.Parameter(torch.tensor(2.3)),
        logit_bias=torch.nn.Parameter(torch.tensor(-0.5)),
        logit_scale2=torch.nn.Parameter(torch.tensor(1.7)),
        logit_bias2=torch.nn.Parameter(torch.tensor(0.2)),
    )
    return SimpleNamespace(model=model, _unwrapped_model=model)


def _run(center, secondary=False, center_global=None):
    """Forward + backward through a BCE loss; returns (logits, dL/dsim, scale.grad, bias.grad)."""
    torch.manual_seed(0)
    stub = _make_stub()
    sim = torch.randn(B, B).requires_grad_(True)
    targs = (torch.rand(B, B) < 0.05).float()
    logits = VLMWrapper.compute_logits(stub, sim, False, center, secondary=secondary, center_global=center_global)
    loss = F.binary_cross_entropy_with_logits(logits, targs)
    loss.backward()
    scale = stub.model.logit_scale2 if secondary else stub.model.logit_scale
    bias = stub.model.logit_bias2 if secondary else stub.model.logit_bias
    return logits.detach(), sim.grad, scale.grad, bias.grad


def test_none_plain():
    logits, _, _, _ = _run(None)
    torch.manual_seed(0)
    sim = torch.randn(B, B)
    assert torch.allclose(logits, sim * torch.tensor(2.3).exp() - 0.5)


def test_sim_centers_forward_and_projects_grad():
    logits, g_sim, _, _ = _run("sim")
    torch.manual_seed(0)
    scaled = torch.randn(B, B) * torch.tensor(2.3).exp()
    assert torch.allclose(logits, scaled - scaled.mean() - 0.5)
    assert abs(g_sim.sum().item()) < 1e-5


@pytest.mark.parametrize("mode", ["grad_proj", "grad_proj2"])
def test_grad_proj_forward_identity(mode):
    logits_none, _, _, _ = _run(None)
    logits_gs, _, _, _ = _run(mode)
    assert torch.equal(logits_gs, logits_none)


@pytest.mark.parametrize("mode", ["grad_proj", "grad_proj2"])
def test_grad_proj_zero_sum_sim_grad(mode):
    _, g_sim, _, _ = _run(mode)
    assert abs(g_sim.sum().item()) < 1e-5
    # the projection changed the gradient (it's not the raw BCE one)
    _, g_sim_none, _, _ = _run(None)
    assert not torch.allclose(g_sim, g_sim_none)


def test_grad_proj_variants_same_encoder_grad():
    _, g_sim_1, _, _ = _run("grad_proj")
    _, g_sim_2, _, _ = _run("grad_proj2")
    assert torch.allclose(g_sim_1, g_sim_2)


def test_grad_proj_scale_bias_grads_raw():
    # sim-node projection: scale AND bias grads must match the None mode
    _, _, g_scale_none, g_bias_none = _run(None)
    _, _, g_scale_gs, g_bias_gs = _run("grad_proj")
    assert torch.allclose(g_scale_gs, g_scale_none)
    assert torch.allclose(g_bias_gs, g_bias_none)


def test_grad_proj2_bias_raw_scale_projected():
    # scaled-sim projection: bias grad raw, scale grad projected (differs from None mode)
    _, _, g_scale_none, g_bias_none = _run(None)
    _, _, g_scale_gs, g_bias_gs = _run("grad_proj2")
    assert torch.allclose(g_bias_gs, g_bias_none)
    assert not torch.allclose(g_scale_gs, g_scale_none)


def test_secondary_honors_own_flag():
    logits, g_sim, _, _ = _run("grad_proj", secondary=True)
    torch.manual_seed(0)
    sim = torch.randn(B, B)
    assert torch.allclose(logits, sim * torch.tensor(1.7).exp() + 0.2)  # forward untouched
    assert abs(g_sim.sum().item()) < 1e-5


def test_grad_proj_center_global_subtracts_constant():
    # chunked path: the precomputed full-batch grad mean is subtracted as a constant at the sim node;
    # scale/bias grads stay raw exactly as in the per-matrix mode
    c = torch.tensor(0.123)
    _, g_none, g_scale_none, g_bias_none = _run(None)
    _, g_cg, g_scale, g_bias = _run("grad_proj", center_global=c)
    torch.testing.assert_close(g_cg, g_none - c)
    torch.testing.assert_close(g_scale, g_scale_none)
    torch.testing.assert_close(g_bias, g_bias_none)


def test_grad_proj2_center_global_subtracts_scaled_constant():
    # constant applied at the scaled-sim node -> reaches sim multiplied by e^t; bias grad stays raw
    c = torch.tensor(0.123)
    _, g_none, _, g_bias_none = _run(None)
    _, g_cg, _, g_bias = _run("grad_proj2", center_global=c)
    torch.testing.assert_close(g_cg, g_none - c * torch.tensor(2.3).exp())
    torch.testing.assert_close(g_bias, g_bias_none)


def test_sim_center_global_matches_full_forward_and_scalar_grads():
    # chunked path passes the global sim mean: forward and scale/bias grads must match the full-batch
    # "sim" mode (the sim-tile grad differs by design -- the projection routes through the in-graph
    # mean, which on the chunked path lives on the embedding leaves, not the tile)
    torch.manual_seed(0)
    m = torch.randn(B, B).mean()
    logits_full, _, g_scale_full, g_bias_full = _run("sim")
    logits_cg, _, g_scale_cg, g_bias_cg = _run("sim", center_global=m)
    torch.testing.assert_close(logits_cg, logits_full)
    torch.testing.assert_close(g_scale_cg, g_scale_full)
    torch.testing.assert_close(g_bias_cg, g_bias_full)


