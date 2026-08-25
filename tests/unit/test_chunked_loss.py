"""
Equivalence tests for the tiled/chunked global-batch BCE-family loss (hardware.loss_chunk_size).

chunked_bce_loss_backward must reproduce the loss and gradients (wrt image/text embeddings and the
primary/secondary logit scale/bias) of the full-batch path (BCECriterion.__call__ /
BifurcatedBCECriterion.__call__ blended by _global_batch_loss), up to floating-point summation
order -- across the full BCE-family config space: mp/sp/tax/phylo targets, cls_imb.norm, a
BCE-family loss2 mix, mix_unit_scale, and bif_bce's two-branch tiles (half-live logit scalars,
row-wise DSMR, targ_mass_neut, per-branch centering).
"""
import importlib
import math
import sys
import types

import numpy as np
import pytest
import torch

from utils.head import compute_sim
from utils.phylo import PhyloVCV as RealPhyloVCV  # captured real class before the fake below


def import_loss_module():
    fake_phylo = types.ModuleType("utils.phylo")

    class DummyPhyloVCV:
        """Constant soft target (0.25); block builder agrees with the full matrix by construction."""
        def __init__(self, dataset: str, split: str, train_pt: str,
                     batch_size: int, htarg_shuf: bool = False, seed: int | None = None) -> None:
            self.dataset = dataset

        def get_targs_batch(self, targ_data_b):
            n = len(targ_data_b)
            return torch.full((n, n), 0.25)

        def make_targ_block_fn(self, targ_data_b, device):
            B = len(targ_data_b)
            return lambda rs, re: torch.full((re - rs, B), 0.25, device=device)

    fake_phylo.PhyloVCV = DummyPhyloVCV
    sys.modules["utils.phylo"] = fake_phylo
    sys.modules.pop("utils.loss", None)
    return importlib.import_module("utils.loss")


L = import_loss_module()
# phylo-target params for get_phylo_vcv's constructor call (DummyPhyloVCV ignores them)
L.configure_phylo_targs(split="D10", train_pt="train", batch_size=4, htarg_shuf=False, seed=None)


def _cfg(crit="bce", targ="mp", dsmr=True, focal_gamma=2.0, sim="cos",
         cls_imb_norm=False, center=None, neut=False):
    return {
        "crit": crit, "sim": sim, "targ": targ,
        "bce": {"targ_mass_neut": neut},  # read by bif_bce only
        "wting": {
            "cls_imb": {"type": "inv_freq", "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999},
                        "norm": cls_imb_norm},
            **({"focal": {"gamma": focal_gamma}} if focal_gamma > 0.0 else {}),  # config load prunes the block when gamma = 0.0
            "bce": {"dsmr": dsmr},
        },
        "logits": {"temp": {"clamp": False}, "bce": {"center": center, "bias": {}}},
    }


CRIT_CLS = {"bce": L.BCECriterion, "bif_bce": L.BifurcatedBCECriterion}


def _make_crit(cfg, K, B):
    cls = CRIT_CLS[cfg["crit"]]
    crit = cls.__new__(cls)  # bypass build_wting (no dataset needed)
    crit.cfg = cfg
    crit.device = torch.device("cpu")
    crit.batch_size = B
    g = torch.Generator().manual_seed(K)
    crit.counts = torch.randint(1, 1000, (K,), generator=g).to(torch.float64)
    crit.wt_mean = 1.0
    return crit


def _params(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "scale": torch.tensor(2.3, requires_grad=True),
        "bias": torch.tensor(-0.5, requires_grad=True),
        "scale2": torch.tensor(1.7, requires_grad=True),
        "bias2": torch.tensor(0.2, requires_grad=True),
    }


class _ZSG(torch.autograd.Function):
    """Mirror of models._ZeroSumGrad (identity forward, g - g.mean() backward)."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        return g - g.mean()


class _ZSGC(torch.autograd.Function):
    """Mirror of models._ZeroSumGradConst (identity forward, g - c backward)."""
    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x

    @staticmethod
    def backward(ctx, g):
        return g - ctx.c, None


def _compute_logits_fn(p):
    """Stub mirroring VLMWrapper.compute_logits's center/half_live semantics (incl. the chunked
    path's center_global hook) over the toy scale/bias params."""
    def compute_logits(sim, clamp, center=None, secondary=False, center_global=None, half_live=False):
        s = p["scale2"] if secondary else p["scale"]
        b = p["bias2"] if secondary else p["bias"]
        if half_live:
            s = 0.5 * s + 0.5 * s.detach()
            b = 0.5 * b + 0.5 * b.detach()
        if clamp:
            s = s.clamp(max=math.log(100))
        if center == "grad_proj":
            sim = _ZSG.apply(sim) if center_global is None else _ZSGC.apply(sim, center_global)
        sim_scaled = sim * s.exp()
        if center == "grad_proj2":
            sim_scaled = _ZSG.apply(sim_scaled) if center_global is None else _ZSGC.apply(sim_scaled, center_global)
        if center == "sim":
            sim_scaled = sim_scaled - (sim_scaled.mean() if center_global is None else center_global * s.exp())
        return sim_scaled + b
    return compute_logits


def _make_targ_data(B, K, R, class_encs_b):
    """targ_data carrying every field any targ_type needs (rank_encs for tax, cid/dataset for phylo)."""
    g = torch.Generator().manual_seed(99)
    rank_encs = torch.randint(0, 3, (B, R), generator=g).tolist()
    return [{"rank_encs": rank_encs[i], "cid": f"c{int(class_encs_b[i])}", "dataset": "cub"} for i in range(B)]


def _full_reference(crit1, crit2, mix, mix_unit_scale, img, txt, class_encs_b, targ_data_b, p):
    """Returns (loss, loss_raw, sims_ref) -- sims_ref is a per-criterion list of branch tuples
    ((sim,) non-bifurcated, (i2t, t2i) bifurcated) whose retained grads after the caller's backward
    are the ground truth for the chunked path's tile-accumulated (branch-summed) grad_sum_sims.
    Mirrors _loss_for_crit_full_batch's branch construction and _global_batch_loss's unit-scale
    blend (a bifurcated loss normalizes by L/2)."""
    clogits = _compute_logits_fn(p)
    sims_ref = []

    def crit_loss(crit, secondary):
        clamp = crit.cfg["logits"]["temp"]["clamp"]
        center = crit.cfg["logits"]["bce"]["center"]
        if crit.bifurcated:
            sims = (
                compute_sim(img, txt.detach(), crit.cfg["sim"]),
                compute_sim(img.detach(), txt, crit.cfg["sim"]),
            )
            crit_logits = tuple(clogits(s, clamp, center, secondary=secondary, half_live=True) for s in sims)
        else:
            sims = (compute_sim(img, txt, crit.cfg["sim"]),)
            crit_logits = clogits(sims[0], clamp, center, secondary=secondary)
        for s in sims:
            s.retain_grad()
        sims_ref.append(sims)
        loss, loss_raw, _ = crit(crit_logits, class_encs_b, targ_data_b, train=True, logit_scale=p["scale2"] if secondary else p["scale"])
        return loss, loss_raw

    loss1, loss1_raw = crit_loss(crit1, False)
    if mix == 0.0:
        return loss1, loss1_raw, sims_ref
    loss2, loss2_raw = crit_loss(crit2, True)
    if mix_unit_scale:
        loss1 = loss1 / (loss1.detach() / (2.0 if crit1.bifurcated else 1.0)).clamp_min(1e-12)
        loss2 = loss2 / (loss2.detach() / (2.0 if crit2.bifurcated else 1.0)).clamp_min(1e-12)
    return (1.0 - mix) * loss1 + mix * loss2, (1.0 - mix) * loss1_raw + mix * loss2_raw, sims_ref


CASES = [
    # (crit1, crit2, targ1, targ2, dsmr, focal, norm_ci, mix, unit_scale, center1, center2, neut)
    ("bce", None,  "mp",    None,  True,  2.0, False, 0.0, False, None, None, False),  # baseline
    ("bce", None,  "sp",    None,  True,  2.0, False, 0.0, False, None, None, False),
    ("bce", None,  "tax",   None,  True,  2.0, False, 0.0, False, None, None, False),
    ("bce", None,  "phylo", None,  True,  2.0, False, 0.0, False, None, None, False),
    ("bce", None,  "mp",    None,  False, 0.0, False, 0.0, False, None, None, False),  # no dsmr, no focal
    ("bce", None,  "mp",    None,  True,  2.0, True,  0.0, False, None, None, False),  # cls_imb.norm
    ("bce", "bce", "mp",    "mp",  True,  2.0, False, 0.3, False, None, None, False),  # mix, no unit scale
    ("bce", "bce", "mp",    "phylo", True, 2.0, False, 0.3, False, None, None, False),  # mixed target types
    ("bce", "bce", "mp",    "mp",  True,  2.0, False, 0.3, True,  None, None, False),  # mix + unit scale
    ("bce", "bce", "tax",   "mp",  True,  2.0, True,  0.3, True,  None, None, False),   # everything at once
    ("bce", "bce", "mp",    "mp",  False, 0.0, False, 0.5, True,  None, None, False),   # unit scale, no weighting
    ("bce", None,  "mp",    None,  True,  2.0, False, 0.0, False, "sim",        None, False),  # in-graph global sim mean
    ("bce", None,  "mp",    None,  True,  2.0, False, 0.0, False, "grad_proj",  None, False),  # constant grad projection
    ("bce", None,  "mp",    None,  True,  2.0, False, 0.0, False, "grad_proj2", None, False),
    ("bce", None,  "phylo", None,  True,  2.0, False, 0.0, False, "grad_proj",  None, False),  # soft targets + projection
    ("bce", "bce", "mp",    "phylo", True, 2.0, False, 0.3, False, "grad_proj2", "sim", False),  # mixed centers under mix
    ("bce", "bce", "mp",    "mp",  True,  2.0, False, 0.3, True,  "grad_proj",  "grad_proj", False),  # unit-scale coeff folding
    # bif_bce: two-branch tiles, 1D per-anchor weighting, half-live logit scalars
    ("bif_bce", None,      "mp",    None,  False, 0.0, False, 0.0, False, None, None, False),  # bif baseline
    ("bif_bce", None,      "sp",    None,  False, 2.0, False, 0.0, False, None, None, False),
    ("bif_bce", None,      "tax",   None,  True,  2.0, False, 0.0, False, None, None, False),  # row-wise dsmr on soft targets
    ("bif_bce", None,      "phylo", None,  True,  2.0, False, 0.0, False, None, None, True),   # + targ_mass_neut
    ("bif_bce", None,      "mp",    None,  True,  2.0, True,  0.0, False, None, None, True),   # cls_imb.norm + dsmr + neut
    ("bif_bce", None,      "mp",    None,  False, 2.0, False, 0.0, False, "sim",        None, False),  # per-branch in-graph mean
    ("bif_bce", None,      "mp",    None,  True,  2.0, False, 0.0, False, "grad_proj",  None, True),   # per-branch grad const
    ("bif_bce", None,      "mp",    None,  False, 2.0, False, 0.0, False, "grad_proj2", None, False),
    ("bif_bce", "bce",     "mp",    "mp",  True,  2.0, False, 0.3, False, None, None, False),  # bif+bce mix, no unit scale
    ("bif_bce", "bce",     "mp",    "mp",  True,  2.0, False, 0.3, True,  None, None, False),  # bif+bce mix, unit scale (L/2)
    ("bce",     "bif_bce", "mp",    "sp",  True,  2.0, False, 0.3, True,  None, None, True),   # bce+bif mix
    ("bif_bce", "bif_bce", "mp",    "phylo", True, 2.0, False, 0.5, True, "grad_proj", "sim", True),  # bif+bif, mixed centers
]


@pytest.mark.parametrize("C", [16, 48])  # 3 row-blocks, and single-block (== full)
@pytest.mark.parametrize("crit1_name,crit2_name,targ1,targ2,dsmr,focal,norm_ci,mix,unit_scale,center1,center2,neut", CASES)
def test_chunked_matches_full(crit1_name, crit2_name, targ1, targ2, dsmr, focal, norm_ci, mix, unit_scale, center1, center2, neut, C):
    device = torch.device("cpu")
    B, K, D, R = 48, 20, 16, 4

    cfg1 = _cfg(crit=crit1_name, targ=targ1, dsmr=dsmr, focal_gamma=focal,
                cls_imb_norm=norm_ci, center=center1, neut=neut)
    crit1 = _make_crit(cfg1, K, B)
    crit2 = None
    if mix != 0.0:
        cfg2 = _cfg(crit=crit2_name, targ=targ2, dsmr=dsmr, focal_gamma=focal,
                    cls_imb_norm=norm_ci, center=center2, neut=neut)
        crit2 = _make_crit(cfg2, K, B)

    g = torch.Generator().manual_seed(0)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    targ_data_b = _make_targ_data(B, K, R, class_encs_b)

    # full-batch reference
    img = img0.clone().requires_grad_(True)
    txt = txt0.clone().requires_grad_(True)
    p = _params(1)
    loss_ref, loss_raw_ref, sims_ref = _full_reference(crit1, crit2, mix, unit_scale, img, txt, class_encs_b, targ_data_b, p)
    loss_ref.backward()
    gsum_ref = [sum(s.grad.double().sum().item() for s in branches) for branches in sims_ref]

    # chunked
    imgc = img0.clone().requires_grad_(True)
    txtc = txt0.clone().requires_grad_(True)
    pc = _params(1)
    loss_c, loss_raw_c, _, gsum_c = L.chunked_bce_loss_backward(
        imgc, txtc, class_encs_b, targ_data_b, crit1, crit2, mix, unit_scale,
        _compute_logits_fn(pc), C, False, device, rank=0, world_size=1,
    )

    torch.testing.assert_close(loss_c, loss_ref.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(loss_raw_c, loss_raw_ref.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(imgc.grad, img.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(txtc.grad, txt.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(pc["scale"].grad, p["scale"].grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(pc["bias"].grad, p["bias"].grad, rtol=1e-4, atol=1e-6)
    assert gsum_c[0] == pytest.approx(gsum_ref[0], rel=1e-4, abs=1e-4)
    if mix != 0.0:
        torch.testing.assert_close(pc["scale2"].grad, p["scale2"].grad, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(pc["bias2"].grad, p["bias2"].grad, rtol=1e-4, atol=1e-6)
        assert gsum_c[1] == pytest.approx(gsum_ref[1], rel=1e-4, abs=1e-4)
    else:
        assert gsum_c[1] is None


def test_stats_min_max_mean_exact():
    B, C, K, D = 48, 16, 20, 16
    crit = _make_crit(_cfg(), K, B)
    g = torch.Generator().manual_seed(3)
    img = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    txt = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    targ_data_b = [None] * B

    sim = compute_sim(img.detach(), txt.detach(), "cos")
    targs = (class_encs_b.unsqueeze(1) == class_encs_b.unsqueeze(0)).float()
    _, _, stats, _ = L.chunked_bce_loss_backward(
        img, txt, class_encs_b, targ_data_b, crit, None, 0.0, False,
        lambda s, clamp, center=None, secondary=False, center_global=None, half_live=False: s * 10.0 - 0.5, C, False, torch.device("cpu"), rank=0, world_size=1,
    )
    assert stats["sim1_min"] == pytest.approx(sim.min().item(), abs=1e-5)
    assert stats["sim1_max"] == pytest.approx(sim.max().item(), abs=1e-5)
    assert stats["sim1_mean"] == pytest.approx(sim.mean().item(), abs=1e-5)
    assert stats["targ1_mean"] == pytest.approx(targs.mean().item(), abs=1e-5)
    assert "sim2_min" not in stats
    # the streamed probability histogram matches a full-batch one over the stub's logits
    # (sim * 10 - 0.5): counts just add across tiles, so it is exact, not subsampled
    p = (sim * 10.0 - 0.5).sigmoid()
    expected = torch.histc(p, bins=L.HIST_BINS, min=0.0, max=1.0) / p.numel()
    assert stats["p1_hist"] == pytest.approx(expected.tolist(), abs=1e-6)


def test_stats_split_by_crit():
    """Under a loss2 mix each branch reports stats over its own sim (per its sim_type) and its own
    (unblended) targets."""
    B, C, K, D = 48, 16, 20, 16
    crit1 = _make_crit(_cfg(targ="mp", sim="cos"), K, B)
    crit2 = _make_crit(_cfg(targ="sp", sim="geo1"), K, B)
    g = torch.Generator().manual_seed(3)
    img = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    txt = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    targ_data_b = [None] * B

    sim1 = compute_sim(img.detach(), txt.detach(), "cos")
    sim2 = compute_sim(img.detach(), txt.detach(), "geo1")
    targs1 = (class_encs_b.unsqueeze(1) == class_encs_b.unsqueeze(0)).float()
    targs2 = torch.eye(B)
    _, _, stats, _ = L.chunked_bce_loss_backward(
        img, txt, class_encs_b, targ_data_b, crit1, crit2, 0.3, False,
        _compute_logits_fn(_params(1)), C, False, torch.device("cpu"), rank=0, world_size=1,
    )
    for tag, sim, targs in (("1", sim1, targs1), ("2", sim2, targs2)):
        assert stats[f"sim{tag}_min"] == pytest.approx(sim.min().item(), abs=1e-5)
        assert stats[f"sim{tag}_max"] == pytest.approx(sim.max().item(), abs=1e-5)
        assert stats[f"sim{tag}_mean"] == pytest.approx(sim.mean().item(), abs=1e-5)
        assert stats[f"targ{tag}_min"] == pytest.approx(targs.min().item(), abs=1e-5)
        assert stats[f"targ{tag}_max"] == pytest.approx(targs.max().item(), abs=1e-5)
        assert stats[f"targ{tag}_mean"] == pytest.approx(targs.mean().item(), abs=1e-5)


def test_batch_diagnostics_off():
    """Disabling either diagnostics component must leave the loss and every gradient identical (the
    hooks/stats only observe): sim_grad_sums=False returns grad_sum_sims (None, None) and
    sim_targ_stats=False returns batch_stats None, each flag independent of the other."""
    B, C, K, D = 48, 16, 20, 16
    crit1 = _make_crit(_cfg(targ="mp"), K, B)
    crit2 = _make_crit(_cfg(crit="bif_bce", targ="mp"), K, B)
    g = torch.Generator().manual_seed(3)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)

    def run(sim_grad_sums, sim_targ_stats):
        img = img0.clone().requires_grad_(True)
        txt = txt0.clone().requires_grad_(True)
        p = _params(1)
        loss, loss_raw, stats, gsum = L.chunked_bce_loss_backward(
            img, txt, class_encs_b, [None] * B, crit1, crit2, 0.3, False,
            _compute_logits_fn(p), C, False, torch.device("cpu"), rank=0, world_size=1,
            sim_grad_sums=sim_grad_sums, sim_targ_stats=sim_targ_stats,
        )
        return loss, loss_raw, stats, gsum, img, txt, p

    loss_on, raw_on, stats_on, gsum_on, img_on, txt_on, p_on = run(True, True)
    assert stats_on is not None and gsum_on[0] is not None and gsum_on[1] is not None
    for sim_grad_sums, sim_targ_stats in ((False, False), (True, False), (False, True)):
        loss_off, raw_off, stats_off, gsum_off, img_off, txt_off, p_off = run(sim_grad_sums, sim_targ_stats)
        assert stats_off == (stats_on if sim_targ_stats else None)
        assert gsum_off == (gsum_on if sim_grad_sums else (None, None))
        torch.testing.assert_close(loss_off, loss_on, rtol=0, atol=0)
        torch.testing.assert_close(raw_off, raw_on, rtol=0, atol=0)
        torch.testing.assert_close(img_off.grad, img_on.grad, rtol=0, atol=0)
        torch.testing.assert_close(txt_off.grad, txt_on.grad, rtol=0, atol=0)
        for key in ("scale", "bias", "scale2", "bias2"):
            torch.testing.assert_close(p_off[key].grad, p_on[key].grad, rtol=0, atol=0)


@pytest.mark.parametrize("cfg_loss,cfg_loss2", [
    ({"crit": "infonce", "targ": "mp"}, {"mix": 0.0, "crit": "bce"}),          # infonce primary
    ({"crit": "bce", "targ": "mp"}, {"mix": 0.3, "crit": "infonce"}),           # infonce secondary (mixed)
])
def test_chunking_unsupported_with_infonce(cfg_loss, cfg_loss2):
    assert not L.chunking_supported(cfg_loss, cfg_loss2)


@pytest.mark.parametrize("cfg_loss,cfg_loss2", [
    ({"crit": "bce", "targ": "phylo"}, {"mix": 0.0, "crit": "bce"}),            # phylo now supported
    ({"crit": "bce", "targ": "mp"}, {"mix": 0.3, "crit": "bce"}),               # bce+bce mix supported
    ({"crit": "bce", "targ": "mp"}, {"mix": 0.0, "crit": "infonce"}),           # infonce loss2 inert at mix=0
    ({"crit": "bif_bce", "targ": "mp"}, {"mix": 0.0, "crit": "bce"}),           # bif_bce primary supported
    ({"crit": "bce", "targ": "mp"}, {"mix": 0.3, "crit": "bif_bce"}),           # bif_bce secondary supported
    ({"crit": "bif_bce", "targ": "mp"}, {"mix": 0.3, "crit": "bif_bce"}),       # bif+bif mix supported
])
def test_chunking_supported(cfg_loss, cfg_loss2):
    assert L.chunking_supported(cfg_loss, cfg_loss2)


def test_chunked_asserts_on_geo_sim_center():
    # center: sim needs the cos mean factorization for an exact global mean; TrainConfig rejects the
    # combo at config time, and the chunked entrypoint guards direct callers too
    B, K = 16, 5
    crit = _make_crit(_cfg(sim="geo1", center="sim"), K, B)
    g = torch.Generator().manual_seed(0)
    img = torch.nn.functional.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
    txt = torch.nn.functional.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    with pytest.raises(AssertionError, match="requires cos sim"):
        L.chunked_bce_loss_backward(img, txt, class_encs_b, [None] * B, crit, None, 0.0, False,
                                    _compute_logits_fn(_params(1)), 8, False, torch.device("cpu"),
                                    rank=0, world_size=1)


def _synthetic_vcv():
    vcv = RealPhyloVCV.__new__(RealPhyloVCV)  # bypass tree loading
    K = 8
    rng = np.random.default_rng(0)
    A = rng.random((K, K))
    targs = (A + A.T) / 2.0
    # Deliberately non-1.0 diagonal (the real target matrix has a unit diagonal -- exp(0)) so the
    # same-cid overwrite is observable rather than a no-op in this test.
    np.fill_diagonal(targs, rng.uniform(0.3, 0.9, size=K))
    targs[0, 0] = 1.0
    vcv.targs = targs
    vcv._cid_to_idx = {f"c{i}": i for i in range(K)}
    return vcv


def test_phylo_block_matches_full():
    """Real PhyloVCV.make_targ_block_fn reproduces the [rs:re, :] block of get_targs_batch (incl. the
    same-cid overwrite), on a synthetic target matrix with all-in-tree cids and repeats."""
    vcv = _synthetic_vcv()
    B = 20
    targ_data_b = [{"cid": f"c{i % 8}"} for i in range(B)]

    full = vcv.get_targs_batch(targ_data_b)
    blk_fn = vcv.make_targ_block_fn(targ_data_b, torch.device("cpu"))
    for rs in range(0, B, 6):
        re = min(rs + 6, B)
        torch.testing.assert_close(blk_fn(rs, re), full[rs:re])


def test_phylo_same_cid_pinned_to_one():
    """Same-cid pairs are forced to 1.0 even when targs' diagonal disagrees (the real matrix's
    diagonal is exp(0) = 1.0; the synthetic matrix keeps it below so the overwrite is observable)."""
    vcv = _synthetic_vcv()
    assert vcv.targs[1, 1] < 1.0
    targ_data_b = [{"cid": "c1"}, {"cid": "c1"}, {"cid": "c2"}]  # samples 0 and 1 share c1
    full = vcv.get_targs_batch(targ_data_b)
    assert full[0, 0] == 1.0 and full[0, 1] == 1.0 and full[1, 0] == 1.0  # same-cid -> 1.0
    assert full[0, 2] == pytest.approx(vcv.targs[1, 2])  # cross-species keeps the target value
