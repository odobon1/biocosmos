"""
Equivalence tests for the tiled/chunked global-batch BCE loss (hardware.loss_chunk_size).

chunked_bce_loss_backward must reproduce the loss and gradients (wrt image/text embeddings and the
primary/secondary logit scale/bias) of the full-batch path (BCECriterion.__call__ blended by
_global_batch_loss), up to floating-point summation order -- across the full BCE config space:
sw/iw/tax/phylo targets, norm.cls_imb, norm.agg, a BCE+BCE loss2 mix, and mix_unit_scale.
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
        def __init__(self, dataset: str, htarg_shuf: bool = False, seed: int | None = None) -> None:
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


def _cfg(targ="sw", dsmr=True, focal_gamma=2.0, agg="prod", freq_type="naive", sim="cos",
         norm_cls_imb=False, norm_agg=False):
    return {
        "type": "bce", "sim": sim, "targ": targ,
        "wting": {
            "cls_imb": {"type": "inv_freq", "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999},
                        "freq_type_2d": freq_type, "wt_mean_type": "per_class"},
            "focal": {"gamma": focal_gamma, "comp_type": 1},
            "dsmr": dsmr, "agg": agg,
            "norm": {"cls_imb": norm_cls_imb, "agg": norm_agg},
        },
        "logits": {"scale": {"clamp": False}, "bias": {}},
    }


def _make_crit(cfg, K, B):
    crit = L.BCECriterion.__new__(L.BCECriterion)  # bypass build_wting (no dataset needed)
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


def _compute_logits_fn(p):
    def compute_logits(sim, clamp, secondary=False):
        s = p["scale2"] if secondary else p["scale"]
        b = p["bias2"] if secondary else p["bias"]
        if clamp:
            s = s.clamp(max=math.log(100))
        return sim * s.exp() + b
    return compute_logits


def _make_targ_data(B, K, R, class_encs_b):
    """targ_data carrying every field any targ_type needs (rank_encs for tax, cid/dataset for phylo)."""
    g = torch.Generator().manual_seed(99)
    rank_encs = torch.randint(0, 3, (B, R), generator=g).tolist()
    return [{"rank_encs": rank_encs[i], "cid": f"c{int(class_encs_b[i])}", "dataset": "cub"} for i in range(B)]


def _full_reference(crit1, crit2, mix, mix_unit_scale, img, txt, class_encs_b, targ_data_b, p):
    clogits = _compute_logits_fn(p)

    def crit_loss(crit, secondary):
        sim = compute_sim(img, txt, crit.cfg["sim"])
        logits = clogits(sim, crit.cfg["logits"]["scale"]["clamp"], secondary=secondary)
        loss, loss_raw, _ = crit(logits, class_encs_b, targ_data_b, train=True)
        return loss, loss_raw

    loss1, loss1_raw = crit_loss(crit1, False)
    if mix == 0.0:
        return loss1, loss1_raw
    loss2, loss2_raw = crit_loss(crit2, True)
    if mix_unit_scale:
        loss1 = loss1 / loss1.detach().clamp_min(1e-12)
        loss2 = loss2 / loss2.detach().clamp_min(1e-12)
    return (1.0 - mix) * loss1 + mix * loss2, (1.0 - mix) * loss1_raw + mix * loss2_raw


CASES = [
    # (targ1, targ2, dsmr, focal, agg, freq, norm_ci, norm_agg, mix, unit_scale)
    ("sw",    None,  True,  2.0, "prod",      "naive",     False, False, 0.0, False),  # baseline
    ("iw",    None,  True,  2.0, "prod",      "naive",     False, False, 0.0, False),
    ("tax",   None,  True,  2.0, "prod",      "naive",     False, False, 0.0, False),
    ("phylo", None,  True,  2.0, "prod",      "naive",     False, False, 0.0, False),
    ("sw",    None,  False, 0.0, "prod",      "naive",     False, False, 0.0, False),  # no dsmr, no focal
    ("sw",    None,  True,  2.0, "mean",      "naive",     False, False, 0.0, False),
    ("sw",    None,  True,  2.0, "geo_mean",  "cmx2",      False, False, 0.0, False),
    ("sw",    None,  True,  2.0, "harm_mean", "pair_prob", False, False, 0.0, False),
    ("sw",    None,  True,  2.0, "prod",      "naive",     True,  False, 0.0, False),  # norm.cls_imb
    ("sw",    None,  True,  2.0, "prod",      "naive",     False, True,  0.0, False),  # norm.agg
    ("sw",    None,  True,  2.0, "geo_mean",  "cmx2",      True,  True,  0.0, False),  # both norms
    ("tax",   None,  True,  2.0, "prod",      "naive",     False, True,  0.0, False),  # norm.agg + soft targets
    ("sw",    "sw",  True,  2.0, "prod",      "naive",     False, False, 0.3, False),  # mix, no unit scale
    ("sw",    "phylo", True, 2.0, "prod",     "naive",     False, False, 0.3, False),  # mixed target types
    ("sw",    "sw",  True,  2.0, "prod",      "naive",     False, False, 0.3, True),   # mix + unit scale
    ("tax",   "sw",  True,  2.0, "mean",      "cmx2",      True,  True,  0.3, True),    # everything at once
    ("sw",    "sw",  False, 0.0, "prod",      "naive",     False, False, 0.5, True),    # unit scale, no weighting
]


@pytest.mark.parametrize("C", [16, 48])  # 3 row-blocks, and single-block (== full)
@pytest.mark.parametrize("targ1,targ2,dsmr,focal,agg,freq,norm_ci,norm_agg,mix,unit_scale", CASES)
def test_chunked_matches_full(targ1, targ2, dsmr, focal, agg, freq, norm_ci, norm_agg, mix, unit_scale, C):
    device = torch.device("cpu")
    B, K, D, R = 48, 20, 16, 4

    cfg1 = _cfg(targ=targ1, dsmr=dsmr, focal_gamma=focal, agg=agg, freq_type=freq,
                norm_cls_imb=norm_ci, norm_agg=norm_agg)
    crit1 = _make_crit(cfg1, K, B)
    crit2 = None
    if mix != 0.0:
        cfg2 = _cfg(targ=targ2, dsmr=dsmr, focal_gamma=focal, agg=agg, freq_type=freq,
                    norm_cls_imb=norm_ci, norm_agg=norm_agg)
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
    loss_ref, loss_raw_ref = _full_reference(crit1, crit2, mix, unit_scale, img, txt, class_encs_b, targ_data_b, p)
    loss_ref.backward()

    # chunked
    imgc = img0.clone().requires_grad_(True)
    txtc = txt0.clone().requires_grad_(True)
    pc = _params(1)
    loss_c, loss_raw_c, _ = L.chunked_bce_loss_backward(
        imgc, txtc, class_encs_b, targ_data_b, crit1, crit2, mix, unit_scale,
        _compute_logits_fn(pc), C, False, device,
    )

    torch.testing.assert_close(loss_c, loss_ref.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(loss_raw_c, loss_raw_ref.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(imgc.grad, img.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(txtc.grad, txt.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(pc["scale"].grad, p["scale"].grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(pc["bias"].grad, p["bias"].grad, rtol=1e-4, atol=1e-6)
    if mix != 0.0:
        torch.testing.assert_close(pc["scale2"].grad, p["scale2"].grad, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(pc["bias2"].grad, p["bias2"].grad, rtol=1e-4, atol=1e-6)


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
    _, _, stats = L.chunked_bce_loss_backward(
        img, txt, class_encs_b, targ_data_b, crit, None, 0.0, False,
        lambda s, clamp, secondary=False: s * 10.0 - 0.5, C, False, torch.device("cpu"),
    )
    assert stats["sim_min"] == pytest.approx(sim.min().item(), abs=1e-5)
    assert stats["sim_max"] == pytest.approx(sim.max().item(), abs=1e-5)
    assert stats["sim_mean"] == pytest.approx(sim.mean().item(), abs=1e-5)
    assert stats["targ_mean"] == pytest.approx((2 * targs.mean() - 1).item(), abs=1e-5)


@pytest.mark.parametrize("cfg_loss,cfg_loss2", [
    ({"type": "infonce2", "targ": "sw"}, {"mix": 0.0, "type": "bce"}),         # infonce primary
    ({"type": "infonce1", "targ": "sw"}, {"mix": 0.0, "type": "bce"}),
    ({"type": "bce", "targ": "sw"}, {"mix": 0.3, "type": "infonce2"}),          # infonce secondary (mixed)
])
def test_validate_chunking_rejects_infonce(cfg_loss, cfg_loss2):
    with pytest.raises(NotImplementedError):
        L.validate_chunking_supported(cfg_loss, cfg_loss2)


@pytest.mark.parametrize("cfg_loss,cfg_loss2", [
    ({"type": "bce", "targ": "phylo"}, {"mix": 0.0, "type": "bce"}),            # phylo now supported
    ({"type": "bce", "targ": "sw"}, {"mix": 0.3, "type": "bce"}),               # bce+bce mix supported
    ({"type": "bce", "targ": "sw"}, {"mix": 0.0, "type": "infonce2"}),          # infonce loss2 inert at mix=0
])
def test_validate_chunking_accepts(cfg_loss, cfg_loss2):
    L.validate_chunking_supported(cfg_loss, cfg_loss2)  # no raise


def _synthetic_vcv():
    vcv = RealPhyloVCV.__new__(RealPhyloVCV)  # bypass tree loading
    K = 8
    rng = np.random.default_rng(0)
    A = rng.random((K, K))
    corr = (A + A.T) / 2.0
    # NON-ultrametric regime (cub/bryo): corr = vcv / max(diag) leaves shallower tips with a diagonal
    # < 1.0, so the same-cid overwrite is load-bearing (only the lepid merge tree is ultrametric).
    np.fill_diagonal(corr, rng.uniform(0.3, 0.9, size=K))
    corr[0, 0] = 1.0  # one tip at max depth
    vcv.corr = corr
    vcv._cid_to_idx = {f"c{i}": i for i in range(K)}
    return vcv


def test_phylo_block_matches_full():
    """Real PhyloVCV.make_targ_block_fn reproduces the [rs:re, :] block of get_targs_batch (incl. the
    same-cid overwrite), on a synthetic correlation matrix with all-in-tree cids and repeats."""
    vcv = _synthetic_vcv()
    B = 20
    targ_data_b = [{"cid": f"c{i % 8}"} for i in range(B)]

    full = vcv.get_targs_batch(targ_data_b)
    blk_fn = vcv.make_targ_block_fn(targ_data_b, torch.device("cpu"))
    for rs in range(0, B, 6):
        re = min(rs + 6, B)
        torch.testing.assert_close(blk_fn(rs, re), full[rs:re])


def test_phylo_same_cid_pinned_to_one():
    """Same-cid pairs are forced to 1.0 even when corr's own diagonal is < 1.0 (non-ultrametric tree),
    i.e. the same-cid overwrite is not a no-op for cub/bryo."""
    vcv = _synthetic_vcv()
    assert vcv.corr[1, 1] < 1.0  # c1 sits below max depth
    targ_data_b = [{"cid": "c1"}, {"cid": "c1"}, {"cid": "c2"}]  # samples 0 and 1 share c1
    full = vcv.get_targs_batch(targ_data_b)
    assert full[0, 0] == 1.0 and full[0, 1] == 1.0 and full[1, 0] == 1.0  # same-cid -> 1.0
    assert full[0, 2] == pytest.approx(vcv.corr[1, 2])  # cross-species keeps the corr value
