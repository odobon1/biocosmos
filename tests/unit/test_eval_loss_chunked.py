"""
Unit tests for VLMWrapper.eval_loss_chunked, the eval loss pass: the fixed-size chunking, each full chunk's stats
(sim_targ_batch_stats' entries + grad_sum_sim, the chunk's sum(dL/dsim)) and that the pass leaves the model's logit
scalars' .grad alone. The REAL methods are bound to a lightweight harness `self` (as test_bif_bce does): single
process, world_size 1, a toy model carrying just the logit scalars.
"""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.loss as L
from models import VLMWrapper

HIST_BINS = 20  # the harness' reporting.learning_curves.hist_bins
KAPPAS = [0.0, 3.0]  # ... and hpsm.kappas


def _cfg(crit):
    """The loss-level config (train.yaml's `loss` block, as the criterion reads it)."""
    return {
        "crit": crit, "sim": "cos", "blend": {"lambda": 0.0, "type": "targ"}, "unitless": False,
        "xnce": {"block_res": None},
        "bce": {"targ_mass_neut": False},
        "wting": {
            "cls_imb": {"type": None, "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999}, "norm": False},
            "bce": {"dsmr": False},
        },
        "logits": {"shared": True, "scale": {"clamp": False}, "bce": {"center": None, "bias": {}}},
    }


def _targ(targ):
    """A target spec (train.yaml's loss1 / loss2 block); the tsm is read by InfoNCE only."""
    return {"targ": targ, "xnce": {"tsm": {"type": "linear", "sm_scale": 1.0}}}


def _make_crit(cfg, K, B):
    cls = {"bce": L.BCECriterion, "xnce": L.InfoNCECriterion}[cfg["crit"]]
    crit = cls.__new__(cls)  # bypass build_wting (no dataset needed)
    crit.cfg = cfg
    crit.targ_specs = L.targ_specs(cfg["blend"]["lambda"], _targ("mp"), _targ("sp"))
    crit.device = torch.device("cpu")
    crit.batch_size = B
    crit.counts = torch.randint(1, 1000, (K,), generator=torch.Generator().manual_seed(K)).to(torch.float64)
    crit.wt_mean = 1.0
    return crit


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.logit_bias = nn.Parameter(torch.tensor(-0.5))


class Harness:
    """Fake VLMWrapper `self` carrying the real methods verbatim."""
    _unwrapped_model = VLMWrapper._unwrapped_model
    compute_logits = VLMWrapper.compute_logits
    _loss_full_batch = VLMWrapper._loss_full_batch
    eval_loss_chunked = VLMWrapper.eval_loss_chunked


def _make_harness(crit):
    h = Harness()
    h.model = Toy().eval()  # the eval pass runs with the model in eval mode; the chunk loss never reads the mode
    h.crit = crit
    h.world_size = 1
    h.cfg = SimpleNamespace(loss=crit.cfg, reporting={"learning_curves": {"hpsm": {"kappas": KAPPAS}, "hist_bins": HIST_BINS}})
    return h


def _data(N, K, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = F.normalize(torch.randn(N, D, generator=g), dim=1)
    txt = F.normalize(torch.randn(N, D, generator=g), dim=1)
    class_encs = torch.randint(0, K, (N,), generator=g)
    return img, txt, class_encs


@pytest.mark.parametrize("crit_name", ["bce", "xnce"])
def test_eval_loss_chunked_reports_each_full_chunks_stats_and_sim_grad_sum(crit_name) -> None:
    # N = 10 over chunks of 4: two full chunks (the trailing 2 dropped), each carrying sim_targ_batch_stats' entries --
    # p_hist for the BCE family alone -- plus grad_sum_sim, sum(dL/dsim) of the training objective on the chunk (the
    # raw BCE loss is detached, so only the train=True objective carries a graph). The pass is called under no_grad
    # (collect_eval_bundle / chunked_loss) and reads the sims' gradient alone, so the model's logit scalars end with
    # no .grad
    N, B, K, D = 10, 4, 5, 8
    img, txt, class_encs = _data(N, K, D)
    h = _make_harness(_make_crit(_cfg(crit_name), K, B))

    with torch.no_grad():
        loss_avg, chunk_stats = h.eval_loss_chunked(img, txt, class_encs, [None] * N, B)

    assert len(chunk_stats) == 2 and loss_avg > 0.0
    for stats in chunk_stats:
        assert len(stats["sim_hist"]) == len(stats["targ_hist"]) == HIST_BINS
        assert len(stats["sim_margin"]) == len(stats["sim_margin_i2t"]) == len(KAPPAS)
        assert ("p_hist" in stats) == (crit_name == "bce")
        assert math.isfinite(stats["grad_sum_sim"])
    assert h.model.logit_scale.grad is None and h.model.logit_bias.grad is None

    # the sim-grad sum is the train strip's measurement (TrainPipeline._step_train: the retained sims' .grad of the
    # training objective after the backward) taken on the same chunk -- the first chunk of the fixed-seed permutation
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:B]
    img_c, txt_c = img[perm].clone().requires_grad_(True), txt[perm].clone().requires_grad_(True)
    loss, _, _, sims, _, _ = h._loss_full_batch(img_c, txt_c, class_encs[perm], [None] * B, True)
    for sim in sims:
        sim.retain_grad()
    loss.backward()
    expected = sum(sim.grad.float().sum().item() for sim in sims)
    assert chunk_stats[0]["grad_sum_sim"] == pytest.approx(expected, abs=1e-5)
    if crit_name == "xnce":  # a row softmax is shift-invariant: dL/dsim sums to zero over the chunk
        assert abs(expected) < 1e-5


def test_eval_loss_chunked_needs_one_full_chunk() -> None:
    N, B, K, D = 3, 4, 5, 8
    img, txt, class_encs = _data(N, K, D)
    h = _make_harness(_make_crit(_cfg("bce"), K, B))
    assert h.eval_loss_chunked(img, txt, class_encs, [None] * N, B) == (None, [])
