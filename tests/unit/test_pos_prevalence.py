"""
pos_prevalence: the expected weighted positive prevalence of a BCE-family loss's B x B target matrix, the
p behind logits.bce.bias.init: pos_prevalence (bias = log(p / (1 - p))). Closed forms (sp: 1/B; mp: the
same-class pair probability; inv_freq gamma 1 on pairs: 1/K; DSMR on binary targets: 1/2) plus Monte Carlo
agreement with uniform without-replacement batch sampling.
"""
import random

import numpy as np
import pytest
import torch

import utils.loss as loss_mod


BATCH_SIZE = 8
COUNTS = [40.0, 12.0, 5.0, float("nan"), 3.0]  # enc 3 absent from the train partition
ENC2CID = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
RANK_ENCS = {"a": [1, 10], "b": [1, 20], "c": [2, 30], "e": [2, 30]}  # (a, b) share the top rank; (c, e) the full path
PHYLO_TARGS = {  # symmetric, unit diagonal
    ("a", "b"): 0.8, ("a", "c"): 0.3, ("a", "e"): 0.1,
    ("b", "c"): 0.5, ("b", "e"): 0.2,
    ("c", "e"): 0.6,
}


class FakeSplit:
    def __init__(self):
        self.enc2cid = ENC2CID
        self.class_counts = {"train": np.array(COUNTS)}


class FakePhyloVCV:
    def get_targs_batch(self, targ_data_b):
        cids = [td["cid"] for td in targ_data_b]
        targs = torch.tensor([[1.0 if a == b else PHYLO_TARGS[tuple(sorted((a, b)))] for b in cids] for a in cids])
        return targs


def make_cfg_loss(crit="bce", targ="mp", wting_type=None, gamma=1.0, dsmr=False, targ_mass_neut=False):
    return {
        "crit": crit,
        "targ": targ,
        "bce": {"targ_mass_neut": targ_mass_neut},
        "wting": {
            "cls_imb": {"type": wting_type, "inv_freq": {"gamma": gamma}, "class_bal": {"beta": 0.9999}, "norm": False},
            "bce": {"dsmr": dsmr},
        },
    }


@pytest.fixture(autouse=True)
def patch_metadata(monkeypatch):
    monkeypatch.setattr(loss_mod, "load_split", lambda dataset, split: FakeSplit())
    monkeypatch.setattr(loss_mod, "compute_rank_encs", lambda dataset, cids: [RANK_ENCS[cid] for cid in cids])
    monkeypatch.setattr(loss_mod, "get_phylo_vcv", lambda dataset: FakePhyloVCV())


def prevalence(cfg_loss):
    return loss_mod.pos_prevalence(cfg_loss, "cub", "D10", "train", BATCH_SIZE)


def present_counts():
    return [n for n in COUNTS if not np.isnan(n)]


def homog_pair_prob_sum():
    # sum over classes of the same-class pair probability (anchor's own slot included), i.e. B * P(same class)
    n = np.array(present_counts())
    N = n.sum()
    return float(((n / N) * (1 + (BATCH_SIZE - 1) * (n - 1) / (N - 1))).sum())


def monte_carlo_prevalence(targ, n_batches=20000, seed=0):
    # mean target over uniformly random B x B entries, batches drawn uniformly without replacement
    rng = random.Random(seed)
    n = present_counts()
    encs = [enc for enc, c in enumerate(COUNTS) if not np.isnan(c)]
    population = [enc for enc, c in zip(encs, n) for _ in range(int(c))]
    total = 0.0
    for _ in range(n_batches):
        batch = rng.sample(population, BATCH_SIZE)
        targ_data_b = [{"cid": ENC2CID[enc], "dataset": "cub", "rank_encs": RANK_ENCS[ENC2CID[enc]]} for enc in batch]
        Y = loss_mod.compute_targets(targ, BATCH_SIZE, torch.tensor(batch), targ_data_b, "cpu")
        total += Y.mean().item()
    return total / n_batches


@pytest.mark.parametrize("crit", ["bce", "bif_bce"])
def test_sp_is_one_over_batch_size(crit):
    assert prevalence(make_cfg_loss(crit=crit, targ="sp")) == pytest.approx(1 / BATCH_SIZE)


@pytest.mark.parametrize("crit", ["bce", "bif_bce"])
def test_mp_is_same_class_pair_probability(crit):
    assert prevalence(make_cfg_loss(crit=crit, targ="mp")) == pytest.approx(homog_pair_prob_sum() / BATCH_SIZE)


def test_mp_reduces_to_sp_with_singleton_classes(monkeypatch):
    class SingletonSplit(FakeSplit):
        def __init__(self):
            super().__init__()
            self.class_counts = {"train": np.array([1.0, 1.0, 1.0, float("nan"), 1.0])}

    monkeypatch.setattr(loss_mod, "load_split", lambda dataset, split: SingletonSplit())
    assert prevalence(make_cfg_loss(targ="mp")) == pytest.approx(1 / BATCH_SIZE)


@pytest.mark.parametrize("targ", ["mp", "tax", "phylo"])
def test_matches_monte_carlo_batch_sampling(targ):
    assert prevalence(make_cfg_loss(targ=targ)) == pytest.approx(monte_carlo_prevalence(targ), abs=5e-3)


def test_soft_targets_exceed_binary_and_stay_in_range():
    p_mp = prevalence(make_cfg_loss(targ="mp"))
    p_tax = prevalence(make_cfg_loss(targ="tax"))
    p_phylo = prevalence(make_cfg_loss(targ="phylo"))
    assert p_mp < p_tax < 1.0
    assert p_mp < p_phylo < 1.0


def test_inv_freq_gamma_one_pair_weighting_equalizes_class_pairs():
    # W = 1 / P_full on every class pair -> every (present) class pair counts equally -> 1 / K under mp
    K = len(present_counts())
    assert prevalence(make_cfg_loss(crit="bce", targ="mp", wting_type="inv_freq", gamma=1.0)) == pytest.approx(1 / K)


def test_inv_freq_gamma_one_anchor_weighting_equalizes_anchor_classes():
    # bif_bce: row weight 1 / n_r -> every anchor class counts equally; row prevalence is its expected same-class
    # slot fraction (1 + (B - 1)(n_r - 1) / (N - 1)) / B
    n = np.array(present_counts())
    N = n.sum()
    expected = float(np.mean((1 + (BATCH_SIZE - 1) * (n - 1) / (N - 1)) / BATCH_SIZE))
    assert prevalence(make_cfg_loss(crit="bif_bce", targ="mp", wting_type="inv_freq", gamma=1.0)) == pytest.approx(expected)


@pytest.mark.parametrize("crit", ["bce", "bif_bce"])
@pytest.mark.parametrize("targ", ["sp", "mp"])
def test_dsmr_balances_binary_targets_to_half(crit, targ):
    assert prevalence(make_cfg_loss(crit=crit, targ=targ, dsmr=True)) == pytest.approx(0.5)


def test_dsmr_on_soft_targets_is_not_half():
    # DSMR equalizes the Y-weighted and (1 - Y)-weighted masses, which is a 1/2 prevalence only for binary Y
    assert prevalence(make_cfg_loss(crit="bce", targ="phylo", dsmr=True)) != pytest.approx(0.5)


def test_targ_mass_neut_weights_rows_by_inverse_expected_mass():
    # bif_bce, mp: row weight 1 / m_r with m_r the anchor class's expected row target mass -> every anchor row
    # contributes unit positive mass, so p = 1 / (B * E_anchor[1 / m_r])
    n = np.array(present_counts())
    N = n.sum()
    m = 1 + (BATCH_SIZE - 1) * (n - 1) / (N - 1)
    expected = 1 / (BATCH_SIZE * float(((n / N) / m).sum()))
    assert prevalence(make_cfg_loss(crit="bif_bce", targ="mp", targ_mass_neut=True)) == pytest.approx(expected)


def test_targ_mass_neut_is_inert_under_sp():
    assert prevalence(make_cfg_loss(crit="bif_bce", targ="sp", targ_mass_neut=True)) == pytest.approx(1 / BATCH_SIZE)
