"""
hard_pair_similarity_margin: the per-row hardness-weighted hard-pair similarity margin. At kappa 0 it
is the continuous-Q mean-separation margin (q-weighted mean sim of the positives minus the
(1-q)-weighted mean of the negatives), generalizing over sp / mp / continuous targets; a growing
kappa shifts each side's weight onto its hard pairs.
"""

import pytest
import torch

from tests.unit.test_loss_targets import import_loss_module


def _sim(B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, B, generator=g) * 2.0 - 1.0


def test_sp_targets_read_diagonal_minus_offdiagonal_mean():
    L = import_loss_module()
    B = 12
    S = _sim(B)
    rows = L.hard_pair_similarity_margin(S, torch.eye(B), 0.0)
    off_mean = (S.sum(1) - S.diag()) / (B - 1)
    torch.testing.assert_close(rows, S.diag() - off_mean)


def test_mp_targets_read_positive_mean_minus_negative_mean():
    L = import_loss_module()
    B = 12
    S = _sim(B, seed=1)
    class_encs = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 0])
    Q = (class_encs[:, None] == class_encs[None, :]).float()
    rows = L.hard_pair_similarity_margin(S, Q, 0.0)
    pos = (S * Q).sum(1) / Q.sum(1)
    neg = (S * (1 - Q)).sum(1) / (1 - Q).sum(1)
    torch.testing.assert_close(rows, pos - neg)


def test_continuous_targets_read_membership_weighted_means():
    L = import_loss_module()
    B = 10
    S = _sim(B, seed=2)
    g = torch.Generator().manual_seed(3)
    Q = torch.rand(B, B, generator=g)
    Q.fill_diagonal_(1.0)
    rows = L.hard_pair_similarity_margin(S, Q, 0.0)
    pos = (S * Q).sum(1) / Q.sum(1)
    neg = (S * (1 - Q)).sum(1) / (1 - Q).sum(1)
    torch.testing.assert_close(rows, pos - neg)


def test_kappa_concentrates_on_hard_pairs():
    L = import_loss_module()
    B = 12
    S = _sim(B, seed=4)
    class_encs = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    Q = (class_encs[:, None] == class_encs[None, :]).float()
    hard = L.hard_pair_similarity_margin(S, Q, 1e3)
    # the positive side collapses onto the least-similar positive, the negative side onto the
    # most-similar negative -- the hardest pairs on each side
    pos_hard = S.masked_fill(Q == 0, float("inf")).min(1).values
    neg_hard = S.masked_fill(Q == 1, float("-inf")).max(1).values
    torch.testing.assert_close(hard, pos_hard - neg_hard)
    # and the hard margin never exceeds the mean-separation one
    assert torch.all(hard <= L.hard_pair_similarity_margin(S, Q, 0.0) + 1e-6)


def test_row_block_matches_full_matrix():
    # tiles hold whole rows, so a row-block's margins are those rows' full-matrix margins
    L = import_loss_module()
    B, C = 16, 4
    S = _sim(B, seed=5)
    Q = torch.eye(B)
    for kappa in (0.0, 2.0):
        full = L.hard_pair_similarity_margin(S, Q, kappa)
        for rs in range(0, B, C):
            torch.testing.assert_close(L.hard_pair_similarity_margin(S[rs:rs + C], Q[rs:rs + C], kappa), full[rs:rs + C])
