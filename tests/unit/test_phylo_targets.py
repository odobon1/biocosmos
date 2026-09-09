"""
python -m pytest tests/unit/test_phylo_targets.py

PhyloVCV target math on a tiny synthetic tree, per kernel: bm (the cosine-normalized
Brownian VCV; beta inert), and laplace / ou (exp decay in the kernel's tree metric --
sqrt-patristic / patristic -- normalized by the pair-probability-frequency-weighted
avg_dist: absent classes excluded, same-class d=0 pairs included; sharpened by beta).
"""

from io import StringIO

import numpy as np
import pytest
import torch
from Bio import Phylo

import utils.phylo as phylo_mod
from utils.imb import _pair_prob_freqs


BETA = 2.0
BATCH_SIZE = 4
# non-ultrametric tree: depths a=2, b=3, c=4; MRCA(a, b) at depth 1, (a, c) and (b, c) split at the root;
# patristic (a,b)=3, (a,c)=6, (b,c)=7
NEWICK = "((a:1, b:2):1, c:4);"
PATRISTIC = np.array([
    [0.0, 3.0, 6.0],
    [3.0, 0.0, 7.0],
    [6.0, 7.0, 0.0],
])
# the tree metric each exp-decay kernel decays in
KERNEL_DISTS = {"laplace": np.sqrt(PATRISTIC), "ou": PATRISTIC}
# bm: C_ab / sqrt(C_aa * C_bb) on the VCV C = [[2, 1, 0], [1, 3, 0], [0, 0, 4]]
BM_TARGS = np.array([
    [1.0, 1.0 / np.sqrt(6.0), 0.0],
    [1.0 / np.sqrt(6.0), 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
# class c (enc 2) absent from the train partition -> excluded from avg_dist
COUNTS = [2.0, 1.0, float("nan")]
ENC2CID = {0: "a", 1: "b", 2: "c"}


class FakeSplit:
    def __init__(self):
        self.enc2cid = ENC2CID
        self.class_counts = {"train": np.array(COUNTS)}


@pytest.fixture(params=["bm", "laplace", "ou"])
def kernel(request):
    return request.param


@pytest.fixture()
def vcv(monkeypatch, kernel):
    tree = Phylo.read(StringIO(NEWICK), "newick")
    monkeypatch.setattr(phylo_mod, "get_tree", lambda dataset: tree)
    monkeypatch.setattr(phylo_mod, "load_split", lambda dataset, split: FakeSplit())
    return phylo_mod.PhyloVCV(dataset="cub", split="D10", train_pt="train", batch_size=BATCH_SIZE, kernel=kernel, beta=BETA)


def test_targets_match_formula(vcv, kernel):
    if kernel == "bm":
        expected = BM_TARGS
    else:
        dists = KERNEL_DISTS[kernel]
        # avg_dist over the present classes (a, b) only, weighted by their pair probabilities over the
        # full (ordered-pair) matrix: (a, b) and (b, a) are separate cells, as in the BxB batch matrix
        counts = torch.tensor(COUNTS, dtype=torch.float64)
        pair_freqs = _pair_prob_freqs(counts, torch.tensor([0, 1]), BATCH_SIZE).numpy()
        avg_dist = (pair_freqs * dists[:2, :2]).sum() / pair_freqs.sum()
        expected = np.exp(-BETA * dists / avg_dist)

    idxs = [vcv._cid_to_idx[cid] for cid in ("a", "b", "c")]
    np.testing.assert_allclose(vcv.targs[np.ix_(idxs, idxs)], expected)


def test_targets_unit_diagonal(vcv):
    np.testing.assert_array_equal(np.diag(vcv.targs), 1.0)
