"""
python -m pytest tests/unit/test_phylo_targets.py

PhyloVCV target math on a tiny synthetic tree: sqrt-patristic distances, the
pair-probability-frequency-weighted avg_dist normalizer (absent classes excluded,
same-class d=0 pairs included), and the exp decay mapping.
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
# non-ultrametric tree: depths a=2, b=3, c=4; patristic (a,b)=3, (a,c)=6, (b,c)=7
NEWICK = "((a:1, b:2):1, c:4);"
# class c (enc 2) absent from the train partition -> excluded from avg_dist
COUNTS = [2.0, 1.0, float("nan")]
ENC2CID = {0: "a", 1: "b", 2: "c"}


class FakeSplit:
    def __init__(self):
        self.enc2cid = ENC2CID
        self.class_counts = {"train": np.array(COUNTS)}


@pytest.fixture()
def vcv(monkeypatch):
    tree = Phylo.read(StringIO(NEWICK), "newick")
    monkeypatch.setattr(phylo_mod, "get_tree", lambda dataset: tree)
    monkeypatch.setattr(phylo_mod, "load_split", lambda dataset, split: FakeSplit())
    return phylo_mod.PhyloVCV(dataset="cub", beta=BETA, split="D10", train_pt="train", batch_size=BATCH_SIZE)


def test_targets_match_formula(vcv):
    dists = np.sqrt(np.array([
        [0.0, 3.0, 6.0],
        [3.0, 0.0, 7.0],
        [6.0, 7.0, 0.0],
    ]))
    # avg_dist over the present classes (a, b) only, weighted by their pair probabilities
    counts = torch.tensor(COUNTS, dtype=torch.float64)
    pair_freqs = _pair_prob_freqs(counts, torch.tensor([0, 1]), BATCH_SIZE).numpy()
    avg_dist = (pair_freqs * dists[:2, :2]).sum() / pair_freqs.sum()
    expected = np.exp(-BETA * dists / avg_dist)

    idxs = [vcv._cid_to_idx[cid] for cid in ("a", "b", "c")]
    np.testing.assert_allclose(vcv.targs[np.ix_(idxs, idxs)], expected)


def test_targets_unit_diagonal(vcv):
    np.testing.assert_array_equal(np.diag(vcv.targs), 1.0)
