import numpy as np
import torch
from Bio.Phylo.BaseTree import Tree, Clade
from itertools import combinations
from typing import Dict, List

from utils.imb import _pair_prob_freqs
from utils.utils import paths, load_pickle, load_split

import pdb


def get_tree(dataset: str) -> Tree:
    if dataset in ("bryo", "cub", "lepid", "nymph"):
        tree = load_pickle(paths["metadata"][dataset] / "tree.pkl")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return tree

class PhyloVCV:
    """
    Phylo target matrix from the dataset's tree: Y = exp(-d / avg_dist), where
    d(a, b) = sqrt(patristic distance) -- the standard deviation of the Brownian-motion
    contrast X_a - X_b, a root-independent tree metric -- and avg_dist normalizes d to
    units of the average sampled pair's distance (_avg_dist). Built once at startup
    (VCV -> distances -> targets); batches index into it (get_targs_batch /
    make_targ_block_fn).
    """

    def __init__(self, dataset: str, split: str, train_pt: str, batch_size: int,
                 htarg_shuf: bool = False, seed: int | None = None) -> None:

        self.tree: Tree = get_tree(dataset)
        root: Clade = self.tree.root
        self._depth: Dict[Clade, float] = {root: 0.0}
        self._cid_to_clade: Dict[str, Clade] = {}

        # populate _depth and _cid_to_clade
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.clades:
                self._depth[child] = self._depth[node] + child.branch_length
                stack.append(child)
            if node.is_terminal():
                self._cid_to_clade[node.name] = node

        self._cids:       List[str]      = sorted(list(self._cid_to_clade.keys()))
        self._cid_to_idx: Dict[str, int] = {cid: i for i, cid in enumerate(self._cids)}

        vcv = self.build_vcv_matrix()

        # sqrt-patristic distance: d(a, b) = sqrt(depth(a) + depth(b) - 2 * depth(MRCA(a, b)))
        tip_depths = np.diag(vcv)
        dists = np.sqrt(tip_depths[:, None] + tip_depths[None, :] - 2.0 * vcv)

        avg_dist = self._avg_dist(dists, dataset, split, train_pt, batch_size)
        self.targs = np.exp(-dists / avg_dist)  # unit diagonal, (0, 1] range

        if htarg_shuf:
            # Scramble which species maps to which position in the (already-built) target
            # matrix. targs itself is untouched, so the full set of pairwise phylo distances is
            # preserved; only the cid -> matrix-index correspondence is randomized. The permutation
            # is derived from `seed`, so all DDP ranks agree and identically-seeded runs match.
            perm = np.random.default_rng(seed).permutation(len(self._cids))
            self._cid_to_idx = {cid: int(perm[i]) for i, cid in enumerate(self._cids)}

    def get_cids(self) -> list[str]:
        return self._cids

    def _avg_dist(self, dists: np.ndarray, dataset: str, split: str, train_pt: str, batch_size: int) -> float:
        """
        Pair-probability-frequency-weighted mean of `dists` over the train partition's classes
        (same-class d=0 pairs included), weighting each class pair by its batch co-occurrence
        probability (_pair_prob_freqs, the 2D BCE class-imbalance counting method). Normalizing
        by it makes the decay unit-free (calibrated to the average sampled pair's distance) and
        comparable across datasets. Computed once from global class counts, so it is a
        dataset-level constant identical across batches and DDP ranks.
        """
        split_obj = load_split(dataset, split)
        counts = torch.tensor(split_obj.class_counts[train_pt], dtype=torch.float64)
        encs = (~torch.isnan(counts)).nonzero(as_tuple=True)[0]  # classes present in the partition
        pair_freqs = _pair_prob_freqs(counts, encs, batch_size).numpy()
        idxs = np.array([self._cid_to_idx[split_obj.enc2cid[int(enc)]] for enc in encs])
        return float((pair_freqs * dists[np.ix_(idxs, idxs)]).sum() / pair_freqs.sum())

    def build_vcv_matrix(self) -> np.ndarray:

        n_cids = len(self._cids)
        vcv    = np.zeros((n_cids, n_cids), dtype=np.float64)

        clade_to_tip_idxs: Dict[Clade, np.ndarray] = {}
        stack = [(self.tree.root, False)]

        while stack:
            node, seen = stack.pop()
            if not seen:
                stack.append((node, True))
                stack.extend((child, False) for child in node.clades)
                continue

            if node.is_terminal():
                idx = self._cid_to_idx[node.name]
                vcv[idx, idx] = self._depth[node]  # variance along diagonal
                clade_to_tip_idxs[node] = np.array([idx], dtype=np.intp)
                continue

            idxs_child_tips = [clade_to_tip_idxs[child] for child in node.clades if child in clade_to_tip_idxs]

            depth = self._depth[node]
            for tips_u, tips_v in combinations(idxs_child_tips, 2):
                vcv[tips_u[:, None], tips_v] += depth
                vcv[tips_v[:, None], tips_u] += depth

            clade_to_tip_idxs[node] = np.concatenate(idxs_child_tips, dtype=np.intp) 

        return vcv
    
    def get_targ(self, cid_a: str, cid_b: str) -> float:
        """
        Returns the phylo target between two samples.
        Currently only used for verification purposes.
        """
        i = self._cid_to_idx[cid_a]
        j = self._cid_to_idx[cid_b]
        return float(self.targs[i, j])

    def get_targs_batch(self, targ_data_b) -> torch.Tensor:
        cids_b = [td["cid"] for td in targ_data_b]

        idxs = [self._cid_to_idx[cid] for cid in cids_b]
        targs = self.targs[np.ix_(idxs, idxs)]  # pt[B, B]; advanced indexing returns a writeable copy

        # same-cid pairs are fully positive (1.0). targs' diagonal is 1.0 (exp(0)) by construction;
        # same-species samples are pinned to it explicitly rather than relying on that.
        cids_arr = np.asarray(cids_b, dtype=object)
        targs[cids_arr[:, None] == cids_arr[None, :]] = 1.0

        return torch.from_numpy(targs).float()

    def make_targ_block_fn(self, targ_data_b, device):
        """
        Closure (rs, re) -> [re-rs, B] phylo target row-block (rows rs:re vs all B cols) matching the
        [rs:re, :] block of get_targs_batch, for the chunked/tiled loss. Target indices and the
        same-cid array are precomputed once; missing cids fail loud up front (as in get_targs_batch).
        """
        cids_b = [td["cid"] for td in targ_data_b]

        idxs = np.array([self._cid_to_idx[cid] for cid in cids_b])
        cids_arr = np.asarray(cids_b, dtype=object)

        def targ_block(rs, re):
            targs = self.targs[np.ix_(idxs[rs:re], idxs)]  # [C, B]; writeable copy
            targs[cids_arr[rs:re][:, None] == cids_arr[None, :]] = 1.0
            return torch.from_numpy(targs).float().to(device)

        return targ_block
