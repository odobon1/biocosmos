from functools import lru_cache
import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree, Clade  # type: ignore[import]
from itertools import combinations
from typing import Dict, List

from utils import paths

import pdb


class PhyloVCV:

    def __init__(self):

        self.tree: Tree = Phylo.read(paths["nymph_phylo_tree"], "newick")

        root:               Clade              = self.tree.root
        self._depth:        Dict[Clade, float] = {root: 0.0}
        self._sid_to_clade: Dict[str, Clade]   = {}

        # populate _depth and _sid_to_clade
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.clades:
                self._depth[child] = self._depth[node] + child.branch_length
                stack.append(child)
            if node.is_terminal():
                self._sid_to_clade[node.name] = node

        self._sids:       List[str]      = sorted(list(self._sid_to_clade.keys()))
        self._sid_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(self._sids)}

        vcv  = self.build_vcv_matrix()
        """
        Aassuming all tips are at the same depth, all elements along the diagonal (variances) are of the same value and 
        dividing by that value equates to the so called correlation matrix e.g. standardizing matrix values to range [0, 1]
        """
        self.corr = vcv / vcv[0, 0]

    def get_sids(self) -> list[str]:
        return self._sids

    def build_vcv_matrix(self) -> np.ndarray:

        n_sids = len(self._sids)
        vcv    = np.zeros((n_sids, n_sids), dtype=np.float64)

        clade_to_tip_idxs: Dict[Clade, np.ndarray] = {}
        stack = [(self.tree.root, False)]

        while stack:
            node, seen = stack.pop()
            if not seen:
                stack.append((node, True))
                stack.extend((child, False) for child in node.clades)
                continue

            if node.is_terminal():
                idx = self._sid_to_idx[node.name]
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
    
    def get_targ(self, sid_a: str, sid_b: str) -> float:
        i = self._sid_to_idx[sid_a]
        j = self._sid_to_idx[sid_b]
        return float(np.clip(self.corr[i, j], 0.0, 1.0))

    def get_targs_batch(self, sids_b) -> torch.Tensor:
        idxs_b = [self._sid_to_idx[s] for s in sids_b]
        targs  = self.corr[np.ix_(idxs_b, idxs_b)]
        return torch.from_numpy(targs).float()
