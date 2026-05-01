import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree, Clade  # type: ignore[import]
from itertools import combinations
from typing import Dict, List

from utils.utils import paths, load_pickle

import pdb


def get_tree(dataset: str) -> Tree:
    if dataset in ("bryo", "cub", "lepid", "nymph"):
        tree = load_pickle(paths["metadata"][dataset] / "tree.pkl")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return tree

class PhyloVCV:

    def __init__(self, dataset: str) -> None:

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
        self._warned_missing_cids: set[str] = set()

        vcv = self.build_vcv_matrix()

        self.corr = vcv / max(np.diag(vcv))

    def get_cids(self) -> list[str]:
        return self._cids

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
        Returns the phylogenetic correlation between two samples.
        Currently only used for verification purposes.
        """
        i = self._cid_to_idx[cid_a]
        j = self._cid_to_idx[cid_b]
        return float(np.clip(self.corr[i, j], 0.0, 1.0))

    def get_targs_batch(self, targ_data_b) -> torch.Tensor:
        cids_b = [td["cid"] for td in targ_data_b]
        B = len(cids_b)

        known_pos = [i for i, cid in enumerate(cids_b) if cid in self._cid_to_idx]
        missing_cids = {cid for cid in cids_b if cid not in self._cid_to_idx}

        missing_new = sorted(missing_cids - self._warned_missing_cids)
        if missing_new:
            self._warned_missing_cids.update(missing_new)
            print(
                "WARNING: "
                f"{len(missing_new)} class ids in batch missing from phylo tree; "
                "using fallback phylo targets for missing ids."
            )

        targs = np.zeros((B, B), dtype=np.float64)

        if known_pos:
            known_idxs = [self._cid_to_idx[cids_b[i]] for i in known_pos]
            targs_known = self.corr[np.ix_(known_idxs, known_idxs)]
            targs[np.ix_(known_pos, known_pos)] = targs_known

        # Fallback: samples with the same cid should remain fully positive.
        cids_arr = np.asarray(cids_b, dtype=object)
        same_cid_mask = cids_arr[:, None] == cids_arr[None, :]
        targs[same_cid_mask] = 1.0

        return torch.from_numpy(targs).float()