"""
python -m tools.readable_tree
"""

from typing import List
import itertools
from Bio import Phylo  # type: ignore[import]

from utils.utils import paths


class Namer:
    """Gives stable, readable labels for unlabeled internal nodes."""
    def __init__(self):
        self._counter = itertools.count(1)

    def label(self, clade) -> str:
        if clade.name and str(clade.name).strip():
            return str(clade.name).strip()
        if not hasattr(clade, "_auto_id"):
            clade._auto_id = next(self._counter)
        return f"internal_{clade._auto_id}"

def count_leaves(clade) -> int:
    return sum(1 for _ in clade.get_terminals())

def ascii_tree_lines(
    clade,
    namer: Namer,
    prefix: str = "",
    is_last: bool = True,
    depth: int = 0,
    cum_dist: float = 0.0,
) -> List[str]:
    """Return list of pretty-printed lines for this clade and its descendants."""
    tee = "├─ " 
    last = "└─ "
    bar = "│  "
    spc = "   "

    label = namer.label(clade)
    leaves = count_leaves(clade)

    if depth == 0:
        line = [f"{label}  [root, leaves={leaves}]"]
    else:
        conn = last if is_last else tee
        line = [f"{prefix}{conn}{label}  [len={clade.branch_length:.6g}, cum={cum_dist:.6g}, leaves={leaves}]"]

    kids = getattr(clade, "clades", []) or []
    n = len(kids)
    for i, child in enumerate(kids):
        child_is_last = (i == n - 1)
        child_prefix = prefix + (spc if is_last else bar)
        edge_len = child.branch_length or 0.0
        line.extend(
            ascii_tree_lines(
                child,
                namer,
                prefix=child_prefix,
                is_last=child_is_last,
                depth=depth + 1,
                cum_dist=cum_dist + edge_len,
            )
        )
    return line

if __name__ == "__main__":
    
    # FPATH_TREE = paths["nymph_phylo_tree"]
    FPATH_TREE = paths["lepid_phylo_tree_r"]
    FPATH_READABLE_TREE = "tools/readable_tree.txt"

    tree = Phylo.read(FPATH_TREE, "newick")
    namer = Namer()
    lines = ascii_tree_lines(tree.root, namer)
    with open(FPATH_READABLE_TREE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")