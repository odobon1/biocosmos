import re
from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree  # type: ignore[import]

from utils.utils import paths


def build_tree_bryo() -> Tree:
    tree = Phylo.read(paths["root"] / "metadata/bryo/SI_Fig1(BIG).newick", "newick")
    tip_names_trunc = set()
    for tip in tree.get_terminals():
        tip_name_trunc = trim_after_all_caps_underscore(tip.name)
        if tip_name_trunc in tip_names_trunc or tip.name.startswith("UNKNOWN"):
            tree.prune(target=tip.name)
        else:
            tip.name = tip_name_trunc
            tip_names_trunc.add(tip_name_trunc)
    return tree

# helper function for build_tree_bryo()
def trim_after_all_caps_underscore(s: str) -> str:
    """
    Remove the first underscore followed by an ALL-CAPS token and everything after it.

    Examples
    --------
    'Fenestrulina_n_sp1_BLEED237' -> 'Fenestrulina_n_sp1'
    'Fenestrulina_sp_SEQ_BLEED1261' -> 'Fenestrulina_sp'
    """
    return re.sub(r'_[A-Z][A-Z0-9_]*$', '', s)