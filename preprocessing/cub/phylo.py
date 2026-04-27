"""
python -m preprocessing.cub.phylo
"""

from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree  # type: ignore[import]

from preprocessing.common.phylo import augment_class_data, prune_tree, augment_tree_with_polytomies
from utils.utils import paths, load_pickle, save_pickle

import pdb


def build_tree_cub() -> Tree:
    tree = Phylo.read(paths["cub_tree_raw"], "newick")
    for tip in tree.get_terminals():
        tip.name = tip.name[8:].lower().replace(" ", "_")
    return tree

def main():
    print("Building CUB tree...")

    tree = build_tree_cub()
    class_data = load_pickle(paths["metadata"]["cub"] / "class_data.pkl")

    tree_poly = augment_tree_with_polytomies(tree, class_data)
    tree_poly_pruned = prune_tree(tree_poly, class_data)

    save_pickle(tree_poly_pruned, paths["metadata"]["cub"] / "tree.pkl")
    print("CUB tree complete")


if __name__ == "__main__":
    main()