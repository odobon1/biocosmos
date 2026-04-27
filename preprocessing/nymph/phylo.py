"""
python -m preprocessing.nymph.phylo
"""

from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree  # type: ignore[import]

from preprocessing.common.phylo import augment_class_data, prune_tree, augment_tree_with_polytomies
from utils.utils import paths, load_pickle, save_pickle

import pdb


def build_tree_nymph() -> Tree:
    tree = Phylo.read(paths["nymph_tree_raw"], "newick")
    for tip in tree.get_terminals():
        if tip.name in ("sp_1", "genus2_annette", "genus2_andromica"):
            tree.prune(target=tip.name)
    return tree

def main():
    print("Building Nymphalidae tree...")

    tree = build_tree_nymph()
    class_data = load_pickle(paths["metadata"]["nymph"] / "class_data.pkl")

    class_data_aug = augment_class_data(class_data, tree)
    tree_pruned = prune_tree(tree, class_data_aug)
    tree_poly = augment_tree_with_polytomies(tree_pruned, class_data_aug)
    tree_poly_pruned = prune_tree(tree_poly, class_data)

    save_pickle(tree_poly_pruned, paths["metadata"]["nymph"] / "tree.pkl")
    print("Nymphalidae tree complete")


if __name__ == "__main__":
    main()