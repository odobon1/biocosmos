"""
python -m preprocessing.cub.phylo
"""

from Bio import Phylo
from Bio.Phylo.BaseTree import Tree

from preprocessing.common.phylo import prune_tree, augment_tree_with_polytomies
from utils.utils import paths, load_pickle, save_pickle

import pdb


def build_tree_cub(class_data) -> Tree:
    common_to_cid = {data["common_name"]: cid for cid, data in class_data.items()}
    tree = Phylo.read(paths["raw_tree"]["cub"], "newick")
    for tip in tree.get_terminals():
        common_name = tip.name[8:].lower().replace(" ", "_")
        tip.name = common_to_cid[common_name]
    return tree

def main():
    print("Building CUB tree...")

    class_data = load_pickle(paths["metadata"]["cub"] / "class_data.pkl")
    tree = build_tree_cub(class_data)

    tree_poly = augment_tree_with_polytomies(tree, class_data)
    tree_poly_pruned = prune_tree(tree_poly, class_data)

    save_pickle(tree_poly_pruned, paths["metadata"]["cub"] / "tree.pkl")
    print("CUB tree complete")


if __name__ == "__main__":
    main()