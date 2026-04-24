"""
python -m preprocessing.cub.phylo
"""

from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree  # type: ignore[import]

from preprocessing.common.phylo import augment_class_data, prune_tree, augment_tree_with_polytomies
from utils.utils import paths, load_pickle, save_pickle


def build_tree_cub() -> Tree:

    path = "data/cub/output.nex"
    trees = list(Phylo.parse(path, "nexus"))

    # GET TREE (FOR NOW, USING ONLY THE FIRST ONE)
    tree = trees[0]

    for tip in tree.get_terminals():
        tip.name = tip.name.lower()

    return tree

def main():
    print("Building CUB tree...")

    tree = build_tree_cub()

    class_data = load_pickle(paths["metadata"]["cub"] / "class_data.pkl")
    class_data = {
        k.replace(" ", "_").lower(): {
            "order": v["order"].lower(),
            "family": v["family"].lower(),
            "genus": v["genus"].lower(),
            "rdpath_imgs": v["rdpath_imgs"],
            "split": v["split"]
        }
        for k, v in class_data.items()
    }

    class_data_aug = augment_class_data(class_data, tree)

    tree_pruned = prune_tree(tree, class_data_aug)
    tree_poly = augment_tree_with_polytomies(tree_pruned, class_data_aug)
    tree_poly_pruned = prune_tree(tree_poly, class_data)

    save_pickle(tree_poly_pruned, paths["metadata"]["cub"] / "tree.pkl")
    print("CUB tree complete")


if __name__ == "__main__":
    main()