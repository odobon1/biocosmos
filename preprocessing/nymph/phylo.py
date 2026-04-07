from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree  # type: ignore[import]

from utils.utils import paths, save_pickle


def build_tree_nymph() -> Tree:
    tree = Phylo.read(paths["nymph_phylo_tree"], "newick")
    return tree

def main():
    print("Building Nymphalidae tree...")
    tree = build_tree_nymph()
    save_pickle(tree, paths["metadata"]["nymph"] / "tree.pkl")
    print("Nymphalidae tree complete")


if __name__ == "__main__":
    main()