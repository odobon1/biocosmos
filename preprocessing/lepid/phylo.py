from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from utils.data import before_second_underscore


def build_tree_lepid() -> Tree:
    tree = Phylo.read(paths["lepid_phylo_tree"], "newick")
    # heliconius_cydno/heliconius_cydno_alithea is the only instance where subspecies + corresponding species are both in tree, remove subspecies
    tree.prune(target="heliconius_cydno_alithea")
    # convert all subspecies names to species-level names by truncating after the second underscore
    for tip in tree.get_terminals():
        if tip.name.count("_") >= 2:
            tip.name = before_second_underscore(tip.name)

    class_data = load_pickle(paths["metadata"]["lepid"] / "class_data.pkl")
    for tip in tree.get_terminals():
        if tip.name not in class_data.keys():
            tree.prune(target=tip.name)

    return tree

def main():
    tree = build_tree_lepid()
    save_pickle(tree, paths["metadata"]["lepid"] / "tree.pkl")


if __name__ == "__main__":
    main()