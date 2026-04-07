from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Clade, Tree  # type: ignore[import]
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from utils.utils import paths, load_pickle, save_pickle
from utils.data import before_second_underscore
from preprocessing.nymph.phylo import build_tree_nymph


def build_tree_lepid() -> Tree:
    tree = Phylo.read(paths["lepid_phylo_tree"], "newick")
    # heliconius_cydno/heliconius_cydno_alithea is the only instance where subspecies + corresponding species are both in tree, remove subspecies
    tree.prune(target="heliconius_cydno_alithea")
    # convert all subspecies names to species-level names by truncating after the second underscore
    for tip in tree.get_terminals():
        if tip.name.count("_") >= 2:
            tip.name = before_second_underscore(tip.name)

    # class_data = load_pickle(paths["metadata"]["lepid"] / "class_data.pkl")
    # for tip in tree.get_terminals():
    #     if tip.name not in class_data.keys():
    #         tree.prune(target=tip.name)

    return tree

def combine_trees_lepid_nymph(
    tree_lepid: Tree,
    tree_nymph: Tree,
) -> Tree:
    """
    Merge the Lepidoptera tree with the Nymphalidae tree.

    Design goals
    ------------
    - Keep the Lepid tree as the global backbone.
    - Expand shared Lepid tips when the Nymph tree gives an unambiguous
      single-shared clade containing additional Nymph-only taxa.
    - Attach any still-unplaced Nymph-only taxa as one residual Nymph subtree
      under the Lepid overlap anchor.

    This preserves the tested Lepid pairwise distances by leaving the Lepid
    backbone untouched, preserves the tested Nymph distances by reusing exact
    Nymph clades for inserted taxa, and keeps the merged tree ultrametric by
    solving each graft stem length against the Lepid tree height.
    """

    out = deepcopy(tree_lepid)
    nymph = deepcopy(tree_nymph)

    lepid_tips = tip_names(out.root)
    nymph_tips = tip_names(nymph.root)
    shared = lepid_tips & nymph_tips

    if not shared:
        raise ValueError("No shared taxa between Lepid and Nymph trees.")

    if nymph_tips <= lepid_tips:
        return out

    target_height = tree_height(out)
    nymph_only = nymph_tips - lepid_tips

    placed_nymph_only: Set[str] = set()
    for shared_sid, nymph_clade, nymph_only_here in collect_single_shared_expansions(
        nymph,
        shared,
        out,
    ):
        shared_tip = out.find_any(name=shared_sid)
        if shared_tip is None:
            continue

        parent = find_parent(out.root, shared_tip)
        if parent is None:
            continue

        shared_depth_in_clade = distance_from_clade(nymph_clade, shared_sid)
        available_branch = shared_tip.branch_length or 0.0
        graft = deepcopy(nymph_clade)
        graft.branch_length = available_branch - shared_depth_in_clade

        replace_child(parent, shared_tip, graft)
        placed_nymph_only |= nymph_only_here

    residual_nymph_only = nymph_only - placed_nymph_only
    if residual_nymph_only:
        anchor = out.common_ancestor(list(shared))
        residual_root = clone_induced_subtree(nymph, residual_nymph_only)
        if residual_root is None:
            return out

        residual_root.branch_length = max(
            0.0,
            target_height - out.distance(out.root, anchor) - subtree_height(residual_root),
        )
        anchor.clades.append(residual_root)

    return out

# combine_trees_lepid_nymph() helper
def tip_names(clade: Clade) -> Set[str]:
    return {tip.name for tip in clade.get_terminals() if tip.name is not None}

# combine_trees_lepid_nymph() helper
def tree_height(tree: Tree) -> float:
    root = tree.root
    return max(tree.distance(root, tip) for tip in tree.get_terminals())

# combine_trees_lepid_nymph() helper
def subtree_height(clade: Clade) -> float:
    if not clade.clades:
        return 0.0
    return max((child.branch_length or 0.0) + subtree_height(child) for child in clade.clades)

# combine_trees_lepid_nymph() helper
def distance_from_clade(root: Clade, target_name: str) -> float:
    path = path_from_root(root, target_name)
    if not path:
        raise ValueError(f"Target {target_name!r} not found under clade.")

    return sum(node.branch_length or 0.0 for node in path[1:])

# combine_trees_lepid_nymph() helper
def path_from_root(root: Clade, target_name: str) -> List[Clade]:
    path: List[Clade] = []

    def dfs(cur: Clade) -> bool:
        path.append(cur)
        if cur.name == target_name and cur.is_terminal():
            return True
        for child in cur.clades:
            if dfs(child):
                return True
        path.pop()
        return False

    found = dfs(root)
    return path if found else []

# combine_trees_lepid_nymph() helper
def find_parent(root: Clade, target: Clade) -> Optional[Clade]:
    for parent in root.find_clades(order="level"):
        for child in parent.clades:
            if child is target:
                return parent
    return None

# combine_trees_lepid_nymph() helper
def replace_child(parent: Clade, old_child: Clade, new_child: Clade) -> None:
    for idx, child in enumerate(parent.clades):
        if child is old_child:
            parent.clades[idx] = new_child
            return
    raise ValueError("old_child not found under parent")

# combine_trees_lepid_nymph() helper
def compress_unary(clade: Clade) -> Clade:
    clade.clades = [compress_unary(child) for child in clade.clades]

    while len(clade.clades) == 1 and clade.name is None:
        child = clade.clades[0]
        clade.name = child.name
        clade.branch_length = (clade.branch_length or 0.0) + (child.branch_length or 0.0)
        clade.clades = child.clades

    return clade

# combine_trees_lepid_nymph() helper
def clone_induced_subtree(tree: Tree, keep: Set[str]) -> Optional[Clade]:
    if not keep:
        return None

    new_tree = deepcopy(tree)
    for tip in list(new_tree.get_terminals()):
        if tip.name not in keep:
            new_tree.prune(tip)

    compress_unary(new_tree.root)
    return deepcopy(new_tree.root)

# combine_trees_lepid_nymph() helper
def collect_single_shared_expansions(
    nymph: Tree,
    shared: Set[str],
    lepid: Tree,
) -> List[Tuple[str, Clade, Set[str]]]:
    """
    Return maximal Nymph clades that contain exactly one shared Lepid tip and
    at least one Nymph-only tip.

    When the shared tip's terminal branch in Lepid is long enough to absorb the
    local Nymph clade depth, we can replace that single Lepid tip with the full
    Nymph clade while preserving all Lepid-vs-Lepid distances.
    """

    parent_by_id: Dict[int, Clade] = {}
    tip_cache: Dict[int, Set[str]] = {}
    fill_tip_cache(nymph.root, tip_cache)

    for parent in nymph.find_clades(order="level"):
        for child in parent.clades:
            parent_by_id[id(child)] = parent

    expansions: List[Tuple[str, Clade, Set[str]]] = []
    for node in nymph.find_clades(order="postorder"):
        desc = tip_cache[id(node)]
        shared_here = desc & shared
        nymph_only_here = desc - shared

        if len(shared_here) != 1 or not nymph_only_here:
            continue

        parent = parent_by_id.get(id(node))
        if parent is not None and len(tip_cache[id(parent)] & shared) == 1:
            continue

        shared_sid = next(iter(shared_here))
        lepid_tip = lepid.find_any(name=shared_sid)
        if lepid_tip is None:
            continue

        shared_depth_in_clade = distance_from_clade(node, shared_sid)
        available_branch = lepid_tip.branch_length or 0.0
        if available_branch + 1e-8 < shared_depth_in_clade:
            continue

        expansions.append((shared_sid, node, nymph_only_here))

    return expansions

# combine_trees_lepid_nymph() helper
def fill_tip_cache(clade: Clade, cache: Dict[int, Set[str]]) -> Set[str]:
    if clade.is_terminal():
        names = {clade.name} if clade.name is not None else set()
    else:
        names = set()
        for child in clade.clades:
            names |= fill_tip_cache(child, cache)

    cache[id(clade)] = names
    return names

def main():
    print("Building Lepidoptera tree...")
    tree_lepid = build_tree_lepid()
    tree_nymph = build_tree_nymph()
    tree_lepid = combine_trees_lepid_nymph(tree_lepid, tree_nymph)
    save_pickle(tree_lepid, paths["metadata"]["lepid"] / "tree.pkl")
    print("Lepidoptera tree complete")


if __name__ == "__main__":
    main()