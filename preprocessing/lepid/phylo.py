"""
python -m preprocessing.lepid.phylo
"""

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

    return tree

def combine_trees_lepid_nymph(
    tree_lepid: Tree,
    tree_nymph: Tree,
    class_data: Dict[str, Dict[str, str | None]],
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
        # First try placing residual taxa by genus near existing Lepid genus clades.
        lepid_genus_to_tips: Dict[str, Set[str]] = {}
        for sid in lepid_tips:
            genus = sid.split("_", 1)[0]
            lepid_genus_to_tips.setdefault(genus, set()).add(sid)

        lepid_genus_to_rank: Dict[str, Dict[str, str]] = {}
        for sid in lepid_tips:
            sid_data = class_data.get(sid)
            if sid_data is None:
                continue
            genus = sid.split("_", 1)[0]
            genus_rank_data = lepid_genus_to_rank.setdefault(genus, {})
            for rank in ("tribe", "subfamily", "family"):
                rank_value = sid_data.get(rank)
                if isinstance(rank_value, str) and rank_value:
                    genus_rank_data[rank] = rank_value

        residual_by_genus: Dict[str, Set[str]] = {}
        for sid in residual_nymph_only:
            genus = sid.split("_", 1)[0]
            residual_by_genus.setdefault(genus, set()).add(sid)

        attached_residual: Set[str] = set()
        for genus in sorted(residual_by_genus.keys()):
            if genus not in lepid_genus_to_tips:
                continue

            genus_residual = residual_by_genus[genus]
            residual_root = clone_induced_subtree(nymph, genus_residual)
            if residual_root is None:
                continue

            anchor = closest_genus_divergence_anchor(
                out,
                genus,
                lepid_tips,
                lepid_genus_to_tips,
                lepid_genus_to_rank,
            )
            anchor_depth = out.distance(out.root, anchor)
            residual_height = subtree_height(residual_root)
            if anchor_depth + residual_height > target_height + 1e-8:
                # No non-negative branch can keep this genus graft ultrametric.
                continue

            attach_subtree_at_anchor(out, anchor, residual_root, target_height)
            attached_residual |= genus_residual

        # Fallback for any unmatched genera: keep original shared-anchor behavior.
        unmatched_residual = residual_nymph_only - attached_residual
        if unmatched_residual:
            anchor = out.common_ancestor(list(shared))
            residual_root = clone_induced_subtree(nymph, unmatched_residual)
            if residual_root is None:
                return out

            attach_subtree_at_anchor(out, anchor, residual_root, target_height)

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
def attach_subtree_at_anchor(tree: Tree, anchor: Clade, residual_root: Clade, target_height: float) -> None:
    """
    Attach residual_root under anchor while preserving all existing Lepid terminals
    and keeping terminal depth at target_height.
    """

    anchor_depth = tree.distance(tree.root, anchor)
    residual_root.branch_length = max(
        0.0,
        target_height - anchor_depth - subtree_height(residual_root),
    )

    if not anchor.is_terminal():
        anchor.clades.append(residual_root)
        return

    parent = find_parent(tree.root, anchor)
    if parent is None:
        # Defensive fallback: this should not happen for a normal rooted tree.
        anchor.clades.append(residual_root)
        return

    # Split a terminal anchor into a small internal node so the original Lepid
    # tip remains terminal and keeps its original root distance.
    old_branch = anchor.branch_length or 0.0
    anchor.branch_length = 0.0

    wrapper = Clade(name=None, branch_length=old_branch)
    wrapper.clades = [anchor, residual_root]
    replace_child(parent, anchor, wrapper)

# combine_trees_lepid_nymph() helper
def closest_genus_divergence_anchor(
    tree: Tree,
    genus: str,
    lepid_tips: Set[str],
    lepid_genus_to_tips: Dict[str, Set[str]],
    lepid_genus_to_rank: Dict[str, Dict[str, str]],
) -> Clade:
    """
    Anchor residual genus grafts at the divergence between the target genus and
    its closest neighboring genus, preferring taxonomic proximity by rank:
    tribe, then subfamily, then family.
    """

    genus_tips = lepid_genus_to_tips[genus]
    genus_anchor = tree.common_ancestor(sorted(genus_tips))

    candidates: Set[str] = set()
    genus_rank_data = lepid_genus_to_rank.get(genus, {})
    for rank in ("tribe", "subfamily", "family"):
        rank_value = genus_rank_data.get(rank)
        if not rank_value:
            continue

        rank_candidates = {
            sid
            for sid in (lepid_tips - genus_tips)
            if lepid_genus_to_rank.get(sid.split("_", 1)[0], {}).get(rank) == rank_value
        }
        if rank_candidates:
            candidates = rank_candidates
            break

    if not candidates:
        candidates = lepid_tips - genus_tips
    if not candidates:
        return genus_anchor

    ref_tip = min(genus_tips)
    closest_tip = min(
        candidates,
        key=lambda sid: (tree.distance(ref_tip, sid), sid),
    )
    return tree.common_ancestor([ref_tip, closest_tip])

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

def augment_class_data(class_data, tree):
    class_data = deepcopy(class_data)

    sids_tree = {tip.name for tip in tree.get_terminals()}

    sids_cd = set(class_data.keys())
    genera_sids_cd = set([sid.split("_")[0] for sid in sids_cd])

    sids_tree_not_in_cd = sids_tree - sids_cd
    genera_sids_tree_not_in_cd = set([sid.split("_")[0] for sid in sids_tree_not_in_cd])

    genera_in_common = genera_sids_tree_not_in_cd & genera_sids_cd
    genera_in_common = genera_in_common - {"emesis"}  # genus "emesis" is in multiple families: {'riodinidae', 'lycaenidae'}

    for genus in genera_in_common:
        sids_genus = [sid for sid in sids_cd if class_data[sid]["genus"] == genus]
        family = class_data[sids_genus[0]]["family"]
        subfamily = class_data[sids_genus[0]]["subfamily"]
        tribe = class_data[sids_genus[0]]["tribe"]
        
        sids_tree_genus = [sid for sid in sids_tree_not_in_cd if sid.split("_")[0] == genus]

        for sid in sids_tree_genus:
            class_data[sid] = {
                "family": family,
                "subfamily": subfamily,
                "tribe": tribe,
                "genus": genus,
            }

    return class_data

def prune_tree(tree, class_data):
    for tip in tree.get_terminals():
        if tip.name not in class_data.keys():
            tree.prune(target=tip.name)
    return tree

def main():
    print("Building Lepidoptera tree...")

    tree_lepid = build_tree_lepid()
    tree_nymph = build_tree_nymph()

    class_data = load_pickle(paths["metadata"]["lepid"] / "class_data.pkl")
    # class data augmented with sids on trees not in class_data but with genera in class_data
    class_data_aug = augment_class_data(class_data, tree_lepid)
    class_data_aug = augment_class_data(class_data_aug, tree_nymph)

    tree_lepid = combine_trees_lepid_nymph(tree_lepid, tree_nymph, class_data_aug)
    tree_lepid = prune_tree(tree_lepid, class_data_aug)

    save_pickle(tree_lepid, paths["metadata"]["lepid"] / "tree.pkl")
    print("Lepidoptera tree complete")


if __name__ == "__main__":
    main()