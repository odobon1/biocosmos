from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import heapq
from typing import Dict, List, Optional, Tuple, Set
from Bio.Phylo.BaseTree import Tree, Clade  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from utils.data import species_to_genus

import pdb


# Default rank order from broad -> specific (excluding species)
RANK_ORDER = ["order", "family", "subfamily", "tribe", "genus"]


def get_leaf_names(tree: Tree) -> List[str]:
    return [leaf.name for leaf in tree.get_terminals() if leaf.name is not None]


# augment_tree_with_polytomies() helper
def get_available_ranks(
    class_data: Dict[str, Dict[str, Optional[str]]],
    candidate_ranks: Optional[List[str]] = None,
) -> List[str]:
    """
    Detect which taxonomic ranks are actually available (non-None) in class_data.
    Returns ranks in order from broadest to most specific.

    Parameters
    ----------
    class_data
        Dict of class-id -> {rank: value, ...}
    candidate_ranks
        Optional list of ranks to check. If None, uses RANK_ORDER.
        Should be ordered from broad to specific.

    Returns
    -------
    available_ranks
        Subset of candidate_ranks that have at least one non-None value in class_data.
        Ordered from broad to specific (genus last).
    """
    if candidate_ranks is None:
        candidate_ranks = RANK_ORDER

    available = []
    for rank in candidate_ranks:
        # Check if this rank has any non-None values in the class_data
        for cid_data in class_data.values():
            if cid_data.get(rank) is not None:
                available.append(rank)
                break

    return available

# augment_tree_with_polytomies() helper
def compress_lineage(
    cid_data: Dict[str, Optional[str]],
    rank_order: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Returns the class's available lineage as a list of (rank, taxon),
    skipping missing ranks (None).

    Parameters
    ----------
    cid_data
        Single class's class_data dict like {"family": "fam1", "subfamily": "sub1", ...}
    rank_order
        Order to process ranks. If None, uses RANK_ORDER. Should go from broad to specific.

    Example:
        {
            "family": "fam1",
            "subfamily": "sub1",
            "tribe": None,
            "genus": "gen1",
        }

    becomes:
        [("family", "fam1"), ("subfamily", "sub1"), ("genus", "gen1")]
    """
    if rank_order is None:
        rank_order = RANK_ORDER

    lineage = []
    for rank in rank_order:
        val = cid_data.get(rank)
        if val is not None:
            lineage.append((rank, val))
    return lineage

# augment_tree_with_polytomies() helper
def get_parent_map_for_classes(
    class_data: Dict[str, Dict[str, Optional[str]]],
    rank_order: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Optional[Tuple[str, str]]]]:
    """
    For each class, returns a mapping:
        rank -> parent(rank, taxon) within that class's compressed lineage

    Parameters
    ----------
    class_data
        Dict of class-id -> {rank: value, ...}
    rank_order
        Order to process ranks. If None, uses RANK_ORDER.

    Example compressed lineage:
        family -> subfamily -> genus

    then:
        family -> None
        subfamily -> ("family", fam)
        genus -> ("subfamily", subfam)
    """
    if rank_order is None:
        rank_order = RANK_ORDER

    out: Dict[str, Dict[str, Optional[Tuple[str, str]]]] = {}

    for cid, row in class_data.items():
        lineage = compress_lineage(row, rank_order)
        parent_info: Dict[str, Optional[Tuple[str, str]]] = {}
        for i, (rank, taxon) in enumerate(lineage):
            if i == 0:
                parent_info[rank] = None
            else:
                parent_info[rank] = lineage[i - 1]
        out[cid] = parent_info

    return out

# augment_tree_with_polytomies() helper
def represented_classes_by_rank_value(
    class_data: Dict[str, Dict[str, Optional[str]]],
    represented: Set[str],
    rank: str,
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for cid in represented:
        rank_value = class_data[cid].get(rank)
        if rank_value is not None:
            out[rank_value].add(cid)
    return out

# augment_tree_with_polytomies() helper
def deepest_mrca_between_groups(
    tree: Tree,
    group1_classes: Set[str],
    group2_classes: Set[str],
    depth_map: Dict,
) -> Optional[Clade]:
    """
    Among all MRCA(a, b) with a in group1, b in group2, return the deepest one.

    This runs in O(number_of_nodes) by propagating group-membership flags upward,
    avoiding pairwise MRCA calls.
    """
    if not group1_classes or not group2_classes:
        return None

    best_node: Optional[Clade] = None
    best_depth = -1.0

    def walk(node: Clade) -> Tuple[bool, bool]:
        nonlocal best_node, best_depth

        if not node.clades:
            name = node.name
            return name in group1_classes, name in group2_classes

        child_flags: List[Tuple[bool, bool]] = [walk(child) for child in node.clades]

        has1 = any(f1 for f1, _ in child_flags)
        has2 = any(f2 for _, f2 in child_flags)
        if not (has1 and has2):
            return has1, has2

        idx_with_1 = {i for i, (f1, _) in enumerate(child_flags) if f1}
        idx_with_2 = {i for i, (_, f2) in enumerate(child_flags) if f2}

        # Node is an MRCA candidate only when group1 and group2 can be drawn
        # from different child subtrees.
        split_across_children = not (len(idx_with_1) == 1 and idx_with_1 == idx_with_2)
        if split_across_children:
            depth = depth_map[node]
            if depth > best_depth:
                best_depth = depth
                best_node = node

        return has1, has2

    walk(tree.root)

    return best_node

# augment_tree_with_polytomies() helper
def find_graft_node_for_class(
    tree: Tree,
    cid: str,
    class_data: Dict[str, Dict[str, Optional[str]]],
    rep_by_rank_taxon: Dict[str, Dict[str, Set[str]]],
    taxa_by_parent: Dict[str, Dict[Optional[Tuple[str, str]], Set[str]]],
    parent_map: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    depth_map: Dict,
    graft_cache: Dict[Tuple[str, str, Optional[Tuple[str, str]]], Optional[Clade]],
    rank_order: Optional[List[str]] = None,
) -> Clade:
    """
    Find the graft node for one missing class using the most specific valid rank.

    Parameters
    ----------
    rank_order
        Order to process ranks. If None, uses RANK_ORDER.
    """
    if rank_order is None:
        rank_order = RANK_ORDER

    lineage = compress_lineage(class_data[cid], rank_order)

    # Search from most specific available rank upward: genus, tribe, subfamily, family
    for rank, taxon in reversed(lineage):
        reps_in_taxon = rep_by_rank_taxon.get(rank, {}).get(taxon, set())
        if not reps_in_taxon:
            continue

        parent = parent_map[cid].get(rank)
        if parent is None:
            sibling_taxa = set(rep_by_rank_taxon.get(rank, {})) - {taxon}
        else:
            sibling_taxa = taxa_by_parent.get(rank, {}).get(parent, set()) - {taxon}

        cache_key = (rank, taxon, parent)
        if cache_key in graft_cache:
            graft_node = graft_cache[cache_key]
        else:
            represented_sibling_classes: Set[str] = set()
            for sib_taxon in sibling_taxa:
                represented_sibling_classes.update(rep_by_rank_taxon.get(rank, {}).get(sib_taxon, set()))

            if not represented_sibling_classes:
                # No siblings at this rank - skip and try higher rank
                graft_cache[cache_key] = None
                continue

            graft_node = deepest_mrca_between_groups(
                tree=tree,
                group1_classes=reps_in_taxon,
                group2_classes=represented_sibling_classes,
                depth_map=depth_map,
            )
            graft_cache[cache_key] = graft_node

        if graft_node is not None:
            return graft_node

    # Final fallback: attach at the root.
    # We deliberately avoid returning a leaf here — grafting onto a leaf turns it
    # into an internal node, silently removing it from get_terminals().
    return tree.root

# augment_tree_with_polytomies() helper
def add_child_as_polytomy(
    node: Clade,
    cid: str,
    branch_length: Optional[float] = 0.0,
) -> None:
    node.clades.append(Clade(name=cid, branch_length=branch_length))

# augment_tree_with_polytomies() helper
def rehome_missing_classes_in_represented_higher_rank(
    tree: Tree,
    cids_missing: Set[str],
    cids_cd_on_tree: Set[str],
    class_data: Dict[str, Dict[str, Optional[str]]],
    rank_order: Optional[List[str]] = None,
) -> None:
    """
    Rehome newly inserted missing classes so each higher-rank class is attached at an
    interpretable divergence anchor.

    Case A (higher-rank class represented in original tree):
        anchor at the most recent inter-class divergence available for that class.

    Case B (higher-rank class not represented in original tree):
        fallback to the most recent inter-family (or next higher-rank class) divergence
        available for the class' family.

    Parameters
    ----------
    rank_order
        Ranks to use for hierarchy. If None, uses RANK_ORDER.
    """
    if rank_order is None:
        rank_order = RANK_ORDER

    if not cids_missing:
        return

    def _build_adj_map(tree_obj: Tree) -> Dict[Clade, List[Tuple[Clade, float]]]:
        adj: Dict[Clade, List[Tuple[Clade, float]]] = defaultdict(list)

        def walk(node: Clade) -> None:
            for child in node.clades:
                w = float(child.branch_length or 0.0)
                adj[node].append((child, w))
                adj[child].append((node, w))
                walk(child)

        walk(tree_obj.root)
        return adj

    def _nearest_eligible_tip_from_ref(
        ref_name: str,
        tip_clade_by_name: Dict[str, Clade],
        adj: Dict[Clade, List[Tuple[Clade, float]]],
        eligible_tips: Set[str],
    ) -> Optional[str]:
        ref_clade = tip_clade_by_name[ref_name]
        heap: List[Tuple[float, int, Clade, Optional[Clade]]] = []
        counter = 0
        heapq.heappush(heap, (0.0, counter, ref_clade, None))
        visited: Set[Clade] = set()

        while heap:
            dist, _, node, parent = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            if (
                node.name is not None
                and node.name != ref_name
                and node.name in eligible_tips
            ):
                return node.name

            for nb, w in adj.get(node, []):
                if parent is not None and nb is parent:
                    continue
                counter += 1
                heapq.heappush(heap, (dist + w, counter, nb, node))

        return None

    def _build_parent_map(tree_obj: Tree) -> Dict[Clade, Optional[Clade]]:
        parent_map: Dict[Clade, Optional[Clade]] = {tree_obj.root: None}

        def walk(node: Clade) -> None:
            for child in node.clades:
                parent_map[child] = node
                walk(child)

        walk(tree_obj.root)
        return parent_map

    tip_clade_by_name: Dict[str, Clade] = {
        tip.name: tip for tip in tree.get_terminals() if tip.name is not None
    }
    adj_map = _build_adj_map(tree)

    # Filter represented_original to only classes actually on this tree
    represented_original_present = {
        cid for cid in cids_cd_on_tree if cid in tip_clade_by_name
    }

    missing_by_genus: Dict[str, Set[str]] = defaultdict(set)
    for cid in cids_missing:
        missing_by_genus[cid.split("_", 1)[0]].add(cid)

    represented_by_genus: Dict[str, Set[str]] = defaultdict(set)
    for cid in represented_original_present:
        represented_by_genus[cid.split("_", 1)[0]].add(cid)

    represented_tips_present = {
        cid
        for cid in represented_original_present
        if cid in tip_clade_by_name and cid in class_data
    }
    represented_tips = sorted(represented_tips_present)

    # Build representation map for higher ranks (exclude genus since we search by genus)
    # Go from most specific to least specific (reverse of rank_order)
    higher_ranks = [r for r in reversed(rank_order) if r != "genus"]
    represented_by_rank_value: Dict[str, Dict[str, List[str]]] = {
        rank: defaultdict(list) for rank in higher_ranks
    }
    for cid in represented_tips:
        row = class_data.get(cid, {})
        for rank in higher_ranks:
            rank_val = row.get(rank)
            if rank_val:
                represented_by_rank_value[rank][rank_val].append(cid)

    planned_rehomes: List[Tuple[List[str], Clade, float]] = []

    for genus in sorted(missing_by_genus.keys()):
        # Case A: genus already represented in original tree.
        ref_candidates_repr = sorted(
            cid
            for cid in represented_by_genus.get(genus, set())
            if tree.find_any(name=cid) is not None
        )
        if ref_candidates_repr:
            ref_tip = ref_candidates_repr[0]
            sample_tax = class_data.get(ref_tip, {})
            candidate_tips: List[str] = []
            # Search from most specific higher rank upward to find closest sibling taxon
            for rank in higher_ranks:
                rank_value = sample_tax.get(rank)
                if not rank_value:
                    continue
                rank_candidates = [
                    cid
                    for cid in represented_by_rank_value[rank].get(rank_value, [])
                    if cid.split("_", 1)[0] != genus
                ]
                if rank_candidates:
                    candidate_tips = rank_candidates
                    break

            if not candidate_tips:
                candidate_tips = [
                    cid for cid in represented_tips if cid.split("_", 1)[0] != genus
                ]

            if not candidate_tips:
                continue

            # class-level nearest-neighbor selection
            closest_tip = _nearest_eligible_tip_from_ref(
                ref_name=ref_tip,
                tip_clade_by_name=tip_clade_by_name,
                adj=adj_map,
                eligible_tips=set(candidate_tips),
            )
            if closest_tip is None:
                continue

            anchor = tree.common_ancestor([ref_tip, closest_tip])
            attach_length = tree.distance(anchor, closest_tip)
            planned_rehomes.append((sorted(missing_by_genus[genus]), anchor, attach_length))
            continue

        # Case B: genus absent in original tree.
        genus_species_any = next(iter(missing_by_genus[genus]))
        genus_tax = class_data.get(genus_species_any, {})
        # Find the first available (non-None) rank value at higher rank level
        top_rank = None
        top_rank_value = None
        for rank in higher_ranks:
            val = genus_tax.get(rank)
            if val is not None:
                top_rank = rank
                top_rank_value = val
                break
        if not top_rank_value:
            continue

        ref_candidates_family = sorted(
            cid
            for cid in represented_tips_present
            if class_data.get(cid, {}).get(top_rank) == top_rank_value
            and cid.split("_", 1)[0] != genus
        )
        if not ref_candidates_family:
            continue

        ref_tip = ref_candidates_family[0]
        candidate_tips = [
            cid
            for cid in represented_tips
            if class_data.get(cid, {}).get(top_rank) not in (None, top_rank_value)
        ]
        if not candidate_tips:
            continue

        closest_tip = _nearest_eligible_tip_from_ref(
            ref_name=ref_tip,
            tip_clade_by_name=tip_clade_by_name,
            adj=adj_map,
            eligible_tips=set(candidate_tips),
        )
        if closest_tip is None:
            continue

        anchor = tree.common_ancestor([ref_tip, closest_tip])
        attach_length = tree.distance(anchor, closest_tip)
        planned_rehomes.append((sorted(missing_by_genus[genus]), anchor, attach_length))

    if not planned_rehomes:
        return

    # Detach all targets first in one pass; this is much faster than repeated prune().
    parent_by_clade = _build_parent_map(tree)
    target_cids = {cid for cids, _, _ in planned_rehomes for cid in cids}
    for cid in target_cids:
        clade = tip_clade_by_name.get(cid)
        if clade is None:
            continue
        parent = parent_by_clade.get(clade)
        if parent is None:
            continue
        parent.clades = [child for child in parent.clades if child is not clade]

    # Reattach grouped by genus-specific anchor and branch length.
    for cids, anchor, attach_length in planned_rehomes:
        for cid in cids:
            add_child_as_polytomy(
                node=anchor,
                cid=cid,
                branch_length=attach_length,
            )

# augment_tree_with_polytomies() helper
def _build_clade_path_map(tree: Tree) -> Dict[Clade, Tuple[int, ...]]:
    path_map: Dict[Clade, Tuple[int, ...]] = {}

    def walk(node: Clade, path: Tuple[int, ...]) -> None:
        path_map[node] = path
        for i, child in enumerate(node.clades):
            walk(child, path + (i,))

    walk(tree.root, ())
    return path_map

# augment_tree_with_polytomies() helper
def _get_clade_by_path(tree: Tree, path: Tuple[int, ...]) -> Clade:
    node = tree.root
    for idx in path:
        node = node.clades[idx]
    return node

_WORKER_TREE: Optional[Tree] = None
_WORKER_CLASS_DATA: Optional[Dict[str, Dict[str, Optional[str]]]] = None
_WORKER_REP_BY_RANK_TAXON: Optional[Dict[str, Dict[str, Set[str]]]] = None
_WORKER_TAXA_BY_PARENT: Optional[Dict[str, Dict[Optional[Tuple[str, str]], Set[str]]]] = None
_WORKER_PARENT_MAP: Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]] = None
_WORKER_DEPTH_MAP: Optional[Dict] = None
_WORKER_PATH_BY_CLADE: Optional[Dict[Clade, Tuple[int, ...]]] = None
_WORKER_GRAFT_CACHE: Optional[Dict[Tuple[str, str, Optional[Tuple[str, str]]], Optional[Clade]]] = None

_WORKER_RANK_ORDER: Optional[List[str]] = None

# augment_tree_with_polytomies() helper
def _worker_init(
    tree: Tree,
    class_data: Dict[str, Dict[str, Optional[str]]],
    rep_by_rank_taxon: Dict[str, Dict[str, Set[str]]],
    taxa_by_parent: Dict[str, Dict[Optional[Tuple[str, str]], Set[str]]],
    parent_map: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    rank_order: Optional[List[str]] = None,
) -> None:
    global _WORKER_TREE
    global _WORKER_CLASS_DATA
    global _WORKER_REP_BY_RANK_TAXON
    global _WORKER_TAXA_BY_PARENT
    global _WORKER_PARENT_MAP
    global _WORKER_DEPTH_MAP
    global _WORKER_PATH_BY_CLADE
    global _WORKER_GRAFT_CACHE
    global _WORKER_RANK_ORDER

    _WORKER_TREE = tree
    _WORKER_CLASS_DATA = class_data
    _WORKER_REP_BY_RANK_TAXON = rep_by_rank_taxon
    _WORKER_TAXA_BY_PARENT = taxa_by_parent
    _WORKER_PARENT_MAP = parent_map
    _WORKER_RANK_ORDER = rank_order if rank_order is not None else RANK_ORDER
    _WORKER_DEPTH_MAP = tree.depths()
    _WORKER_PATH_BY_CLADE = _build_clade_path_map(tree)
    _WORKER_GRAFT_CACHE = {}

# augment_tree_with_polytomies() helper
def _find_graft_node_path_for_class_worker(
    cid: str,
) -> Tuple[str, Tuple[int, ...]]:
    if (
        _WORKER_TREE is None
        or _WORKER_CLASS_DATA is None
        or _WORKER_REP_BY_RANK_TAXON is None
        or _WORKER_TAXA_BY_PARENT is None
        or _WORKER_PARENT_MAP is None
        or _WORKER_DEPTH_MAP is None
        or _WORKER_PATH_BY_CLADE is None
        or _WORKER_GRAFT_CACHE is None
        or _WORKER_RANK_ORDER is None
    ):
        raise RuntimeError("Worker context not initialized")

    graft_node = find_graft_node_for_class(
        tree=_WORKER_TREE,
        cid=cid,
        class_data=_WORKER_CLASS_DATA,
        rep_by_rank_taxon=_WORKER_REP_BY_RANK_TAXON,
        taxa_by_parent=_WORKER_TAXA_BY_PARENT,
        parent_map=_WORKER_PARENT_MAP,
        depth_map=_WORKER_DEPTH_MAP,
        graft_cache=_WORKER_GRAFT_CACHE,
        rank_order=_WORKER_RANK_ORDER,
    )
    return cid, _WORKER_PATH_BY_CLADE[graft_node]

def augment_tree_with_polytomies(
    tree: Tree,
    class_data: Dict[str, Dict[str, Optional[str]]],
    branch_length: Optional[float] = 0.0,
    n_workers: int = 4,
) -> Tree:
    """
    Add missing classes to a phylogenetic tree using class_data-guided polytomy insertion.

    Rules:
    - Search upward through each class' compressed lineage (skip None ranks).
    - Use the most specific rank where:
        * the class' taxon has at least one original-tree representative
        * at least one sibling taxon at that rank also has an original-tree representative
    - Graft at the deepest MRCA between represented members of the target taxon
      and represented members of represented sibling taxa.

    Parameters
    ----------
    tree
        Bio.Phylo tree with some subset of classes already present as leaves.
    class_data
        Mapping like:
            {
                "A": {
                    "family": "family1",
                    "subfamily": "sub1" or None,
                    "tribe": "tribe1" or None,
                    "genus": "genus1",
                },
                ...
            }
        Available ranks are auto-detected from this class_data.
    branch_length
        Branch length to assign to newly grafted leaves.
    n_workers
        Number of worker processes for placement computation.
        Use 1 for serial execution.

    Returns
    -------
    augmented_tree
        Augmented tree with missing classes added as polytomies.
    """
    tree = deepcopy(tree)

    rank_order = get_available_ranks(class_data)

    parent_map = get_parent_map_for_classes(class_data, rank_order)

    cids_cd = set(class_data.keys())
    cids_cd_on_tree = set(get_leaf_names(tree)) & cids_cd
    cids_missing = sorted(cids_cd - cids_cd_on_tree)

    if not cids_missing:
        return tree
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    # Precompute once: for each (rank, taxon) -> set of represented classes
    rep_by_rank_taxon: Dict[str, Dict[str, Set[str]]] = {
        rank: represented_classes_by_rank_value(class_data, cids_cd_on_tree, rank)
        for rank in rank_order
    }

    # Precompute once: for each rank, group taxa by their compressed-lineage parent
    taxa_by_parent: Dict[str, Dict[Optional[Tuple[str, str]], Set[str]]] = {
        rank: defaultdict(set) for rank in rank_order
    }
    for _, cid_data in class_data.items():
        lineage = compress_lineage(cid_data, rank_order)
        for i, (rank, taxon) in enumerate(lineage):
            parent: Optional[Tuple[str, str]] = lineage[i - 1] if i > 0 else None
            taxa_by_parent[rank][parent].add(taxon)

    chunksize = max(1, len(cids_missing) // max(1, n_workers * 8))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(tree, class_data, rep_by_rank_taxon, taxa_by_parent, parent_map, rank_order),
    ) as executor:
        for cid, graft_path in executor.map(_find_graft_node_path_for_class_worker, cids_missing, chunksize=chunksize):
            add_child_as_polytomy(
                node=_get_clade_by_path(tree, graft_path),
                cid=cid,
                branch_length=branch_length,
            )

    rehome_missing_classes_in_represented_higher_rank(
        tree=tree,
        cids_missing=set(cids_missing),
        cids_cd_on_tree=cids_cd_on_tree,
        class_data=class_data,
        rank_order=rank_order,
    )

    return tree

def prune_tree(tree, class_data):
    tree = deepcopy(tree)
    for tip in tree.get_terminals():
        if tip.name not in class_data.keys():
            tree.prune(target=tip.name)
    return tree

def augment_class_data(class_data, tree):
    """
    Infer taxonomy for classes on the tree that are not in class_data,
    using existing classes from the same genus.

    This function works with any set of taxonomic ranks present in class_data
    (e.g., family+subfamily+tribe+genus for Lepid, or just subfamily+genus for Nymph).

    Parameters
    ----------
    class_data
        Dict mapping class ID -> {rank: value, ...}
    tree
        Bio.Phylo tree with terminal nodes named by class ID

    Returns
    -------
    augmented_class_data
        class_data extended with entries for tree classes not originally in it.
        Each new entry copies all ranks from a representative of the same genus.
    """
    class_data = deepcopy(class_data)

    cids_tree = {tip.name for tip in tree.get_terminals()}

    cids_cd = set(class_data.keys())
    genera_cids_cd = set([species_to_genus(cid) for cid in cids_cd])

    cids_tree_non_cd = cids_tree - cids_cd
    genera_cids_tree_non_cd = set([species_to_genus(cid) for cid in cids_tree_non_cd])

    genera_in_common = genera_cids_tree_non_cd & genera_cids_cd

    for genus in genera_in_common:
        # Get a representative class from this genus that's already in class_data
        cids_genus = [cid for cid in cids_cd if class_data[cid]["genus"] == genus]
        if not cids_genus:
            continue

        # Skip ambiguous genera where any non-genus taxonomy rank has
        # conflicting non-None values across classes already in class_data.
        # This keeps us from inferring incorrect ranks for missing tree tips.
        # Note: exclude "species" from ambiguity check since different species
        # in the same genus will have different species values by definition.
        rank_fields = [
            rank
            for rank in class_data[cids_genus[0]].keys()
            if rank not in {"genus", "common_name", "species"}
        ]
        genus_is_ambiguous = False
        for rank in rank_fields:
            values = {
                class_data[cid].get(rank)
                for cid in cids_genus
                if class_data[cid].get(rank) is not None
            }
            if len(values) > 1:
                genus_is_ambiguous = True
                break
        if genus_is_ambiguous:
            continue

        # Extract available ranks from the representative (only non-None values)
        representative = class_data[cids_genus[0]]
        ranks_to_copy = {rank: representative[rank] for rank in representative if rank != "common_name"}
        
        # Find tree classes of higher-rank class not yet in class_data
        cids_tree_genus = [cid for cid in cids_tree_non_cd if cid.split("_")[0] == genus]

        # Add each one to class_data with the inferred ranks
        for cid in cids_tree_genus:
            class_data[cid] = ranks_to_copy.copy()

    return class_data