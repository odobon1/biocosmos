"""
python -m preprocessing.cub.phylo

The consensus backbone (data/cub/1_tree-consensus-...phy) is missing 10 of CUB's 200
species. Those gaps are filled from the Jetz et al. (birdtree.org) Hackett Stage2
posterior sample (data/cub/AllBirdsHackett1.tre): each missing species is attached at
its majority-rule sister clade across the 1000 posterior trees, at the median
attachment age, leaving every backbone placement and branch length untouched.

The cid->Jetz mapping (GBIF) and the pruned posterior sample are cached in
preprocessing/cub/intermediaries/jetz_cache.pkl.gz (git-tracked, ~3.5 MB) and
rebuilt automatically when the class set changes, so re-runs skip both GBIF and
the 464 MB posterior parse.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
from io import StringIO
import pickle
from statistics import median
import threading
from typing import Dict, List, Optional, Set, Tuple

from Bio import Phylo
from Bio.Phylo.BaseTree import Tree, Clade
import requests

from preprocessing.common.gbif import gbif_name_candidates
from preprocessing.common.phylo import prune_tree, augment_tree_with_polytomies
from utils.utils import paths, load_pickle, save_pickle

import pdb


FPATH_JETZ_CACHE = paths["preproc"]["cub"] / "intermediaries" / "jetz_cache.pkl.gz"


# class ids whose Jetz counterpart can't be reached via GBIF synonymy (post-2012
# splits/lumps and renames); values are verbatim Jetz tip names
CID_TO_JETZ_CORRECTIONS = {
    "ailuroedus_maculosus": "Ailuroedus_melanotis",
    "ammospiza_nelsoni": "Ammodramus_nelsoni",
    "colibri_cyanotus": "Colibri_thalassinus",
    "larus_smithsonianus": "Larus_argentatus",
    "mareca_strepera": "Anas_strepera",
    "setophaga_aestiva": "Dendroica_petechia",
    "troglodytes_hiemalis": "Troglodytes_troglodytes",
    "vermivora_cyanoptera": "Vermivora_pinus",
}


def build_tree_cub(class_data) -> Tree:
    common_to_cid = {data["common_name"]: cid for cid, data in class_data.items()}
    tree = Phylo.read(paths["raw_tree"]["cub"], "newick")
    for tip in tree.get_terminals():
        common_name = tip.name[8:].lower().replace(" ", "_")
        tip.name = common_to_cid[common_name]
    return tree

def map_cids_to_jetz(cids: List[str], jetz_names: Set[str], n_workers: int = 16) -> Dict[str, str]:
    """
    cid -> Jetz tip name, matched via GBIF nomenclature (accepted name + synonyms)
    since Jetz uses 2012-era taxonomy. cids with no match are simply absent from
    the result. Fails loudly if two cids claim the same Jetz tip. GBIF lookups run
    in a thread pool with a connection-reusing session per thread.
    """
    jetz_by_lower = {name.lower(): name for name in jetz_names}
    local = threading.local()

    def match(cid: str) -> Tuple[str, Optional[str]]:
        if cid in CID_TO_JETZ_CORRECTIONS:
            return cid, CID_TO_JETZ_CORRECTIONS[cid]
        if not hasattr(local, "session"):
            local.session = requests.Session()
        for candidate in gbif_name_candidates(cid, session=local.session):
            if candidate in jetz_by_lower:
                return cid, jetz_by_lower[candidate]
        return cid, None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        cid_to_jetz = {cid: jetz for cid, jetz in executor.map(match, cids) if jetz is not None}

    claimed = Counter(cid_to_jetz.values())
    collisions = {tip for tip, n in claimed.items() if n > 1}
    if collisions:
        raise ValueError(f"Multiple cids map to the same Jetz tip(s): {collisions}")
    return cid_to_jetz

# load_pruned_jetz_trees() helper
def induced_subtree(clade: Clade, keep: Set[str]) -> Optional[Clade]:
    """Copy of `clade` containing only tips in `keep`, unary nodes collapsed."""
    if clade.is_terminal():
        if clade.name in keep:
            return Clade(name=clade.name, branch_length=clade.branch_length or 0.0)
        return None
    kids = [induced_subtree(child, keep) for child in clade.clades]
    kids = [kid for kid in kids if kid is not None]
    if not kids:
        return None
    if len(kids) == 1:
        kids[0].branch_length += clade.branch_length or 0.0
        return kids[0]
    node = Clade(branch_length=clade.branch_length or 0.0)
    node.clades = kids
    return node

class PrunedTree:
    """One posterior tree pruned to the mapped cids, with lookups for attachment queries."""

    def __init__(self, root: Clade) -> None:
        self.root = root
        self.parent: Dict[int, Clade] = {}
        self.depth: Dict[int, float] = {id(root): 0.0}
        self.tip_by_name: Dict[str, Clade] = {}

        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.clades:
                self.parent[id(child)] = node
                self.depth[id(child)] = self.depth[id(node)] + (child.branch_length or 0.0)
                stack.append(child)
            if node.is_terminal():
                self.tip_by_name[node.name] = node

        self.height = max(self.depth[id(tip)] for tip in self.tip_by_name.values())

    def tips_under(self, node: Clade, restrict: Set[str]) -> Set[str]:
        found = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.is_terminal():
                if n.name in restrict:
                    found.add(n.name)
            else:
                stack.extend(n.clades)
        return found

    def attachment(self, missing: str, ref: Set[str]) -> Optional[Tuple[frozenset, float]]:
        """
        Sister clade (restricted to `ref`) and attachment age of `missing` in this
        tree: walking rootward from the missing tip, the first ancestor with any
        ref tips below it is where `missing` joins the tree induced by `ref`.
        """
        node = self.tip_by_name.get(missing)
        if node is None:
            return None
        prev = node
        while id(node) in self.parent:
            node = self.parent[id(node)]
            sisters = set()
            for child in node.clades:
                if child is not prev:
                    sisters |= self.tips_under(child, ref)
            if sisters:
                age = self.height - self.depth[id(node)]
                return frozenset(sisters), age
            prev = node
        return None

_JETZ_KEEP: Optional[Set[str]] = None
_JETZ_TO_CID: Optional[Dict[str, str]] = None

# build_pruned_jetz_newicks() helper
def _jetz_worker_init(jetz_to_cid: Dict[str, str]) -> None:
    global _JETZ_KEEP, _JETZ_TO_CID
    _JETZ_TO_CID = jetz_to_cid
    _JETZ_KEEP = set(jetz_to_cid.keys())

# build_pruned_jetz_newicks() helper
def _prune_jetz_line(line: str) -> str:
    tree = Phylo.read(StringIO(line), "newick")
    root = induced_subtree(tree.root, _JETZ_KEEP)
    for tip in root.get_terminals():
        tip.name = _JETZ_TO_CID[tip.name]
    out = StringIO()
    # 17 significant digits round-trips doubles exactly
    Phylo.write(Tree(root=root), out, "newick", format_branch_length="%.17g")
    return out.getvalue()

def build_pruned_jetz_newicks(jetz_to_cid: Dict[str, str], n_workers: int = 8) -> List[str]:
    """
    Every posterior tree pruned to the mapped cids (tips renamed to cids), as
    newick strings, parsed in parallel across the posterior sample's lines.
    """
    with open(paths["raw_tree"]["cub_jetz"]) as f:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_jetz_worker_init, initargs=(jetz_to_cid,)
        ) as executor:
            return list(executor.map(_prune_jetz_line, f, chunksize=10))

def load_jetz_cache(cids: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    The cid->Jetz mapping and pruned posterior newicks, from the cached
    intermediary when it matches the current class set, else rebuilt (GBIF +
    full posterior parse) and re-cached.
    """
    if FPATH_JETZ_CACHE.is_file():
        with gzip.open(FPATH_JETZ_CACHE, "rb") as f:
            cache = pickle.load(f)
        if cache["cids"] == cids:
            return cache["cid_to_jetz"], cache["pruned_newicks"]

    print("  building Jetz cache (GBIF mapping + posterior pruning)...")
    cid_to_jetz = map_cids_to_jetz(cids, read_jetz_tip_names())
    pruned_newicks = build_pruned_jetz_newicks({jetz: cid for cid, jetz in cid_to_jetz.items()})
    FPATH_JETZ_CACHE.parent.mkdir(exist_ok=True)
    with gzip.open(FPATH_JETZ_CACHE, "wb") as f:
        pickle.dump({"cids": cids, "cid_to_jetz": cid_to_jetz, "pruned_newicks": pruned_newicks}, f)
    return cid_to_jetz, pruned_newicks

# graft_missing_from_jetz() helper
def backbone_maps(tree: Tree) -> Tuple[Dict[int, Clade], Dict[int, float], Dict[str, Clade]]:
    parent: Dict[int, Clade] = {}
    depth: Dict[int, float] = {id(tree.root): 0.0}
    tip_by_name: Dict[str, Clade] = {}
    stack = [tree.root]
    while stack:
        node = stack.pop()
        for child in node.clades:
            parent[id(child)] = node
            depth[id(child)] = depth[id(node)] + (child.branch_length or 0.0)
            stack.append(child)
        if node.is_terminal():
            tip_by_name[node.name] = node
    return parent, depth, tip_by_name

def graft_missing_from_jetz(
    tree: Tree,
    pruned_jetz: List[PrunedTree],
    cids_missing: List[str],
) -> None:
    """
    Sequentially attach each missing cid to the backbone as sister to its
    majority-rule attachment clade, splitting that clade's stem edge at the median
    attachment age. Already-inserted cids join the reference set, so groups of
    missing relatives (e.g. congeners) recover their own internal structure.
    """
    for cid in sorted(cids_missing):
        parent, depth, tip_by_name = backbone_maps(tree)
        ref = set(tip_by_name.keys())

        votes = Counter()
        ages: Dict[frozenset, List[float]] = {}
        for pt in pruned_jetz:
            att = pt.attachment(cid, ref)
            if att is None:
                continue
            sisters, age = att
            votes[sisters] += 1
            ages.setdefault(sisters, []).append(age)
        if not votes:
            print(f"WARNING: no Jetz attachment found for {cid}; leaving it to polytomy grafting")
            continue
        sisters, n_votes = votes.most_common(1)[0]
        age = median(ages[sisters])
        print(f"  {cid}: sister to {sorted(sisters)[:4]}{'...' if len(sisters) > 4 else ''} "
              f"({len(sisters)} tips, {n_votes}/{len(pruned_jetz)} trees, age {age:.1f})")

        if len(sisters) == 1:
            anchor = tip_by_name[next(iter(sisters))]
        else:
            anchor = tree.common_ancestor(sorted(sisters))
        anchor_parent = parent.get(id(anchor))

        sister_tip_depths = [depth[id(tip_by_name[name])] for name in sisters]
        present = sum(sister_tip_depths) / len(sister_tip_depths)
        depth_new = present - age

        if anchor_parent is None:
            # attachment above the root: new node becomes the root
            depth_new = min(depth_new, 0.0)
            new_node = Clade(branch_length=None)
            anchor.branch_length = 0.0 - depth_new
            tree.root = new_node
        else:
            # split the anchor's stem edge at depth_new (clamped into the edge)
            depth_new = min(max(depth_new, depth[id(anchor_parent)]), depth[id(anchor)])
            new_node = Clade(branch_length=depth_new - depth[id(anchor_parent)])
            anchor.branch_length = depth[id(anchor)] - depth_new
            anchor_parent.clades[anchor_parent.clades.index(anchor)] = new_node
        new_node.clades = [anchor, Clade(name=cid, branch_length=max(present - depth_new, 0.0))]

def main():
    print("Building CUB tree...")

    class_data = load_pickle(paths["metadata"]["cub"] / "class_data.pkl")
    tree = build_tree_cub(class_data)

    cids_missing = sorted(set(class_data.keys()) - {tip.name for tip in tree.get_terminals()})
    if cids_missing:
        print(f"Filling {len(cids_missing)} backbone gaps from the Jetz posterior sample...")
        cid_to_jetz, pruned_newicks = load_jetz_cache(sorted(class_data.keys()))
        unmapped = [cid for cid in cids_missing if cid not in cid_to_jetz]
        if unmapped:
            print(f"WARNING: no Jetz tip found for {unmapped}; leaving them to polytomy grafting")
        pruned_jetz = [PrunedTree(Phylo.read(StringIO(nwk), "newick").root) for nwk in pruned_newicks]
        graft_missing_from_jetz(tree, pruned_jetz, [cid for cid in cids_missing if cid in cid_to_jetz])

    tree_poly = augment_tree_with_polytomies(tree, class_data)
    tree_poly_pruned = prune_tree(tree_poly, class_data)

    save_pickle(tree, paths["metadata"]["cub"] / "tree_prepoly.pkl")
    save_pickle(tree_poly_pruned, paths["metadata"]["cub"] / "tree.pkl")
    print("CUB tree complete")

# main() helper
def read_jetz_tip_names() -> Set[str]:
    with open(paths["raw_tree"]["cub_jetz"]) as f:
        tree = Phylo.read(StringIO(f.readline()), "newick")
    return {tip.name for tip in tree.get_terminals()}


if __name__ == "__main__":
    main()
