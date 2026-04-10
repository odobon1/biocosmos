"""
python -m pytest tests/unit/test_phylo_merge.py

All of this is from within the context of only the merge being performed (no pruning with class_data_aug)
"""

from itertools import combinations
import random
import pytest  # type: ignore[import]

from preprocessing.lepid.phylo import build_tree_lepid, combine_trees_lepid_nymph
from preprocessing.nymph.phylo import build_tree_nymph
from utils.utils import paths, load_pickle
from preprocessing.lepid.phylo import augment_class_data


SAMPLE_SIZE = 50


@pytest.fixture(scope="module")
def tree_bundle():
    tree_lepid = build_tree_lepid()
    tree_nymph = build_tree_nymph()
    class_data = load_pickle(paths["metadata"]["lepid"] / "class_data.pkl")
    # class data augmented with sids on trees not in class_data but with genera in class_data
    class_data_aug = augment_class_data(class_data, tree_lepid)
    class_data_aug = augment_class_data(class_data_aug, tree_nymph)
    tree_merge = combine_trees_lepid_nymph(tree_lepid, tree_nymph, class_data_aug)

    sids_lepid = {tip.name for tip in tree_lepid.get_terminals()}
    sids_nymph = {tip.name for tip in tree_nymph.get_terminals()}
    sids_merge = {tip.name for tip in tree_merge.get_terminals()}

    sids_lepid_only = sids_lepid - sids_nymph
    sids_nymph_only = sids_nymph - sids_lepid
    sids_shared = sids_lepid & sids_nymph

    sids_lepid_cdnymph = {
        sid
        for sid in sids_lepid
        if class_data_aug.get(sid, {}).get("family", "") == "nymphalidae"
    }

    bundle = {
        "tree": {
            "lepid": tree_lepid,
            "nymph": tree_nymph,
            "merge": tree_merge,
        },
        "sids": {
            "lepid": sids_lepid,
            "nymph": sids_nymph,
            "merge": sids_merge,
            "nymph_only": sids_nymph_only,
            "lepid_only_non_cdnymph": sids_lepid_only - sids_lepid_cdnymph,
            "shared_cdnymph": sids_shared & sids_lepid_cdnymph,
        }
    }

    return bundle

def test_merge_preserves_all_terminal_taxa(tree_bundle) -> None:
    """
    Test that all sids from both input trees are present in the merged tree.
    """
    sids_lepid = tree_bundle["sids"]["lepid"]
    sids_nymph = tree_bundle["sids"]["nymph"]
    sids_merge = tree_bundle["sids"]["merge"]

    sids_lepid_missing_in_merge = sids_lepid - sids_merge
    sids_nymph_missing_in_merge = sids_nymph - sids_merge

    assert not sids_lepid_missing_in_merge, (
        "Merged tree is missing Lepid terminals: "
        f"{sorted(sids_lepid_missing_in_merge)[:10]}"
    )
    assert not sids_nymph_missing_in_merge, (
        "Merged tree is missing Nymph terminals: "
        f"{sorted(sids_nymph_missing_in_merge)[:10]}"
    )

def test_merge_ultrametric(tree_bundle) -> None:
    """
    Test that all tips of merged tree are equidistant from root within a small tolerance
    """
    tree_merge = tree_bundle["tree"]["merge"]

    tips = tree_merge.get_terminals()
    root = tree_merge.root
    distances = [tree_merge.distance(root, tip) for tip in tips]

    min_dist = min(distances)
    max_dist = max(distances)

    assert (max_dist - min_dist) <= 1e-4, (
        "Merged tree is not ultrametric within tolerance: "
        f"min={min_dist}, max={max_dist}"
    )

def test_merge_tree_depth(tree_bundle) -> None:
    """
    Test that depth of merged tree is approximately the same as depth of Lepid tree.
    """
    tree_lepid = tree_bundle["tree"]["lepid"]
    tree_merge = tree_bundle["tree"]["merge"]

    lepid_tips = tree_lepid.get_terminals()
    merge_tips = tree_merge.get_terminals()

    depth_lepid = max(tree_lepid.distance(tree_lepid.root, tip) for tip in lepid_tips)
    depth_merge = max(tree_merge.distance(tree_merge.root, tip) for tip in merge_tips)

    assert depth_merge == pytest.approx(depth_lepid), (
        "Merged tree depth changed relative to Lepid tree: "
        f"lepid={depth_lepid}, merged={depth_merge}"
    )

def _sample(values: set[str], k: int) -> list[str]:
    vals = sorted(values)
    if len(vals) <= k:
        return vals
    return random.sample(vals, k)

def test_merge_preserves_dists_lepid_non_nymph_backbone(tree_bundle) -> None:
    """
    All non-Nymphalidae Lepid-only taxa lie entirely outside the replaced Nymphalidae subtree,
    so their backbone distances must be identical between the Lepid source tree and the merged
    tree. Samples SAMPLE_SIZE taxa from that partition and checks every pair.
    """
    tree_lepid = tree_bundle["tree"]["lepid"]
    tree_merge = tree_bundle["tree"]["merge"]
    sids_lepid_only_non_cdnymph_sample = _sample(tree_bundle["sids"]["lepid_only_non_cdnymph"], SAMPLE_SIZE)

    for sid_a, sid_b in combinations(sids_lepid_only_non_cdnymph_sample, 2):
        dist_lepid = tree_lepid.distance(sid_a, sid_b)
        dist_merge = tree_merge.distance(sid_a, sid_b)

        assert dist_merge == pytest.approx(dist_lepid), (
            f"Pairwise distance changed for {sid_a} vs {sid_b}: "
            f"lepid={dist_lepid}, merged={dist_merge}"
        )

def test_merge_preserves_dists_nymph(tree_bundle) -> None:
    """
    The Nymph subtree is inserted without any branch-length scaling: the Nymph root is
    attached at depth T - h_nymph so that every Nymph tip reaches exactly depth T.
    Therefore all pairwise distances between shared (Lepid ∩ Nymph) Nymphalidae taxa
    must be identical in the Nymph source tree and the merged tree.
    """
    tree_nymph = tree_bundle["tree"]["nymph"]
    tree_merge = tree_bundle["tree"]["merge"]
    sids_shared_cdnymph_sample = _sample(tree_bundle["sids"]["shared_cdnymph"], SAMPLE_SIZE)

    for sid_a, sid_b in combinations(sids_shared_cdnymph_sample, 2):
        dist_nymph = tree_nymph.distance(sid_a, sid_b)
        dist_merge = tree_merge.distance(sid_a, sid_b)
        assert dist_merge == pytest.approx(dist_nymph), (
            f"Nymph pairwise distance changed for {sid_a} vs {sid_b}: "
            f"nymph={dist_nymph}, merged={dist_merge}"
        )

def test_merge_polytomy(tree_bundle) -> None:
    """
    Colias taxa that appear only in the Nymph tree (colias_palaeno) have no congeneric
    entry in the Lepid tree, so they are grafted via the nearest-divergence heuristic.
    The nearest Lepid anchor for the Colias genus is Zerene; all Nymph-only Colias tips
    should be attached as a polytomy at that same divergence distance.
    """
    tree_merge = tree_bundle["tree"]["merge"]

    sid_nearest = "zerene_cesonia"
    dist_colias_nearest = tree_merge.distance("colias_palaeno", sid_nearest)

    assert abs(tree_merge.distance("colias_palaeno", "colias_croceus") - dist_colias_nearest) <= 1e-4
    assert abs(tree_merge.distance("colias_palaeno", "colias_hyale") - dist_colias_nearest) <= 1e-4

def test_merge_preserves_all_lepid_only_non_nymph_tips(tree_bundle) -> None:
    """
    Every Lepid terminal that is not in the Nymph tree and not classified as Nymphalidae
    lies outside the replaced subtree and must survive verbatim in the merged tree. This
    catches over-pruning regressions from the MRCA replacement step.
    """
    sids_merge = tree_bundle["sids"]["merge"]
    sids_lepid_only_non_nymph = tree_bundle["sids"]["lepid_only_non_cdnymph"]
    missing = sids_lepid_only_non_nymph - sids_merge
    assert not missing, f"Merged tree dropped non-Nymph Lepid-only tips: {sorted(missing)[:10]}"

def test_merge_preserves_dists_lepid_only_non_nymph(tree_bundle) -> None:
    """
    Pairwise distances between non-Nymphalidae Lepid-only taxa must be identical in the
    Lepid source tree and the merged tree. Unlike test_merge_preserves_dists_lepid_non_nymph_backbone
    this test samples from the full lepid_only_non_nymph partition rather than the backbone
    subset, providing broader coverage.
    """
    tree_lepid = tree_bundle["tree"]["lepid"]
    tree_merge = tree_bundle["tree"]["merge"]

    sids_lepid_only_non_cdnymph_sample = _sample(tree_bundle["sids"]["lepid_only_non_cdnymph"], SAMPLE_SIZE)
    for sid_a, sid_b in combinations(sids_lepid_only_non_cdnymph_sample, 2):
        dist_lepid = tree_lepid.distance(sid_a, sid_b)
        dist_merge = tree_merge.distance(sid_a, sid_b)

        assert dist_merge == pytest.approx(dist_lepid), (
            f"Pairwise distance changed for {sid_a} vs {sid_b}: "
            f"lepid={dist_lepid}, merged={dist_merge}"
        )

def test_merge_preserves_dists_nymph_only(tree_bundle) -> None:
    """
    Nymph-only taxa (present in Nymph tree but absent from Lepid) are part of the Nymph
    subtree inserted without scaling. Their pairwise distances must therefore be exactly
    preserved in the merged tree.
    """
    tree_nymph = tree_bundle["tree"]["nymph"]
    tree_merge = tree_bundle["tree"]["merge"]

    sids_nymph_only_sample = _sample(tree_bundle["sids"]["nymph_only"], SAMPLE_SIZE)
    for sid_a, sid_b in combinations(sids_nymph_only_sample, 2):
        dist_nymph = tree_nymph.distance(sid_a, sid_b)
        dist_merge = tree_merge.distance(sid_a, sid_b)
        assert dist_merge == pytest.approx(dist_nymph), (
            f"Nymph-only pairwise distance changed for {sid_a} vs {sid_b}: "
            f"nymph={dist_nymph}, merged={dist_merge}"
        )

def test_merge_preserves_dists_nymph_only_and_shared(tree_bundle) -> None:
    """
    Cross-pair distances between Nymph-only taxa and shared (Lepid ∩ Nymph) Nymphalidae
    taxa are preserved exactly because the entire Nymph subtree is inserted without scaling.
    Checks every pair in the cross-product of sampled Nymph-only x shared_cdnymph.
    """
    sids_nymph_only_sample = _sample(tree_bundle["sids"]["nymph_only"], SAMPLE_SIZE)
    sids_shared_cdnymph_sample = _sample(tree_bundle["sids"]["shared_cdnymph"], SAMPLE_SIZE)
    tree_nymph = tree_bundle["tree"]["nymph"]
    tree_merge = tree_bundle["tree"]["merge"]
    for sid_a in sids_nymph_only_sample:
        for sid_b in sids_shared_cdnymph_sample:
            dist_nymph = tree_nymph.distance(sid_a, sid_b)
            dist_merge = tree_merge.distance(sid_a, sid_b)
            assert dist_merge == pytest.approx(dist_nymph), (
                f"Nymph cross-pair distance changed for {sid_a} vs {sid_b}: "
                f"nymph={dist_nymph}, merged={dist_merge}"
            )
