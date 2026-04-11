"""
python -m pytest tests/unit/test_phylo.py

Tests for the phylo preprocessing pipeline: tree merge and taxonomic polytomy
fill.  Integration tests use real trees via the shared tree_bundle fixture;
unit tests for rehome_missing_species_in_represented_genera operate on small
synthetic trees.
"""

from io import StringIO
from itertools import combinations
import random

import pytest  # type: ignore[import]
from Bio import Phylo  # type: ignore[import]

from preprocessing.lepid.phylo import build_tree_lepid, combine_trees_lepid_nymph
from preprocessing.nymph.phylo import build_tree_nymph
from utils.utils import paths, load_pickle
from preprocessing.common.phylo import (
    augment_tree_with_polytomies,
    rehome_missing_species_in_represented_genera,
    prune_tree,
    augment_class_data,
)


SAMPLE_SIZE = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tree(newick: str):
    return Phylo.read(StringIO(newick), "newick")


def _sample(values: set[str], k: int) -> list[str]:
    vals = sorted(values)
    if len(vals) <= k:
        return vals
    return random.sample(vals, k)


# ---------------------------------------------------------------------------
# Fixtures — integration (real trees)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tree_bundle():

    tree_lepid = build_tree_lepid()
    tree_nymph = build_tree_nymph()

    class_data = load_pickle(paths["metadata"]["lepid"] / "class_data.pkl")
    # class data augmented with sids on trees not in class_data but with genera in class_data
    class_data_aug = augment_class_data(class_data, tree_lepid)
    class_data_aug = augment_class_data(class_data_aug, tree_nymph)

    tree_merge = combine_trees_lepid_nymph(tree_lepid, tree_nymph, class_data_aug)
    # capture pre-prune sids before prune_tree mutates tree_merge in-place
    sids_merge = {tip.name for tip in tree_merge.get_terminals()}
    tree_merge_pruned = prune_tree(tree_merge, class_data_aug)
    # tree_merge IS tree_merge_pruned (same object); capture post-prune sids for
    # distance-test sampling so we never query a tip that was pruned away
    sids_merge_pruned = {tip.name for tip in tree_merge.get_terminals()}

    tree_poly = augment_tree_with_polytomies(tree_merge_pruned, class_data_aug)
    tree_poly_pruned = prune_tree(tree_poly, class_data)

    sids_lepid = {tip.name for tip in tree_lepid.get_terminals()}
    sids_nymph = {tip.name for tip in tree_nymph.get_terminals()}

    sids_lepid_only = sids_lepid - sids_nymph
    sids_nymph_only = sids_nymph - sids_lepid
    sids_shared = sids_lepid & sids_nymph

    sids_lepid_cdnymph = {
        sid
        for sid in sids_lepid
        if class_data_aug.get(sid, {}).get("family", "") == "nymphalidae"
    }

    sids_nymph_only_cdnymph = {
        sid
        for sid in sids_nymph_only
        if class_data_aug.get(sid, {}).get("family", "") == "nymphalidae"
    }

    bundle = {
        "tree": {
            "lepid": tree_lepid,
            "nymph": tree_nymph,
            "merge": tree_merge,
            "merge_pruned": tree_merge_pruned,
            "poly": tree_poly,
            "poly_pruned": tree_poly_pruned,
        },
        "sids": {
            "lepid": sids_lepid,
            "nymph": sids_nymph,
            "merge": sids_merge,
            "nymph_only": sids_nymph_only,
            "nymph_only_cdnymph": sids_nymph_only_cdnymph,
            # intersect with post-prune merge tree so distance tests never query a pruned tip
            "lepid_only_non_cdnymph": (sids_lepid_only - sids_lepid_cdnymph) & sids_merge_pruned,
            "shared_cdnymph": sids_shared & sids_lepid_cdnymph,
        }
    }

    return bundle


# ---------------------------------------------------------------------------
# Fixtures — unit (synthetic trees)
# ---------------------------------------------------------------------------

@pytest.fixture()
def pyrisitia_tree():
    """
    Tiny ultrametric tree mimicking the Pyrisitia / Leucidia situation.

    Original tree (before augmentation):
        ((pyrisitia_lisa:14.69, leucidia_elvina:14.69):14.69,
          colias_palaeno:29.38);

    Post-augmentation (wrong placement at root, branch_length=0):
        ((pyrisitia_lisa:14.69, leucidia_elvina:14.69):14.69,
          colias_palaeno:29.38,
          pyrisitia_chamberlaini:0.0, pyrisitia_venusta:0.0);

    After rehome the three pairs should each be ~29.38:
        pyrisitia_chamberlaini <-> pyrisitia_lisa  ≈ 29.38
        pyrisitia_chamberlaini <-> pyrisitia_venusta ≈ 29.38
        pyrisitia_chamberlaini <-> leucidia_elvina  ≈ 29.38
    """
    newick = (
        "((pyrisitia_lisa:14.69, leucidia_elvina:14.69):14.69, "
        "colias_palaeno:29.38, "
        "pyrisitia_chamberlaini:0.0, pyrisitia_venusta:0.0);"
    )
    return _build_tree(newick)


@pytest.fixture()
def pyrisitia_taxonomy():
    return {
        "pyrisitia_lisa": {
            "family": "Pieridae",
            "subfamily": "Coliadinae",
            "tribe": "Coliadini",
            "genus": "pyrisitia",
        },
        "pyrisitia_chamberlaini": {
            "family": "Pieridae",
            "subfamily": "Coliadinae",
            "tribe": "Coliadini",
            "genus": "pyrisitia",
        },
        "pyrisitia_venusta": {
            "family": "Pieridae",
            "subfamily": "Coliadinae",
            "tribe": "Coliadini",
            "genus": "pyrisitia",
        },
        "leucidia_elvina": {
            "family": "Pieridae",
            "subfamily": "Coliadinae",
            "tribe": "Coliadini",
            "genus": "leucidia",
        },
        "colias_palaeno": {
            "family": "Pieridae",
            "subfamily": "Coliadinae",
            "tribe": "Coliadini",
            "genus": "colias",
        },
    }


# ---------------------------------------------------------------------------
# Integration tests: tree merge
# ---------------------------------------------------------------------------

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
    Colias taxa that appear only in the Nymph tree but whose genus already exists on the
    retained Lepid backbone (colias_palaeno) must be rehomed to the Colias-vs-Zerene
    divergence anchor instead of inheriting the broad Nymph merge anchor.
    """
    tree_merge = tree_bundle["tree"]["merge"]

    sid_nearest = "zerene_cesonia"
    dist_colias_nearest = tree_merge.distance("colias_palaeno", sid_nearest)

    assert abs(tree_merge.distance("colias_palaeno", "colias_croceus") - dist_colias_nearest) <= 1e-4
    assert abs(tree_merge.distance("colias_palaeno", "colias_hyale") - dist_colias_nearest) <= 1e-4
    assert tree_merge.distance("colias_palaeno", "pyrisitia_lisa") == pytest.approx(
        tree_merge.distance("colias_croceus", "pyrisitia_lisa")
    )

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
    Nymph-only Nymphalidae taxa remain inside the intact inserted Nymph subtree, so their
    pairwise distances must be exactly preserved in the merged tree.
    """
    tree_nymph = tree_bundle["tree"]["nymph"]
    tree_merge = tree_bundle["tree"]["merge"]

    sids_nymph_only_sample = _sample(tree_bundle["sids"]["nymph_only_cdnymph"], SAMPLE_SIZE)
    for sid_a, sid_b in combinations(sids_nymph_only_sample, 2):
        dist_nymph = tree_nymph.distance(sid_a, sid_b)
        dist_merge = tree_merge.distance(sid_a, sid_b)
        assert dist_merge == pytest.approx(dist_nymph), (
            f"Nymph-only pairwise distance changed for {sid_a} vs {sid_b}: "
            f"nymph={dist_nymph}, merged={dist_merge}"
        )

def test_merge_preserves_dists_nymph_only_and_shared(tree_bundle) -> None:
    """
    Cross-pair distances between Nymph-only Nymphalidae taxa and shared (Lepid ∩ Nymph)
    Nymphalidae taxa are preserved exactly because that clade is inserted without scaling.
    Checks every pair in the cross-product of sampled nymph_only_cdnymph x shared_cdnymph.
    """
    sids_nymph_only_sample = _sample(tree_bundle["sids"]["nymph_only_cdnymph"], SAMPLE_SIZE)
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


# ---------------------------------------------------------------------------
# Integration tests: poly tree
# ---------------------------------------------------------------------------

def test_poly_ultrametric(tree_bundle) -> None:
    """
    Test that all tips of the poly-pruned tree are equidistant from root within
    a small tolerance.
    """
    tree_poly_pruned = tree_bundle["tree"]["poly_pruned"]

    tips = tree_poly_pruned.get_terminals()
    root = tree_poly_pruned.root
    distances = [tree_poly_pruned.distance(root, tip) for tip in tips]

    min_dist = min(distances)
    max_dist = max(distances)

    assert (max_dist - min_dist) <= 1e-4, (
        "Poly-pruned tree is not ultrametric within tolerance: "
        f"min={min_dist}, max={max_dist}"
    )

def test_poly_tree_depth(tree_bundle) -> None:
    """
    Test that depth of the poly-pruned tree is approximately the same as depth
    of the Lepid tree.
    """
    tree_lepid = tree_bundle["tree"]["lepid"]
    tree_poly_pruned = tree_bundle["tree"]["poly_pruned"]

    lepid_tips = tree_lepid.get_terminals()
    poly_tips = tree_poly_pruned.get_terminals()

    depth_lepid = max(tree_lepid.distance(tree_lepid.root, tip) for tip in lepid_tips)
    depth_poly = max(tree_poly_pruned.distance(tree_poly_pruned.root, tip) for tip in poly_tips)

    assert depth_poly == pytest.approx(depth_lepid), (
        "Poly-pruned tree depth changed relative to Lepid tree: "
        f"lepid={depth_lepid}, poly_pruned={depth_poly}"
    )


# ---------------------------------------------------------------------------
# Unit tests: rehome_missing_species_in_represented_genera
# ---------------------------------------------------------------------------

class TestPolytomyRehomeMissingSpeciesInRepresentedGenera:

    def test_poly_missing_pyrisitia_get_correct_distances(
        self, pyrisitia_tree, pyrisitia_taxonomy
    ):
        """
        Missing Pyrisitia species should be rehomed to the Pyrisitia/Leucidia
        divergence anchor (not the root), giving ~29.38 distances to all
        congeners and to sibling-tribe leucidia.

        This is a regression test for the bug where missing congeners were
        assigned 0-length branches to the root, resulting in wrong distances
        (~19.95 instead of ~29.38).
        """
        represented_original = {"pyrisitia_lisa", "leucidia_elvina", "colias_palaeno"}
        missing_species = {"pyrisitia_chamberlaini", "pyrisitia_venusta"}
        expected = 29.38

        rehome_missing_species_in_represented_genera(
            tree=pyrisitia_tree,
            missing_species=missing_species,
            represented_original=represented_original,
            taxonomy=pyrisitia_taxonomy,
        )

        pairs = [
            ("pyrisitia_chamberlaini", "pyrisitia_lisa"),
            ("pyrisitia_chamberlaini", "pyrisitia_venusta"),
            ("pyrisitia_chamberlaini", "leucidia_elvina"),
            ("pyrisitia_lisa", "leucidia_elvina"),
        ]
        for a, b in pairs:
            d = pyrisitia_tree.distance(a, b)
            assert abs(d - expected) < 0.01, (
                f"{a} <-> {b}: got {d:.4f}, expected ~{expected}"
            )

    def test_poly_noop_when_no_missing_species(self, pyrisitia_tree, pyrisitia_taxonomy):
        """
        When missing_species is empty the tree should be unchanged.
        """
        leaves_before = sorted(
            l.name for l in pyrisitia_tree.get_terminals() if l.name
        )
        rehome_missing_species_in_represented_genera(
            tree=pyrisitia_tree,
            missing_species=set(),
            represented_original={"pyrisitia_lisa", "leucidia_elvina", "colias_palaeno"},
            taxonomy=pyrisitia_taxonomy,
        )
        leaves_after = sorted(
            l.name for l in pyrisitia_tree.get_terminals() if l.name
        )
        assert leaves_before == leaves_after

    def test_poly_rehomes_genus_with_single_missing_species(self, pyrisitia_taxonomy):
        """
        A genus with a single missing species should still be rehomed to the
        nearest inter-genus divergence anchor.
        """
        newick = (
            "((pyrisitia_lisa:14.69, leucidia_elvina:14.69):14.69, "
            "colias_palaeno:29.38, "
            "pyrisitia_chamberlaini:0.0);"
        )
        tree = _build_tree(newick)

        rehome_missing_species_in_represented_genera(
            tree=tree,
            missing_species={"pyrisitia_chamberlaini"},
            represented_original={"pyrisitia_lisa", "leucidia_elvina", "colias_palaeno"},
            taxonomy=pyrisitia_taxonomy,
        )

        d_after = tree.distance("pyrisitia_chamberlaini", "pyrisitia_lisa")
        assert abs(d_after - 29.38) < 0.01

    def test_poly_family_fallback_for_unrepresented_genus(self):
        """
        If a missing genus has no original-tree representative, species in that
        genus should be rehomed using the most recent inter-family divergence.
        """
        # Represented on original tree: genus1_a, genus1_b, genus2_d, genus4_i
        # Missing: genus1_c, genus2_e/genus2_f, genus3_g/genus3_h
        newick = (
            "(((genus1_a:1,genus1_b:1):1,genus2_d:2):2,genus4_i:4,"
            "genus1_c:0,genus2_e:0,genus2_f:0,genus3_g:0,genus3_h:0);"
        )
        tree = _build_tree(newick)
        taxonomy = {
            "genus1_a": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus1"},
            "genus1_b": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus1"},
            "genus1_c": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus1"},
            "genus2_d": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus2"},
            "genus2_e": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus2"},
            "genus2_f": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus2"},
            "genus3_g": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus3"},
            "genus3_h": {"family": "family1", "subfamily": None, "tribe": None, "genus": "genus3"},
            "genus4_i": {"family": "family2", "subfamily": None, "tribe": None, "genus": "genus4"},
        }

        rehome_missing_species_in_represented_genera(
            tree=tree,
            missing_species={"genus1_c", "genus2_e", "genus2_f", "genus3_g", "genus3_h"},
            represented_original={"genus1_a", "genus1_b", "genus2_d", "genus4_i"},
            taxonomy=taxonomy,
        )

        # Genus1 and genus2 rehome to their inter-genus divergence anchor.
        assert tree.distance("genus1_c", "genus1_a") == pytest.approx(tree.distance("genus1_b", "genus2_d"))
        assert tree.distance("genus1_c", "genus2_d") == pytest.approx(tree.distance("genus1_b", "genus2_d"))

        # Genus3 (no representative) falls back to family-level divergence.
        assert tree.distance("genus3_g", "genus3_h") == pytest.approx(tree.distance("genus1_a", "genus4_i"))
