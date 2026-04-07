from itertools import combinations
import pytest  # type: ignore[import]

from preprocessing.lepid.phylo import build_tree_lepid
from preprocessing.lepid.phylo_merge import combine_trees_lepid_nymph
from preprocessing.nymph.phylo import build_tree_nymph


# sids that are in Nymph family dir (class_data.pkl) with genera on Lepid tree but not on Nymph tree
PAIRWISE_DISTANCE_SIDS_LEPID = [
    'coenyra_hebe',
    'mandarinia_regalis',
    'dryadula_phaetusa',
    'drucina_leonata',
    'praepronophila_emma',
    'lamasia_lyncides',
    'corderopedaliodes_corderoi',
    'bletogona_mycalesis',
    'orinoma_damaris',
    'pherepedaliodes_pheretiades',
    'sasakia_charonda',
    'thaleropis_ionia',
    'ptychandra_lorquinii',
    'pandita_sinope',
    'daedalma_dinias',
    'neomaenas_servilia',
    'eretris_porphyria',
    'laparus_doris',
    'patsuia_sinensium',
    'podotricha_euchroia',
    'miriamica_weiskei',
    'physcaeneura_panda',
    'callarge_sagitta'
 ]

# sids on Nymph tree but not Lepid tree
PAIRWISE_DISTANCE_SIDS_NYMPH = [
    'aglais_caschmirensis',
    'aglais_urticae',
    'aglais_milberti',
]

PAIRWISE_DISTANCE_SIDS_CROSS_FAMILY = [
    "macrosoma_rubedinaria",
    "macrosoma_tipulata",
    "abantis_ja",
    "abraximorpha_davidii",
    "achlyodes_busirus",
    "acleros_ploetzi",
    "adlerodea_petrovna",
    "adopaeoides_prittwitzi",
    "aeromachus_stigmata",
    "aethilla_lavochrea",
    "agathymus_aryxna",
    "agathymus_mariae",
    "thisbe_irenea",
    "thisbe_lycorias",
    "zabuella_tenellus",
    "zemeros_flegyas",
]

@pytest.fixture(scope="module")
def merged_tree_bundle():
    tree_lepid = build_tree_lepid()
    tree_nymph = build_tree_nymph()
    tree_merged = combine_trees_lepid_nymph(tree_lepid, tree_nymph)
    return tree_lepid, tree_nymph, tree_merged


def test_merge_preserves_all_terminal_taxa(merged_tree_bundle) -> None:
    tree_lepid, tree_nymph, tree_merged = merged_tree_bundle

    sids_lepid = {tip.name for tip in tree_lepid.get_terminals()}
    sids_nymph = {tip.name for tip in tree_nymph.get_terminals()}
    sids_merged = {tip.name for tip in tree_merged.get_terminals()}

    missing_from_merged_vs_lepid = sids_lepid - sids_merged
    missing_from_merged_vs_nymph = sids_nymph - sids_merged

    assert not missing_from_merged_vs_lepid, (
        "Merged tree is missing Lepid terminals: "
        f"{sorted(missing_from_merged_vs_lepid)[:10]}"
    )
    assert not missing_from_merged_vs_nymph, (
        "Merged tree is missing Nymph terminals: "
        f"{sorted(missing_from_merged_vs_nymph)[:10]}"
    )


def test_merged_tree_tips_same_distance_from_root(merged_tree_bundle) -> None:
    _, _, tree_merged = merged_tree_bundle

    tips = tree_merged.get_terminals()
    assert tips, "Merged tree has no terminal taxa."

    root = tree_merged.root
    distances = [tree_merged.distance(root, tip) for tip in tips]

    min_dist = min(distances)
    max_dist = max(distances)

    assert (max_dist - min_dist) <= 1e-4, (
        "Merged tree is not ultrametric within tolerance: "
        f"min={min_dist}, max={max_dist}"
    )


def test_merge_preserves_tree_depth(merged_tree_bundle) -> None:
    tree_lepid, _, tree_merged = merged_tree_bundle

    lepid_tips = tree_lepid.get_terminals()
    merged_tips = tree_merged.get_terminals()

    assert lepid_tips, "Lepid tree has no terminal taxa."
    assert merged_tips, "Merged tree has no terminal taxa."

    depth_lepid = max(tree_lepid.distance(tree_lepid.root, tip) for tip in lepid_tips)
    depth_merged = max(tree_merged.distance(tree_merged.root, tip) for tip in merged_tips)

    assert depth_merged == pytest.approx(depth_lepid), (
        "Merged tree depth changed relative to Lepid tree: "
        f"lepid={depth_lepid}, merged={depth_merged}"
    )


def test_merge_preserves_pairwise_distances_lepid(
    merged_tree_bundle,
) -> None:
    tree_lepid, _, tree_merged = merged_tree_bundle

    sids_lepid = {tip.name for tip in tree_lepid.get_terminals()}
    sids_merged = {tip.name for tip in tree_merged.get_terminals()}

    missing_lepid = sorted(set(PAIRWISE_DISTANCE_SIDS_LEPID) - sids_lepid)
    missing_merged = sorted(set(PAIRWISE_DISTANCE_SIDS_LEPID) - sids_merged)

    assert not missing_lepid, f"Specified SIDs missing from Lepid tree: {missing_lepid}"
    assert not missing_merged, f"Specified SIDs missing from merged tree: {missing_merged}"

    for sid_a, sid_b in combinations(PAIRWISE_DISTANCE_SIDS_LEPID, 2):
        dist_lepid = tree_lepid.distance(sid_a, sid_b)
        dist_merged = tree_merged.distance(sid_a, sid_b)

        assert dist_merged == pytest.approx(dist_lepid), (
            f"Pairwise distance changed for {sid_a} vs {sid_b}: "
            f"lepid={dist_lepid}, merged={dist_merged}"
        )


def test_merge_preserves_pairwise_distances_nymph(
    merged_tree_bundle,
) -> None:
    _, tree_nymph, tree_merged = merged_tree_bundle

    sids_nymph = {tip.name for tip in tree_nymph.get_terminals()}
    sids_merged = {tip.name for tip in tree_merged.get_terminals()}

    missing_nymph = sorted(set(PAIRWISE_DISTANCE_SIDS_NYMPH) - sids_nymph)
    missing_merged = sorted(set(PAIRWISE_DISTANCE_SIDS_NYMPH) - sids_merged)

    assert not missing_nymph, f"Specified SIDs missing from Nymph tree: {missing_nymph}"
    assert not missing_merged, f"Specified SIDs missing from merged tree: {missing_merged}"

    for sid_a, sid_b in combinations(PAIRWISE_DISTANCE_SIDS_NYMPH, 2):
        dist_nymph = tree_nymph.distance(sid_a, sid_b)
        dist_merged = tree_merged.distance(sid_a, sid_b)

        assert dist_merged == pytest.approx(dist_nymph), (
            f"Pairwise distance changed for {sid_a} vs {sid_b}: "
            f"nymph={dist_nymph}, merged={dist_merged}"
        )


def test_merge_preserves_pairwise_distances_cross_family(
    merged_tree_bundle,
) -> None:
    tree_lepid, _, tree_merged = merged_tree_bundle

    sids_lepid = {tip.name for tip in tree_lepid.get_terminals()}
    sids_merged = {tip.name for tip in tree_merged.get_terminals()}

    missing_lepid = sorted(set(PAIRWISE_DISTANCE_SIDS_CROSS_FAMILY) - sids_lepid)
    missing_merged = sorted(set(PAIRWISE_DISTANCE_SIDS_CROSS_FAMILY) - sids_merged)

    assert not missing_lepid, f"Specified SIDs missing from Lepid tree: {missing_lepid}"
    assert not missing_merged, f"Specified SIDs missing from merged tree: {missing_merged}"

    for sid_a, sid_b in combinations(PAIRWISE_DISTANCE_SIDS_CROSS_FAMILY, 2):
        dist_lepid = tree_lepid.distance(sid_a, sid_b)
        dist_merged = tree_merged.distance(sid_a, sid_b)

        assert dist_merged == pytest.approx(dist_lepid), (
            f"Pairwise distance changed for {sid_a} vs {sid_b}: "
            f"lepid={dist_lepid}, merged={dist_merged}"
        )
