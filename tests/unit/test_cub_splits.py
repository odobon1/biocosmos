"""
conda run -n biocosmos_b200 python -m pytest tests/unit/test_cub_splits.py
"""

from types import SimpleNamespace

import pytest

from preprocessing.cub.splits import (
    _build_classdir_to_cid,
    _build_img_ptrs,
    _build_skeys_from_rfpaths,
    _choose_ood_val_sids,
    _normalize_cub_rfpath,
    _split_train_into_train_idval_oodval,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_class_data(*entries):
    """Each entry: (cid, class_dir_name, species_sid)."""
    return {
        cid: {"rdpath_imgs": f"images/{class_dir}", "species": sid}
        for cid, class_dir, sid in entries
    }


def _rfpaths_for(class_dir, n):
    return [f"{class_dir}/img_{i:04d}.jpg" for i in range(n)]


def _make_class_data_and_rfpaths(entries):
    """
    entries: list of (class_dir, species_sid, n_imgs)
    Returns (class_data, flat rfpaths list).
    """
    class_data = {}
    rfpaths = []
    for idx, (class_dir, sid, n) in enumerate(entries):
        class_data[f"cid_{idx}"] = {"rdpath_imgs": f"images/{class_dir}", "species": sid}
        rfpaths += _rfpaths_for(class_dir, n)
    return class_data, rfpaths


def _uniform_pool(n_sids, samps_per_sid):
    """Uniform train pool: n_sids species × samps_per_sid samples each."""
    return {
        (f"sid_{i:03d}", j)
        for i in range(n_sids)
        for j in range(samps_per_sid)
    }


def _cfg(pct_partition=0.1, pct_ood_tol=0.005, seed=42):
    return SimpleNamespace(seed=seed, pct_partition=pct_partition, pct_ood_tol=pct_ood_tol)


# ─── _normalize_cub_rfpath ───────────────────────────────────────────────────

def test_normalize_cub_rfpath_raises_on_already_clean_path_without_images_prefix():
    with pytest.raises(ValueError, match="Could not parse CUB rfpath"):
        _normalize_cub_rfpath("001.Black_footed_Albatross/img_001.jpg")


def test_normalize_cub_rfpath_strips_leading_prefix():
    raw = "/some/long/prefix/images/001.Black_footed_Albatross/img_001.jpg"
    assert _normalize_cub_rfpath(raw) == "001.Black_footed_Albatross/img_001.jpg"


def test_normalize_cub_rfpath_raises_when_images_segment_is_absent():
    with pytest.raises(ValueError, match="Could not parse CUB rfpath"):
        _normalize_cub_rfpath("/no/IMGS/prefix/here.jpg")


# ─── _build_classdir_to_cid ──────────────────────────────────────────────────

def test_build_classdir_to_cid_maps_each_class_dir_to_its_cid():
    class_data = _make_class_data(
        ("albatross", "001.Black_footed_Albatross", "diomedea_nigripes"),
        ("laysan", "002.Laysan_Albatross", "phoebastria_immutabilis"),
    )
    result = _build_classdir_to_cid(class_data)

    assert result["001.Black_footed_Albatross"] == "albatross"
    assert result["002.Laysan_Albatross"] == "laysan"
    assert len(result) == 2


def test_build_classdir_to_cid_raises_on_rdpath_without_images_prefix():
    class_data = {
        "bad_cid": {"rdpath_imgs": "no_images_prefix/class_dir", "species": "some_sp"}
    }
    with pytest.raises(ValueError, match="Invalid rdpath_imgs"):
        _build_classdir_to_cid(class_data)


# ─── _build_img_ptrs ─────────────────────────────────────────────────────────

def test_build_img_ptrs_assigns_sequential_samp_idxs_per_species():
    class_data, rfpaths = _make_class_data_and_rfpaths([
        ("001.Albatross", "diomedea_nigripes", 3),
        ("002.Laysan", "phoebastria_immutabilis", 2),
    ])
    _, sid_2_samp_idxs, _, _, n_samps_dict = _build_img_ptrs(rfpaths, class_data)

    assert sid_2_samp_idxs["diomedea_nigripes"] == [0, 1, 2]
    assert sid_2_samp_idxs["phoebastria_immutabilis"] == [0, 1]
    assert n_samps_dict == {"diomedea_nigripes": 3, "phoebastria_immutabilis": 2}


def test_build_img_ptrs_rfpath_2_skey_covers_all_rfpaths():
    class_data, rfpaths = _make_class_data_and_rfpaths([
        ("001.Albatross", "diomedea_nigripes", 2),
        ("002.Laysan", "phoebastria_immutabilis", 3),
    ])
    _, _, rfpath_2_skey, sids, _ = _build_img_ptrs(rfpaths, class_data)

    assert set(rfpath_2_skey.keys()) == set(rfpaths)
    assert set(sids) == {"diomedea_nigripes", "phoebastria_immutabilis"}


def test_build_img_ptrs_skey_values_are_unique():
    class_data, rfpaths = _make_class_data_and_rfpaths([
        ("001.Albatross", "diomedea_nigripes", 4),
        ("002.Laysan", "phoebastria_immutabilis", 4),
    ])
    _, _, rfpath_2_skey, _, _ = _build_img_ptrs(rfpaths, class_data)

    skeys = list(rfpath_2_skey.values())
    assert len(skeys) == len(set(skeys))


def test_build_img_ptrs_raises_on_bad_rfpath_format():
    class_data = _make_class_data(("cid", "001.Albatross", "some_species"))
    bad_rfpaths = ["only_one_part.jpg"]  # only 1 path segment → ValueError

    with pytest.raises(ValueError, match="Unexpected CUB rfpath format"):
        _build_img_ptrs(bad_rfpaths, class_data)


def test_build_img_ptrs_raises_on_unknown_class_dir():
    class_data = _make_class_data(("cid", "001.Albatross", "some_species"))
    rfpaths = ["999.UnknownClass/img_0001.jpg"]

    with pytest.raises(KeyError, match="missing from class_data mapping"):
        _build_img_ptrs(rfpaths, class_data)


# ─── _build_skeys_from_rfpaths ───────────────────────────────────────────────

def test_build_skeys_from_rfpaths_returns_correct_skeys():
    rfpath_2_skey = {
        "001.Cls/img_0.jpg": ("sid_a", 0),
        "001.Cls/img_1.jpg": ("sid_a", 1),
    }
    result = _build_skeys_from_rfpaths(list(rfpath_2_skey.keys()), rfpath_2_skey, "test")

    assert result == {("sid_a", 0), ("sid_a", 1)}


def test_build_skeys_from_rfpaths_raises_on_unknown_rfpath():
    rfpath_2_skey = {"001.Cls/img_0.jpg": ("sid_a", 0)}

    with pytest.raises(KeyError, match="not found in global lookup"):
        _build_skeys_from_rfpaths(["MISSING/img.jpg"], rfpath_2_skey, "test")


def test_build_skeys_from_rfpaths_raises_on_duplicate_rfpaths():
    rfpath_2_skey = {"001.Cls/img_0.jpg": ("sid_a", 0)}

    with pytest.raises(ValueError, match="Duplicate rfpaths detected"):
        _build_skeys_from_rfpaths(
            ["001.Cls/img_0.jpg", "001.Cls/img_0.jpg"],
            rfpath_2_skey,
            "test",
        )


# ─── _choose_ood_val_sids ────────────────────────────────────────────────────

def _uniform_sids_and_skeys(n_sids, samps_per_sid):
    sids = [f"sid_{i:03d}" for i in range(n_sids)]
    sid_2_skeys = {sid: [(sid, j) for j in range(samps_per_sid)] for sid in sids}
    return sids, sid_2_skeys


def test_choose_ood_val_sids_returns_empty_when_pct_partition_is_zero():
    sids, _ = _uniform_sids_and_skeys(10, 5)
    result = _choose_ood_val_sids(
        sids_train=set(sids),
        n_sids_total_target=10,
        cfg=_cfg(pct_partition=0.0),
    )
    assert result == set()


def test_choose_ood_val_sids_returns_exact_class_count():
    n_sids = 20
    sids, _ = _uniform_sids_and_skeys(n_sids, 10)
    cfg = _cfg(pct_partition=0.1)

    result = _choose_ood_val_sids(
        sids_train=set(sids),
        n_sids_total_target=n_sids,
        cfg=cfg,
    )

    assert len(result) == round(n_sids * cfg.pct_partition)


def test_choose_ood_val_sids_result_is_subset_of_train():
    n_sids = 20
    sids, _ = _uniform_sids_and_skeys(n_sids, 5)
    cfg = _cfg(pct_partition=0.15)

    result = _choose_ood_val_sids(
        sids_train=set(sids),
        n_sids_total_target=n_sids,
        cfg=cfg,
    )

    assert result.issubset(set(sids))


def test_choose_ood_val_sids_raises_when_target_sid_count_exceeds_available():
    sids, _ = _uniform_sids_and_skeys(5, 3)

    # n_sids_total_target=100 → target = round(100*0.5) = 50 species >> 5 available
    with pytest.raises(ValueError, match="Target OOD-val sid count.*exceeds available"):
        _choose_ood_val_sids(
            sids_train=set(sids),
            n_sids_total_target=100,
            cfg=_cfg(pct_partition=0.5),
        )


# ─── _split_train_into_train_idval_oodval ────────────────────────────────────

def test_split_train_partitions_cover_pool_completely():
    pool = _uniform_pool(20, 10)
    train, id_val, ood_val = _split_train_into_train_idval_oodval(
        skeys_train_pool=pool,
        n_sids_total_target=20,
        n_samps_total_target=200,
        cfg=_cfg(pct_partition=0.1, pct_ood_tol=0.005),
    )
    assert train | id_val | ood_val == pool


def test_split_train_partitions_are_mutually_disjoint():
    pool = _uniform_pool(20, 10)
    train, id_val, ood_val = _split_train_into_train_idval_oodval(
        skeys_train_pool=pool,
        n_sids_total_target=20,
        n_samps_total_target=200,
        cfg=_cfg(pct_partition=0.1, pct_ood_tol=0.005),
    )
    assert not (train & id_val)
    assert not (train & ood_val)
    assert not (id_val & ood_val)


def test_split_train_id_val_hits_exact_sample_count():
    n_samps_total = 200
    pct = 0.1
    pool = _uniform_pool(20, 10)
    _, id_val, _ = _split_train_into_train_idval_oodval(
        skeys_train_pool=pool,
        n_sids_total_target=20,
        n_samps_total_target=n_samps_total,
        cfg=_cfg(pct_partition=pct, pct_ood_tol=0.005),
    )
    assert len(id_val) == round(n_samps_total * pct)


def test_split_train_ood_val_sample_proportion_within_tolerance():
    n_samps_total = 200
    pct = 0.1
    tol = 0.005
    pool = _uniform_pool(20, 10)
    _, _, ood_val = _split_train_into_train_idval_oodval(
        skeys_train_pool=pool,
        n_sids_total_target=20,
        n_samps_total_target=n_samps_total,
        cfg=_cfg(pct_partition=pct, pct_ood_tol=tol),
    )
    pct_actual = len(ood_val) / n_samps_total
    assert abs(pct_actual - pct) < tol


def test_split_train_every_id_species_has_at_least_one_train_sample():
    pool = _uniform_pool(20, 10)
    train, id_val, ood_val = _split_train_into_train_idval_oodval(
        skeys_train_pool=pool,
        n_sids_total_target=20,
        n_samps_total_target=200,
        cfg=_cfg(pct_partition=0.1, pct_ood_tol=0.005),
    )
    ood_sids = {sid for sid, _ in ood_val}
    id_sids = {sid for sid, _ in (train | id_val)} - ood_sids
    train_sids = {sid for sid, _ in train}

    for sid in id_sids:
        assert sid in train_sids, f"Species '{sid}' has no samples in train partition"


def test_split_train_ood_val_sids_may_coincide_with_id_test_class_universe():
    # The function must not exclude OOD-val choices based on what's in fixed test partitions.
    # Simply verify: the function succeeds and produces non-empty OOD-val without knowledge
    # of any test partitions — overlap is allowed by design.
    pool = _uniform_pool(20, 10)
    _, _, ood_val = _split_train_into_train_idval_oodval(
        skeys_train_pool=pool,
        n_sids_total_target=20,
        n_samps_total_target=200,
        cfg=_cfg(pct_partition=0.1, pct_ood_tol=0.005),
    )
    assert len(ood_val) > 0


def test_split_train_raises_when_id_val_target_exceeds_multis_pool():
    # After OOD-val takes round(20*0.1)=2 species, remaining 18 are all singletons
    # (1 sample each). n_samps_total_target=200 → id_val target = round(200*0.1)=20,
    # but only 18 multi-eligible samples exist (all singletons → 0 multis actually),
    # so the check fires.
    pool = {(f"sid_{i:03d}", 0) for i in range(20)}  # 20 singletons, 1 sample each
    with pytest.raises(ValueError, match="Target ID-val sample count.*exceeds available"):
        _split_train_into_train_idval_oodval(
            skeys_train_pool=pool,
            n_sids_total_target=20,
            n_samps_total_target=200,  # large target → id_val needs 20 multis, none exist
            cfg=_cfg(pct_partition=0.1, pct_ood_tol=0.005),
        )