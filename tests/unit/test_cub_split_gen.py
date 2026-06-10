"""
conda run -n biocosmos_b200 python -m pytest tests/unit/test_cub_split_gen.py
"""

from types import SimpleNamespace

import pytest

from preprocessing.cub.split_gen import (
    _build_classdir_to_cid,
    _build_img_ptrs,
    _class_dir_to_common_name,
    _normalize_cub_rfpath,
)
from preprocessing.cub.split_gen_utils import build_data_indexes_cub


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_class_data(*entries):
    """Each entry: (cid, class_dir_name, species_cid)."""
    return {
        cid: {"species": cid, "common_name": _class_dir_to_common_name(class_dir)}
        for cid, class_dir, cid in entries
    }


def _rfpaths_for(class_dir, n):
    return [f"{class_dir}/img_{i:04d}.jpg" for i in range(n)]


def _make_class_data_and_rfpaths(entries):
    """
    entries: list of (class_dir, species_cid, n_imgs)
    Returns (class_data, flat rfpaths list).
    """
    class_data = {}
    rfpaths = []
    for idx, (class_dir, cid, n) in enumerate(entries):
        class_data[f"cid_{idx}"] = {"species": cid, "common_name": _class_dir_to_common_name(class_dir)}
        rfpaths += _rfpaths_for(class_dir, n)
    return class_data, rfpaths


def _uniform_pool(n_cids, samps_per_cid):
    """Uniform train pool: n_cids species × samps_per_cid samples each."""
    return {
        (f"cid_{i:03d}", j)
        for i in range(n_cids)
        for j in range(samps_per_cid)
    }


def _cfg(pct_partition=0.1, pct_ood_tol=0.005, seed=42):
    return SimpleNamespace(seed=seed, pct_partition=pct_partition, pct_ood_tol=pct_ood_tol)


# ─── _normalize_cub_rfpath ───────────────────────────────────────────────────

def test_normalize_cub_rfpath_strips_leading_prefix():
    raw = "/some/long/prefix/images/001.Black_footed_Albatross/img_001.jpg"
    assert _normalize_cub_rfpath(raw) == "001.Black_footed_Albatross/img_001.jpg"


# ─── _build_classdir_to_cid ──────────────────────────────────────────────────

def test_build_classdir_to_cid_maps_each_class_dir_to_its_species_cid():
    class_data = _make_class_data(
        ("albatross", "001.Black_footed_Albatross", "diomedea_nigripes"),
        ("laysan", "002.Laysan_Albatross", "phoebastria_immutabilis"),
    )
    result = _build_classdir_to_cid(class_data)

    assert result["black_footed_albatross"] == "diomedea_nigripes"
    assert result["laysan_albatross"] == "phoebastria_immutabilis"
    assert len(result) == 2


def test_build_img_ptrs_rfpath_2_skey_covers_all_rfpaths():
    class_data, rfpaths = _make_class_data_and_rfpaths([
        ("001.Albatross", "diomedea_nigripes", 2),
        ("002.Laysan", "phoebastria_immutabilis", 3),
    ])
    _, rfpath_2_skey, cids = _build_img_ptrs(rfpaths, class_data)

    assert set(rfpath_2_skey.keys()) == set(rfpaths)
    assert set(cids) == {"diomedea_nigripes", "phoebastria_immutabilis"}


def test_build_img_ptrs_skey_values_are_unique():
    class_data, rfpaths = _make_class_data_and_rfpaths([
        ("001.Albatross", "diomedea_nigripes", 4),
        ("002.Laysan", "phoebastria_immutabilis", 4),
    ])
    _, rfpath_2_skey, _ = _build_img_ptrs(rfpaths, class_data)

    skeys = list(rfpath_2_skey.values())
    assert len(skeys) == len(set(skeys))


def test_build_img_ptrs_raises_on_unknown_class_dir():
    class_data = _make_class_data(("cid", "001.Albatross", "some_species"))
    rfpaths = ["999.UnknownClass/img_0001.jpg"]

    with pytest.raises(KeyError):
        _build_img_ptrs(rfpaths, class_data)


# ─── build_data_indexes_cub ──────────────────────────────────────────────────

def _make_cub_fixtures():
    cids = ["sp_a", "sp_b", "sp_c", "sp_d", "sp_e"]
    img_ptrs = {
        "sp_a": {0: "001.Sp_A/img0.jpg", 1: "001.Sp_A/img1.jpg"},
        "sp_b": {0: "002.Sp_B/img0.jpg", 1: "002.Sp_B/img1.jpg"},
        "sp_c": {0: "003.Sp_C/img0.jpg"},
        "sp_d": {0: "004.Sp_D/img0.jpg"},
        "sp_e": {0: "005.Sp_E/img0.jpg"},
    }
    skeys_pts = {
        "train":   {("sp_a", 0), ("sp_b", 0)},
        "id_val":  {("sp_a", 1)},
        "id_test": {("sp_b", 1)},
        "ood_val": {("sp_c", 0), ("sp_d", 0)},
        "ood_test":{("sp_e", 0)},
    }
    skeys_pts["trainval"] = (
        skeys_pts["train"]
        | skeys_pts["id_val"]
        | skeys_pts["ood_val"]
    )
    skeys_pts["whole"] = (
        skeys_pts["train"]
        | skeys_pts["id_val"]
        | skeys_pts["id_test"]
        | skeys_pts["ood_val"]
        | skeys_pts["ood_test"]
    )
    return cids, img_ptrs, skeys_pts


def test_cub_data_indexes_partition_size_composition():
    cids, img_ptrs, skeys_pts = _make_cub_fixtures()
    all_cids = sorted({cid for skeys in skeys_pts.values() for cid, _ in skeys})
    cid2enc = {cid: i for i, cid in enumerate(all_cids)}
    data_indexes = build_data_indexes_cub(skeys_pts, img_ptrs, cid2enc)

    assert len(data_indexes["trainval"]) == (
        len(data_indexes["train"])
        + len(data_indexes["val"]["id"])
        + len(data_indexes["val"]["ood"])
    )
    assert len(data_indexes["whole"]) == (
        len(data_indexes["trainval"])
        + len(data_indexes["test"]["id"])
        + len(data_indexes["test"]["ood"])
    )