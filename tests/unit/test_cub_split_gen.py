import pytest

from preprocessing.cub.split_gen_utils import (
    _build_classdir_to_cid,
    build_img_ptrs,
    _class_dir_to_common_name,
    normalize_cub_rfpath,
)

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


# ─── normalize_cub_rfpath ────────────────────────────────────────────────────

def test_normalize_cub_rfpath_strips_leading_prefix():
    raw = "/some/long/prefix/images/001.Black_footed_Albatross/img_001.jpg"
    assert normalize_cub_rfpath(raw) == "001.Black_footed_Albatross/img_001.jpg"


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
    _, rfpath_2_skey, cids = build_img_ptrs(rfpaths, class_data=class_data)

    assert set(rfpath_2_skey.keys()) == set(rfpaths)
    assert set(cids) == {"diomedea_nigripes", "phoebastria_immutabilis"}


def test_build_img_ptrs_skey_values_are_unique():
    class_data, rfpaths = _make_class_data_and_rfpaths([
        ("001.Albatross", "diomedea_nigripes", 4),
        ("002.Laysan", "phoebastria_immutabilis", 4),
    ])
    _, rfpath_2_skey, _ = build_img_ptrs(rfpaths, class_data=class_data)

    skeys = list(rfpath_2_skey.values())
    assert len(skeys) == len(set(skeys))


def test_build_img_ptrs_raises_on_unknown_class_dir():
    class_data = _make_class_data(("cid", "001.Albatross", "some_species"))
    rfpaths = ["999.UnknownClass/img_0001.jpg"]

    with pytest.raises(KeyError):
        build_img_ptrs(rfpaths, class_data=class_data)
