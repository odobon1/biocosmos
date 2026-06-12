"""
python -m pytest tests/unit/test_split_gen.py
"""

from types import SimpleNamespace
import pandas as pd

from preprocessing.common.split_gen import (
    build_data_indexes,
    build_nshot,
    build_skeys_trainval,
)
from preprocessing.nymph.split_gen_utils import build_cid_2_samp_idxs


def test_build_cid_2_samp_idxs_defaults_to_all_samples() -> None:
    cids = ["cid_a", "cid_b"]
    img_ptrs = {
        "cid_a": {0: "cid_a/img0.png", 1: "cid_a/img1.png"},
        "cid_b": {0: "cid_b/img0.png"},
    }

    cid_2_samp_idxs = build_cid_2_samp_idxs(cids, img_ptrs)

    assert cid_2_samp_idxs == {
        "cid_a": [0, 1],
        "cid_b": [0],
    }


def test_build_cid_2_samp_idxs_filters_to_requested_position() -> None:
    cids = ["cid_a", "cid_b"]
    img_ptrs = {
        "cid_a": {
            0: "cid_a/img0.png",
            1: "cid_a/img1.png",
            2: "cid_a/img2.png",
        },
        "cid_b": {
            0: "cid_b/img3.png",
            1: "cid_b/img4.png",
        },
    }
    df_metadata = pd.DataFrame(
        {
            "mask_name": ["img0.png", "img1.png", "img2.png", "img3.png", "img4.png"],
            "class_dv": ["dorsal", "ventral", "dorsal", "ventral", "dorsal"],
        }
    )

    cid_2_samp_idxs = build_cid_2_samp_idxs(
        cids,
        img_ptrs,
        pos_filter="dorsal",
        df_metadata=df_metadata,
    )

    assert cid_2_samp_idxs == {
        "cid_a": [0, 2],
        "cid_b": [1],
    }


def test_build_skeys_trainval_unions_train_and_validation() -> None:
    skeys_pts = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        "ood_val": {("cid_b", 0)},
        "id_test": {("cid_a", 2)},
        "ood_test": {("cid_c", 0)},
    }

    trainval = build_skeys_trainval(skeys_pts)

    assert trainval == {
        ("cid_a", 0),
        ("cid_a", 1),
        ("cid_b", 0),
    }


def test_build_skeys_trainval_excludes_id_test_overlap() -> None:
    skeys_pts = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        "ood_val": {("cid_b", 0), ("cid_c", 0)},
        "id_test": {("cid_b", 0), ("cid_d", 0)},
        "ood_test": {("cid_e", 0)},
    }

    trainval = build_skeys_trainval(skeys_pts)

    assert trainval == {
        ("cid_a", 0),
        ("cid_a", 1),
        ("cid_c", 0),
    }


def test_build_nshot_buckets_ood_borrowed_id_test_by_trainval_cardinality() -> None:
    # cid_ood is an OOD species: 2 samples in ood_val (trainval), 1 borrowed into id_test.
    # trainval cardinality for cid_ood = 2.
    # With nst_seps=[2]: bisect_left([2], 2) == 0 → "few" bucket.
    # With nst_seps=[2]: bisect_left([2], 3) == 1 → "many" bucket.
    cfg = SimpleNamespace(nst_names=["few", "many"], nst_seps=[2])
    skeys_pts = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        # cid_ood sample borrowed into id_test
        "id_test": {("cid_ood", 0)},
        # cid_ood has 2 samples remaining in ood_val (and thus in trainval)
        "ood_val": {("cid_ood", 1), ("cid_ood", 2)},
        "ood_test": set(),
    }
    skeys_pts["trainval"] = build_skeys_trainval(skeys_pts)
    # trainval = {(cid_a,0), (cid_a,1), (cid_ood,1), (cid_ood,2)}
    # cid_ood trainval count = 2 → bisect_left([2], 2) == 0 → "few" bucket

    nshot = build_nshot(skeys_pts, cfg)

    # cid_ood's borrowed id_test sample must appear in "few" id_test bucket (trainval count=2)
    assert ("cid_ood", 0) in nshot["buckets"]["few"]["id_test"]
    # No cid_ood samples in id_val buckets (OOD species are not ID)
    for bucket_name in cfg.nst_names:
        assert ("cid_ood", 0) not in nshot["buckets"][bucket_name]["id_val"]
    # Total id_test skeys across all buckets equals len(skeys_pts["id_test"])
    all_id_test_bucketed = set()
    for bucket_name in cfg.nst_names:
        all_id_test_bucketed.update(nshot["buckets"][bucket_name]["id_test"])
    assert all_id_test_bucketed == skeys_pts["id_test"]


def test_build_nshot_uses_train_for_id_val_and_trainval_for_id_test() -> None:
    cfg = SimpleNamespace(nst_names=["few", "many"], nst_seps=[2])
    skeys_pts = {
        "train": {("cid_a", 0), ("cid_b", 0), ("cid_b", 1), ("cid_b", 2)},
        "id_val": {("cid_a", 1)},
        "id_test": {("cid_a", 2), ("cid_b", 3)},
        "ood_val": set(),
        "ood_test": set(),
    }
    skeys_pts["trainval"] = build_skeys_trainval(skeys_pts)

    nshot = build_nshot(skeys_pts, cfg)

    # cid_a: train=1 → few bucket for id_val; trainval=2 → few bucket for id_test
    assert nshot["buckets"]["few"]["id_val"] == {("cid_a", 1)}
    assert nshot["buckets"]["few"]["id_test"] == {("cid_a", 2)}
    # cid_b: train=3 → many bucket for id_val; trainval=3 → many bucket for id_test
    assert nshot["buckets"]["many"]["id_val"] == set()
    assert nshot["buckets"]["many"]["id_test"] == {("cid_b", 3)}


def test_build_data_indexes_partition_size_composition():
    img_ptrs = {
        "sp_a": {0: "sp_a/img0.jpg", 1: "sp_a/img1.jpg"},
        "sp_b": {0: "sp_b/img0.jpg", 1: "sp_b/img1.jpg"},
        "sp_c": {0: "sp_c/img0.jpg"},
        "sp_d": {0: "sp_d/img0.jpg"},
        "sp_e": {0: "sp_e/img0.jpg"},
    }
    skeys_pts = {
        "train":    {("sp_a", 0), ("sp_b", 0)},
        "id_val":   {("sp_a", 1)},
        "id_test":  {("sp_b", 1)},
        "ood_val":  {("sp_c", 0), ("sp_d", 0)},
        "ood_test": {("sp_e", 0)},
    }
    skeys_pts["trainval"] = skeys_pts["train"] | skeys_pts["id_val"] | skeys_pts["ood_val"]
    skeys_pts["whole"] = skeys_pts["trainval"] | skeys_pts["id_test"] | skeys_pts["ood_test"]

    all_cids = sorted({cid for pt in skeys_pts.values() for cid, _ in pt})
    cid2enc = {cid: i for i, cid in enumerate(all_cids)}
    skey2meta = {skey: None for skey in skeys_pts["whole"]}
    data_indexes = build_data_indexes(skeys_pts, img_ptrs, cid2enc, skey2meta)

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
