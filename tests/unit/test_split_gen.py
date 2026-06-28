"""
python -m pytest tests/unit/test_split_gen.py
"""

import pytest
from types import SimpleNamespace

from preprocessing.common.split_gen import (
    add_trainval,
    build_data_indexes,
    build_nshot,
)
from utils.data import assemble_data_index


def test_add_trainval_unions_partitions() -> None:
    skeys_pts = {
        "train": {("cid_a", 0)},
        "val_id": {("cid_a", 1)},
        "val_ood": {("cid_b", 0)},
        "test_id": {("cid_a", 2)},
        "test_ood": {("cid_c", 0)},
    }

    add_trainval(skeys_pts)

    assert skeys_pts["trainval"] == {
        ("cid_a", 0),
        ("cid_a", 1),
        ("cid_b", 0),
    }


def test_build_nshot_buckets_ood_val_species_in_test_id_by_trainval_cardinality() -> None:
    # cid_ood lands in BOTH test_id and val_ood: ID-test draws a sample into test_id
    # (species stays in the pool), then OOD-val claims its remaining samples. Since
    # val_ood ⊆ trainval, cid_ood's trainval cardinality is 2 and it must be bucketed
    # into trainval/test by that cardinality.
    # With nst_seps=[2]: bisect_left([2], 2) == 0 → "few" bucket.
    # With nst_seps=[2]: bisect_left([2], 3) == 1 → "many" bucket.
    cfg = SimpleNamespace(nst_names=["few", "many"], nst_seps=[2])
    skeys_pts = {
        "train": {("cid_a", 0)},
        "val_id": {("cid_a", 1)},
        # cid_ood sample drawn into test_id during ID-test sampling
        "test_id": {("cid_ood", 0)},
        # cid_ood has 2 samples remaining in val_ood (and thus in trainval)
        "val_ood": {("cid_ood", 1), ("cid_ood", 2)},
        "test_ood": set(),
    }
    add_trainval(skeys_pts)
    # trainval = {(cid_a,0), (cid_a,1), (cid_ood,1), (cid_ood,2)}
    # cid_ood trainval count = 2 → bisect_left([2], 2) == 0 → "few" bucket

    nshot = build_nshot(skeys_pts, cfg)

    # cid_ood's test_id sample must appear in "few" trainval/test bucket (trainval count=2)
    assert "cid_ood" in nshot["buckets"]["trainval/test"]["few"]
    # No cid_ood entry in val_id buckets (OOD species are not ID)
    for bucket_name in cfg.nst_names:
        assert "cid_ood" not in nshot["buckets"]["train/val"][bucket_name]
    # Total test_id cids across all buckets equals cids in skeys_pts["test_id"]
    all_test_id_bucketed = set()
    for bucket_name in cfg.nst_names:
        all_test_id_bucketed.update(nshot["buckets"]["trainval/test"][bucket_name])
    assert all_test_id_bucketed == {cid for cid, _ in skeys_pts["test_id"]}


def test_build_nshot_uses_train_for_val_id_and_trainval_for_test_id() -> None:
    cfg = SimpleNamespace(nst_names=["few", "many"], nst_seps=[2])
    skeys_pts = {
        "train": {("cid_a", 0), ("cid_b", 0), ("cid_b", 1), ("cid_b", 2)},
        "val_id": {("cid_a", 1)},
        "test_id": {("cid_a", 2), ("cid_b", 3)},
        "val_ood": set(),
        "test_ood": set(),
    }
    add_trainval(skeys_pts)

    nshot = build_nshot(skeys_pts, cfg)

    # cid_a: train=1 → few bucket for val_id; trainval=2 → few bucket for test_id
    assert nshot["buckets"]["train/val"]["few"] == {"cid_a"}
    assert nshot["buckets"]["trainval/test"]["few"] == {"cid_a"}
    # cid_b: train=3 (no val_id sample) → many bucket for test_id only
    assert nshot["buckets"]["train/val"]["many"] == set()
    assert nshot["buckets"]["trainval/test"]["many"] == {"cid_b"}


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
        "val_id":   {("sp_a", 1)},
        "test_id":  {("sp_b", 1)},
        "val_ood":  {("sp_c", 0), ("sp_d", 0)},
        "test_ood": {("sp_e", 0)},
    }
    skeys_all = (
        skeys_pts["train"] | skeys_pts["val_id"] | skeys_pts["val_ood"]
        | skeys_pts["test_id"] | skeys_pts["test_ood"]
    )

    all_cids = sorted({cid for pt in skeys_pts.values() for cid, _ in pt})
    cid2enc = {cid: i for i, cid in enumerate(all_cids)}
    skey2meta = {skey: None for skey in skeys_all}
    data_indexes = build_data_indexes(skeys_pts, img_ptrs, cid2enc, skey2meta)

    assert "trainval" not in data_indexes
    assert len(data_indexes["train"]) == len(skeys_pts["train"])
    assert len(data_indexes["val"]["id"]) == len(skeys_pts["val_id"])
    assert len(data_indexes["val"]["ood"]) == len(skeys_pts["val_ood"])
    assert len(data_indexes["test"]["id"]) == len(skeys_pts["test_id"])
    assert len(data_indexes["test"]["ood"]) == len(skeys_pts["test_ood"])


def test_assemble_data_index_composites_and_raw():
    img_ptrs = {
        "sp_a": {0: "sp_a/img0.jpg", 1: "sp_a/img1.jpg"},
        "sp_b": {0: "sp_b/img0.jpg", 1: "sp_b/img1.jpg"},
        "sp_c": {0: "sp_c/img0.jpg"},
        "sp_d": {0: "sp_d/img0.jpg"},
        "sp_e": {0: "sp_e/img0.jpg"},
    }
    skeys_pts = {
        "train":    {("sp_a", 0), ("sp_b", 0)},
        "val_id":   {("sp_a", 1)},
        "test_id":  {("sp_b", 1)},
        "val_ood":  {("sp_c", 0), ("sp_d", 0)},
        "test_ood": {("sp_e", 0)},
    }
    skeys_all = (
        skeys_pts["train"] | skeys_pts["val_id"] | skeys_pts["val_ood"]
        | skeys_pts["test_id"] | skeys_pts["test_ood"]
    )
    all_cids = sorted({cid for pt in skeys_pts.values() for cid, _ in pt})
    cid2enc = {cid: i for i, cid in enumerate(all_cids)}
    skey2meta = {skey: None for skey in skeys_all}
    data_indexes = build_data_indexes(skeys_pts, img_ptrs, cid2enc, skey2meta)

    def rfpaths(part):
        return {d["rfpath"] for d in assemble_data_index(data_indexes, part)}

    # raw partitions return their underlying lists
    assert rfpaths("train") == {"sp_a/img0.jpg", "sp_b/img0.jpg"}
    assert rfpaths("val_ood") == {"sp_c/img0.jpg", "sp_d/img0.jpg"}

    # trainval = train + val(id) + val(ood); whole additionally adds test(id) + test(ood)
    assert rfpaths("trainval") == rfpaths("train") | rfpaths("val_id") | rfpaths("val_ood")
    assert rfpaths("whole") == (
        rfpaths("train") | rfpaths("val_id") | rfpaths("val_ood")
        | rfpaths("test_id") | rfpaths("test_ood")
    )
    # composites concatenate disjoint partitions (no dedup)
    assert len(assemble_data_index(data_indexes, "whole")) == sum(
        len(skeys) for skeys in skeys_pts.values()
    )

    with pytest.raises(KeyError):
        assemble_data_index(data_indexes, "bogus")
