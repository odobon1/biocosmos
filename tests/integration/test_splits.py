"""
python -m pytest tests/integration/test_splits.py
"""

import pytest

from utils.utils import load_pickle, paths


DATASETS = ["bryo", "cub", "lepid", "nymph"]  # all tests run on each dataset


@pytest.fixture(scope="module", params=DATASETS)
def split_data(request):
    dataset = request.param
    split = load_pickle(paths["metadata"][dataset] / "splits/D10/split.pkl")
    enc2cid = split.enc2cid
    pts = ["train", "val_id", "val_ood", "test_id", "test_ood", "trainval", "whole"]
    return {
        "nshot_val": {
            name: set(split.nshot["buckets"]["train/val"][name])
            for name in split.nshot["names"]
        },
        "nshot_test": {
            name: set(split.nshot["buckets"]["trainval/test"][name])
            for name in split.nshot["names"]
        },
        "rfpaths": {
            pt: {d["rfpath"] for d in split.get_data(pt)}
            for pt in pts
        },
        "cids": {
            pt: {enc2cid[d["class_enc"]] for d in split.get_data(pt)}
            for pt in pts
        },
    }


# ─── rfpath coverage ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_trainval_covers_train_and_val_partitions(split_data):
    r = split_data["rfpaths"]
    assert r["trainval"] == r["train"] | r["val_id"] | r["val_ood"]


@pytest.mark.integration
def test_whole_covers_all_partitions(split_data):
    r = split_data["rfpaths"]
    assert r["whole"] == (
        r["train"] | r["val_id"] | r["val_ood"] | r["test_id"] | r["test_ood"]
    )


# ─── rfpath disjointness ──────────────────────────────────────────────────────

@pytest.mark.integration
def test_trainval_disjoint_from_test(split_data):
    r = split_data["rfpaths"]
    assert r["trainval"] & r["test_id"] == set()
    assert r["trainval"] & r["test_ood"] == set()


@pytest.mark.integration
def test_train_eval_partitions_disjoint(split_data):
    r = split_data["rfpaths"]

    assert r["train"] & r["val_id"] == set()
    assert r["train"] & r["val_ood"] == set()
    assert r["train"] & r["test_id"] == set()
    assert r["train"] & r["test_ood"] == set()

    assert r["val_id"] & r["val_ood"] == set()
    assert r["val_id"] & r["test_id"] == set()
    assert r["val_id"] & r["test_ood"] == set()

    assert r["val_ood"] & r["test_id"] == set()
    assert r["val_ood"] & r["test_ood"] == set()

    assert r["test_id"] & r["test_ood"] == set()


# ─── cid consistency ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_id_cids_disjoint_from_val_ood_cids(split_data):
    c = split_data["cids"]
    assert c["train"] & c["val_ood"] == set()


@pytest.mark.integration
def test_trainval_cids_disjoint_from_test_ood_cids(split_data):
    c = split_data["cids"]
    assert c["trainval"] & c["test_ood"] == set()


@pytest.mark.integration
def test_trainval_cids_composition(split_data):
    c = split_data["cids"]
    assert c["trainval"] == c["train"] | c["val_ood"]
    assert len(c["trainval"]) == len(c["train"]) + len(c["val_ood"])


# ─── n-shot buckets ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_nshot_val_buckets_disjoint(split_data):
    """Each class belongs to at most one n-shot bucket (by val_id membership)."""
    buckets = split_data["nshot_val"]
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert buckets[a] & buckets[b] == set(), (
                f"cids appear in multiple n-shot buckets: {a} and {b}"
            )


@pytest.mark.integration
def test_nshot_test_buckets_disjoint(split_data):
    """Each class belongs to at most one n-shot bucket (by trainval membership)."""
    buckets = split_data["nshot_test"]
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert buckets[a] & buckets[b] == set(), (
                f"cids appear in multiple n-shot buckets: {a} and {b}"
            )
