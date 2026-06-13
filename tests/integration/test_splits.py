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
    di = split.data_indexes
    enc2cid = split.enc2cid
    return {
        "nshot_val_id": {
            name: set(split.nshot["buckets"]["train/val"][name])
            for name in split.nshot["names"]
        },
        "rfpaths": {
            "whole":    {d["rfpath"] for d in di["whole"]},
            "train":    {d["rfpath"] for d in di["train"]},
            "trainval": {d["rfpath"] for d in di["trainval"]},
            "val_id":   {d["rfpath"] for d in di["val"]["id"]},
            "val_ood":  {d["rfpath"] for d in di["val"]["ood"]},
            "test_id":  {d["rfpath"] for d in di["test"]["id"]},
            "test_ood": {d["rfpath"] for d in di["test"]["ood"]},
        },
        "cids": {
            "whole":    {enc2cid[d["class_enc"]] for d in di["whole"]},
            "train":    {enc2cid[d["class_enc"]] for d in di["train"]},
            "trainval": {enc2cid[d["class_enc"]] for d in di["trainval"]},
            "val_id":   {enc2cid[d["class_enc"]] for d in di["val"]["id"]},
            "val_ood":  {enc2cid[d["class_enc"]] for d in di["val"]["ood"]},
            "test_id":  {enc2cid[d["class_enc"]] for d in di["test"]["id"]},
            "test_ood": {enc2cid[d["class_enc"]] for d in di["test"]["ood"]},
        },
    }


# ─── rfpath coverage ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_whole_covers_all_partitions(split_data):
    r = split_data["rfpaths"]
    assert r["whole"] == r["trainval"] | r["test_id"] | r["test_ood"]


@pytest.mark.integration
def test_trainval_covers_train_and_val_partitions(split_data):
    r = split_data["rfpaths"]
    assert r["trainval"] == r["train"] | r["val_id"] | r["val_ood"]


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


@pytest.mark.integration
def test_whole_cids_composition(split_data):
    c = split_data["cids"]
    assert c["whole"] == c["trainval"] | c["test_ood"], (
        f"{len(c['whole'])} != {len(c['trainval'])} + {len(c['test_ood'])} "
        f"= {len(c['trainval']) + len(c['test_ood'])}"
    )
    assert len(c["whole"]) == len(c["trainval"]) + len(c["test_ood"])


# ─── n-shot buckets ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_nshot_val_id_buckets_disjoint(split_data):
    """Each class belongs to at most one n-shot bucket (by val_id membership)."""
    buckets = split_data["nshot_val_id"]
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert buckets[a] & buckets[b] == set(), (
                f"cids appear in multiple n-shot buckets: {a} and {b}"
            )
