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
    return {
        "rfpaths": {
            "whole":    {d["rfpath"] for d in di["whole"]},
            "train":    {d["rfpath"] for d in di["train"]},
            "trainval": {d["rfpath"] for d in di["trainval"]},
            "id_val":   {d["rfpath"] for d in di["val"]["id"]},
            "ood_val":  {d["rfpath"] for d in di["val"]["ood"]},
            "id_test":  {d["rfpath"] for d in di["test"]["id"]},
            "ood_test": {d["rfpath"] for d in di["test"]["ood"]},
        },
        "cids": {
            "whole":    {d["cid"] for d in di["whole"]},
            "train":    {d["cid"] for d in di["train"]},
            "trainval": {d["cid"] for d in di["trainval"]},
            "id_val":   {d["cid"] for d in di["val"]["id"]},
            "ood_val":  {d["cid"] for d in di["val"]["ood"]},
            "id_test":  {d["cid"] for d in di["test"]["id"]},
            "ood_test": {d["cid"] for d in di["test"]["ood"]},
        },
    }


# ─── rfpath coverage ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_whole_covers_all_partitions(split_data):
    r = split_data["rfpaths"]
    assert r["whole"] == r["trainval"] | r["id_test"] | r["ood_test"]


@pytest.mark.integration
def test_trainval_covers_train_and_val_partitions(split_data):
    r = split_data["rfpaths"]
    assert r["trainval"] == r["train"] | r["id_val"] | r["ood_val"]


# ─── rfpath disjointness ──────────────────────────────────────────────────────

@pytest.mark.integration
def test_trainval_disjoint_from_test(split_data):
    r = split_data["rfpaths"]
    assert r["trainval"] & r["id_test"] == set()
    assert r["trainval"] & r["ood_test"] == set()


@pytest.mark.integration
def test_train_eval_partitions_disjoint(split_data):
    r = split_data["rfpaths"]

    assert r["train"] & r["id_val"] == set()
    assert r["train"] & r["ood_val"] == set()
    assert r["train"] & r["id_test"] == set()
    assert r["train"] & r["ood_test"] == set()

    assert r["id_val"] & r["ood_val"] == set()
    assert r["id_val"] & r["id_test"] == set()
    assert r["id_val"] & r["ood_test"] == set()

    assert r["ood_val"] & r["id_test"] == set()
    assert r["ood_val"] & r["ood_test"] == set()

    assert r["id_test"] & r["ood_test"] == set()


# ─── cid consistency ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_id_cids_disjoint_from_ood_val_cids(split_data):
    c = split_data["cids"]
    assert c["train"] & c["ood_val"] == set()


@pytest.mark.integration
def test_trainval_cids_disjoint_from_ood_test_cids(split_data):
    c = split_data["cids"]
    assert c["trainval"] & c["ood_test"] == set()


@pytest.mark.integration
def test_trainval_cids_composition(split_data):
    c = split_data["cids"]
    assert c["trainval"] == c["train"] | c["ood_val"]
    assert len(c["trainval"]) == len(c["train"]) + len(c["ood_val"])


@pytest.mark.integration
def test_whole_cids_composition(split_data):
    c = split_data["cids"]
    assert c["whole"] == c["trainval"] | c["ood_test"], (
        f"{len(c['whole'])} != {len(c['trainval'])} + {len(c['ood_test'])} "
        f"= {len(c['trainval']) + len(c['ood_test'])}"
    )
    assert len(c["whole"]) == len(c["trainval"]) + len(c["ood_test"])
