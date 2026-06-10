"""
python -m pytest tests/unit/test_split_gen.py
"""

from types import SimpleNamespace
import pandas as pd

from preprocessing.common.split_gen import (
    build_dev_skeys_partitions,
    build_id_eval_nshot,
    build_trainval_skeys_partition,
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


def test_build_cid_2_samp_idxs_returns_empty_list_when_species_has_no_matches() -> None:
    cids = ["cid_a"]
    img_ptrs = {
        "cid_a": {
            0: "cid_a/img0.png",
            1: "cid_a/img1.png",
        },
    }
    df_metadata = pd.DataFrame(
        {
            "mask_name": ["img0.png", "img1.png"],
            "class_dv": ["ventral", "ventral"],
        }
    )

    cid_2_samp_idxs = build_cid_2_samp_idxs(
        cids,
        img_ptrs,
        pos_filter="dorsal",
        df_metadata=df_metadata,
    )

    assert cid_2_samp_idxs == {"cid_a": []}


def test_build_dev_skeys_partitions_uses_first_samples_per_partition() -> None:
    skeys_pts = {
        "train": {
            ("cid_b", 2),
            ("cid_a", 0),
            ("cid_c", 1),
            ("cid_b", 1),
        },
        "id_val": {
            ("cid_x", 2),
            ("cid_x", 0),
            ("cid_x", 1),
        },
        "id_test": {("cid_y", 2)},
        "ood_val": {("cid_z", 3), ("cid_z", 1)},
    }

    skeys_pts_dev = build_dev_skeys_partitions(skeys_pts, size_dev=2)

    assert set(skeys_pts_dev.keys()) == set(skeys_pts.keys())
    assert skeys_pts_dev["train"] == {("cid_a", 0), ("cid_b", 1)}
    assert skeys_pts_dev["id_val"] == {("cid_x", 0), ("cid_x", 1)}
    assert skeys_pts_dev["id_test"] == {("cid_y", 2)}
    assert skeys_pts_dev["ood_val"] == {("cid_z", 1), ("cid_z", 3)}


def test_build_dev_skeys_partitions_caps_to_train_size() -> None:
    skeys_pts = {
        "train": {("cid_a", 0), ("cid_b", 1)},
        "id_val": {("cid_x", 2), ("cid_x", 1), ("cid_x", 0)},
        "ood_test": set(),
    }

    skeys_pts_dev = build_dev_skeys_partitions(skeys_pts, size_dev=10)

    assert skeys_pts_dev["train"] == {("cid_a", 0), ("cid_b", 1)}
    assert skeys_pts_dev["id_val"] == {
        ("cid_x", 0),
        ("cid_x", 1),
        ("cid_x", 2),
    }
    assert skeys_pts_dev["ood_test"] == set()


def test_build_trainval_skeys_partition_unions_train_and_validation() -> None:
    skeys_pts = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        "ood_val": {("cid_b", 0)},
        "id_test": {("cid_a", 2)},
        "ood_test": {("cid_c", 0)},
    }

    trainval = build_trainval_skeys_partition(skeys_pts)

    assert trainval == {
        ("cid_a", 0),
        ("cid_a", 1),
        ("cid_b", 0),
    }


def test_build_trainval_skeys_partition_excludes_id_test_overlap() -> None:
    skeys_pts = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        "ood_val": {("cid_b", 0), ("cid_c", 0)},
        "id_test": {("cid_b", 0), ("cid_d", 0)},
        "ood_test": {("cid_e", 0)},
    }

    trainval = build_trainval_skeys_partition(skeys_pts)

    assert trainval == {
        ("cid_a", 0),
        ("cid_a", 1),
        ("cid_c", 0),
    }


def test_build_id_eval_nshot_buckets_ood_borrowed_id_test_by_trainval_cardinality() -> None:
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
    skeys_pts["trainval"] = build_trainval_skeys_partition(skeys_pts)
    # trainval = {(cid_a,0), (cid_a,1), (cid_ood,1), (cid_ood,2)}
    # cid_ood trainval count = 2 → bisect_left([2], 2) == 0 → "few" bucket

    id_eval_nshot = build_id_eval_nshot(skeys_pts, cfg)

    # cid_ood's borrowed id_test sample must appear in "few" id_test bucket (trainval count=2)
    assert ("cid_ood", 0) in id_eval_nshot["buckets"]["few"]["id_test"]
    # No cid_ood samples in id_val buckets (OOD species are not ID)
    for bucket_name in cfg.nst_names:
        assert ("cid_ood", 0) not in id_eval_nshot["buckets"][bucket_name]["id_val"]
    # Total id_test skeys across all buckets equals len(skeys_pts["id_test"])
    all_id_test_bucketed = set()
    for bucket_name in cfg.nst_names:
        all_id_test_bucketed.update(id_eval_nshot["buckets"][bucket_name]["id_test"])
    assert all_id_test_bucketed == skeys_pts["id_test"]


def test_build_id_eval_nshot_uses_train_for_id_val_and_trainval_for_id_test() -> None:
    cfg = SimpleNamespace(nst_names=["few", "many"], nst_seps=[2])
    skeys_pts = {
        "train": {("cid_a", 0), ("cid_b", 0), ("cid_b", 1), ("cid_b", 2)},
        "id_val": {("cid_a", 1)},
        "id_test": {("cid_a", 2), ("cid_b", 3)},
        "ood_val": set(),
        "ood_test": set(),
    }
    skeys_pts["trainval"] = build_trainval_skeys_partition(skeys_pts)

    id_eval_nshot = build_id_eval_nshot(skeys_pts, cfg)

    # cid_a: train=1 → few bucket for id_val; trainval=2 → few bucket for id_test
    assert id_eval_nshot["buckets"]["few"]["id_val"] == {("cid_a", 1)}
    assert id_eval_nshot["buckets"]["few"]["id_test"] == {("cid_a", 2)}
    # cid_b: train=3 → many bucket for id_val; trainval=3 → many bucket for id_test
    assert id_eval_nshot["buckets"]["many"]["id_val"] == set()
    assert id_eval_nshot["buckets"]["many"]["id_test"] == {("cid_b", 3)}
