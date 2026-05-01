"""
python -m pytest tests/unit/test_splits.py
"""

from types import SimpleNamespace
import pandas as pd  # type: ignore[import]
import pytest  # type: ignore[import]

from preprocessing.common.splits import (
    build_dev_skeys_partitions,
    build_id_eval_nshot,
    build_id_partitions,
    build_trainval_skeys_partition,
)
from preprocessing.nymph.splits_utils import build_cid_2_samp_idxs


def test_build_cid_2_samp_idxs_defaults_to_all_samples() -> None:
    cids = ["cid_a", "cid_b"]
    img_ptrs = {
        "cid_a": {0: "cid_a/img0.png", 1: "cid_a/img1.png"},
        "cid_b": {0: "cid_b/img0.png"},
    }

    cid_2_samp_idxs = build_cid_2_samp_idxs(cids, img_ptrs=img_ptrs)

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
        pos_filter="dorsal",
        img_ptrs=img_ptrs,
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
        pos_filter="dorsal",
        img_ptrs=img_ptrs,
        df_metadata=df_metadata,
    )

    assert cid_2_samp_idxs == {"cid_a": []}


def test_gen_id_partitions_keeps_filtered_singleton_real_sample_index(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(seed=7, pct_eval=0.1, pct_partition=0.05)
    cids_id = {"cid_single", "cid_multi"}
    cid_2_samp_idxs = {
        "cid_single": [4],
        "cid_multi": [0, 2],
    }
    n_samps_dict = {
        "cid_single": 1,
        "cid_multi": 2,
    }

    # This test targets singleton index retention only; keep strat_split out of scope.
    monkeypatch.setattr("preprocessing.common.splits.strat_split", lambda **kwargs: (set(), set(), set()))

    skeys_train, skeys_id_val, skeys_id_test, cid_2_skeys_id, _, _, _ = build_id_partitions(
        cids_id=cids_id,
        cid_2_samp_idxs=cid_2_samp_idxs,
        n_samps_dict=n_samps_dict,
        cfg=cfg,
    )

    assert ("cid_single", 4) in skeys_train
    assert ("cid_single", 0) not in skeys_train
    assert cid_2_skeys_id["cid_single"] == [("cid_single", 4)]
    assert all(skey[0] != "cid_single" for skey in skeys_id_val.union(skeys_id_test))


def test_build_dev_skeys_partitions_mirrors_keys_with_first_train_samples() -> None:
    skeys_partitions = {
        "train": {
            ("cid_b", 2),
            ("cid_a", 0),
            ("cid_c", 1),
            ("cid_b", 1),
        },
        "id_val": {("cid_x", 1)},
        "id_test": {("cid_y", 2)},
        "ood_val": {("cid_z", 3)},
    }

    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, size_dev=2)

    expected = {("cid_a", 0), ("cid_b", 1)}
    assert set(skeys_partitions_dev.keys()) == set(skeys_partitions.keys())
    for key in skeys_partitions_dev:
        assert skeys_partitions_dev[key] == expected


def test_build_dev_skeys_partitions_caps_to_train_size() -> None:
    skeys_partitions = {
        "train": {("cid_a", 0), ("cid_b", 1)},
        "id_val": set(),
    }

    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, size_dev=10)

    expected = {("cid_a", 0), ("cid_b", 1)}
    assert skeys_partitions_dev["train"] == expected
    assert skeys_partitions_dev["id_val"] == expected


def test_build_dev_skeys_partitions_rejects_invalid_size() -> None:
    skeys_partitions = {"train": {("cid_a", 0)}}

    with pytest.raises(ValueError, match="size_dev must be greater than 0"):
        build_dev_skeys_partitions(skeys_partitions, size_dev=0)


def test_build_trainval_skeys_partition_unions_train_and_validation() -> None:
    skeys_partitions = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        "ood_val": {("cid_b", 0)},
        "id_test": {("cid_a", 2)},
        "ood_test": {("cid_c", 0)},
    }

    trainval = build_trainval_skeys_partition(skeys_partitions)

    assert trainval == {
        ("cid_a", 0),
        ("cid_a", 1),
        ("cid_b", 0),
    }


def test_build_trainval_skeys_partition_excludes_id_test_overlap() -> None:
    skeys_partitions = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        "ood_val": {("cid_b", 0), ("cid_c", 0)},
        "id_test": {("cid_b", 0), ("cid_d", 0)},
        "ood_test": {("cid_e", 0)},
    }

    trainval = build_trainval_skeys_partition(skeys_partitions)

    assert trainval == {
        ("cid_a", 0),
        ("cid_a", 1),
        ("cid_c", 0),
    }


def test_build_id_eval_nshot_uses_train_for_id_val_and_trainval_for_id_test() -> None:
    cfg = SimpleNamespace(nst_names=["few", "many"], nst_seps=[2])
    cids_id = {"cid_a", "cid_b"}
    cid_2_skeys_id = {
        "cid_a": [("cid_a", 0), ("cid_a", 1), ("cid_a", 2)],
        "cid_b": [("cid_b", 0), ("cid_b", 1), ("cid_b", 2), ("cid_b", 3)],
    }
    skeys_partitions = {
        "train": {("cid_a", 0), ("cid_b", 0), ("cid_b", 1), ("cid_b", 2)},
        "id_val": {("cid_a", 1)},
        "id_test": {("cid_a", 2), ("cid_b", 3)},
        "ood_val": set(),
        "ood_test": set(),
    }
    skeys_partitions["trainval"] = build_trainval_skeys_partition(skeys_partitions)

    id_eval_nshot = build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id)

    assert id_eval_nshot["buckets"]["few"]["id_val"] == {("cid_a", 1)}
    assert id_eval_nshot["buckets"]["few"]["trainval"] == {
        ("cid_a", 0),
        ("cid_a", 1),
    }
    assert id_eval_nshot["buckets"]["few"]["id_test"] == {("cid_a", 2)}
    assert id_eval_nshot["buckets"]["many"]["trainval"] == {
        ("cid_b", 0),
        ("cid_b", 1),
        ("cid_b", 2),
    }
    assert id_eval_nshot["buckets"]["many"]["id_test"] == {("cid_b", 3)}


def test_build_id_eval_nshot_buckets_ood_borrowed_id_test_by_trainval_cardinality() -> None:
    # cid_ood is an OOD species: 2 samples in ood_val (trainval), 1 borrowed into id_test.
    # trainval cardinality for cid_ood = 2.
    # With nst_seps=[2]: bisect_left([2], 2) == 0 → "few" bucket.
    # With nst_seps=[2]: bisect_left([2], 3) == 1 → "many" bucket.
    cfg = SimpleNamespace(nst_names=["few", "many"], nst_seps=[2])
    cids_id = {"cid_a"}
    cid_2_skeys_id = {
        "cid_a": [("cid_a", 0), ("cid_a", 1)],
    }
    skeys_partitions = {
        "train": {("cid_a", 0)},
        "id_val": {("cid_a", 1)},
        # cid_ood sample borrowed into id_test
        "id_test": {("cid_ood", 0)},
        # cid_ood has 2 samples remaining in ood_val (and thus in trainval)
        "ood_val": {("cid_ood", 1), ("cid_ood", 2)},
        "ood_test": set(),
    }
    skeys_partitions["trainval"] = build_trainval_skeys_partition(skeys_partitions)
    # trainval = {(cid_a,0), (cid_a,1), (cid_ood,1), (cid_ood,2)}
    # cid_ood trainval count = 2 → bisect_left([2], 2) == 0 → "few" bucket

    id_eval_nshot = build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id)

    # cid_ood's borrowed id_test sample must appear in "few" id_test bucket (trainval count=2)
    assert ("cid_ood", 0) in id_eval_nshot["buckets"]["few"]["id_test"]
    # No cid_ood samples in id_val or trainval buckets (OOD species are not ID)
    for bucket_name in cfg.nst_names:
        assert ("cid_ood", 0) not in id_eval_nshot["buckets"][bucket_name]["id_val"]
        assert ("cid_ood", 0) not in id_eval_nshot["buckets"][bucket_name]["trainval"]
    # Total id_test skeys across all buckets equals len(skeys_partitions["id_test"])
    all_id_test_bucketed = set()
    for bucket_name in cfg.nst_names:
        all_id_test_bucketed.update(id_eval_nshot["buckets"][bucket_name]["id_test"])
    assert all_id_test_bucketed == skeys_partitions["id_test"]


def test_build_id_partitions_can_draw_id_test_from_extra_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(seed=5, pct_eval=0.5, pct_partition=0.1)
    cids_id = {"cid_a"}
    cid_2_samp_idxs = {"cid_a": [0, 1, 2, 3]}
    n_samps_dict = {"cid_a": 4}
    skeys_id_test_extra = {("cid_ood", 0), ("cid_ood", 1)}

    # First call (ID test from ID+extra, choose_partition="test"):
    #   strat_split returns (rem, val_tmp, test_tmp); chosen = test_tmp = {cid_ood 0}.
    # Second call (ID val from remaining ID pool, choose_partition="val"):
    #   strat_split returns (rem, val_tmp, test_tmp); chosen = val_tmp = {cid_a 1}.
    #   test_tmp must be empty so nothing is restored into rem/train.
    strat_calls = [
        (
            {("cid_a", 0), ("cid_a", 1), ("cid_a", 2), ("cid_a", 3), ("cid_ood", 0), ("cid_ood", 1)},
            {("cid_a", 0)},
            {("cid_ood", 0)},
        ),
        (
            {("cid_a", 2), ("cid_a", 3)},
            {("cid_a", 1)},
            set(),
        ),
    ]

    def strat_split_stub(**kwargs):
        return strat_calls.pop(0)

    monkeypatch.setattr("preprocessing.common.splits.strat_split", strat_split_stub)

    (
        skeys_train,
        skeys_id_val,
        skeys_id_test,
        _,
        _,
        _,
        skeys_id_test_extra_taken,
    ) = build_id_partitions(
        cids_id=cids_id,
        cid_2_samp_idxs=cid_2_samp_idxs,
        n_samps_dict=n_samps_dict,
        cfg=cfg,
        skeys_id_test_extra=skeys_id_test_extra,
    )

    assert ("cid_ood", 0) in skeys_id_test
    assert skeys_id_test_extra_taken == {("cid_ood", 0)}
    assert skeys_id_val == {("cid_a", 1)}
    assert skeys_train == {("cid_a", 2), ("cid_a", 3)}