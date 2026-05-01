"""
python -m pytest tests/integration/test_bryo_split.py
"""

from itertools import combinations
from types import SimpleNamespace

import pytest  # type: ignore[import]

from preprocessing.bryo.splits_utils import build_data_indexes_bryo
from preprocessing.common.splits import (
    build_genus_2_cids,
    build_id_partitions,
    build_n_insts_2_classes_g,
    build_ood_partitions,
    build_trainval_skeys_partition,
)


def _make_img_ptrs(genera_n_samps: dict[str, int]) -> dict:
    return {
        genus: {idx: f"{genus}/img{idx:04d}.jpg" for idx in range(n)}
        for genus, n in genera_n_samps.items()
    }


@pytest.mark.integration
def test_bryo_split_partitions_do_not_overlap() -> None:
    genera_n_samps = {
        "genA": 20, "genB": 15, "genC": 10, "genD": 8, "genE": 5, "genF": 3,
        "genX": 12, "genY": 9, "genZ": 7, "genW": 4,
    }

    img_ptrs_all = _make_img_ptrs(genera_n_samps)

    cids = sorted(genera_n_samps)
    cid_2_samp_idxs = {
        cid: list(sorted(img_ptrs_all[cid].keys()))
        for cid in cids
    }
    n_samps_dict = {cid: len(idxs) for cid, idxs in cid_2_samp_idxs.items()}

    cfg = SimpleNamespace(
        seed=42, pct_partition=0.1, pct_eval=0.2, pct_ood_tol=0.15,
        nst_names=["1-4", "5+"], nst_seps=[4], size_dev=5,
    )

    genus_2_cids = build_genus_2_cids(cids)
    n_insts_2_classes_g = build_n_insts_2_classes_g(cids)

    cids_id, _, _, skeys_ood_val, skeys_ood_test = build_ood_partitions(
        n_insts_2_classes_g,
        genus_2_cids,
        set(cids),
        cid_2_samp_idxs,
        n_samps_dict,
        cfg,
    )

    skeys_train, skeys_id_val, skeys_id_test, _, _, _, skeys_id_test_extra_taken = build_id_partitions(
        cids_id,
        cid_2_samp_idxs,
        n_samps_dict,
        cfg,
        skeys_id_test_extra=skeys_ood_val,
    )
    skeys_ood_val = skeys_ood_val - skeys_id_test_extra_taken

    assert not (skeys_id_test & skeys_ood_val), "ID test must not overlap with OOD val"

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }
    skeys_partitions["trainval"] = build_trainval_skeys_partition(skeys_partitions)

    data_indexes = build_data_indexes_bryo(
        sorted(genera_n_samps),
        skeys_partitions,
        img_ptrs=img_ptrs_all,
    )

    partitions = {
        "train": data_indexes["train"],
        "validation/id": data_indexes["validation"]["id"],
        "validation/ood": data_indexes["validation"]["ood"],
        "test/id": data_indexes["test"]["id"],
        "test/ood": data_indexes["test"]["ood"],
    }

    trainval_skeys = {(datum["cid"], datum["rfpath"]) for datum in data_indexes["trainval"]}
    expected_trainval_skeys = {
        (datum["cid"], datum["rfpath"])
        for datum in data_indexes["train"]
    }
    expected_trainval_skeys |= {
        (datum["cid"], datum["rfpath"])
        for datum in data_indexes["validation"]["id"]
    }
    expected_trainval_skeys |= {
        (datum["cid"], datum["rfpath"])
        for datum in data_indexes["validation"]["ood"]
    }
    assert trainval_skeys == expected_trainval_skeys

    skeys_by_partition = {
        name: {(datum["cid"], datum["rfpath"]) for datum in part}
        for name, part in partitions.items()
    }

    for name_a, name_b in combinations(skeys_by_partition, 2):
        if {name_a, name_b} == {"validation/ood", "test/id"}:
            # ID-test is intentionally allowed to sample from OOD-val pool.
            continue
        overlap = skeys_by_partition[name_a] & skeys_by_partition[name_b]
        assert not overlap, (
            f"Overlap found between '{name_a}' and '{name_b}': "
            f"{len(overlap)} shared (cid, rfpath) pairs."
        )

    n_total = sum(len(skeys_partitions[name]) for name in ("train", "id_val", "id_test", "ood_val", "ood_test"))
    pct_id_val = len(skeys_partitions["id_val"]) / n_total
    pct_id_test = len(skeys_partitions["id_test"]) / n_total
    pct_ood_val = len(skeys_partitions["ood_val"]) / n_total
    pct_ood_test = len(skeys_partitions["ood_test"]) / n_total

    assert abs(pct_id_val - cfg.pct_partition) < cfg.pct_ood_tol
    assert abs(pct_id_test - cfg.pct_partition) < cfg.pct_ood_tol
    assert abs(pct_ood_val - cfg.pct_partition) < cfg.pct_ood_tol
    assert abs(pct_ood_test - cfg.pct_partition) < cfg.pct_ood_tol