"""
python -m pytest tests/integration/test_bryo_split.py
"""

from itertools import combinations
from types import SimpleNamespace

import pytest  # type: ignore[import]

from preprocessing.bryo.splits_utils import (
    build_data_indexes_bryo,
    build_ood_skeys,
    split_ood_genera_val_test,
)
from preprocessing.common.splits import build_id_partitions


def _make_img_ptrs(genera_n_samps: dict[str, int]) -> dict:
    return {
        genus: {idx: f"{genus}/img{idx:04d}.jpg" for idx in range(n)}
        for genus, n in genera_n_samps.items()
    }


@pytest.mark.integration
def test_bryo_split_partitions_do_not_overlap() -> None:
    # 6 ID genera (varying sample counts) + 4 OOD genera
    genera_n_samps = {
        "genA": 20, "genB": 15, "genC": 10, "genD": 8, "genE": 5, "genF": 3,
        "oodX": 12, "oodY": 9, "oodZ": 7, "oodW": 4,
    }
    genera_ood = {"oodX", "oodY", "oodZ", "oodW"}
    genera_id = set(genera_n_samps) - genera_ood

    img_ptrs_all = _make_img_ptrs(genera_n_samps)

    sid_2_samp_idxs = {
        sid: list(sorted(img_ptrs_all[sid].keys()))
        for sid in sorted(genera_n_samps)
    }
    n_samps_dict = {sid: len(idxs) for sid, idxs in sid_2_samp_idxs.items()}
    n_samps_dict_id = {sid: n_samps_dict[sid] for sid in genera_id}

    cfg = SimpleNamespace(seed=42, pct_eval=0.2, nst_names=["1-4", "5+"], nst_seps=[4])

    genera_ood_val, genera_ood_test = split_ood_genera_val_test(genera_ood, n_samps_dict, cfg.seed)
    skeys_ood_val = build_ood_skeys(genera_ood_val, sid_2_samp_idxs)
    skeys_ood_test = build_ood_skeys(genera_ood_test, sid_2_samp_idxs)

    skeys_train, skeys_id_val, skeys_id_test, _, _, _ = build_id_partitions(
        genera_id,
        sid_2_samp_idxs,
        n_samps_dict_id,
        cfg,
    )

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }

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

    skeys_by_partition = {
        name: set(zip(part["sids"], part["rfpaths"]))
        for name, part in partitions.items()
    }

    for name_a, name_b in combinations(skeys_by_partition, 2):
        overlap = skeys_by_partition[name_a] & skeys_by_partition[name_b]
        assert not overlap, (
            f"Overlap found between '{name_a}' and '{name_b}': "
            f"{len(overlap)} shared (sid, rfpath) pairs."
        )