"""
conda run -n biocosmos_b200 python -m pytest tests/unit/test_nymph_split_indexes.py
"""

import pandas as pd

from preprocessing.nymph.splits_utils import build_data_indexes


def _make_nymph_fixtures():
    cids = ["sp_a", "sp_b", "sp_c", "sp_d", "sp_e"]
    img_ptrs = {
        "sp_a": {0: "sp_a/img0.png", 1: "sp_a/img1.png"},
        "sp_b": {0: "sp_b/img0.png", 1: "sp_b/img1.png"},
        "sp_c": {0: "sp_c/img0.png"},
        "sp_d": {0: "sp_d/img0.png"},
        "sp_e": {0: "sp_e/img0.png"},
    }
    df_metadata = pd.DataFrame(
        {
            "mask_name": ["img0.png", "img1.png"],
            "class_dv":  ["dorsal",   "ventral"],
            "sex":        ["female",   "male"],
        }
    )
    skeys_partitions = {
        "train":    {("sp_a", 0), ("sp_b", 0)},
        "id_val":   {("sp_a", 1)},
        "id_test":  {("sp_b", 1)},
        "ood_val":  {("sp_c", 0), ("sp_d", 0)},
        "ood_test": {("sp_e", 0)},
    }
    skeys_partitions["trainval"] = (
        skeys_partitions["train"]
        | skeys_partitions["id_val"]
        | skeys_partitions["ood_val"]
    )
    skeys_partitions["whole"] = (
        skeys_partitions["train"]
        | skeys_partitions["id_val"]
        | skeys_partitions["id_test"]
        | skeys_partitions["ood_val"]
        | skeys_partitions["ood_test"]
    )
    return cids, img_ptrs, df_metadata, skeys_partitions


def test_nymph_data_indexes_partition_size_composition():
    cids, img_ptrs, df_metadata, skeys_partitions = _make_nymph_fixtures()
    data_indexes = build_data_indexes(cids, skeys_partitions, img_ptrs=img_ptrs, df_metadata=df_metadata)

    assert len(data_indexes["trainval"]) == (
        len(data_indexes["train"])
        + len(data_indexes["validation"]["id"])
        + len(data_indexes["validation"]["ood"])
    )
    assert len(data_indexes["whole"]) == (
        len(data_indexes["trainval"])
        + len(data_indexes["test"]["id"])
        + len(data_indexes["test"]["ood"])
    )
