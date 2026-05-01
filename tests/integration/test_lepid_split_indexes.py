"""
python -m pytest tests/integration/test_lepid_split_indexes.py
"""

import pandas as pd  # type: ignore[import]
import pytest  # type: ignore[import]

from preprocessing.lepid.splits_utils import build_data_indexes_lepid


@pytest.mark.integration
def test_lepid_data_indexes_match_split_rfpath_invariants() -> None:
    cids = [
        "gena_speca",
        "genb_specb",
        "genc_specc",
        "gend_specd",
        "gene_spece",
    ]
    cid_2_family = {
        "gena_speca": "fam_a",
        "genb_specb": "fam_a",
        "genc_specc": "fam_b",
        "gend_specd": "fam_b",
        "gene_spece": "fam_c",
    }
    img_ptrs = {
        "gena_speca": {
            0: "fam_a/gena_speca/img0.png",
            1: "fam_a/gena_speca/img1.png",
        },
        "genb_specb": {
            0: "fam_a/genb_specb/img0.png",
            1: "fam_a/genb_specb/img1.png",
        },
        "genc_specc": {
            0: "fam_b/genc_specc/img0.png",
        },
        "gend_specd": {
            0: "fam_b/gend_specd/img0.png",
        },
        "gene_spece": {
            0: "fam_c/gene_spece/img0.png",
        },
    }
    df_metadata = pd.DataFrame(
        {
            "mask_name": [
                "img0.png",
                "img1.png",
            ],
            "class_dv": [
                "dorsal",
                "ventral",
            ],
            "sex": [
                "female",
                "male",
            ],
        }
    )
    skeys_partitions = {
        "train": {
            ("gena_speca", 0),
            ("genb_specb", 0),
        },
        "id_val": {
            ("gena_speca", 1),
        },
        "id_test": {
            ("genb_specb", 1),
        },
        "ood_val": {
            ("genc_specc", 0),
            ("gend_specd", 0),
        },
        "ood_test": {
            ("gene_spece", 0),
        },
    }
    skeys_partitions["trainval"] = {
        ("gena_speca", 0),
        ("genb_specb", 0),
        ("gena_speca", 1),
        ("genc_specc", 0),
        ("gend_specd", 0),
    }

    data_indexes = build_data_indexes_lepid(
        cids=cids,
        skeys_partitions=skeys_partitions,
        cid_2_family=cid_2_family,
        img_ptrs=img_ptrs,
        df_metadata=df_metadata,
    )

    rfpaths_train = {datum["rfpath"] for datum in data_indexes["train"]}
    rfpaths_trainval = {datum["rfpath"] for datum in data_indexes["trainval"]}
    rfpaths_id_val = {datum["rfpath"] for datum in data_indexes["validation"]["id"]}
    rfpaths_ood_val = {datum["rfpath"] for datum in data_indexes["validation"]["ood"]}
    rfpaths_id_test = {datum["rfpath"] for datum in data_indexes["test"]["id"]}
    rfpaths_ood_test = {datum["rfpath"] for datum in data_indexes["test"]["ood"]}

    assert (rfpaths_train | rfpaths_id_val | rfpaths_ood_val) == rfpaths_trainval
    assert len(rfpaths_trainval & (rfpaths_id_test | rfpaths_ood_test)) == 0
    assert len(rfpaths_train & (rfpaths_id_test | rfpaths_ood_test | rfpaths_id_val | rfpaths_ood_val)) == 0
    assert len(rfpaths_id_val & rfpaths_ood_val) == 0
    assert len(rfpaths_id_test & rfpaths_ood_test) == 0