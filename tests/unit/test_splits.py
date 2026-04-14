"""
python -m pytest tests/unit/test_splits.py
"""

from types import SimpleNamespace
import pandas as pd  # type: ignore[import]
import pytest  # type: ignore[import]

from preprocessing.common.splits import build_id_partitions
from preprocessing.nymph.splits_utils import build_sid_2_samp_idxs


def test_build_sid_2_samp_idxs_defaults_to_all_samples() -> None:
    sids = ["sid_a", "sid_b"]
    img_ptrs = {
        "sid_a": {0: "sid_a/img0.png", 1: "sid_a/img1.png"},
        "sid_b": {0: "sid_b/img0.png"},
    }

    sid_2_samp_idxs = build_sid_2_samp_idxs(sids, img_ptrs=img_ptrs)

    assert sid_2_samp_idxs == {
        "sid_a": [0, 1],
        "sid_b": [0],
    }


def test_build_sid_2_samp_idxs_filters_to_requested_position() -> None:
    sids = ["sid_a", "sid_b"]
    img_ptrs = {
        "sid_a": {
            0: "sid_a/img0.png",
            1: "sid_a/img1.png",
            2: "sid_a/img2.png",
        },
        "sid_b": {
            0: "sid_b/img3.png",
            1: "sid_b/img4.png",
        },
    }
    df_metadata = pd.DataFrame(
        {
            "mask_name": ["img0.png", "img1.png", "img2.png", "img3.png", "img4.png"],
            "class_dv": ["dorsal", "ventral", "dorsal", "ventral", "dorsal"],
        }
    )

    sid_2_samp_idxs = build_sid_2_samp_idxs(
        sids,
        pos_filter="dorsal",
        img_ptrs=img_ptrs,
        df_metadata=df_metadata,
    )

    assert sid_2_samp_idxs == {
        "sid_a": [0, 2],
        "sid_b": [1],
    }


def test_build_sid_2_samp_idxs_returns_empty_list_when_species_has_no_matches() -> None:
    sids = ["sid_a"]
    img_ptrs = {
        "sid_a": {
            0: "sid_a/img0.png",
            1: "sid_a/img1.png",
        },
    }
    df_metadata = pd.DataFrame(
        {
            "mask_name": ["img0.png", "img1.png"],
            "class_dv": ["ventral", "ventral"],
        }
    )

    sid_2_samp_idxs = build_sid_2_samp_idxs(
        sids,
        pos_filter="dorsal",
        img_ptrs=img_ptrs,
        df_metadata=df_metadata,
    )

    assert sid_2_samp_idxs == {"sid_a": []}


def test_gen_id_partitions_keeps_filtered_singleton_real_sample_index(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(seed=7, pct_eval=0.1)
    sids_id = {"sid_single", "sid_multi"}
    sid_2_samp_idxs = {
        "sid_single": [4],
        "sid_multi": [0, 2],
    }
    n_samps_dict = {
        "sid_single": 1,
        "sid_multi": 2,
    }

    # This test targets singleton index retention only; keep strat_split out of scope.
    monkeypatch.setattr("preprocessing.common.splits.strat_split", lambda **kwargs: (set(), set(), set()))

    skeys_train, skeys_id_val, skeys_id_test, sid_2_skeys_id, _, _ = build_id_partitions(
        sids_id=sids_id,
        sid_2_samp_idxs=sid_2_samp_idxs,
        n_samps_dict=n_samps_dict,
        cfg=cfg,
    )

    assert ("sid_single", 4) in skeys_train
    assert ("sid_single", 0) not in skeys_train
    assert sid_2_skeys_id["sid_single"] == [("sid_single", 4)]
    assert all(skey[0] != "sid_single" for skey in skeys_id_val.union(skeys_id_test))
