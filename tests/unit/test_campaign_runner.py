from pathlib import Path
import json
import pytest

import campaign_runner as cr


def test_load_or_create_baseline_reuses_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "campaign_name", "cmp_a")
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline_a = {
        "campaign_name": "dev",
        "setting_name": "dev_setting",
        "seed": 1,
        "dataset": "cub",
        "split_name": "D10",
    }
    baseline_b = {
        "campaign_name": "changed",
        "setting_name": "changed",
        "seed": 2,
        "dataset": "lepid",
        "split_name": "dev",
    }

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: baseline_a)
    out_first = cr._load_or_create_baseline()

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: baseline_b)
    out_second = cr._load_or_create_baseline()

    assert out_first == baseline_a
    assert out_second == baseline_a



def test_run_campaign_matrix_and_dataset_outer_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "campaign_name", "cmp_b")
    monkeypatch.setattr(cr, "seed0", 42)
    monkeypatch.setattr(cr, "num_seeds", 2)
    monkeypatch.setattr(cr, "DATASETS", ("cub", "lepid"))
    monkeypatch.setattr(
        cr,
        "baseline_overrides",
        [
            {"loss": {"targ": "aligned"}, "name": "iw"},
            {"loss": {"targ": "phylo"}, "name": "hp"},
        ],
    )
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign_name": "base_campaign",
        "setting_name": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split_name": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline", lambda: baseline)

    scheduled = []

    def _fake_get_config_train(cfg_dict):
        return cfg_dict

    def _fake_run_training(cfg, imgs_mem=None):
        scheduled.append((cfg["seed"], cfg["dataset"], cfg["setting_name"], cfg["loss"]["targ"]))

    monkeypatch.setattr(cr, "get_config_train", _fake_get_config_train)
    monkeypatch.setattr(cr, "run_training", _fake_run_training)
    monkeypatch.setattr(cr, "_build_img_cache", lambda _dataset, _cfg_dict: None)

    cr.run_campaign()

    assert len(scheduled) == 8

    assert scheduled[0] == (42, "cub", "iw", "aligned")
    assert scheduled[1] == (42, "cub", "hp", "phylo")
    assert scheduled[2] == (43, "cub", "iw", "aligned")
    assert scheduled[3] == (43, "cub", "hp", "phylo")
    assert scheduled[4] == (42, "lepid", "iw", "aligned")



def test_run_campaign_writes_explicit_aligned_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "campaign_name", "cmp_c")
    monkeypatch.setattr(cr, "seed0", 7)
    monkeypatch.setattr(cr, "num_seeds", 1)
    monkeypatch.setattr(cr, "DATASETS", ("cub",))
    monkeypatch.setattr(
        cr,
        "baseline_overrides",
        [
            {"loss/targ": "aligned", "name": "iw"},
        ],
    )
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign_name": "base_campaign",
        "setting_name": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split_name": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline", lambda: baseline)
    monkeypatch.setattr(cr, "get_config_train", lambda cfg_dict: cfg_dict)
    monkeypatch.setattr(cr, "run_training", lambda _cfg, imgs_mem=None: None)
    monkeypatch.setattr(cr, "_build_img_cache", lambda _dataset, _cfg_dict: None)

    cr.run_campaign()

    fpath = Path(tmp_path) / "cmp_c" / "iw" / "overrides.json"
    assert fpath.exists()

    with open(fpath) as f:
        data = json.load(f)

    assert data["loss"]["targ"] == "aligned"


def test_expand_settings_raises_on_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate baseline_overrides name"):
        cr._expand_settings(
            [
                {"loss": {"targ": "aligned"}, "name": "dup"},
                {"loss": {"targ": "phylo"}, "name": "dup"},
            ]
        )