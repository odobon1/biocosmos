from pathlib import Path
import json
import pytest
import subprocess

import campaign_runner as cr


def test_load_or_create_baseline_reuses_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN_NAME", "cmp_a")
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
    monkeypatch.setattr(cr, "CAMPAIGN_NAME", "cmp_b")
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "NUM_SEEDS", 2)
    monkeypatch.setattr(cr, "DATASETS", ("cub", "lepid"))
    monkeypatch.setattr(
        cr,
        "BASELINE_OVERRIDES",
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

    def _fake_run_trial_subprocess(cfg_fpath: Path):
        with open(cfg_fpath) as f:
            cfg = json.load(f)
        scheduled.append((cfg["seed"], cfg["dataset"], cfg["setting_name"], cfg["loss"]["targ"]))

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign()

    assert len(scheduled) == 8

    assert scheduled[0] == (42, "cub", "iw", "aligned")
    assert scheduled[1] == (42, "cub", "hp", "phylo")
    assert scheduled[2] == (43, "cub", "iw", "aligned")
    assert scheduled[3] == (43, "cub", "hp", "phylo")
    assert scheduled[4] == (42, "lepid", "iw", "aligned")



def test_run_campaign_writes_explicit_aligned_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN_NAME", "cmp_c")
    monkeypatch.setattr(cr, "SEED0", 7)
    monkeypatch.setattr(cr, "NUM_SEEDS", 1)
    monkeypatch.setattr(cr, "DATASETS", ("cub",))
    monkeypatch.setattr(
        cr,
        "BASELINE_OVERRIDES",
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
    monkeypatch.setattr(cr, "_run_trial_subprocess", lambda _cfg_fpath: None)

    cr.run_campaign()

    fpath = Path(tmp_path) / "cmp_c" / "iw" / "overrides.json"
    assert fpath.exists()

    with open(fpath) as f:
        data = json.load(f)

    assert data["loss"]["targ"] == "aligned"


def test_expand_settings_raises_on_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate BASELINE_OVERRIDES name"):
        cr._expand_settings(
            [
                {"loss": {"targ": "aligned"}, "name": "dup"},
                {"loss": {"targ": "phylo"}, "name": "dup"},
            ]
        )


def test_log_trial_error_includes_subprocess_stderr_tail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN_NAME", "cmp_d")
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    err = subprocess.CalledProcessError(
        returncode=1,
        cmd=["torchrun", "..."],
        stderr="line1\nline2\nline3",
    )

    cr._log_trial_error(
        campaign_dir=cr._campaign_dir(),
        idx=3,
        total=10,
        seed=42,
        dataset="cub",
        setting_name="iw",
        exc=err,
    )

    log_fpath = Path(tmp_path) / "cmp_d" / "campaign_errors.log"
    assert log_fpath.exists()

    with open(log_fpath) as f:
        text = f.read()

    assert "TRIAL FAILED" in text
    assert "stderr (tail)" in text
    assert "line3" in text