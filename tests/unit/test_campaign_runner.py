from pathlib import Path
import json
import pytest
import subprocess

import campaign_runner as cr


def test_load_or_create_baseline_reuses_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_a")
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline_a = {
        "campaign": "dev",
        "split": "D10",
    }
    baseline_b = {
        "campaign": "changed",
        "split": "dev",
    }

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: baseline_a)
    out_first = cr._load_or_create_baseline_config()

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: baseline_b)
    out_second = cr._load_or_create_baseline_config()

    assert out_first == baseline_a
    assert out_second == baseline_a


def test_load_or_create_manifold_viz_reuses_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_mviz")
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    cfg_a = {"tsne": {"perplexity": 30, "n_iter": 1000}}
    cfg_b = {"tsne": {"perplexity": 5, "n_iter": 250}}

    monkeypatch.setattr(cr, "load_manifold_viz_config_dict", lambda: cfg_a)
    out_first = cr._load_or_create_manifold_viz_config()

    monkeypatch.setattr(cr, "load_manifold_viz_config_dict", lambda: cfg_b)
    out_second = cr._load_or_create_manifold_viz_config()

    assert out_first == cfg_a
    assert out_second == cfg_a


def test_run_campaign_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_b")
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "NUM_SEEDS", 2)
    monkeypatch.setattr(cr, "DATASETS", ("cub", "lepid"))
    monkeypatch.setattr(
        cr,
        "BASELINE_OVERRIDES",
        [
            {"loss.targ": "aligned", "name": "iw"},
            {"loss.targ": "phylo", "name": "hp"},
        ],
    )
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda: {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}})

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict):
        scheduled.append((cfg_dict["seed"], cfg_dict["dataset"], cfg_dict["setting"], cfg_dict["loss"]["targ"]))

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign()

    assert len(scheduled) == 8

    assert set(scheduled) == {
        (seed, dataset, setting, targ)
        for seed in (42, 43)
        for dataset in ("cub", "lepid")
        for setting, targ in (("iw", "aligned"), ("hp", "phylo"))
    }



def test_run_campaign_writes_explicit_aligned_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_c")
    monkeypatch.setattr(cr, "SEED0", 7)
    monkeypatch.setattr(cr, "NUM_SEEDS", 1)
    monkeypatch.setattr(cr, "DATASETS", ("cub",))
    monkeypatch.setattr(
        cr,
        "BASELINE_OVERRIDES",
        [
            {"loss.targ": "aligned", "name": "iw"},
        ],
    )
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda: {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}})
    monkeypatch.setattr(cr, "_run_trial_subprocess", lambda _cfg_fpath: None)

    cr.run_campaign()

    fpath = Path(tmp_path) / "cmp_c" / "iw" / "overrides.json"
    assert fpath.exists()

    with open(fpath) as f:
        data = json.load(f)

    assert data["loss.targ"] == "aligned"


def test_run_campaign_marks_complete_after_successful_trial(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_complete")
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "NUM_SEEDS", 1)
    monkeypatch.setattr(cr, "DATASETS", ("cub",))
    monkeypatch.setattr(cr, "BASELINE_OVERRIDES", [{"loss.targ": "aligned", "name": "iw"}])
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda: {"n_stoch_layers": 1})

    dpath_trial = Path(tmp_path) / "cmp_complete" / "iw" / "cub" / "42"

    # a real trial writes its metadata (complete still False) + leaves a chkpts/in_progress dir behind;
    # the campaign runner is what cleans up and flips complete=True on a clean exit
    def _fake_run_trial_subprocess(cfg_dict: dict):
        (dpath_trial / "chkpts" / "in_progress").mkdir(parents=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False}, f)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign()

    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is True
    assert not (dpath_trial / "chkpts" / "in_progress").exists()


def test_run_campaign_leaves_trial_incomplete_when_subprocess_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_fail")
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "NUM_SEEDS", 1)
    monkeypatch.setattr(cr, "DATASETS", ("cub",))
    monkeypatch.setattr(cr, "BASELINE_OVERRIDES", [{"loss.targ": "aligned", "name": "iw"}])
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda: {"n_stoch_layers": 1})

    dpath_trial = Path(tmp_path) / "cmp_fail" / "iw" / "cub" / "42"

    # trial reaches final eval (writes metadata) then the subprocess crashes during finalization
    def _fake_run_trial_subprocess(cfg_dict: dict):
        dpath_trial.mkdir(parents=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False}, f)
        raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign()

    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is False


def test_expand_settings_raises_on_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate BASELINE_OVERRIDES name"):
        cr._expand_settings(
            [
                {"loss": {"targ": "aligned"}, "name": "dup"},
                {"loss": {"targ": "phylo"}, "name": "dup"},
            ]
        )


def test_run_campaign_allows_opt_override_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_opt")
    monkeypatch.setattr(cr, "SEED0", 9)
    monkeypatch.setattr(cr, "NUM_SEEDS", 1)
    monkeypatch.setattr(cr, "DATASETS", ("cub",))
    monkeypatch.setattr(
        cr,
        "BASELINE_OVERRIDES",
        [
            {"opt.l2reg": 0.33, "opt.beta2": 0.88, "name": "opt_tune"},
        ],
    )
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
        "arch": {"model_type": "clip_vitb16", "non_causal": False},
        "opt": {
            "lr": {"decay_factor": 1.0e-3},
            "l2reg": None,
            "beta1": 0.9,
            "beta2": None,
            "eps": 1.0e-6,
        },
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda: {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}})

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict):
        scheduled.append(cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign()

    assert len(scheduled) == 1
    assert scheduled[0]["opt"]["l2reg"] == 0.33
    assert scheduled[0]["opt"]["beta2"] == 0.88


def test_log_trial_error_writes_to_trial_dir_with_stderr(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_d")
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    dpath_trial = cr._dpath_campaign() / "iw" / "cub" / "42"
    err = subprocess.CalledProcessError(
        returncode=1,
        cmd=["torchrun", "..."],
        stderr="line1\nline2\nline3",
    )

    cr._log_trial_error(
        dpath_trial=dpath_trial,
        idx_trial=3,
        n_trials=10,
        seed=42,
        dataset="cub",
        setting="iw",
        exc=err,
    )

    log_fpath = dpath_trial / "error.log"
    assert log_fpath.exists()

    text = log_fpath.read_text()

    assert "TRIAL FAILED" in text
    assert "stderr" in text
    assert "line3" in text


def test_log_trial_error_strips_precrash_noise(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "CAMPAIGN", "cmp_d")
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    dpath_trial = cr._dpath_campaign() / "iw" / "cub" / "42"
    stderr = (
        "W0626 21:53:01 site-packages/torch/distributed/run.py:766] warning spam\n"
        "Eval (val):  50%|#####     | 5/10 [00:01<00:01,  4.5it/s]\n"
        "Traceback (most recent call last):\n"
        '  File "campaign_trial_runner.py", line 23, in main\n'
        "ValueError: batch_size 32000 exceeds training set size 4096\n"
    )
    err = subprocess.CalledProcessError(returncode=1, cmd=["torchrun"], stderr=stderr)

    cr._log_trial_error(
        dpath_trial=dpath_trial,
        idx_trial=1,
        n_trials=1,
        seed=42,
        dataset="cub",
        setting="iw",
        exc=err,
    )

    text = (dpath_trial / "error.log").read_text()
    assert "Traceback (most recent call last):" in text
    assert "ValueError: batch_size 32000 exceeds training set size 4096" in text
    assert "warning spam" not in text
    assert "it/s" not in text