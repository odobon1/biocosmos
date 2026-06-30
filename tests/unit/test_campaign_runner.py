from pathlib import Path
import json
import pytest
import subprocess

import campaign_runner as cr


def test_load_or_create_baseline_reuses_existing_file(tmp_path, monkeypatch) -> None:
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
    out_first = cr._load_or_create_baseline_config("cmp_a")

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: baseline_b)
    out_second = cr._load_or_create_baseline_config("cmp_a")

    assert out_first == baseline_a
    assert out_second == baseline_a


def test_load_or_create_manifold_viz_reuses_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    cfg_a = {"tsne": {"perplexity": 30, "n_iter": 1000}}
    cfg_b = {"tsne": {"perplexity": 5, "n_iter": 250}}

    monkeypatch.setattr(cr, "load_manifold_viz_config_dict", lambda: cfg_a)
    out_first = cr._load_or_create_manifold_viz_config("cmp_mviz")

    monkeypatch.setattr(cr, "load_manifold_viz_config_dict", lambda: cfg_b)
    out_second = cr._load_or_create_manifold_viz_config("cmp_mviz")

    assert out_first == cfg_a
    assert out_second == cfg_a


def test_run_campaign_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
        "dev": {"traintime_evals": False},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda campaign: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda campaign: {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}})

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append((cfg_dict["seed"], cfg_dict["dataset"], cfg_dict["setting"], cfg_dict["loss"]["targ"]))

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_b",
        n_trials=2,
        datasets=("cub", "lepid"),
        baseline_overrides=[
            {"loss.targ": "aligned", "name": "iw"},
            {"loss.targ": "phylo", "name": "hp"},
        ],
    )

    assert len(scheduled) == 8

    assert set(scheduled) == {
        (seed, dataset, setting, targ)
        for seed in (42, 43)
        for dataset in ("cub", "lepid")
        for setting, targ in (("iw", "aligned"), ("hp", "phylo"))
    }



def test_run_campaign_writes_explicit_aligned_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 7)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
        "dev": {"traintime_evals": False},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda campaign: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda campaign: {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}})
    monkeypatch.setattr(cr, "_run_trial_subprocess", lambda _cfg_dict, spare_render_pid=None: None)

    cr.run_campaign(
        campaign="cmp_c",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[
            {"loss.targ": "aligned", "name": "iw"},
        ],
    )

    fpath = Path(tmp_path) / "cmp_c" / "iw" / "overrides.json"
    assert fpath.exists()

    with open(fpath) as f:
        data = json.load(f)

    assert data["loss.targ"] == "aligned"


def test_run_campaign_marks_complete_after_successful_trial(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
        "dev": {"traintime_evals": False},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda campaign: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda campaign: {"n_stoch_layers": 1})
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    dpath_trial = Path(tmp_path) / "cmp_complete" / "iw" / "cub" / "42"

    # a real trial writes its metadata (complete still False) + leaves a chkpts/in_progress dir behind;
    # the campaign runner is what cleans up and flips complete=True on a clean exit
    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        (dpath_trial / "chkpts" / "in_progress").mkdir(parents=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False}, f)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_complete",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[{"loss.targ": "aligned", "name": "iw"}],
    )

    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is True
    assert not (dpath_trial / "chkpts" / "in_progress").exists()


def test_run_campaign_leaves_trial_incomplete_when_subprocess_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
        "dev": {"traintime_evals": False},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda campaign: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda campaign: {"n_stoch_layers": 1})

    dpath_trial = Path(tmp_path) / "cmp_fail" / "iw" / "cub" / "42"

    # trial reaches final eval (writes metadata) then the subprocess crashes during finalization
    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        dpath_trial.mkdir(parents=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False}, f)
        raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_fail",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[{"loss.targ": "aligned", "name": "iw"}],
    )

    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is False


def test_expand_settings_raises_on_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate baseline_overrides name"):
        cr._expand_settings(
            [
                {"loss": {"targ": "aligned"}, "name": "dup"},
                {"loss": {"targ": "phylo"}, "name": "dup"},
            ]
        )


def test_run_campaign_allows_opt_override_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 9)
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
        "dev": {"traintime_evals": False},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda campaign: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda campaign: {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}})

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append(cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_opt",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[
            {"opt.l2reg": 0.33, "opt.beta2": 0.88, "name": "opt_tune"},
        ],
    )

    assert len(scheduled) == 1
    assert scheduled[0]["opt"]["l2reg"] == 0.33
    assert scheduled[0]["opt"]["beta2"] == 0.88


def test_log_trial_error_writes_to_trial_dir_with_stderr(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    dpath_trial = cr._dpath_campaign("cmp_d") / "iw" / "cub" / "42"
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
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    dpath_trial = cr._dpath_campaign("cmp_d") / "iw" / "cub" / "42"
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


def test_write_manifest_buckets_and_formats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})
    dpath_campaign = Path(tmp_path) / "cmp_manifest"

    def _make_trial(setting, dataset, seed, complete=None, errored=False):
        d = dpath_campaign / setting / dataset / str(seed)
        d.mkdir(parents=True, exist_ok=True)
        if complete is not None:
            with open(d / "trial_metadata.json", "w") as f:
                json.dump({"dataset": dataset, "complete": complete}, f)
        if errored:
            (d / "error.log").write_text("boom")

    _make_trial("hp", "cub", 42, complete=True)            # completed
    _make_trial("hp", "lepid", 42, complete=False, errored=True)  # failed
    # hp/nymph/42 -> in progress: it's a resume-after-failure, so it carries a stale error.log; the
    # running trial (passed explicitly) must outrank that error.log and bucket as In Progress, not Failed
    _make_trial("hp", "nymph", 42, errored=True)
    # hp/bryo/42  -> queued (no dir at all)

    trials = [
        ("hp", "cub", 42),
        ("hp", "lepid", 42),
        ("hp", "nymph", 42),
        ("hp", "bryo", 42),
    ]
    cr._write_manifest("cmp_manifest", trials, in_progress=("hp", "nymph", 42))

    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "hp/lepid/42\n"
        "\n"
        "✅ Completed:\n"
        "hp/cub/42\n"
        "\n"
        "🏃 In Progress:\n"
        "hp/nymph/42\n"
        "\n"
        "⏳ Queued:\n"
        "hp/bryo/42\n"
    )


def test_write_manifest_completed_beats_stale_error_log(tmp_path, monkeypatch) -> None:
    # a trial that failed once then succeeded on resume keeps its old error.log; complete=True wins
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})
    dpath_campaign = Path(tmp_path) / "cmp_manifest_resume"
    d = dpath_campaign / "hp" / "cub" / "42"
    d.mkdir(parents=True)
    with open(d / "trial_metadata.json", "w") as f:
        json.dump({"dataset": "cub", "complete": True}, f)
    (d / "error.log").write_text("old failure")

    cr._write_manifest("cmp_manifest_resume", [("hp", "cub", 42)], in_progress=None)

    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "\n"
        "✅ Completed:\n"
        "hp/cub/42\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
    )


def test_write_manifest_shows_all_headers_at_kickoff(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})
    dpath_campaign = Path(tmp_path) / "cmp_manifest_kickoff"
    dpath_campaign.mkdir(parents=True)

    cr._write_manifest("cmp_manifest_kickoff", [("hp", "cub", 42), ("hp", "lepid", 42)], in_progress=None)

    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "\n"
        "✅ Completed:\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
        "hp/cub/42\n"
        "hp/lepid/42\n"
    )


def test_run_campaign_writes_manifest_tracking_outcomes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
        "dev": {"traintime_evals": False},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda campaign: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda campaign: {"n_stoch_layers": 1})

    # keep the post-trial render off the real subprocess path
    class _FakeProc:
        pid = 1234
        def poll(self): return 0
        def wait(self): return 0
        def terminate(self): pass
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: _FakeProc())

    dpath_campaign = Path(tmp_path) / "cmp_manifest_run"

    # snapshot the manifest mid-trial (the start write marks the running trial In Progress); assert after
    # the run so an AssertionError here can't be swallowed by run_campaign's per-trial except Exception.
    in_progress_snapshots = []

    # cub completes cleanly; lepid crashes during the run
    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        cur = f"{cfg_dict['setting']}/{cfg_dict['dataset']}/{cfg_dict['seed']}"
        in_progress_snapshots.append((cur, (dpath_campaign / "manifest.log").read_text()))
        d = dpath_campaign / cfg_dict["setting"] / cfg_dict["dataset"] / str(cfg_dict["seed"])
        (d / "chkpts" / "in_progress").mkdir(parents=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False}, f)
        if cfg_dict["dataset"] == "lepid":
            raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_manifest_run",
        n_trials=1,
        datasets=("cub", "lepid"),
        baseline_overrides=[{"loss.targ": "aligned", "name": "iw"}],
    )

    # each trial showed under In Progress while it was running
    assert [cur for cur, _ in in_progress_snapshots] == ["iw/cub/42", "iw/lepid/42"]
    for cur, snapshot in in_progress_snapshots:
        assert f"🏃 In Progress:\n{cur}\n" in snapshot

    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "iw/lepid/42\n"
        "\n"
        "✅ Completed:\n"
        "iw/cub/42\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
    )


def test_run_campaign_clears_in_progress_on_interrupt(tmp_path, monkeypatch) -> None:
    # a campaign abort (Ctrl-C / SIGTERM / scancel) must not leave the killed trial frozen as In Progress
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    baseline = {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "aligned", "type": "bce", "sim": "cos"},
        "dev": {"traintime_evals": False},
    }
    monkeypatch.setattr(cr, "_load_or_create_baseline_config", lambda campaign: baseline)
    monkeypatch.setattr(cr, "_load_or_create_manifold_viz_config", lambda campaign: {"n_stoch_layers": 1})

    dpath_campaign = Path(tmp_path) / "cmp_manifest_interrupt"

    # the trial gets killed mid-run: leaves chkpts/in_progress + incomplete metadata, no error.log
    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        d = dpath_campaign / cfg_dict["setting"] / cfg_dict["dataset"] / str(cfg_dict["seed"])
        (d / "chkpts" / "in_progress").mkdir(parents=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False}, f)
        raise KeyboardInterrupt

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_manifest_interrupt",
        n_trials=1,
        datasets=("cub", "lepid"),
        baseline_overrides=[{"loss.targ": "aligned", "name": "iw"}],
    )

    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "\n"
        "✅ Completed:\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
        "iw/cub/42\n"
        "iw/lepid/42\n"
    )
