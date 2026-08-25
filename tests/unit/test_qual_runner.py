import json
import pytest
import torch

import campaign_runner as cr
import qual_runner as qr


@pytest.fixture(autouse=True)
def _stub_kickoff_config_validation(monkeypatch):
    """run_campaign constructs every setting's effective TrainConfig at kickoff (fail-fast validation);
    the minimal baselines these tests inject can't build the real one (and there's no SLURM alloc in
    the test env), so stub the constructor (mirrors test_campaign_runner)."""
    monkeypatch.setattr(cr, "get_config_train", lambda cfg_dict: None)


BASE_SNAPSHOT = {
    "train": {
        "campaign": "base_campaign",
        "setting": "base_setting",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "sp", "crit": "bce", "sim": "cos"},
        "loss2": {"mix": 0.0},
        "dev": {"del_base_eval_cache": {"campaign": False, "trial": False}},
    },
    "hardware": {"max_retries": 2, "use_img_cache": False},
    "manif_viz": {"eval_duration": 1500},
    "model_specific": {},
    "dataset_specific": {},
}

OVERRIDES = {"phylo": {"loss2.mix": 0.3}, "phylo2": {"loss2.mix": 0.4}, "sp": {"loss.targ": "sp"}}


def _make_base_campaign(tmp_path, name="base", settings=("phylo", "phylo2", "sp"), datasets=("cub",), seeds=(42,), complete=True, n_gpus=None):
    """Fake completed base campaign on disk: recorded matrix in campaign_metadata.json, frozen
    cfg_baseline.json, and per-setting dirs with overrides.json + complete trials."""
    dpath = tmp_path / name
    dpath.mkdir(parents=True)
    meta = {
        "duration": "0-00:01:00",
        "commit": "deadbeef",
        "n_gpus": torch.cuda.device_count() if n_gpus is None else n_gpus,
        "n_cpus": 8,
        "ram": 64,
        "memory": {"ram": None, "vram": None},
        "n_crashes": {"ram": 0, "vram": 0, "other": 0},
        "settings": list(settings),
        "datasets": list(datasets),
        "seeds": list(seeds),
        "runtime_img_cache": {"cub": None},
    }
    (dpath / "campaign_metadata.json").write_text(json.dumps(meta))
    (dpath / "cfg_baseline.json").write_text(json.dumps(BASE_SNAPSHOT))
    for setting in settings:
        dpath_setting = dpath / "settings" / setting
        dpath_setting.mkdir(parents=True)
        (dpath_setting / "overrides.json").write_text(json.dumps(OVERRIDES[setting]))
        (dpath_setting / "setting_metadata.json").write_text(json.dumps({"n_crashes": {"ram": 0, "vram": 0, "other": 0}}))
        for dataset in datasets:
            for seed in seeds:
                d = dpath_setting / dataset / str(seed)
                d.mkdir(parents=True)
                with open(d / "trial_metadata.json", "w") as f:
                    json.dump({"dataset": dataset, "complete": complete, "runtime": {"trial": "100.0"}, "progress": {"epoch": 1, "n_epochs": 1, "n_samps_seen": 4096}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)
                (d / "data_trial.pkl").write_text("payload")  # marker: the copy carries trial contents
    return dpath


def _wire(tmp_path, monkeypatch, continue_campaign=True):
    """Point the runners at tmp_path and fake the trial subprocess (leaves the state a real one
    would; run_campaign flips complete=True). Returns the launched-trial log."""
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    monkeypatch.setattr(qr, "load_train_config_dict", lambda: {"dev": {"continue_campaign": continue_campaign}})
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        scheduled.append((cfg_dict["setting"], cfg_dict["dataset"], cfg_dict["seed"], cfg_dict["idx_seed"], cfg_dict["split"]))
        d = tmp_path / cfg_dict["campaign"] / "settings" / cfg_dict["setting"] / cfg_dict["dataset"] / str(cfg_dict["seed"])
        (d / "chkpts" / "in_progress").mkdir(parents=True, exist_ok=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "10.0"}, "progress": {"epoch": 1, "n_epochs": 1, "n_samps_seen": 4096}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)
    return scheduled


def test_run_qual_campaign_copies_qualified_and_runs_only_new_seeds(tmp_path, monkeypatch) -> None:
    scheduled = _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path)

    qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["phylo2", "sp"])

    dpath_qual = tmp_path / "base_qual"
    # the frozen config comes from the base campaign, not the live yamls
    assert json.loads((dpath_qual / "cfg_baseline.json").read_text()) == BASE_SNAPSHOT
    # qualified settings copied wholesale (overrides + trial contents, still complete); the
    # unqualified 'phylo' is left behind
    for setting in ("phylo2", "sp"):
        assert json.loads((dpath_qual / "settings" / setting / "overrides.json").read_text()) == OVERRIDES[setting]
        assert (dpath_qual / "settings" / setting / "cub" / "42" / "data_trial.pkl").read_text() == "payload"
        assert json.loads((dpath_qual / "settings" / setting / "cub" / "42" / "trial_metadata.json").read_text())["complete"] is True
    assert not (dpath_qual / "settings" / "phylo").exists()

    # only the seeds above the base campaign's run, against the frozen base config (split D10), with
    # idx_seed continuing the sweep (the manif_viz seed window keys off it)
    assert scheduled == [
        ("phylo2", "cub", 43, 1, "D10"),
        ("sp", "cub", 43, 1, "D10"),
        ("phylo2", "cub", 44, 2, "D10"),
        ("sp", "cub", 44, 2, "D10"),
    ]

    meta = json.loads((dpath_qual / "campaign_metadata.json").read_text())
    assert meta["settings"] == ["phylo2", "sp"]
    assert meta["datasets"] == ["cub"]
    assert meta["seeds"] == [42, 43, 44]


def test_run_qual_campaign_pure_copy_when_counts_match(tmp_path, monkeypatch) -> None:
    # n_trials_qual == the base's trial count -> everything is copied, nothing new runs
    scheduled = _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path)

    qr.run_qual_campaign(n_trials_qual=1, base_campaign="base", qualified_settings=["sp"])

    assert scheduled == []
    assert (tmp_path / "base_qual" / "settings" / "sp" / "cub" / "42" / "trial_metadata.json").exists()


def test_run_qual_campaign_missing_base_raises(tmp_path, monkeypatch) -> None:
    _wire(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="nope"):
        qr.run_qual_campaign(n_trials_qual=3, base_campaign="nope", qualified_settings=["sp"])
    assert not (tmp_path / "nope_qual").exists()


def test_run_qual_campaign_incomplete_base_raises_before_side_effects(tmp_path, monkeypatch) -> None:
    _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="not complete"):
        qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["sp"])
    assert not (tmp_path / "base_qual").exists()


def test_run_qual_campaign_unknown_setting_raises(tmp_path, monkeypatch) -> None:
    _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path)
    with pytest.raises(ValueError, match="not in base_campaign"):
        qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["sp", "bogus"])


def test_run_qual_campaign_rejects_bad_qualified_lists(tmp_path, monkeypatch) -> None:
    _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=[])
    with pytest.raises(ValueError, match="duplicates"):
        qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["sp", "sp"])


def test_run_qual_campaign_n_trials_below_base_raises(tmp_path, monkeypatch) -> None:
    _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path, seeds=(42, 43))
    with pytest.raises(ValueError, match="n_trials_qual"):
        qr.run_qual_campaign(n_trials_qual=1, base_campaign="base", qualified_settings=["sp"])


def test_run_qual_campaign_gpu_mismatch_raises(tmp_path, monkeypatch) -> None:
    _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path, n_gpus=torch.cuda.device_count() + 1)
    with pytest.raises(RuntimeError, match="GPU count mismatch"):
        qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["sp"])


def test_run_qual_campaign_relaunch_skips_done_and_copies_newly_qualified(tmp_path, monkeypatch) -> None:
    scheduled = _wire(tmp_path, monkeypatch)
    _make_base_campaign(tmp_path)

    qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["phylo2", "sp"])
    n_first = len(scheduled)

    # relaunch with 'phylo' newly qualified: its base trial is copied in and only its missing seeds
    # run -- everything from the first launch is already complete and skipped
    qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["phylo2", "sp", "phylo"])

    assert (tmp_path / "base_qual" / "settings" / "phylo" / "cub" / "42" / "data_trial.pkl").exists()
    assert scheduled[n_first:] == [
        ("phylo", "cub", 43, 1, "D10"),
        ("phylo", "cub", 44, 2, "D10"),
    ]
    meta = json.loads((tmp_path / "base_qual" / "campaign_metadata.json").read_text())
    assert meta["settings"] == ["phylo2", "sp", "phylo"]


def test_launch_loads_config_and_forwards_completed_flag(tmp_path, monkeypatch) -> None:
    (tmp_path / "quals").mkdir()
    (tmp_path / "quals" / "dev.yaml").write_text("n_trials_qual: 3\nbase_campaign: base\nqualified_settings: [sp]\n")
    monkeypatch.setattr(qr, "paths", {"config": tmp_path})
    calls = []
    monkeypatch.setattr(qr, "run_qual_campaign", lambda **kwargs: calls.append(kwargs) or True)

    assert qr.launch("dev") is True
    assert calls == [{"n_trials_qual": 3, "base_campaign": "base", "qualified_settings": ["sp"]}]


def test_run_qual_campaign_dedupes_name_when_not_continuing(tmp_path, monkeypatch) -> None:
    scheduled = _wire(tmp_path, monkeypatch, continue_campaign=False)
    _make_base_campaign(tmp_path)
    (tmp_path / "base_qual").mkdir()  # name taken -> falls back to base_qual2

    qr.run_qual_campaign(n_trials_qual=3, base_campaign="base", qualified_settings=["sp"])

    assert (tmp_path / "base_qual2" / "settings" / "sp" / "cub" / "42" / "trial_metadata.json").exists()
    assert {(setting, seed) for setting, _, seed, _, _ in scheduled} == {("sp", 43), ("sp", 44)}
