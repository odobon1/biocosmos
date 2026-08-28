from pathlib import Path
import json
import os
import pytest
import subprocess
import yaml

import campaign_runner as cr
from utils.utils import PrintLog


# a single no-override coord: the degenerate no-HPO case, so a campaign's matrix is its arms alone
_BASE_COORD = [[{"name": "base"}]]


@pytest.fixture(autouse=True)
def _stub_kickoff_config_validation(monkeypatch):
    """run_campaign constructs every arm x coord's effective TrainConfig at kickoff (fail-fast validation);
    the minimal baselines these tests inject can't build the real one (and there's no SLURM alloc in the
    test env), so stub the constructor. The kickoff loop itself still runs --
    test_run_campaign_invalid_config_fails_at_kickoff re-patches it to raise."""
    monkeypatch.setattr(cr, "get_config_train", lambda cfg_dict: None)


def _dpath_trial(tmp_path, cfg_dict) -> Path:
    return (tmp_path / cfg_dict["campaign"] / "datasets" / cfg_dict["dataset"] / "arms" / cfg_dict["arm"] / "coords"
            / cfg_dict["coord"] / str(cfg_dict["seed"]))


def _leave_completed_trial(tmp_path, cfg_dict) -> None:
    """Mimic a real successful trial subprocess: leave chkpts/in_progress + incomplete metadata behind so
    run_campaign's success path (rmtree in_progress + flip complete=True) has something to act on."""
    d = _dpath_trial(tmp_path, cfg_dict)
    (d / "chkpts" / "in_progress").mkdir(parents=True, exist_ok=True)
    with open(d / "trial_metadata.json", "w") as f:
        json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"epoch": 1, "n_epochs": 35, "n_samps_seen": 200_000}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)


def _setup_completing_campaign(tmp_path, monkeypatch) -> list:
    """Wire run_campaign so trials complete cleanly without real subprocesses/renders: each fake trial
    leaves the chkpts/in_progress dir + incomplete metadata behind (the runner flips complete=True).
    Returns the list of (arm, coord, dataset, seed) tuples each launched trial was invoked with."""
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    baseline = {
        "campaign": "base_campaign",
        "arm": "base_arm",
        "coord": "base_coord",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "sp", "crit": "bce", "sim": "cos"},
        "dev": {"del_base_eval_cache": {"campaign": False, "trial": False}},
    }
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2, "use_img_cache": False},
        "manif_viz": {"eval_duration": 1500},
        "model_specific": {},
        "dataset_specific": {},
    })
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    scheduled: list = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append((cfg_dict["arm"], cfg_dict["coord"], cfg_dict["dataset"], cfg_dict["seed"]))
        d = _dpath_trial(tmp_path, cfg_dict)
        (d / "chkpts" / "in_progress").mkdir(parents=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"epoch": 1, "n_epochs": 35, "n_samps_seen": 200_000}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)
    return scheduled


def test_load_or_create_campaign_config_reuses_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    train_a = {"campaign": "dev", "split": "D10"}
    hw_a = {"mixed_prec": True, "prefetch_factor": 4}
    mviz_a = {"tsne": {"perplexity": 30, "n_iter": 1000}}
    ms_a = {"siglip": {"wd": 0.0, "beta2": 0.95}, "clip": {"wd": 0.2, "beta2": 0.98}}
    ds_a = {"cub": {"n_epochs": 100}, "lepid": {"n_epochs": 20}}

    train_b = {"campaign": "changed", "split": "dev"}
    hw_b = {"mixed_prec": False, "prefetch_factor": 2}
    mviz_b = {"tsne": {"perplexity": 5, "n_iter": 250}}
    ms_b = {"siglip": {"wd": 0.1, "beta2": 0.5}, "clip": {"wd": 0.3, "beta2": 0.7}}
    ds_b = {"cub": {"n_epochs": 5}, "lepid": {"n_epochs": 2}}

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: train_a)
    monkeypatch.setattr(cr, "load_hardware_config_dict", lambda: hw_a)
    monkeypatch.setattr(cr, "load_manif_viz_config_dict", lambda: mviz_a)
    monkeypatch.setattr(cr, "load_model_specific_config_dict", lambda: ms_a)
    monkeypatch.setattr(cr, "load_dataset_specific_config_dict", lambda: ds_a)
    out_first = cr._load_or_create_campaign_config("cmp_a")

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: train_b)
    monkeypatch.setattr(cr, "load_hardware_config_dict", lambda: hw_b)
    monkeypatch.setattr(cr, "load_manif_viz_config_dict", lambda: mviz_b)
    monkeypatch.setattr(cr, "load_model_specific_config_dict", lambda: ms_b)
    monkeypatch.setattr(cr, "load_dataset_specific_config_dict", lambda: ds_b)
    out_second = cr._load_or_create_campaign_config("cmp_a")

    # the five sources are bundled into one snapshot and frozen on first launch
    expected = {"train": train_a, "hardware": hw_a, "manif_viz": mviz_a, "model_specific": ms_a, "dataset_specific": ds_a}
    assert out_first == expected
    assert out_second == expected


def test_load_or_create_campaign_config_keeps_unresolved_nulls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    train_cfg = {
        "campaign": "dev",
        "n_epochs": None,
        "n_chkpts": None,
        "arch": {"model_type": "siglip_vitb16"},
        "opt": {"wd": None, "beta2": None},
    }
    monkeypatch.setattr(cr, "load_train_config_dict", lambda: train_cfg)
    monkeypatch.setattr(cr, "load_hardware_config_dict", lambda: {"max_retries": 2, "use_img_cache": False})
    monkeypatch.setattr(cr, "load_manif_viz_config_dict", lambda: {})
    monkeypatch.setattr(cr, "load_model_specific_config_dict", lambda: {"siglip": {"wd": 0.0, "beta2": 0.95}})
    monkeypatch.setattr(cr, "load_dataset_specific_config_dict", lambda: {"cub": {"n_epochs": 100, "n_chkpts": 50}})

    snapshot = cr._load_or_create_campaign_config("cmp_ms")

    # model-family and dataset-specific defaults are NOT resolved into the train snapshot -- they stay
    # null so a per-arm/coord arch.model_type override / the trial's dataset can pick up the matching
    # value per trial (resolution happens in the trial, from the model_specific/dataset_specific snapshots).
    assert snapshot["train"]["opt"]["wd"] is None
    assert snapshot["train"]["opt"]["beta2"] is None
    assert snapshot["train"]["n_epochs"] is None
    assert snapshot["train"]["n_chkpts"] is None


def _stub_campaign_config(monkeypatch, dev=None, hardware=None, manif_viz=None, train_extra=None) -> dict:
    """Inject a minimal frozen campaign snapshot; returns its train baseline."""
    baseline = {
        "campaign": "base_campaign",
        "arm": "base_arm",
        "coord": "base_coord",
        "seed": 0,
        "dataset": "cub",
        "split": "D10",
        "loss": {"targ": "sp", "crit": "bce", "sim": "cos"},
        "dev": dev or {"del_base_eval_cache": {"campaign": False, "trial": False}},
        **(train_extra or {}),
    }
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": hardware or {"max_retries": 2, "use_img_cache": False},
        "manif_viz": manif_viz or {"eval_duration": 1500},
        "model_specific": {},
        "dataset_specific": {},
    })
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)
    return baseline


def test_run_campaign_matrix(tmp_path, monkeypatch) -> None:
    # arms x coords x datasets x seeds: every combination is launched once, with the arm's and the
    # coord's overrides both applied to the trial config
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch, train_extra={"opt": {"lr": {"init": 1.0e-5}}})

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append((cfg_dict["seed"], cfg_dict["dataset"], cfg_dict["arm"], cfg_dict["coord"],
                          cfg_dict["loss"]["targ"], cfg_dict["opt"]["lr"]["init"]))
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_b",
        n_trials=2,
        datasets=("cub", "lepid"),
        ablation_arms=[[
            {"loss.targ": "sp", "name": "sp"},
            {"loss.targ": "phylo", "name": "hp"},
        ]],
        hpo_coords=[[{"opt.lr.init": [2.0e-5, 2.0e-6]}]],
    )

    assert len(scheduled) == 16

    assert set(scheduled) == {
        (seed, dataset, arm, coord, targ, lr)
        for seed in (42, 43)
        for dataset in ("cub", "lepid")
        for arm, targ in (("sp", "sp"), ("hp", "phylo"))
        for coord, lr in (("LR-2.0e-5", 2.0e-5), ("LR-2.0e-6", 2.0e-6))
    }
    # seed-major, then dataset, arm, coord (the inner loop) -- the cycle order the stats levels key off
    assert scheduled[:4] == [
        (42, "cub", "sp", "LR-2.0e-5", "sp", 2.0e-5),
        (42, "cub", "sp", "LR-2.0e-6", "sp", 2.0e-6),
        (42, "cub", "hp", "LR-2.0e-5", "phylo", 2.0e-5),
        (42, "cub", "hp", "LR-2.0e-6", "phylo", 2.0e-6),
    ]
    assert scheduled[4][1] == "lepid" and scheduled[8][0] == 43

    meta = json.loads((tmp_path / "cmp_b" / "campaign_metadata.json").read_text())
    assert meta["arms"] == ["sp", "hp"]
    assert meta["coords"] == ["LR-2.0e-5", "LR-2.0e-6"]


def test_run_campaign_raises_on_arm_coord_key_collision_before_side_effects(tmp_path, monkeypatch) -> None:
    # arms and coords merge into one trial config, so a key claimed by both is rejected at kickoff, before
    # any filesystem side effect (no campaign dir)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    with pytest.raises(ValueError, match=r"loss\.targ.*appear in both ablation_arms and hpo_coords"):
        cr.run_campaign(
            campaign="cmp_collide",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
            hpo_coords=[[{"loss.targ": "mp", "name": "mp"}]],
        )

    assert not (tmp_path / "cmp_collide").exists()


def test_run_campaign_writes_split_overrides(tmp_path, monkeypatch) -> None:
    # a coord dir's overrides.json records the arm's and the coord's declared overrides separately
    monkeypatch.setattr(cr, "SEED0", 7)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch, train_extra={"opt": {"lr": {"init": 1.0e-5}}})
    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        scheduled.append(cfg_dict["_overrides"])
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_c",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=[[{"opt.lr.init": 2.0e-5, "name": "lr"}]],
    )

    fpath = tmp_path / "cmp_c" / "datasets" / "cub" / "arms" / "sp" / "coords" / "lr" / "overrides.json"
    assert fpath.exists()
    with open(fpath) as f:
        assert json.load(f) == {"arm": {"loss.targ": "sp"}, "coord": {"opt.lr.init": 2.0e-5}}
    # the trial itself gets the merged set
    assert scheduled == [{"loss.targ": "sp", "opt.lr.init": 2.0e-5}]


def test_run_campaign_defers_coord_dir_until_trial_launch(tmp_path, monkeypatch) -> None:
    # a coord's dir (datasets/<dataset>/arms/<arm>/coords/<coord>/, holding overrides.json) is created at
    # its first trial's launch, not at campaign kickoff -- a planned arm whose trials never start leaves
    # no arms/<arm>/ dir
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch)

    dpath_arms = tmp_path / "cmp_defer" / "datasets" / "cub" / "arms"

    # abort the campaign during the first trial (arm 'sp'): its overrides.json must already be
    # in place at launch, while 'hp' -- planned but never launched -- must have no dir at all
    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        assert (dpath_arms / cfg_dict["arm"] / "coords" / cfg_dict["coord"] / "overrides.json").exists()
        raise KeyboardInterrupt

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_defer",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[
            {"loss.targ": "sp", "name": "sp"},
            {"loss.targ": "phylo", "name": "hp"},
        ]],
        hpo_coords=_BASE_COORD,
    )

    assert (dpath_arms / "sp" / "coords" / "base" / "overrides.json").exists()
    assert not (dpath_arms / "hp").exists()


def test_run_campaign_marks_complete_after_successful_trial(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch)

    dpath_trial = tmp_path / "cmp_complete" / "datasets" / "cub" / "arms" / "sp" / "coords" / "base" / "42"

    # a real trial writes its metadata (complete still False) + leaves a chkpts/in_progress dir behind;
    # the campaign runner is what cleans up and flips complete=True on a clean exit
    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        (dpath_trial / "chkpts" / "in_progress").mkdir(parents=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"epoch": 1, "n_epochs": 35, "n_samps_seen": 200_000}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_complete",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is True
    assert not (dpath_trial / "chkpts").exists()  # the whole tree goes: it only ever held the resume state


def _campaign_table_fpaths(dpath_campaign: Path, dataset: str, arm: str) -> list[Path]:
    groups = ("native", "native_macro", "joint", "joint_macro")
    dpath_dataset = dpath_campaign / "datasets" / dataset
    fpaths = []
    for criterion in ("map", "acc"):
        for group in groups:
            fpaths.append(dpath_dataset / "arms" / arm / "arm_stats" / criterion / group / "metrics.png")
            for kind in ("arm_coords", "arms"):
                fpaths.append(dpath_dataset / "dataset_stats" / kind / criterion / group / "metrics.png")
                fpaths.append(dpath_campaign / "campaign_stats" / kind / criterion / f"{group}.xlsx")
    return fpaths


@pytest.mark.parametrize("interrupted", [False, True])
def test_run_campaign_renders_tables_at_exit(tmp_path, monkeypatch, interrupted: bool) -> None:
    # trials render each stats level only when a seed completes across that level's cycle, so the runner
    # renders every level once on the way out -- covering a campaign that ends mid-cycle, whether it ran
    # to the end (a trial that never succeeds) or was interrupted
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch, hardware={"max_retries": 0, "use_img_cache": False})

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        if interrupted:
            raise KeyboardInterrupt
        raise RuntimeError("trial died")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    completed = cr.run_campaign(
        campaign="cmp_render",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    # an interrupted run reports False so the campaign queue stops instead of launching the next entry
    assert completed is (not interrupted)
    # no trial ever completed, so the tables are empty -- the point is that they were written at all
    for fpath in _campaign_table_fpaths(tmp_path / "cmp_render", "cub", "sp"):
        assert fpath.exists(), fpath


def test_run_campaign_del_base_eval_cache_campaign_deletes_only_at_creation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache", "root": tmp_path / "root"})
    _stub_campaign_config(monkeypatch, dev={"del_base_eval_cache": {"campaign": True, "trial": False}})

    dpath_cache = tmp_path / "root" / "base_eval_cache"
    dpath_cache.mkdir(parents=True)
    (dpath_cache / "combo.pkl").touch()

    seen_at_launch = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        seen_at_launch.append(dpath_cache.exists())
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    kwargs = dict(campaign="cmp_delc", n_trials=1, datasets=("cub",), ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]], hpo_coords=_BASE_COORD)
    cr.run_campaign(**kwargs)
    assert seen_at_launch == [False]  # first launch: cache deleted before the trial ran

    # a relaunch of the existing campaign is not a new beginning: the cache survives it
    dpath_cache.mkdir(parents=True)
    (dpath_cache / "combo.pkl").touch()
    cr.run_campaign(**kwargs)
    assert (dpath_cache / "combo.pkl").exists()


def test_run_campaign_del_base_eval_cache_trial_deletes_before_each_trial(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache", "root": tmp_path / "root"})
    _stub_campaign_config(monkeypatch, dev={"del_base_eval_cache": {"campaign": False, "trial": True}})

    dpath_cache = tmp_path / "root" / "base_eval_cache"
    dpath_cache.mkdir(parents=True)
    (dpath_cache / "combo.pkl").touch()

    seen_at_launch = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        # record whether the cache survived to this trial's launch, then rebuild it the way the
        # trial's base eval would
        seen_at_launch.append(dpath_cache.exists())
        dpath_cache.mkdir(parents=True)
        (dpath_cache / "combo.pkl").touch()
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_delt",
        n_trials=2,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    assert seen_at_launch == [False, False]  # deleted before every trial, not just the first
    assert (dpath_cache / "combo.pkl").exists()  # the last trial's rebuild is left in place


def test_run_campaign_retries_then_fails_trial_without_progress(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch)

    dpath_trial = tmp_path / "cmp_fail" / "datasets" / "cub" / "arms" / "sp" / "coords" / "base" / "42"

    # every attempt crashes without ever writing a checkpoint (no forward progress), so the runner retries
    # up to the no-progress cap and then gives up, leaving the trial incomplete with an error.log.
    # each attempt fails a different way (vram, ram, other) to exercise the crash classifier end-to-end
    crash_stderrs = [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB",
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: Signal 9 (SIGKILL) received by PID 12345",
        "boom",
    ]
    calls = []
    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        calls.append((cfg_dict["arm"], cfg_dict["coord"], cfg_dict["dataset"], cfg_dict["seed"]))
        dpath_trial.mkdir(parents=True, exist_ok=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"epoch": 1, "n_epochs": 35, "n_samps_seen": 200_000}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)
        raise subprocess.CalledProcessError(1, ["torchrun"], stderr=crash_stderrs[len(calls) - 1])

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_fail",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    # one initial attempt + max_retries (2, from the injected hardware config) no-progress resume attempts
    assert len(calls) == 2 + 1
    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is False
    # the fatal error.log carries the aggregate failure= label (vram + ram + other crashes -> Mixed),
    # which the manifest surfaces on the Failed line
    assert "failure=Mixed" in (dpath_trial / "error.log").read_text()
    assert "--- Mixed" in (tmp_path / "cmp_fail" / "manifest.log").read_text(encoding="utf-8")
    # every crash (all three, at the same 200k checkpoint) also lands its own file under errors/
    assert sorted(p.name for p in (dpath_trial / "errors").iterdir()) == [
        "error-200000-0.log", "error-200000-1.log", "error-200000-2.log",
    ]
    # all three crashes are tallied at the campaign level, bucketed by cause (that file is never rewritten
    # by the mock); the mock rewrites trial_metadata fresh each attempt, so the trial-level counter
    # reflects only the last one (an 'other' crash)
    with open(tmp_path / "cmp_fail" / "campaign_metadata.json") as f:
        assert json.load(f)["n_crashes"] == {"ram": 1, "vram": 1, "other": 1}
    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["n_crashes"] == {"ram": 0, "vram": 0, "other": 1}


def test_bump_crash_counts_reaches_coord_metadata(tmp_path) -> None:
    # the coord-level counter (coord_metadata.json, the trial dir's parent) is bumped alongside the trial's
    # and the campaign's, and only when it exists (a crash can precede the subprocess writing it)
    dpath_campaign = tmp_path / "cmp"
    dpath_trial = dpath_campaign / "datasets" / "cub" / "arms" / "sp" / "coords" / "base" / "42"
    dpath_trial.mkdir(parents=True)
    (dpath_campaign / "campaign_metadata.json").write_text(json.dumps({"n_crashes": {"ram": 0, "vram": 0, "other": 0}}))

    cr._bump_crash_counts(dpath_trial, dpath_campaign, "vram")
    assert json.loads((dpath_campaign / "campaign_metadata.json").read_text())["n_crashes"] == {"ram": 0, "vram": 1, "other": 0}

    (dpath_trial.parent / "coord_metadata.json").write_text(json.dumps({"n_crashes": {"ram": 0, "vram": 0, "other": 0}}))
    cr._bump_crash_counts(dpath_trial, dpath_campaign, "ram")
    assert json.loads((dpath_trial.parent / "coord_metadata.json").read_text())["n_crashes"] == {"ram": 1, "vram": 0, "other": 0}
    assert json.loads((dpath_campaign / "campaign_metadata.json").read_text())["n_crashes"] == {"ram": 1, "vram": 1, "other": 0}


def test_run_campaign_invalid_config_fails_at_kickoff(tmp_path, monkeypatch) -> None:
    # an arm x coord whose effective TrainConfig fails validation kills the campaign at kickoff, with the
    # failing combination named -- no trial is ever launched
    scheduled = _setup_completing_campaign(tmp_path, monkeypatch)

    def _fake_get_config_train(cfg_dict):
        if cfg_dict["arm"] == "bad":
            raise ValueError("batch_size (1024) must be an exact multiple of world_size (2) x hardware.loss_chunk_size (1024)")

    monkeypatch.setattr(cr, "get_config_train", _fake_get_config_train)

    with pytest.raises(ValueError, match="invalid config for arm 'bad' / coord 'base' on dataset 'cub'"):
        cr.run_campaign(
            campaign="cmp_badcfg",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[{"loss.targ": "sp", "name": "sp"}, {"loss.targ": "mp", "name": "bad"}]],
            hpo_coords=_BASE_COORD,
        )
    assert scheduled == []


def test_classify_crash_buckets() -> None:
    def cpe(stderr):
        return subprocess.CalledProcessError(1, ["torchrun"], stderr=stderr)

    assert cr._classify_crash(cpe("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB")) == "vram"
    assert cr._classify_crash(cpe("Signal 9 (SIGKILL) received by PID 12345")) == "ram"
    assert cr._classify_crash(cpe("RuntimeError: DataLoader worker (pid 123) is killed by signal: Killed.")) == "ram"
    # a CUDA OOM's teardown can drag SIGKILL noise into stderr -- the root cause wins
    assert cr._classify_crash(cpe("CUDA out of memory\nSignal 9 (SIGKILL) received by PID 12345")) == "vram"
    assert cr._classify_crash(cpe("boom")) == "other"
    assert cr._classify_crash(RuntimeError("crashed before the subprocess produced stderr")) == "other"


def test_run_campaign_retries_recover_across_flakes_that_make_progress(tmp_path, monkeypatch) -> None:
    # a trial that flakes repeatedly but advances its checkpoint each time is resumed indefinitely: the
    # no-progress counter resets on every forward step, so more flakes than the cap still recover.
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch)

    dpath_trial = tmp_path / "cmp_flaky" / "datasets" / "cub" / "arms" / "sp" / "coords" / "base" / "42"
    fpath_ckpt = dpath_trial / "chkpts" / "in_progress" / "train_state.pt"

    # flake on max_retries+1 attempts (more than the no-progress cap of 2 injected above), but advance the
    # checkpoint before each crash; succeed on the next. Each flake made progress, so none count as stalled.
    n_flakes = 2 + 1
    calls = {"n": 0}
    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        calls["n"] += 1
        fpath_ckpt.parent.mkdir(parents=True, exist_ok=True)
        fpath_ckpt.write_text(f"state-{calls['n']}")
        os.utime(fpath_ckpt, (calls["n"] * 1000, calls["n"] * 1000))  # strictly-increasing mtime = progress
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"epoch": 1, "n_epochs": 35, "n_samps_seen": 200_000}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)
        if calls["n"] <= n_flakes:
            raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_flaky",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    assert calls["n"] == n_flakes + 1  # every progressing flake was retried; final attempt completed
    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is True
    assert not fpath_ckpt.parent.exists()  # chkpts/in_progress removed on success
    assert not (dpath_trial / "error.log").exists()
    with open(tmp_path / "cmp_flaky" / "campaign_metadata.json") as f:
        assert json.load(f)["n_crashes"] == {"ram": 0, "vram": 0, "other": n_flakes}  # each recovered flake is counted at the campaign level


def test_expand_combo_groups_raises_on_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate hpo_coords name"):
        cr._expand_combo_groups(
            [
                [
                    {"loss.targ": "sp", "name": "dup"},
                    {"loss.targ": "phylo", "name": "dup"},
                ]
            ],
            "hpo_coords",
        )


def test_expand_combo_groups_rejects_empty_list() -> None:
    # zero combo groups would expand to one nameless member (an empty Cartesian product), whose dir
    # name would vanish from the artifact path -- a no-HPO campaign spells its single coord out instead
    with pytest.raises(ValueError, match="hpo_coords must list at least one combo group"):
        cr._expand_combo_groups([], "hpo_coords")
    assert cr._expand_combo_groups(_BASE_COORD, "hpo_coords") == [("base", {})]


def test_expand_combo_groups_single_combo_group_unchanged() -> None:
    # one combo group -> members expand unchanged: names are not joined and payloads carry through as-is
    members = cr._expand_combo_groups(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                {"loss.targ": "sp", "name": "sp"},
            ]
        ],
        "ablation_arms",
    )
    assert members == [
        ("hp", {"loss2.mix": 0.3, "loss2.targ": "phylo"}),
        ("sp", {"loss.targ": "sp"}),
    ]


def test_expand_combo_groups_derives_name_from_overrides_when_omitted() -> None:
    # an item without a 'name' is named by its overrides: 'key-value' pairs joined by '_', with
    # keys/values mapped through CFG_PARAM_ALIASES / CFG_PARAM_VALUE_ALIASES when an alias exists
    # (e.g. loss2.mix -> Mix, mp -> MP) and anything unaliased passing through verbatim
    members = cr._expand_combo_groups(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo"},
                {"loss.targ": "mp"},
                {"loss.targ": "sp", "name": "sp"},
                {"loss.sim": "geo1"},
            ]
        ],
        "ablation_arms",
    )
    assert members == [
        ("Mix-0.3_Targ2-hp", {"loss2.mix": 0.3, "loss2.targ": "phylo"}),
        ("Targ-MP", {"loss.targ": "mp"}),
        ("sp", {"loss.targ": "sp"}),
        ("loss.sim-geo1", {"loss.sim": "geo1"}),
    ]


def test_expand_combo_groups_universal_value_aliases() -> None:
    # True/False/None values map through CFG_UNIVERSAL_VALUE_ALIASES (key-independent) -> T/F/N;
    # numeric values equal to a bool (0.0 == False, 1.0 == True) must NOT alias -- the lookup is
    # identity-guarded against Python's bool/int equality
    members = cr._expand_combo_groups(
        [
            [
                {"htarg_shuf": True, "chain_floor": None},
                {"dv_batching": False},
                {"loss2.mix": 0.0},
                {"loss2.mix": 1.0},
            ]
        ],
        "ablation_arms",
    )
    assert [name for name, _ in members] == [
        "htarg_shuf-T_chain_floor-N",
        "dv_batching-F",
        "Mix-0.0",
        "Mix-1.0",
    ]


def test_expand_combo_groups_combo_list_expands_item_and_appends_to_name() -> None:
    # a list-valued override is a combo list: the item expands to one member per list value, the
    # (aliased) 'key-value' pair appended to the explicit 'name'
    members = cr._expand_combo_groups(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": [1024, 2048], "name": "hp"},
                {"loss.targ": "mp", "batch_size": [1024, 2048], "name": "mp"},
            ]
        ],
        "ablation_arms",
    )
    assert members == [
        ("hp_BS-1k", {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 1024}),
        ("hp_BS-2k", {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 2048}),
        ("mp_BS-1k", {"loss.targ": "mp", "batch_size": 1024}),
        ("mp_BS-2k", {"loss.targ": "mp", "batch_size": 2048}),
    ]


def test_fmt_name_value_floats_read_like_yaml() -> None:
    # nonzero floats below 1e-2 render in scientific notation with the shortest mantissa (a decimal point
    # kept, the exponent unpadded) -- including the 1e-4 .. 1e-2 band Python itself renders as decimals
    # (0.0002 -> '0.0002'), so an LR sweep reads uniformly; larger floats, zero, ints and strings pass
    # through str()
    assert cr._fmt_name_value(2.0e-4) == "2.0e-4"
    assert cr._fmt_name_value(7e-06) == "7.0e-6"
    assert cr._fmt_name_value(1.131e-4) == "1.131e-4"
    assert cr._fmt_name_value(2.828e-5) == "2.828e-5"
    assert cr._fmt_name_value(1.0e-3) == "1.0e-3"
    assert cr._fmt_name_value(-2.0e-4) == "-2.0e-4"
    assert cr._fmt_name_value(0.01) == "0.01"
    assert cr._fmt_name_value(0.05) == "0.05"
    assert cr._fmt_name_value(0.3) == "0.3"
    assert cr._fmt_name_value(0.0) == "0.0"
    assert cr._fmt_name_value(2048) == "2048"
    assert cr._fmt_name_value("sp") == "sp"


def test_expand_combo_groups_multiple_combo_lists_cross_within_item() -> None:
    # several combo lists in one item cross with each other, the last-listed key varying fastest;
    # scientific-notation floats read like the YAML that declared them (7e-06 -> '7.0e-6', 2.0e-4 ->
    # '2.0e-4' rather than Python's '0.0002')
    members = cr._expand_combo_groups(
        [
            [
                {"loss2.mix": 0.3, "batch_size": [1024, 2048], "opt.lr.init": [7.0e-6, 2.0e-4], "name": "hp"},
            ]
        ],
        "hpo_coords",
    )
    assert [name for name, _ in members] == [
        "hp_BS-1k_LR-7.0e-6",
        "hp_BS-1k_LR-2.0e-4",
        "hp_BS-2k_LR-7.0e-6",
        "hp_BS-2k_LR-2.0e-4",
    ]
    assert dict(members)["hp_BS-2k_LR-2.0e-4"] == {"loss2.mix": 0.3, "batch_size": 2048, "opt.lr.init": 2.0e-4}


def test_expand_combo_groups_combo_list_in_unnamed_item_folds_into_derived_name() -> None:
    # in an unnamed item the expanded value is named like any other override, in declared position --
    # the dev_new.yaml shape: unnamed combo-list items, one per combo group, crossing into a coord grid
    members = cr._expand_combo_groups([[{"loss.targ": "mp", "batch_size": [1024, 2048]}]], "ablation_arms")
    assert members == [
        ("Targ-MP_BS-1k", {"loss.targ": "mp", "batch_size": 1024}),
        ("Targ-MP_BS-2k", {"loss.targ": "mp", "batch_size": 2048}),
    ]
    coords = cr._expand_combo_groups(
        [[{"opt.lr.init": [2.0e-6, 2.0e-5]}], [{"loss.logits.temp.init": [0.0, 0.1]}]],
        "hpo_coords",
    )
    assert [name for name, _ in coords] == [
        "LR-2.0e-6_Tau-0.0", "LR-2.0e-6_Tau-0.1", "LR-2.0e-5_Tau-0.0", "LR-2.0e-5_Tau-0.1",
    ]
    assert dict(coords)["LR-2.0e-5_Tau-0.1"] == {"opt.lr.init": 2.0e-5, "loss.logits.temp.init": 0.1}


def test_expand_combo_groups_derived_and_explicit_names_join_across_combo_groups() -> None:
    # derived names compose with explicit ones the same way in the cross-combo-group join
    members = cr._expand_combo_groups(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                {"loss.targ": "mp"},
            ],
            [
                {"batch_size": 2048, "name": "2k"},
                {"batch_size": 1024},
            ],
        ],
        "ablation_arms",
    )

    assert len(members) == 4
    assert dict(members) == {
        "hp_2k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 2048},
        "hp_BS-1k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 1024},
        "Targ-MP_2k": {"loss.targ": "mp", "batch_size": 2048},
        "Targ-MP_BS-1k": {"loss.targ": "mp", "batch_size": 1024},
    }


def test_expand_combo_groups_cartesian_product_merges_and_joins_names() -> None:
    # two combo groups -> every cross-combo-group combination; payloads merge, names join with '_' in combo-group order
    members = cr._expand_combo_groups(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                {"loss.targ": "mp", "name": "mp"},
            ],
            [
                {"batch_size": 2048, "name": "2k"},
                {"batch_size": 1024, "name": "1k"},
            ],
        ],
        "ablation_arms",
    )

    assert len(members) == 4
    assert dict(members) == {
        "hp_2k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 2048},
        "hp_1k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 1024},
        "mp_2k": {"loss.targ": "mp", "batch_size": 2048},
        "mp_1k": {"loss.targ": "mp", "batch_size": 1024},
    }


def test_expand_combo_groups_generalizes_to_three_combo_groups() -> None:
    members = cr._expand_combo_groups(
        [
            [{"a": 1, "name": "x"}, {"a": 2, "name": "y"}],
            [{"b": 1, "name": "p"}, {"b": 2, "name": "q"}],
            [{"c": 1, "name": "m"}, {"c": 2, "name": "n"}],
        ],
        "ablation_arms",
    )

    assert len(members) == 8
    assert {name for name, _ in members} == {
        "x_p_m", "x_p_n", "x_q_m", "x_q_n",
        "y_p_m", "y_p_n", "y_q_m", "y_q_n",
    }
    assert dict(members)["y_q_n"] == {"a": 2, "b": 2, "c": 2}


def test_expand_combo_groups_raises_on_cross_combo_group_key_collision() -> None:
    # the same override key appears in two combo groups -> two values would fight to define it when merged
    with pytest.raises(ValueError, match="ablation_arms key.*collide between combo groups"):
        cr._expand_combo_groups(
            [
                [
                    {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                    {"loss.targ": "mp", "name": "mp"},
                ],
                [
                    {"loss2.mix": 0.4, "loss2.targ": "phylo", "name": "hp4"},
                    {"loss.targ": "mp", "name": "sw2"},
                ],
            ],
            "ablation_arms",
        )


def test_expand_matrix_raises_on_arm_coord_key_collision() -> None:
    # arms and coords are one override space: a key in an arm combo group and a coord combo group collides
    # exactly like one shared between two combo groups of the same list
    with pytest.raises(ValueError, match=r"batch_size.*appear in both ablation_arms and hpo_coords"):
        cr._expand_matrix(
            [[{"loss.targ": "mp", "batch_size": 2048, "name": "mp"}]],
            [[{"batch_size": [1024, 2048]}]],
        )
    arms, coords = cr._expand_matrix(
        [[{"loss.targ": "mp", "name": "mp"}, {"loss.targ": "sp", "name": "sp"}]],
        [[{"batch_size": [1024, 2048]}]],
    )
    assert arms == [("mp", {"loss.targ": "mp"}), ("sp", {"loss.targ": "sp"})]
    assert coords == [("BS-1k", {"batch_size": 1024}), ("BS-2k", {"batch_size": 2048})]


def test_expand_combo_groups_null_name_skips_component() -> None:
    # an item with an explicit `name: null` contributes no name component; the override still applies,
    # so the member is named by the other combo groups alone (e.g. 'hp' x null -> 'hp')
    members = cr._expand_combo_groups(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                {"loss.targ": "mp", "name": "mp"},
            ],
            [
                {"loss.wting.cls_imb.type": "inv_freq", "name": "if"},
                {"loss.wting.cls_imb.type": "class_bal", "name": None},
            ],
        ],
        "ablation_arms",
    )
    assert dict(members) == {
        "hp_if": {"loss2.mix": 0.3, "loss2.targ": "phylo", "loss.wting.cls_imb.type": "inv_freq"},
        "hp": {"loss2.mix": 0.3, "loss2.targ": "phylo", "loss.wting.cls_imb.type": "class_bal"},
        "mp_if": {"loss.targ": "mp", "loss.wting.cls_imb.type": "inv_freq"},
        "mp": {"loss.targ": "mp", "loss.wting.cls_imb.type": "class_bal"},
    }


def test_expand_combo_groups_raises_on_multiple_null_names_in_group() -> None:
    # two null-named items in one combo group would give two members the same joined name
    with pytest.raises(ValueError, match="at most one item per combo group may set"):
        cr._expand_combo_groups(
            [
                [
                    {"loss.targ": "mp", "name": "mp"},
                    {"loss.targ": "sp", "name": None},
                    {"loss.targ": "phylo", "name": None},
                ]
            ],
            "ablation_arms",
        )


def test_expand_combo_groups_raises_when_every_combo_group_has_null_name() -> None:
    # a null item in every combo group -> the all-null combination yields an empty name
    with pytest.raises(ValueError, match="at least one combo group must have"):
        cr._expand_combo_groups(
            [
                [
                    {"loss.targ": "mp", "name": "mp"},
                    {"loss.targ": "sp", "name": None},
                ],
                [
                    {"batch_size": 2048, "name": "2k"},
                    {"batch_size": 1024, "name": None},
                ],
            ],
            "ablation_arms",
        )


def test_run_campaign_crosses_arms_with_coords(tmp_path, monkeypatch) -> None:
    # the arms x coords product flows through run_campaign: each combination schedules with both sides'
    # overrides merged into its config, and its coord dir gets the two sides recorded separately
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch)

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append((cfg_dict["arm"], cfg_dict["coord"], cfg_dict["loss"]["targ"], cfg_dict["loss"]["sim"]))
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_groups",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}, {"loss.targ": "phylo", "name": "hp"}]],
        hpo_coords=[[{"loss.sim": "cos", "name": "cos"}, {"loss.sim": "l2", "name": "l2"}]],
    )

    assert set(scheduled) == {
        ("sp", "cos", "sp", "cos"),
        ("sp", "l2", "sp", "l2"),
        ("hp", "cos", "phylo", "cos"),
        ("hp", "l2", "phylo", "l2"),
    }

    with open(tmp_path / "cmp_groups" / "datasets" / "cub" / "arms" / "hp" / "coords" / "l2" / "overrides.json") as f:
        data = json.load(f)
    assert data == {"arm": {"loss.targ": "phylo"}, "coord": {"loss.sim": "l2"}}


def test_run_campaign_allows_opt_override_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 9)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch, train_extra={
        "arch": {"model_type": "clip_vitb16", "clip": {"non_causal": False}},
        "opt": {
            "lr": {"decay_factor": 1.0e-3},
            "wd": None,
            "beta1": 0.9,
            "beta2": None,
            "eps": 1.0e-6,
        },
    })

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append(cfg_dict)
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_opt",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"opt.wd": 0.33, "name": "wd"}]],
        hpo_coords=[[{"opt.beta2": 0.88, "name": "b2"}]],
    )

    assert len(scheduled) == 1
    assert scheduled[0]["opt"]["wd"] == 0.33
    assert scheduled[0]["opt"]["beta2"] == 0.88


def test_log_trial_error_writes_to_trial_dir_with_stderr(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    dpath_trial = cr._dpath_campaign("cmp_d") / "sp" / "base" / "cub" / "42"
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
        arm="sp",
        coord="base",
        exc=err,
        failure="VRAM",
    )

    log_fpath = dpath_trial / "error.log"
    assert log_fpath.exists()

    text = log_fpath.read_text()

    assert "TRIAL FAILED" in text
    assert "arm=sp, coord=base, failure=VRAM" in text  # the marker PrintLog.manifest parses for the Failed section
    assert "stderr" in text
    assert "line3" in text


def test_log_trial_error_strips_precrash_noise(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    dpath_trial = cr._dpath_campaign("cmp_d") / "sp" / "base" / "cub" / "42"
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
        arm="sp",
        coord="base",
        exc=err,
        failure="Other",
    )

    text = (dpath_trial / "error.log").read_text()
    assert "Traceback (most recent call last):" in text
    assert "ValueError: batch_size 32000 exceeds training set size 4096" in text
    assert "warning spam" not in text
    assert "it/s" not in text


def test_log_crash_writes_per_crash_files_indexed_by_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    dpath_trial = cr._dpath_campaign("cmp_c") / "sp" / "base" / "cub" / "42"
    dpath_trial.mkdir(parents=True, exist_ok=True)

    # errors/ is created only once a crash occurs; before the first checkpoint there's no metadata, so
    # progress reads as 0
    assert not (dpath_trial / "errors").exists()
    cr._log_crash(dpath_trial, subprocess.CalledProcessError(1, ["torchrun"], stderr="boom-early"))
    assert (dpath_trial / "errors" / "error-0-0.log").exists()
    assert "boom-early" in (dpath_trial / "errors" / "error-0-0.log").read_text()

    # once a checkpoint advances progress, crashes are keyed by samples seen; repeated crashes at the same
    # sample count get incrementing indices instead of clobbering
    with open(dpath_trial / "trial_metadata.json", "w") as f:
        json.dump({"progress": {"epoch": 3, "n_epochs": 35, "n_samps_seen": 1_058_816}}, f)
    cr._log_crash(dpath_trial, subprocess.CalledProcessError(1, ["torchrun"], stderr="boom-a"))
    cr._log_crash(dpath_trial, subprocess.CalledProcessError(1, ["torchrun"], stderr="boom-b"))

    assert "boom-a" in (dpath_trial / "errors" / "error-1058816-0.log").read_text()
    assert "boom-b" in (dpath_trial / "errors" / "error-1058816-1.log").read_text()


def test_manifest_buckets_and_formats(tmp_path) -> None:
    dpath_campaign = Path(tmp_path) / "cmp_manifest"

    def _make_trial(dataset, arm, coord, seed, complete=None, failure=None, runtime=None, epoch=0):
        d = dpath_campaign / "datasets" / dataset / "arms" / arm / "coords" / coord / str(seed)
        d.mkdir(parents=True, exist_ok=True)
        if complete is not None:
            with open(d / "trial_metadata.json", "w") as f:
                json.dump({
                    "dataset": dataset,
                    "complete": complete,
                    "runtime": {"trial": runtime},
                    "progress": {"epoch": epoch, "n_epochs": 35, "n_samps_seen": 0},
                }, f)
        if failure is not None:  # errored, with the failure= marker _log_trial_error writes
            (d / "error.log").write_text(f"TRIAL FAILED\n  seed={seed}, dataset={dataset}, arm={arm}, coord={coord}, failure={failure}\nboom")

    _make_trial("cub", "hp", "c0", 42, complete=True, runtime="113723.9", epoch=35)  # completed
    _make_trial("lepid", "hp", "c0", 42, complete=False, failure="VRAM", runtime="3723.4", epoch=19)  # failed
    # moss/hp/c0/42 -> failed before ever writing metadata (e.g. crashed at startup): no runtime to show
    _make_trial("moss", "hp", "c0", 42, failure="Mixed")
    # nymph/hp/c0/42 -> in progress: it's a resume-after-failure, so it carries a stale error.log; the
    # running trial (passed explicitly) must outrank that error.log and bucket as In Progress, not Failed
    _make_trial("nymph", "hp", "c0", 42, failure="Other")
    # cub/hp/c1/42  -> queued (no dir at all)

    trials = [
        ("cub", "hp", "c0", 42),
        ("lepid", "hp", "c0", 42),
        ("moss", "hp", "c0", 42),
        ("nymph", "hp", "c0", 42),
        ("cub", "hp", "c1", 42),
    ]
    PrintLog.manifest(dpath_campaign, trials, in_progress=("nymph", "hp", "c0", 42))

    # Completed/Failed entries carry the trial wall-clock, dash-padded per section (min 3 dashes at the
    # longest trial id) so the times line up
    text = (dpath_campaign / "manifest.log").read_text(encoding="utf-8")
    assert text == (
        "❌ Failed:\n"
        "lepid/hp/c0/42 --- 0-01:02:03 --- 19/35 --- VRAM\n"
        "moss/hp/c0/42 ---- n/a --- Mixed\n"
        "\n"
        "✅ Completed:\n"
        "cub/hp/c0/42 --- 1-07:35:23\n"
        "\n"
        "🏃 In Progress:\n"
        "nymph/hp/c0/42\n"
        "\n"
        "⏳ Queued:\n"
        "cub/hp/c1/42\n"
    )


def test_manifest_completed_beats_stale_error_log(tmp_path) -> None:
    # a trial that failed once then succeeded on resume keeps its old error.log; complete=True wins
    dpath_campaign = Path(tmp_path) / "cmp_manifest_resume"
    d = dpath_campaign / "datasets" / "cub" / "arms" / "hp" / "coords" / "c0" / "42"
    d.mkdir(parents=True)
    with open(d / "trial_metadata.json", "w") as f:
        json.dump({
            "dataset": "cub",
            "complete": True,
            "runtime": {"trial": "45296.0"},
            "progress": {"epoch": 35, "n_epochs": 35, "n_samps_seen": 4_000_000},
        }, f)
    (d / "error.log").write_text("old failure")

    PrintLog.manifest(dpath_campaign, [("cub", "hp", "c0", 42)], in_progress=None)

    text = (dpath_campaign / "manifest.log").read_text(encoding="utf-8")
    assert text == (
        "❌ Failed:\n"
        "\n"
        "✅ Completed:\n"
        "cub/hp/c0/42 --- 0-12:34:56\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
    )


def test_manifest_shows_all_headers_at_kickoff(tmp_path) -> None:
    dpath_campaign = Path(tmp_path) / "cmp_manifest_kickoff"
    dpath_campaign.mkdir(parents=True)

    PrintLog.manifest(dpath_campaign, [("cub", "hp", "c0", 42), ("lepid", "hp", "c0", 42)], in_progress=None)

    text = (dpath_campaign / "manifest.log").read_text(encoding="utf-8")
    assert text == (
        "❌ Failed:\n"
        "\n"
        "✅ Completed:\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
        "cub/hp/c0/42\n"
        "lepid/hp/c0/42\n"
    )


def test_run_campaign_writes_manifest_tracking_outcomes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch)

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

    # cub completes cleanly; lepid crashes without progress on every attempt (retried up to the cap)
    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        cur = f"{cfg_dict['dataset']}/{cfg_dict['arm']}/{cfg_dict['coord']}/{cfg_dict['seed']}"
        in_progress_snapshots.append((cur, (dpath_campaign / "manifest.log").read_text(encoding="utf-8")))
        d = _dpath_trial(tmp_path, cfg_dict)
        (d / "chkpts" / "in_progress").mkdir(parents=True, exist_ok=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"epoch": 1, "n_epochs": 35, "n_samps_seen": 200_000}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)
        if cfg_dict["dataset"] == "lepid":
            raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_manifest_run",
        n_trials=1,
        datasets=("cub", "lepid"),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    # each trial showed under In Progress while it was running (lepid appears once per retry; collapse them)
    order = [cur for cur, _ in in_progress_snapshots]
    distinct_order = [cur for i, cur in enumerate(order) if i == 0 or cur != order[i - 1]]
    assert distinct_order == ["cub/sp/base/42", "lepid/sp/base/42"]
    for cur, snapshot in in_progress_snapshots:
        assert f"🏃 In Progress:\n{cur}\n" in snapshot

    text = (dpath_campaign / "manifest.log").read_text(encoding="utf-8")
    assert text == (
        "❌ Failed:\n"
        "lepid/sp/base/42 --- 0-01:01:01 --- 1/35 --- Other\n"
        "\n"
        "✅ Completed:\n"
        "cub/sp/base/42 --- 0-01:01:01\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
    )


def test_run_campaign_clears_in_progress_on_interrupt(tmp_path, monkeypatch) -> None:
    # a campaign abort (Ctrl-C / SIGTERM / scancel) must not leave the killed trial frozen as In Progress
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})
    _stub_campaign_config(monkeypatch)

    dpath_campaign = Path(tmp_path) / "cmp_manifest_interrupt"

    # the trial gets killed mid-run: leaves chkpts/in_progress + incomplete metadata, no error.log
    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        d = _dpath_trial(tmp_path, cfg_dict)
        (d / "chkpts" / "in_progress").mkdir(parents=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"epoch": 1, "n_epochs": 35, "n_samps_seen": 200_000}, "n_crashes": {"ram": 0, "vram": 0, "other": 0}}, f)
        raise KeyboardInterrupt

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_manifest_interrupt",
        n_trials=1,
        datasets=("cub", "lepid"),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    text = (dpath_campaign / "manifest.log").read_text(encoding="utf-8")
    assert text == (
        "❌ Failed:\n"
        "\n"
        "✅ Completed:\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
        "cub/sp/base/42\n"
        "lepid/sp/base/42\n"
    )


def test_run_campaign_persists_and_grows_matrix(tmp_path, monkeypatch) -> None:
    scheduled = _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_grow",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    with open(tmp_path / "cmp_grow" / "campaign_metadata.json") as f:
        meta = json.load(f)
    assert meta["arms"] == ["sp"]
    assert meta["coords"] == ["base"]
    assert meta["datasets"] == ["cub"]
    assert meta["seeds"] == [42]
    assert scheduled == [("sp", "base", "cub", 42)]

    # relaunch with added arms, coords, datasets, and seeds -> matrix grows, no error
    n_before = len(scheduled)
    cr.run_campaign(
        campaign="cmp_grow",
        n_trials=2,
        datasets=("cub", "lepid"),
        ablation_arms=[[
            {"loss.targ": "sp", "name": "sp"},
            {"loss.targ": "phylo", "name": "hp"},
        ]],
        hpo_coords=[[{"name": "base"}, {"loss.sim": "geo1"}]],
    )

    with open(tmp_path / "cmp_grow" / "campaign_metadata.json") as f:
        meta = json.load(f)
    assert meta["arms"] == ["sp", "hp"]
    assert meta["coords"] == ["base", "loss.sim-geo1"]
    assert meta["datasets"] == ["cub", "lepid"]
    assert meta["seeds"] == [42, 43]

    # the already-completed cub/sp/base/42 trial is skipped; only the 15 newly-added trials run
    relaunch_calls = scheduled[n_before:]
    assert ("sp", "base", "cub", 42) not in relaunch_calls
    assert len(relaunch_calls) == 2 * 2 * 2 * 2 - 1


def test_run_campaign_records_commit_hash_on_first_launch(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_commit",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(cr.__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    meta = json.loads((tmp_path / "cmp_commit" / "campaign_metadata.json").read_text())
    assert meta["commit"] == head


def test_run_campaign_raises_on_duplicate_name_before_side_effects(tmp_path, monkeypatch) -> None:
    # the dup-name check is hoisted to the top of run_campaign, so it must fire before any filesystem
    # side effect -- no campaign dir / time.pkl / campaign_metadata.json is created
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {}, "img_cache": tmp_path / "img_cache"})

    with pytest.raises(ValueError, match="Duplicate ablation_arms name"):
        cr.run_campaign(
            campaign="cmp_dup",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[
                {"loss.targ": "sp", "name": "dup"},
                {"loss.targ": "phylo", "name": "dup"},
            ]],
            hpo_coords=_BASE_COORD,
        )

    assert not (tmp_path / "cmp_dup").exists()


def test_run_campaign_relaunch_survives_duration_only_metadata_rewrite(tmp_path, monkeypatch) -> None:
    # mirrors production: between launches a trial (utils/train.py update_campaign_time) rewrites
    # campaign_metadata.json with only 'duration' changed; the matrix keys must survive for the
    # relaunch's removal check to read them, and the trial-written duration must survive the relaunch
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_roundtrip",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    fpath_meta = tmp_path / "cmp_roundtrip" / "campaign_metadata.json"
    meta = json.loads(fpath_meta.read_text())
    meta["duration"] = "0-01:23:45"  # whole-dict rewrite, duration only (what update_campaign_time does)
    fpath_meta.write_text(json.dumps(meta))

    # additive relaunch must not error and must preserve the trial-written duration
    cr.run_campaign(
        campaign="cmp_roundtrip",
        n_trials=1,
        datasets=("cub", "lepid"),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    meta = json.loads(fpath_meta.read_text())
    assert meta["duration"] == "0-01:23:45"
    assert meta["datasets"] == ["cub", "lepid"]


def test_run_campaign_raises_on_removed_arm(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_rm_arm",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[
            {"loss.targ": "sp", "name": "sp"},
            {"loss.targ": "phylo", "name": "hp"},
        ]],
        hpo_coords=_BASE_COORD,
    )

    with pytest.raises(RuntimeError, match="arms removed.*hp"):
        cr.run_campaign(
            campaign="cmp_rm_arm",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
            hpo_coords=_BASE_COORD,
        )


def test_run_campaign_raises_on_removed_coord(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_rm_coord",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=[[{"loss.sim": ["cos", "geo1"]}]],
    )

    with pytest.raises(RuntimeError, match="coords removed.*loss.sim-geo1"):
        cr.run_campaign(
            campaign="cmp_rm_coord",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
            hpo_coords=[[{"loss.sim": ["cos"]}]],
        )


def test_run_campaign_raises_on_removed_dataset(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_rm_dataset",
        n_trials=1,
        datasets=("cub", "lepid"),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    with pytest.raises(RuntimeError, match="datasets removed.*lepid"):
        cr.run_campaign(
            campaign="cmp_rm_dataset",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
            hpo_coords=_BASE_COORD,
        )


def test_run_campaign_raises_on_removed_seed(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_rm_seed",
        n_trials=2,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    with pytest.raises(RuntimeError, match="seeds removed.*43"):
        cr.run_campaign(
            campaign="cmp_rm_seed",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
            hpo_coords=_BASE_COORD,
        )


def _stub_img_cache_campaign(tmp_path, monkeypatch, use_img_cache: bool) -> None:
    monkeypatch.setattr(cr, "SEED0", 42)
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path, "imgs": {"bryo": None, "cub": None}, "img_cache": tmp_path / "img_cache"})
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": {"campaign": "c", "arm": "a", "coord": "c", "seed": 0, "dataset": "cub", "split": "D10", "loss": {"targ": "sp", "crit": "bce", "sim": "cos"}, "dev": {"del_base_eval_cache": {"campaign": False, "trial": False}}},
        "hardware": {"max_retries": 2, "use_img_cache": use_img_cache},
        "manif_viz": {},
        "model_specific": {},
        "dataset_specific": {},
    })
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)


def test_run_campaign_use_img_cache_missing_pack_errors_before_trials(tmp_path, monkeypatch) -> None:
    _stub_img_cache_campaign(tmp_path, monkeypatch, use_img_cache=True)
    launched = []
    monkeypatch.setattr(cr, "_run_trial_subprocess", lambda *a, **k: launched.append(1))

    with pytest.raises(FileNotFoundError, match="cub"):
        cr.run_campaign(
            campaign="cmp_ic_missing",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
            hpo_coords=_BASE_COORD,
        )
    assert launched == []


def test_run_campaign_use_img_cache_records_staging_runtime(tmp_path, monkeypatch) -> None:
    _stub_img_cache_campaign(tmp_path, monkeypatch, use_img_cache=True)
    (tmp_path / "img_cache" / "cub").mkdir(parents=True)
    (tmp_path / "img_cache" / "cub" / "meta.json").write_text("{}")
    monkeypatch.setattr(cr, "stage_img_cache", lambda ds: 1.2345)
    monkeypatch.setattr(cr, "_run_trial_subprocess", lambda cfg_dict, spare_render_pid=None: _leave_completed_trial(tmp_path, cfg_dict))

    cr.run_campaign(
        campaign="cmp_ic_rt",
        n_trials=1,
        datasets=("cub",),
        ablation_arms=[[{"loss.targ": "sp", "name": "sp"}]],
        hpo_coords=_BASE_COORD,
    )

    meta = json.loads((tmp_path / "cmp_ic_rt" / "campaign_metadata.json").read_text())
    # staged dataset gets round(seconds, 2); datasets not in this campaign stay null
    assert meta["runtime_img_cache"] == {"bryo": None, "cub": 1.23}


@pytest.mark.parametrize("side", ["arm", "coord"])
def test_run_campaign_use_img_cache_override_checked_at_startup(tmp_path, monkeypatch, side: str) -> None:
    # baseline use_img_cache=False, but one arm (or one coord) enables it via hw.* override: the
    # missing-pack error must still fire at startup, before any trial launches
    _stub_img_cache_campaign(tmp_path, monkeypatch, use_img_cache=False)
    launched = []
    monkeypatch.setattr(cr, "_run_trial_subprocess", lambda *a, **k: launched.append(1))

    ic = [[{"hw.use_img_cache": True, "name": "ic"}]]
    with pytest.raises(FileNotFoundError, match="cub"):
        cr.run_campaign(
            campaign="cmp_ic_override",
            n_trials=1,
            datasets=("cub",),
            ablation_arms=ic if side == "arm" else [[{"loss.targ": "sp", "name": "sp"}]],
            hpo_coords=ic if side == "coord" else _BASE_COORD,
        )
    assert launched == []


def _run_launch_camp(tmp_path, monkeypatch, continue_campaign, suffix=None) -> str:
    """Run cr._launch_camp() with stubbed config loading and a no-op run_campaign; returns the
    campaign name run_campaign was launched with."""
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})
    monkeypatch.setattr(cr, "_load_campaign_config", lambda name: {
        "suffix": suffix, "n_trials": 1, "datasets": ["cub"], "ablation_arms": [[{"name": "s"}]], "hpo_coords": _BASE_COORD,
    })
    monkeypatch.setattr(cr, "load_train_config_dict", lambda: {"dev": {"continue_campaign": continue_campaign}})
    launched = []
    monkeypatch.setattr(cr, "run_campaign", lambda campaign, **kwargs: launched.append((campaign, kwargs)))
    cr._launch_camp("dev")
    assert launched[0][1]["ablation_arms"] == [[{"name": "s"}]] and launched[0][1]["hpo_coords"] == _BASE_COORD
    return launched[0][0]


def test_launch_camp_continue_campaign_true_keeps_existing_name(tmp_path, monkeypatch) -> None:
    (tmp_path / "dev").mkdir()
    assert _run_launch_camp(tmp_path, monkeypatch, continue_campaign=True) == "dev"


def test_launch_camp_continue_campaign_false_keeps_name_without_collision(tmp_path, monkeypatch) -> None:
    assert _run_launch_camp(tmp_path, monkeypatch, continue_campaign=False) == "dev"


def test_launch_camp_continue_campaign_false_dedupes_to_first_free_name(tmp_path, monkeypatch) -> None:
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev2").mkdir()
    assert _run_launch_camp(tmp_path, monkeypatch, continue_campaign=False) == "dev3"


def test_launch_camp_continue_campaign_false_dedupes_suffixed_name(tmp_path, monkeypatch) -> None:
    (tmp_path / "dev_foobar").mkdir()
    assert _run_launch_camp(tmp_path, monkeypatch, continue_campaign=False, suffix="foobar") == "dev_foobar2"


def _wire_queue(tmp_path, monkeypatch) -> Path:
    """Point the runner at a real config dir (camp_queue.yaml + camps/ live under it) and
    neutralize pytest's own argv; returns the config dir."""
    dpath_config = tmp_path / "config"
    (dpath_config / "camps").mkdir(parents=True)
    monkeypatch.setattr(cr, "paths", {"config": dpath_config})
    monkeypatch.setattr(cr.sys, "argv", ["campaign_runner"])
    return dpath_config


def _write_queue(dpath_config: Path, entries: list[str]) -> None:
    (dpath_config / "camp_queue.yaml").write_text(yaml.safe_dump({"campaigns": entries}))


def _add_camp_config(dpath_config: Path, *names: str) -> None:
    for name in names:
        (dpath_config / "camps" / f"{name}.yaml").write_text("{}")


def _stub_run_queue_entry(dpath_config, monkeypatch, on_run=None) -> list[str]:
    """Stub cr._run_queue_entry to just record dispatched specs (returning True); `on_run(spec)`
    can rewrite camp_queue.yaml mid-run to simulate live edits. Returns the record list."""
    ran = []

    def _fake_run_queue_entry(spec: str) -> bool:
        ran.append(spec)
        if on_run is not None:
            on_run(spec)
        return True

    monkeypatch.setattr(cr, "_run_queue_entry", _fake_run_queue_entry)
    return ran


def test_main_runs_queue_in_order_and_picks_up_added_entries(tmp_path, monkeypatch) -> None:
    dpath_config = _wire_queue(tmp_path, monkeypatch)
    _add_camp_config(dpath_config, "a", "b", "c")
    _write_queue(dpath_config, ["camp.a", "camp.b"])

    # camp.c is appended while camp.a runs -- the re-read after each campaign picks it up
    def _append_during_a(spec: str) -> None:
        if spec == "camp.a":
            _write_queue(dpath_config, ["camp.a", "camp.b", "camp.c"])

    ran = _stub_run_queue_entry(dpath_config, monkeypatch, on_run=_append_during_a)
    cr.main()
    assert ran == ["camp.a", "camp.b", "camp.c"]


def test_main_picks_up_entries_inserted_at_any_position(tmp_path, monkeypatch) -> None:
    dpath_config = _wire_queue(tmp_path, monkeypatch)
    _add_camp_config(dpath_config, "a", "b", "c", "d")
    _write_queue(dpath_config, ["camp.a", "camp.b"])

    # while camp.a runs, camp.c is inserted at the front and camp.d in the middle: already-run
    # entries are consumed by occurrence, so the insertions run next, in file order
    def _insert_during_a(spec: str) -> None:
        if spec == "camp.a":
            _write_queue(dpath_config, ["camp.c", "camp.a", "camp.d", "camp.b"])

    ran = _stub_run_queue_entry(dpath_config, monkeypatch, on_run=_insert_during_a)
    cr.main()
    assert ran == ["camp.a", "camp.c", "camp.d", "camp.b"]


def test_main_duplicate_entry_queues_a_second_run(tmp_path, monkeypatch) -> None:
    dpath_config = _wire_queue(tmp_path, monkeypatch)
    _add_camp_config(dpath_config, "a")
    _write_queue(dpath_config, ["camp.a", "camp.a"])

    ran = _stub_run_queue_entry(dpath_config, monkeypatch)
    cr.main()
    assert ran == ["camp.a", "camp.a"]


@pytest.mark.parametrize("interrupt", ["returns_false", "raises"])
def test_main_interrupted_campaign_stops_the_queue(tmp_path, monkeypatch, interrupt: str) -> None:
    # both interrupt shapes stop the queue: run_campaign's caught-interrupt False, and a raw
    # KeyboardInterrupt from an unguarded window (e.g. between trials)
    dpath_config = _wire_queue(tmp_path, monkeypatch)
    _add_camp_config(dpath_config, "a", "b")
    _write_queue(dpath_config, ["camp.a", "camp.b"])

    ran = []

    def _fake_run_queue_entry(spec: str) -> bool:
        ran.append(spec)
        if interrupt == "raises":
            raise KeyboardInterrupt
        return False

    monkeypatch.setattr(cr, "_run_queue_entry", _fake_run_queue_entry)
    cr.main()
    assert ran == ["camp.a"]


def test_main_empty_queue_exits_immediately(tmp_path, monkeypatch) -> None:
    dpath_config = _wire_queue(tmp_path, monkeypatch)
    (dpath_config / "camp_queue.yaml").write_text("campaigns:\n")  # blank list parses to None

    ran = _stub_run_queue_entry(dpath_config, monkeypatch)
    cr.main()
    assert ran == []


def test_main_rejects_arguments(tmp_path, monkeypatch) -> None:
    _wire_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(cr.sys, "argv", ["campaign_runner", "--dev"])
    with pytest.raises(SystemExit, match="camp_queue"):
        cr.main()


def test_main_invalid_next_entry_fails_before_running_anything(tmp_path, monkeypatch) -> None:
    dpath_config = _wire_queue(tmp_path, monkeypatch)
    _write_queue(dpath_config, ["bogus.a"])

    ran = _stub_run_queue_entry(dpath_config, monkeypatch)
    with pytest.raises(SystemExit, match="camp.<name>"):
        cr.main()
    assert ran == []


def test_main_invalid_pending_entry_warns_then_fails_when_reached(tmp_path, monkeypatch, capsys) -> None:
    # a bad entry behind valid ones only warns at first sight (fixable in place before its turn);
    # left unfixed, it hard-fails when it becomes the entry about to run
    dpath_config = _wire_queue(tmp_path, monkeypatch)
    _add_camp_config(dpath_config, "a")
    _write_queue(dpath_config, ["camp.a", "camp.missing"])

    ran = _stub_run_queue_entry(dpath_config, monkeypatch)
    with pytest.raises(SystemExit, match="missing"):
        cr.main()
    assert ran == ["camp.a"]
    assert "pending entry 'camp.missing' is invalid" in capsys.readouterr().out


def test_stash_nccl_dumps_moves_dumps_and_strips_prefix(tmp_path) -> None:
    (tmp_path / "nccl_trace_sp_base_cub_42_rank0").write_text("dump0")
    (tmp_path / "nccl_trace_sp_base_cub_42_rank1").write_text("dump1")
    (tmp_path / "cfg_baseline.json").write_text("{}")

    cr._stash_nccl_dumps(tmp_path)

    assert (tmp_path / "nccl_traces" / "sp_base_cub_42_rank0").read_text() == "dump0"
    assert (tmp_path / "nccl_traces" / "sp_base_cub_42_rank1").read_text() == "dump1"
    assert not list(tmp_path.glob("nccl_trace_*"))
    assert (tmp_path / "cfg_baseline.json").exists()


def test_stash_nccl_dumps_without_dumps_creates_nothing(tmp_path) -> None:
    (tmp_path / "cfg_baseline.json").write_text("{}")

    cr._stash_nccl_dumps(tmp_path)

    assert not (tmp_path / "nccl_traces").exists()


def test_stash_nccl_dumps_accumulates_into_existing_dir(tmp_path) -> None:
    (tmp_path / "nccl_traces").mkdir()
    (tmp_path / "nccl_traces" / "earlier_cub_7_rank0").write_text("old")
    (tmp_path / "nccl_trace_sp_base_cub_42_rank0").write_text("new")

    cr._stash_nccl_dumps(tmp_path)

    assert (tmp_path / "nccl_traces" / "earlier_cub_7_rank0").read_text() == "old"
    assert (tmp_path / "nccl_traces" / "sp_base_cub_42_rank0").read_text() == "new"
