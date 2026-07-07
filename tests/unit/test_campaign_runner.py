from pathlib import Path
import json
import os
import pytest
import subprocess

import campaign_runner as cr
from utils.utils import PrintLog


def _leave_completed_trial(tmp_path, cfg_dict) -> None:
    """Mimic a real successful trial subprocess: leave chkpts/in_progress + incomplete metadata behind so
    run_campaign's success path (rmtree in_progress + flip complete=True) has something to act on."""
    d = tmp_path / cfg_dict["campaign"] / "settings" / cfg_dict["setting"] / cfg_dict["dataset"] / str(cfg_dict["seed"])
    (d / "chkpts" / "in_progress").mkdir(parents=True, exist_ok=True)
    with open(d / "trial_metadata.json", "w") as f:
        json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"n_samps_seen": 200_000, "sample_volume": 4_000_000}}, f)


def _setup_completing_campaign(tmp_path, monkeypatch) -> list:
    """Wire run_campaign so trials complete cleanly without real subprocesses/renders: each fake trial
    leaves the chkpts/in_progress dir + incomplete metadata behind (the runner flips complete=True).
    Returns the list of (setting, dataset, seed) tuples each launched trial was invoked with."""
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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1},
        "model_specific": {},
    })
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    scheduled: list = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append((cfg_dict["setting"], cfg_dict["dataset"], cfg_dict["seed"]))
        d = tmp_path / cfg_dict["campaign"] / "settings" / cfg_dict["setting"] / cfg_dict["dataset"] / str(cfg_dict["seed"])
        (d / "chkpts" / "in_progress").mkdir(parents=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"n_samps_seen": 200_000, "sample_volume": 4_000_000}}, f)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)
    return scheduled


def test_load_or_create_campaign_config_reuses_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    train_a = {"campaign": "dev", "split": "D10"}
    hw_a = {"mixed_prec": True, "prefetch_factor": 4}
    mviz_a = {"tsne": {"perplexity": 30, "n_iter": 1000}}
    ms_a = {"siglip": {"l2reg": 0.0, "beta2": 0.95}, "clip": {"l2reg": 0.2, "beta2": 0.98}}

    train_b = {"campaign": "changed", "split": "dev"}
    hw_b = {"mixed_prec": False, "prefetch_factor": 2}
    mviz_b = {"tsne": {"perplexity": 5, "n_iter": 250}}
    ms_b = {"siglip": {"l2reg": 0.1, "beta2": 0.5}, "clip": {"l2reg": 0.3, "beta2": 0.7}}

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: train_a)
    monkeypatch.setattr(cr, "load_hardware_config_dict", lambda: hw_a)
    monkeypatch.setattr(cr, "load_manifold_viz_config_dict", lambda: mviz_a)
    monkeypatch.setattr(cr, "load_model_specific_config_dict", lambda: ms_a)
    out_first = cr._load_or_create_campaign_config("cmp_a")

    monkeypatch.setattr(cr, "load_train_config_dict", lambda: train_b)
    monkeypatch.setattr(cr, "load_hardware_config_dict", lambda: hw_b)
    monkeypatch.setattr(cr, "load_manifold_viz_config_dict", lambda: mviz_b)
    monkeypatch.setattr(cr, "load_model_specific_config_dict", lambda: ms_b)
    out_second = cr._load_or_create_campaign_config("cmp_a")

    # the four sources are bundled into one snapshot and frozen on first launch
    expected = {"train": train_a, "hardware": hw_a, "manifold_viz": mviz_a, "model_specific": ms_a}
    assert out_first == expected
    assert out_second == expected


def test_load_or_create_campaign_config_keeps_model_specific_nulls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    train_cfg = {
        "campaign": "dev",
        "arch": {"model_type": "siglip_vitb16"},
        "opt": {"l2reg": None, "beta2": None},
    }
    monkeypatch.setattr(cr, "load_train_config_dict", lambda: train_cfg)
    monkeypatch.setattr(cr, "load_hardware_config_dict", lambda: {"max_retries": 2})
    monkeypatch.setattr(cr, "load_manifold_viz_config_dict", lambda: {})
    monkeypatch.setattr(cr, "load_model_specific_config_dict", lambda: {"siglip": {"l2reg": 0.0, "beta2": 0.95}})

    snapshot = cr._load_or_create_campaign_config("cmp_ms")

    # model-family defaults are NOT resolved into the train snapshot -- they stay null so a per-setting
    # arch.model_type override can pick up the matching family per trial (resolution happens in the
    # trial, from the model_specific snapshot).
    assert snapshot["train"]["opt"]["l2reg"] is None
    assert snapshot["train"]["opt"]["beta2"] is None


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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}},
        "model_specific": {},
    })

    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append((cfg_dict["seed"], cfg_dict["dataset"], cfg_dict["setting"], cfg_dict["loss"]["targ"]))
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_b",
        n_trials=2,
        datasets=("cub", "lepid"),
        baseline_overrides=[[
            {"loss.targ": "aligned", "name": "iw"},
            {"loss.targ": "phylo", "name": "hp"},
        ]],
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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}},
        "model_specific": {},
    })
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)
    monkeypatch.setattr(
        cr,
        "_run_trial_subprocess",
        lambda cfg_dict, spare_render_pid=None: _leave_completed_trial(tmp_path, cfg_dict),
    )

    cr.run_campaign(
        campaign="cmp_c",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[[
            {"loss.targ": "aligned", "name": "iw"},
        ]],
    )

    fpath = Path(tmp_path) / "cmp_c" / "settings" / "iw" / "overrides.json"
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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1},
        "model_specific": {},
    })
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    dpath_trial = Path(tmp_path) / "cmp_complete" / "settings" / "iw" / "cub" / "42"

    # a real trial writes its metadata (complete still False) + leaves a chkpts/in_progress dir behind;
    # the campaign runner is what cleans up and flips complete=True on a clean exit
    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        (dpath_trial / "chkpts" / "in_progress").mkdir(parents=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"n_samps_seen": 200_000, "sample_volume": 4_000_000}}, f)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_complete",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is True
    assert not (dpath_trial / "chkpts" / "in_progress").exists()


def test_run_campaign_retries_then_fails_trial_without_progress(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1},
        "model_specific": {},
    })

    dpath_trial = Path(tmp_path) / "cmp_fail" / "settings" / "iw" / "cub" / "42"

    # every attempt crashes without ever writing a checkpoint (no forward progress), so the runner retries
    # up to the no-progress cap and then gives up, leaving the trial incomplete with an error.log.
    calls = []
    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        calls.append((cfg_dict["setting"], cfg_dict["dataset"], cfg_dict["seed"]))
        dpath_trial.mkdir(parents=True, exist_ok=True)
        with open(dpath_trial / "trial_metadata.json", "w") as f:
            json.dump({"dataset": "cub", "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"n_samps_seen": 200_000, "sample_volume": 4_000_000}}, f)
        raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_fail",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    # one initial attempt + max_retries (2, from the injected hardware config) no-progress resume attempts
    assert len(calls) == 2 + 1
    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is False
    assert (dpath_trial / "error.log").exists()


def test_run_campaign_retries_recover_across_flakes_that_make_progress(tmp_path, monkeypatch) -> None:
    # a trial that flakes repeatedly but advances its checkpoint each time is resumed indefinitely: the
    # no-progress counter resets on every forward step, so more flakes than the cap still recover.
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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1},
        "model_specific": {},
    })
    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    dpath_trial = Path(tmp_path) / "cmp_flaky" / "settings" / "iw" / "cub" / "42"
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
            json.dump({"dataset": "cub", "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"n_samps_seen": 200_000, "sample_volume": 4_000_000}}, f)
        if calls["n"] <= n_flakes:
            raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_flaky",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    assert calls["n"] == n_flakes + 1  # every progressing flake was retried; final attempt completed
    with open(dpath_trial / "trial_metadata.json") as f:
        assert json.load(f)["complete"] is True
    assert not fpath_ckpt.parent.exists()  # chkpts/in_progress removed on success
    assert not (dpath_trial / "error.log").exists()


def test_expand_settings_raises_on_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate baseline_overrides name"):
        cr._expand_settings(
            [
                [
                    {"loss.targ": "aligned", "name": "dup"},
                    {"loss.targ": "phylo", "name": "dup"},
                ]
            ]
        )


def test_expand_settings_single_combo_group_unchanged() -> None:
    # one combo group -> members expand unchanged: names are not joined and payloads carry through as-is
    settings = cr._expand_settings(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                {"loss.targ": "aligned", "name": "iw"},
            ]
        ]
    )
    assert settings == [
        ("hp", {"loss2.mix": 0.3, "loss2.targ": "phylo"}),
        ("iw", {"loss.targ": "aligned"}),
    ]


def test_expand_settings_derives_name_from_overrides_when_omitted() -> None:
    # an item without a 'name' is named by its overrides: 'key-value' pairs joined by '_', with
    # keys/values mapped through CFG_PARAM_ALIASES / CFG_PARAM_VALUE_ALIASES when an alias exists
    # and anything unaliased (e.g. loss2.mix) passing through verbatim
    settings = cr._expand_settings(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo"},
                {"loss.targ": "multipos"},
                {"loss.targ": "aligned", "name": "iw"},
            ]
        ]
    )
    assert settings == [
        ("loss2.mix-0.3_L2T-hp", {"loss2.mix": 0.3, "loss2.targ": "phylo"}),
        ("L1T-multipos", {"loss.targ": "multipos"}),
        ("iw", {"loss.targ": "aligned"}),
    ]


def test_expand_settings_combo_list_expands_item_and_appends_to_name() -> None:
    # a list-valued override is a combo list: the item expands to one setting per list value, the
    # (aliased) 'key-value' pair appended to the explicit 'name'
    settings = cr._expand_settings(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": [1024, 2048], "name": "hp"},
                {"loss.targ": "multipos", "batch_size": [1024, 2048], "name": "sw"},
            ]
        ]
    )
    assert settings == [
        ("hp_bs-1k", {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 1024}),
        ("hp_bs-2k", {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 2048}),
        ("sw_bs-1k", {"loss.targ": "multipos", "batch_size": 1024}),
        ("sw_bs-2k", {"loss.targ": "multipos", "batch_size": 2048}),
    ]


def test_expand_settings_multiple_combo_lists_cross_within_item() -> None:
    # several combo lists in one item cross with each other, the last-listed key varying fastest;
    # scientific-notation floats read like the YAML that declared them (7e-06 -> '7.0e-6')
    settings = cr._expand_settings(
        [
            [
                {"loss2.mix": 0.3, "batch_size": [1024, 2048], "opt.lr.init": [7.0e-6, 1.2e-5], "name": "hp"},
            ]
        ]
    )
    assert [name for name, _ in settings] == [
        "hp_bs-1k_opt.lr.init-7.0e-6",
        "hp_bs-1k_opt.lr.init-1.2e-5",
        "hp_bs-2k_opt.lr.init-7.0e-6",
        "hp_bs-2k_opt.lr.init-1.2e-5",
    ]
    assert dict(settings)["hp_bs-2k_opt.lr.init-1.2e-5"] == {"loss2.mix": 0.3, "batch_size": 2048, "opt.lr.init": 1.2e-5}


def test_expand_settings_combo_list_in_unnamed_item_folds_into_derived_name() -> None:
    # in an unnamed item the expanded value is named like any other override, in declared position
    settings = cr._expand_settings([[{"loss.targ": "multipos", "batch_size": [1024, 2048]}]])
    assert settings == [
        ("L1T-multipos_bs-1k", {"loss.targ": "multipos", "batch_size": 1024}),
        ("L1T-multipos_bs-2k", {"loss.targ": "multipos", "batch_size": 2048}),
    ]


def test_expand_settings_derived_and_explicit_names_join_across_combo_groups() -> None:
    # derived names compose with explicit ones the same way in the cross-combo-group join
    settings = cr._expand_settings(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                {"loss.targ": "multipos"},
            ],
            [
                {"batch_size": 2048, "name": "2k"},
                {"batch_size": 1024},
            ],
        ]
    )

    assert len(settings) == 4
    assert dict(settings) == {
        "hp_2k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 2048},
        "hp_bs-1k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 1024},
        "L1T-multipos_2k": {"loss.targ": "multipos", "batch_size": 2048},
        "L1T-multipos_bs-1k": {"loss.targ": "multipos", "batch_size": 1024},
    }


def test_expand_settings_cartesian_product_merges_and_joins_names() -> None:
    # two combo groups -> every cross-combo-group combination; payloads merge, names join with '_' in combo-group order
    settings = cr._expand_settings(
        [
            [
                {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                {"loss.targ": "multipos", "name": "sw"},
            ],
            [
                {"batch_size": 2048, "name": "2k"},
                {"batch_size": 1024, "name": "1k"},
            ],
        ]
    )

    assert len(settings) == 4
    assert dict(settings) == {
        "hp_2k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 2048},
        "hp_1k": {"loss2.mix": 0.3, "loss2.targ": "phylo", "batch_size": 1024},
        "sw_2k": {"loss.targ": "multipos", "batch_size": 2048},
        "sw_1k": {"loss.targ": "multipos", "batch_size": 1024},
    }


def test_expand_settings_generalizes_to_three_combo_groups() -> None:
    settings = cr._expand_settings(
        [
            [{"a": 1, "name": "x"}, {"a": 2, "name": "y"}],
            [{"b": 1, "name": "p"}, {"b": 2, "name": "q"}],
            [{"c": 1, "name": "m"}, {"c": 2, "name": "n"}],
        ]
    )

    assert len(settings) == 8
    assert {name for name, _ in settings} == {
        "x_p_m", "x_p_n", "x_q_m", "x_q_n",
        "y_p_m", "y_p_n", "y_q_m", "y_q_n",
    }
    assert dict(settings)["y_q_n"] == {"a": 2, "b": 2, "c": 2}


def test_expand_settings_raises_on_cross_combo_group_key_collision() -> None:
    # the same override key appears in two combo groups -> two values would fight to define it when merged
    with pytest.raises(ValueError, match="collide between combo groups"):
        cr._expand_settings(
            [
                [
                    {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
                    {"loss.targ": "multipos", "name": "sw"},
                ],
                [
                    {"loss2.mix": 0.4, "loss2.targ": "phylo", "name": "hp4"},
                    {"loss.targ": "multipos", "name": "sw2"},
                ],
            ]
        )


def test_run_campaign_expands_combo_groups(tmp_path, monkeypatch) -> None:
    # the Cartesian product flows through run_campaign: each combined setting schedules and gets its
    # own merged overrides.json
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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}},
        "model_specific": {},
    })

    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append((cfg_dict["setting"], cfg_dict["loss"]["targ"], cfg_dict["loss"]["sim"]))
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_groups",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[
            [{"loss.targ": "aligned", "name": "iw"}, {"loss.targ": "phylo", "name": "hp"}],
            [{"loss.sim": "cos", "name": "cos"}, {"loss.sim": "l2", "name": "l2"}],
        ],
    )

    assert set(scheduled) == {
        ("iw_cos", "aligned", "cos"),
        ("iw_l2", "aligned", "l2"),
        ("hp_cos", "phylo", "cos"),
        ("hp_l2", "phylo", "l2"),
    }

    with open(Path(tmp_path) / "cmp_groups" / "settings" / "hp_l2" / "overrides.json") as f:
        data = json.load(f)
    assert data == {"loss.targ": "phylo", "loss.sim": "l2"}


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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1, "tsne": {"perplexity": 30, "n_iter": 1000}},
        "model_specific": {},
    })

    monkeypatch.setattr(cr, "_spawn_render", lambda *a, **k: None)

    scheduled = []

    def _fake_run_trial_subprocess(cfg_dict: dict, spare_render_pid=None):
        scheduled.append(cfg_dict)
        _leave_completed_trial(tmp_path, cfg_dict)

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_opt",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[[
            {"opt.l2reg": 0.33, "opt.beta2": 0.88, "name": "opt_tune"},
        ]],
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


def test_manifest_buckets_and_formats(tmp_path) -> None:
    dpath_campaign = Path(tmp_path) / "cmp_manifest"

    def _make_trial(setting, dataset, seed, complete=None, errored=False, runtime=None, n_samps_seen=0):
        d = dpath_campaign / "settings" / setting / dataset / str(seed)
        d.mkdir(parents=True, exist_ok=True)
        if complete is not None:
            with open(d / "trial_metadata.json", "w") as f:
                json.dump({
                    "dataset": dataset,
                    "complete": complete,
                    "runtime": {"trial": runtime},
                    "progress": {"n_samps_seen": n_samps_seen, "sample_volume": 4_000_000},
                }, f)
        if errored:
            (d / "error.log").write_text("boom")

    _make_trial("hp", "cub", 42, complete=True, runtime="113723.9", n_samps_seen=4_000_000)  # completed
    _make_trial("hp", "lepid", 42, complete=False, errored=True, runtime="3723.4", n_samps_seen=2_200_000)  # failed
    # hp/moss/42 -> failed before ever writing metadata (e.g. crashed at startup): no runtime to show
    _make_trial("hp", "moss", 42, errored=True)
    # hp/nymph/42 -> in progress: it's a resume-after-failure, so it carries a stale error.log; the
    # running trial (passed explicitly) must outrank that error.log and bucket as In Progress, not Failed
    _make_trial("hp", "nymph", 42, errored=True)
    # hp/bryo/42  -> queued (no dir at all)

    trials = [
        ("hp", "cub", 42),
        ("hp", "lepid", 42),
        ("hp", "moss", 42),
        ("hp", "nymph", 42),
        ("hp", "bryo", 42),
    ]
    PrintLog.manifest(dpath_campaign, trials, in_progress=("hp", "nymph", 42))

    # Completed/Failed entries carry the trial wall-clock, dash-padded per section (min 3 dashes at the
    # longest trial id) so the times line up
    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "hp/lepid/42 --- 0-01:02:03 --- 2.2M/4.0M\n"
        "hp/moss/42 ---- n/a\n"
        "\n"
        "✅ Completed:\n"
        "hp/cub/42 --- 1-07:35:23\n"
        "\n"
        "🏃 In Progress:\n"
        "hp/nymph/42\n"
        "\n"
        "⏳ Queued:\n"
        "hp/bryo/42\n"
    )


def test_manifest_completed_beats_stale_error_log(tmp_path) -> None:
    # a trial that failed once then succeeded on resume keeps its old error.log; complete=True wins
    dpath_campaign = Path(tmp_path) / "cmp_manifest_resume"
    d = dpath_campaign / "settings" / "hp" / "cub" / "42"
    d.mkdir(parents=True)
    with open(d / "trial_metadata.json", "w") as f:
        json.dump({
            "dataset": "cub",
            "complete": True,
            "runtime": {"trial": "45296.0"},
            "progress": {"n_samps_seen": 4_000_000, "sample_volume": 4_000_000},
        }, f)
    (d / "error.log").write_text("old failure")

    PrintLog.manifest(dpath_campaign, [("hp", "cub", 42)], in_progress=None)

    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "\n"
        "✅ Completed:\n"
        "hp/cub/42 --- 0-12:34:56\n"
        "\n"
        "🏃 In Progress:\n"
        "\n"
        "⏳ Queued:\n"
    )


def test_manifest_shows_all_headers_at_kickoff(tmp_path) -> None:
    dpath_campaign = Path(tmp_path) / "cmp_manifest_kickoff"
    dpath_campaign.mkdir(parents=True)

    PrintLog.manifest(dpath_campaign, [("hp", "cub", 42), ("hp", "lepid", 42)], in_progress=None)

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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1},
        "model_specific": {},
    })

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
        cur = f"{cfg_dict['setting']}/{cfg_dict['dataset']}/{cfg_dict['seed']}"
        in_progress_snapshots.append((cur, (dpath_campaign / "manifest.log").read_text()))
        d = dpath_campaign / "settings" / cfg_dict["setting"] / cfg_dict["dataset"] / str(cfg_dict["seed"])
        (d / "chkpts" / "in_progress").mkdir(parents=True, exist_ok=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"n_samps_seen": 200_000, "sample_volume": 4_000_000}}, f)
        if cfg_dict["dataset"] == "lepid":
            raise subprocess.CalledProcessError(1, ["torchrun"], stderr="boom")

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_manifest_run",
        n_trials=1,
        datasets=("cub", "lepid"),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    # each trial showed under In Progress while it was running (lepid appears once per retry; collapse them)
    order = [cur for cur, _ in in_progress_snapshots]
    distinct_order = [cur for i, cur in enumerate(order) if i == 0 or cur != order[i - 1]]
    assert distinct_order == ["iw/cub/42", "iw/lepid/42"]
    for cur, snapshot in in_progress_snapshots:
        assert f"🏃 In Progress:\n{cur}\n" in snapshot

    text = (dpath_campaign / "manifest.log").read_text()
    assert text == (
        "❌ Failed:\n"
        "iw/lepid/42 --- 0-01:01:01 --- 0.2M/4.0M\n"
        "\n"
        "✅ Completed:\n"
        "iw/cub/42 --- 0-01:01:01\n"
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
    monkeypatch.setattr(cr, "_load_or_create_campaign_config", lambda campaign: {
        "train": baseline,
        "hardware": {"max_retries": 2},
        "manifold_viz": {"n_stoch_layers": 1},
        "model_specific": {},
    })

    dpath_campaign = Path(tmp_path) / "cmp_manifest_interrupt"

    # the trial gets killed mid-run: leaves chkpts/in_progress + incomplete metadata, no error.log
    def _fake_run_trial_subprocess(cfg_dict, spare_render_pid=None):
        d = dpath_campaign / "settings" / cfg_dict["setting"] / cfg_dict["dataset"] / str(cfg_dict["seed"])
        (d / "chkpts" / "in_progress").mkdir(parents=True)
        with open(d / "trial_metadata.json", "w") as f:
            json.dump({"dataset": cfg_dict["dataset"], "complete": False, "runtime": {"trial": "3661.0"}, "progress": {"n_samps_seen": 200_000, "sample_volume": 4_000_000}}, f)
        raise KeyboardInterrupt

    monkeypatch.setattr(cr, "_run_trial_subprocess", _fake_run_trial_subprocess)

    cr.run_campaign(
        campaign="cmp_manifest_interrupt",
        n_trials=1,
        datasets=("cub", "lepid"),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
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


def test_run_campaign_persists_and_grows_matrix(tmp_path, monkeypatch) -> None:
    scheduled = _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_grow",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    with open(tmp_path / "cmp_grow" / "campaign_metadata.json") as f:
        meta = json.load(f)
    assert meta["settings"] == ["iw"]
    assert meta["datasets"] == ["cub"]
    assert meta["seeds"] == [42]
    assert scheduled == [("iw", "cub", 42)]

    # relaunch with added settings, datasets, and seeds -> matrix grows, no error
    n_before = len(scheduled)
    cr.run_campaign(
        campaign="cmp_grow",
        n_trials=2,
        datasets=("cub", "lepid"),
        baseline_overrides=[[
            {"loss.targ": "aligned", "name": "iw"},
            {"loss.targ": "phylo", "name": "hp"},
        ]],
    )

    with open(tmp_path / "cmp_grow" / "campaign_metadata.json") as f:
        meta = json.load(f)
    assert meta["settings"] == ["iw", "hp"]
    assert meta["datasets"] == ["cub", "lepid"]
    assert meta["seeds"] == [42, 43]

    # the already-completed iw/cub/42 trial is skipped; only the 7 newly-added trials run
    relaunch_calls = scheduled[n_before:]
    assert ("iw", "cub", 42) not in relaunch_calls
    assert len(relaunch_calls) == 7


def test_run_campaign_raises_on_duplicate_name_before_side_effects(tmp_path, monkeypatch) -> None:
    # the dup-name check is hoisted to the top of run_campaign, so it must fire before any filesystem
    # side effect -- no campaign dir / time.pkl / campaign_metadata.json is created
    monkeypatch.setattr(cr, "paths", {"artifacts": tmp_path})

    with pytest.raises(ValueError, match="Duplicate baseline_overrides name"):
        cr.run_campaign(
            campaign="cmp_dup",
            n_trials=1,
            datasets=("cub",),
            baseline_overrides=[[
                {"loss.targ": "aligned", "name": "dup"},
                {"loss.targ": "phylo", "name": "dup"},
            ]],
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
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
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
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    meta = json.loads(fpath_meta.read_text())
    assert meta["duration"] == "0-01:23:45"
    assert meta["datasets"] == ["cub", "lepid"]


def test_run_campaign_raises_on_removed_setting(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_rm_setting",
        n_trials=1,
        datasets=("cub",),
        baseline_overrides=[[
            {"loss.targ": "aligned", "name": "iw"},
            {"loss.targ": "phylo", "name": "hp"},
        ]],
    )

    with pytest.raises(RuntimeError, match="settings removed.*hp"):
        cr.run_campaign(
            campaign="cmp_rm_setting",
            n_trials=1,
            datasets=("cub",),
            baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
        )


def test_run_campaign_raises_on_removed_dataset(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_rm_dataset",
        n_trials=1,
        datasets=("cub", "lepid"),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    with pytest.raises(RuntimeError, match="datasets removed.*lepid"):
        cr.run_campaign(
            campaign="cmp_rm_dataset",
            n_trials=1,
            datasets=("cub",),
            baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
        )


def test_run_campaign_raises_on_removed_seed(tmp_path, monkeypatch) -> None:
    _setup_completing_campaign(tmp_path, monkeypatch)

    cr.run_campaign(
        campaign="cmp_rm_seed",
        n_trials=2,
        datasets=("cub",),
        baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
    )

    with pytest.raises(RuntimeError, match="seeds removed.*43"):
        cr.run_campaign(
            campaign="cmp_rm_seed",
            n_trials=1,
            datasets=("cub",),
            baseline_overrides=[[{"loss.targ": "aligned", "name": "iw"}]],
        )
