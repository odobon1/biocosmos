import json

import pytest

import test as test_script


def _write_trainval(dpath_trainval, metadata) -> None:
    dpath_trainval.mkdir(parents=True)
    (dpath_trainval / "cfg_baseline.json").write_text(json.dumps({"train": {}}))
    (dpath_trainval / "phase_metadata.json").write_text(json.dumps(metadata))


def test_plan_requires_trainval_phase(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(test_script, "paths", {"artifacts": tmp_path})

    with pytest.raises(FileNotFoundError, match="trainval"):
        test_script._plan("camp")


def test_plan_combos_follow_the_matrix_in_campaign_order(tmp_path, monkeypatch) -> None:
    # datasets-major, then arms, then each arm's picked coord(s), all in the matrix's order -- the same
    # (dataset, arm, coord) sequence the trainval phase ran
    monkeypatch.setattr(test_script, "paths", {"artifacts": tmp_path})
    metadata = {
        "seeds": [42, 43],
        "matrix": {"cub": {"a1": ["c1"], "a2": ["c2"]}, "bryo": {"a1": ["c1", "c3"], "a2": ["c2"]}},
    }
    _write_trainval(tmp_path / "camp" / "trainval", metadata)

    cfg_snapshot, metadata_out, combos = test_script._plan("camp")

    assert cfg_snapshot == {"train": {}}
    assert metadata_out["seeds"] == [42, 43]
    assert combos == [("cub", "a1", "c1"), ("cub", "a2", "c2"),
                      ("bryo", "a1", "c1"), ("bryo", "a1", "c3"), ("bryo", "a2", "c2")]


def test_pending_skips_fully_scored_trials(tmp_path) -> None:
    # a trial is scored once ALL its per-group score files are on disk; a partial write (crash
    # mid-save) leaves it pending, so a relaunch re-runs it
    dpath_test = tmp_path / "test"
    combos = [("cub", "a1", "c1")]
    dpath_seeds = test_script._dpath_coord(dpath_test, "cub", "a1", "c1") / "_seeds"
    for group_key in test_script._EVAL_GROUPS:  # seed 42: fully scored
        (dpath_seeds / "42").mkdir(parents=True, exist_ok=True)
        (dpath_seeds / "42" / f"{group_key}.json").write_text("{}")
    (dpath_seeds / "43").mkdir(parents=True)  # seed 43: partially scored
    (dpath_seeds / "43" / "native.json").write_text("{}")

    pending = test_script._pending(dpath_test, combos, [42, 43, 44])

    assert pending == {("cub", "a1", "c1"): [43, 44]}


def test_seed_test_tree_copies_metadata_and_coord_configs(tmp_path) -> None:
    # the test tree is made self-contained for table rendering: the planned matrix + seeds plus each
    # coord's config.json/overrides.json from the trainval tree; refreshed each run
    dpath_trainval = tmp_path / "trainval"
    dpath_test = tmp_path / "test"
    metadata = {
        "seeds": [42],
        "matrix": {"cub": {"a1": ["c1"]}},
        "n_gpus": 4, "n_crashes": {"ram": 0, "vram": 0, "other": 0},  # runner-side fields stay behind
    }
    dpath_coord_tv = test_script._dpath_coord(dpath_trainval, "cub", "a1", "c1")
    dpath_coord_tv.mkdir(parents=True)
    (dpath_coord_tv / "config.json").write_text(json.dumps({"loss1": {"targ": "sp"}}))
    (dpath_coord_tv / "overrides.json").write_text(json.dumps({"arm": {}, "coord": {}}))

    test_script._seed_test_tree(dpath_test, dpath_trainval, metadata, [("cub", "a1", "c1")])

    written = json.loads((dpath_test / "phase_metadata.json").read_text())
    assert written == {"seeds": [42], "matrix": {"cub": {"a1": ["c1"]}}}
    dpath_coord_test = test_script._dpath_coord(dpath_test, "cub", "a1", "c1")
    assert json.loads((dpath_coord_test / "config.json").read_text()) == {"loss1": {"targ": "sp"}}
    assert json.loads((dpath_coord_test / "overrides.json").read_text()) == {"arm": {}, "coord": {}}
