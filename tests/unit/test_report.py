import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from openpyxl import load_workbook

from utils import report
from utils.train import ArtifactManager
from utils.utils import load_pickle, save_pickle, paths


# {group key: reported name} for the groups in play, the shape utils.config.eval_groups hands the
# renderers; the fixtures below write every group's files, so all four are exercised
_EVAL_GROUPS = {"native": "Native", "native_macro": "Native-Macro", "joint": "Joint", "joint_macro": "Joint-Macro"}
_GROUP_KEYS = tuple(_EVAL_GROUPS)
_SUPP_OFF = {"primitive": False, "n_shot": False}
_SUPP_PRIM = {"primitive": True, "n_shot": False}
_MAP_LABELS = ["All", "ID", "OOD", "I2T", "I2I", "T2I"]
_PRIM_MAP_LABELS = _MAP_LABELS + ["ID I2T", "ID I2I", "ID T2I", "OOD I2T", "OOD I2I", "OOD T2I"]


def _dpath_coord(dpath_phase, dataset, arm, coord):
    return dpath_phase / "_datasets" / dataset / "_arms" / arm / "_coords" / coord


def _write_meta(dpath_phase, arms, coords, datasets, matrix=None, seeds=(42,)) -> None:
    """The phase's phase_metadata.json, its `matrix` ({dataset: {arm: [coords]}}, the planned combos) defaulting
    to every coord under every arm on every dataset -- the screening phase's shape -- and its planned trial
    `seeds` (only arm_complete reads those)."""
    if matrix is None:
        matrix = {dataset: {arm: list(coords) for arm in arms} for dataset in datasets}
    (dpath_phase / "phase_metadata.json").write_text(json.dumps({"matrix": matrix, "seeds": list(seeds)}))


def _captured(grids, *tail):
    """The captured table grid whose scores.png sits under the dir sequence `tail` (e.g. 'arms', 'map', 'native')."""
    return next(grid for fpath, grid in grids if fpath.parent.parts[-len(tail):] == tail)


def test_aggregate_metric_stats_ste_spread() -> None:
    # ste = std / sqrt(n): 7.07 / sqrt(2) = 5.00
    trials = [
        {"scores": {"comp": {"map": {"all": "0.50"}}}},
        {"scores": {"comp": {"map": {"all": "0.60"}}}},
    ]

    out = report._aggregate_metric_stats(trials, "ste")

    assert out["scores"]["comp"]["map"]["all"] == "55.00 ± 5.00"


def test_aggregate_metric_stats_single_trial_returns_leaves_verbatim() -> None:
    trials = [{"scores": {"comp": {"map": {"all": "0.5029"}}}}]

    out = report._aggregate_metric_stats(trials, "std")

    assert out == {"scores": {"comp": {"map": {"all": "0.5029"}}}}


def test_update_metric_stats_counts_trials_lacking_complete_flag(tmp_path, monkeypatch) -> None:
    # completion is now marked by the orchestrator after stats run, so update_metric_stats must aggregate
    # trials by their written selected-checkpoint (_selected/<criterion>/) metrics -- not by a `complete` flag that
    # isn't set yet; each criterion aggregates its own _selected files into its own coord_stats/<criterion>/ subtree
    dataset = "cub"
    dpath_coord = tmp_path
    for seed, map_v, acc_v in (("42", "0.50", "0.30"), ("43", "0.60", "0.40")):
        for criterion, all_v in (("map", map_v), ("acc", acc_v)):
            dpath_selected = dpath_coord / "_seeds" / seed / "evals" / "_selected" / criterion
            dpath_selected.mkdir(parents=True)
            for group_key in _GROUP_KEYS:
                (dpath_selected / f"{group_key}.json").write_text(json.dumps({
                    "scores": {"comp": {"map": {"all": all_v}}},
                    "loss_raw": {"id": "0.70", "ood": None},
                    "sim": {"mean": "0.0925"},
                    "targ": {"mean": "-0.9895"},
                    "chkpt": "1/1 (0.0M/0.0M samples)",
                    "killed": False,
                }))

    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_metric_stats(_EVAL_GROUPS, "std")

    stats = json.loads((dpath_coord / "coord_stats" / "map" / "native" / "metrics.json").read_text())
    assert stats["n_trials"] == 2
    assert stats["scores"]["comp"]["map"]["all"] == "55.00 ± 7.07"
    # only the scores tree is aggregated; the non-score fields are dropped
    assert set(stats) == {"n_trials", "scores"}

    listview_text = (dpath_coord / "coord_stats" / "map" / "native" / "metrics_listview.json").read_text()
    listview = json.loads(listview_text)
    assert listview["n_trials"] == 2
    assert set(listview) == {"n_trials", "scores"}
    assert listview["scores"]["comp"]["map"]["all"] == ["50.00", "60.00"]
    # each leaf list stays on a single line
    assert '"all": ["50.00", "60.00"]' in listview_text
    # the acc tree aggregates the acc-selected _selected files, separately from map's
    stats_acc = json.loads((dpath_coord / "coord_stats" / "acc" / "native" / "metrics.json").read_text())
    assert stats_acc["scores"]["comp"]["map"]["all"] == "35.00 ± 7.07"
    # one aggregate + one listview file per criterion x eval group
    for criterion in ("map", "acc"):
        for group_key in _GROUP_KEYS:
            assert (dpath_coord / "coord_stats" / criterion / group_key / "metrics.json").exists()
            assert (dpath_coord / "coord_stats" / criterion / group_key / "metrics_listview.json").exists()


def test_stats_are_rendered_only_for_the_eval_groups_in_play(tmp_path, monkeypatch) -> None:
    # every renderer keys off the eval_groups map it is handed (reporting.yaml's `eval`), not the set of
    # <group>.json files on disk: a group left out gets no coord_stats subtree even when its metrics are there
    for seed in ("42", "43"):
        for criterion in ("map", "acc"):
            dpath_selected = tmp_path / "_seeds" / seed / "evals" / "_selected" / criterion
            dpath_selected.mkdir(parents=True)
            for group_key in _GROUP_KEYS:
                (dpath_selected / f"{group_key}.json").write_text(json.dumps({
                    "scores": {"comp": {"map": {"all": "0.50"}}},
                    "loss_raw": {"id": "0.70", "ood": None},
                    "sim": {"mean": "0.0925"},
                    "targ": {"mean": "-0.9895"},
                    "chkpt": "1/1 (0.0M/0.0M samples)",
                    "killed": False,
                }))

    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", "cub")

    report.update_metric_stats({"native": "Standard", "joint_macro": "GZSL"}, "std")

    for criterion in ("map", "acc"):
        assert (tmp_path / "coord_stats" / criterion / "native").exists()
        assert (tmp_path / "coord_stats" / criterion / "joint_macro").exists()
        assert not (tmp_path / "coord_stats" / criterion / "native_macro").exists()
        assert not (tmp_path / "coord_stats" / criterion / "joint").exists()


def _write_trial_evals(dpath_trial, chkpt_scores, n_chkpts=None, killed=None):
    """A trial's per-checkpoint eval files: chkpt_scores[i] = (comp.map.all, comp.acc.i2t) at
    checkpoint i, index 0 being the base eval. n_chkpts defaults to the last written index, i.e. a
    completed trial; pass a larger one to leave the trial short of its final eval -- or, with
    killed=<that last index>, a trial killed there (trial_metadata.json's killed field)."""
    n_chkpts = len(chkpt_scores) - 1 if n_chkpts is None else n_chkpts
    for idx_eval, (map_v, acc_v) in enumerate(chkpt_scores):
        dpath_chkpt = dpath_trial / "evals" / ("base" if idx_eval == 0 else f"eval{idx_eval}")
        dpath_chkpt.mkdir(parents=True)
        for group_key in _GROUP_KEYS:
            (dpath_chkpt / f"{group_key}.json").write_text(json.dumps({
                "scores": {"comp": {"map": {"all": map_v}, "acc": {"i2t": acc_v}}},
                "chkpt": f"{idx_eval}/{n_chkpts} (0.0M/0.0M samples)",
            }))
    (dpath_trial / "trial_metadata.json").write_text(json.dumps({"killed": killed}))


def test_update_chkpt_selection_picks_argmax_of_the_mean_curve(tmp_path, monkeypatch) -> None:
    # the coord picks ONE checkpoint index per criterion x group -- argmax over the across-trial
    # MEAN curve, not each trial's own argmax -- and every trial is scored there. Trial 42 peaks at
    # chkpt 1 and trial 43 at chkpt 3, but the mean peaks at 2, so BOTH are scored at chkpt 2.
    dataset = "cub"
    dpath_coord = tmp_path
    (tmp_path / "coord_metadata.json").write_text(json.dumps({"n_crashes": {}, "best_chkpt": {}}))
    _write_trial_evals(dpath_coord / "_seeds" / "42", (("0.10", "0.10"), ("0.90", "0.90"), ("0.50", "0.50"), ("0.20", "0.20")))
    _write_trial_evals(dpath_coord / "_seeds" / "43", (("0.10", "0.10"), ("0.10", "0.10"), ("0.70", "0.70"), ("0.80", "0.80")))

    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_chkpt_selection(_EVAL_GROUPS, "std")

    chkpt_means = load_pickle(dpath_coord / "coord_stats" / "map" / "native" / "chkpt_means.pkl")
    assert chkpt_means["n_trials"] == 2
    assert list(chkpt_means["chkpts"]) == [0, 1, 2, 3]  # the base eval leads the curve
    assert chkpt_means["means"] == pytest.approx([0.10, 0.50, 0.60, 0.50])
    assert chkpt_means["idx_best"] == 2  # argmax over 1.., earliest on ties

    # _selected: both trials scored at the SAME checkpoint -- neither trial's own argmax
    for seed, score in (("42", "0.50"), ("43", "0.70")):
        sel = json.loads((dpath_coord / "_seeds" / seed / "evals" / "_selected" / "map" / "native.json").read_text())
        assert sel["chkpt"].startswith("2/3")
        assert sel["scores"]["comp"]["map"]["all"] == score  # 42's own best was 0.90, 43's 0.80

    # _best: each trial at its OWN argmax (42 peaked at chkpt 1, 43 at chkpt 3)
    for seed, idx, score in (("42", 1, "0.90"), ("43", 3, "0.80")):
        own = json.loads((dpath_coord / "_seeds" / seed / "evals" / "_best" / "map" / "native.json").read_text())
        assert own["chkpt"].startswith(f"{idx}/3")
        assert own["scores"]["comp"]["map"]["all"] == score

    metadata = json.loads((tmp_path / "coord_metadata.json").read_text())
    assert metadata["best_chkpt"]["map"]["native"] == {
        "idx": 2, "n_trials": 2, "mean": "0.6000",
    }
    for criterion in ("map", "acc"):
        for group_key in _GROUP_KEYS:
            assert (dpath_coord / "coord_stats" / criterion / group_key / "chkpt_means.pkl").exists()
            assert (dpath_coord / "coord_stats" / criterion / group_key / "chkpt_means.png").exists()
            assert metadata["best_chkpt"][criterion][group_key]["idx"] == 2


def test_update_chkpt_selection_excludes_unfinished_trials(tmp_path, monkeypatch) -> None:
    # a trial short of its final eval doesn't enter the mean at all (nor get a _selected/ or _best/):
    # here 43 stopped at chkpt 1 of 2
    dataset = "cub"
    dpath_coord = tmp_path
    (tmp_path / "coord_metadata.json").write_text(json.dumps({"n_crashes": {}, "best_chkpt": {}}))
    _write_trial_evals(dpath_coord / "_seeds" / "42", (("0.90", "0.90"), ("0.10", "0.10"), ("0.30", "0.30")))
    _write_trial_evals(dpath_coord / "_seeds" / "43", (("0.10", "0.10"), ("0.99", "0.99")), n_chkpts=2)

    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_chkpt_selection(_EVAL_GROUPS, "std")

    chkpt_means = load_pickle(dpath_coord / "coord_stats" / "map" / "native" / "chkpt_means.pkl")
    assert chkpt_means["n_trials"] == 1  # only trial 42 counted
    assert chkpt_means["means"] == pytest.approx([0.90, 0.10, 0.30])  # 42's curve alone
    assert list(chkpt_means["spreads"]) == [0.0] * 3  # ddof=1 undefined for one trial -> flat band
    assert chkpt_means["idx_best"] == 0  # the base eval competes like any checkpoint, and 0.90 tops the curve
    own = json.loads((dpath_coord / "_seeds" / "42" / "evals" / "_best" / "map" / "native.json").read_text())
    assert own["chkpt"].startswith("0/2")  # the trial's own argmax lands on the base eval too
    assert not (dpath_coord / "_seeds" / "43" / "evals" / "_selected").exists()
    assert not (dpath_coord / "_seeds" / "43" / "evals" / "_best").exists()


def test_update_chkpt_selection_base_eval_competes_like_any_checkpoint(tmp_path, monkeypatch) -> None:
    # the base eval (index 0) competes like any checkpoint: here it tops the mean
    # curve, so every trial's _selected/ is a copy of its evals/base/, and 42's own argmax is the
    # base too while 43's is chkpt 2
    dataset = "cub"
    dpath_coord = tmp_path
    (tmp_path / "coord_metadata.json").write_text(json.dumps({"n_crashes": {}, "best_chkpt": {}}))
    _write_trial_evals(dpath_coord / "_seeds" / "42", (("0.90", "0.90"), ("0.10", "0.10"), ("0.30", "0.30")))
    _write_trial_evals(dpath_coord / "_seeds" / "43", (("0.50", "0.50"), ("0.20", "0.20"), ("0.60", "0.60")))

    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_chkpt_selection(_EVAL_GROUPS, "std")

    chkpt_means = load_pickle(dpath_coord / "coord_stats" / "map" / "native" / "chkpt_means.pkl")
    assert chkpt_means["means"] == pytest.approx([0.70, 0.15, 0.45])
    assert chkpt_means["idx_best"] == 0
    for seed, score in (("42", "0.90"), ("43", "0.50")):
        sel = json.loads((dpath_coord / "_seeds" / seed / "evals" / "_selected" / "map" / "native.json").read_text())
        assert sel["chkpt"].startswith("0/2")
        assert sel["scores"]["comp"]["map"]["all"] == score
    for seed, idx, score in (("42", 0, "0.90"), ("43", 2, "0.60")):
        own = json.loads((dpath_coord / "_seeds" / seed / "evals" / "_best" / "map" / "native.json").read_text())
        assert own["chkpt"].startswith(f"{idx}/2")
        assert own["scores"]["comp"]["map"]["all"] == score
    metadata = json.loads((tmp_path / "coord_metadata.json").read_text())
    assert metadata["best_chkpt"]["map"]["native"] == {"idx": 0, "n_trials": 2, "mean": "0.7000"}


def test_chkpt_dpaths_completes_a_killed_trial_at_its_kill_checkpoint(tmp_path) -> None:
    # a trial killed at eval k (kill_thresh; trial_metadata.json's killed = k) is complete with its
    # evals ending there, where the same evals without the marker are a trial still short of its final
    _write_trial_evals(tmp_path / "killed", (("0.5", "0.5"), ("0.4", "0.4"), ("0.3", "0.3")), n_chkpts=5, killed=2)
    _write_trial_evals(tmp_path / "short", (("0.5", "0.5"), ("0.4", "0.4"), ("0.3", "0.3")), n_chkpts=5)

    assert [d.name for d in report._chkpt_dpaths(tmp_path / "killed")] == ["base", "eval1", "eval2"]
    assert report._chkpt_dpaths(tmp_path / "short") is None


def test_update_chkpt_selection_scores_killed_trials_at_their_own_best(tmp_path, monkeypatch) -> None:
    # trial 43 was killed at chkpt 1 of 3 (nothing beat its base), so it has no eval at the coord's
    # selected index: it stays out of the mean curve (42's alone, selecting chkpt 2) and is scored at
    # its own best eval -- the base here, since nothing beat it -- its _selected copy flagged killed
    dataset = "cub"
    dpath_coord = tmp_path
    (tmp_path / "coord_metadata.json").write_text(json.dumps({"n_crashes": {}, "best_chkpt": {}}))
    _write_trial_evals(dpath_coord / "_seeds" / "42", (("0.10", "0.10"), ("0.30", "0.30"), ("0.60", "0.60"), ("0.50", "0.50")))
    _write_trial_evals(dpath_coord / "_seeds" / "43", (("0.40", "0.40"), ("0.20", "0.20")), n_chkpts=3, killed=1)

    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_chkpt_selection(_EVAL_GROUPS, "std")

    chkpt_means = load_pickle(dpath_coord / "coord_stats" / "map" / "native" / "chkpt_means.pkl")
    assert chkpt_means["n_trials"] == 1
    assert chkpt_means["means"] == pytest.approx([0.10, 0.30, 0.60, 0.50])
    assert chkpt_means["idx_best"] == 2
    sel = json.loads((dpath_coord / "_seeds" / "42" / "evals" / "_selected" / "map" / "native.json").read_text())
    assert sel["chkpt"].startswith("2/3") and sel["killed"] is False
    sel = json.loads((dpath_coord / "_seeds" / "43" / "evals" / "_selected" / "map" / "native.json").read_text())
    assert sel["chkpt"].startswith("0/3") and sel["scores"]["comp"]["map"]["all"] == "0.40" and sel["killed"] is True
    own = json.loads((dpath_coord / "_seeds" / "43" / "evals" / "_best" / "map" / "native.json").read_text())
    assert own["chkpt"].startswith("0/3") and "killed" not in own
    metadata = json.loads((tmp_path / "coord_metadata.json").read_text())
    assert metadata["best_chkpt"]["map"]["native"] == {"idx": 2, "n_trials": 1, "mean": "0.6000"}


def test_update_chkpt_selection_selects_over_killed_trials_when_all_were_killed(tmp_path, monkeypatch) -> None:
    # every trial killed at chkpt 1 of 3: their equally short curves form the mean curve (so the coord
    # still has a selection for the qual pick / trainval stop); each is still scored at its own best --
    # the base eval, which nothing beat
    dataset = "cub"
    dpath_coord = tmp_path
    (tmp_path / "coord_metadata.json").write_text(json.dumps({"n_crashes": {}, "best_chkpt": {}}))
    _write_trial_evals(dpath_coord / "_seeds" / "42", (("0.50", "0.50"), ("0.20", "0.20")), n_chkpts=3, killed=1)
    _write_trial_evals(dpath_coord / "_seeds" / "43", (("0.40", "0.40"), ("0.30", "0.30")), n_chkpts=3, killed=1)

    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_chkpt_selection(_EVAL_GROUPS, "std")

    chkpt_means = load_pickle(dpath_coord / "coord_stats" / "map" / "native" / "chkpt_means.pkl")
    assert chkpt_means["n_trials"] == 2
    assert list(chkpt_means["chkpts"]) == [0, 1]
    assert chkpt_means["means"] == pytest.approx([0.45, 0.25])
    assert chkpt_means["idx_best"] == 0
    for seed, score in (("42", "0.50"), ("43", "0.40")):
        sel = json.loads((dpath_coord / "_seeds" / seed / "evals" / "_selected" / "map" / "native.json").read_text())
        assert sel["chkpt"].startswith("0/3") and sel["scores"]["comp"]["map"]["all"] == score and sel["killed"] is True


def test_sweep_completion_levels(tmp_path, monkeypatch) -> None:
    # each stats level re-renders only once a seed has completed across its own cycle -- the arm's
    # coords (arm_metrics), the dataset's arms x coords (dataset_metrics), the whole matrix
    # (phase_metrics) -- since each trial completion reselects only its own coord's checkpoint. A
    # trial short of its final eval doesn't count.
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)
    _write_meta(tmp_path, ["sp", "mp"], ["c0", "c1"], ["cub", "lepid"])
    scores = (("0.10", "0.10"), ("0.30", "0.30"))
    for arm in ("sp", "mp"):
        for coord in ("c0", "c1"):
            for dataset in ("cub", "lepid"):
                _write_trial_evals(_dpath_coord(tmp_path, dataset, arm, coord) / "_seeds" / "42", scores)
                # seed 43 has run everywhere but mp/c1/lepid, where it stopped short of its final eval
                short = (arm, coord, dataset) == ("mp", "c1", "lepid")
                _write_trial_evals(_dpath_coord(tmp_path, dataset, arm, coord) / "_seeds" / "43", scores, n_chkpts=2 if short else None)

    assert report.seed_sweep_complete(42)
    assert report.dataset_sweep_complete(42, "lepid")
    assert report.arm_sweep_complete(42, "lepid", "mp")

    assert not report.seed_sweep_complete(43)
    assert report.dataset_sweep_complete(43, "cub")  # cub's cycle closed regardless of lepid's
    assert not report.dataset_sweep_complete(43, "lepid")
    assert report.arm_sweep_complete(43, "lepid", "sp")  # sp's coords are all in on lepid
    assert not report.arm_sweep_complete(43, "lepid", "mp")  # mp/c1 still short

    dpath_eval = _dpath_coord(tmp_path, "lepid", "mp", "c1") / "_seeds" / "43" / "evals" / "eval2"
    dpath_eval.mkdir()
    for group_key in _GROUP_KEYS:  # its final eval lands -> every cycle closes
        (dpath_eval / f"{group_key}.json").write_text(json.dumps({
            "scores": {"comp": {"map": {"all": "0.30"}, "acc": {"i2t": "0.30"}}},
            "chkpt": "2/2 (0.0M/0.0M samples)",
        }))

    assert report.arm_sweep_complete(43, "lepid", "mp")
    assert report.dataset_sweep_complete(43, "lepid")
    assert report.seed_sweep_complete(43)


def _comp(base: float) -> dict:
    return {
        "acc": {"i2t": f"{base + 0.06:.4f}"},
        "map": {
            "all": f"{base:.4f}",
            "ood": f"{base + 0.01:.4f}",
            "id": f"{base + 0.02:.4f}",
            "i2t": f"{base + 0.03:.4f}",
            "i2i": f"{base + 0.04:.4f}",
            "t2i": f"{base + 0.05:.4f}",
        },
    }


def _scores_grp(comp: dict) -> dict:
    # full per-grp scores subtree as written to a per-group metrics file: comp + per-partition primitive scores
    prim = {"map": {"i2t": "0.10", "i2i": "0.10", "t2i": "0.10"}, "acc": {"i2t": "0.10"}}
    return {"comp": comp, "id": prim, "ood": prim}


def _write_group_metrics(dpath_selected, scores_grp: dict, macro: dict | None = None, acc_selected: dict | None = None,
                         killed: bool = False) -> None:
    # trial-end materialization (report.update_chkpt_selection) writes one metrics file per eval
    # group under each selection criterion; fixtures reuse one subtree per averaging axis across
    # both sets, and the same content for both criteria unless acc_selected supplies the
    # acc-criterion subtree. A completed trial also always has its trial_metadata.json (hardware
    # readings) and its coord's coord_metadata.json (crash counters), which the always-on
    # 'Hardware Performance' sheet reads, and its coord's chkpt_means.pkl (the across-trial mean
    # curve the convergence plots read): placeholders are written here (the coord's only if
    # absent -- a curve whose selected point is the coord's first trial's score), for tests to
    # overwrite when they assert on them.
    macro = scores_grp if macro is None else macro
    dpath_trial = dpath_selected.parent.parent  # <coord>/_seeds/<seed>/evals/_selected
    dpath_coord = dpath_trial.parents[1]
    for criterion, (grp_std, grp_macro) in (("map", (scores_grp, macro)),
                                            ("acc", (acc_selected or scores_grp, acc_selected or macro))):
        (dpath_selected / criterion).mkdir(parents=True, exist_ok=True)
        score_key, metric = report.BEST_CRITERIA[criterion]
        for group_key, grp in (("native", grp_std), ("native_macro", grp_macro), ("joint", grp_std), ("joint_macro", grp_macro)):
            (dpath_selected / criterion / f"{group_key}.json").write_text(json.dumps({"scores": grp, "killed": killed}))
            fpath_means = dpath_coord / "coord_stats" / criterion / group_key / "chkpt_means.pkl"
            if not fpath_means.exists():
                fpath_means.parent.mkdir(parents=True, exist_ok=True)
                save_pickle({
                    "n_trials": 1,
                    "chkpts": np.arange(2),
                    "means": np.array([0.0, float(grp["comp"][score_key][metric])]),
                    "spreads": np.zeros(2),
                    "idx_best": 1,
                }, fpath_means)
    (dpath_trial / "trial_metadata.json").write_text(json.dumps({
        "runtime": {"train": {"mean": "1.00"}, "eval": {"mean": "1.00"}, "trial": "1.00"},
        "memory": {"ram": "1.0/128.0 GB", "vram": "1.0/178.4 GB"},
        "killed": 1 if killed else None,  # the eval index a killed trial stopped at
        "progress": {"epoch": 2, "n_epochs": 2, "n_samps_seen": 200},  # the strips' epoch axis reads n_epochs
    }))
    # the strip figures read every completed trial's recorded series, and every trial records its logit scale
    # (frozen or not), so these carry a minimal one: two batches, a BCE-family loss's empty bound trio. The
    # strip tests (_write_strip_trial) write their own trials with the series under test
    save_pickle({"epoch": {
        "n_samps_seen": [100, 200], "scale": [10.0, 10.0], "logit_scale": [2.3, 2.3],
        **{f"{p}alpha_req_{stat}": [] for p in ("", "log_") for stat in ("min", "mean", "max")},
    }}, dpath_trial / "data_trial.pkl")
    fpath_meta_coord = dpath_coord / "coord_metadata.json"
    if not fpath_meta_coord.exists():
        # horizon: the coord's sample volume (utils.train.save_metadata_coord writes it for every coord);
        # the strip figures read it for their epoch axis, and skip these trials, which have no evals/
        fpath_meta_coord.write_text(json.dumps({
            "n_crashes": {"ram": 0, "vram": 0, "other": 0},
            "horizon": {"n_samps": {"total": 200, "warmup": 0}, "n_steps": {"total": 2, "warmup": 0}},
        }))


def _entry(base: float, killed: bool = False) -> dict:
    # a _collect_comps entry: the comp score maps plus the trial's killed flag
    return {**_comp(base), "killed": killed}


def test_stats_table_grid_formats_by_trial_count() -> None:
    # row keys are tuples of key cells (one per header); the trial count goes on the last key cell
    grid, killed_rows = report._stats_table_grid(
        ("Arm", "Coord"),
        ("All", "ID", "OOD", "I2T", "I2I", "T2I"),
        [
            (("hp", "c0"), [_entry(0.50), _entry(0.60)]),
            (("mp", "c0"), [_entry(0.50)]),
            (("sp", "c0"), []),
        ],
        "map",
        "std",
    )
    assert killed_rows == set()

    assert grid[0] == ["Arm", "Coord", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert [row[:2] for row in grid[1:]] == [["hp", "c0 (2)"], ["mp", "c0 (1)"], ["sp", "c0 (0)"]]
    assert grid[1][2] == "55.00 ± 7.07"  # 2 trials: mean ± std
    assert grid[2][2] == "50.00"  # 1 trial: mean only
    assert grid[3][2] == "-"  # 0 trials
    assert grid[1][3] == "57.00 ± 7.07"  # "ID" column reads score key "id"


def test_stats_table_grid_ste_spread() -> None:
    grid, _ = report._stats_table_grid(
        ("Arm",),
        ("All",),
        [(("hp",), [_entry(0.50), _entry(0.60)])],
        "map",
        "ste",
    )

    assert grid[1][1] == "55.00 ± 5.00"  # ste = std / sqrt(n): 7.07 / sqrt(2)


def test_stats_table_grid_single_acc_column() -> None:
    grid, _ = report._stats_table_grid(
        ("Arm",),
        ("I2T",),
        [(("hp",), [_entry(0.50), _entry(0.60)])],
        "acc",
        "std",
    )

    assert grid == [["Arm", "I2T"], ["hp (2)", "61.00 ± 7.07"]]


def test_stats_table_grid_marks_killed_rows() -> None:
    # a row aggregating a killed trial (kill_thresh) shows the killed count on its key cell and is
    # reported for the renderers' yellow shading; its scores aggregate like any other
    grid, killed_rows = report._stats_table_grid(
        ("Coord",),
        ("All",),
        [(("c0",), [_entry(0.50), _entry(0.60, killed=True)]), (("c1",), [_entry(0.40)])],
        "map",
        "std",
    )

    assert grid == [["Coord", "All"], ["c0 (2, 1 killed)", "55.00 ± 7.07"], ["c1 (1)", "40.00"]]
    assert killed_rows == {1}


def test_update_arm_metrics_writes_pngs(tmp_path, monkeypatch) -> None:
    # the arm's arm_metrics/performance/ tables + convergence plots for one dataset, with one 'Coord' row per coord of
    # the arm with >= 1 completed trial there -- "c1" is planned in phase_metadata.json but has no
    # trials, so it gets no row; an arm with no dir on the dataset (planned "mp", never launched) is
    # skipped outright. bold_high=True + heatmap=True exercise the real matplotlib styling paths
    # (winner bold + heatmap shading) end-to-end.
    dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _scores_grp(_comp(0.50)))
    _write_meta(tmp_path, ["hp", "mp"], ["c0", "c1"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_arm_metrics("cub", "hp", _EVAL_GROUPS, "std", True, False, True, _SUPP_OFF)
    report.update_arm_metrics("cub", "mp", _EVAL_GROUPS, "std", True, False, True, _SUPP_OFF)
    dpath_stats = tmp_path / "_datasets" / "cub" / "_arms" / "hp" / "arm_metrics" / "performance"
    for criterion in ("map", "acc"):
        for group_key in _GROUP_KEYS:
            assert (dpath_stats / criterion / group_key / "scores.png").exists()
            assert (dpath_stats / criterion / group_key / "convergence.png").exists()
    assert not (tmp_path / "_datasets" / "cub" / "_arms" / "mp").exists()


def test_update_arm_metrics_rows_and_curves(tmp_path, monkeypatch) -> None:
    # grid contents: a 'Coord' key column, one row per coord with local trials (campaign order), the
    # criterion's own scores; the convergence plot overlays those coords' mean curves, the winner being
    # the highest mean at its own selected checkpoint (b: 0.60 at chkpt 2 -- a peaks earlier but lower)
    for coord, base in (("a", 0.40), ("b", 0.60)):
        dpath_selected = _dpath_coord(tmp_path, "cub", "hp", coord) / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_comp(base)))
    _write_chkpt_means(tmp_path, "cub", "hp", "a", [0.10, 0.40, 0.30], 1)
    _write_chkpt_means(tmp_path, "cub", "hp", "b", [0.10, 0.20, 0.60], 2)
    _write_meta(tmp_path, ["hp"], ["a", "b", "c"], ["cub"])

    grids, plotted = [], []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)
    monkeypatch.setattr(report, "_render_stats_table", lambda grid, n_keys, title, fpath, bold_high, heatmap, killed_rows: grids.append((fpath, grid)))
    monkeypatch.setattr(report, "_plot_convergence", lambda curves, idx_win, score_name, title, fpath: plotted.append((fpath, curves, idx_win)))

    report.update_arm_metrics("cub", "hp", _EVAL_GROUPS, "std", False, False, False, _SUPP_OFF)
    assert len(grids) == 8 and len(plotted) == 8  # map + acc per eval group
    assert _captured(grids, "performance", "map", "native") == [
        ["Coord", *_MAP_LABELS],
        ["a (1)", "40.00", "42.00", "41.00", "43.00", "44.00", "45.00"],
        ["b (1)", "60.00", "62.00", "61.00", "63.00", "64.00", "65.00"],
    ]
    assert _captured(grids, "performance", "acc", "native") == [["Coord", "I2T"], ["a (1)", "46.00"], ["b (1)", "66.00"]]
    fpath, curves, idx_win = next(p for p in plotted if p[0].parent.parts[-3:] == ("performance", "map", "native"))
    assert fpath.name == "convergence.png"
    assert [row for row, _, _ in curves] == [("a",), ("b",)]
    assert curves[idx_win][0] == ("b",)


def _write_complete_trial(dpath_phase, dataset, arm, coord, seed, base) -> None:
    """A completed trial of (arm, coord) on `dataset` at `seed`: its per-checkpoint evals (peaking at `base`,
    so it reads as complete), its selected-checkpoint metrics, and the learning_curves/ dir the best_coord/
    mirror copies -- each png's bytes naming the trial it came from."""
    dpath_trial = _dpath_coord(dpath_phase, dataset, arm, coord) / "_seeds" / str(seed)
    _write_trial_evals(dpath_trial, (("0.10", "0.10"), (f"{base:.4f}", f"{base:.4f}")))
    _write_group_metrics(dpath_trial / "evals" / "_selected", _scores_grp(_comp(base)))
    dpath_curves = dpath_trial / "learning_curves"
    (dpath_curves / "scores").mkdir(parents=True)
    (dpath_curves / "general.png").write_text(f"{arm}/{coord}/{seed} general")
    (dpath_curves / "scores" / "native.png").write_text(f"{arm}/{coord}/{seed} scores")


def test_update_arm_metrics_mirrors_the_best_coords_learning_curves(tmp_path, monkeypatch) -> None:
    # a complete arm gets best_coord/learning_curves/_seeds/<seed>/, a copy of its BEST coord's per-trial
    # learning curves -- highest across-trial mean Native mAP comp All, b's 0.60/0.70 over a's 0.40/0.50 --
    # one dir per seed, whole (scores/ included), with the losing coord's curves nowhere in it
    dpath_phase = tmp_path / "_screen"
    for coord, bases in (("a", (0.40, 0.50)), ("b", (0.60, 0.70))):
        for seed, base in zip((42, 43), bases):
            _write_complete_trial(dpath_phase, "cub", "hp", coord, seed, base)
    _write_meta(dpath_phase, ["hp"], ["a", "b"], ["cub"], seeds=[42, 43])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)

    report.update_arm_metrics("cub", "hp", _EVAL_GROUPS, "std", False, False, False, _SUPP_OFF)

    dpath_curves = dpath_phase / "_datasets" / "cub" / "_arms" / "hp" / "best_coord" / "learning_curves"
    assert sorted(p.name for p in dpath_curves.iterdir()) == ["42", "43"]
    assert (dpath_curves / "42" / "general.png").read_text() == "hp/b/42 general"
    assert (dpath_curves / "43" / "general.png").read_text() == "hp/b/43 general"
    assert (dpath_curves / "43" / "scores" / "native.png").read_text() == "hp/b/43 scores"


def test_best_coord_curves_wait_for_every_planned_trial(tmp_path, monkeypatch) -> None:
    # the dir stands for a FINISHED arm -- every planned coord x seed -- not for a closed seed cycle: with
    # b's seed 43 missing, seed 42's cycle is closed everywhere (arm_sweep_complete) yet nothing is written
    dpath_phase = tmp_path / "_screen"
    for coord in ("a", "b"):
        for seed in (42, 43):
            if (coord, seed) != ("b", 43):
                _write_complete_trial(dpath_phase, "cub", "hp", coord, seed, 0.50)
    _write_meta(dpath_phase, ["hp"], ["a", "b"], ["cub"], seeds=[42, 43])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)
    dpath_best = dpath_phase / "_datasets" / "cub" / "_arms" / "hp" / "best_coord"

    assert report.arm_sweep_complete(42, "cub", "hp")
    assert not report.arm_complete("cub", "hp")
    report.update_best_coord_curves("cub", "hp")
    assert not dpath_best.exists()

    _write_complete_trial(dpath_phase, "cub", "hp", "b", 43, 0.50)  # the arm's last trial lands
    report.update_best_coord_curves("cub", "hp")
    assert sorted(p.name for p in (dpath_best / "learning_curves").iterdir()) == ["42", "43"]


def test_best_coord_curves_are_dropped_when_the_arm_grows(tmp_path, monkeypatch) -> None:
    # a coord added to the arm (an item added to a combo group / list) leaves it short of its plan again: the
    # mirror the earlier completion wrote is deleted, and written afresh against the NEW coord set once the
    # arm completes over it -- c wins there, so its curves replace b's
    dpath_phase = tmp_path / "_screen"
    for coord, base in (("a", 0.40), ("b", 0.60)):
        _write_complete_trial(dpath_phase, "cub", "hp", coord, 42, base)
    _write_meta(dpath_phase, ["hp"], ["a", "b"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)
    dpath_curves = dpath_phase / "_datasets" / "cub" / "_arms" / "hp" / "best_coord" / "learning_curves"

    report.update_best_coord_curves("cub", "hp")
    assert (dpath_curves / "42" / "general.png").read_text() == "hp/b/42 general"

    _write_meta(dpath_phase, ["hp"], ["a", "b", "c"], ["cub"])  # "c" added, no trials of it yet
    report.update_best_coord_curves("cub", "hp")
    assert not dpath_curves.parent.exists()

    _write_complete_trial(dpath_phase, "cub", "hp", "c", 42, 0.90)
    report.update_best_coord_curves("cub", "hp")
    assert (dpath_curves / "42" / "general.png").read_text() == "hp/c/42 general"


def test_best_coord_curves_are_screening_only(tmp_path, monkeypatch) -> None:
    # the qual matrix reduces each arm to its pick(s), so there best_coord/ would just duplicate the pick's own
    # coord dir -- only the _screen phase gets it, keyed off the phase dir's name as dataset_metrics/arms/ is
    dpath_phase = tmp_path / "qual"
    _write_complete_trial(dpath_phase, "cub", "hp", "c0", 42, 0.50)
    _write_meta(dpath_phase, ["hp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)

    report.update_best_coord_curves("cub", "hp")

    assert not (dpath_phase / "_datasets" / "cub" / "_arms" / "hp" / "best_coord").exists()


def _write_strip_trial(dpath_phase, dataset, arm, coord, seed, scale, n_epochs=2, samps_per_epoch=100,
                       reqs=True) -> None:
    """A completed trial carrying what the strip figures read: its evals (so it reads as complete), the
    pair the epoch axis is rebuilt from (coord_metadata.json's sample volume over trial_metadata.json's
    n_epochs), and a data_trial.pkl whose `scale` series is given, its logalpha counterpart the same
    values negated so the two figures are told apart. reqs=False leaves the target-implied bound trio
    empty, as a BCE-family loss does."""
    dpath_coord = _dpath_coord(dpath_phase, dataset, arm, coord)
    dpath_trial = dpath_coord / "_seeds" / str(seed)
    _write_trial_evals(dpath_trial, (("0.10", "0.10"), ("0.20", "0.20")))
    (dpath_trial / "trial_metadata.json").write_text(json.dumps(
        {"killed": None, "progress": {"epoch": n_epochs, "n_epochs": n_epochs, "n_samps_seen": n_epochs * samps_per_epoch}}))
    (dpath_coord / "coord_metadata.json").write_text(json.dumps(
        {"n_crashes": {}, "horizon": {"n_samps": {"total": n_epochs * samps_per_epoch}}}))
    n = len(scale)
    req = {stat: [float(i) + off for i in range(n)] if reqs else [] for stat, off in (("min", 0.0), ("mean", 1.0), ("max", 2.0))}
    save_pickle({"epoch": {
        "n_samps_seen": [(i + 1) * n_epochs * samps_per_epoch / n for i in range(n)],
        "scale": list(scale),
        "logit_scale": [-v for v in scale],
        **{f"alpha_req_{stat}": vals for stat, vals in req.items()},
        **{f"log_alpha_req_{stat}": vals for stat, vals in req.items()},
    }}, dpath_trial / "data_trial.pkl")


def test_update_coord_strips_writes_per_seed_and_agg_figures(tmp_path, monkeypatch) -> None:
    # the arm's logit-scale strips: one figure per seed plus the agg one, each in both parameterizations
    # (alpha, logalpha), under arm_metrics/coord_strips/
    dpath_phase = tmp_path / "_screen"
    for coord in ("a", "b"):
        for seed in (42, 43):
            _write_strip_trial(dpath_phase, "cub", "hp", coord, seed, [10.0, 11.0, 12.0, 13.0])
    _write_meta(dpath_phase, ["hp"], ["a", "b"], ["cub"], seeds=[42, 43])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)

    report.update_coord_strips("cub", "hp", "std")

    dpath_strips = dpath_phase / "_datasets" / "cub" / "_arms" / "hp" / "arm_metrics" / "coord_strips"
    for name in ("alpha", "logalpha"):
        assert (dpath_strips / "agg" / f"{name}.png").exists()
        assert (dpath_strips / "seeds" / "42" / f"{name}.png").exists()
        assert (dpath_strips / "seeds" / "43" / f"{name}.png").exists()


def test_update_coord_strips_rows_and_aggregation(tmp_path, monkeypatch) -> None:
    # a seed's figure draws that seed's own trial per coord, unbanded, on an epoch axis rebuilt from the
    # recorded sample volume / n_epochs; the agg one draws the mean over the coord's trials with a spread
    # band on the scale line AND on every bound line. A coord with no completed trial at a seed is left
    # out of that seed's figure; the strips keep campaign coord order.
    dpath_phase = tmp_path / "_screen"
    _write_strip_trial(dpath_phase, "cub", "hp", "a", 42, [10.0, 20.0])
    _write_strip_trial(dpath_phase, "cub", "hp", "a", 43, [30.0, 40.0])
    _write_strip_trial(dpath_phase, "cub", "hp", "b", 42, [1.0, 2.0])  # b never ran seed 43
    _write_meta(dpath_phase, ["hp"], ["a", "b"], ["cub"], seeds=[42, 43])

    calls = []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)
    monkeypatch.setattr(report, "_render_strips", lambda strips, fpath, *args: calls.append((fpath, strips)))

    report.update_coord_strips("cub", "hp", "std")

    figures = {(fpath.parent.name, fpath.stem): strips for fpath, strips in calls}
    assert [coord for coord, *_ in figures[("42", "alpha")]] == ["a", "b"]
    assert [coord for coord, *_ in figures[("43", "alpha")]] == ["a"]
    _, x, (vals, spread), reqs = figures[("42", "alpha")][0]
    assert x == pytest.approx([0.0, 1.0])  # epochs: n_samps_seen / samps_per_epoch, anchored at 0
    assert vals == pytest.approx([10.0, 20.0]) and spread is None
    assert [s for _, s in reqs.values()] == [None, None, None]  # the bound lines are unbanded too
    _, _, (vals, spread), reqs = figures[("agg", "alpha")][0]
    assert vals == pytest.approx([20.0, 30.0])  # a's two trials, meaned
    assert spread == pytest.approx([np.std([10.0, 30.0], ddof=1)] * 2)
    assert all(s == pytest.approx([0.0, 0.0]) for _, s in reqs.values())  # identical across a's trials
    # the logalpha figure reads the log-scale series (negated here), not the alpha one
    _, _, (vals, _), _ = figures[("agg", "logalpha")][0]
    assert vals == pytest.approx([-20.0, -30.0])


def test_update_coord_strips_keep_frozen_and_boundless_coords(tmp_path, monkeypatch) -> None:
    # a coord with a frozen logit scale keeps its strip -- the series is recorded frozen or not, a flat line
    # there -- as does one whose loss records no target-implied bounds (BCE family), with no bound lines on
    # it. An arm with no completed trial at all gets no figure.
    dpath_phase = tmp_path / "_screen"
    _write_strip_trial(dpath_phase, "cub", "hp", "a", 42, [14.0, 14.0], reqs=False)  # frozen
    _write_strip_trial(dpath_phase, "cub", "hp", "b", 42, [1.0, 2.0], reqs=False)
    _write_meta(dpath_phase, ["hp", "mp"], ["a", "b"], ["cub"], seeds=[42])

    calls = []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)
    monkeypatch.setattr(report, "_render_strips", lambda strips, fpath, *args: calls.append((fpath, strips)))

    report.update_coord_strips("cub", "hp", "std")

    assert [[coord for coord, *_ in strips] for _, strips in calls] == [["a", "b"]] * 4
    assert all(reqs == {} for _, strips in calls for *_, reqs in strips)
    _, _, (vals, _), _ = calls[0][1][0]
    assert vals == pytest.approx([14.0, 14.0])

    calls.clear()
    report.update_coord_strips("cub", "mp", "std")  # planned, nothing run
    assert calls == []


def test_render_strips_tick_a_flat_strip_at_its_value(tmp_path, monkeypatch) -> None:
    # the strip figures draw the trial figures' scale panel, y axis included (_pin_flat_yaxis): a frozen scale
    # with no bound beside it gets its value as the one tick; one read against a bound trio autoscales over
    # both, as a learnable scale does
    seen = {}

    def finish(fig, axes, *args, **kwargs):
        seen["yticks"] = [[t.get_text() for t in ax.get_yticklabels()] for ax in axes]
        seen["ylims"] = [ax.get_ylim() for ax in axes]
        plt.close(fig)

    monkeypatch.setattr(report, "_finish_curves", finish)
    x, flat = np.array([0.0, 1.0, 2.0]), (np.full(3, 14.2857), None)
    reqs = {stat: (np.array([3.0, 4.0, 5.0]) + off, None) for stat, off in (("min", 0.0), ("mean", 1.0), ("max", 2.0))}
    strips = [("frozen", x, flat, {}), ("frozen_bounded", x, flat, reqs), ("learnable", x, (np.array([1.0, 2.0, 3.0]), None), {})]

    report._render_strips(strips, tmp_path / "x.png", r"\alpha", "t", "std", 12, 8, 8, 1, 10, 1.8, 0.8)

    assert seen["yticks"][0] == ["14.29"]
    lo, hi = seen["ylims"][0]
    assert lo < 14.2857 < hi and (lo + hi) / 2 == pytest.approx(14.2857)  # the line centred on its tick
    for idx, lowest in ((1, 3.0), (2, 1.0)):  # autoscaled: several ticks, over everything the strip draws
        assert len(seen["yticks"][idx]) > 1
        assert seen["ylims"][idx][0] <= lowest
    assert seen["ylims"][1][1] >= 14.2857


def test_strip_blocks_group_by_the_coords_top_level_dimension() -> None:
    # combo-group names join with '_' and the campaign crosses the first group slowest, so the component
    # before the first '_' is the coarsest sweep the names carry: one block per LR here
    strips = [(coord,) for coord in ("LR-1e-5_Alpha-100", "LR-1e-5_Alpha-10", "LR-2e-5_Alpha-100")]

    assert report._strip_blocks(strips, ["ax0", "ax1", "ax2"]) == [["ax0", "ax1"], ["ax2"]]
    # one combo group (no '_' level) would box every strip on its own -- no boxes at all instead
    assert report._strip_blocks([("c0",), ("c1",)], ["ax0", "ax1"]) == []


def test_finish_curves_boxes_blocks_when_asked(tmp_path) -> None:
    # box_blocks closes the block's sides too: the double width runs down every panel of it, boxing the
    # run rather than just ruling it off (the strip figures' coord groups)
    fig, axs = plt.subplots(4, 1)
    axes = list(axs)

    report._finish_curves(fig, axes, [], {}, "t", tmp_path / "boxed.png", 8, 1, axes_blocks=[axes[1:3]],
                          box_blocks=True)

    widths = {ax: {side: spine.get_linewidth() for side, spine in ax.spines.items()} for ax in axes}
    assert widths[axes[1]] == {"left": 2, "right": 2, "top": 2, "bottom": 1}
    assert widths[axes[2]] == {"left": 2, "right": 2, "top": 1, "bottom": 2}
    assert set(widths[axes[0]].values()) == {1} and set(widths[axes[3]].values()) == {1}


def test_finish_curves_rules_off_panel_blocks(tmp_path) -> None:
    # a block (plot_alpha_curves' dL/dalpha aggs) is ruled at twice the panel border's width on the two
    # horizontals bounding it -- the top of its first panel, the bottom of its last -- and nowhere else:
    # its sides, the edges shared between its panels, and every panel outside it keep the panel width
    fig, axs = plt.subplots(5, 1)
    axes = list(axs)

    report._finish_curves(fig, axes, [], {}, "t", tmp_path / "blocks.png", 8, 1, axes_blocks=[axes[1:4]])

    widths = {ax: {side: spine.get_linewidth() for side, spine in ax.spines.items()} for ax in axes}
    assert set(widths[axes[0]].values()) == {1}  # panels outside a block are untouched
    assert set(widths[axes[4]].values()) == {1}
    assert widths[axes[1]] == {"left": 1, "right": 1, "top": 2, "bottom": 1}
    assert widths[axes[2]] == {"left": 1, "right": 1, "top": 1, "bottom": 1}
    assert widths[axes[3]] == {"left": 1, "right": 1, "top": 1, "bottom": 2}


def test_plot_convergence_draws_a_base_selection(tmp_path) -> None:
    # with the base eval a candidate the curves start at checkpoint 0 on a symlog axis, so a
    # selection there (row a) is drawable; without it the log axis starts at checkpoint 1
    curves = [(("a",), np.array([0.6, 0.4, 0.5]), 0), (("b",), np.array([0.2, 0.3, 0.1]), 1)]
    report._plot_convergence(curves, 0, "mAP", "with base", tmp_path / "with_base.png")
    report._plot_convergence(curves, 1, "mAP", "second row", tmp_path / "without_base.png")
    assert (tmp_path / "with_base.png").exists()
    assert (tmp_path / "without_base.png").exists()


def _write_chkpt_means(dpath_phase, dataset, arm, coord, means, idx_best) -> None:
    """The coord's across-trial mean curve on `dataset`, the same artifact update_chkpt_selection
    writes, for every criterion x group."""
    for criterion in ("map", "acc"):
        for group_key in _GROUP_KEYS:
            dpath_group = _dpath_coord(dpath_phase, dataset, arm, coord) / "coord_stats" / criterion / group_key
            dpath_group.mkdir(parents=True, exist_ok=True)
            save_pickle(
                {
                    "n_trials": 1,
                    "chkpts": np.arange(len(means)),
                    "means": np.array(means),
                    "spreads": np.zeros(len(means)),
                    "idx_best": idx_best,
                },
                dpath_group / "chkpt_means.pkl",
            )


def test_update_dataset_metrics_writes_pngs(tmp_path, monkeypatch) -> None:
    # the dataset's dataset_metrics/ tables + convergence plots, both kinds: arm_coords/ ('Arm' + 'Coord'
    # rows) and arms/ ('Arm' rows at each arm's best coord). "mp" is planned but has no completed trials
    # in this dataset -> no row in either; a dataset with no dir (planned "bryo", never launched) is skipped.
    dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _scores_grp(_comp(0.50)))
    _write_meta(tmp_path, ["hp", "mp"], ["c0"], ["cub", "bryo"])

    grids = []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)
    render = report._render_stats_table

    def _spy(grid, n_keys, title, fpath, bold_high, heatmap, killed_rows):
        grids.append((fpath, grid))
        render(grid, n_keys, title, fpath, bold_high, heatmap, killed_rows)

    monkeypatch.setattr(report, "_render_stats_table", _spy)

    report.update_dataset_metrics("cub", _EVAL_GROUPS, "std", True, False, True, _SUPP_OFF)
    report.update_dataset_metrics("bryo", _EVAL_GROUPS, "std", True, False, True, _SUPP_OFF)
    dpath_stats = tmp_path / "_datasets" / "cub" / "dataset_metrics"
    for kind in ("arm_coords", "arms"):
        for criterion in ("map", "acc"):
            for group_key in _GROUP_KEYS:
                assert (dpath_stats / kind / "performance" / criterion / group_key / "scores.png").exists()
                assert (dpath_stats / kind / "performance" / criterion / group_key / "convergence.png").exists()
    assert not (tmp_path / "_datasets" / "bryo").exists()
    assert _captured(grids, "arm_coords", "performance", "map", "native") == [
        ["Arm", "Coord", *_MAP_LABELS],
        ["hp", "c0 (1)", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00"],
    ]
    assert _captured(grids, "arms", "performance", "acc", "native") == [["Arm", "I2T"], ["hp (1)", "56.00"]]


def test_update_dataset_metrics_qual_phase_skips_arms(tmp_path, monkeypatch) -> None:
    # the qual matrix reduces each arm to its pick(s), so its dataset_metrics/arms/ would just
    # duplicate arm_coords/ -- only the _screen phase gets it (keyed off the phase dir's name)
    dpath_phase = tmp_path / "qual"
    dpath_selected = _dpath_coord(dpath_phase, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _scores_grp(_comp(0.50)))
    _write_meta(dpath_phase, ["hp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)

    report.update_dataset_metrics("cub", _EVAL_GROUPS, "std", False, False, False, _SUPP_OFF)
    dpath_stats = dpath_phase / "_datasets" / "cub" / "dataset_metrics"
    assert (dpath_stats / "arm_coords" / "performance" / "map" / "native" / "scores.png").exists()
    assert not (dpath_stats / "arms").exists()


def test_update_dataset_metrics_arms_use_best_coord_per_arm(tmp_path, monkeypatch) -> None:
    # arms/: each arm at its best coord for THIS dataset, per criterion x group -- on the map tables the
    # coord with the highest mean comp.map.all (a: c0 60 > c1 40; b: c1 50 > c0 30), on the acc tables the
    # highest comp.acc.i2t (a: c1 80 > c0 20; b: c0 70 > c1 10) -- ties to the first coord in campaign
    # order (c: both 0.50 -> c0). The arms convergence plot overlays those picks' curves; arm_coords/ keeps
    # every (arm, coord) as its own row, campaign order.
    vals = {  # (arm, coord) -> (map all, acc i2t)
        ("a", "c0"): (0.60, "0.20"), ("a", "c1"): (0.40, "0.80"),
        ("b", "c0"): (0.30, "0.70"), ("b", "c1"): (0.50, "0.10"),
        ("c", "c0"): (0.50, "0.50"), ("c", "c1"): (0.50, "0.50"),
    }
    for (arm, coord), (all_v, acc_v) in vals.items():
        dpath_selected = _dpath_coord(tmp_path, "cub", arm, coord) / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_full_comp(all_v, acc_v)))
    _write_meta(tmp_path, ["a", "b", "c"], ["c0", "c1"], ["cub"])

    grids, plotted = [], []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)
    monkeypatch.setattr(report, "_render_stats_table", lambda grid, n_keys, title, fpath, bold_high, heatmap, killed_rows: grids.append((fpath, grid)))
    monkeypatch.setattr(report, "_plot_convergence", lambda curves, idx_win, score_name, title, fpath: plotted.append((fpath, curves, idx_win)))

    report.update_dataset_metrics("cub", _EVAL_GROUPS, "std", False, False, False, _SUPP_OFF)
    assert [r[:2] for r in _captured(grids, "arms", "performance", "map", "native")] == [["Arm", "All"], ["a (1)", "60.00"], ["b (1)", "50.00"], ["c (1)", "50.00"]]
    assert _captured(grids, "arms", "performance", "acc", "native") == [["Arm", "I2T"], ["a (1)", "80.00"], ["b (1)", "70.00"], ["c (1)", "50.00"]]
    assert [r[:3] for r in _captured(grids, "arm_coords", "performance", "map", "native")] == [
        ["Arm", "Coord", "All"],
        ["a", "c0 (1)", "60.00"], ["a", "c1 (1)", "40.00"],
        ["b", "c0 (1)", "30.00"], ["b", "c1 (1)", "50.00"],
        ["c", "c0 (1)", "50.00"], ["c", "c1 (1)", "50.00"],
    ]
    # the arms convergence curves are the picked coords' (placeholder curves: [0, score], selected at 1)
    _, curves, idx_win = next(p for p in plotted if p[0].parent.parts[-4:] == ("arms", "performance", "map", "native"))
    assert [(row, float(means[idx_best])) for row, means, idx_best in curves] == [(("a",), 0.60), (("b",), 0.50), (("c",), 0.50)]
    assert curves[idx_win][0] == ("a",)
    _, curves, _ = next(p for p in plotted if p[0].parent.parts[-4:] == ("arms", "performance", "acc", "native"))
    assert [(row, float(means[idx_best])) for row, means, idx_best in curves] == [(("a",), 0.80), (("b",), 0.70), (("c",), 0.50)]
    _, curves, _ = next(p for p in plotted if p[0].parent.parts[-4:] == ("arm_coords", "performance", "map", "native"))
    assert [row for row, _, _ in curves] == [("a", "c0"), ("a", "c1"), ("b", "c0"), ("b", "c1"), ("c", "c0"), ("c", "c1")]


def test_pick_best_coords_by_native_map_all(tmp_path, monkeypatch) -> None:
    # the qual phase's selection: per (arm, dataset), the planned coord with the highest across-trial mean Native
    # mAP composite All -- criterion map, group native: a's c1 has the higher acc and the higher native_macro
    # mAP, but c0 wins on native comp.map.all (60 > 40); ties go to the first coord in campaign order (b: 50 =
    # 50 -> c0); a coord is a candidate only where it has completed trials (a on bryo: c1 alone); an arm with
    # none there (b on bryo) or anywhere (planned "c") gets no entry
    vals = {  # (arm, coord, dataset) -> (map all, acc i2t)
        ("a", "c0", "cub"): (0.60, "0.20"), ("a", "c1", "cub"): (0.40, "0.80"),
        ("b", "c0", "cub"): (0.50, "0.10"), ("b", "c1", "cub"): (0.50, "0.90"),
        ("a", "c1", "bryo"): (0.30, "0.10"),
    }
    for (arm, coord, dataset), (all_v, acc_v) in vals.items():
        dpath_selected = _dpath_coord(tmp_path, dataset, arm, coord) / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_full_comp(all_v, acc_v)), macro=_scores_grp(_full_comp(0.99 if coord == "c1" else 0.01)))
    _write_meta(tmp_path, ["a", "b", "c"], ["c0", "c1"], ["cub", "bryo"])
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    assert report.pick_best_coords() == {("a", "cub"): "c0", ("b", "cub"): "c0", ("a", "bryo"): "c1"}


def test_phase_matrix_drives_sweeps_and_rows(tmp_path, monkeypatch) -> None:
    # the phase's planned matrix -- not arms x coords -- is what the sweep gates and table rows key off: in a
    # qual-shaped tree each arm has one planned coord (sp: c0, mp: c1; "c2" is a campaign coord planned nowhere
    # here), so a seed's cycles close once THOSE have completed, and the tables carry only those rows
    matrix = {"cub": {"sp": ["c0"], "mp": ["c1"]}}
    _write_meta(tmp_path, ["sp", "mp"], ["c0", "c1", "c2"], ["cub"], matrix)
    scores = (("0.10", "0.10"), ("0.30", "0.30"))
    _write_trial_evals(_dpath_coord(tmp_path, "cub", "sp", "c0") / "_seeds" / "42", scores)
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    assert report.arm_sweep_complete(42, "cub", "sp")
    assert not report.dataset_sweep_complete(42, "cub")  # mp/c1 still to come
    _write_trial_evals(_dpath_coord(tmp_path, "cub", "mp", "c1") / "_seeds" / "42", scores)
    assert report.dataset_sweep_complete(42, "cub")
    assert report.seed_sweep_complete(42)

    for arm, coord, base in (("sp", "c0", 0.50), ("mp", "c1", 0.40)):
        _write_group_metrics(_dpath_coord(tmp_path, "cub", arm, coord) / "_seeds" / "42" / "evals" / "_selected", _scores_grp(_comp(base)))
    grids = []
    monkeypatch.setattr(report, "_render_stats_table", lambda grid, n_keys, title, fpath, bold_high, heatmap, killed_rows: grids.append((fpath, grid)))
    monkeypatch.setattr(report, "_plot_convergence", lambda *a: None)

    report.update_arm_metrics("cub", "sp", _EVAL_GROUPS, "std", False, False, False, _SUPP_OFF)
    report.update_dataset_metrics("cub", _EVAL_GROUPS, "std", False, False, False, _SUPP_OFF)
    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    assert [r[0] for r in _captured(grids, "performance", "map", "native")] == ["Coord", "c0 (1)"]
    assert [r[:2] for r in _captured(grids, "arm_coords", "performance", "map", "native")] == [["Arm", "Coord"], ["sp", "c0 (1)"], ["mp", "c1 (1)"]]
    assert [r[0] for r in _captured(grids, "arms", "performance", "map", "native")] == ["Arm", "sp (1)", "mp (1)"]
    grid = [[c.value for c in r] for r in load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx").active.iter_rows()]
    assert grid[4][:3] == ["sp", "c0 (1)", "50.00"] and grid[5][:3] == ["mp", "c1 (1)", "40.00"]
    grid = [[c.value for c in r] for r in load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx").active.iter_rows()]
    assert grid[4][:3] == ["sp", "c0 (1)", "50.00"] and grid[5][:3] == ["mp", "c1 (1)", "40.00"]


def test_update_dataset_metrics_ordered_localized_per_metric(tmp_path, monkeypatch) -> None:
    # ordered=True: each png orders its rows by its own metric's means over ITS dataset's trials
    # only. The bryo data makes cub's local orders the opposite of the cross-dataset ones: cub-local
    # mAP gives a=60 > b=40 -> [a, b] (cross-dataset means 35 vs 40 would say [b, a]), and cub-local
    # acc gives b=80 > a=20 -> [b, a] (cross-dataset means 55 vs 45 would say [a, b]). "c" completed
    # only in bryo -> no row in cub's pngs (blank rows are xlsx-only).
    comp_vals = {  # (arm, dataset) -> (map "all", acc "i2t")
        ("a", "cub"): (0.60, "0.20"), ("a", "bryo"): (0.10, "0.90"),
        ("b", "cub"): (0.40, "0.80"), ("b", "bryo"): (0.40, "0.10"),
        ("c", "bryo"): (0.90, "0.90"),
    }
    for (arm, dataset), (all_v, acc_v) in comp_vals.items():
        dpath_selected = _dpath_coord(tmp_path, dataset, arm, "c0") / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_full_comp(all_v, acc_v)))
    _write_meta(tmp_path, ["a", "b", "c"], ["c0"], ["cub", "bryo"])

    grids = []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)
    monkeypatch.setattr(report, "_render_stats_table", lambda grid, n_keys, title, fpath, bold_high, heatmap, killed_rows: grids.append((fpath, grid)))

    report.update_dataset_metrics("cub", _EVAL_GROUPS, "std", False, True, False, _SUPP_OFF)
    assert len(grids) == 16  # map + acc per eval group, both kinds
    grid_map = _captured(grids, "arm_coords", "performance", "map", "native")
    assert grid_map[0] == ["Arm", "Coord", *_MAP_LABELS]
    assert [r[:2] for r in grid_map[1:]] == [["a", "c0 (1)"], ["b", "c0 (1)"]]  # cub-local mAP order; no "c" row
    assert grid_map[1][2] == "60.00"
    assert grid_map[2][2] == "40.00"
    grid_acc = _captured(grids, "arm_coords", "performance", "acc", "native")
    assert grid_acc == [["Arm", "Coord", "I2T"], ["b", "c0 (1)", "80.00"], ["a", "c0 (1)", "20.00"]]  # cub-local acc order [b, a]
    # the arms tables (each arm at its only coord) order the same way
    assert [r[0] for r in _captured(grids, "arms", "performance", "map", "native")[1:]] == ["a (1)", "b (1)"]
    assert [r[0] for r in _captured(grids, "arms", "performance", "acc", "native")[1:]] == ["b (1)", "a (1)"]


def test_update_phase_metrics_qual_phase_skips_arms_workbooks(tmp_path, monkeypatch) -> None:
    # the qual matrix reduces each arm to its pick(s), so its arms/ workbooks would just duplicate
    # arm_coords/ -- only the _screen phase gets them (keyed off the phase dir's name)
    dpath_phase = tmp_path / "qual"
    for seed, base in (("42", 0.50), ("43", 0.60)):
        dpath_selected = _dpath_coord(dpath_phase, "cub", "hp", "c0") / "_seeds" / seed / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_comp(base)))
    _write_meta(dpath_phase, ["hp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", dpath_phase)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    assert (dpath_phase / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx").exists()
    assert not (dpath_phase / "phase_metrics" / "arms").exists()


def test_update_phase_metrics_writes_stacked_tables(tmp_path, monkeypatch) -> None:
    # (the arms workbook, keyed like the arm_coords one by 'Arm' + 'Coord' -- here the arm's best coord
    # in each dataset, '-' where it has no trials there and in the Mean table -- see
    # test_update_phase_metrics_arm_coords_layout for the (arm, coord)-keyed one)
    # two datasets -> two stacked tables sharing the same rows (in phase_metadata order). "hp" has 2
    # cub trials (mean ± spread) and none in bryo -> a blank "-" row in the Bryozoa table; "mp" has no
    # completed trials anywhere -> no rows at all until its first trial completes.
    for seed, base in (("42", 0.50), ("43", 0.60)):
        dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / seed / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_comp(base)))
    _write_meta(tmp_path, ["hp", "mp"], ["c0"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    fpath_xlsx = tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx"
    assert fpath_xlsx.exists()
    wb = load_workbook(fpath_xlsx)
    ws = wb.active

    def rows_as_lists():
        return [[c.value for c in row] for row in ws.iter_rows()]

    grid = rows_as_lists()
    # campaign banner ("<parent-dir> - <campaign> (<eval group name>)") + blank row, then the dataset tables, then the
    # always-on cross-dataset mean table (one row per arm) at the bottom. "mp" (no completed
    # trials anywhere) gets no rows; "hp" completed only in cub, so the Bryozoa table still gets its
    # blank "-" row. mean cells are point values (no spread), unlike the per-dataset "± spread".
    # (the banner names the campaign, the phase dir's parent -- here tmp_path stands in for the phase dir)
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; mAP-selection)"
    assert grid[1][:8] == [None] * 8  # blank row below the campaign banner
    assert grid[2][0] == "CUB"
    assert grid[3][:8] == ["Arm", "Coord", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[4][:3] == ["hp", "c0 (2)", "55.00 ± 7.07"]  # hp: 2 trials, at its cub pick
    assert grid[5][:8] == [None] * 8  # spacer row -- no "mp" row
    assert grid[6][0] == "Bryozoa"
    assert grid[7][:8] == ["Arm", "Coord", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[8][:3] == ["hp", "- (0)", "-"]  # hp's blank entries still added for the trial-less dataset: no pick there
    assert grid[10][0] == "Mean"  # cross-dataset mean table sits at the bottom
    assert grid[11][:2] == ["Arm", "Coord"]
    assert grid[12][:8] == ["hp", "-", "55.00", "57.00", "56.00", "58.00", "59.00", "60.00"]  # the pick is per dataset: '-'
    assert not any(v in ("mp", "mp (0)") for r in grid for v in r)
    # per-seed blocks to the right, one blank separator column apart; "seed <seed>" labels sit in the
    # campaign-banner row; seed blocks have no Mean table and their dataset tables sit in the same
    # rows as the aggregate block's (dataset tables lead everywhere), with plain key cells (no
    # counts), that seed's raw values, and "-" where the seed's trial hasn't completed
    assert grid[0][9] == "seed 42"
    assert grid[0][18] == "seed 43"
    assert grid[2][9] == "CUB"
    assert grid[3][9:17] == ["Arm", "Coord", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[4][9:17] == ["hp", "c0", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00"]   # seed 42 CUB, aligned with aggregate CUB
    assert grid[6][9] == "Bryozoa"
    assert grid[8][9:17] == ["hp", "-", "-", "-", "-", "-", "-", "-"]                           # seed 42 Bryozoa
    assert grid[4][18:26] == ["hp", "c0", "60.00", "62.00", "61.00", "63.00", "64.00", "65.00"]  # seed 43 CUB
    assert grid[10][9] is None  # seed blocks have no Mean table (bottom band stays blank)
    assert not any(r[9] == "Mean" for r in grid)  # no trial-mean table in seed blocks
    assert all(r[8] is None and r[17] is None for r in grid)  # separator columns stay empty
    # snug column widths: longest header/data cell + 2; empty separator columns get a small fixed width
    assert ws.column_dimensions["A"].width == len("Arm") + 2
    assert ws.column_dimensions["B"].width == len("c0 (2)") + 2
    assert ws.column_dimensions["C"].width == len("55.00 ± 7.07") + 2
    assert ws.column_dimensions["I"].width == 3
    # bold_high=False: data cells stay unbolded (only header row + key columns bold)
    assert ws.cell(row=5, column=3).font.bold is not True
    # heatmap=False: data cells are left unshaded
    assert ws.cell(row=5, column=3).fill.patternType is None
    # "All Borders": thin black gridlines on every table cell, incl. all cells of the merged title banner
    assert ws.cell(row=3, column=1).border.top.style == "thin"
    assert ws.cell(row=3, column=1).border.top.color.rgb[-6:] == "000000"
    assert ws.cell(row=3, column=8).border.right.color.rgb[-6:] == "000000"  # banner's far merged edge
    assert ws.cell(row=5, column=3).border.left.color.rgb[-6:] == "000000"  # data cell
    # campaign + table titles are left-aligned in their cells
    assert ws.cell(row=1, column=1).alignment.horizontal == "left"
    assert ws.cell(row=3, column=1).alignment.horizontal == "left"
    # key cells are left-aligned; the key headers and score cells stay centered
    assert ws.cell(row=5, column=1).alignment.horizontal == "left"  # "hp" (dataset table)
    assert ws.cell(row=5, column=2).alignment.horizontal == "left"  # "c0 (2)" (dataset table)
    assert ws.cell(row=13, column=1).alignment.horizontal == "left"  # "hp" (Mean table)
    assert ws.cell(row=4, column=1).alignment.horizontal == "center"  # "Arm" header
    assert ws.cell(row=4, column=2).alignment.horizontal == "center"  # "Coord" header
    assert ws.cell(row=5, column=3).alignment.horizontal == "center"  # score cell
    # 2nd sheet: the accuracy analog (single I2T column per table), same layout/row order.
    # hp's cub trials have acc i2t 56.00/66.00 -> mean 61.00 (± 7.07 in the per-dataset table).
    assert wb.sheetnames == ["Composite mAP", "Composite I2T Accuracy", "Hardware Performance"]
    agrid = [[c.value for c in row] for row in wb["Composite I2T Accuracy"].iter_rows()]
    assert agrid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; mAP-selection)"
    assert agrid[2][0] == "CUB"
    assert agrid[3][:3] == ["Arm", "Coord", "I2T"]
    assert agrid[4][:3] == ["hp", "c0 (2)", "61.00 ± 7.07"]
    assert agrid[6][0] == "Bryozoa"
    assert agrid[8][:3] == ["hp", "- (0)", "-"]
    assert agrid[10][0] == "Mean"
    assert agrid[12][:3] == ["hp", "-", "61.00"]
    # acc seed blocks (3-wide, so at cols E-G and I-K)
    assert agrid[0][4] == "seed 42"
    assert agrid[0][8] == "seed 43"
    assert agrid[2][4] == "CUB"
    assert agrid[4][4:7] == ["hp", "c0", "56.00"]   # seed 42 CUB, aligned with aggregate CUB
    assert agrid[4][8:11] == ["hp", "c0", "66.00"]  # seed 43 CUB
    assert agrid[8][4:7] == ["hp", "-", "-"]        # seed 42 Bryozoa
    # 3rd sheet: the hardware analog, same layout/row order with the hw readings as columns (the
    # Mean table appends the crash totals); values asserted in test_update_phase_metrics_hw_sheet
    hgrid = [[c.value for c in row] for row in wb["Hardware Performance"].iter_rows()]
    assert hgrid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; mAP-selection)"
    assert hgrid[2][0] == "CUB"
    assert hgrid[3][:7] == ["Arm", "Coord", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM"]
    assert hgrid[4][:2] == ["hp", "c0 (2)"]
    assert hgrid[6][0] == "Bryozoa"
    assert hgrid[8][:3] == ["hp", "- (0)", "-"]
    assert hgrid[10][0] == "Mean"
    assert hgrid[11][:10] == ["Arm", "Coord", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM",
                              "Total Crashes RAM", "Total Crashes VRAM", "Total Crashes Other"]
    assert hgrid[12][:2] == ["hp", "-"]
    assert hgrid[0][11] == "seed 42"  # one separator past the 10-wide Mean table (the block's widest)
    assert hgrid[4][11:13] == ["hp", "c0"]


def test_update_phase_metrics_arm_coords_layout(tmp_path, monkeypatch) -> None:
    # the arm_coords workbooks key every table by two columns, 'Arm' + 'Coord' (the trial count on the
    # coord cell), one row per (arm, coord) with a completed trial somewhere, in campaign order (arms,
    # then coords within each); every block sits one column further right accordingly, and the mAP
    # sheet's banner title spans both key columns ahead of the group headers
    for (arm, coord), base in ((("hp", "c0"), 0.50), (("hp", "c1"), 0.60), (("mp", "c1"), 0.40)):
        dpath_selected = _dpath_coord(tmp_path, "cub", arm, coord) / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_comp(base)))
    _write_meta(tmp_path, ["hp", "mp"], ["c0", "c1"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_PRIM, False)

    wb = load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx")
    ws = wb.active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    merged = {str(m) for m in ws.merged_cells.ranges}
    # 14-wide tables: Arm + Coord + 6 comp + 6 prim
    assert grid[2][:3] == ["CUB", None, "Composite Scores"]
    assert grid[2][8] == "Primitive Scores"
    assert "A3:B3" in merged and "C3:H3" in merged and "I3:N3" in merged
    assert grid[3][:14] == ["Arm", "Coord", *_PRIM_MAP_LABELS]
    assert grid[4][:4] == ["hp", "c0 (1)", "50.00", "52.00"]
    assert grid[5][:3] == ["hp", "c1 (1)", "60.00"]
    assert grid[6][:3] == ["mp", "c1 (1)", "40.00"]  # (mp, c0) has no trials anywhere -> no row
    assert grid[7][:14] == [None] * 14
    assert grid[8][0] == "Bryozoa"
    assert grid[10][:3] == ["hp", "c0 (0)", "-"]
    assert grid[14][0] == "Mean"
    assert grid[15][:2] == ["Arm", "Coord"]
    assert grid[16][:3] == ["hp", "c0", "50.00"]
    assert grid[18][:3] == ["mp", "c1", "40.00"]
    # seed block: one separator past the 14-wide aggregate, its rows aligned with the aggregate's
    assert grid[0][15] == "seed 42"
    assert grid[3][15:17] == ["Arm", "Coord"]
    assert grid[4][15:18] == ["hp", "c0", "50.00"]
    assert grid[6][15:18] == ["mp", "c1", "40.00"]
    assert all(r[14] is None for r in grid)
    # both key cells styled as row labels: left-aligned, bold, header grey; score cells not
    assert ws.cell(row=5, column=1).alignment.horizontal == "left" and ws.cell(row=5, column=2).alignment.horizontal == "left"
    assert ws.cell(row=5, column=2).font.bold is True and ws.cell(row=5, column=2).fill.fgColor.rgb[-6:] == "EAEAEA"
    assert ws.cell(row=5, column=3).font.bold is not True and ws.cell(row=5, column=3).alignment.horizontal == "center"
    # the accuracy sheet keeps its full-width merged title banner over the 5-wide tables
    ws_acc = wb["Composite I2T Accuracy"]
    agrid = [[c.value for c in r] for r in ws_acc.iter_rows()]
    assert "A3:E3" in {str(m) for m in ws_acc.merged_cells.ranges}
    assert agrid[3][:5] == ["Arm", "Coord", "I2T", "ID I2T", "OOD I2T"]
    assert agrid[4][:3] == ["hp", "c0 (1)", "56.00"]
    assert agrid[0][6] == "seed 42"
    # the hardware sheet: 7-wide dataset tables, 10-wide Mean -> seed block one separator past col 10
    hgrid = [[c.value for c in r] for r in wb["Hardware Performance"].iter_rows()]
    assert hgrid[3][:3] == ["Arm", "Coord", "Time Trial"]
    assert hgrid[4][:2] == ["hp", "c0 (1)"]
    assert hgrid[15][:10] == ["Arm", "Coord", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM",
                              "Total Crashes RAM", "Total Crashes VRAM", "Total Crashes Other"]
    assert hgrid[16][:2] == ["hp", "c0"]
    assert hgrid[0][11] == "seed 42"


def test_update_phase_metrics_bold_high(tmp_path, monkeypatch) -> None:
    # bold_high=True: the highest-mean cell in each score column is bolded. "hp" (base 0.60)
    # outranks "mp" (base 0.50) in every column, so hp's cells bold and mp's do not; "sp" completed
    # only in bryo, so its CUB row is blank "-" -- ignored and never bolded.
    for arm, dataset, base in (("hp", "cub", 0.60), ("mp", "cub", 0.50), ("sp", "bryo", 0.10)):
        dpath_selected = _dpath_coord(tmp_path, dataset, arm, "c0") / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_comp(base)))
    _write_meta(tmp_path, ["hp", "mp", "sp"], ["c0"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", True, False, False, _SUPP_OFF, False)

    ws = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx").active
    # campaign banner + blank row, then the CUB table first: banner row 3, header row 4, arm rows
    # 5/6/7 = hp/mp/sp; score cols C..H = All/ID/OOD/I2T/I2I/T2I (the Mean table sits at the bottom)
    assert ws.cell(row=3, column=1).value == "CUB"
    assert ws.cell(row=4, column=1).value == "Arm"
    assert ws.cell(row=4, column=2).value == "Coord"
    for score_col in range(3, 9):
        assert ws.cell(row=5, column=score_col).font.bold is True       # hp wins -> bold
        assert ws.cell(row=6, column=score_col).font.bold is not True   # mp loses -> not bold
        assert ws.cell(row=7, column=score_col).value == "-"            # sp: no cub trials
        assert ws.cell(row=7, column=score_col).font.bold is not True   # "-" never bolds
    # same in the arm_coords workbook, keyed the same way
    ws = load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx").active
    assert ws.cell(row=4, column=2).value == "Coord"
    for score_col in range(3, 9):
        assert ws.cell(row=5, column=score_col).font.bold is True
        assert ws.cell(row=6, column=score_col).font.bold is not True


def test_update_phase_metrics_per_group_files(tmp_path, monkeypatch) -> None:
    # one workbook per eval group under phase_metrics/{arm_coords,arms}/performance/<criterion>/<group>/, each
    # reading its own comp map: native_macro/'s <- the best-checkpoint native_macro.json, not the (different)
    # standard values
    dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(
        dpath_selected,
        _scores_grp(_comp(0.50)),        # standard All -> 50.00
        macro=_scores_grp(_comp(0.30)),  # macro All -> 30.00
    )
    _write_meta(tmp_path, ["hp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    dpath_stats = tmp_path / "phase_metrics"
    for kind in ("arm_coords", "arms"):
        for criterion in ("map", "acc"):
            assert sorted(p.parent.name for p in (dpath_stats / kind / "performance" / criterion).glob("*/metrics.xlsx")) == [
                "joint", "joint_macro", "native", "native_macro"
            ]
    ws = load_workbook(dpath_stats / "arms" / "performance" / "map" / "native" / "metrics.xlsx").active
    assert ws.cell(row=4, column=3).value == "All"
    assert [ws.cell(row=5, column=c).value for c in (1, 2, 3)] == ["hp", "c0 (1)", "50.00"]
    ws_macro = load_workbook(dpath_stats / "arms" / "performance" / "map" / "native_macro" / "metrics.xlsx").active
    assert ws_macro.cell(row=5, column=3).value == "30.00"  # macro, not standard's 50.00
    ws_ac = load_workbook(dpath_stats / "arm_coords" / "performance" / "map" / "native_macro" / "metrics.xlsx").active
    assert [ws_ac.cell(row=5, column=c).value for c in (1, 2, 3)] == ["hp", "c0 (1)", "30.00"]


def test_update_phase_metrics_criterion_sourcing(tmp_path, monkeypatch) -> None:
    # one workbook set per selection criterion: BOTH sheets of <kind>/<criterion>/ source
    # that criterion's best checkpoints -- the map/ workbook's accuracy sheet holds the acc scores
    # AT the best-mAP checkpoint (not the best-acc ones), and vice versa; banners name the selection
    dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(
        dpath_selected,
        _scores_grp(_comp(0.50)),               # map-best checkpoint: mAP All 50.00, acc I2T 56.00
        acc_selected=_scores_grp(_comp(0.30)),  # acc-best checkpoint: mAP All 30.00, acc I2T 36.00
    )
    _write_meta(tmp_path, ["hp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    wb_map = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")
    grid = [[c.value for c in r] for r in wb_map.active.iter_rows()]
    agrid = [[c.value for c in r] for r in wb_map["Composite I2T Accuracy"].iter_rows()]
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; mAP-selection)"
    assert grid[4][:3] == ["hp", "c0 (1)", "50.00"]   # mAP at the map-best checkpoint
    assert agrid[4][:3] == ["hp", "c0 (1)", "56.00"]  # acc at the map-best checkpoint

    wb_acc = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "acc" / "native" / "metrics.xlsx")
    grid = [[c.value for c in r] for r in wb_acc.active.iter_rows()]
    agrid = [[c.value for c in r] for r in wb_acc["Composite I2T Accuracy"].iter_rows()]
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; Acc-selection)"
    assert grid[4][:3] == ["hp", "c0 (1)", "30.00"]   # mAP at the acc-best checkpoint
    assert agrid[4][:3] == ["hp", "c0 (1)", "36.00"]  # acc at the acc-best checkpoint


def _full_comp(all_v: float, acc_v: str = "0.10") -> dict:
    # comp with controllable map "all" + acc "i2t"; other leaves fixed (irrelevant to ordering / heatmap rows)
    return {
        "acc": {"i2t": acc_v},
        "map": {"all": f"{all_v:.4f}", "id": "0.10", "ood": "0.10", "i2t": "0.10", "i2i": "0.10", "t2i": "0.10"},
    }


def test_update_phase_metrics_ordered_per_sheet_metric(tmp_path, monkeypatch) -> None:
    # ordered=True: each sheet orders its rows by its own metric's Mean-table first column, so the
    # two sheets may disagree. Campaign order is [a, b]; mAP mean-All gives a=mean(20,40)=30.00 <
    # b=mean(40,40)=40.00 -> mAP sheet flips to [b, a], while acc mean-I2T gives a=80.00 > b=20.00
    # -> accuracy sheet keeps [a, b].
    for arm, acc_v, cub_all, bryo_all in (("a", "0.80", 0.20, 0.40), ("b", "0.20", 0.40, 0.40)):
        for dataset, all_v in (("cub", cub_all), ("bryo", bryo_all)):
            dpath_selected = _dpath_coord(tmp_path, dataset, arm, "c0") / "_seeds" / "42" / "evals" / "_selected"
            dpath_selected.mkdir(parents=True)
            _write_group_metrics(dpath_selected, _scores_grp(_full_comp(all_v, acc_v)))
    _write_meta(tmp_path, ["a", "b"], ["c0"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, True, False, _SUPP_OFF, False)

    wb = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")
    grid = [[c.value for c in r] for r in wb.active.iter_rows()]
    # campaign banner + blank row, then the dataset tables (Mean at the bottom); rows ordered
    # by the mean table's "All" column -> b before a
    assert grid[2][0] == "CUB"
    assert grid[3][:8] == ["Arm", "Coord", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[4][:3] == ["b", "c0 (1)", "40.00"]
    assert grid[5][:3] == ["a", "c0 (1)", "20.00"]
    assert grid[7][0] == "Bryozoa"
    assert grid[9][:3] == ["b", "c0 (1)", "40.00"]
    assert grid[10][:3] == ["a", "c0 (1)", "40.00"]
    assert grid[12][0] == "Mean"
    assert grid[14][:3] == ["b", "-", "40.00"]
    assert grid[15][:3] == ["a", "-", "30.00"]
    # the seed block's rows are pinned to the sheet's mean-derived order too (CUB table, aligned rows)
    assert grid[0][9] == "seed 42"
    assert [grid[4][9], grid[5][9]] == ["b", "a"]
    # the accuracy sheet orders by its own acc mean-'I2T' column -> [a, b], unlike the mAP sheet
    agrid = [[c.value for c in r] for r in wb["Composite I2T Accuracy"].iter_rows()]
    assert agrid[3][:3] == ["Arm", "Coord", "I2T"]
    assert agrid[4][:3] == ["a", "c0 (1)", "80.00"]
    assert agrid[5][:3] == ["b", "c0 (1)", "20.00"]
    assert agrid[12][0] == "Mean"
    assert agrid[14][:3] == ["a", "-", "80.00"]
    assert agrid[15][:3] == ["b", "-", "20.00"]
    assert agrid[4][4:7] == ["a", "c0", "80.00"]  # acc seed block keeps the acc sheet's [a, b] order (aligned rows)
    # the arm_coords workbook orders its (arm, coord) rows the same way
    wb_ac = load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx")
    grid = [[c.value for c in r] for r in wb_ac.active.iter_rows()]
    assert grid[4][:3] == ["b", "c0 (1)", "40.00"]
    assert grid[5][:3] == ["a", "c0 (1)", "20.00"]


def _fill_rgb(ws, row, col):
    # last 6 hex chars (RGB) of a cell's fill, or None when the cell is unshaded
    fill = ws.cell(row=row, column=col).fill
    return None if fill.patternType is None else fill.fgColor.rgb[-6:]


def test_update_phase_metrics_heatmap(tmp_path, monkeypatch) -> None:
    # heatmap=True: value/100 maps to white->#4caf50 over a fixed range, regardless of the column's
    # other cells; 20/50/80 -> #dbefdc / #a6d7a8 / #70bf73. One dataset, so the Mean "All" column
    # mirrors the values and is shaded too; "d" (no completed trials anywhere) gets no row at all.
    for arm, all_v in (("a", 0.20), ("b", 0.50), ("c", 0.80)):
        dpath_selected = _dpath_coord(tmp_path, "cub", arm, "c0") / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_full_comp(all_v)))
    _write_meta(tmp_path, ["a", "b", "c", "d"], ["c0"], ["cub"])  # "d" has no trials

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, True, _SUPP_OFF, False)

    ws = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx").active
    # campaign banner + blank row; CUB table first: banner row 3, header row 4, "All" column is col C,
    # arm rows 5/6/7 = a/b/c
    assert ws.cell(row=8, column=1).value is None  # spacer right after c -> no "d" row
    assert _fill_rgb(ws, 5, 3) == "DBEFDC"  # 20 -> t=0.20
    assert _fill_rgb(ws, 6, 3) == "A6D7A8"  # 50 -> t=0.50
    assert _fill_rgb(ws, 7, 3) == "70BF73"  # 80 -> t=0.80
    # the trailing Mean table is shaded too (arm rows 11/12/13)
    assert _fill_rgb(ws, 11, 3) == "DBEFDC"
    assert _fill_rgb(ws, 13, 3) == "70BF73"
    # key cells never shade (both workbooks' coord column included)
    assert _fill_rgb(ws, 5, 2) == "EAEAEA"
    ws = load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx").active
    assert _fill_rgb(ws, 5, 2) == "EAEAEA" and _fill_rgb(ws, 5, 3) == "DBEFDC"


def test_update_phase_metrics_shades_killed_rows_yellow(tmp_path, monkeypatch) -> None:
    # a row aggregating a killed trial (kill_thresh) swaps the heatmap ramp for the killed yellow in
    # every score cell -- the dataset table, the Mean table, and the killed seed's block (the seed block
    # of a surviving sibling keeps the ramp) -- and its key cell counts the kill; a row without one is
    # untouched
    for seed, killed in (("42", False), ("43", True)):
        dpath_selected = _dpath_coord(tmp_path, "cub", "a", "c0") / "_seeds" / seed / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_full_comp(0.20)), killed=killed)
    dpath_selected = _dpath_coord(tmp_path, "cub", "b", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _scores_grp(_full_comp(0.80)))
    _write_meta(tmp_path, ["a", "b"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, True, _SUPP_OFF, False)

    ws = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx").active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    # CUB table: banner row 3, header row 4, arm rows 5 (a) / 6 (b); Mean table rows 10 (a) / 11 (b)
    assert grid[4][:3] == ["a", "c0 (2, 1 killed)", "20.00 ± 0.00"]
    assert grid[5][:3] == ["b", "c0 (1)", "80.00"]
    assert _fill_rgb(ws, 5, 3) == report._KILLED_HEX and _fill_rgb(ws, 5, 8) == report._KILLED_HEX
    assert _fill_rgb(ws, 6, 3) == "70BF73"
    assert _fill_rgb(ws, 10, 3) == report._KILLED_HEX and _fill_rgb(ws, 11, 3) == "70BF73"
    # seed blocks (one blank separator column past the 8-column aggregate block): seed 42 first
    assert grid[2][9] == "CUB" and grid[4][9:12] == ["a", "c0", "20.00"]
    assert _fill_rgb(ws, 5, 12) == "DBEFDC"  # seed 42's a survived: the ramp
    assert _fill_rgb(ws, 5, 21) == report._KILLED_HEX  # seed 43's a was killed
    assert _fill_rgb(ws, 6, 21) is None  # seed 43 never ran b: '-' stays unshaded


def test_update_phase_metrics_hw_sheet_x_marks_killed_rows(tmp_path, monkeypatch) -> None:
    # the 'Hardware Performance' sheet's counterpart of the score sheets' yellow rows: a row holding a
    # killed trial (kill_thresh) reads 'X' across its readings -- the dataset table, the Mean table
    # (its crash totals still counted) and the killed seed's block, while the surviving sibling seed's
    # block keeps its readings; a row without a kill is untouched
    for seed, killed in (("42", False), ("43", True)):
        dpath_selected = _dpath_coord(tmp_path, "cub", "a", "c0") / "_seeds" / seed / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_full_comp(0.20)), killed=killed)
    dpath_selected = _dpath_coord(tmp_path, "cub", "b", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _scores_grp(_full_comp(0.80)))
    (_dpath_coord(tmp_path, "cub", "a", "c0") / "coord_metadata.json").write_text(json.dumps({"n_crashes": {"ram": 2, "vram": 0, "other": 1}}))
    _write_meta(tmp_path, ["a", "b"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, True, _SUPP_OFF, False)

    ws = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")["Hardware Performance"]
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    # CUB table (7 wide): header row 3, arm rows 4 (a) / 5 (b); Mean table (10 wide): rows 9 (a) / 10 (b)
    assert grid[4][:7] == ["a", "c0 (2)", "X", "X", "X", "X", "X"]
    assert grid[5][:7] == ["b", "c0 (1)", "1", "1", "1", "1", "1"]
    assert grid[9][:10] == ["a", "-", "X", "X", "X", "X", "X", "2", "0", "1"]
    assert grid[10][:10] == ["b", "-", "1", "1", "1", "1", "1", "0", "0", "0"]
    # seed blocks past the 10-wide aggregate block: seed 42 at col 11, seed 43 at col 19
    assert grid[0][11] == "seed 42" and grid[4][11:18] == ["a", "c0", "1", "1", "1", "1", "1"]
    assert grid[0][19] == "seed 43" and grid[4][19:26] == ["a", "c0", "X", "X", "X", "X", "X"]
    assert grid[5][19:26] == ["b", "c0", "-", "-", "-", "-", "-"]  # seed 43 never ran b (key cell: the arm's best coord)
    assert ws.cell(row=5, column=3).fill.patternType is None  # hardware cells stay unstyled


def test_render_stats_table_shades_killed_rows_yellow(tmp_path) -> None:
    # the png renderer's killed rows take the same yellow (a smoke render with the real styling path)
    grid = [["Coord", "All"], ["c0 (2, 1 killed)", "55.00 ± 7.07"], ["c1 (1)", "40.00"]]
    report._render_stats_table(grid, 1, "mAP", tmp_path / "metrics.png", True, True, {1})
    assert (tmp_path / "metrics.png").exists()


def _prim_scores_grp() -> dict:
    # _scores_grp with distinct per-partition primitive values (comp base 0.50)
    grp = _scores_grp(_comp(0.50))
    grp["id"] = {"map": {"i2t": "0.61", "i2i": "0.62", "t2i": "0.63"}, "acc": {"i2t": "0.64"}}
    grp["ood"] = {"map": {"i2t": "0.71", "i2i": "0.72", "t2i": "0.73"}, "acc": {"i2t": "0.74"}}
    return grp


def test_update_phase_metrics_supp_primitive(tmp_path, monkeypatch) -> None:
    # supp_scores.primitive appends the per-partition primitive score columns: ID/OOD x I2T/I2I/T2I
    # on the mAP sheet, ID I2T / OOD I2T on the accuracy sheet -- in every table, incl. the seed blocks
    dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _prim_scores_grp())
    _write_meta(tmp_path, ["hp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_PRIM, False)

    wb = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")
    ws = wb.active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    # mAP banners split: title merged over the two key columns + grey merged 'Composite Scores'/'Primitive
    # Scores' group headers
    merged = {str(m) for m in ws.merged_cells.ranges}
    assert grid[2][:3] == ["CUB", None, "Composite Scores"]
    assert grid[2][8] == "Primitive Scores"
    assert grid[6][:3] == ["Mean", None, "Composite Scores"]
    assert grid[6][8] == "Primitive Scores"
    assert "A3:B3" in merged and "C3:H3" in merged and "I3:N3" in merged
    assert ws.cell(row=3, column=3).fill.fgColor.rgb[-6:] == "EAEAEA"  # group headers get the header grey
    assert ws.cell(row=3, column=1).fill.patternType is None           # title cell stays unfilled
    assert grid[3][:14] == ["Arm", "Coord", *_PRIM_MAP_LABELS]
    assert grid[4][:14] == ["hp", "c0 (1)", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00",
                            "61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]  # CUB row
    assert grid[8][8:14] == ["61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]  # Mean row (bottom)
    # seed block starts after the 14-wide aggregate + separator; its CUB table row-aligns with the aggregate's
    assert grid[0][15] == "seed 42"
    assert grid[2][15:18] == ["CUB", None, "Composite Scores"]
    assert grid[2][23] == "Primitive Scores"
    assert grid[3][15:29] == ["Arm", "Coord", *_PRIM_MAP_LABELS]
    assert grid[4][23:29] == ["61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]
    # the accuracy sheet keeps full-width merged title banners (no group headers)
    ws_acc = wb["Composite I2T Accuracy"]
    agrid = [[c.value for c in r] for r in ws_acc.iter_rows()]
    assert agrid[2][:2] == ["CUB", None]
    assert "A3:E3" in {str(m) for m in ws_acc.merged_cells.ranges}
    assert agrid[3][:5] == ["Arm", "Coord", "I2T", "ID I2T", "OOD I2T"]
    assert agrid[4][:5] == ["hp", "c0 (1)", "56.00", "64.00", "74.00"]  # CUB row
    assert agrid[8][:5] == ["hp", "-", "56.00", "64.00", "74.00"]  # Mean row (bottom)
    assert agrid[0][6] == "seed 42"


def test_update_arm_metrics_supp_primitive(tmp_path, monkeypatch) -> None:
    # supp_scores.primitive appends the per-partition primitive score columns to the png grids too
    dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _prim_scores_grp())
    _write_meta(tmp_path, ["hp"], ["c0"], ["cub"])

    grids = []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)
    monkeypatch.setattr(report, "_render_stats_table", lambda grid, n_keys, title, fpath, bold_high, heatmap, killed_rows: grids.append((fpath, grid)))

    report.update_arm_metrics("cub", "hp", _EVAL_GROUPS, "std", False, False, False, _SUPP_PRIM)
    assert len(grids) == 8  # map + acc per eval group
    grid_map = _captured(grids, "map", "native")
    assert grid_map[0] == ["Coord", *_PRIM_MAP_LABELS]
    assert grid_map[1] == ["c0 (1)", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00",
                           "61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]
    grid_acc = _captured(grids, "acc", "native")
    assert grid_acc[0] == ["Coord", "I2T", "ID I2T", "OOD I2T"]
    assert grid_acc[1] == ["c0 (1)", "56.00", "64.00", "74.00"]


def _nshot_scores_grp(nshot_map: dict, nshot_acc: dict) -> dict:
    # _prim_scores_grp + the ID partition's n-shot bucket scores, as the eval writes them: only
    # buckets with classes in the eval partition are present
    grp = _prim_scores_grp()
    grp["id"]["map"]["n-shot"] = nshot_map
    grp["id"]["acc"]["n-shot"] = nshot_acc
    return grp


_NSHOT_FULL = ({"few-shot": "0.31", "med-shot": "0.32", "many-shot": "0.33"},
               {"few-shot": "0.41", "med-shot": "0.42", "many-shot": "0.43"})
_NSHOT_NO_FEW = ({"med-shot": "0.52", "many-shot": "0.53"}, {"med-shot": "0.62", "many-shot": "0.63"})


def test_update_phase_metrics_supp_n_shot(tmp_path, monkeypatch) -> None:
    # supp_scores.n_shot appends one column per ID-partition n-shot bucket to the right of every
    # table (after the primitive columns when both are on): the bucket's composite mAP on the mAP
    # sheet (under an 'N-Shot Scores' group header), its I2T accuracy on the accuracy sheet. Bucket
    # names come from the eval files, merged in their order: bryo (first in campaign order) lacks
    # few-shot (no classes there, as on the dev split), cub has all three -> [few, med, many]; the
    # absent bucket renders "-" in bryo's rows and is left out of the Mean (cub's value alone).
    for dataset, nshot in (("bryo", _NSHOT_NO_FEW), ("cub", _NSHOT_FULL)):
        dpath_selected = _dpath_coord(tmp_path, dataset, "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _nshot_scores_grp(*nshot))
    _write_meta(tmp_path, ["hp"], ["c0"], ["bryo", "cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, {"primitive": True, "n_shot": True}, False)

    wb = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")
    ws = wb.active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    merged = {str(m) for m in ws.merged_cells.ranges}
    # 17-wide tables: Arm + Coord + 6 comp + 6 prim + 3 n-shot; three group headers over the banner
    assert grid[2][:3] == ["Bryozoa", None, "Composite Scores"]
    assert grid[2][8] == "Primitive Scores"
    assert grid[2][14] == "N-Shot Scores"
    assert "C3:H3" in merged and "I3:N3" in merged and "O3:Q3" in merged
    assert grid[3][:17] == ["Arm", "Coord", *_PRIM_MAP_LABELS, "few-shot", "med-shot", "many-shot"]
    assert grid[4][14:17] == ["-", "52.00", "53.00"]  # Bryozoa: no few-shot bucket
    assert grid[6][0] == "CUB"
    assert grid[8][14:17] == ["31.00", "32.00", "33.00"]
    assert grid[10][0] == "Mean"
    assert grid[11][14:17] == ["few-shot", "med-shot", "many-shot"]
    assert grid[12][14:17] == ["31.00", "42.00", "43.00"]  # few-shot: cub alone; others mean bryo/cub
    # seed block (one separator past the 17-wide aggregate) carries the columns too
    assert grid[0][18] == "seed 42"
    assert grid[3][32:35] == ["few-shot", "med-shot", "many-shot"]
    assert grid[4][32:35] == ["-", "52.00", "53.00"]
    assert grid[8][32:35] == ["31.00", "32.00", "33.00"]
    # accuracy sheet: the buckets' I2T accuracies, full-width banner as before
    ws_acc = wb["Composite I2T Accuracy"]
    agrid = [[c.value for c in r] for r in ws_acc.iter_rows()]
    assert "A3:H3" in {str(m) for m in ws_acc.merged_cells.ranges}
    assert agrid[3][:8] == ["Arm", "Coord", "I2T", "ID I2T", "OOD I2T", "few-shot", "med-shot", "many-shot"]
    assert agrid[4][5:8] == ["-", "62.00", "63.00"]
    assert agrid[8][5:8] == ["41.00", "42.00", "43.00"]
    assert agrid[12][5:8] == ["41.00", "52.00", "53.00"]

    # n_shot alone: the bucket columns follow the composite ones directly, with just the two groups
    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, {"primitive": False, "n_shot": True}, False)

    ws = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx").active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    assert grid[2][:3] == ["Bryozoa", None, "Composite Scores"]
    assert grid[2][8] == "N-Shot Scores"
    assert "I3:K3" in {str(m) for m in ws.merged_cells.ranges}
    assert grid[3][:11] == ["Arm", "Coord", *_MAP_LABELS, "few-shot", "med-shot", "many-shot"]
    assert grid[8][8:11] == ["31.00", "32.00", "33.00"]


def test_update_arm_metrics_supp_n_shot(tmp_path, monkeypatch) -> None:
    # supp_scores.n_shot appends the bucket columns to the png grids too
    dpath_selected = _dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42" / "evals" / "_selected"
    dpath_selected.mkdir(parents=True)
    _write_group_metrics(dpath_selected, _nshot_scores_grp(*_NSHOT_FULL))
    _write_meta(tmp_path, ["hp"], ["c0"], ["cub"])

    grids = []
    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)
    monkeypatch.setattr(report, "_render_stats_table", lambda grid, n_keys, title, fpath, bold_high, heatmap, killed_rows: grids.append((fpath, grid)))

    report.update_arm_metrics("cub", "hp", _EVAL_GROUPS, "std", False, False, False, {"primitive": False, "n_shot": True})
    grid_map = _captured(grids, "map", "native")
    assert grid_map[0] == ["Coord", *_MAP_LABELS, "few-shot", "med-shot", "many-shot"]
    assert grid_map[1][7:] == ["31.00", "32.00", "33.00"]
    grid_acc = _captured(grids, "acc", "native")
    assert grid_acc[0] == ["Coord", "I2T", "few-shot", "med-shot", "many-shot"]
    assert grid_acc[1][2:] == ["41.00", "42.00", "43.00"]


def test_update_phase_metrics_overrides_bands(tmp_path, monkeypatch) -> None:
    # overrides=True renders the config bands in left column bands the score blocks shift right past,
    # aligned with the aggregate block's bottom Mean table (whose key columns label their rows): "Arm
    # Overrides" with one column per param declared in ablation_arms (union of the rows' overrides.json
    # 'arm' keys, first-seen order) and, in the arm_coords workbooks, "Coord Overrides" likewise for
    # hpo_coords. Values resolve from each row's config.json -- "-" when the param is absent there
    # (inert under that config: mp has loss.blend.lambda 0.0, so clean_metadata dropped its loss2 subtree).
    # loss.loss1.targ resolves to "mp" for EVERY row, so its column is omitted (uniform columns
    # differentiate nothing). Config cells get no winner-bold/heatmap styling despite
    # bold_high/heatmap on. The arms workbooks get the arm band only: an arm's coord is picked per
    # dataset, so it has no single coord config to show.
    arms = {
        "hp": ({"loss.blend.lambda": 0.3, "loss.loss2.targ": "phylo"}, {"loss": {"blend": {"lambda": 0.3}, "loss1": {"targ": "mp"}, "loss2": {"targ": "phylo"}}}),
        "mp": ({"loss.loss1.targ": "mp"}, {"loss": {"loss1": {"targ": "mp"}}}),
    }
    coords = {"lo": 1.0e-5, "hi": 1.0e-4}
    base = 0.50
    for arm, (arm_ov, meta) in arms.items():
        for coord, lr in coords.items():
            dpath_coord = _dpath_coord(tmp_path, "cub", arm, coord)
            dpath_selected = dpath_coord / "_seeds" / "42" / "evals" / "_selected"
            dpath_selected.mkdir(parents=True)
            _write_group_metrics(dpath_selected, _scores_grp(_comp(base)))
            base -= 0.05  # (hp, lo) best for hp, (mp, lo) best for mp
            (dpath_coord / "overrides.json").write_text(json.dumps({"arm": arm_ov, "coord": {"lr.init": lr}}))
            (dpath_coord / "config.json").write_text(json.dumps({**meta, "lr": {"init": lr}}))
    _write_meta(tmp_path, ["hp", "mp"], ["lo", "hi"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", True, False, True, _SUPP_OFF, True)

    wb = load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx")
    ws = wb.active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    merged = {str(m) for m in ws.merged_cells.ranges}
    # bands: Arm Overrides (2 params) at A..B + separator C, Coord Overrides (1 param) at D + separator E;
    # the score blocks start at F. 4 rows (hp/lo, hp/hi, mp/lo, mp/hi): CUB table rows 5..8, Mean banner row 10
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; mAP-selection)"  # campaign banner stays top-left
    assert grid[2][5] == "CUB"
    assert grid[3][5:7] == ["Arm", "Coord"]
    assert grid[9][5] == "Mean"
    assert grid[9][0] == "Arm Overrides" and grid[9][3] == "Coord Overrides"
    assert "A10:B10" in merged and "D10:D10" not in merged  # a 1-wide band title is not merged
    assert grid[10][:2] == ["loss.blend.lambda", "loss.loss2.targ"]
    assert grid[10][3] == "lr.init"
    assert not any(v == "loss.loss1.targ" for r in grid for v in r)
    assert grid[10][5:7] == ["Arm", "Coord"]  # Mean header shares the row
    assert grid[11][:2] == ["0.3", "phylo"] and grid[11][3] == "1e-05" and grid[11][5:7] == ["hp", "lo"]
    assert grid[12][:2] == ["0.3", "phylo"] and grid[12][3] == "0.0001" and grid[12][5:7] == ["hp", "hi"]
    assert grid[13][:2] == ["-", "-"] and grid[13][3] == "1e-05" and grid[13][5:7] == ["mp", "lo"]  # loss.loss2.* inert for mp
    assert grid[14][:2] == ["-", "-"] and grid[14][3] == "0.0001" and grid[14][5:7] == ["mp", "hi"]
    # param-name headers styled like other headers; config cells skip score styling entirely
    assert ws.cell(row=11, column=1).fill.fgColor.rgb[-6:] == "EAEAEA"
    assert ws.cell(row=11, column=1).font.bold is True
    assert ws.cell(row=11, column=4).font.bold is True
    assert ws.cell(row=12, column=1).fill.patternType is None
    assert ws.cell(row=12, column=1).font.bold is not True
    assert ws.cell(row=12, column=4).fill.patternType is None
    assert all(r[2] is None and r[4] is None for r in grid)  # separator columns stay empty
    # seed block one separator past the 8-wide aggregate score block (cols F..M)
    assert grid[0][14] == "seed 42"
    assert grid[2][14] == "CUB"
    # same treatment on the accuracy sheet (3-wide score tables at F..H, seed at col J)
    ws_acc = wb["Composite I2T Accuracy"]
    agrid = [[c.value for c in r] for r in ws_acc.iter_rows()]
    assert agrid[2][5] == "CUB"
    assert agrid[9][5] == "Mean"
    assert agrid[9][0] == "Arm Overrides" and agrid[9][3] == "Coord Overrides"
    assert agrid[10][:2] == ["loss.blend.lambda", "loss.loss2.targ"] and agrid[10][3] == "lr.init"
    assert agrid[11][:2] == ["0.3", "phylo"]
    assert agrid[13][:2] == ["-", "-"]
    assert agrid[0][9] == "seed 42"

    # the arms workbook: the Arm Overrides band alone (A..B + separator C, score blocks at D), one row
    # per arm at its best coord (lo for both, named in the Coord key column; '-' in the Mean table)
    wb = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")
    grid = [[c.value for c in r] for r in wb.active.iter_rows()]
    assert not any(v == "Coord Overrides" for r in grid for v in r)
    assert grid[2][3] == "CUB"
    assert grid[4][3:6] == ["hp", "lo (1)", "50.00"]
    assert grid[5][3:6] == ["mp", "lo (1)", "40.00"]
    assert grid[7][0] == "Arm Overrides" and grid[7][3] == "Mean"
    assert grid[8][:2] == ["loss.blend.lambda", "loss.loss2.targ"] and grid[8][3:5] == ["Arm", "Coord"]
    assert grid[9][:2] == ["0.3", "phylo"] and grid[9][3:5] == ["hp", "-"]
    assert grid[10][:2] == ["-", "-"] and grid[10][3:5] == ["mp", "-"]
    assert all(r[2] is None for r in grid)
    assert grid[0][12] == "seed 42"


def test_update_phase_metrics_overrides_all_uniform_omits_bands(tmp_path, monkeypatch) -> None:
    # every overridden param resolves to the same value for every row -> no column survives on either
    # side, so both bands are omitted entirely and the score blocks sit leftmost
    for arm, base in (("hp", 0.50), ("mp", 0.40)):
        dpath_coord = _dpath_coord(tmp_path, "cub", arm, "c0")
        dpath_selected = dpath_coord / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_comp(base)))
        (dpath_coord / "overrides.json").write_text(json.dumps({"arm": {"loss.loss1.targ": "mp"}, "coord": {"lr.init": 1.0e-5}}))
        (dpath_coord / "config.json").write_text(json.dumps({"loss": {"loss1": {"targ": "mp"}}, "lr": {"init": 1.0e-5}}))
    _write_meta(tmp_path, ["hp", "mp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, True)

    for kind in ("arm_coords", "arms"):
        wb = load_workbook(tmp_path / "phase_metrics" / kind / "performance" / "map" / "native" / "metrics.xlsx")
        grid = [[c.value for c in r] for r in wb.active.iter_rows()]
        assert not any(v in ("Arm Overrides", "Coord Overrides") for r in grid for v in r)
        assert grid[2][0] == "CUB"  # score blocks leftmost: no band, no separator column


def test_update_phase_metrics_arms_workbook_picks_best_coord_per_dataset(tmp_path, monkeypatch) -> None:
    # the arms workbooks show each arm at its best coord PER DATASET (the workbook's criterion's comp
    # score, highest across-trial mean): on cub a's c0 (60 > 40), on bryo a's c1 (30 > 20). The Mean
    # table then averages those per-dataset bests (45.00), the seed blocks carry the best coord's
    # trial, the accuracy sheet shows the mAP-picked coord's acc (not the best acc), and the hardware
    # sheet reads the best coords' trials and sums their coord_metadata crash counts (cub c0: 1 ram,
    # bryo c1: 2 vram). Under the acc criterion the pick flips (cub: c1, acc 80 > 20; bryo: c0, 70 > 10).
    vals = {  # (dataset, coord) -> (map all, acc i2t), arm 'a' throughout
        ("cub", "c0"): (0.60, "0.20"), ("cub", "c1"): (0.40, "0.80"),
        ("bryo", "c0"): (0.20, "0.70"), ("bryo", "c1"): (0.30, "0.10"),
    }
    trial_t = {("cub", "c0"): "100.00", ("cub", "c1"): "200.00", ("bryo", "c0"): "300.00", ("bryo", "c1"): "400.00"}
    crashes = {
        ("cub", "c0"): {"ram": 1, "vram": 0, "other": 0}, ("cub", "c1"): {"ram": 5, "vram": 5, "other": 5},
        ("bryo", "c0"): {"ram": 7, "vram": 7, "other": 7}, ("bryo", "c1"): {"ram": 0, "vram": 2, "other": 0},
    }
    for (dataset, coord), (all_v, acc_v) in vals.items():
        dpath_coord = _dpath_coord(tmp_path, dataset, "a", coord)
        dpath_selected = dpath_coord / "_seeds" / "42" / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_full_comp(all_v, acc_v)))
        (dpath_coord / "_seeds" / "42" / "trial_metadata.json").write_text(json.dumps({
            "runtime": {"train": {"mean": "1.00"}, "eval": {"mean": "1.00"}, "trial": trial_t[(dataset, coord)]},
            "memory": {"ram": "1.0/128.0 GB", "vram": "1.0/178.4 GB"},
            "killed": None,
        }))
        (dpath_coord / "coord_metadata.json").write_text(json.dumps({"n_crashes": crashes[(dataset, coord)]}))
    _write_meta(tmp_path, ["a"], ["c0", "c1"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    # every table's 'Coord' key cell names the dataset's pick (the Mean table's shows '-': per dataset)
    wb = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")
    grid = [[c.value for c in r] for r in wb.active.iter_rows()]
    assert grid[3][:3] == ["Arm", "Coord", "All"]
    assert grid[4][:3] == ["a", "c0 (1)", "60.00"]   # CUB: c0
    assert grid[8][:3] == ["a", "c1 (1)", "30.00"]   # Bryozoa: c1
    assert grid[12][:3] == ["a", "-", "45.00"]       # Mean of the per-dataset bests
    assert grid[4][9:12] == ["a", "c0", "60.00"] and grid[8][9:12] == ["a", "c1", "30.00"]  # seed 42 block
    agrid = [[c.value for c in r] for r in wb["Composite I2T Accuracy"].iter_rows()]
    assert agrid[4][:3] == ["a", "c0 (1)", "20.00"] and agrid[8][:3] == ["a", "c1 (1)", "10.00"]  # the mAP-picked coords' acc
    hgrid = [[c.value for c in r] for r in wb["Hardware Performance"].iter_rows()]
    assert hgrid[4][:3] == ["a", "c0 (1)", "100"]    # cub c0's trial
    assert hgrid[8][:3] == ["a", "c1 (1)", "400"]    # bryo c1's trial
    assert hgrid[12][:3] == ["a", "-", "250"]
    assert hgrid[12][7:10] == ["1", "2", "0"]        # crash totals: cub c0's + bryo c1's
    # the arm_coords workbook keeps every coord as its own row, so nothing is picked there
    grid = [[c.value for c in r] for r in load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx").active.iter_rows()]
    assert grid[4][:3] == ["a", "c0 (1)", "60.00"] and grid[5][:3] == ["a", "c1 (1)", "40.00"]
    # acc-selection workbook: the pick flips per dataset
    wb_acc = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "acc" / "native" / "metrics.xlsx")
    grid = [[c.value for c in r] for r in wb_acc.active.iter_rows()]
    assert grid[4][:3] == ["a", "c1 (1)", "40.00"]  # cub: c1 (acc 80) -> its mAP 40
    assert grid[8][:3] == ["a", "c0 (1)", "20.00"]  # bryo: c0 (acc 70) -> its mAP 20
    agrid = [[c.value for c in r] for r in wb_acc["Composite I2T Accuracy"].iter_rows()]
    assert agrid[4][:3] == ["a", "c1 (1)", "80.00"] and agrid[8][:3] == ["a", "c0 (1)", "70.00"]
    hgrid = [[c.value for c in r] for r in wb_acc["Hardware Performance"].iter_rows()]
    assert hgrid[4][:3] == ["a", "c1 (1)", "200"] and hgrid[8][:3] == ["a", "c0 (1)", "300"]
    assert hgrid[12][7:10] == ["12", "12", "12"]  # cub c1's 5/5/5 + bryo c0's 7/7/7


def test_update_phase_metrics_hw_sheet(tmp_path, monkeypatch) -> None:
    # the always-on 3rd sheet, "Hardware Performance", mirrors the score sheets' layout (campaign
    # banner, per-dataset tables + Mean table, per-seed blocks, overrides bands, mAP-sheet row order)
    # with per-trial readings from trial_metadata.json as columns, meaned over the same trials as the
    # score tables, rounded to the nearest int. Dataset tables mean that dataset's completed trials
    # ("<arm> (n)" labels, "-" row where an arm has none), seed-block tables carry that seed's single
    # trial, and the Mean table means the per-dataset trial means across datasets -- hp's cub trial
    # times (100.4, 200.4) mean to 150.4, then with bryo's 350.0 -> 250.2 -> "250" (a pooled per-trial
    # mean would give 217: the two-level aggregation matters) -- plus the Total Crashes RAM/VRAM/Other
    # columns (Mean table only, crash totals don't decompose per dataset/seed) summed over the
    # per-dataset coord_metadata.json n_crashes (per-row totals across seeds + datasets). Hardware
    # cells get no winner-bold/heatmap styling despite bold_high/heatmap on; the score sheets carry no
    # hardware tables.
    hw_vals = {  # (arm, dataset, seed) -> (trial, train mean, eval mean, ram, vram)
        ("hp", "cub", "42"): ("100.40", "10.10", "5.10", "100.2/128.0 GB", "20.2/178.4 GB"),
        ("hp", "cub", "43"): ("200.40", "20.10", "7.10", "110.2/128.0 GB", "24.2/178.4 GB"),
        ("hp", "bryo", "42"): ("350.00", "30.10", "9.10", "120.2/128.0 GB", "30.2/178.4 GB"),
        ("mp", "cub", "42"): ("63.49", "7.70", "6.49", "117.2/128.0 GB", "26.3/178.4 GB"),
    }
    for (arm, dataset, seed), (trial_t, train_t, eval_t, ram, vram) in hw_vals.items():
        dpath_trial = _dpath_coord(tmp_path, dataset, arm, "c0") / "_seeds" / seed
        dpath_selected = dpath_trial / "evals" / "_selected"
        dpath_selected.mkdir(parents=True)
        _write_group_metrics(dpath_selected, _scores_grp(_comp(0.50)))
        (dpath_trial / "trial_metadata.json").write_text(json.dumps({
            "runtime": {"train": {"mean": train_t}, "eval": {"mean": eval_t}, "trial": trial_t},
            "memory": {"ram": ram, "vram": vram},
            "killed": None,
        }))
    # the per-dataset coord files: overrides/config identical across a coord's datasets, the
    # crash counters per dataset (hp's 2/1/0 total is split across cub and bryo)
    for arm, dataset, overrides, meta, crashes in (
        ("hp", "cub", {"loss.blend.lambda": 0.3}, {"loss": {"blend": {"lambda": 0.3}}}, {"ram": 1, "vram": 1, "other": 0}),
        ("hp", "bryo", {"loss.blend.lambda": 0.3}, {"loss": {"blend": {"lambda": 0.3}}}, {"ram": 1, "vram": 0, "other": 0}),
        ("mp", "cub", {"loss.loss1.targ": "mp"}, {"loss": {"loss1": {"targ": "mp"}}}, {"ram": 0, "vram": 0, "other": 3}),
    ):
        dpath_coord = _dpath_coord(tmp_path, dataset, arm, "c0")
        (dpath_coord / "overrides.json").write_text(json.dumps({"arm": overrides, "coord": {}}))
        (dpath_coord / "config.json").write_text(json.dumps({**meta}))
        (dpath_coord / "coord_metadata.json").write_text(json.dumps({"n_crashes": crashes}))
    _write_meta(tmp_path, ["hp", "mp"], ["c0"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_phase_metrics(_EVAL_GROUPS, "std", True, False, True, _SUPP_OFF, True)

    wb = load_workbook(tmp_path / "phase_metrics" / "arms" / "performance" / "map" / "native" / "metrics.xlsx")
    assert wb.sheetnames == ["Composite mAP", "Composite I2T Accuracy", "Hardware Performance"]
    ws = wb["Hardware Performance"]
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    merged = {str(m) for m in ws.merged_cells.ranges}
    # overrides band at A..B + separator C; aggregate block at D -- dataset tables 7 wide (D..J),
    # the Mean table 10 (D..M, crash columns appended), so the block spans D..M and seed 42 starts
    # one separator later at O
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; mAP-selection)"
    assert grid[12][0] == "Arm Overrides"
    assert "A13:B13" in merged
    assert grid[13][:2] == ["loss.blend.lambda", "loss.loss1.targ"]
    assert grid[14][:2] == ["0.3", "-"]
    assert grid[15][:2] == ["-", "mp"]
    # CUB table: merged title banner, Arm + Coord (the dataset's pick) + hw header, '<coord> (n)'
    # labels; hp means its 2 cub trials, mp its 1
    assert grid[2][3] == "CUB"
    assert "D3:J3" in merged
    assert grid[3][3:10] == ["Arm", "Coord", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM"]
    assert grid[4][3:10] == ["hp", "c0 (2)", "150", "15", "6", "105", "22"]
    assert grid[5][3:10] == ["mp", "c0 (1)", "63", "8", "6", "117", "26"]
    # Bryozoa table: hp's single trial passes through; mp has no bryo trials -> "-" row, no pick
    assert grid[7][3] == "Bryozoa"
    assert grid[9][3:10] == ["hp", "c0 (1)", "350", "30", "9", "120", "30"]
    assert grid[10][3:10] == ["mp", "- (0)", "-", "-", "-", "-", "-"]
    # Mean table: cross-dataset means of the per-dataset trial means + the crash-total columns
    # (the pick is per dataset: '-' in its Coord column)
    assert grid[12][3] == "Mean"
    assert "D13:M13" in merged
    assert grid[13][3:13] == ["Arm", "Coord", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM",
                              "Total Crashes RAM", "Total Crashes VRAM", "Total Crashes Other"]
    assert grid[14][3:13] == ["hp", "-", "250", "23", "8", "113", "26", "2", "1", "0"]
    # mp: single cub trial, values pass straight through the two-level mean before rounding
    assert grid[15][3:13] == ["mp", "-", "63", "8", "6", "117", "26", "0", "0", "3"]
    # seed blocks: per-dataset tables only (no Mean), plain labels; seed 42 at O..U, seed 43 at W..AC
    assert grid[0][14] == "seed 42"
    assert grid[0][22] == "seed 43"
    assert grid[2][14] == "CUB"
    assert grid[4][14:21] == ["hp", "c0", "100", "10", "5", "100", "20"]   # seed 42 CUB, hp's 42 trial alone
    assert grid[5][14:21] == ["mp", "c0", "63", "8", "6", "117", "26"]     # mp's only trial
    assert grid[9][14:21] == ["hp", "c0", "350", "30", "9", "120", "30"]   # seed 42 Bryozoa
    assert grid[10][14:21] == ["mp", "-", "-", "-", "-", "-", "-"]         # mp: no bryo trial
    assert grid[12][14] is None  # no Mean table in seed blocks
    assert grid[4][22:29] == ["hp", "c0", "200", "20", "7", "110", "24"]   # seed 43 CUB, hp's 43 trial alone
    assert grid[5][22:29] == ["mp", "c0", "-", "-", "-", "-", "-"]         # mp has no 43 trial (its cub pick still named)
    # separator columns between the band and blocks stay empty
    assert all(r[2] is None and r[13] is None and r[21] is None for r in grid)
    # header + key cells styled like the score sheets' (key cells left-aligned); value
    # cells get no winner-bold/heatmap styling despite bold_high/heatmap on
    assert ws.cell(row=4, column=6).font.bold is True
    assert ws.cell(row=4, column=6).fill.fgColor.rgb[-6:] == "EAEAEA"
    assert ws.cell(row=5, column=4).font.bold is True
    assert ws.cell(row=5, column=4).alignment.horizontal == "left"
    assert ws.cell(row=5, column=5).font.bold is True  # "c0 (2)": a key cell
    assert ws.cell(row=5, column=5).alignment.horizontal == "left"
    assert ws.cell(row=6, column=6).font.bold is not True  # hp's 150 would be the Time Trial "winner"
    assert ws.cell(row=5, column=6).font.bold is not True
    assert ws.cell(row=5, column=6).fill.patternType is None
    # the score sheets carry no hardware tables; their overrides bands stay leftmost
    for sheet in ("Composite mAP", "Composite I2T Accuracy"):
        sgrid = [[c.value for c in r] for r in wb[sheet].iter_rows()]
        assert sgrid[12][0] == "Arm Overrides"
        assert sgrid[12][3] == "Mean"
        assert not any(v in ("Hardware Performance", "Time Trial") for r in sgrid for v in r)
    # the arm_coords workbook's crash totals are per (arm, coord) row -- the same sums here, one key column over
    hgrid = [[c.value for c in r] for r in load_workbook(tmp_path / "phase_metrics" / "arm_coords" / "performance" / "map" / "native" / "metrics.xlsx")["Hardware Performance"].iter_rows()]
    assert hgrid[14][3:5] == ["hp", "c0"] and hgrid[14][10:13] == ["2", "1", "0"]
    assert hgrid[15][3:5] == ["mp", "c0"] and hgrid[15][10:13] == ["0", "0", "3"]


def _write_test_scores(dpath_trial, scores_grp, chkpt, macro=None) -> None:
    """A test-scored trial's per-group score files (_seeds/<seed>/<group>.json), the shape test.py
    writes: {'chkpt': saved checkpoint index, 'scores': the group's scores}; the same subtree stands
    in for both standard groups (and `macro` for the macro ones)."""
    macro = scores_grp if macro is None else macro
    dpath_trial.mkdir(parents=True, exist_ok=True)
    for group_key, grp in (("native", scores_grp), ("native_macro", macro), ("joint", scores_grp), ("joint_macro", macro)):
        (dpath_trial / f"{group_key}.json").write_text(json.dumps({"chkpt": chkpt, "scores": grp}))


def test_update_test_stats_writes_stacked_tables_with_chkpt_column(tmp_path, monkeypatch) -> None:
    # one workbook per eval group at <test dir>/map/test_<group>.xlsx, laid out like the campaign
    # arm_coords workbooks -- score sheets of stacked per-dataset tables + Mean table + per-seed
    # blocks -- but with a third 'Chkpt' key column (the combo's saved-checkpoint index) and no
    # 'Hardware Performance' sheet. hp/c0 has 2 scored cub trials (chkpt 7) and none in bryo (blank
    # row, '-' chkpt); mp has none anywhere -> no rows. The Mean table shows '-' for chkpt (a
    # per-dataset value).
    for seed, base in (("42", 0.50), ("43", 0.60)):
        _write_test_scores(_dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / seed, _scores_grp(_comp(base)), 7)
    _write_meta(tmp_path, ["hp", "mp"], ["c0"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_test_stats(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    for group_key in _GROUP_KEYS:
        assert (tmp_path / "map" / f"test_{group_key}.xlsx").exists()
    wb = load_workbook(tmp_path / "map" / "test_native.xlsx")
    assert wb.sheetnames == ["Composite mAP", "Composite I2T Accuracy"]  # no hardware sheet: test runs no training
    ws = wb.active
    grid = [[c.value for c in row] for row in ws.iter_rows()]
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.parent.name} (Native; test)"
    assert grid[2][0] == "CUB"
    assert grid[3][:9] == ["Arm", "Coord", "Chkpt", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[4][:4] == ["hp", "c0 (2)", "7", "55.00 ± 7.07"]
    assert grid[6][0] == "Bryozoa"
    assert grid[8][:4] == ["hp", "c0 (0)", "-", "-"]  # no bryo trials: blank row, no chkpt either
    assert grid[10][0] == "Mean"
    assert grid[12][:9] == ["hp", "c0", "-", "55.00", "57.00", "56.00", "58.00", "59.00", "60.00"]
    assert not any(v in ("mp", "mp (0)") for r in grid for v in r)  # no completed trials anywhere -> no rows
    # the chkpt cell is a key column: header-styled (bold, grey, left-aligned), never score-styled
    assert ws.cell(row=5, column=3).font.bold is True
    assert ws.cell(row=5, column=3).fill.fgColor.rgb[-6:] == "EAEAEA"
    assert ws.cell(row=5, column=3).alignment.horizontal == "left"
    # per-seed blocks one separator past the 9-wide aggregate block, chkpt carried (same value per seed)
    assert grid[0][10] == "seed 42"
    assert grid[0][20] == "seed 43"
    assert grid[2][10] == "CUB"
    assert grid[4][10:19] == ["hp", "c0", "7", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00"]
    assert grid[8][10:13] == ["hp", "c0", "-"]  # seed 42 bryo: no trial
    assert grid[4][20:24] == ["hp", "c0", "7", "60.00"]
    assert all(r[9] is None and r[19] is None for r in grid)  # separator columns stay empty
    assert not any(r[10] == "Mean" for r in grid)  # seed blocks have no Mean table
    # accuracy sheet: same layout over the single I2T column (4-wide tables, seed 42 block at col F)
    agrid = [[c.value for c in row] for row in wb["Composite I2T Accuracy"].iter_rows()]
    assert agrid[3][:4] == ["Arm", "Coord", "Chkpt", "I2T"]
    assert agrid[4][:4] == ["hp", "c0 (2)", "7", "61.00 ± 7.07"]
    assert agrid[12][:4] == ["hp", "c0", "-", "61.00"]
    assert agrid[0][5] == "seed 42"
    assert agrid[4][5:9] == ["hp", "c0", "7", "56.00"]


def test_update_test_stats_chkpt_per_dataset(tmp_path, monkeypatch) -> None:
    # a coord's qual-selected checkpoint differs per dataset, so each dataset table (aggregate and
    # seed blocks alike) carries its own value; the cross-dataset Mean table has no single one -> '-'
    _write_test_scores(_dpath_coord(tmp_path, "cub", "hp", "c0") / "_seeds" / "42", _scores_grp(_comp(0.50)), 3)
    _write_test_scores(_dpath_coord(tmp_path, "bryo", "hp", "c0") / "_seeds" / "42", _scores_grp(_comp(0.60)), 9)
    _write_meta(tmp_path, ["hp"], ["c0"], ["cub", "bryo"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_test_stats(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, False)

    grid = [[c.value for c in row] for row in load_workbook(tmp_path / "map" / "test_native.xlsx").active.iter_rows()]
    assert grid[2][0] == "CUB"
    assert grid[4][:4] == ["hp", "c0 (1)", "3", "50.00"]
    assert grid[6][0] == "Bryozoa"
    assert grid[8][:4] == ["hp", "c0 (1)", "9", "60.00"]
    assert grid[10][0] == "Mean"
    assert grid[12][:4] == ["hp", "c0", "-", "55.00"]
    assert grid[4][10:13] == ["hp", "c0", "3"] and grid[8][10:13] == ["hp", "c0", "9"]  # seed 42 block


def test_update_test_stats_overrides_bands(tmp_path, monkeypatch) -> None:
    # overrides=True renders the Arm/Coord Overrides bands from the test tree's config.json /
    # overrides.json copies, exactly like the phase workbooks: one column per differing declared
    # param, aligned with the aggregate Mean table's rows, score blocks shifted right past them
    for arm, targ, base in (("hp", "phylo", 0.50), ("mp", "mp", 0.40)):
        dpath_coord = _dpath_coord(tmp_path, "cub", arm, "c0")
        _write_test_scores(dpath_coord / "_seeds" / "42", _scores_grp(_comp(base)), 5)
        (dpath_coord / "overrides.json").write_text(json.dumps({"arm": {"loss.loss1.targ": targ}, "coord": {"lr.init": 1.0e-5}}))
        (dpath_coord / "config.json").write_text(json.dumps({"loss": {"loss1": {"targ": targ}}, "lr": {"init": 1.0e-5}}))
    _write_meta(tmp_path, ["hp", "mp"], ["c0"], ["cub"])

    monkeypatch.setattr(ArtifactManager, "dpath_phase", tmp_path)

    report.update_test_stats(_EVAL_GROUPS, "std", False, False, False, _SUPP_OFF, True)

    grid = [[c.value for c in row] for row in load_workbook(tmp_path / "map" / "test_native.xlsx").active.iter_rows()]
    # the arm band's one differing param at col A + separator B; lr.init is uniform -> the coord
    # band is omitted; score blocks start at C, band rows aligned with the Mean banner (row 8)
    assert grid[2][2] == "CUB"
    assert grid[3][2:6] == ["Arm", "Coord", "Chkpt", "All"]
    assert grid[4][2:6] == ["hp", "c0 (1)", "5", "50.00"]
    assert grid[5][2:6] == ["mp", "c0 (1)", "5", "40.00"]
    assert grid[7][0] == "Arm Overrides" and grid[7][2] == "Mean"
    assert grid[8][0] == "loss.loss1.targ" and grid[8][2:5] == ["Arm", "Coord", "Chkpt"]
    assert grid[9][0] == "phylo" and grid[9][2:5] == ["hp", "c0", "-"]
    assert grid[10][0] == "mp" and grid[10][2:5] == ["mp", "c0", "-"]
    assert not any(v == "Coord Overrides" for r in grid for v in r)
    assert all(r[1] is None for r in grid)  # band separator column stays empty


def test_fold_hist_columns_halves_on_overflow() -> None:
    # one histogram per batch, folded to at most `threshold` columns: the group size is the smallest
    # power of two that fits, so the strip halves (1 -> 2 -> 4 batches per column) as batches pile up
    cols = [[float(i), float(i) + 1] for i in range(10)]  # 2 "bins" per column, values track the index

    grid, stride = report._fold_hist_columns(cols, 16)  # under budget -> untouched
    assert stride == 1
    assert grid.tolist() == cols

    grid, stride = report._fold_hist_columns(cols, 5)  # 10 > 5 -> pairs averaged, 5 columns
    assert stride == 2
    assert grid.tolist() == [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5], [8.5, 9.5]]

    grid, stride = report._fold_hist_columns(cols, 4)  # 10//2=5 still over -> groups of 4
    assert stride == 4
    # only FULL groups: batches 0-3 and 4-7; the trailing 8, 9 are dropped, not drawn as a
    # 2-batch column sitting next to 4-batch ones
    assert grid.tolist() == [[1.5, 2.5], [5.5, 6.5]]


def test_fold_hist_columns_stride_is_smallest_power_of_two_that_fits() -> None:
    import math as _math
    for n_batches in (129, 200, 512, 1000):
        cols = [[1.0] * 10 for _ in range(n_batches)]
        grid, stride = report._fold_hist_columns(cols, 128)
        assert stride & (stride - 1) == 0  # power of two
        assert len(grid) == n_batches // stride <= 128
        assert n_batches // (stride // 2) > 128  # one step smaller would overflow
        assert n_batches - len(grid) * stride < stride  # at most one partial group dropped
        assert grid.shape[1] == 10  # bins preserved


def test_fit_ylabel_shrinks_a_caption_taller_than_its_axes() -> None:
    # the label is rotated, so its height is its longest line's length: a two-line caption fits the same way
    for caption in ("n-shot Acc. (ID I2T)", "n-shot Acc.\n(ID I2T)", r"$\mathcal{E}_{\text{ir}} = D_{\mathrm{KL}}(y\|p^*)$"):
        fig = report.plt.figure(figsize=(4, 1))
        ax = fig.add_subplot()
        ax.set_ylabel(caption, fontsize=40)
        assert ax.yaxis.label.get_window_extent().height > ax.bbox.height
        report._fit_ylabel(ax)
        assert ax.yaxis.label.get_fontsize() < 40
        assert 0.95 * ax.bbox.height < ax.yaxis.label.get_window_extent().height <= ax.bbox.height  # no smaller than it must be
        report.plt.close(fig)


def test_fit_ylabel_keeps_a_caption_that_fits() -> None:
    fig = report.plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    ax.set_ylabel("n-shot Acc.\n(ID I2T)", fontsize=12)
    report._fit_ylabel(ax)
    assert ax.yaxis.label.get_fontsize() == 12
    report.plt.close(fig)


def test_residual_mark_keys_off_provenance_not_magnitude() -> None:
    # one mark (report._RESID_UNVALIDATED), off resid_paths alone: any row off the plain p* - y subtraction makes
    # the batch unvalidated, WHATEVER the value reads -- it can come out a noise floor, an exact zero or a
    # plausible number, so magnitude certifies nothing in either direction. A residual off an exact path (hard
    # closed form, feasible exact zero) carries none, being as trustworthy at 1e-40 as at 1e-2
    tiny, fine = 1e-30, 1e-3
    data_epoch = {
        "dalpha_sum_abs_full": [[1.0, 0.5, 0.5]] * 5,
        "dalpha_sum_abs_res": [[v, v, v] for v in (tiny, tiny, 0.0, tiny, fine)],
        "resid_paths": [
            [1.0, 0.0, 0.0],  # hard closed form, tiny: exact -- not marked
            [0.0, 0.0, 1.0],  # subtracted, tiny
            [0.0, 1.0, 0.0],  # feasible exact zero: exact -- not marked
            [0.5, 0.0, 0.5],  # any subtracted row puts the batch's aggregate in doubt
            [0.0, 0.0, 1.0],  # subtracted and plausible-looking: marked all the same
        ],
    }
    assert report._resid_unvalidated(data_epoch).tolist() == [False, True, False, True, True]


def test_a_wrong_residual_on_a_well_fitted_batch_is_marked_whatever_it_reads() -> None:
    # why the mark takes no magnitude test: a floor scaled by A_full (the term against the full gradient) scales
    # with a gradient magnitude that shrinks with the fit, while the subtraction's error is set by the order-one
    # entries of p* and y and does not. Four genuine unit vectors fitted to rows [0.6, 0.4, 0, 0] at alpha 25:
    # A_full is down at 3e-16, so such a floor would sit at 6e-30 and pass the batch -- whose kl_ir reads
    # exactly 0 against a true 2.3e-22. Provenance marks it regardless
    import math
    import torch
    from tests.unit.test_loss_targets import import_loss_module
    L = import_loss_module()
    alpha = 25.0
    theta = math.acos(1.0 - math.log(1.5) / alpha)  # s_1 - s_2 = log(0.6 / 0.4) / alpha
    u1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    u2 = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float64)
    V = torch.stack([u1, u2, -u1, -u2])
    S = V @ V.T  # a real cosine Gram matrix
    Y = torch.tensor([[.6, .4, 0, 0], [.4, .6, 0, 0], [0, 0, .6, .4], [0, 0, .4, .6]], dtype=torch.float64)
    stats = L.infonce_batch_stats(S, (Y > 0).double(), Y, alpha * S, torch.tensor(math.log(alpha), dtype=torch.float64), False)

    r = math.exp(-2 * alpha)
    true_kl_ir = -0.6 * math.log1p(-2 * r / (1 + 2 * r))  # floor {the zeros}, interior {0.4}, cap {0.6}
    assert true_kl_ir == pytest.approx(2.3e-22, rel=0.02)
    assert abs(stats["kl_ir"] - true_kl_ir) > 0.5 * true_kl_ir  # the reported value is simply wrong
    data_epoch = {key: [val] for key, val in stats.items()}
    assert stats["resid_paths"] == [0.0, 0.0, 1.0]
    floor = 100 * np.finfo(np.float64).eps * stats["dalpha_sum_abs_full"][0]
    assert stats["dalpha_sum_abs_res"][0] > floor  # the magnitude test this mark replaced would have passed it
    assert report._resid_unvalidated(data_epoch).tolist() == [True]


def _alpha_data_epoch(n, **series):
    """A data_epoch carrying every series plot_alpha_curves indexes, empty unless given."""
    keys = ["scale", "scale2", "logit_scale", "logit_scale2", "logit_scale_grad", "logit_scale2_grad",
            "dlogalpha_correction", "resid_paths",
            *(f"{p}alpha_req_{stat}" for p in ("", "log_") for stat in ("min", "mean", "max")),
            *(f"{p}_{agg}_{comp}" for p in ("dalpha", "dlogalpha") for agg in report._DALPHA_AGGS
              for comp in ("full", "struct", "res", "sres", "ires"))]
    return {**{key: [] for key in keys}, **series}


def _plot_alpha(data_epoch, n, tmp_path, monkeypatch, scale_key, prefix, sym):
    """plot_alpha_curves with the save intercepted: the panels' y labels, top to bottom, and the lines drawn
    on each (label -> y data)."""
    seen = {}

    def finish(fig, axes, axes_hist, legend_handles, *args, **kwargs):
        seen["panels"] = [(ax.get_ylabel(), {line.get_label(): list(line.get_ydata()) for line in ax.get_lines()})
                          for ax in axes]
        seen["marks"] = {ax.get_ylabel(): sorted({patch.get_label() for patch in ax.patches} - {"_nolegend_"})
                         for ax in axes}
        seen["blocks"] = [[ax.get_ylabel() for ax in block] for block in kwargs.get("axes_blocks", ())]
        seen["ylims"] = {ax.get_ylabel(): ax.get_ylim() for ax in axes}
        seen["yticks"] = {ax.get_ylabel(): list(ax.get_yticks()) for ax in axes}
        fig.canvas.draw()  # fills the autoscaled panels' tick labels in
        seen["yticklabels"] = {ax.get_ylabel(): [t.get_text() for t in ax.get_yticklabels()] for ax in axes}
        plt.close(fig)

    monkeypatch.setattr(report, "_finish_curves", finish)
    report.plot_alpha_curves(data_epoch, list(range(n)), tmp_path, 10, 8, 8, 1.0, 8.0, 1.0, 1.0,
                             scale_key, prefix, sym, plot_title="t", output_filename="x.png")
    _plot_alpha.seen = seen
    return seen["panels"]


def test_logalpha_figure_draws_the_scale_parameters_grad_with_the_correction(tmp_path, monkeypatch) -> None:
    # what the scale actually received, measured: model.logit_scale.grad (read post-backward, pre-step) gets
    # its own panel directly under the log alpha it belongs to, with loss.infonce.block_residuals' correction
    # drawn on it -- so the gradient without the intervention is the one line minus the other. Captioned
    # (.grad) because the analytical dlogalpha full-term panel further down carries the same nabla, and the
    # two need not agree (blend coefficients, loss.unitless)
    n = 4
    grad, corr = [-0.25, -0.20, -0.10, -0.05], [0.17, 0.12, 0.05, 0.01]
    data_epoch = _alpha_data_epoch(n, scale=[2.0, 2.2, 2.5, 2.7], logit_scale=[0.7, 0.8, 0.9, 1.0],
                                   logit_scale_grad=grad, dlogalpha_correction=corr)

    panels = _plot_alpha(data_epoch, n, tmp_path, monkeypatch, "logit_scale", "dlogalpha", r"\log \alpha")
    assert [label for label, _ in panels] == [r"$\log \alpha$", r"$\nabla_{\log \alpha} \mathcal{L}$" + "\n(.grad)"]
    lines = panels[1][1]
    assert lines[".grad"] == grad and lines["correction"] == corr

    # the parameter is log alpha, so the alpha figure gets no such panel
    panels = _plot_alpha(data_epoch, n, tmp_path, monkeypatch, "scale", "dalpha", r"\alpha")
    assert [label for label, _ in panels] == [r"$\alpha$"]

    # separate logit scalars: each scale's own grad under its own panel; the correction is the primary's
    data_epoch = _alpha_data_epoch(n, logit_scale=[0.7] * n, logit_scale2=[0.4] * n, logit_scale_grad=grad,
                                   logit_scale2_grad=[0.3] * n, dlogalpha_correction=corr)
    panels = _plot_alpha(data_epoch, n, tmp_path, monkeypatch, "logit_scale", "dlogalpha", r"\log \alpha")
    assert [label for label, _ in panels] == [
        r"$\log \alpha$", r"$\nabla_{\log \alpha} \mathcal{L}$" + "\n(.grad)",
        r"$\log \alpha_2$", r"$\nabla_{\log \alpha_2} \mathcal{L}$" + "\n(.grad)"]
    assert "correction" in panels[1][1] and "correction" not in panels[3][1]
    assert panels[3][1][".grad"] == [0.3] * n


def test_a_frozen_scale_keeps_its_panel_on_both_figures(tmp_path, monkeypatch) -> None:
    # a frozen scale (loss.logits.scale.freeze) is recorded all the same: a flat line on the alpha figure and
    # on the logalpha one, captioned as frozen and with no .grad panel under it (a frozen parameter has none
    # -- the empty .grad series is how the figure reads it as frozen). The y axis of a panel drawing the one
    # value and nothing else has no range to autoscale: it gets that value as its only tick, the line centred
    n = 4
    alpha = 1.0 / 0.07
    frozen = dict(scale=[alpha] * n, logit_scale=[np.log(alpha)] * n)

    for scale_key, prefix, sym, val, tick in (("scale", "dalpha", r"\alpha", alpha, "14.29"),
                                              ("logit_scale", "dlogalpha", r"\log \alpha", np.log(alpha), "2.659")):
        panels = _plot_alpha(_alpha_data_epoch(n, **frozen), n, tmp_path, monkeypatch, scale_key, prefix, sym)
        label = rf"${sym}$" + "\n(frozen)"
        assert [caption for caption, _ in panels] == [label]  # the scale's panel, and no .grad one
        assert panels[0][1] == {rf"${sym}$": [val] * n}  # the recorded value, drawn as is
        assert _plot_alpha.seen["yticklabels"][label] == [tick]
        lo, hi = _plot_alpha.seen["ylims"][label]
        assert lo < val < hi and (lo + hi) / 2 == pytest.approx(val)

    # read against its target-implied bounds the panel has a range, and autoscales over the pair -- that
    # comparison is what a frozen scale's panel is kept for
    reqs = {f"alpha_req_{stat}": [v, v + 1.0, v + 2.0, v + 3.0] for stat, v in (("min", 2.0), ("mean", 5.0), ("max", 20.0))}
    _plot_alpha(_alpha_data_epoch(n, **frozen, **reqs), n, tmp_path, monkeypatch, "scale", "dalpha", r"\alpha")
    label = r"$\alpha$" + "\n(frozen)"
    lo, hi = _plot_alpha.seen["ylims"][label]
    assert lo <= 2.0 and hi >= 23.0 and len(_plot_alpha.seen["yticklabels"][label]) > 1

    # bounds all at infinity (hard binary targets under the linear tsm) draw nothing: the one value again
    inf = {f"alpha_req_{stat}": [float("inf")] * n for stat in ("min", "mean", "max")}
    _plot_alpha(_alpha_data_epoch(n, **frozen, **inf), n, tmp_path, monkeypatch, "scale", "dalpha", r"\alpha")
    assert _plot_alpha.seen["yticklabels"][label] == ["14.29"]

    # a learnable scale is untouched: no frozen caption, its .grad panel under it, an autoscaled axis
    learnable = dict(scale=[2.0, 2.2, 2.5, 2.7], logit_scale=[0.7, 0.8, 0.9, 1.0], logit_scale_grad=[-0.2] * n)
    panels = _plot_alpha(_alpha_data_epoch(n, **learnable), n, tmp_path, monkeypatch, "logit_scale", "dlogalpha", r"\log \alpha")
    assert [caption for caption, _ in panels] == [r"$\log \alpha$", r"$\nabla_{\log \alpha} \mathcal{L}$" + "\n(.grad)"]
    assert len(_plot_alpha.seen["yticklabels"][r"$\log \alpha$"]) > 1

    # separate logit scalars share loss.logits.scale.freeze, so the second scale is frozen alongside
    both = dict(logit_scale=[0.7] * n, logit_scale2=[0.7] * n)
    panels = _plot_alpha(_alpha_data_epoch(n, **both), n, tmp_path, monkeypatch, "logit_scale", "dlogalpha", r"\log \alpha")
    assert [caption for caption, _ in panels] == [r"$\log \alpha$" + "\n(frozen)", r"$\log \alpha_2$" + "\n(frozen)"]


def _dalpha_series(n, prefix, res_abs):
    """The twenty-five dalpha-family series of one figure, the residual comps' magnitudes at `res_abs`."""
    series = {}
    for agg in report._DALPHA_AGGS:
        for comp in ("full", "struct", "res", "sres", "ires"):
            val = res_abs if (agg in ("sum_abs", "row_abs") and comp in ("res", "sres", "ires")) else 0.5
            series[f"{prefix}_{agg}_{comp}"] = [[val, val, val]] * n
    return series


def test_residual_panels_carry_the_provenance_marks_and_the_blocks_stay_aligned(tmp_path, monkeypatch) -> None:
    # the residual comps are hatched "unvalidated" over every batch with rows off the subtraction, whatever
    # their magnitude; full / struct never are. The analytical full term says so in its caption, sharing its
    # symbol with the measured .grad panel. And the agg blocks are ruled around the dalpha panels -- NOT shifted
    # up by the .grad panel that sits among the scale ones
    n = 4
    paths = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    data_epoch = _alpha_data_epoch(n, logit_scale=[0.7] * n, logit_scale_grad=[-0.1] * n, resid_paths=paths,
                                   **_dalpha_series(n, "dalpha", 1e-30), **_dalpha_series(n, "dlogalpha", 1e-30))
    _plot_alpha(data_epoch, n, tmp_path, monkeypatch, "logit_scale", "dlogalpha", r"\log \alpha")
    marks, blocks = _plot_alpha.seen["marks"], _plot_alpha.seen["blocks"]

    nabla = lambda sup: rf"$\nabla_{{\log \alpha}}{sup} \mathcal{{L}}$"
    for sup in (r"^{\text{R}}", r"^{\text{SR}}", r"^{\text{IR}}"):
        assert marks[nabla(sup)] == ["unvalidated"]
    assert marks[nabla("") + "\n(analytic)"] == [] and marks[nabla(r"^{\text{S}}")] == []
    assert marks[nabla("") + "\n(.grad)"] == []
    # each block is one agg's five comps, in order: the sums' block opens on the analytical full term, not
    # on a scale or .grad panel
    assert [len(block) for block in blocks] == [5, 5, 5, 5, 5]
    assert blocks[0][0] == nabla("") + "\n(analytic)" and blocks[0][-1] == nabla(r"^{\text{IR}}")
    # the blocks in order: the sums, the per-pair magnitude A and |sum| / A, the per-anchor magnitude B and
    # |sum| / B -- a ratio captioned as the quotient it is, over its magnitude block's own symbol
    sups = ("", r"^{\text{S}}", r"^{\text{R}}", r"^{\text{SR}}", r"^{\text{IR}}")
    for idx_block, mag in ((1, "A"), (3, "B")):
        mags = [rf"\text{{{mag}}}_{{\log \alpha}}{sup}" for sup in sups]
        assert blocks[idx_block] == [f"${m}$" for m in mags]
        assert blocks[idx_block + 1] == [rf"$\frac{{|{nabla(sup)[1:-1]}|}}{{{m}}}$" for sup, m in zip(sups, mags)]
    # the logalpha figure's ratio panels repeat the alpha figure's, so both ratio blocks are left blank there
    # -- and drawn on the alpha figure, the per-anchor one included
    drawn = lambda: {label: bool(lines) for label, lines in _plot_alpha.seen["panels"]}
    assert not any(drawn()[label] for idx_block in (2, 4) for label in blocks[idx_block])
    assert all(drawn()[label] for idx_block in (0, 1, 3) for label in blocks[idx_block])
    _plot_alpha(data_epoch, n, tmp_path, monkeypatch, "scale", "dalpha", r"\alpha")
    assert all(drawn().values()) and len(drawn()) == 25
    assert _plot_alpha.seen["blocks"][4][1] == r"$\frac{|\nabla_{\alpha}^{\text{S}} \mathcal{L}|}{\text{B}_{\alpha}^{\text{S}}}$"

    # the mark does not move with the magnitude: a plausible-looking residual is hatched just the same
    data_epoch.update(_dalpha_series(n, "dalpha", 1e-3))
    data_epoch.update(_dalpha_series(n, "dlogalpha", 1e-3))
    _plot_alpha(data_epoch, n, tmp_path, monkeypatch, "logit_scale", "dlogalpha", r"\log \alpha")
    assert _plot_alpha.seen["marks"][nabla(r"^{\text{R}}")] == ["unvalidated"]


def test_ratio_panels_keep_their_end_values_inside_the_limits(tmp_path, monkeypatch) -> None:
    # a ratio of exactly 1 (perfect coherence: a hard target's SR / IR terms, throughout) or exactly 0 (total
    # cancellation) is a reading, not a missing one -- but at y limits of exactly (0, 1) it lies under the
    # panel's border, which is drawn over the curves, and the panel reads as empty. So both ends sit strictly
    # inside the displayed limits, on both ratio blocks, the ticks staying on the ratio's own range
    n = 4
    series = _dalpha_series(n, "dalpha", 1e-3)
    for agg in report._DALPHA_RATIO_AGGS:
        series[f"dalpha_{agg}_sres"] = [[1.0, 1.0, 1.0]] * n
        series[f"dalpha_{agg}_ires"] = [[0.0, 0.0, 0.0]] * n
    panels = dict(_plot_alpha(_alpha_data_epoch(n, **series), n, tmp_path, monkeypatch, "scale", "dalpha", r"\alpha"))
    ylims, yticks, blocks = (_plot_alpha.seen[key] for key in ("ylims", "yticks", "blocks"))

    ratio_labels = blocks[2] + blocks[4]  # |sum| / A, |sum| / B
    assert len(ratio_labels) == 10
    for label in ratio_labels:
        lo, hi = ylims[label]
        drawn = [v for line in panels[label].values() for v in line]
        assert drawn and lo < min(drawn) and max(drawn) < hi, label
        assert lo < 0.0 and 1.0 < hi  # whatever this trial drew: the ends are inside on every ratio panel
        assert yticks[label] == [0.0, 0.5, 1.0]
    # the end values really were drawn, and drawn unchanged: padding the view, not offsetting the data
    assert {v for line in panels[blocks[2][3]].values() for v in line} == {1.0}
    assert {v for line in panels[blocks[4][4]].values() for v in line} == {0.0}


def test_kl_figure_marks_its_residual_panels_unvalidated(tmp_path, monkeypatch) -> None:
    # E_R / E_SR / E_IR are the residual family's, so they carry the same unvalidated mark; the total and the
    # structural part, which difference nothing against y, do not. A trial whose every row came off an exact
    # path draws no mark at all
    seen = {}

    def finish(fig, axes, axes_hist, legend_handles, *args, **kwargs):
        seen["marks"] = [sorted({patch.get_label() for patch in ax.patches} - {"_nolegend_"}) for ax in axes]
        seen["legends"] = len(legend_handles)
        plt.close(fig)

    monkeypatch.setattr(report, "_finish_curves", finish)
    n = 3
    kl = {key: [0.1] * n for key in ("kl", "kl_s", "kl_ir", "kl_sr")}
    args = (list(range(n)), tmp_path, 10, 8, 8, 1.0, 8.0, 1.0, 1.0)

    report.plot_kl_curves({**kl, "resid_paths": [[0.5, 0.0, 0.5]] * n}, *args, plot_title="t", output_filename="x.png")
    assert seen["marks"] == [[], [], ["unvalidated"], ["unvalidated"], ["unvalidated"]] and seen["legends"] == 3

    report.plot_kl_curves({**kl, "resid_paths": [[1.0, 0.0, 0.0]] * n}, *args, plot_title="t", output_filename="x.png")
    assert seen["marks"] == [[], [], [], [], []] and seen["legends"] == 0
