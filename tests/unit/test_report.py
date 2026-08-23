import json

import numpy as np
import pytest
from openpyxl import load_workbook

from utils import report
from utils.train import ArtifactManager
from utils.utils import load_pickle, save_pickle, paths


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
    # trials by their written best-checkpoint (_best/<criterion>/) metrics -- not by a `complete` flag that
    # isn't set yet; each criterion aggregates its own _best files into its own stats/<criterion>/ subtree
    dataset = "cub"
    dpath_dataset = tmp_path / dataset
    for seed, map_v, acc_v in (("42", "0.50", "0.30"), ("43", "0.60", "0.40")):
        for criterion, all_v in (("map", map_v), ("acc", acc_v)):
            dpath_best = dpath_dataset / seed / "evals" / "_best" / criterion
            dpath_best.mkdir(parents=True)
            for group_key in ("native", "native_macro", "joint", "joint_macro"):
                (dpath_best / f"{group_key}.json").write_text(json.dumps({
                    "scores": {"comp": {"map": {"all": all_v}}},
                    "loss_raw": {"id": "0.70", "ood": None},
                    "sim": {"mean": "0.0925"},
                    "targ": {"mean": "-0.9895"},
                    "chkpt": "1/1 (0.0M/0.0M samples)",
                }))

    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_metric_stats("std")

    stats = json.loads((dpath_dataset / "stats" / "map" / "native" / "metrics.json").read_text())
    assert stats["n_trials"] == 2
    assert stats["scores"]["comp"]["map"]["all"] == "55.00 ± 7.07"
    # only the scores tree is aggregated; the non-score fields are dropped
    assert set(stats) == {"n_trials", "scores"}

    listview_text = (dpath_dataset / "stats" / "map" / "native" / "metrics_listview.json").read_text()
    listview = json.loads(listview_text)
    assert listview["n_trials"] == 2
    assert set(listview) == {"n_trials", "scores"}
    assert listview["scores"]["comp"]["map"]["all"] == ["50.00", "60.00"]
    # each leaf list stays on a single line
    assert '"all": ["50.00", "60.00"]' in listview_text
    # the acc tree aggregates the acc-selected _best files, separately from map's
    stats_acc = json.loads((dpath_dataset / "stats" / "acc" / "native" / "metrics.json").read_text())
    assert stats_acc["scores"]["comp"]["map"]["all"] == "35.00 ± 7.07"
    # one aggregate + one listview file per criterion x eval group
    for criterion in ("map", "acc"):
        for group_key in ("native", "native_macro", "joint", "joint_macro"):
            assert (dpath_dataset / "stats" / criterion / group_key / "metrics.json").exists()
            assert (dpath_dataset / "stats" / criterion / group_key / "metrics_listview.json").exists()


_GROUP_KEYS = ("native", "native_macro", "joint", "joint_macro")
_SUPP_OFF = {"primitive": False, "n_shot": False}
_SUPP_PRIM = {"primitive": True, "n_shot": False}


def _write_trial_evals(dpath_trial, chkpt_scores, n_chkpts=None):
    """A trial's per-checkpoint eval files: chkpt_scores[i] = (comp.map.all, comp.acc.i2t) at
    checkpoint i, index 0 being the base eval. n_chkpts defaults to the last written index, i.e. a
    completed trial; pass a larger one to leave the trial short of its final eval."""
    n_chkpts = len(chkpt_scores) - 1 if n_chkpts is None else n_chkpts
    for idx_eval, (map_v, acc_v) in enumerate(chkpt_scores):
        dpath_chkpt = dpath_trial / "evals" / ("base" if idx_eval == 0 else f"eval{idx_eval}")
        dpath_chkpt.mkdir(parents=True)
        for group_key in _GROUP_KEYS:
            (dpath_chkpt / f"{group_key}.json").write_text(json.dumps({
                "scores": {"comp": {"map": {"all": map_v}, "acc": {"i2t": acc_v}}},
                "chkpt": f"{idx_eval}/{n_chkpts} (0.0M/0.0M samples)",
            }))


def test_update_chkpt_selection_picks_argmax_of_the_mean_curve(tmp_path, monkeypatch) -> None:
    # the setting picks ONE checkpoint index per criterion x group -- argmax over the across-trial
    # MEAN curve, not each trial's own argmax -- and every trial is scored there. Trial 42 peaks at
    # chkpt 1 and trial 43 at chkpt 3, but the mean peaks at 2, so BOTH are scored at chkpt 2.
    dataset = "cub"
    dpath_dataset = tmp_path / dataset
    (tmp_path / "setting_metadata.json").write_text(json.dumps({"n_crashes": {}, "best_chkpt": {}}))
    _write_trial_evals(dpath_dataset / "42", (("0.10", "0.10"), ("0.90", "0.90"), ("0.50", "0.50"), ("0.20", "0.20")))
    _write_trial_evals(dpath_dataset / "43", (("0.10", "0.10"), ("0.10", "0.10"), ("0.70", "0.70"), ("0.80", "0.80")))

    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_chkpt_selection("std")

    chkpt_means = load_pickle(dpath_dataset / "stats" / "map" / "native" / "chkpt_means.pkl")
    assert chkpt_means["n_trials"] == 2
    assert list(chkpt_means["chkpts"]) == [0, 1, 2, 3]  # the base eval leads the curve
    assert chkpt_means["means"] == pytest.approx([0.10, 0.50, 0.60, 0.50])
    assert chkpt_means["idx_best"] == 2  # argmax over 1.., earliest on ties

    # both trials scored at the SAME checkpoint -- neither trial's own argmax
    for seed, score in (("42", "0.50"), ("43", "0.70")):
        best = json.loads((dpath_dataset / seed / "evals" / "_best" / "map" / "native.json").read_text())
        assert best["chkpt"].startswith("2/3")
        assert best["scores"]["comp"]["map"]["all"] == score  # 42's own best was 0.90, 43's 0.80

    metadata = json.loads((tmp_path / "setting_metadata.json").read_text())
    assert metadata["best_chkpt"]["cub"]["map"]["native"] == {
        "idx": 2, "n_trials": 2, "mean": "0.6000",
    }
    for criterion in ("map", "acc"):
        for group_key in _GROUP_KEYS:
            assert (dpath_dataset / "stats" / criterion / group_key / "chkpt_means.pkl").exists()
            assert (dpath_dataset / "stats" / criterion / group_key / "chkpt_means.png").exists()
            assert metadata["best_chkpt"]["cub"][criterion][group_key]["idx"] == 2


def test_update_chkpt_selection_excludes_base_and_unfinished_trials(tmp_path, monkeypatch) -> None:
    # the base eval (index 0) is plotted but never selectable, and a trial short of its final eval
    # doesn't enter the mean at all (nor get a _best/): here 43 stopped at chkpt 1 of 2
    dataset = "cub"
    dpath_dataset = tmp_path / dataset
    (tmp_path / "setting_metadata.json").write_text(json.dumps({"n_crashes": {}, "best_chkpt": {}}))
    _write_trial_evals(dpath_dataset / "42", (("0.90", "0.90"), ("0.10", "0.10"), ("0.30", "0.30")))
    _write_trial_evals(dpath_dataset / "43", (("0.10", "0.10"), ("0.99", "0.99")), n_chkpts=2)

    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_chkpt_selection("std")

    chkpt_means = load_pickle(dpath_dataset / "stats" / "map" / "native" / "chkpt_means.pkl")
    assert chkpt_means["n_trials"] == 1  # only trial 42 counted
    assert chkpt_means["means"] == pytest.approx([0.90, 0.10, 0.30])  # 42's curve alone
    assert list(chkpt_means["spreads"]) == [0.0] * 3  # ddof=1 undefined for one trial -> flat band
    assert chkpt_means["idx_best"] == 2  # 0.90 at base is higher, but base can't win
    assert not (dpath_dataset / "43" / "evals" / "_best").exists()


def test_seed_sweep_complete_needs_every_setting_and_dataset(tmp_path, monkeypatch) -> None:
    # the campaign-level tables wait for a seed to have finished across the WHOLE matrix, since each
    # trial completion reselects only its own (setting, dataset)
    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["sp", "mp"], "datasets": ["cub", "lepid"]}))
    scores = (("0.10", "0.10"), ("0.30", "0.30"))
    for setting in ("sp", "mp"):
        for dataset in ("cub", "lepid"):
            _write_trial_evals(tmp_path / "settings" / setting / dataset / "42", scores)
    # seed 43 has run everywhere but mp/lepid, where it stopped short of its final eval
    for setting, dataset, n_chkpts in (("sp", "cub", None), ("sp", "lepid", None), ("mp", "cub", None), ("mp", "lepid", 2)):
        _write_trial_evals(tmp_path / "settings" / setting / dataset / "43", scores, n_chkpts=n_chkpts)

    assert report.seed_sweep_complete(42)
    assert not report.seed_sweep_complete(43)

    (tmp_path / "settings" / "mp" / "lepid" / "43" / "evals" / "eval2").mkdir()
    for group_key in _GROUP_KEYS:  # its final eval lands -> the sweep closes
        (tmp_path / "settings" / "mp" / "lepid" / "43" / "evals" / "eval2" / f"{group_key}.json").write_text(json.dumps({
            "scores": {"comp": {"map": {"all": "0.30"}, "acc": {"i2t": "0.30"}}},
            "chkpt": "2/2 (0.0M/0.0M samples)",
        }))

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


def _write_group_metrics(dpath_best, scores_grp: dict, macro: dict | None = None, acc_selected: dict | None = None) -> None:
    # trial-end materialization (report.update_chkpt_selection) writes one metrics file per eval
    # group under each selection criterion; fixtures reuse one subtree per averaging axis across
    # both sets, and the same content for both criteria unless acc_selected supplies the
    # acc-criterion subtree. A completed trial also always has its trial_metadata.json (hardware
    # readings) and its setting's setting_metadata.json (crash counters), which the always-on
    # 'Hardware Performance' sheet reads: placeholder ones are written here (the setting's only
    # if absent), for tests to overwrite when they assert on them.
    macro = scores_grp if macro is None else macro
    for criterion, (grp_std, grp_macro) in (("map", (scores_grp, macro)),
                                            ("acc", (acc_selected or scores_grp, acc_selected or macro))):
        (dpath_best / criterion).mkdir(parents=True, exist_ok=True)
        for group_key, grp in (("native", grp_std), ("native_macro", grp_macro), ("joint", grp_std), ("joint_macro", grp_macro)):
            (dpath_best / criterion / f"{group_key}.json").write_text(json.dumps({"scores": grp}))
    dpath_trial = dpath_best.parent.parent  # <setting>/<dataset>/<seed>/evals/_best
    (dpath_trial / "trial_metadata.json").write_text(json.dumps({
        "runtime": {"train": {"mean": "1.00"}, "eval": {"mean": "1.00"}, "trial": "1.00"},
        "memory": {"ram": "1.0/128.0 GB", "vram": "1.0/178.4 GB"},
    }))
    fpath_meta_setting = dpath_trial.parent.parent / "setting_metadata.json"
    if not fpath_meta_setting.exists():
        fpath_meta_setting.write_text(json.dumps({"n_crashes": {"ram": 0, "vram": 0, "other": 0}}))


def test_stats_table_grid_formats_by_trial_count() -> None:
    grid = report._stats_table_grid(
        ("All", "ID", "OOD", "I2T", "I2I", "T2I"),
        [
            ("hp", [_comp(0.50)["map"], _comp(0.60)["map"]]),
            ("mp", [_comp(0.50)["map"]]),
            ("sp", []),
        ],
        "std",
    )

    assert grid[0] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert [row[0] for row in grid[1:]] == ["hp (2)", "mp (1)", "sp (0)"]
    assert grid[1][1] == "55.00 ± 7.07"  # 2 trials: mean ± std
    assert grid[2][1] == "50.00"  # 1 trial: mean only
    assert grid[3][1] == "-"  # 0 trials
    assert grid[1][2] == "57.00 ± 7.07"  # "ID" column reads score key "id"


def test_stats_table_grid_ste_spread() -> None:
    grid = report._stats_table_grid(
        ("All",),
        [("hp", [_comp(0.50)["map"], _comp(0.60)["map"]])],
        "ste",
    )

    assert grid[1][1] == "55.00 ± 5.00"  # ste = std / sqrt(n): 7.07 / sqrt(2)


def test_stats_table_grid_single_acc_column() -> None:
    grid = report._stats_table_grid(
        ("I2T",),
        [("hp", [_comp(0.50)["acc"], _comp(0.60)["acc"]])],
        "std",
    )

    assert grid == [["Setting", "I2T"], ["hp (2)", "61.00 ± 7.07"]]


def test_update_stats_tables_writes_pngs(tmp_path, monkeypatch) -> None:
    # "mp" is planned in campaign_metadata.json but has no completed trials in this dataset, so it
    # gets no row (exclusion asserted in test_update_stats_tables_ordered_localized_per_metric);
    # trials are counted by their written best-checkpoint metrics, same as update_metric_stats.
    # bold_high=True + heatmap=True exercise the real matplotlib styling paths (winner bold +
    # heatmap shading) end-to-end.
    dataset = "cub"
    dpath_best = tmp_path / "settings" / "hp" / dataset / "42" / "evals" / "_best"
    dpath_best.mkdir(parents=True)
    _write_group_metrics(dpath_best, _scores_grp(_comp(0.50)))
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp", "mp"], "datasets": [dataset]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_stats_tables("std", True, False, True, _SUPP_OFF)

    for group_key in ("native", "native_macro", "joint", "joint_macro"):
        assert (tmp_path / "stats" / dataset / "map" / group_key / "metrics.png").exists()
        assert (tmp_path / "stats" / dataset / "acc" / group_key / "metrics.png").exists()


def _write_chkpt_means(dpath_campaign, setting, dataset, means, idx_best) -> None:
    """`setting`'s across-trial mean curve, the same artifact update_chkpt_selection writes, for
    every criterion x group."""
    for criterion in ("map", "acc"):
        for group_key in _GROUP_KEYS:
            dpath_group = dpath_campaign / "settings" / setting / dataset / "stats" / criterion / group_key
            dpath_group.mkdir(parents=True)
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


def test_update_convergence_plots_writes_pngs(tmp_path, monkeypatch) -> None:
    dataset = "cub"
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["a", "b"], "datasets": [dataset]}))
    _write_chkpt_means(tmp_path, "a", dataset, [0.10, 0.40, 0.30], 1)
    _write_chkpt_means(tmp_path, "b", dataset, [0.10, 0.20, 0.60], 2)

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    report.update_convergence_plots()

    for criterion in ("map", "acc"):
        for group_key in _GROUP_KEYS:
            assert (tmp_path / "stats" / dataset / criterion / group_key / "convergence.png").exists()


def test_update_convergence_plots_winner_is_highest_selected_mean(tmp_path, monkeypatch) -> None:
    # the winner is the setting with the highest mean at its OWN selected checkpoint ("b", 0.60 at
    # chkpt 2 -- "a" peaks earlier but lower); "c" never completed a trial here, so it has no
    # chkpt_means.pkl and doesn't make the plot at all
    dataset = "cub"
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["a", "b", "c"], "datasets": [dataset]}))
    _write_chkpt_means(tmp_path, "a", dataset, [0.10, 0.40, 0.30], 1)
    _write_chkpt_means(tmp_path, "b", dataset, [0.10, 0.20, 0.60], 2)

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)
    plotted = []
    monkeypatch.setattr(report, "_plot_convergence", lambda curves, idx_win, *a: plotted.append((curves, idx_win)))

    report.update_convergence_plots()

    curves, idx_win = plotted[0]
    assert [setting for setting, _, _ in curves] == ["a", "b"]
    assert curves[idx_win][0] == "b"


def test_update_stats_tables_ordered_localized_per_metric(tmp_path, monkeypatch) -> None:
    # ordered=True: each png orders setting rows by its own metric's means over ITS dataset's trials
    # only. The bryo data makes cub's local orders the opposite of the cross-dataset ones: cub-local
    # mAP gives a=60 > b=40 -> [a, b] (cross-dataset means 35 vs 40 would say [b, a]), and cub-local
    # acc gives b=80 > a=20 -> [b, a] (cross-dataset means 55 vs 45 would say [a, b]). "c" completed
    # only in bryo -> no row in cub's pngs (blank rows are xlsx-only).
    comp_vals = {  # (setting, dataset) -> (map "all", acc "i2t")
        ("a", "cub"): (0.60, "0.20"), ("a", "bryo"): (0.10, "0.90"),
        ("b", "cub"): (0.40, "0.80"), ("b", "bryo"): (0.40, "0.10"),
        ("c", "bryo"): (0.90, "0.90"),
    }
    for (setting, dataset), (all_v, acc_v) in comp_vals.items():
        dpath_best = tmp_path / "settings" / setting / dataset / "42" / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _scores_grp(_full_comp(all_v, acc_v)))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["a", "b", "c"], "datasets": ["cub", "bryo"]})
    )

    grids = []
    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", "cub")
    monkeypatch.setattr(
        report, "_render_stats_table",
        lambda grid, title, fpath, bold_high, heatmap: grids.append(grid),
    )

    report.update_stats_tables("std", False, True, False, _SUPP_OFF)

    assert len(grids) == 8  # map + acc per eval group
    grid_map, grid_acc = grids[0], grids[1]  # native pair (fixture repeats one subtree per group)
    assert grid_map[0] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert [r[0] for r in grid_map[1:]] == ["a (1)", "b (1)"]  # cub-local mAP order; no "c" row
    assert grid_map[1][1] == "60.00"
    assert grid_map[2][1] == "40.00"
    assert grid_acc[0] == ["Setting", "I2T"]
    assert grid_acc[1] == ["b (1)", "80.00"]  # cub-local acc order [b, a]
    assert grid_acc[2] == ["a (1)", "20.00"]
    assert len(grid_acc) == 3  # no "c" row here either


def test_update_metrics_xlsx_writes_stacked_tables(tmp_path, monkeypatch) -> None:
    # two datasets -> two stacked tables sharing the same setting rows (in campaign_metadata order).
    # "hp" has 2 cub trials (mean ± spread) and none in bryo -> a blank "-" row in the Bryozoa table;
    # "mp" has no completed trials anywhere -> no rows at all until its first trial completes.
    settings = ["hp", "mp"]
    for seed, base in (("42", 0.50), ("43", 0.60)):
        dpath_best = tmp_path / "settings" / "hp" / "cub" / seed / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _scores_grp(_comp(base)))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": settings, "datasets": ["cub", "bryo"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, False, False, _SUPP_OFF, False)

    fpath_xlsx = tmp_path / "stats" / "metrics" / "map" / "native.xlsx"
    assert fpath_xlsx.exists()
    wb = load_workbook(fpath_xlsx)
    ws = wb.active

    def rows_as_lists():
        return [[c.value for c in row] for row in ws.iter_rows()]

    grid = rows_as_lists()
    # campaign banner ("<parent-dir> - <campaign> (<eval group name>)") + blank row, then the dataset tables, then the
    # always-on cross-dataset mean table (one row per setting) at the bottom. "mp" (no completed
    # trials anywhere) gets no rows; "hp" completed only in cub, so the Bryozoa table still gets its
    # blank "-" row. mean cells are point values (no spread), unlike the per-dataset "± spread".
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.name} (Native; mAP-selection)"
    assert grid[1][:7] == [None] * 7  # blank row below the campaign banner
    assert grid[2][0] == "CUB"
    assert grid[3][:7] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[4][:2] == ["hp (2)", "55.00 ± 7.07"]  # hp: 2 trials
    assert grid[5][:7] == [None] * 7  # spacer row -- no "mp" row
    assert grid[6][0] == "Bryozoa"
    assert grid[7][:7] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[8][:2] == ["hp (0)", "-"]  # hp's blank entries still added for the trial-less dataset
    assert grid[10][0] == "Mean"  # cross-dataset mean table sits at the bottom
    assert grid[11][0] == "Setting"
    assert grid[12][:7] == ["hp", "55.00", "57.00", "56.00", "58.00", "59.00", "60.00"]
    assert not any(v in ("mp", "mp (0)") for r in grid for v in r)
    # per-seed blocks to the right, one blank separator column apart; "seed <seed>" labels sit in the
    # campaign-banner row; seed blocks have no Mean table and their dataset tables sit in the same
    # rows as the aggregate block's (dataset tables lead everywhere), with plain setting labels (no
    # counts), that seed's raw values, and "-" where the seed's trial hasn't completed
    assert grid[0][8] == "seed 42"
    assert grid[0][16] == "seed 43"
    assert grid[2][8] == "CUB"
    assert grid[3][8:15] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[4][8:15] == ["hp", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00"]   # seed 42 CUB, aligned with aggregate CUB
    assert grid[6][8] == "Bryozoa"
    assert grid[8][8:15] == ["hp", "-", "-", "-", "-", "-", "-"]                           # seed 42 Bryozoa
    assert grid[4][16:23] == ["hp", "60.00", "62.00", "61.00", "63.00", "64.00", "65.00"]  # seed 43 CUB
    assert grid[10][8] is None  # seed blocks have no Mean table (bottom band stays blank)
    assert not any(r[8] == "Mean" for r in grid)  # no trial-mean table in seed blocks
    assert all(r[7] is None and r[15] is None for r in grid)  # separator columns stay empty
    # snug column widths: longest header/data cell + 2; empty separator columns get a small fixed width
    assert ws.column_dimensions["A"].width == len("Setting") + 2
    assert ws.column_dimensions["B"].width == len("55.00 ± 7.07") + 2
    assert ws.column_dimensions["H"].width == 3
    # bold_high=False: data cells stay unbolded (only header row + setting column bold)
    assert ws.cell(row=5, column=2).font.bold is not True
    # heatmap=False: data cells are left unshaded
    assert ws.cell(row=5, column=2).fill.patternType is None
    # "All Borders": thin black gridlines on every table cell, incl. all cells of the merged title banner
    assert ws.cell(row=3, column=1).border.top.style == "thin"
    assert ws.cell(row=3, column=1).border.top.color.rgb[-6:] == "000000"
    assert ws.cell(row=3, column=7).border.right.color.rgb[-6:] == "000000"  # banner's far merged edge
    assert ws.cell(row=5, column=2).border.left.color.rgb[-6:] == "000000"  # data cell
    # campaign + table titles are left-aligned in their cells
    assert ws.cell(row=1, column=1).alignment.horizontal == "left"
    assert ws.cell(row=3, column=1).alignment.horizontal == "left"
    # setting names are left-aligned; the Setting header and score cells stay centered
    assert ws.cell(row=5, column=1).alignment.horizontal == "left"  # "hp (2)" (dataset table)
    assert ws.cell(row=13, column=1).alignment.horizontal == "left"  # "hp" (Mean table)
    assert ws.cell(row=4, column=1).alignment.horizontal == "center"  # "Setting" header
    assert ws.cell(row=5, column=2).alignment.horizontal == "center"  # score cell
    # 2nd sheet: the accuracy analog (single I2T column per table), same layout/row order.
    # hp's cub trials have acc i2t 56.00/66.00 -> mean 61.00 (± 7.07 in the per-dataset table).
    assert wb.sheetnames == ["Composite mAP", "Composite I2T Accuracy", "Hardware Performance"]
    agrid = [[c.value for c in row] for row in wb["Composite I2T Accuracy"].iter_rows()]
    assert agrid[0][0] == f"{paths['root'].parent.name} - {tmp_path.name} (Native; mAP-selection)"
    assert agrid[2][0] == "CUB"
    assert agrid[3][:2] == ["Setting", "I2T"]
    assert agrid[4][:2] == ["hp (2)", "61.00 ± 7.07"]
    assert agrid[6][0] == "Bryozoa"
    assert agrid[8][:2] == ["hp (0)", "-"]
    assert agrid[10][0] == "Mean"
    assert agrid[12][:2] == ["hp", "61.00"]
    # acc seed blocks (2-wide, so at cols D-E and G-H)
    assert agrid[0][3] == "seed 42"
    assert agrid[0][6] == "seed 43"
    assert agrid[2][3] == "CUB"
    assert agrid[4][3:5] == ["hp", "56.00"]   # seed 42 CUB, aligned with aggregate CUB
    assert agrid[4][6:8] == ["hp", "66.00"]   # seed 43 CUB
    assert agrid[8][3:5] == ["hp", "-"]       # seed 42 Bryozoa
    # 3rd sheet: the hardware analog, same layout/row order with the hw readings as columns (the
    # Mean table appends the crash totals); values asserted in test_update_metrics_xlsx_hw_sheet
    hgrid = [[c.value for c in row] for row in wb["Hardware Performance"].iter_rows()]
    assert hgrid[0][0] == f"{paths['root'].parent.name} - {tmp_path.name} (Native; mAP-selection)"
    assert hgrid[2][0] == "CUB"
    assert hgrid[3][:6] == ["Setting", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM"]
    assert hgrid[4][0] == "hp (2)"
    assert hgrid[6][0] == "Bryozoa"
    assert hgrid[8][:2] == ["hp (0)", "-"]
    assert hgrid[10][0] == "Mean"
    assert hgrid[11][:9] == ["Setting", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM",
                             "Total Crashes RAM", "Total Crashes VRAM", "Total Crashes Other"]
    assert hgrid[12][0] == "hp"
    assert hgrid[0][10] == "seed 42"  # one separator past the 9-wide Mean table (the block's widest)
    assert hgrid[4][10] == "hp"


def test_update_metrics_xlsx_bold_high(tmp_path, monkeypatch) -> None:
    # bold_high=True: the highest-mean setting cell in each score column is bolded. "hp" (base 0.60)
    # outranks "mp" (base 0.50) in every column, so hp's cells bold and mp's do not; "sp" completed
    # only in bryo, so its CUB row is blank "-" -- ignored and never bolded.
    settings = ["hp", "mp", "sp"]
    for setting, dataset, base in (("hp", "cub", 0.60), ("mp", "cub", 0.50), ("sp", "bryo", 0.10)):
        dpath_best = tmp_path / "settings" / setting / dataset / "42" / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _scores_grp(_comp(base)))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": settings, "datasets": ["cub", "bryo"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", True, False, False, _SUPP_OFF, False)

    ws = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx").active
    # campaign banner + blank row, then the CUB table first: banner row 3, header row 4, setting rows
    # 5/6/7 = hp/mp/sp; score cols B..G = All/ID/OOD/I2T/I2I/T2I (the Mean table sits at the bottom)
    assert ws.cell(row=3, column=1).value == "CUB"
    assert ws.cell(row=4, column=1).value == "Setting"
    for score_col in range(2, 8):
        assert ws.cell(row=5, column=score_col).font.bold is True       # hp wins -> bold
        assert ws.cell(row=6, column=score_col).font.bold is not True   # mp loses -> not bold
        assert ws.cell(row=7, column=score_col).value == "-"            # sp: no cub trials
        assert ws.cell(row=7, column=score_col).font.bold is not True   # "-" never bolds


def test_update_metrics_xlsx_per_group_files(tmp_path, monkeypatch) -> None:
    # one workbook per eval group under stats/metrics/, each reading its own comp map:
    # native_macro.xlsx <- the best-checkpoint native_macro.json, not the (different) standard values
    dpath_best = tmp_path / "settings" / "hp" / "cub" / "42" / "evals" / "_best"
    dpath_best.mkdir(parents=True)
    _write_group_metrics(
        dpath_best,
        _scores_grp(_comp(0.50)),        # standard All -> 50.00
        macro=_scores_grp(_comp(0.30)),  # macro All -> 30.00
    )
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["hp"], "datasets": ["cub"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, False, False, _SUPP_OFF, False)

    dpath_metrics = tmp_path / "stats" / "metrics"
    for criterion in ("map", "acc"):
        assert sorted(p.name for p in (dpath_metrics / criterion).glob("*.xlsx")) == [
            "joint.xlsx", "joint_macro.xlsx", "native.xlsx", "native_macro.xlsx"
        ]
    ws = load_workbook(dpath_metrics / "map" / "native.xlsx").active
    assert ws.cell(row=4, column=2).value == "All"
    assert ws.cell(row=5, column=1).value == "hp (1)"
    assert ws.cell(row=5, column=2).value == "50.00"
    ws_macro = load_workbook(dpath_metrics / "map" / "native_macro.xlsx").active
    assert ws_macro.cell(row=5, column=2).value == "30.00"  # macro, not standard's 50.00


def test_update_metrics_xlsx_criterion_sourcing(tmp_path, monkeypatch) -> None:
    # one workbook set per selection criterion: BOTH sheets of stats/metrics/<criterion>/ source
    # that criterion's best checkpoints -- the map/ workbook's accuracy sheet holds the acc scores
    # AT the best-mAP checkpoint (not the best-acc ones), and vice versa; banners name the selection
    dpath_best = tmp_path / "settings" / "hp" / "cub" / "42" / "evals" / "_best"
    dpath_best.mkdir(parents=True)
    _write_group_metrics(
        dpath_best,
        _scores_grp(_comp(0.50)),               # map-best checkpoint: mAP All 50.00, acc I2T 56.00
        acc_selected=_scores_grp(_comp(0.30)),  # acc-best checkpoint: mAP All 30.00, acc I2T 36.00
    )
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp"], "datasets": ["cub"]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, False, False, _SUPP_OFF, False)

    wb_map = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx")
    grid = [[c.value for c in r] for r in wb_map.active.iter_rows()]
    agrid = [[c.value for c in r] for r in wb_map["Composite I2T Accuracy"].iter_rows()]
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.name} (Native; mAP-selection)"
    assert grid[4][:2] == ["hp (1)", "50.00"]   # mAP at the map-best checkpoint
    assert agrid[4][:2] == ["hp (1)", "56.00"]  # acc at the map-best checkpoint

    wb_acc = load_workbook(tmp_path / "stats" / "metrics" / "acc" / "native.xlsx")
    grid = [[c.value for c in r] for r in wb_acc.active.iter_rows()]
    agrid = [[c.value for c in r] for r in wb_acc["Composite I2T Accuracy"].iter_rows()]
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.name} (Native; Acc-selection)"
    assert grid[4][:2] == ["hp (1)", "30.00"]   # mAP at the acc-best checkpoint
    assert agrid[4][:2] == ["hp (1)", "36.00"]  # acc at the acc-best checkpoint


def _full_comp(all_v: float, acc_v: str = "0.10") -> dict:
    # comp with controllable map "all" + acc "i2t"; other leaves fixed (irrelevant to ordering / heatmap rows)
    return {
        "acc": {"i2t": acc_v},
        "map": {"all": f"{all_v:.4f}", "id": "0.10", "ood": "0.10", "i2t": "0.10", "i2i": "0.10", "t2i": "0.10"},
    }


def test_update_metrics_xlsx_ordered_per_sheet_metric(tmp_path, monkeypatch) -> None:
    # ordered=True: each sheet orders its columns by its own metric's Mean-table first row, so the
    # two sheets may disagree. Campaign order is [a, b]; mAP mean-All gives a=mean(20,40)=30.00 <
    # b=mean(40,40)=40.00 -> mAP sheet flips to [b, a], while acc mean-I2T gives a=80.00 > b=20.00
    # -> accuracy sheet keeps [a, b].
    for setting, acc_v, cub_all, bryo_all in (("a", "0.80", 0.20, 0.40), ("b", "0.20", 0.40, 0.40)):
        for dataset, all_v in (("cub", cub_all), ("bryo", bryo_all)):
            dpath_best = tmp_path / "settings" / setting / dataset / "42" / "evals" / "_best"
            dpath_best.mkdir(parents=True)
            _write_group_metrics(dpath_best, _scores_grp(_full_comp(all_v, acc_v)))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["a", "b"], "datasets": ["cub", "bryo"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, True, False, _SUPP_OFF, False)

    wb = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx")
    grid = [[c.value for c in r] for r in wb.active.iter_rows()]
    # campaign banner + blank row, then the dataset tables (Mean at the bottom); setting rows ordered
    # by the mean table's "All" column -> b before a
    assert grid[2][0] == "CUB"
    assert grid[3][:7] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[4][:2] == ["b (1)", "40.00"]
    assert grid[5][:2] == ["a (1)", "20.00"]
    assert grid[7][0] == "Bryozoa"
    assert grid[9][:2] == ["b (1)", "40.00"]
    assert grid[10][:2] == ["a (1)", "40.00"]
    assert grid[12][0] == "Mean"
    assert grid[14][:2] == ["b", "40.00"]
    assert grid[15][:2] == ["a", "30.00"]
    # the seed block's rows are pinned to the sheet's mean-derived order too (CUB table, aligned rows)
    assert grid[0][8] == "seed 42"
    assert [grid[4][8], grid[5][8]] == ["b", "a"]
    # the accuracy sheet orders by its own acc mean-'I2T' column -> [a, b], unlike the mAP sheet
    agrid = [[c.value for c in r] for r in wb["Composite I2T Accuracy"].iter_rows()]
    assert agrid[3][:2] == ["Setting", "I2T"]
    assert agrid[4][:2] == ["a (1)", "80.00"]
    assert agrid[5][:2] == ["b (1)", "20.00"]
    assert agrid[12][0] == "Mean"
    assert agrid[14][:2] == ["a", "80.00"]
    assert agrid[15][:2] == ["b", "20.00"]
    assert agrid[4][3:5] == ["a", "80.00"]  # acc seed block keeps the acc sheet's [a, b] order (aligned rows)


def _fill_rgb(ws, row, col):
    # last 6 hex chars (RGB) of a cell's fill, or None when the cell is unshaded
    fill = ws.cell(row=row, column=col).fill
    return None if fill.patternType is None else fill.fgColor.rgb[-6:]


def test_update_metrics_xlsx_heatmap(tmp_path, monkeypatch) -> None:
    # heatmap=True: value/100 maps to white->#ff5533 over a fixed range, regardless of the column's
    # other cells; 20/50/80 -> #ffddd6 / #ffaa99 / #ff775c. One dataset, so the Mean "All" column
    # mirrors the values and is shaded too; "d" (no completed trials anywhere) gets no row at all.
    for setting, all_v in (("a", 0.20), ("b", 0.50), ("c", 0.80)):
        dpath_best = tmp_path / "settings" / setting / "cub" / "42" / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _scores_grp(_full_comp(all_v)))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["a", "b", "c", "d"], "datasets": ["cub"]})  # "d" has no trials
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, False, True, _SUPP_OFF, False)

    ws = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx").active
    # campaign banner + blank row; CUB table first: banner row 3, header row 4, "All" column is col B,
    # setting rows 5/6/7 = a/b/c
    assert ws.cell(row=8, column=1).value is None  # spacer right after c -> no "d" row
    assert _fill_rgb(ws, 5, 2) == "FFDDD6"  # 20 -> t=0.20
    assert _fill_rgb(ws, 6, 2) == "FFAA99"  # 50 -> t=0.50
    assert _fill_rgb(ws, 7, 2) == "FF775C"  # 80 -> t=0.80
    # the trailing Mean table is shaded too (setting rows 11/12/13)
    assert _fill_rgb(ws, 11, 2) == "FFDDD6"
    assert _fill_rgb(ws, 13, 2) == "FF775C"


def _prim_scores_grp() -> dict:
    # _scores_grp with distinct per-partition primitive values (comp base 0.50)
    grp = _scores_grp(_comp(0.50))
    grp["id"] = {"map": {"i2t": "0.61", "i2i": "0.62", "t2i": "0.63"}, "acc": {"i2t": "0.64"}}
    grp["ood"] = {"map": {"i2t": "0.71", "i2i": "0.72", "t2i": "0.73"}, "acc": {"i2t": "0.74"}}
    return grp


_PRIM_MAP_HEADER = ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I",
                    "ID I2T", "ID I2I", "ID T2I", "OOD I2T", "OOD I2I", "OOD T2I"]


def test_update_metrics_xlsx_supp_primitive(tmp_path, monkeypatch) -> None:
    # supp_scores.primitive appends the per-partition primitive score columns: ID/OOD x I2T/I2I/T2I
    # on the mAP sheet, ID I2T / OOD I2T on the accuracy sheet -- in every table, incl. the seed blocks
    dpath_best = tmp_path / "settings" / "hp" / "cub" / "42" / "evals" / "_best"
    dpath_best.mkdir(parents=True)
    _write_group_metrics(dpath_best, _prim_scores_grp())
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp"], "datasets": ["cub"]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, False, False, _SUPP_PRIM, False)

    wb = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx")
    ws = wb.active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    # mAP banners split: unmerged title + grey merged 'Composite Scores'/'Primitive Scores' group headers
    assert grid[2][:2] == ["CUB", "Composite Scores"]
    assert grid[2][7] == "Primitive Scores"
    assert grid[6][:2] == ["Mean", "Composite Scores"]
    assert grid[6][7] == "Primitive Scores"
    assert "B3:G3" in {str(m) for m in ws.merged_cells.ranges} and "H3:M3" in {str(m) for m in ws.merged_cells.ranges}
    assert ws.cell(row=3, column=2).fill.fgColor.rgb[-6:] == "EAEAEA"  # group headers get the header grey
    assert ws.cell(row=3, column=1).fill.patternType is None           # title cell stays unfilled
    assert grid[3][:13] == _PRIM_MAP_HEADER
    assert grid[4][:13] == ["hp (1)", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00",
                            "61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]  # CUB row
    assert grid[8][7:13] == ["61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]  # Mean row (bottom)
    # seed block starts after the 13-wide aggregate + separator; its CUB table row-aligns with the aggregate's
    assert grid[0][14] == "seed 42"
    assert grid[2][14:16] == ["CUB", "Composite Scores"]
    assert grid[2][21] == "Primitive Scores"
    assert grid[3][14:27] == _PRIM_MAP_HEADER
    assert grid[4][21:27] == ["61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]
    # the accuracy sheet keeps full-width merged title banners (no group headers)
    ws_acc = wb["Composite I2T Accuracy"]
    agrid = [[c.value for c in r] for r in ws_acc.iter_rows()]
    assert agrid[2][:2] == ["CUB", None]
    assert "A3:D3" in {str(m) for m in ws_acc.merged_cells.ranges}
    assert agrid[3][:4] == ["Setting", "I2T", "ID I2T", "OOD I2T"]
    assert agrid[4][:4] == ["hp (1)", "56.00", "64.00", "74.00"]  # CUB row
    assert agrid[8][:4] == ["hp", "56.00", "64.00", "74.00"]  # Mean row (bottom)
    assert agrid[0][5] == "seed 42"


def test_update_stats_tables_supp_primitive(tmp_path, monkeypatch) -> None:
    # supp_scores.primitive appends the per-partition primitive score columns to the png grids too
    dpath_best = tmp_path / "settings" / "hp" / "cub" / "42" / "evals" / "_best"
    dpath_best.mkdir(parents=True)
    _write_group_metrics(dpath_best, _prim_scores_grp())
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp"], "datasets": ["cub"]}))

    grids = []
    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", "cub")
    monkeypatch.setattr(
        report, "_render_stats_table",
        lambda grid, title, fpath, bold_high, heatmap: grids.append(grid),
    )

    report.update_stats_tables("std", False, False, False, _SUPP_PRIM)

    assert len(grids) == 8  # map + acc per eval group
    grid_map, grid_acc = grids[0], grids[1]  # native pair (fixture repeats one subtree per group)
    assert grid_map[0] == _PRIM_MAP_HEADER
    assert grid_map[1] == ["hp (1)", "50.00", "52.00", "51.00", "53.00", "54.00", "55.00",
                           "61.00", "62.00", "63.00", "71.00", "72.00", "73.00"]
    assert grid_acc[0] == ["Setting", "I2T", "ID I2T", "OOD I2T"]
    assert grid_acc[1] == ["hp (1)", "56.00", "64.00", "74.00"]


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


def test_update_metrics_xlsx_supp_n_shot(tmp_path, monkeypatch) -> None:
    # supp_scores.n_shot appends one column per ID-partition n-shot bucket to the right of every
    # table (after the primitive columns when both are on): the bucket's composite mAP on the mAP
    # sheet (under an 'N-Shot Scores' group header), its I2T accuracy on the accuracy sheet. Bucket
    # names come from the eval files, merged in their order: bryo (first in campaign order) lacks
    # few-shot (no classes there, as on the dev split), cub has all three -> [few, med, many]; the
    # absent bucket renders "-" in bryo's rows and is left out of the Mean (cub's value alone).
    for dataset, nshot in (("bryo", _NSHOT_NO_FEW), ("cub", _NSHOT_FULL)):
        dpath_best = tmp_path / "settings" / "hp" / dataset / "42" / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _nshot_scores_grp(*nshot))
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp"], "datasets": ["bryo", "cub"]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, False, False, {"primitive": True, "n_shot": True}, False)

    wb = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx")
    ws = wb.active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    merged = {str(m) for m in ws.merged_cells.ranges}
    # 16-wide tables: Setting + 6 comp + 6 prim + 3 n-shot; three group headers over the banner
    assert grid[2][:2] == ["Bryozoa", "Composite Scores"]
    assert grid[2][7] == "Primitive Scores"
    assert grid[2][13] == "N-Shot Scores"
    assert "B3:G3" in merged and "H3:M3" in merged and "N3:P3" in merged
    assert grid[3][:16] == _PRIM_MAP_HEADER + ["few-shot", "med-shot", "many-shot"]
    assert grid[4][13:16] == ["-", "52.00", "53.00"]  # Bryozoa: no few-shot bucket
    assert grid[6][0] == "CUB"
    assert grid[8][13:16] == ["31.00", "32.00", "33.00"]
    assert grid[10][0] == "Mean"
    assert grid[11][13:16] == ["few-shot", "med-shot", "many-shot"]
    assert grid[12][13:16] == ["31.00", "42.00", "43.00"]  # few-shot: cub alone; others mean bryo/cub
    # seed block (one separator past the 16-wide aggregate) carries the columns too
    assert grid[0][17] == "seed 42"
    assert grid[3][30:33] == ["few-shot", "med-shot", "many-shot"]
    assert grid[4][30:33] == ["-", "52.00", "53.00"]
    assert grid[8][30:33] == ["31.00", "32.00", "33.00"]
    # accuracy sheet: the buckets' I2T accuracies, full-width banner as before
    ws_acc = wb["Composite I2T Accuracy"]
    agrid = [[c.value for c in r] for r in ws_acc.iter_rows()]
    assert "A3:G3" in {str(m) for m in ws_acc.merged_cells.ranges}
    assert agrid[3][:7] == ["Setting", "I2T", "ID I2T", "OOD I2T", "few-shot", "med-shot", "many-shot"]
    assert agrid[4][4:7] == ["-", "62.00", "63.00"]
    assert agrid[8][4:7] == ["41.00", "42.00", "43.00"]
    assert agrid[12][4:7] == ["41.00", "52.00", "53.00"]

    # n_shot alone: the bucket columns follow the composite ones directly, with just the two groups
    report.update_metrics_xlsx("std", False, False, False, {"primitive": False, "n_shot": True}, False)

    ws = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx").active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    assert grid[2][:2] == ["Bryozoa", "Composite Scores"]
    assert grid[2][7] == "N-Shot Scores"
    assert "H3:J3" in {str(m) for m in ws.merged_cells.ranges}
    assert grid[3][:10] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I", "few-shot", "med-shot", "many-shot"]
    assert grid[8][7:10] == ["31.00", "32.00", "33.00"]


def test_update_stats_tables_supp_n_shot(tmp_path, monkeypatch) -> None:
    # supp_scores.n_shot appends the bucket columns to the png grids too
    dpath_best = tmp_path / "settings" / "hp" / "cub" / "42" / "evals" / "_best"
    dpath_best.mkdir(parents=True)
    _write_group_metrics(dpath_best, _nshot_scores_grp(*_NSHOT_FULL))
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp"], "datasets": ["cub"]}))

    grids = []
    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", "cub")
    monkeypatch.setattr(
        report, "_render_stats_table",
        lambda grid, title, fpath, bold_high, heatmap: grids.append(grid),
    )

    report.update_stats_tables("std", False, False, False, {"primitive": False, "n_shot": True})

    grid_map, grid_acc = grids[0], grids[1]
    assert grid_map[0] == ["Setting", "All", "ID", "OOD", "I2T", "I2I", "T2I", "few-shot", "med-shot", "many-shot"]
    assert grid_map[1][7:] == ["31.00", "32.00", "33.00"]
    assert grid_acc[0] == ["Setting", "I2T", "few-shot", "med-shot", "many-shot"]
    assert grid_acc[1][2:] == ["41.00", "42.00", "43.00"]


def test_update_metrics_xlsx_baseline_overrides(tmp_path, monkeypatch) -> None:
    # baseline_overrides=True renders a "Baseline Overrides" config table in a left column band that
    # the score blocks shift right past, aligned with the aggregate block's bottom Mean table (whose
    # Setting column labels its rows): one column per overridden param (union of the settings'
    # overrides.json keys, first-seen order), values resolved from each setting's
    # config.json -- "-" when the param is absent there (inert under that config: mp has
    # loss2.mix 0.0, so clean_metadata dropped its loss2 subtree). loss.targ resolves to "mp" for
    # BOTH settings, so its column is omitted (uniform columns differentiate nothing). Config cells
    # get no winner-bold/heatmap styling despite bold_high/heatmap on.
    for setting, base, overrides, meta in (
        ("hp", 0.50, {"loss2.mix": 0.3, "loss2.targ": "phylo"},
         {"loss": {"targ": "mp"}, "loss2": {"mix": 0.3, "targ": "phylo"}}),
        ("mp", 0.40, {"loss.targ": "mp"},
         {"loss": {"targ": "mp"}}),
    ):
        dpath_setting = tmp_path / "settings" / setting
        dpath_best = dpath_setting / "cub" / "42" / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _scores_grp(_comp(base)))
        (dpath_setting / "overrides.json").write_text(json.dumps(overrides))
        (dpath_setting / "config.json").write_text(json.dumps({**meta}))
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp", "mp"], "datasets": ["cub"]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", True, False, True, _SUPP_OFF, True)

    wb = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx")
    ws = wb.active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    # score blocks shift right past the 2-wide overrides band + separator col C: aggregate at D..J
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.name} (Native; mAP-selection)"  # campaign banner stays top-left
    assert grid[2][3] == "CUB"
    assert grid[7][3] == "Mean"
    # overrides table in the left band, aligned with the Mean table; param cols in first-seen order
    # (hp's overrides, then mp's); no Setting column of its own; uniform loss.targ column omitted
    assert grid[7][0] == "Baseline Overrides"
    assert "A8:B8" in {str(m) for m in ws.merged_cells.ranges}
    assert grid[8][:2] == ["loss2.mix", "loss2.targ"]
    assert not any(v == "loss.targ" for r in grid for v in r)
    assert grid[8][3:5] == ["Setting", "All"]  # Mean header shares the row
    assert grid[9][:2] == ["0.3", "phylo"]
    assert grid[9][3] == "hp"  # labeled by the Mean's Setting column
    assert grid[10][:2] == ["-", "-"]  # loss2.* inert for mp -> "-"
    assert grid[10][3] == "mp"
    # param-name header styled like other headers; config cells skip score styling entirely
    assert ws.cell(row=9, column=1).fill.fgColor.rgb[-6:] == "EAEAEA"
    assert ws.cell(row=9, column=1).font.bold is True
    assert ws.cell(row=10, column=1).fill.patternType is None
    assert ws.cell(row=10, column=1).font.bold is not True
    assert all(r[2] is None for r in grid)  # separator col C stays empty
    # seed block one separator past the aggregate score block (cols D..J)
    assert grid[0][11] == "seed 42"
    assert grid[2][11] == "CUB"
    # same treatment on the accuracy sheet (2-wide score tables at D..E, seed at col H)
    ws_acc = wb["Composite I2T Accuracy"]
    agrid = [[c.value for c in r] for r in ws_acc.iter_rows()]
    assert agrid[2][3] == "CUB"
    assert agrid[7][3] == "Mean"
    assert agrid[7][0] == "Baseline Overrides"
    assert "A8:B8" in {str(m) for m in ws_acc.merged_cells.ranges}
    assert agrid[8][:2] == ["loss2.mix", "loss2.targ"]
    assert agrid[9][:2] == ["0.3", "phylo"]
    assert agrid[10][:2] == ["-", "-"]
    assert agrid[0][6] == "seed 42"


def test_update_metrics_xlsx_baseline_overrides_all_uniform_omits_table(tmp_path, monkeypatch) -> None:
    # every overridden param resolves to the same value for every setting -> no column survives, so
    # the Baseline Overrides band is omitted entirely and the score blocks sit leftmost
    for setting, base in (("hp", 0.50), ("mp", 0.40)):
        dpath_setting = tmp_path / "settings" / setting
        dpath_best = dpath_setting / "cub" / "42" / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _scores_grp(_comp(base)))
        (dpath_setting / "overrides.json").write_text(json.dumps({"loss.targ": "mp"}))
        (dpath_setting / "config.json").write_text(json.dumps({"loss": {"targ": "mp"}}))
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp", "mp"], "datasets": ["cub"]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", False, False, False, _SUPP_OFF, True)

    wb = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx")
    grid = [[c.value for c in r] for r in wb.active.iter_rows()]
    assert not any(v == "Baseline Overrides" for r in grid for v in r)
    assert grid[2][0] == "CUB"  # score blocks leftmost: no band, no separator column


def test_update_metrics_xlsx_hw_sheet(tmp_path, monkeypatch) -> None:
    # the always-on 3rd sheet, "Hardware Performance", mirrors the score sheets' layout (campaign
    # banner, per-dataset tables + Mean table, per-seed blocks, Baseline Overrides band, mAP-sheet
    # row order) with per-trial readings from trial_metadata.json as columns, meaned over the same
    # trials as the score tables, rounded to the nearest int. Dataset tables mean that dataset's
    # completed trials ("<setting> (n)" labels, "-" row where a setting has none), seed-block
    # tables carry that seed's single trial, and the Mean table means the per-dataset trial means
    # across datasets -- hp's cub trial times (100.4, 200.4) mean to 150.4, then with bryo's 350.0
    # -> 250.2 -> "250" (a pooled per-trial mean would give 217: the two-level aggregation
    # matters) -- plus the Total Crashes RAM/VRAM/Other columns (Mean table only, crash totals
    # don't decompose per dataset/seed) straight from setting_metadata.json's n_crashes
    # (per-setting totals across seeds + datasets). Hardware cells get no winner-bold/heatmap
    # styling despite bold_high/heatmap on; the score sheets carry no hardware tables.
    hw_vals = {  # (setting, dataset, seed) -> (trial, train mean, eval mean, ram, vram)
        ("hp", "cub", "42"): ("100.40", "10.10", "5.10", "100.2/128.0 GB", "20.2/178.4 GB"),
        ("hp", "cub", "43"): ("200.40", "20.10", "7.10", "110.2/128.0 GB", "24.2/178.4 GB"),
        ("hp", "bryo", "42"): ("350.00", "30.10", "9.10", "120.2/128.0 GB", "30.2/178.4 GB"),
        ("mp", "cub", "42"): ("63.49", "7.70", "6.49", "117.2/128.0 GB", "26.3/178.4 GB"),
    }
    for (setting, dataset, seed), (trial_t, train_t, eval_t, ram, vram) in hw_vals.items():
        dpath_trial = tmp_path / "settings" / setting / dataset / seed
        dpath_best = dpath_trial / "evals" / "_best"
        dpath_best.mkdir(parents=True)
        _write_group_metrics(dpath_best, _scores_grp(_comp(0.50)))
        (dpath_trial / "trial_metadata.json").write_text(json.dumps({
            "runtime": {"train": {"mean": train_t}, "eval": {"mean": eval_t}, "trial": trial_t},
            "memory": {"ram": ram, "vram": vram},
        }))
    for setting, overrides, meta, crashes in (
        ("hp", {"loss2.mix": 0.3}, {"loss2": {"mix": 0.3}}, {"ram": 2, "vram": 1, "other": 0}),
        ("mp", {"loss.targ": "mp"}, {"loss": {"targ": "mp"}}, {"ram": 0, "vram": 0, "other": 3}),
    ):
        (tmp_path / "settings" / setting / "overrides.json").write_text(json.dumps(overrides))
        (tmp_path / "settings" / setting / "config.json").write_text(json.dumps({**meta}))
        (tmp_path / "settings" / setting / "setting_metadata.json").write_text(json.dumps({"n_crashes": crashes}))
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp", "mp"], "datasets": ["cub", "bryo"]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    report.update_metrics_xlsx("std", True, False, True, _SUPP_OFF, True)

    wb = load_workbook(tmp_path / "stats" / "metrics" / "map" / "native.xlsx")
    assert wb.sheetnames == ["Composite mAP", "Composite I2T Accuracy", "Hardware Performance"]
    ws = wb["Hardware Performance"]
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    merged = {str(m) for m in ws.merged_cells.ranges}
    # overrides band at A..B + separator C; aggregate block at D -- dataset tables 6 wide (D..I),
    # the Mean table 9 (D..L, crash columns appended), so the block spans D..L and seed 42 starts
    # one separator later at N
    assert grid[0][0] == f"{paths['root'].parent.name} - {tmp_path.name} (Native; mAP-selection)"
    assert grid[12][0] == "Baseline Overrides"
    assert "A13:B13" in merged
    assert grid[13][:2] == ["loss2.mix", "loss.targ"]
    assert grid[14][:2] == ["0.3", "-"]
    assert grid[15][:2] == ["-", "mp"]
    # CUB table: merged title banner, Setting + hw header, '<setting> (n)' labels; hp means its 2
    # cub trials, mp its 1
    assert grid[2][3] == "CUB"
    assert "D3:I3" in merged
    assert grid[3][3:9] == ["Setting", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM"]
    assert grid[4][3:9] == ["hp (2)", "150", "15", "6", "105", "22"]
    assert grid[5][3:9] == ["mp (1)", "63", "8", "6", "117", "26"]
    # Bryozoa table: hp's single trial passes through; mp has no bryo trials -> "-" row
    assert grid[7][3] == "Bryozoa"
    assert grid[9][3:9] == ["hp (1)", "350", "30", "9", "120", "30"]
    assert grid[10][3:9] == ["mp (0)", "-", "-", "-", "-", "-"]
    # Mean table: cross-dataset means of the per-dataset trial means + the crash-total columns
    assert grid[12][3] == "Mean"
    assert "D13:L13" in merged
    assert grid[13][3:12] == ["Setting", "Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM",
                              "Total Crashes RAM", "Total Crashes VRAM", "Total Crashes Other"]
    assert grid[14][3:12] == ["hp", "250", "23", "8", "113", "26", "2", "1", "0"]
    # mp: single cub trial, values pass straight through the two-level mean before rounding
    assert grid[15][3:12] == ["mp", "63", "8", "6", "117", "26", "0", "0", "3"]
    # seed blocks: per-dataset tables only (no Mean), plain labels; seed 42 at N..S, seed 43 at U..Z
    assert grid[0][13] == "seed 42"
    assert grid[0][20] == "seed 43"
    assert grid[2][13] == "CUB"
    assert grid[4][13:19] == ["hp", "100", "10", "5", "100", "20"]   # seed 42 CUB, hp's 42 trial alone
    assert grid[5][13:19] == ["mp", "63", "8", "6", "117", "26"]     # mp's only trial
    assert grid[9][13:19] == ["hp", "350", "30", "9", "120", "30"]   # seed 42 Bryozoa
    assert grid[10][13:19] == ["mp", "-", "-", "-", "-", "-"]        # mp: no bryo trial
    assert grid[12][13] is None  # no Mean table in seed blocks
    assert grid[4][20:26] == ["hp", "200", "20", "7", "110", "24"]   # seed 43 CUB, hp's 43 trial alone
    assert grid[5][20:26] == ["mp", "-", "-", "-", "-", "-"]         # mp has no 43 trial
    # separator columns between the band and blocks stay empty
    assert all(r[2] is None and r[12] is None and r[19] is None for r in grid)
    # header + setting cells styled like the score sheets' (setting names left-aligned); value
    # cells get no winner-bold/heatmap styling despite bold_high/heatmap on
    assert ws.cell(row=4, column=5).font.bold is True
    assert ws.cell(row=4, column=5).fill.fgColor.rgb[-6:] == "EAEAEA"
    assert ws.cell(row=5, column=4).font.bold is True
    assert ws.cell(row=5, column=4).alignment.horizontal == "left"
    assert ws.cell(row=6, column=5).font.bold is not True  # hp's 150 would be the Time Trial "winner"
    assert ws.cell(row=5, column=5).font.bold is not True
    assert ws.cell(row=5, column=5).fill.patternType is None
    # the score sheets carry no hardware tables; their overrides bands stay leftmost
    for sheet in ("Composite mAP", "Composite I2T Accuracy"):
        sgrid = [[c.value for c in r] for r in wb[sheet].iter_rows()]
        assert sgrid[12][0] == "Baseline Overrides"
        assert sgrid[12][3] == "Mean"
        assert not any(v in ("Hardware Performance", "Time Trial") for r in sgrid for v in r)


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
