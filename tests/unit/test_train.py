import json

from utils.train import ArtifactManager


def test_aggregate_metric_stats_keeps_none_leaf_across_trials() -> None:
    # loss_raw["ood"] is never computed -> None in every trial's metrics.json. Aggregating across
    # >1 completed trial must keep it None, not attempt float(None).
    trials = [
        {"scores": {"comp": {"map": {"all": "0.50"}}}, "loss_raw": {"id": "0.7029", "ood": None}},
        {"scores": {"comp": {"map": {"all": "0.60"}}}, "loss_raw": {"id": "0.7005", "ood": None}},
    ]

    out = ArtifactManager._aggregate_metric_stats(trials, "std")

    assert out["loss_raw"]["ood"] is None
    assert out["loss_raw"]["id"] == "0.7017 ± 0.0017"
    assert out["scores"]["comp"]["map"]["all"] == "55.00 ± 7.07"


def test_aggregate_metric_stats_ste_spread() -> None:
    # ste = std / sqrt(n): 7.07 / sqrt(2) = 5.00
    trials = [
        {"scores": {"comp": {"map": {"all": "0.50"}}}},
        {"scores": {"comp": {"map": {"all": "0.60"}}}},
    ]

    out = ArtifactManager._aggregate_metric_stats(trials, "ste")

    assert out["scores"]["comp"]["map"]["all"] == "55.00 ± 5.00"


def test_aggregate_metric_stats_single_trial_returns_leaves_verbatim() -> None:
    trials = [{"loss_raw": {"id": "0.7029", "ood": None}}]

    out = ArtifactManager._aggregate_metric_stats(trials, "std")

    assert out == {"loss_raw": {"id": "0.7029", "ood": None}}


def test_update_metric_stats_counts_trials_lacking_complete_flag(tmp_path, monkeypatch) -> None:
    # completion is now marked by the orchestrator after stats run, so update_metric_stats must aggregate
    # trials by their written final-eval metrics -- not by a `complete` flag that isn't set yet
    dataset = "cub"
    dpath_dataset = tmp_path / dataset
    for seed, all_v in (("42", "0.50"), ("43", "0.60")):
        dpath_final = dpath_dataset / seed / "evals" / "final"
        dpath_final.mkdir(parents=True)
        (dpath_final / "metrics.json").write_text(json.dumps({
            "scores": {"comp": {"map": {"all": all_v}}},
            "loss_raw": {"id": "0.70", "ood": None},
            "n_samps_seen": "100/100",
        }))

    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    ArtifactManager.update_metric_stats("std")

    stats = json.loads((dpath_dataset / "stats" / "metrics.json").read_text())
    assert stats["n_trials"] == 2
    assert stats["loss_raw"]["ood"] is None
    assert stats["scores"]["comp"]["map"]["all"] == "55.00 ± 7.07"

    listview_text = (dpath_dataset / "stats" / "metrics_listview.json").read_text()
    listview = json.loads(listview_text)
    assert listview["n_trials"] == 2
    assert listview["loss_raw"]["ood"] is None
    assert listview["loss_raw"]["id"] == ["0.7000", "0.7000"]
    assert listview["scores"]["comp"]["map"]["all"] == ["50.00", "60.00"]
    # each leaf list stays on a single line
    assert '"all": ["50.00", "60.00"]' in listview_text


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


def test_stats_table_grid_formats_by_trial_count() -> None:
    grid = ArtifactManager._stats_table_grid(
        "Composite mAP",
        ("All", "OOD", "ID", "I2T", "I2I", "T2I"),
        [
            ("hp", [_comp(0.50)["map"], _comp(0.60)["map"]]),
            ("sw", [_comp(0.50)["map"]]),
            ("iw", []),
        ],
        "std",
    )

    assert grid[0] == ["Composite mAP", "hp (2)", "sw (1)", "iw (0)"]
    assert [row[0] for row in grid[1:]] == ["All", "OOD", "ID", "I2T", "I2I", "T2I"]
    assert grid[1][1] == "55.00 ± 7.07"  # 2 trials: mean ± std
    assert grid[1][2] == "50.00"  # 1 trial: mean only
    assert grid[1][3] == "-"  # 0 trials
    assert grid[2][1] == "56.00 ± 7.07"  # "OOD" row reads score key "ood"


def test_stats_table_grid_ste_spread() -> None:
    grid = ArtifactManager._stats_table_grid(
        "Composite mAP",
        ("All",),
        [("hp", [_comp(0.50)["map"], _comp(0.60)["map"]])],
        "ste",
    )

    assert grid[1][1] == "55.00 ± 5.00"  # ste = std / sqrt(n): 7.07 / sqrt(2)


def test_stats_table_grid_single_acc_row() -> None:
    grid = ArtifactManager._stats_table_grid(
        "Composite I2T Accuracy",
        ("I2T",),
        [("hp", [_comp(0.50)["acc"], _comp(0.60)["acc"]])],
        "std",
    )

    assert grid == [["Composite I2T Accuracy", "hp (2)"], ["I2T", "61.00 ± 7.07"]]


def test_update_stats_tables_writes_pngs(tmp_path, monkeypatch) -> None:
    # "sw" is planned in campaign_metadata.json but has no trial dirs yet -- it must still get a
    # column; trials are counted by their written final-eval metrics, same as update_metric_stats
    dataset = "cub"
    dpath_final = tmp_path / "settings" / "hp" / dataset / "42" / "evals" / "final"
    dpath_final.mkdir(parents=True)
    (dpath_final / "metrics.json").write_text(json.dumps({
        "scores": {"closed_set": {"standard": {"comp": _comp(0.50)}}},
    }))
    (tmp_path / "campaign_metadata.json").write_text(json.dumps({"settings": ["hp", "sw"]}))

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)
    monkeypatch.setattr(ArtifactManager, "dataset", dataset)

    ArtifactManager.update_stats_tables("closed_standard", "std")

    assert (tmp_path / "stats" / dataset / "map.png").exists()
    assert (tmp_path / "stats" / dataset / "acc.png").exists()
