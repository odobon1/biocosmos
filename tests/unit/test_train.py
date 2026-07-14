import json

from openpyxl import load_workbook

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


def test_update_metrics_xlsx_writes_stacked_tables(tmp_path, monkeypatch) -> None:
    # two datasets -> two stacked tables sharing the same setting columns (in campaign_metadata order),
    # so a downstream re-sort keeps columns aligned. "cub" has 2 trials for hp (mean ± spread) and 0 for
    # sw (a "-" column); "bryo" has none at all.
    settings = ["hp", "sw"]
    for seed, base in (("42", 0.50), ("43", 0.60)):
        dpath_final = tmp_path / "settings" / "hp" / "cub" / seed / "evals" / "final"
        dpath_final.mkdir(parents=True)
        (dpath_final / "metrics.json").write_text(json.dumps({
            "scores": {"closed_set": {"standard": {"comp": _comp(base)}}},
        }))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": settings, "datasets": ["cub", "bryo"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    ArtifactManager.update_metrics_xlsx("closed_standard", "std", False, False, None)

    fpath_xlsx = tmp_path / "stats" / "metrics.xlsx"
    assert fpath_xlsx.exists()
    ws = load_workbook(fpath_xlsx).active

    def rows_as_lists():
        return [[c.value for c in row] for row in ws.iter_rows()]

    grid = rows_as_lists()
    # always-on harmonic table (7x3), spacer, then the two dataset tables. ordered=False -> columns stay in
    # campaign order [hp, sw]. hp has trials only in cub (mean 55.00), so its harmonic 'All' = 55.00; sw has
    # none anywhere -> "-". harmonic cells are point values (no spread), unlike the per-dataset "± spread".
    assert grid[0][0] == "Harmonic Mean"
    assert grid[1] == ["mAP Composite", "hp", "sw"]
    assert grid[2] == ["All", "55.00", "-"]
    # dataset tables follow, sharing the same [hp, sw] column order
    assert grid[9][0] == "CUB"
    assert grid[10] == ["mAP Composite", "hp (2)", "sw (0)"]
    assert [r[0] for r in grid[11:17]] == ["All", "ID", "OOD", "I2T", "I2I", "T2I"]
    assert grid[11][1] == "55.00 ± 7.07"  # hp: 2 trials
    assert grid[11][2] == "-"             # sw: 0 trials
    assert grid[17] == [None, None, None]  # spacer row
    assert grid[18][0] == "Bryozoa"
    assert grid[19] == ["mAP Composite", "hp (0)", "sw (0)"]
    assert grid[20][1] == "-"             # bryo: no trials for any setting
    # bold_high=False: data cells stay unbolded (only header row + label column bold)
    assert ws.cell(row=12, column=2).font.bold is not True
    # heatmap=None: data cells are left unshaded
    assert ws.cell(row=12, column=2).fill.patternType is None


def test_update_metrics_xlsx_bold_high(tmp_path, monkeypatch) -> None:
    # bold_high=True: the highest-mean setting cell in each score row is bolded. "hp" (base 0.60)
    # outranks "sw" (base 0.50) in every row, so hp's cell bolds and sw's does not; a "-" column
    # (no trials) is ignored and never bolds.
    settings = ["hp", "sw", "iw"]
    for setting, base in (("hp", 0.60), ("sw", 0.50)):
        dpath_final = tmp_path / "settings" / setting / "cub" / "42" / "evals" / "final"
        dpath_final.mkdir(parents=True)
        (dpath_final / "metrics.json").write_text(json.dumps({
            "scores": {"closed_set": {"standard": {"comp": _comp(base)}}},
        }))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": settings, "datasets": ["cub"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    ArtifactManager.update_metrics_xlsx("closed_standard", "std", True, False, None)

    ws = load_workbook(tmp_path / "stats" / "metrics.xlsx").active
    # harmonic table occupies rows 1-8; the CUB per-dataset table starts at row 10 (banner), header row 11,
    # score rows 12..17 = All/ID/OOD/I2T/I2I/T2I; cols B/C/D = hp/sw/iw
    assert ws.cell(row=10, column=1).value == "CUB"
    assert ws.cell(row=11, column=1).value == "mAP Composite"
    for score_row in range(12, 18):
        assert ws.cell(row=score_row, column=2).font.bold is True       # hp wins -> bold
        assert ws.cell(row=score_row, column=3).font.bold is not True   # sw loses -> not bold
        assert ws.cell(row=score_row, column=4).value == "-"            # iw: no trials
        assert ws.cell(row=score_row, column=4).font.bold is not True   # "-" never bolds


def test_update_metrics_xlsx_selects_eval_group(tmp_path, monkeypatch) -> None:
    # table_eval_group routes which comp map is read: 'closed_macro' -> scores.closed_set.per_class,
    # not the (different) scores.closed_set.standard values.
    dpath_final = tmp_path / "settings" / "hp" / "cub" / "42" / "evals" / "final"
    dpath_final.mkdir(parents=True)
    (dpath_final / "metrics.json").write_text(json.dumps({
        "scores": {"closed_set": {
            "standard": {"comp": _comp(0.50)},   # All -> 50.00
            "per_class": {"comp": _comp(0.30)},  # All -> 30.00
        }},
    }))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["hp"], "datasets": ["cub"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    ArtifactManager.update_metrics_xlsx("closed_macro", "std", False, False, None)

    ws = load_workbook(tmp_path / "stats" / "metrics.xlsx").active
    assert ws.cell(row=3, column=1).value == "All"
    assert ws.cell(row=3, column=2).value == "30.00"  # per_class, not standard's 50.00


def _full_map(all_v: float) -> dict:
    # comp map with a controllable "all"; other rows fixed (irrelevant to column ordering)
    return {"all": f"{all_v:.4f}", "id": "0.10", "ood": "0.10", "i2t": "0.10", "i2i": "0.10", "t2i": "0.10"}


def test_update_metrics_xlsx_ordered_harmonic(tmp_path, monkeypatch) -> None:
    # ordered=True: prepend a "Harmonic Mean" table (harmonic mean of each setting/row's per-dataset means)
    # and order every table's columns by its "All" row, descending. Campaign order is [a, b]; harmonic-All
    # gives a=hmean(20,80)=32.00 and b=hmean(40,40)=40.00, so columns flip to [b, a] across all tables.
    for setting, cub_all, bryo_all in (("a", 0.20, 0.80), ("b", 0.40, 0.40)):
        for dataset, all_v in (("cub", cub_all), ("bryo", bryo_all)):
            dpath_final = tmp_path / "settings" / setting / dataset / "42" / "evals" / "final"
            dpath_final.mkdir(parents=True)
            (dpath_final / "metrics.json").write_text(json.dumps({
                "scores": {"closed_set": {"standard": {"comp": {"map": _full_map(all_v)}}}},
            }))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["a", "b"], "datasets": ["cub", "bryo"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    ArtifactManager.update_metrics_xlsx("closed_standard", "std", False, True, None)

    ws = load_workbook(tmp_path / "stats" / "metrics.xlsx").active
    grid = [[c.value for c in r] for r in ws.iter_rows()]
    # harmonic table on top, columns ordered by its "All" row -> b before a
    assert grid[0][0] == "Harmonic Mean"
    assert grid[1] == ["mAP Composite", "b", "a"]
    assert grid[2] == ["All", "40.00", "32.00"]
    # per-dataset tables follow, sharing the same [b, a] column order
    assert grid[9][0] == "CUB"
    assert grid[10] == ["mAP Composite", "b (1)", "a (1)"]
    assert grid[11] == ["All", "40.00", "20.00"]
    assert grid[18][0] == "Bryozoa"
    assert grid[20] == ["All", "40.00", "80.00"]


def _fill_rgb(ws, row, col):
    # last 6 hex chars (RGB) of a cell's fill, or None when the cell is unshaded
    fill = ws.cell(row=row, column=col).fill
    return None if fill.patternType is None else fill.fgColor.rgb[-6:]


def test_update_metrics_xlsx_heatmap_scaled(tmp_path, monkeypatch) -> None:
    # heatmap='scaled': each row's min -> white (#ffffff), max -> #ff5533, rest linearly interpolated;
    # '-' cells stay unshaded. One dataset, so the harmonic "All" row mirrors the values 20/50/80.
    for setting, all_v in (("a", 0.20), ("b", 0.50), ("c", 0.80)):
        dpath_final = tmp_path / "settings" / setting / "cub" / "42" / "evals" / "final"
        dpath_final.mkdir(parents=True)
        (dpath_final / "metrics.json").write_text(json.dumps({
            "scores": {"closed_set": {"standard": {"comp": {"map": _full_map(all_v)}}}},
        }))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["a", "b", "c", "d"], "datasets": ["cub"]})  # "d" has no trials
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    ArtifactManager.update_metrics_xlsx("closed_standard", "std", False, False, "scaled")

    ws = load_workbook(tmp_path / "stats" / "metrics.xlsx").active
    # harmonic "All" row is row 3; cols B/C/D/E = a/b/c/d (values 20/50/80/-)
    assert _fill_rgb(ws, 3, 2) == "FFFFFF"  # row min -> white
    assert _fill_rgb(ws, 3, 3) == "FFAA99"  # midpoint (t=0.5) -> interpolated
    assert _fill_rgb(ws, 3, 4) == "FF5533"  # row max -> #ff5533
    assert _fill_rgb(ws, 3, 5) is None      # "-" cell never shaded
    # per-dataset tables are shaded too (CUB "All" row is row 12)
    assert _fill_rgb(ws, 12, 2) == "FFFFFF"
    assert _fill_rgb(ws, 12, 4) == "FF5533"


def test_update_metrics_xlsx_heatmap_fixed(tmp_path, monkeypatch) -> None:
    # heatmap='fixed': value/100 maps to white->#ff5533 regardless of the row's other cells;
    # 20/50/80 -> #ffddd6 / #ffaa99 / #ff775c.
    for setting, all_v in (("a", 0.20), ("b", 0.50), ("c", 0.80)):
        dpath_final = tmp_path / "settings" / setting / "cub" / "42" / "evals" / "final"
        dpath_final.mkdir(parents=True)
        (dpath_final / "metrics.json").write_text(json.dumps({
            "scores": {"closed_set": {"standard": {"comp": {"map": _full_map(all_v)}}}},
        }))
    (tmp_path / "campaign_metadata.json").write_text(
        json.dumps({"settings": ["a", "b", "c"], "datasets": ["cub"]})
    )

    monkeypatch.setattr(ArtifactManager, "dpath_campaign", tmp_path)

    ArtifactManager.update_metrics_xlsx("closed_standard", "std", False, False, "fixed")

    ws = load_workbook(tmp_path / "stats" / "metrics.xlsx").active
    assert _fill_rgb(ws, 3, 2) == "FFDDD6"  # 20 -> t=0.20
    assert _fill_rgb(ws, 3, 3) == "FFAA99"  # 50 -> t=0.50
    assert _fill_rgb(ws, 3, 4) == "FF775C"  # 80 -> t=0.80


def test_load_base_eval_cache_requires_projections_for_viz_trials(tmp_path, monkeypatch) -> None:
    # a crash between the metrics write and the projections write leaves metrics.json without
    # projections.npz -- a viz trial must read that as a miss (recompute) rather than hit the
    # missing projections.npz downstream in _copy_base_eval; non-viz trials still reuse the metrics
    monkeypatch.setattr(ArtifactManager, "base_eval_cache_dpath", lambda: tmp_path)
    metrics = {"scores": {"comp": {"map": {"all": "0.50"}}}}

    assert ArtifactManager.load_base_eval_cache(require_projections=False) is None  # no cache at all

    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    assert ArtifactManager.load_base_eval_cache(require_projections=True) is None  # partial cache
    assert ArtifactManager.load_base_eval_cache(require_projections=False) == metrics

    (tmp_path / "projections.npz").write_bytes(b"")
    assert ArtifactManager.load_base_eval_cache(require_projections=True) == metrics
