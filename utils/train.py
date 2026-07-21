import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from dataclasses import asdict
from datetime import datetime, timezone
import os
import time
import random
import numpy as np
import torch
import shutil
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from utils.utils import (
    paths,
    save_pickle,
    load_pickle,
    save_json,
    save_json_listview,
    load_json,
    TimeTracker,
    Timer,
    DATASET_ALIAS2NAME,
)
from utils.ddp import rank0

import pdb


def format_scores(scores):
    if scores is None:
        return None
    if isinstance(scores, dict):
        return {k: format_scores(v) for k, v in scores.items()}
    return f"{float(scores):.4f}"

def parse_scores(scores):
    if scores is None:
        return None
    if isinstance(scores, dict):
        return {k: parse_scores(v) for k, v in scores.items()}
    return float(scores)

def format_mem(mem):
    # {"ram": (used_bytes, total_bytes), "vram": ...} -> {"ram": "4.2/128.0 GB", "vram": ...}
    return {k: f"{used / 2**30:.1f}/{total / 2**30:.1f} GB" for k, (used, total) in mem.items()}

def merge_mem(mem_prev, mem_new):
    """Per key keep whichever formatted 'used/total GB' reading has the higher used -- a running max
    across snapshots (kept as a pair so used/total stay from the same reading if totals ever differ
    across relaunches). None (no reading yet) is always superseded."""
    def used(s):
        return float(s.split("/")[0])
    return {
        k: mem_new[k] if mem_prev[k] is None or used(mem_new[k]) > used(mem_prev[k]) else mem_prev[k]
        for k in mem_new
    }


class TrialData:

    def __init__(self, dpath_trial):

        self.fpath_data = dpath_trial / "data_trial.pkl"  #!

        self.data_epoch = {
            "n_samps_seen": [],
            "lr": [],
            "loss_train": [],
            "loss_raw_train": [],
            "grad_norm_model": [],
            "sim_min": [],
            "sim_max": [],
            "sim_median": [],
            "sim_mean": [],
            "targ_min": [],
            "targ_max": [],
            "targ_median": [],
            "targ_mean": [],
        }
        self.data_eval = {
            "n_samps_seen": [],
        }
        self.data = {
            "epoch": self.data_epoch,
            "eval": self.data_eval,
        }

        self.n_evals = 0

        self.eval_metrics = None  # most recent eval metrics
        self.time_eval = None  # most recent eval time

        self.timer_trial = Timer()
        self.timer_trial.start()

    def update_train_batch(self, n_samps_seen, lr=None, loss_train=None, loss_raw_train=None, grad_norm_model=None, batch_stats=None):

        self.data_epoch["n_samps_seen"].append(n_samps_seen)

        if lr is not None:
            self.data_epoch["lr"].append(lr)
        if loss_train is not None:
            self.data_epoch["loss_train"].append(loss_train)
        if loss_raw_train is not None:
            self.data_epoch["loss_raw_train"].append(loss_raw_train)
        if grad_norm_model is not None:
            self.data_epoch["grad_norm_model"].append(grad_norm_model)
        if batch_stats is not None:
            for stat_key, stat_value in batch_stats.items():
                self.data_epoch[stat_key].append(stat_value)

    def update_eval(self, n_samps_seen):

        def append_nested(dst, src):
            for score_name, score_value in src.items():
                if isinstance(score_value, dict):
                    if score_name not in dst:
                        dst[score_name] = {}
                    append_nested(dst[score_name], score_value)
                else:
                    if score_name not in dst:
                        dst[score_name] = []
                    dst[score_name].append(score_value)

        self.n_evals += 1

        self.data_eval["n_samps_seen"].append(n_samps_seen)

        for k, v in self.eval_metrics.items():
            if k not in self.data_eval:
                self.data_eval[k] = {}
            append_nested(self.data_eval[k], v)

    @classmethod
    def resume(cls, dpath_trial, trial_state):
        obj = cls(dpath_trial)
        obj.data = load_pickle(obj.fpath_data)
        obj.data_epoch = obj.data["epoch"]
        obj.data_eval = obj.data["eval"]
        obj.n_evals = trial_state["n_evals"]
        obj.timer_trial = Timer()
        obj.timer_trial.set_elapsed_time(trial_state["timer_trial_elapsed"])
        obj.timer_trial.start()
        return obj

    @rank0
    def save(self):
        save_pickle(self.data, self.fpath_data)


class ArtifactManager:

    dpath_campaign = None
    dpath_setting = None
    dpath_trial = None
    fpath_metadata_trial = None
    dpath_model_final = None
    dpath_eval_final = None
    dpath_model_checkpoint = None
    resuming = False
    dataset = None
    split = None

    # table_eval_group -> (scores set_key, group key, human-readable name); drives map.png and metrics.xlsx
    _TABLE_EVAL_GROUPS = {
        "closed_standard": ("closed_set", "standard", "Closed-Set, Standard"),
        "closed_macro": ("closed_set", "per_class", "Closed-Set, Macro"),
        "full_standard": ("full_set", "standard", "Full-Set, Standard"),
        "full_macro": ("full_set", "per_class", "Full-Set, Macro"),
    }

    @staticmethod
    def set_paths(cfg_train):

        ArtifactManager.dpath_campaign = paths["artifacts"] / cfg_train.campaign
        ArtifactManager.dpath_setting = ArtifactManager.dpath_campaign / "settings" / cfg_train.setting
        ArtifactManager.dataset = cfg_train.dataset
        ArtifactManager.split = cfg_train.split

        trial_name = cfg_train.seed
        ArtifactManager.dpath_trial = ArtifactManager.dpath_setting / cfg_train.dataset / str(trial_name)
        ArtifactManager.fpath_metadata_trial = ArtifactManager.dpath_trial / "trial_metadata.json"

        ArtifactManager.dpath_model_final = ArtifactManager.dpath_trial / "chkpts/final"
        ArtifactManager.dpath_eval_final = ArtifactManager.dpath_trial / "evals/final"
        ArtifactManager.dpath_model_checkpoint = ArtifactManager.dpath_trial / "chkpts/in_progress"

        if ArtifactManager.dpath_trial.exists():
            if (ArtifactManager.dpath_model_checkpoint / "train_state.pt").exists():
                ArtifactManager.resuming = True
            else:
                shutil.rmtree(ArtifactManager.dpath_trial, ignore_errors=True)
                ArtifactManager.resuming = False
        else:
            ArtifactManager.resuming = False

    @staticmethod
    @rank0
    def create_trial_dirs():
        if ArtifactManager.resuming:
            return
        for subdir in ("logs", "chkpts", "chkpts/in_progress", "learning_curves"):
            (ArtifactManager.dpath_trial / subdir).mkdir(parents=True)

    @staticmethod
    @rank0
    def update_campaign_time():

        def format_duration(seconds: float) -> str:
            seconds = int(seconds)
            days, seconds = divmod(seconds, 86400)
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
        
        fpath_pkl = ArtifactManager.dpath_campaign / "time.pkl"
        fpath_json = ArtifactManager.dpath_campaign / "campaign_metadata.json"

        time_data = load_pickle(fpath_pkl)

        time_last_updated = time_data["last_updated"]
        time_curr = time.time()

        time_elapsed = time_data["elapsed"]
        time_elapsed = time_elapsed + (time_curr - time_last_updated)

        time_data["last_updated"] = time_curr
        time_data["elapsed"] = time_elapsed
        save_pickle(time_data, fpath_pkl)
        
        metadata_camp = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")
        metadata_camp["duration"] = format_duration(time_elapsed)
        save_json(metadata_camp, fpath_json)

    @staticmethod
    @rank0
    def update_campaign_memory(mem):
        # campaign-level memory = running max across every trial's snapshots (== max across
        # trial-level values, maintained incrementally so it survives an OOM-killed trial)
        fpath_meta = ArtifactManager.dpath_campaign / "campaign_metadata.json"
        metadata_camp = load_json(fpath_meta)
        metadata_camp["memory"] = merge_mem(metadata_camp["memory"], format_mem(mem))
        save_json(metadata_camp, fpath_meta)

    @staticmethod
    @rank0
    def save_metadata_setting(cfg_train):
        
        def clean_metadata(metadata):

            del metadata["campaign"]
            del metadata["setting"]
            del metadata["seed"]
            del metadata["idx_seed"]
            del metadata["dataset"]
            del metadata["split"]

            del metadata["dev"]
            del metadata["stats"]
            
            metadata["loss"].pop("wting", None)
            metadata["loss"].pop("focal", None)
            metadata["loss"].pop("dsmr", None)

            if "loss2" in metadata:
                metadata["loss2"].pop("wting", None)
                metadata["loss2"].pop("focal", None)
                metadata["loss2"].pop("dsmr", None)

            if metadata["loss2"]["mix"] == 0.0:
                del metadata["loss2"]
        
        fpath_meta = ArtifactManager.dpath_setting / "setting_metadata.json"
        metadata = asdict(cfg_train)
        clean_metadata(metadata)
        if fpath_meta.exists():
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Setting params changed!"
        else:
            save_json(metadata, fpath_meta)

    @staticmethod
    def _get_trial_runtime_data(data: TrialData, idx_epoch: int, time_tracker: TimeTracker):

        def fmt(seconds):
            return f"{seconds:.2f}" if seconds is not None else None

        def mean_bucket(name):
            return {"mean": fmt(time_tracker.mean(name)), "n": time_tracker.n(name)}

        # train mean keyed on idx_epoch (epochs *started*) to match its per-epoch cadence; shows
        # "0.00" between the first epoch's start and finish, as before.
        if idx_epoch > 0:
            mean_time_train = fmt(time_tracker.mean("train") or 0.0)
        else:
            mean_time_train = None

        time_trial = data.timer_trial.get_elapsed_time()
        # remainder of the trial wall-clock not attributed to a tracked bucket -- checkpoint I/O,
        # sync/barriers, and any in-progress epoch's train time not yet folded into the train mean
        time_other = time_trial - time_tracker.attributed()

        runtime_data = {
            "train": {"mean": mean_time_train, "n": idx_epoch},
            "eval": mean_bucket("eval"),
            "viz_compute": mean_bucket("viz_compute"),
            "other": fmt(time_other),
            "trial": fmt(time_trial),
        }

        return runtime_data

    @staticmethod
    @rank0
    def save_metadata_trial(data: TrialData, idx_epoch: int, time_tracker: TimeTracker, n_samps_seen: int, sample_volume: int, mem, init_flag=False):
        runtime_data = ArtifactManager._get_trial_runtime_data(data, idx_epoch, time_tracker)
        progress_data = {"n_samps_seen": n_samps_seen, "sample_volume": sample_volume}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if init_flag:
            metadata_trial = {
                "dataset": ArtifactManager.dataset,
                "split": ArtifactManager.split,
                "runtime": runtime_data,
                "progress": progress_data,
                "memory": format_mem(mem),
                "datetime_start": now,
                "datetime_last_seen": now,
                "complete": False,
            }
        else:
            metadata_trial = load_json(ArtifactManager.fpath_metadata_trial)
            metadata_trial["runtime"] = runtime_data
            metadata_trial["progress"] = progress_data
            metadata_trial["memory"] = merge_mem(metadata_trial["memory"], format_mem(mem))
            metadata_trial["datetime_last_seen"] = now
        save_json(metadata_trial, ArtifactManager.fpath_metadata_trial)

    @staticmethod
    @rank0
    def save_eval_data(dpath_model, eval_metrics, n_samps_seen_chkpt, n_samps_seen):
        dpath_model.mkdir(parents=True, exist_ok=True)
        fpath_meta = dpath_model / "metrics.json"
        metadata = {
            **format_scores(eval_metrics),
            "n_samps_seen": f"{n_samps_seen_chkpt:,}/{n_samps_seen:,}",
        }
        save_json(metadata, fpath_meta)

    @staticmethod
    def _spread(nums, spread_type):
        spread = nums.std(ddof=1)
        return spread / np.sqrt(len(nums)) if spread_type == "ste" else spread

    @staticmethod
    def _aggregate_metric_stats(values, spread_type, percent=True):
        first = values[0]
        if first is None:  # leaf is None for every trial (e.g. loss_raw["ood"] is never computed)
            return None
        if isinstance(first, dict):
            return {
                k: ArtifactManager._aggregate_metric_stats(
                    [v[k] for v in values], spread_type, percent=percent and k not in ("loss_raw", "sim", "targ")
                )
                for k in first
            }
        if len(values) == 1:
            return values[0]
        nums = np.array([float(v) for v in values])
        mean = nums.mean()
        spread = ArtifactManager._spread(nums, spread_type)
        if percent:
            return f"{mean * 100:.2f} ± {spread * 100:.2f}"
        return f"{mean:.4f} ± {spread:.4f}"

    @staticmethod
    def _listview_metric_stats(values, percent=True):
        first = values[0]
        if first is None:  # leaf is None for every trial (mirrors _aggregate_metric_stats)
            return None
        if isinstance(first, dict):
            return {
                k: ArtifactManager._listview_metric_stats(
                    [v[k] for v in values], percent=percent and k not in ("loss_raw", "sim", "targ")
                )
                for k in first
            }
        if percent:
            return [f"{float(v) * 100:.2f}" for v in values]
        return [f"{float(v):.4f}" for v in values]

    @staticmethod
    @rank0
    def update_metric_stats(spread_type):
        dpath_dataset = ArtifactManager.dpath_setting / ArtifactManager.dataset
        metric_dicts = []
        for dpath_trial in sorted(dpath_dataset.iterdir()):
            # a written final-eval metrics file is the signal a trial finished; the `complete` flag is
            # marked later (campaign_runner, after a clean exit) so it can't gate this aggregation
            fpath_metrics = dpath_trial / "evals/final/metrics.json"
            if not fpath_metrics.exists():
                continue
            metrics = load_json(fpath_metrics)
            metrics.pop("n_samps_seen", None)
            metric_dicts.append(metrics)

        if not metric_dicts:
            return

        n_trials = len(metric_dicts)
        stats = {
            "n_trials": n_trials,
            **ArtifactManager._aggregate_metric_stats(metric_dicts, spread_type),
        }
        listview = {
            "n_trials": n_trials,
            **ArtifactManager._listview_metric_stats(metric_dicts),
        }
        dpath_stats = dpath_dataset / "stats"
        dpath_stats.mkdir(parents=True, exist_ok=True)
        save_json(stats, dpath_stats / "metrics.json")
        save_json_listview(listview, dpath_stats / "metrics_listview.json")

    @staticmethod
    def _stats_table_grid(corner, row_labels, setting_score_maps, spread_type):
        """Build a composite-score table's cell grid from [(setting, [score dict per completed
        trial]), ...]: a header row of `corner` + '<setting> (n_trials)' columns, then one row per
        label in `row_labels` (each read from the score dicts by its lowercased key) with each cell
        '-' (0 trials), 'XX.XX' (1 trial, mean), or 'XX.XX ± XX.XX' (>1 trial, mean ± spread)."""
        grid = [[corner, *(f"{setting} ({len(score_maps)})" for setting, score_maps in setting_score_maps)]]
        for label in row_labels:
            row = [label]
            for _, score_maps in setting_score_maps:
                nums = np.array([float(score_map[label.lower()]) for score_map in score_maps]) * 100
                if len(nums) == 0:
                    row.append("-")
                elif len(nums) == 1:
                    row.append(f"{nums[0]:.2f}")
                else:
                    row.append(f"{nums.mean():.2f} ± {ArtifactManager._spread(nums, spread_type):.2f}")
            grid.append(row)
        return grid

    @staticmethod
    def _render_stats_table(grid, title, fpath):
        fig, ax = plt.subplots(figsize=(1.2 + 1.5 * (len(grid[0]) - 1), 0.7 + 0.3 * len(grid)))
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=12)
        table = ax.table(cellText=grid, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        table.auto_set_column_width(list(range(len(grid[0]))))
        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#eaeaea")
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    @rank0
    def update_stats_tables(table_eval_group, spread_type):
        """Render the campaign-level composite-score summary tables for this trial's dataset to
        artifacts/<campaign>/stats/<dataset>/map.png (comp mAP: All/OOD/ID/I2T/I2I/T2I rows) and
        acc.png (comp I2T accuracy: single row): one column per setting (all settings planned in
        campaign_metadata.json, including ones with no trials yet), stats aggregated across each
        setting's completed trials. Re-rendered at each trial completion."""
        set_key, grp, group_name = ArtifactManager._TABLE_EVAL_GROUPS[table_eval_group]

        settings = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")["settings"]
        setting_comps = []
        for setting in settings:
            dpath_dataset = ArtifactManager.dpath_campaign / "settings" / setting / ArtifactManager.dataset
            comps = []
            if dpath_dataset.exists():
                for dpath_trial in sorted(dpath_dataset.iterdir()):
                    # same completion signal as update_metric_stats: a written final-eval metrics file
                    fpath_metrics = dpath_trial / "evals/final/metrics.json"
                    if fpath_metrics.exists():
                        comps.append(load_json(fpath_metrics)["scores"][set_key][grp]["comp"])
            setting_comps.append((setting, comps))

        title_suffix = f" -- {DATASET_ALIAS2NAME[ArtifactManager.dataset]} ({group_name})"
        dpath_stats = ArtifactManager.dpath_campaign / "stats" / ArtifactManager.dataset
        dpath_stats.mkdir(parents=True, exist_ok=True)

        grid_map = ArtifactManager._stats_table_grid(
            "Composite mAP",
            ("All", "OOD", "ID", "I2T", "I2I", "T2I"),
            [(setting, [comp["map"] for comp in comps]) for setting, comps in setting_comps],
            spread_type,
        )
        ArtifactManager._render_stats_table(grid_map, f"Composite mAP{title_suffix}", dpath_stats / "map.png")

        grid_acc = ArtifactManager._stats_table_grid(
            "Composite I2T Accuracy",
            ("I2T",),
            [(setting, [comp["acc"] for comp in comps]) for setting, comps in setting_comps],
            spread_type,
        )
        ArtifactManager._render_stats_table(grid_acc, f"Composite I2T Accuracy{title_suffix}", dpath_stats / "acc.png")

    @staticmethod
    @rank0
    def update_metrics_xlsx(table_eval_group, spread_type, bold_high, ordered, heatmap):
        """Write artifacts/<campaign>/stats/metrics.xlsx: one composite-mAP table per campaign dataset,
        stacked vertically -- a bold dataset-title banner, then a 7 x (n_settings + 1) table (header row
        'mAP Composite' + '<setting> (n_trials)' columns, then All/ID/OOD/I2T/I2I/T2I rows), then a blank
        spacer row before the next dataset. Cells are '-' (0 trials), 'XX.XX' (1 trial, mean) or
        'XX.XX ± XX.XX' (>1 trial, mean ± spread), aggregated across each setting's completed trials'
        scores.<set_key>.<grp>.comp.map for the eval group selected by table_eval_group -- the same grid
        that backs map.png. A 'Harmonic Mean' summary table is always stacked on top -- each cell the
        harmonic mean, across datasets, of that setting/row's per-dataset mean (a point value, no spread;
        combining per-dataset spreads is not meaningful). When bold_high is True, the highest-mean setting
        cell in each score row is bolded (ties included; '-' cells ignored). When ordered is True, every
        table's columns are ordered by the harmonic table's 'All' row (descending; settings with no
        completed trials anywhere sort last); when False, columns keep the fixed campaign_metadata order. A
        single column order is shared across all tables so they stay aligned. heatmap shades each score
        cell white->#ff5533 by intensity: None leaves cells unshaded; 'scaled' maps each row's min->max to
        white->#ff5533; 'fixed' maps a fixed 0.00->100.00 to white->#ff5533. '-' cells are never shaded.
        Regenerated at each trial completion."""
        set_key, grp, _ = ArtifactManager._TABLE_EVAL_GROUPS[table_eval_group]
        metadata = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")
        settings, datasets = metadata["settings"], metadata["datasets"]
        row_labels = ("All", "ID", "OOD", "I2T", "I2I", "T2I")

        # collect each (setting, dataset)'s completed-trial comp maps once (empty list -> no trials yet)
        maps_by = {}
        for setting in settings:
            for dataset in datasets:
                dpath_dataset = ArtifactManager.dpath_campaign / "settings" / setting / dataset
                maps = []
                if dpath_dataset.exists():
                    for dpath_trial in sorted(dpath_dataset.iterdir()):
                        # same completion signal as update_stats_tables: a written final-eval metrics file
                        fpath_metrics = dpath_trial / "evals/final/metrics.json"
                        if fpath_metrics.exists():
                            maps.append(load_json(fpath_metrics)["scores"][set_key][grp]["comp"]["map"])
                maps_by[(setting, dataset)] = maps

        def dataset_means(setting, label):
            # this setting/row's mean (percent) per dataset, over datasets with completed trials
            key = label.lower()
            return [np.mean([float(m[key]) for m in maps_by[(setting, dataset)]]) * 100
                    for dataset in datasets if maps_by[(setting, dataset)]]

        def harmonic_mean(vals):
            if not vals:
                return None
            if any(v == 0 for v in vals):
                return 0.0
            return len(vals) / sum(1.0 / v for v in vals)

        # harmonic-mean summary across datasets is always shown; `ordered` only sets the shared column order
        hmeans = {(s, label): harmonic_mean(dataset_means(s, label)) for s in settings for label in row_labels}
        if ordered:
            # order columns by the harmonic 'All' row, descending; no-data settings (None) keep campaign order last
            settings = sorted(settings, key=lambda s: (hmeans[(s, "All")] is not None, hmeans[(s, "All")] or 0.0), reverse=True)
        hgrid = [["mAP Composite", *settings]]
        for label in row_labels:
            hgrid.append([label] + ["-" if hmeans[(s, label)] is None else f"{hmeans[(s, label)]:.2f}" for s in settings])

        tables = [("Harmonic Mean", hgrid)]  # (title, cell grid) in top-to-bottom render order
        for dataset in datasets:
            grid = ArtifactManager._stats_table_grid(
                "mAP Composite",
                row_labels,
                [(setting, maps_by[(setting, dataset)]) for setting in settings],
                spread_type,
            )
            tables.append((DATASET_ALIAS2NAME[dataset], grid))

        def cell_mean(val):
            return None if val == "-" else float(val.split(" ± ")[0])

        def heat_fill(frac):
            # linear white (#ffffff) -> #ff5533 interpolation over frac in [0, 1]
            frac = max(0.0, min(1.0, frac))
            g = round(255 - (255 - 0x55) * frac)
            b = round(255 - (255 - 0x33) * frac)
            return PatternFill("solid", fgColor=f"FF{g:02X}{b:02X}")

        bold = Font(bold=True)
        center = Alignment(horizontal="center", vertical="center")
        header_fill = PatternFill("solid", fgColor="EAEAEA")
        thin = Side(style="thin", color="BBBBBB")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        wb = Workbook()
        ws = wb.active
        ws.title = "Composite mAP"
        n_cols = len(settings) + 1

        row = 1
        for title_text, grid in tables:
            title = ws.cell(row=row, column=1, value=title_text)
            title.font = bold
            title.alignment = center
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)
            row += 1

            for r, grid_row in enumerate(grid):
                means = {} if r == 0 else {c: cell_mean(v) for c, v in enumerate(grid_row) if c > 0 and v != "-"}
                winners = set()
                if bold_high and means:
                    top = max(means.values())
                    winners = {c for c, m in means.items() if m == top}
                row_min = min(means.values()) if means else 0.0
                row_max = max(means.values()) if means else 0.0
                for c, val in enumerate(grid_row):
                    cell = ws.cell(row=row, column=c + 1, value=val)
                    cell.alignment = center
                    cell.border = border
                    if r == 0 or c == 0:
                        cell.font = bold
                        cell.fill = header_fill
                        continue
                    if c in winners:
                        cell.font = bold
                    if heatmap and c in means:
                        if heatmap == "fixed":
                            frac = means[c] / 100.0
                        elif row_max > row_min:  # scaled across the row's data cells
                            frac = (means[c] - row_min) / (row_max - row_min)
                        else:  # scaled but row has one value (or all equal) -> lowest color
                            frac = 0.0
                        cell.fill = heat_fill(frac)
                row += 1
            row += 1  # blank spacer row between tables

        ws.column_dimensions["A"].width = 16
        for c in range(2, n_cols + 1):
            ws.column_dimensions[get_column_letter(c)].width = 18

        dpath_stats = ArtifactManager.dpath_campaign / "stats"
        dpath_stats.mkdir(parents=True, exist_ok=True)
        wb.save(dpath_stats / "metrics.xlsx")

    @staticmethod
    def base_eval_cache_fpath(cfg_train):
        # one pickle per combo in a flat dir, named by the serialized combo key -- a save touches
        # only its own combo's file (no shared-file rewrite) and a load reads only its own
        fname = "__".join(str(c) for c in ArtifactManager.base_eval_key(cfg_train)) + ".pkl"
        return paths["root"] / "base_eval_cache" / fname

    @staticmethod
    def base_eval_key(cfg_train):
        """Combo key for one base-eval reading: the config settings that determine the base model's
        eval output. Numerics-level knobs (hw mixed_prec, t-SNE perplexity) are
        deliberately not keyed. Family-inert components are normalized to None so equivalent configs
        share one entry: non_causal is CLIP-only, vis_proj is SigLIP-only, and seed only enters
        through the random init of a linear/mlp vis_proj head."""
        from models import CLIP_MODELS, SIGLIP_MODELS  # local: models pulls open_clip/transformers, too heavy for module import
        model_type = cfg_train.arch["model_type"]
        non_causal = cfg_train.arch["clip"]["non_causal"] if model_type in CLIP_MODELS else None
        vis_proj = cfg_train.arch["siglip"]["vis_proj"] if model_type in SIGLIP_MODELS else None
        seed = cfg_train.seed if vis_proj is not None else None
        return (
            model_type,
            cfg_train.img_norm,
            cfg_train.dataset,
            cfg_train.split,
            non_causal,
            cfg_train.text_template["eval"],
            vis_proj,
            seed,
        )

    @staticmethod
    @rank0
    def load_base_eval_cache(cfg_train, require_projections, require_embs):
        # @rank0: a concurrent same-combo campaign can create/replace this combo's file at any
        # moment, so independent per-rank reads could disagree on hit/miss; rank 0 alone reads and
        # the caller broadcasts the decision. Entries carry only what the caching trial computed
        # (metrics always; projections for viz trials; embs for pooled trials): a trial must read
        # an entry missing a piece it needs as a miss (recompute, overwriting the entry with the
        # richer version) rather than trip _write_base_eval on the missing piece downstream.
        # Leaner entries stay valid hits for trials that don't need the missing pieces.
        fpath = ArtifactManager.base_eval_cache_fpath(cfg_train)
        if not fpath.exists():
            return None
        entry = load_pickle(fpath)
        if require_projections and entry["projections"] is None:
            return None
        if require_embs and entry["embs"] is None:
            return None
        return entry

    @staticmethod
    @rank0
    def save_base_eval_cache(cfg_train, eval_metrics):
        """Write this combo's entry to its own cache file and return the entry. The npz arrays are
        ingested from this trial's evals/_base/, where compute_projections just wrote them
        (projections absent for non-viz trials, embs for non-pooled trials). Written via temp file +
        atomic replace: concurrent same-combo campaigns overwrite each other with equivalent entries,
        and readers never see a torn file; other combos' files are untouched."""
        entry = {
            "metrics": {k: format_scores(v) for k, v in eval_metrics.items() if k not in ("loss_raw", "sim", "targ")},
            "projections": None,
            "embs": None,
        }
        dpath_base = ArtifactManager.dpath_trial / "evals" / "_base"
        for name in ("projections", "embs"):
            fpath_npz = dpath_base / f"{name}.npz"
            if fpath_npz.exists():
                entry[name] = dict(np.load(fpath_npz))
        fpath = ArtifactManager.base_eval_cache_fpath(cfg_train)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath_tmp = fpath.with_name(f"{fpath.name}.{os.uname().nodename}.{os.getpid()}.tmp")  # node+pid: campaigns on different nodes can share a pid
        save_pickle(entry, fpath_tmp)
        fpath_tmp.replace(fpath)
        return entry

    @staticmethod
    @rank0
    def save_train_state(train_pipe, idx_batch):
        state = {
            "model": train_pipe.modelw._unwrapped_model.state_dict(),
            "norm_mean": train_pipe.modelw.norm_mean,
            "norm_std": train_pipe.modelw.norm_std,
            "optimizer": train_pipe.opt.state_dict(),
            "lr_sched": train_pipe.lr_sched.state_dict(),
            "n_samps_seen": train_pipe.n_samps_seen,
            "n_batches_seen": train_pipe.n_batches_seen,
            "idx_epoch": train_pipe.idx_epoch,
            "idx_batch": idx_batch,
            "chkpt_thresh": train_pipe.chkpt_thresh,
            "times": train_pipe.time_tracker.state_dict(),
        }
        torch.save(state, ArtifactManager.dpath_model_checkpoint / "train_state.pt")

    @staticmethod
    def save_rng_states(rank):
        rng_state = {
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "rng_numpy": np.random.get_state(),
            "rng_random": random.getstate(),
        }
        torch.save(rng_state, ArtifactManager.dpath_model_checkpoint / f"rng_state_rank{rank}.pt")

    @staticmethod
    @rank0
    def save_trial_state(data):
        state = {
            "n_evals": data.n_evals,
            "timer_trial_elapsed": data.timer_trial.get_elapsed_time(),
        }
        save_pickle(state, ArtifactManager.dpath_model_checkpoint / "trial_state.pkl")

    @staticmethod
    def load_train_state():
        return torch.load(
            ArtifactManager.dpath_model_checkpoint / "train_state.pt",
            map_location="cpu",
            weights_only=False,
        )

    @staticmethod
    def load_rng_state(rank):
        return torch.load(
            ArtifactManager.dpath_model_checkpoint / f"rng_state_rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )

    @staticmethod
    def load_trial_state():
        return load_pickle(ArtifactManager.dpath_model_checkpoint / "trial_state.pkl")


def _samples_seen_tick_formatter(value, _pos):
    return f"{value / 1_000_000:g}"

@rank0
def plot_metrics(
        data_tracker, 
        dpath_trial,
        fontsize_axes=12, 
        fontsize_ticks=8, 
        fontsize_legend=8,
        subplot_border_width=1,
        figsize=(10, 16),
        height_ratios=[2, 2, 2, 2, 2, 1, 1, 1],
    ):
    data = data_tracker.data
    data_epoch = data["epoch"]
    data_eval = data["eval"]
    title_suffix = f" -- {ArtifactManager.dpath_setting.name}, {DATASET_ALIAS2NAME[ArtifactManager.dataset]}"

    # eval panels (retrieval / n-shot / accuracy) are populated only when eval ran;
    # train panels (loss / grad norm / lr) plot whenever train data is present (e.g. train_pt=trainval).
    partitions = [k for k in data_eval.get("scores", {}).get("closed_set", {}).get("standard", {}).keys() if k != "comp"]

    x_eval = data_eval["n_samps_seen"]
    x_train = data_epoch["n_samps_seen"]

    bucket_partition = "id" if "id" in partitions else None
    bucket_comp_keys_standard = [
        key for key in data_eval.get("scores", {}).get("closed_set", {}).get("standard", {}).get(bucket_partition, {}).get("map", {}).get("n-shot", {}).keys()
    ]
    bucket_comp_keys_full_set = [
        key
        for key in data_eval.get("scores", {}).get("full_set", {}).get("standard", {}).get(bucket_partition, {}).get("map", {}).get("n-shot", {}).keys()
    ]

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_standard,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="standard",
        full_set=False,
        retrieval_ylabel="mAP Scores",
        accuracy_ylabel="I2T Accuracy",
        nshot_accuracy_ylabel="n-shot Accuracy (ID)",
        plot_title=f"Train Metrics{title_suffix}",
        output_filename="closed_standard.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_standard,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="per_class",
        full_set=False,
        retrieval_ylabel="Macro mAP Scores",
        accuracy_ylabel="I2T Per-Class Accuracy",
        nshot_accuracy_ylabel="n-shot Per-Class\nAccuracy (ID)",
        plot_title=f"Train Metrics (Macro){title_suffix}",
        output_filename="closed_macro.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_full_set,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="standard",
        full_set=True,
        retrieval_ylabel="Full-Set mAP Scores",
        accuracy_ylabel="Full-Set I2T Accuracy",
        nshot_accuracy_ylabel="Full-Set n-shot Accuracy (ID)",
        plot_title=f"Train Metrics (Full-Set){title_suffix}",
        output_filename="full_standard.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_full_set,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="per_class",
        full_set=True,
        retrieval_ylabel="Full-Set Macro mAP Scores",
        accuracy_ylabel="Full-Set I2T Per-Class Accuracy",
        nshot_accuracy_ylabel="Full-Set n-shot Per-Class\nAccuracy (ID)",
        plot_title=f"Train Metrics (Macro Full-Set){title_suffix}",
        output_filename="full_macro.png",
    )

def plot_composite_metrics(
    data_epoch,
    data_eval,
    x_train,
    x_eval,
    dpath_trial,
    partitions,
    bucket_partition,
    bucket_comp_keys,
    fontsize_axes,
    fontsize_ticks,
    fontsize_legend,
    subplot_border_width,
    figsize,
    height_ratios,
    partition_metric_group,
    full_set,
    retrieval_ylabel,
    accuracy_ylabel,
    nshot_accuracy_ylabel,
    plot_title,
    output_filename,
):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    ax0 = fig.add_subplot(gs[0, 0])

    id_partition = "id" if "id" in partitions else None
    ood_partition = "ood" if "ood" in partitions else None
    retrieval_specs = (
        ("i2t", "I2T", "blue"),
        ("i2i", "I2I", "red"),
        ("t2i", "T2I", "green"),
    )
    set_key = "full_set" if full_set else "closed_set"
    style_specs = (
        (id_partition, "ID", "-"),
        (ood_partition, "OOD", "--"),
    )
    for partition, partition_label, linestyle in style_specs:
        if partition is None:
            continue
        partition_group_scores = data_eval.get("scores", {}).get(set_key, {}).get(partition_metric_group, {}).get(partition, {})
        partition_group_scores = partition_group_scores.get("map", {})
        for metric_name, metric_label, color in retrieval_specs:
            if metric_name in partition_group_scores:
                ax0.plot(
                    x_eval,
                    partition_group_scores[metric_name],
                    label=f"{partition_label} {metric_label}",
                    color=color,
                    linestyle=linestyle,
                )

    ax0.set_ylabel(retrieval_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax0.set_ylim(0, 1)
    if partitions:
        ax0.legend(loc="lower right", fontsize=fontsize_legend)
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    is_macro_plot = partition_metric_group == "per_class"
    id_mode_scores = data_eval.get("scores", {}).get(set_key, {}).get(partition_metric_group, {}).get(bucket_partition, {})
    nshot_key = "n-shot"
    if is_macro_plot:
        nshot_ylabel = "Full-Set n-shot Macro mAP (ID)" if full_set else "n-shot Macro mAP (ID)"
    else:
        nshot_ylabel = "Full-Set n-shot mAP (ID)" if full_set else "n-shot mAP (ID)"
    comp_nshot = id_mode_scores.get("map", {}).get(nshot_key, {})
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            label = key
            maybe_plot(ax1, x_eval, comp_nshot, key, label, linestyle=":")
        if comp_nshot:
            ax1.legend(loc="lower right", fontsize=fontsize_legend)
    ax1.set_ylabel(nshot_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    for partition in partitions:
        partition_group_scores = data_eval.get("scores", {}).get(set_key, {}).get(partition_metric_group, {}).get(partition, {})
        partition_group_scores = partition_group_scores.get("acc", {})
        if "i2t" in partition_group_scores:
            ax2.plot(
                x_eval,
                partition_group_scores["i2t"],
                label="-".join([s.upper() if i == 0 else s.title() for i, s in enumerate(partition.split("_"))]),
            )
    ax2.set_ylabel(accuracy_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax2.set_ylim(0, 1)
    if partitions:
        ax2.legend(loc="lower right", fontsize=fontsize_legend)
    ax2.grid(True)
    ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    comp_nshot_acc = id_mode_scores.get("acc", {}).get("n-shot", {})
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax3, x_eval, comp_nshot_acc, key, key, linestyle=":")
        if comp_nshot_acc:
            ax3.legend(loc="lower right", fontsize=fontsize_legend)
    ax3.set_ylabel(nshot_accuracy_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax3.set_ylim(0, 1)
    ax3.grid(True)
    ax3.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
    if len(data_epoch.get("loss_train", [])) == len(x_train):
        ax4.plot(x_train, data_epoch["loss_train"], label="Train Loss")
    if len(data_epoch.get("loss_raw_train", [])) == len(x_train):
        ax4.plot(x_train, data_epoch["loss_raw_train"], label="Train Loss (Raw)")
    for partition in partitions:
        if len(data_eval.get("loss_raw", {}).get(partition, [])) == len(x_eval):
            ax4.plot(
                x_eval,
                data_eval["loss_raw"][partition],
                label=f'{"-".join([s.upper() if i == 0 else s.title() for i, s in enumerate(partition.split("_"))])} Val Loss',
            )
    ax4.set_ylabel("Loss", fontsize=fontsize_axes, fontweight="bold")
    ax4.set_yscale("log")
    ax4.minorticks_on()
    ax4.grid(which="minor", axis="y")
    ax4.legend(loc="upper right", fontsize=fontsize_legend)
    ax4.grid(True)
    ax4.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)
    if len(data_epoch.get("grad_norm_model", [])) == len(x_train):
        ax5.plot(x_train, data_epoch["grad_norm_model"], color="green")
    ax5.set_ylabel("Grad Norm", fontsize=fontsize_axes, fontweight="bold")
    ax5.set_yscale("log")
    ax5.minorticks_on()
    ax5.grid(which="minor", axis="y")
    ax5.grid(True)
    ax5.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax6 = fig.add_subplot(gs[6, 0], sharex=ax0)
    color_sim = "tab:blue"
    color_targ = "tab:orange"
    # targets first, then similarity; min/max solid, mean dashed, median dotted
    for stat_prefix, stat_color in (("targ", color_targ), ("sim", color_sim)):
        for stat_name, stat_linestyle in (("min", "-"), ("max", "-"), ("mean", "--"), ("median", ":")):
            stat_key = f"{stat_prefix}_{stat_name}"
            if len(data_epoch.get(stat_key, [])) == len(x_train):
                ax6.plot(x_train, data_epoch[stat_key], color=stat_color, linestyle=stat_linestyle)
    ax6.set_ylabel("Similarity / Target", fontsize=fontsize_axes, fontweight="bold")
    ax6.set_ylim(-1.0, 1.0)
    ax6.legend(
        handles=[
            Line2D([0], [0], color=color_sim, lw=1.5, label="Similarity"),
            Line2D([0], [0], color=color_targ, lw=1.5, label="Target"),
            Line2D([0], [0], color="gray", lw=1.5, linestyle="-", label="Min/Max"),
            Line2D([0], [0], color="gray", lw=1.5, linestyle="--", label="Mean"),
            Line2D([0], [0], color="gray", lw=1.5, linestyle=":", label="Median"),
        ],
        loc="upper center",
        ncol=5,
        fontsize=fontsize_legend,
    )
    ax6.grid(True)
    ax6.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax7 = fig.add_subplot(gs[7, 0], sharex=ax0)
    if len(data_epoch.get("lr", [])) == len(x_train):
        ax7.plot(x_train, data_epoch["lr"])
    ax7.set_ylabel("Learning Rate", fontsize=fontsize_axes, fontweight="bold")
    ax7.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax7.yaxis.set_offset_position("right")
    ax7.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax7.yaxis.get_offset_text().set_visible(False)
    ax7.set_xlabel("Samples Seen (M)", fontsize=fontsize_axes, fontweight="bold")
    ax7.xaxis.set_major_formatter(FuncFormatter(_samples_seen_tick_formatter))
    ax7.grid(True)
    ax7.tick_params(labelsize=fontsize_ticks)

    for ax in (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7):
        ax.label_outer()

    for idx_ax, ax in enumerate((ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7)):
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_border_width)
            spine.set_edgecolor("black")
        if idx_ax % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    fig.suptitle(plot_title, fontweight="bold", y=0.98, fontsize=20)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plots_dir = dpath_trial / "learning_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / output_filename)
    plt.close(fig)

def maybe_plot(ax, x, data, key, label, **kwargs):
    """
    Helper for plot_metrics() (N-Shot Composites)
    """
    if key in data and len(data[key]) > 0:
        ax.plot(x, data[key], label=label, **kwargs)