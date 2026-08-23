"""
Campaign reporting/presentation: metric-stats aggregation, per-eval-group composite-score
summary tables (stats/<dataset>/map/*.png, acc/*.png) and metrics workbooks
(stats/metrics/*.xlsx), and per-trial learning-curve plots. Everything here renders
from artifacts already on disk and reads its paths from ArtifactManager; trial/checkpoint state
I/O lives in utils/train.py.
"""

import math
import shutil

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"  # Computer Modern for math ylabels (LaTeX look)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from utils.ddp import rank0
from utils.train import ArtifactManager, BEST_CRITERIA
from utils.utils import (
    paths,
    save_json,
    save_json_listview,
    save_pickle,
    load_json,
    DATASET_ALIAS2NAME,
)

import pdb


# eval group key (the scores group key, also the stats artifact filename) -> display name;
# every stats table/xlsx artifact is rendered once per group
_EVAL_GROUPS = {
    "native": "Native",
    "native_macro": "Native-Macro",
    "joint": "Joint",
    "joint_macro": "Joint-Macro",
}

# checkpoint-selection criterion (BEST_CRITERIA / evals/_best/ subdir) -> banner display name
_SELECTION_NAMES = {"map": "mAP-selection", "acc": "Acc-selection"}

# criterion -> display name of the comp score it tracks (chkpt-mean curves)
_CRITERION_SCORE_NAMES = {"map": "Composite mAP", "acc": "Composite I2T Accuracy"}

# "Hardware Performance" table columns; _HW_LABELS key the per-trial dicts built by _collect_hw,
# _HW_CRASH_LABELS the per-setting crash totals it reads from setting_metadata.json
_HW_LABELS = ("Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM")
_HW_CRASH_LABELS = ("Total Crashes RAM", "Total Crashes VRAM", "Total Crashes Other")

# Learning-curve palette for the eval panels. Three visual families, so the two orthogonal rollups
# of the same quantity don't read as peers: maroon = the composite headline (also _mark_best's star
# color), grayscale = the partition rollups (ID/OOD), hue = the modality rollups. The modality hues
# sit ~120 deg apart and dark enough to stay legible where lines bunch; I2I keeps Okabe-Ito's
# vermillion, which is both far from maroon and the safest partner for the other two under
# red-green color blindness. Linestyle still carries partition (solid ID, dashed OOD), and the line
# weights rank the families so the composite reads as the answer rather than one line among six.
_COLOR_COMP = "maroon"
_COLOR_PARTITION = "#707070"
_COLOR_I2T = "#26418F"  # deep indigo blue
_COLOR_I2I = "#D55E00"  # vermillion
_COLOR_T2I = "#1B7837"  # dark forest green
_LW_COMP = 2.2
_LW_PARTITION = 1.8
_LW_MODALITY = 1.5

# Max heatmap columns in a P/Y strip before adjacent ones are folded together. Each recorded batch
# contributes one histogram column; once there are more than this, every 2 consecutive columns are
# averaged into one (then every 4, every 8, ...) so the strip keeps a readable cell width instead of
# collapsing into a smear. Columns therefore oscillate between this and half of it as training runs.
P_HEATMAP_HORIZONTAL_THRESHOLD = 128
# density colormaps: blues for the predictions, rising off the white page so empty bins vanish into
# it -- a cool hue inferno never reaches, so the adjacent strips stay distinct. Deeper and more
# saturated than matplotlib's Blues, whose muted navy leaves a sparse strip washed out. A thermal
# ramp for the targets, running black -> purple -> orange -> yellow -> white as density rises, so
# that strip reads as a dark thermal image rather than an ink-on-paper one.
_P_CMAP = LinearSegmentedColormap.from_list("p_density", ["#FFFFFF", "#4C7FE8", "#0A1FB0"])
_Y_CMAP = plt.get_cmap("inferno")
# most of the mass sits in one bin (a BCE run starts with every pair near 0), so a linear ramp would
# leave the rest invisible -- sqrt scaling lifts the sparse bins into view
_HIST_NORM = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
# panel background for the learning curves' line plots (the heatmap strips paint over their own)
_BG_LINE_PANEL = "#FAF7F0"

def _fold_hist_columns(cols, threshold):
    """(folded columns, group size) for a P heatmap strip: one histogram per recorded batch folded
    down to at most `threshold` columns. The group size is the smallest power of two that fits the
    budget -- so columns average 1, then 2, then 4, ... consecutive batches, halving the strip each
    time it would overflow -- and each column is the bin-wise mean over its group. Only FULL groups
    are returned: a trailing remainder of fewer than `stride` batches is dropped rather than drawn
    as a column averaging fewer batches than its neighbours."""
    cols = np.asarray(cols, dtype=float)
    stride = 1
    while len(cols) // stride > threshold:
        stride *= 2
    n_full = len(cols) // stride
    grid = np.array([cols[i * stride:(i + 1) * stride].mean(axis=0) for i in range(n_full)])
    return grid, stride


def _spread(nums, spread_type):
    spread = nums.std(ddof=1)
    return spread / np.sqrt(len(nums)) if spread_type == "ste" else spread

def _aggregate_metric_stats(values, spread_type):
    first = values[0]
    if isinstance(first, dict):
        return {k: _aggregate_metric_stats([v[k] for v in values], spread_type) for k in first}
    if len(values) == 1:
        return values[0]
    nums = np.array([float(v) for v in values])
    mean = nums.mean()
    spread = _spread(nums, spread_type)
    return f"{mean * 100:.2f} ± {spread * 100:.2f}"

def _listview_metric_stats(values):
    first = values[0]
    if isinstance(first, dict):
        return {k: _listview_metric_stats([v[k] for v in values]) for k in first}
    return [f"{float(v) * 100:.2f}" for v in values]

@rank0
def update_metric_stats(spread_type):
    dpath_dataset = ArtifactManager.dpath_setting / ArtifactManager.dataset
    dpath_stats = dpath_dataset / "stats"
    for criterion in BEST_CRITERIA:
        for group_key in _EVAL_GROUPS:
            metric_dicts = []
            for dpath_trial in sorted(dpath_dataset.iterdir()):
                # update_chkpt_selection runs first and (re)writes evals/_best/<criterion>/ for
                # exactly the completed trials, so their presence still marks the set to aggregate
                fpath_metrics = dpath_trial / f"evals/_best/{criterion}/{group_key}.json"
                if not fpath_metrics.exists():
                    continue
                metrics = load_json(fpath_metrics)
                for key in ("chkpt", "loss_raw", "sim", "targ"):  # non-score fields aren't aggregated
                    metrics.pop(key)
                metric_dicts.append(metrics)

            if not metric_dicts:
                return

            n_trials = len(metric_dicts)
            stats = {
                "n_trials": n_trials,
                **_aggregate_metric_stats(metric_dicts, spread_type),
            }
            listview = {
                "n_trials": n_trials,
                **_listview_metric_stats(metric_dicts),
            }
            dpath_group = dpath_stats / criterion / group_key
            dpath_group.mkdir(parents=True, exist_ok=True)
            save_json(stats, dpath_group / "metrics.json")
            save_json_listview(listview, dpath_group / "metrics_listview.json")

def _chkpt_dpaths(dpath_trial):
    """[evals/base, evals/eval1, .., evals/eval<n_chkpts>] once the trial's FINAL eval is on disk,
    else None -- the trial-completion signal for every setting-level aggregation here. Each eval
    file's chkpt field carries 'k/n_chkpts', so the highest-numbered eval dir says whether k has
    reached n_chkpts without n_chkpts being threaded in from config. (The old signal, a written
    evals/_best/, can't serve any more: _best/ is now derived from the completed trials rather
    than written by each trial for itself.)"""
    dpath_evals = dpath_trial / "evals"
    dpaths_eval = sorted(dpath_evals.glob("eval*"), key=lambda dpath: int(dpath.name[len("eval"):]))
    if not dpaths_eval:
        return None
    chkpt = load_json(dpaths_eval[-1] / f"{next(iter(_EVAL_GROUPS))}.json")["chkpt"]
    idx_eval, n_chkpts = chkpt.split()[0].split("/")
    return [dpath_evals / "base", *dpaths_eval] if idx_eval == n_chkpts else None

def seed_sweep_complete(seed):
    """True once `seed` has a completed trial in EVERY (setting, dataset) of the campaign -- i.e. one
    full pass of the matrix. The cross-dataset workbooks re-render only at these points: every
    trial completion reselects its own (setting, dataset)'s checkpoint, so mid-sweep the
    cross-dataset artifacts would mix settings reselected against different trial counts."""
    metadata = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")
    return all(
        _chkpt_dpaths(ArtifactManager.dpath_campaign / "settings" / setting / dataset / str(seed)) is not None
        for setting in metadata["settings"]
        for dataset in metadata["datasets"]
    )

def _plot_chkpt_means(means, spreads, idx_best, n_trials, spread_type, score_name, title, fpath):
    chkpts = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(chkpts, means, color="blue", label=f"mean (n={n_trials})")
    ax.fill_between(chkpts, means - spreads, means + spreads, color="blue", alpha=0.2, label=f"± {spread_type}")
    ax.axvline(idx_best, color="red", linestyle="--", linewidth=1, label=f"selected ({idx_best}, {means[idx_best]:.4f})")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Checkpoint", fontsize=10, fontweight="bold")
    ax.set_ylabel(score_name, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

@rank0
def update_chkpt_selection(spread_type):
    """Checkpoint selection, per criterion x eval group, for this setting/dataset: the ONE checkpoint
    index every one of its trials is scored at, argmaxed over the across-trial MEAN curve rather than
    per trial -- argmax(mean(...)), not mean(argmax(...)). Each criterion curves the comp score it
    selects on (BEST_CRITERIA: map -> comp.map.all, acc -> comp.acc.i2t) at every checkpoint of every
    completed trial (_chkpt_dpaths); index 0 is the base eval, plotted but never a candidate, so the
    winner is argmax over 1..n_chkpts with the earliest taking ties.

    The selection MOVES as trials land, so all of this is rewritten from scratch at each trial
    completion, for every completed trial of the setting/dataset -- not just the one that finished:
      - evals/_best/<criterion>/<group>.json in each trial: a copy of ITS eval<idx_best> file
      - stats/<criterion>/<group>/chkpt_means.pkl ({'n_trials', 'chkpts', 'means', 'spreads',
        'idx_best'}) + chkpt_means.png (mean curve, mean +- spread band, selection marked)
      - setting_metadata.json's best_chkpt[<dataset>][<criterion>][<group>]
    """
    dpath_dataset = ArtifactManager.dpath_setting / ArtifactManager.dataset
    best_chkpt = {criterion: {} for criterion in BEST_CRITERIA}
    for criterion, (score_key, metric) in BEST_CRITERIA.items():
        for group_key, group_name in _EVAL_GROUPS.items():
            trials = []  # (trial dir, its comp score at each checkpoint), completed trials only
            for dpath_trial in sorted(dpath_dataset.iterdir()):
                dpaths_chkpt = _chkpt_dpaths(dpath_trial)
                if dpaths_chkpt is None:
                    continue
                trials.append((dpath_trial, [
                    float(load_json(dpath / f"{group_key}.json")["scores"]["comp"][score_key][metric])
                    for dpath in dpaths_chkpt
                ]))

            if not trials:
                return

            curves = np.array([curve for _, curve in trials])
            n_trials = len(curves)
            # a lone trial has no ddof=1 spread -> flat (invisible) band
            spreads = np.zeros(curves.shape[1]) if n_trials == 1 else np.array([_spread(col, spread_type) for col in curves.T])
            means = curves.mean(axis=0)
            idx_best = int(np.argmax(means[1:])) + 1  # base (index 0) is not a candidate; argmax keeps the earliest tie

            for dpath_trial, _ in trials:
                dpath_best = dpath_trial / "evals" / "_best" / criterion
                dpath_best.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    dpath_trial / "evals" / f"eval{idx_best}" / f"{group_key}.json",
                    dpath_best / f"{group_key}.json",
                )

            best_chkpt[criterion][group_key] = {
                "idx": idx_best,
                "n_trials": n_trials,
                "mean": f"{means[idx_best]:.4f}",
            }

            dpath_group = dpath_dataset / "stats" / criterion / group_key
            dpath_group.mkdir(parents=True, exist_ok=True)
            save_pickle(
                {
                    "n_trials": n_trials,
                    "chkpts": np.arange(curves.shape[1]),
                    "means": means,
                    "spreads": spreads,
                    "idx_best": idx_best,
                },
                dpath_group / "chkpt_means.pkl",
            )
            score_name = _CRITERION_SCORE_NAMES[criterion]
            title = (
                f"{score_name} per Checkpoint -- {ArtifactManager.dpath_setting.name}, "
                f"{DATASET_ALIAS2NAME[ArtifactManager.dataset]} ({group_name})"
            )
            _plot_chkpt_means(means, spreads, idx_best, n_trials, spread_type, score_name, title,
                              dpath_group / "chkpt_means.png")

    fpath_meta = ArtifactManager.dpath_setting / "setting_metadata.json"
    metadata = load_json(fpath_meta)
    metadata["best_chkpt"][ArtifactManager.dataset] = best_chkpt
    save_json(metadata, fpath_meta)

def _stats_table_grid(labels, setting_score_maps, spread_type):
    """Build a composite-score table's cell grid from [(setting, [score dict per completed
    trial]), ...]: a header row of 'Setting' + one column per label in `labels`, then one row
    per setting -- '<setting> (n_trials)' + each label's cell (read from the score dicts by
    its lowercased key): '-' (0 trials, or none carrying the score), 'XX.XX' (1 trial, mean), or
    'XX.XX ± XX.XX' (>1 trial, mean ± spread)."""
    grid = [["Setting", *labels]]
    for setting, score_maps in setting_score_maps:
        row = [f"{setting} ({len(score_maps)})"]
        for label in labels:
            key = label.lower()
            # an n-shot bucket with no classes in this dataset's eval partition is absent from its files
            nums = np.array([float(score_map[key]) for score_map in score_maps if key in score_map]) * 100
            if len(nums) == 0:
                row.append("-")
            elif len(nums) == 1:
                row.append(f"{nums[0]:.2f}")
            else:
                row.append(f"{nums.mean():.2f} ± {_spread(nums, spread_type):.2f}")
        grid.append(row)
    return grid

def _collect_comps(settings, datasets, criterion):
    """Per eval group, each (setting, dataset)'s completed-trial score maps, keyed by trial seed
    (the trial dir name; empty dict -> no trials yet): comps_all[group_key][(setting, dataset)]
    [seed] is a {'map': ..., 'acc': ..., 'nshot': [...]} entry -- 'map'/'acc' flat label->score
    dicts merging the comp scores with the per-partition primitives ('id i2t' ... 'ood t2i') and
    the ID partition's n-shot bucket scores (keyed by lowercased bucket name), so table labels map
    to keys by lowercasing; 'nshot' the trial's bucket names in the file's (split) order. The eval
    writes the 'n-shot' dicts only for buckets with classes in the eval partition (the dev split
    drops some), so a bucket can be absent from a dataset's files. Each group reads its own
    best-checkpoint metrics file for the given selection criterion (evals/_best/<criterion>/),
    whose presence is also the completion signal, same as update_metric_stats."""
    comps_all = {group_key: {} for group_key in _EVAL_GROUPS}
    for setting in settings:
        for dataset in datasets:
            dpath_dataset = ArtifactManager.dpath_campaign / "settings" / setting / dataset
            comps = {group_key: {} for group_key in _EVAL_GROUPS}
            if dpath_dataset.exists():
                for dpath_trial in sorted(dpath_dataset.iterdir()):
                    for group_key in _EVAL_GROUPS:
                        fpath_metrics = dpath_trial / f"evals/_best/{criterion}/{group_key}.json"
                        if fpath_metrics.exists():
                            scores_grp = load_json(fpath_metrics)["scores"]
                            nshot_map = scores_grp["id"]["map"].get("n-shot", {})
                            nshot_acc = scores_grp["id"]["acc"].get("n-shot", {})
                            comps[group_key][dpath_trial.name] = {
                                "map": {**scores_grp["comp"]["map"],
                                        **{f"{p} {m}": scores_grp[p]["map"][m] for p in ("id", "ood") for m in ("i2t", "i2i", "t2i")},
                                        **{b.lower(): v for b, v in nshot_map.items()}},
                                "acc": {**scores_grp["comp"]["acc"],
                                        **{f"{p} i2t": scores_grp[p]["acc"]["i2t"] for p in ("id", "ood")},
                                        **{b.lower(): v for b, v in nshot_acc.items()}},
                                "nshot": [b.lower() for b in {**nshot_map, **nshot_acc}],
                            }
            for group_key in _EVAL_GROUPS:
                comps_all[group_key][(setting, dataset)] = comps[group_key]
    return comps_all

def _nshot_names(comps_all):
    """The n-shot bucket names (_collect_comps' 'nshot' lists) seen across every collected trial of
    comps_all ({criterion: {group_key: comps_by}}), in the files' bucket order. A dataset's files
    may lack a bucket (no classes there), so the per-trial lists are merged: each unseen name is
    inserted right after its predecessor in that trial's list."""
    names = []
    for comps_all_crit in comps_all.values():
        for comps_by in comps_all_crit.values():
            for comps in comps_by.values():
                for comp in comps.values():
                    for i, name in enumerate(comp["nshot"]):
                        if name not in names:
                            names.insert(names.index(comp["nshot"][i - 1]) + 1 if i else 0, name)
    return names

def _collect_hw(settings, datasets):
    """Each (setting, dataset)'s completed-trial hardware/wall-clock readings, parsed from
    trial_metadata.json and keyed by trial seed (the trial dir name, like _collect_comps): one
    {_HW_LABELS label -> float} dict per trial -- runtime.trial / runtime.train.mean /
    runtime.eval.mean are float-seconds strings, memory.ram / memory.vram are 'used/total GB'
    strings (numerator taken). A written best-checkpoint (evals/_best/map/) metrics file is the
    completion signal, same as _collect_comps (native.json stands in for the set -- all per-group
    files are materialized together at trial end). Also each setting's n_crashes totals ({'ram'/'vram'/'other' -> int},
    summed across all its trials -- seeds + datasets, completed or not) from
    setting_metadata.json, whose counters survive the trial-dir wipes that reset
    trial_metadata's."""
    hw_by = {}
    for setting in settings:
        for dataset in datasets:
            dpath_dataset = ArtifactManager.dpath_campaign / "settings" / setting / dataset
            trials = {}
            if dpath_dataset.exists():
                for dpath_trial in sorted(dpath_dataset.iterdir()):
                    if (dpath_trial / "evals/_best/map/native.json").exists():
                        meta = load_json(dpath_trial / "trial_metadata.json")
                        trials[dpath_trial.name] = {
                            "Time Trial": float(meta["runtime"]["trial"]),
                            "Mean Time Train": float(meta["runtime"]["train"]["mean"]),
                            "Mean Time Eval": float(meta["runtime"]["eval"]["mean"]),
                            "Peak RAM": float(meta["memory"]["ram"].split("/")[0]),
                            "Peak VRAM": float(meta["memory"]["vram"].split("/")[0]),
                        }
            hw_by[(setting, dataset)] = trials
    crashes_by = {
        setting: load_json(ArtifactManager.dpath_campaign / "settings" / setting / "setting_metadata.json")["n_crashes"]
        for setting in settings
    }
    return hw_by, crashes_by

def _score_labels(supp_scores, nshot_names):
    """(mAP labels, acc labels) for the stats tables: the composite columns, then the enabled
    supplemental groups (supp_scores: {'primitive', 'n_shot'} -> bool) -- primitive appends the
    per-partition primitive score columns (ID/OOD x modality), n_shot the ID-partition n-shot
    bucket columns (one per name in nshot_names: the bucket's composite mAP / I2T accuracy)."""
    map_labels = ("All", "ID", "OOD", "I2T", "I2I", "T2I")
    acc_labels = ("I2T",)
    if supp_scores["primitive"]:
        map_labels += ("ID I2T", "ID I2I", "ID T2I", "OOD I2T", "OOD I2I", "OOD T2I")
        acc_labels += ("ID I2T", "OOD I2T")
    if supp_scores["n_shot"]:
        map_labels += tuple(nshot_names)
        acc_labels += tuple(nshot_names)
    return map_labels, acc_labels

def _cross_dataset_means(settings, datasets, comps_by, score_key, labels):
    """xmeans[(setting, label)]: arithmetic mean, across datasets with completed trials, of that
    setting/label's per-dataset mean comp score (percent), read from comp[score_key][label.lower()]
    (score_key: 'map' or 'acc'); None when no dataset's trials carry the score."""

    def dataset_means(setting, label):
        key = label.lower()
        means = []
        for dataset in datasets:
            # an n-shot bucket absent from a dataset's files (no classes there) leaves it out of the mean
            vals = [float(comp[score_key][key]) for comp in comps_by[(setting, dataset)].values() if key in comp[score_key]]
            if vals:
                means.append(np.mean(vals) * 100)
        return means

    xmeans = {}
    for setting in settings:
        for label in labels:
            vals = dataset_means(setting, label)
            xmeans[(setting, label)] = np.mean(vals) if vals else None
    return xmeans

def _order_settings(settings, xmeans, label):
    # order setting rows by the mean for `label`, descending (ties keep campaign order); callers
    # filter out settings with no completed trials before ordering, so every mean is numeric
    return sorted(settings, key=lambda s: xmeans[(s, label)], reverse=True)

def _col_styles(grid, bold_high):
    """Per-column data-cell styling for one rendered table, shared by the png and xlsx tables:
    styles[c] for each score-label column c -- row -> mean for numeric cells ('-' skipped) and
    the bold-winner rows (highest mean, ties included; empty unless bold_high)."""
    styles = {}
    for c in range(1, len(grid[0])):
        means = {r: float(grid[r][c].split(" ± ")[0]) for r in range(1, len(grid)) if grid[r][c] != "-"}
        winners = set()
        if bold_high and means:
            top = max(means.values())
            winners = {r for r, m in means.items() if m == top}
        styles[c] = (means, winners)
    return styles

def _heat_hex(mean):
    """Heatmap cell color as 'RRGGBB': linear white (#ffffff) -> #ff5533 interpolation over a
    fixed 0.00 -> 100.00."""
    frac = max(0.0, min(1.0, mean / 100.0))
    g = round(255 - (255 - 0x55) * frac)
    b = round(255 - (255 - 0x33) * frac)
    return f"FF{g:02X}{b:02X}"

def _render_stats_table(grid, title, fpath, bold_high, heatmap):
    fig, ax = plt.subplots(figsize=(1.2 + 1.5 * (len(grid[0]) - 1), 0.7 + 0.3 * len(grid)))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    table = ax.table(cellText=grid, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    table.auto_set_column_width(list(range(len(grid[0]))))
    styles = _col_styles(grid, bold_high)
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eaeaea")
            continue
        means, winners = styles[col]
        if row in winners:
            cell.set_text_props(fontweight="bold")
        if heatmap and row in means:
            cell.set_facecolor(f"#{_heat_hex(means[row])}")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

@rank0
def update_stats_tables(spread_type, bold_high, ordered, heatmap, supp_scores):
    """Render the campaign-level composite-score summary tables for this trial's dataset, one pair
    per eval group: artifacts/<campaign>/stats/<dataset>/map/<group>.png (comp mAP: All/ID/OOD/
    I2T/I2I/T2I score columns) and acc/<group>.png (comp I2T accuracy: single I2T column) -- each
    png sources its own selection criterion's best checkpoints (map pngs from evals/_best/map/,
    acc pngs from evals/_best/acc/). supp_scores ({'primitive', 'n_shot'} -> bool) appends the
    enabled supplemental score columns (_score_labels) to the right of both: one
    row per setting with >= 1 completed trial in this dataset (settings without local trials are
    omitted -- no blank rows in the pngs), stats aggregated across each setting's completed
    trials. bold_high/ordered/heatmap style the tables the same way as the metrics workbooks:
    bold_high bolds each score column's highest-mean cell (ties included; '-' cells ignored),
    ordered orders each table's setting rows by its own metric's mean over THIS dataset's
    completed trials (map pngs by the mAP 'All' column, acc pngs by the acc 'I2T' column) --
    localized per dataset and per group, independent of the cross-dataset order used in the
    workbooks -- heatmap shades score cells white->#ff5533 over a fixed 0.00->100.00 (as in
    update_metrics_xlsx). Re-rendered at each trial completion."""
    settings_all = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")["settings"]
    dataset = ArtifactManager.dataset
    comps_all = {criterion: _collect_comps(settings_all, (dataset,), criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _best files, all criteria and groups
    # together, so row presence is criterion- and group-independent: a setting gets a row only
    # once it has >= 1 completed trial in THIS dataset (no blank rows)
    comps_ref = next(iter(comps_all["map"].values()))
    settings = [s for s in settings_all if comps_ref[(s, dataset)]]
    map_labels, acc_labels = _score_labels(supp_scores, _nshot_names(comps_all))

    dpath_stats = ArtifactManager.dpath_campaign / "stats" / dataset
    (dpath_stats / "map").mkdir(parents=True, exist_ok=True)
    (dpath_stats / "acc").mkdir(parents=True, exist_ok=True)

    for group_key, group_name in _EVAL_GROUPS.items():
        comps_map = comps_all["map"][group_key]
        comps_acc = comps_all["acc"][group_key]

        def ordered_settings(comps_by, score_key, label):
            # localized order: this dataset's per-setting trial means (single-dataset degenerate
            # case of _cross_dataset_means), not the cross-dataset means backing the workbook order
            means = _cross_dataset_means(settings, (dataset,), comps_by, score_key, (label,))
            return _order_settings(settings, means, label)

        settings_map = ordered_settings(comps_map, "map", "All") if ordered else settings
        settings_acc = ordered_settings(comps_acc, "acc", "I2T") if ordered else settings
        title_suffix = f" -- {DATASET_ALIAS2NAME[dataset]} ({group_name})"

        grid_map = _stats_table_grid(
            map_labels,
            [(setting, [comp["map"] for comp in comps_map[(setting, dataset)].values()]) for setting in settings_map],
            spread_type,
        )
        _render_stats_table(grid_map, f"Composite mAP{title_suffix}", dpath_stats / "map" / f"{group_key}.png", bold_high, heatmap)

        grid_acc = _stats_table_grid(
            acc_labels,
            [(setting, [comp["acc"] for comp in comps_acc[(setting, dataset)].values()]) for setting in settings_acc],
            spread_type,
        )
        _render_stats_table(grid_acc, f"Composite I2T Accuracy{title_suffix}", dpath_stats / "acc" / f"{group_key}.png", bold_high, heatmap)

@rank0
def update_metrics_xlsx(spread_type, bold_high, ordered, heatmap, supp_scores, baseline_overrides):
    """Write one workbook per selection criterion x eval group to
    artifacts/<campaign>/stats/metrics/{map,acc}/<group>.xlsx, each with three sheets: 'Composite
    mAP' (comp map scores, All/ID/OOD/I2T/I2I/T2I score columns), 'Composite I2T Accuracy'
    (comp acc, single I2T column) and 'Hardware Performance' (see below) -- the score sheets
    source the workbook's own criterion's best checkpoints (evals/_best/<criterion>/), so e.g.
    the map/ workbooks' accuracy sheet holds the
    acc scores at the best-mAP checkpoint and vice versa. supp_scores ({'primitive', 'n_shot'} ->
    bool) appends the enabled supplemental score columns (_score_labels: the per-partition
    primitive scores, then the ID-partition n-shot bucket scores -- one column per bucket, '-'
    where a dataset's files lack the bucket) to the right of both sheets' tables, and splits each
    mAP-sheet table banner into the title cell (first column, unmerged) plus grey merged
    'Composite Scores' / 'Primitive Scores' / 'N-Shot Scores' group headers over their column
    groups (the accuracy sheet keeps full-width merged title banners). Each sheet
    opens with a bold '<repo-parent-dir> - <campaign> (<eval group name>; <selection name>)' title
    cell (e.g. 'bc_dev - dev (Native; mAP-selection)') and a blank row, then stacks one table per campaign dataset vertically -- a bold left-aligned
    title banner, then a table of header row 'Setting' + one column per score label and one
    '<setting> (n_trials)' row per setting, then a blank spacer row before the next dataset -- with
    the always-shown 'Mean' summary table at the bottom: one row per setting, each cell the
    arithmetic mean, across datasets with completed trials, of that setting/label's per-dataset
    mean (a point value, no spread). Cells are '-' (0 trials), 'XX.XX' (1 trial, mean) or 'XX.XX ± XX.XX'
    (>1 trial, mean ± spread), aggregated across each setting's completed trials'
    scores.comp for the workbook's eval group. A setting gets
    rows only once it has >= 1 completed trial in some dataset -- it then appears in every table of
    both sheets, with blank '-' rows in dataset tables lacking its trials; settings with no
    completed trials anywhere are omitted entirely. When bold_high is
    True, the highest-mean setting cell in each score column is bolded (ties included; '-' cells
    ignored). When ordered is True, each sheet's setting rows are ordered by its own metric's
    Mean-table first score column -- 'All' for mAP, 'I2T' for accuracy -- descending, settings with
    no completed trials anywhere last; when False, rows keep the fixed campaign_metadata order.
    Within a sheet one row order is shared across all tables, but the two sheets' orders may differ.
    heatmap shades each score cell white->#ff5533 by value over a fixed 0.00->100.00 (False leaves
    cells unshaded). '-' cells are never shaded. To the right of this aggregate block sit per-seed
    blocks (one blank separator column apart): a 'seed <seed>' label in the campaign-banner row,
    then the per-dataset tables only (no Mean summary) built from that seed's trials alone,
    sitting in the same rows as the aggregate block's dataset tables -- plain setting labels (no
    trial counts), single-trial 'XX.XX' cells, '-' where that seed's trial hasn't completed --
    sharing the aggregate block's setting rows/order. baseline_overrides adds a 'Baseline
    Overrides' config table to each sheet, in a left column band that the score blocks shift
    right past (one blank separator column between), vertically aligned with the aggregate
    block's bottom Mean table so the Mean's Setting column labels its rows: one column per param
    overridden in the campaign's baseline_overrides (union of the settings' overrides.json keys,
    first-seen order), each cell the setting's effective value resolved from its
    config.json -- '-' when the param is absent there, the signal that it is inert
    under that configuration (e.g. loss2.* with loss2.mix 0.0). Params whose effective value is
    identical across every setting row are omitted (they differentiate nothing); when every param
    is uniform the table is omitted entirely. This table gets no
    winner-bold/heatmap styling. The third sheet, 'Hardware Performance', mirrors the score
    sheets' layout (same campaign banner, aggregate block of per-dataset tables + bottom Mean
    table, per-seed blocks, Baseline Overrides band, and the mAP sheet's setting-row order) with
    hardware readings in place of scores: Time Trial / Mean Time Train / Mean Time Eval (whole
    seconds) and Peak RAM / Peak VRAM (whole GB) columns, per-trial readings parsed from
    trial_metadata.json (float-seconds runtime strings; 'used/total GB' memory strings, numerator
    taken), every cell rounded to the nearest int. Each table aggregates the same trials as its
    score-sheet counterpart: dataset tables the mean across that dataset's completed trials
    ('<setting> (n_trials)' labels, '-' rows where the setting has none there), seed-block
    tables that seed's single-trial readings ('-' where its trial hasn't completed), and the
    Mean table the mean across datasets with completed trials of the setting's per-dataset trial
    means -- plus Total Crashes RAM / VRAM / Other columns (Mean table only, since they don't
    decompose per dataset/seed), each cell the setting's crash total of that cause across all its
    trials (seeds + datasets, completed or not), read from setting_metadata.json's n_crashes.
    Hardware cells get no winner-bold/heatmap styling. Column widths hug each column's longest
    header/data cell (banner/label text overflows); blank separator columns get a small ~square
    width. Regenerated at each trial completion."""
    metadata = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")
    settings, datasets = metadata["settings"], metadata["datasets"]

    comps_all = {criterion: _collect_comps(settings, datasets, criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _best files, all criteria and groups
    # together, so row presence and seeds are criterion- and group-independent. a setting gets
    # rows only once it has >= 1 completed trial in some dataset; it then appears in every dataset
    # table (blank '-' row where that dataset has no trials for it yet)
    comps_ref = next(iter(comps_all["map"].values()))
    settings = [s for s in settings if any(comps_ref[(s, dataset)] for dataset in datasets)]
    seeds = sorted({seed for comps in comps_ref.values() for seed in comps}, key=int)

    if baseline_overrides:
        dpath_settings = ArtifactManager.dpath_campaign / "settings"
        okeys = []  # union of overridden params, first-seen order across settings (campaign order)
        for s in settings:
            for key in load_json(dpath_settings / s / "overrides.json"):
                if key not in okeys:
                    okeys.append(key)
        metadata_by = {s: load_json(dpath_settings / s / "config.json") for s in settings}

        def override_value(setting, key):
            # a param absent from the setting's metadata is inert under that configuration -> '-'
            node = metadata_by[setting]
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return "-"
                node = node[part]
            return str(node)

        # a param whose effective value is identical across every setting row differentiates
        # nothing -- drop the column (and with it the whole table when no column survives)
        okeys = [key for key in okeys if len({override_value(s, key) for s in settings}) > 1]

    hw_by, crashes_by = _collect_hw(settings, datasets)

    def overrides_grid(rows):
        # the 'Baseline Overrides' grid: header of param names only -- no setting column; the rows
        # align with (and are labeled by) the aggregate Mean table's setting rows
        if not (baseline_overrides and okeys):
            return None
        return [list(okeys)] + [[override_value(s, key) for key in okeys] for s in rows]

    def build_blocks(comps_by, score_key, labels):
        """The sheet's blocks, left to right: (label, [(title, cell grid), ...]) -- the aggregate
        block (label None): one table per campaign dataset, then the always-shown 'Mean'
        cross-dataset summary table at the bottom; then one block per completed seed (label
        'seed <seed>'): the per-dataset tables only (no Mean summary), built from that seed's
        trials alone -- plain setting labels (no trial counts), single-trial 'XX.XX' cells, '-'
        where that seed's trial hasn't completed. Setting rows are shared across all blocks --
        when ordered, pinned to the aggregate Mean-table's first score column (labels[0]),
        descending. Also returns ogrid, the 'Baseline Overrides' grid (param-name header + one
        value row per setting in this sheet's row order), or None when disabled or nothing is
        overridden, and the sheet's setting-row order."""
        xmeans = _cross_dataset_means(settings, datasets, comps_by, score_key, labels)
        rows = _order_settings(settings, xmeans, labels[0]) if ordered else settings

        tables = []
        for dataset in datasets:
            grid = _stats_table_grid(
                labels,
                [(setting, [comp[score_key] for comp in comps_by[(setting, dataset)].values()]) for setting in rows],
                spread_type,
            )
            tables.append((DATASET_ALIAS2NAME[dataset], grid))
        xgrid = [["Setting", *labels]]
        for s in rows:
            xgrid.append([s] + ["-" if xmeans[(s, label)] is None else f"{xmeans[(s, label)]:.2f}" for label in labels])
        tables.append(("Mean", xgrid))
        blocks = [(None, tables)]

        for seed in seeds:
            stables = []
            for dataset in datasets:
                grid = [["Setting", *labels]]
                for s in rows:
                    comp = comps_by[(s, dataset)].get(seed)
                    grid.append([s] + ["-" if comp is None or label.lower() not in comp[score_key]
                                       else f"{float(comp[score_key][label.lower()]) * 100:.2f}"
                                       for label in labels])
                stables.append((DATASET_ALIAS2NAME[dataset], grid))
            blocks.append((f"seed {seed}", stables))
        return blocks, overrides_grid(rows), rows

    def build_hw_blocks(rows):
        """The 'Hardware Performance' sheet's blocks, structured like build_blocks' (aggregate
        block of per-dataset tables + Mean table, then per-seed blocks) over the hw readings:
        header 'Setting' + _HW_LABELS (the Mean table appends the _HW_CRASH_LABELS crash totals),
        cells the rounded mean of the row's per-trial readings ('-' when the setting has none
        there); the Mean table means the per-dataset trial means across datasets."""

        def hw_row(label, readings):
            if not readings:
                return [label] + ["-"] * len(_HW_LABELS)
            return [label] + [str(round(np.mean([r[hw_label] for r in readings]))) for hw_label in _HW_LABELS]

        tables = []
        for dataset in datasets:
            grid = [["Setting", *_HW_LABELS]]
            for s in rows:
                readings = list(hw_by[(s, dataset)].values())
                grid.append(hw_row(f"{s} ({len(readings)})", readings))
            tables.append((DATASET_ALIAS2NAME[dataset], grid))
        xgrid = [["Setting", *_HW_LABELS, *_HW_CRASH_LABELS]]
        for s in rows:
            dataset_means = [
                {hw_label: np.mean([r[hw_label] for r in hw_by[(s, dataset)].values()]) for hw_label in _HW_LABELS}
                for dataset in datasets if hw_by[(s, dataset)]
            ]
            xgrid.append(hw_row(s, dataset_means) + [str(crashes_by[s][kind]) for kind in ("ram", "vram", "other")])
        tables.append(("Mean", xgrid))
        blocks = [(None, tables)]

        for seed in seeds:
            stables = []
            for dataset in datasets:
                grid = [["Setting", *_HW_LABELS]]
                for s in rows:
                    trial = hw_by[(s, dataset)].get(seed)
                    grid.append(hw_row(s, [] if trial is None else [trial]))
                stables.append((DATASET_ALIAS2NAME[dataset], grid))
            blocks.append((f"seed {seed}", stables))
        return blocks

    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    header_fill = PatternFill("solid", fgColor="EAEAEA")
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def write_sheet(ws, blocks, groups, ogrid, group_name, styled=True):
        """groups: None -> each table's banner is its title merged across the full table width;
        else [(group_title, n_group_cols), ...] -> the title sits unmerged in the block's first
        column, followed by one grey merged group-header cell per group (e.g. 'Composite Scores'
        over the composite columns, 'Primitive Scores' over the primitive columns). ogrid
        (None or a header + one-value-row-per-setting grid, sheet row order) renders as the
        'Baseline Overrides' left column band, one blank separator column after it, with the
        score blocks all shifted right past it -- vertically aligned with the aggregate block's
        bottom Mean table so the Mean's Setting column labels its rows; band cells likewise get
        no winner-bold/heatmap styling. styled=False skips the winner-bold/heatmap styling of
        data cells altogether (the hardware sheet's readings aren't scores)."""
        widths = {}  # col idx -> longest header/data cell text (banner/label cells overflow instead)

        campaign = ws.cell(row=1, column=1, value=f"{paths['root'].parent.name} - {ArtifactManager.dpath_campaign.name} ({group_name})")
        campaign.font = bold
        campaign.alignment = left

        bands = [("Baseline Overrides", ogrid)] if ogrid else []
        offset = sum(len(g[0]) + 1 for _, g in bands)  # left bands + their separator columns
        col0 = 1 + offset  # blocks side by side, one blank separator column apart
        for block_label, tables in blocks:
            if block_label is not None:
                label_cell = ws.cell(row=1, column=col0, value=block_label)  # campaign-banner row, atop the block
                label_cell.font = bold
                label_cell.alignment = left
            # banner row + blank row above the tables; dataset tables lead in every block, so they
            # sit in the same rows across blocks (only the aggregate has the trailing Mean table)
            row = 3
            for title_text, grid in tables:
                n_cols = len(grid[0])  # corner + one col per label (the hw Mean table carries extra crash columns)
                title = ws.cell(row=row, column=col0, value=title_text)
                title.font = bold
                title.alignment = left
                for c in range(col0, col0 + n_cols):  # border every cell of the banner so the merged ranges' edges all render
                    ws.cell(row=row, column=c).border = border
                if groups is None:
                    ws.merge_cells(start_row=row, start_column=col0, end_row=row, end_column=col0 + n_cols - 1)
                else:
                    widths[col0] = max(widths.get(col0, 0), len(title_text))  # unmerged title must fit its column
                    gcol = col0 + 1
                    for group_title, n_group in groups:
                        for c in range(gcol, gcol + n_group):  # fill every cell so the merged range renders grey
                            ws.cell(row=row, column=c).fill = header_fill
                        gcell = ws.cell(row=row, column=gcol, value=group_title)
                        gcell.font = bold
                        gcell.alignment = center
                        ws.merge_cells(start_row=row, start_column=gcol, end_row=row, end_column=gcol + n_group - 1)
                        gcol += n_group
                row += 1

                styles = _col_styles(grid, bold_high and styled)
                for r, grid_row in enumerate(grid):
                    for c, val in enumerate(grid_row):
                        cell = ws.cell(row=row, column=col0 + c, value=val)
                        cell.alignment = left if c == 0 and r > 0 else center  # setting names left-aligned
                        cell.border = border
                        widths[col0 + c] = max(widths.get(col0 + c, 0), len(val))
                        if r == 0 or c == 0:
                            cell.font = bold
                            cell.fill = header_fill
                            continue
                        means, winners = styles[c]
                        if r in winners:
                            cell.font = bold
                        if heatmap and styled and r in means:
                            cell.fill = PatternFill("solid", fgColor=_heat_hex(means[r]))
                    row += 1
                row += 1  # blank spacer row between tables
            # widest table decides the block's width (the hw Mean table carries extra crash columns)
            col0 += max(len(grid[0]) for _, grid in tables) + 1

        # banner row of the aggregate block's bottom Mean table (dataset tables precede it)
        band_row = 3 + sum(len(grid) + 2 for _, grid in blocks[0][1][:-1])
        band_col = 1
        for band_title, band_grid in bands:
            b_cols = len(band_grid[0])
            title = ws.cell(row=band_row, column=band_col, value=band_title)
            title.font = bold
            title.alignment = left
            for c in range(band_col, band_col + b_cols):
                ws.cell(row=band_row, column=c).border = border
            ws.merge_cells(start_row=band_row, start_column=band_col, end_row=band_row, end_column=band_col + b_cols - 1)
            row = band_row + 1
            for r, grid_row in enumerate(band_grid):
                for c, val in enumerate(grid_row):
                    cell = ws.cell(row=row, column=band_col + c, value=val)
                    cell.alignment = center
                    cell.border = border
                    widths[band_col + c] = max(widths.get(band_col + c, 0), len(val))
                    if r == 0:  # header row (param names)
                        cell.font = bold
                        cell.fill = header_fill
                row += 1
            band_col += b_cols + 1

        for c in range(1, max(widths) + 1):
            # snug fit to each column's longest cell; blank separator columns get a small ~square width
            ws.column_dimensions[get_column_letter(c)].width = widths[c] + 2 if c in widths else 3

    nshot_names = _nshot_names(comps_all)
    map_labels, acc_labels = _score_labels(supp_scores, nshot_names)
    # with supplemental columns, mAP-sheet banners split into title + 'Composite Scores' + one
    # group header per enabled supplemental group; the accuracy sheet keeps full-width title banners
    map_groups = [("Composite Scores", 6)]
    if supp_scores["primitive"]:
        map_groups.append(("Primitive Scores", 6))
    if supp_scores["n_shot"] and nshot_names:
        map_groups.append(("N-Shot Scores", len(nshot_names)))
    if len(map_groups) == 1:
        map_groups = None
    # one workbook set per selection criterion: every score in stats/metrics/<criterion>/ (both
    # sheets) comes from that criterion's best checkpoints (e.g. the map/ workbooks' accuracy
    # sheet holds the acc scores at the best-mAP checkpoint), with the banner naming the selection
    for criterion, selection_name in _SELECTION_NAMES.items():
        dpath_metrics = ArtifactManager.dpath_campaign / "stats" / "metrics" / criterion
        dpath_metrics.mkdir(parents=True, exist_ok=True)
        for group_key, group_name in _EVAL_GROUPS.items():
            comps_by = comps_all[criterion][group_key]
            wb = Workbook()
            ws_map = wb.active
            ws_map.title = "Composite mAP"
            map_blocks, map_ogrid, map_rows = build_blocks(comps_by, "map", map_labels)
            write_sheet(ws_map, map_blocks, map_groups, map_ogrid, f"{group_name}; {selection_name}")
            acc_blocks, acc_ogrid, _ = build_blocks(comps_by, "acc", acc_labels)
            write_sheet(wb.create_sheet("Composite I2T Accuracy"), acc_blocks, None, acc_ogrid, f"{group_name}; {selection_name}")
            # the hardware sheet shares the mAP sheet's setting-row order (and so its overrides band)
            write_sheet(wb.create_sheet("Hardware Performance"), build_hw_blocks(map_rows), None, map_ogrid,
                        f"{group_name}; {selection_name}", styled=False)
            wb.save(dpath_metrics / f"{group_key}.xlsx")


@rank0
def plot_metrics(
        data_tracker,
        dpath_trial,
        nshot_bucket_names,
        epoch_size,
        fontsize_axes=12,
        fontsize_ticks=8,
        fontsize_legend=8,
        subplot_border_width=1,
        figsize=(10, 16),
        height_ratios=[2, 2, 2, 2, 2, 1, 1, 1, 0.5, 1, 0.5, 0.5],
    ):
    data = data_tracker.data
    data_epoch = data["epoch"]
    data_eval = data["eval"]
    title_prefix = f"{ArtifactManager.dpath_setting.name}, {DATASET_ALIAS2NAME[ArtifactManager.dataset]}"

    # eval panels (retrieval / n-shot / accuracy) are populated only when eval ran;
    # train panels (loss / grad norm / lr) plot whenever train data is present (e.g. train_pt=trainval).
    has_eval = "scores" in data_eval

    # tracked in samples under the hood; plotted in epoch units
    x_eval = [v / epoch_size for v in data_eval["n_samps_seen"]]
    # n_samps_seen is stamped post-batch, but each batch's metrics (loss, grads, stats, lr) are
    # measured on the pre-step model -- stamp at batch start so the train curves anchor at 0
    x_train = [0.0, *(v / epoch_size for v in data_epoch["n_samps_seen"][:-1])]

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        has_eval,
        nshot_bucket_names,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        group_key="native",
        plot_title=f"{title_prefix}, Native",
        output_filename="native.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        has_eval,
        nshot_bucket_names,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        group_key="native_macro",
        plot_title=f"{title_prefix}, Native Macro",
        output_filename="native_macro.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        has_eval,
        nshot_bucket_names,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        group_key="joint",
        plot_title=f"{title_prefix}, Joint",
        output_filename="joint.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        has_eval,
        nshot_bucket_names,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        group_key="joint_macro",
        plot_title=f"{title_prefix}, Joint Macro",
        output_filename="joint_macro.png",
    )

def plot_composite_metrics(
    data_epoch,
    data_eval,
    x_train,
    x_eval,
    dpath_trial,
    has_eval,
    bucket_comp_keys,
    fontsize_axes,
    fontsize_ticks,
    fontsize_legend,
    subplot_border_width,
    figsize,
    height_ratios,
    group_key,
    plot_title,
    output_filename,
):
    # loss2 active (mix != 0) -> its sim-grad sum gets its own strip between the loss1 strip and the
    # S panel, so each series keeps its own y-scale
    has_loss2 = len(data_epoch["grad_sum_sim2"]) == len(x_train)
    if has_loss2:
        height_ratios = [*height_ratios[:9], 0.5, *height_ratios[9:]]
    # one Y-stats panel per loss branch whose targets carry distributional signal -- TrainPipeline
    # records targ stats only for phylo/tax branches (sp/mp targets are 0/1 indicators), so a branch
    # with no series gets no panel, and neither qualifying leaves none at all. Subscripted per loss
    # whenever loss2 is active, even when only one branch qualifies. The base list's single Y slot
    # (second to last, before LR) is replaced by one per panel.
    targ_panels = [
        (f"targ{tag}_hist", f"Y{sub}" if has_loss2 else "Y")
        for tag, sub in (("1", "₁"), ("2", "₂"))
        if len(data_epoch[f"targ{tag}_hist"]) == len(x_train)
    ]
    # P strips (sigmoid(logits), the predicted pair probabilities) sit between S and Y, on Y's [0, 1]
    # axis so predictions and targets read against each other. Recorded only for BCE-family branches,
    # so an InfoNCE branch has no series and gets no panel.
    p_panels = [
        (f"p{tag}_hist", f"P{sub}" if has_loss2 else "P")
        for tag, sub in (("1", "₁"), ("2", "₂"))
        if len(data_epoch[f"p{tag}_hist"]) == len(x_train)
    ]
    height_ratios = [
        *height_ratios[:-2],
        *[height_ratios[-2]] * (len(p_panels) + len(targ_panels)),
        height_ratios[-1],
    ]
    # each tracked logit scalar (TrialData temp*/bias* series; empty when untracked) gets an LR-height
    # strip between the Y panels and LR, temps first; labels are subscripted per loss whenever loss2 is
    # active, even if only one of the pair is tracked
    scalar_panels = [
        (key, rf"${sym}_{tag}$" if has_loss2 else rf"${sym}$")
        for key, sym, tag in (("temp1", r"\tau", 1), ("temp2", r"\tau", 2), ("bias1", "b", 1), ("bias2", "b", 2))
        if len(data_epoch[key]) == len(x_train)
    ]
    height_ratios = [*height_ratios[:-1], *[0.5] * len(scalar_panels), height_ratios[-1]]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    ax0 = fig.add_subplot(gs[0, 0])

    retrieval_specs = (
        ("i2t", "I2T", _COLOR_I2T),
        ("i2i", "I2I", _COLOR_I2I),
        ("t2i", "T2I", _COLOR_T2I),
    )
    comp_scores = data_eval["scores"][group_key]["comp"] if has_eval else {}
    if has_eval:
        comp_map = comp_scores["map"]
        # the composite is the series checkpoint selection argmaxes -- heaviest and on top
        ax0.plot(x_eval, comp_map["all"], label="All", color=_COLOR_COMP, linewidth=_LW_COMP, zorder=4)
        ax0.plot(x_eval, comp_map["id"], label="ID", color=_COLOR_PARTITION, linewidth=_LW_PARTITION)
        ax0.plot(x_eval, comp_map["ood"], label="OOD", color=_COLOR_PARTITION, linestyle="--", linewidth=_LW_PARTITION)
        for metric_name, metric_label, color in retrieval_specs:
            ax0.plot(x_eval, comp_map[metric_name], label=metric_label, color=color, linewidth=_LW_MODALITY)
        _mark_best(ax0, x_eval, comp_map["all"], fontsize_legend)
    ax0.set_ylabel("mAP Composite", fontsize=fontsize_axes, fontweight="bold")
    ax0.set_ylim(0, 1)
    if has_eval:
        ax0.legend(loc="lower left", ncol=len(ax0.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    if has_eval:
        for partition, partition_label, linestyle in (("id", "ID", "-"), ("ood", "OOD", "--")):
            partition_map = data_eval["scores"][group_key][partition]["map"]
            for metric_name, metric_label, color in retrieval_specs:
                ax1.plot(
                    x_eval,
                    partition_map[metric_name],
                    label=f"{partition_label} {metric_label}",
                    color=color,
                    linestyle=linestyle,
                    linewidth=_LW_MODALITY,
                )

    ax1.set_ylabel("mAP Primitive", fontsize=fontsize_axes, fontweight="bold")
    ax1.set_ylim(0, 1)
    if has_eval:
        ax1.legend(loc="lower left", ncol=len(ax1.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    id_mode_scores = data_eval["scores"][group_key]["id"] if has_eval else {}
    comp_nshot = id_mode_scores["map"].get("n-shot", {}) if has_eval else {}
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax2, x_eval, comp_nshot, key, key)
        if comp_nshot:
            ax2.legend(loc="lower left", ncol=len(ax2.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax2.set_ylabel("n-shot mAP (ID)", fontsize=fontsize_axes, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    if has_eval:
        for partition, partition_label, linestyle in (("id", "ID", "-"), ("ood", "OOD", "--")):
            ax3.plot(
                x_eval,
                data_eval["scores"][group_key][partition]["acc"]["i2t"],
                label=partition_label,
                color=_COLOR_I2T,  # this panel is all-I2T, so it keeps that modality's hue
                linestyle=linestyle,
                linewidth=_LW_MODALITY,
            )
        comp_acc = comp_scores["acc"]["i2t"]
        ax3.plot(x_eval, comp_acc, label="Comp", color=_COLOR_COMP, linewidth=_LW_COMP, zorder=4)
        _mark_best(ax3, x_eval, comp_acc, fontsize_legend)
    ax3.set_ylabel("I2T Acc.", fontsize=fontsize_axes, fontweight="bold")
    ax3.set_ylim(0, 1)
    if has_eval:
        ax3.legend(loc="lower left", ncol=len(ax3.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax3.grid(True)
    ax3.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
    comp_nshot_acc = id_mode_scores["acc"].get("n-shot", {}) if has_eval else {}
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax4, x_eval, comp_nshot_acc, key, key)
        if comp_nshot_acc:
            ax4.legend(loc="lower left", ncol=len(ax4.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax4.set_ylabel("n-shot Acc.\n(ID I2T)", fontsize=fontsize_axes, fontweight="bold")
    ax4.set_ylim(0, 1)
    ax4.grid(True)
    ax4.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)
    if len(data_epoch["loss_train"]) == len(x_train):
        ax5.plot(x_train, data_epoch["loss_train"], label="Train", color="tab:orange", zorder=3)
    if len(data_epoch["loss_raw_train"]) == len(x_train):
        ax5.plot(x_train, data_epoch["loss_raw_train"], label="Train (Raw)", color="tab:blue")
    if has_eval:
        for partition, partition_label, loss_color in (("id", "ID", "tab:green"), ("ood", "OOD", "tab:red")):
            ax5.plot(x_eval, data_eval["loss_raw"][partition], label=f"{partition_label} Val", color=loss_color)
    ax5.set_ylabel(r"$\mathcal{L}$", fontsize=fontsize_axes + 4)
    ax5.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground="black")])
    ax5.set_yscale("log")
    ax5.minorticks_on()
    ax5.grid(which="minor", axis="y")
    ax5.legend(loc="upper center", ncol=len(ax5.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax5.grid(True)
    ax5.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax6 = fig.add_subplot(gs[6, 0], sharex=ax0)
    if len(data_epoch["grad_norm_model"]) == len(x_train):
        ax6.plot(x_train, data_epoch["grad_norm_model"], color="tab:orange")
    ax6.set_ylabel(r"$\|\nabla_{\theta}\mathcal{L}\|$", fontsize=fontsize_axes + 4)
    # CM mathtext has no bold symbol fonts; a thin stroke outline fakes the bold
    ax6.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground="black")])
    ax6.set_yscale("log")
    ax6.minorticks_on()
    ax6.grid(which="minor", axis="y")
    ax6.grid(True)
    ax6.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    # the step the optimizer actually took, directly under the gradient that produced it: Adam
    # rescales per parameter, so the two need not track each other
    ax6b = fig.add_subplot(gs[7, 0], sharex=ax0)
    if len(data_epoch["delta_norm_model"]) == len(x_train):
        ax6b.plot(x_train, data_epoch["delta_norm_model"], color="tab:brown")
    ax6b.set_ylabel(r"$\|\Delta\theta\|$", fontsize=fontsize_axes + 4)
    ax6b.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground="black")])
    ax6b.set_yscale("log")
    ax6b.minorticks_on()
    ax6b.grid(which="minor", axis="y")
    ax6b.grid(True)
    ax6b.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax7 = fig.add_subplot(gs[8, 0], sharex=ax0)
    if len(data_epoch["grad_sum_sim1"]) == len(x_train):
        ax7.plot(x_train, data_epoch["grad_sum_sim1"], color="tab:orange", linewidth=1.0)
    ax7.axhline(0.0, color="gray", linewidth=0.5)
    ax7.set_ylabel(r"$\sum \nabla_S \mathcal{L}_1$" if has_loss2 else r"$\sum \nabla_S \mathcal{L}$", fontsize=fontsize_axes - 1)
    ax7.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.6, foreground="black")])
    ax7.grid(True)
    ax7.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax6b, ax7]

    if has_loss2:
        ax7b = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax7b.plot(x_train, data_epoch["grad_sum_sim2"], color="tab:orange", linewidth=1.0)
        ax7b.axhline(0.0, color="gray", linewidth=0.5)
        ax7b.set_ylabel(r"$\sum \nabla_S \mathcal{L}_2$", fontsize=fontsize_axes - 1)
        ax7b.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.6, foreground="black")])
        ax7b.grid(True)
        ax7b.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax7b)

    # sim1/sim2 are always identical in practice, so the one S panel shows loss1's. Min/max solid,
    # mean dashed, median dotted; teal/rose is a dark, mutually contrasting pair that also stays
    # clear of the orange gradient panels above and the purple temp panels below.
    axes_hist = []  # the heatmap strips, which keep the colormap's own background

    def add_stat_panel(stat_prefix, stat_ylabel, stat_ylim, stat_color):
        ax = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        legend_styles = [
            Line2D([0], [0], color=stat_color, lw=1.0, linestyle=stat_linestyle, label=stat_label)
            for stat_linestyle, stat_label in (("-", "Min/Max"), ("--", "Mean"), (":", "Median"))
        ]
        for stat_name, stat_linestyle in (("min", "-"), ("max", "-"), ("mean", "--"), ("median", ":")):
            stat_key = f"{stat_prefix}_{stat_name}"
            if len(data_epoch[stat_key]) == len(x_train):
                ax.plot(x_train, data_epoch[stat_key], color=stat_color, linestyle=stat_linestyle, linewidth=1.0)
        ax.set_ylabel(stat_ylabel, fontsize=fontsize_axes, fontweight="bold")
        ax.set_ylim(*stat_ylim)
        ax.legend(handles=legend_styles, loc="upper center", ncol=len(legend_styles), fontsize=fontsize_legend)
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    def add_hist_panel(hist_key, ylabel, cmap):
        """The branch's per-batch distribution histograms as a density heatmap strip."""
        ax = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        axes_hist.append(ax)
        grid, stride = _fold_hist_columns(data_epoch[hist_key], P_HEATMAP_HORIZONTAL_THRESHOLD)
        # column i spans batches [i*stride, (i+1)*stride), so its edges sit at those batches' x. The
        # last edge lands one past the final batch only when the groups tile the run exactly; there
        # it extends by one mean batch step.
        step = (x_train[-1] - x_train[0]) / max(1, len(x_train) - 1)
        x_edges = np.array([
            x_train[i] if i < len(x_train) else x_train[-1] + step
            for i in (c * stride for c in range(len(grid) + 1))
        ])
        ax.pcolormesh(x_edges, np.linspace(0.0, 1.0, grid.shape[1] + 1), grid.T,
                      cmap=cmap, norm=_HIST_NORM, shading="flat")
        ax.set_ylabel(ylabel, fontsize=fontsize_axes, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    add_stat_panel("sim1", "S", (-1.0, 1.0), "#008080")
    for hist_key, label in p_panels:
        add_hist_panel(hist_key, label, _P_CMAP)
    for hist_key, label in targ_panels:
        add_hist_panel(hist_key, label, _Y_CMAP)

    for key, label in scalar_panels:
        ax = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax.plot(x_train, data_epoch[key], color="tab:purple" if key.startswith("temp") else "blue")
        ax.set_ylabel(label, fontsize=fontsize_axes + 4)
        ax.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground="black")])
        if key.startswith("temp"):
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    ax10 = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
    if len(data_epoch["lr"]) == len(x_train):
        ax10.plot(x_train, data_epoch["lr"], color="red")
    ax10.set_ylabel("η", fontsize=fontsize_axes + 6, fontweight="bold")
    ax10.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax10.yaxis.set_offset_position("right")
    ax10.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax10.yaxis.get_offset_text().set_visible(False)
    ax10.set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")
    ax10.grid(True)
    ax10.tick_params(labelsize=fontsize_ticks)
    axes.append(ax10)

    for ax in axes:
        ax.label_outer()

    for idx_ax, ax in enumerate(axes):
        if ax not in axes_hist:
            ax.set_facecolor(_BG_LINE_PANEL)
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
    fig.savefig(plots_dir / output_filename, dpi=300)
    plt.close(fig)

def maybe_plot(ax, x, data, key, label, **kwargs):
    """
    Helper for plot_metrics() (N-Shot Composites)
    """
    if key in data and len(data[key]) > 0:
        ax.plot(x, data[key], label=label, **kwargs)

def _mark_best(ax, x, ys, fontsize):
    """
    Star + score label ('XX.X', percent) at a composite series' max -- the point checkpoint
    selection picks.
    """
    idx = max(range(len(ys)), key=ys.__getitem__)
    ax.plot(x[idx], ys[idx], marker="*", color=_COLOR_COMP, markersize=12, zorder=5)
    ax.annotate(f"{ys[idx] * 100:.1f}", (x[idx], ys[idx]), textcoords="offset points",
                xytext=(0, 7), ha="center", color=_COLOR_COMP, fontsize=fontsize, fontweight="bold")
