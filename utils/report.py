"""
Campaign reporting/presentation: per-coord metric-stats aggregation + checkpoint selection
(datasets/<dataset>/arms/<arm>/coords/<coord>/coord_stats/), the per-eval-group composite-score summary
tables + convergence plots at every cross-coord level -- per arm (arms/<arm>/arm_stats/), per dataset
(datasets/<dataset>/dataset_stats/{arm_coords,arms}/) -- each {map,acc}/<group>/{metrics,convergence}.png,
the campaign workbooks (campaign_stats/{arm_coords,arms}/{map,acc}/<group>.xlsx), and per-trial
learning-curve plots -- all under one phase dir of the campaign (artifacts/<campaign>/<phase>/,
ArtifactManager.dpath_phase). The phase's campaign_metadata.json 'matrix' ({dataset: {arm: [coords]}}) is the
planned (dataset, arm, coord) set every sweep gate and table row here keys off: every arm x coord in the
screening phase, each arm's picked coord(s) in the qual phase. Everything here renders from artifacts already
on disk and reads its paths from ArtifactManager; trial/checkpoint state I/O lives in utils/train.py.
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
    load_pickle,
    load_json,
    DATASET_ALIAS2NAME,
)

import pdb


# eval group key (the scores group key, also the stats artifact file/dir name) -> display name;
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
# _HW_CRASH_LABELS the per-row crash totals summed from the coord_metadata.json n_crashes it reads
# (one counter per _CRASH_KINDS cause, in _HW_CRASH_LABELS order)
_HW_LABELS = ("Time Trial", "Mean Time Train", "Mean Time Eval", "Peak RAM", "Peak VRAM")
_HW_CRASH_LABELS = ("Total Crashes RAM", "Total Crashes VRAM", "Total Crashes Other")
_CRASH_KINDS = ("ram", "vram", "other")

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
    dpath_coord = ArtifactManager.dpath_coord
    dpath_stats = dpath_coord / "coord_stats"
    for criterion in BEST_CRITERIA:
        for group_key in _EVAL_GROUPS:
            metric_dicts = []
            for dpath_trial in sorted(dpath_coord.iterdir()):
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

def _dpath_coord(dataset, arm, coord):
    """The coord dir holding (arm, coord)'s trials on `dataset`: datasets/<dataset>/arms/<arm>/coords/<coord>."""
    return ArtifactManager.dpath_phase / "datasets" / dataset / "arms" / arm / "coords" / coord

def _coord_label(dpath_coord):
    """'<arm>/<coord>' of a coord dir (datasets/<dataset>/arms/<arm>/coords/<coord>), for plot titles."""
    return f"{dpath_coord.parent.parent.name}/{dpath_coord.name}"

def _chkpt_dpaths(dpath_trial):
    """[evals/base, evals/eval1, .., evals/eval<n_chkpts>] once the trial's FINAL eval is on disk,
    else None -- the trial-completion signal for every coord-level aggregation here. Each eval
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

def _trial_complete(dataset, arm, coord, seed):
    return _chkpt_dpaths(_dpath_coord(dataset, arm, coord) / str(seed)) is not None

def _metadata():
    """The phase's campaign_metadata.json: 'arms' / 'coords' / 'datasets' in campaign order, 'seeds', and
    'matrix' ({dataset: {arm: [coords]}}) -- the phase's planned (dataset, arm, coord) combos, which every
    sweep gate and table row here keys off (every arm x coord in the screening phase, each arm's picked
    coord(s) in the qual phase)."""
    return load_json(ArtifactManager.dpath_phase / "campaign_metadata.json")

def _matrix_arm_coords(metadata, datasets):
    """The planned (arm, coord) pairs over `datasets` (their union), in campaign order."""
    matrix = metadata["matrix"]
    return [(arm, coord) for arm in metadata["arms"] for coord in metadata["coords"]
            if any(coord in matrix[dataset][arm] for dataset in datasets)]

def arm_sweep_complete(seed, dataset, arm):
    """True once `seed` has a completed trial in EVERY planned coord of `arm` on `dataset` -- the arm's cycle
    of the seed sweep. The arm's arm_stats/ re-render only at these points (train.py): every trial
    completion reselects only its own coord's checkpoint, so mid-cycle the arm's tables would mix
    coords reselected against different trial counts."""
    return all(_trial_complete(dataset, arm, coord, seed) for coord in _metadata()["matrix"][dataset][arm])

def dataset_sweep_complete(seed, dataset):
    """True once `seed` has a completed trial in EVERY planned (arm, coord) on `dataset` -- the dataset's
    cycle of the seed sweep; gates the dataset_stats/ re-render the way arm_sweep_complete gates arm_stats/."""
    return all(
        _trial_complete(dataset, arm, coord, seed)
        for arm, coords in _metadata()["matrix"][dataset].items()
        for coord in coords
    )

def seed_sweep_complete(seed):
    """True once `seed` has a completed trial in EVERY planned (dataset, arm, coord) of the phase -- i.e. one
    full pass of the matrix; gates the campaign_stats/ workbooks' re-render."""
    return all(
        _trial_complete(dataset, arm, coord, seed)
        for dataset, arms in _metadata()["matrix"].items()
        for arm, coords in arms.items()
        for coord in coords
    )

def _plot_chkpt_means(means, spreads, idx_best, n_trials, spread_type, score_name, title, fpath):
    chkpts = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(chkpts, means, color="blue", label=f"mean (n={n_trials})")
    ax.fill_between(chkpts, means - spreads, means + spreads, color="blue", alpha=0.2, label=f"± {spread_type}")
    # star + value on the selected point itself, matching how the learning curves mark theirs
    ax.plot(idx_best, means[idx_best], marker="*", color="blue", markersize=14, linestyle="none",
            zorder=5, label=f"selected ({idx_best})")
    ax.annotate(f"{means[idx_best]:.4f}", (idx_best, means[idx_best]), textcoords="offset points",
                xytext=(0, 9), ha="center", color="blue", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Checkpoint", fontsize=10, fontweight="bold")
    ax.set_ylabel(score_name, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)  # 'best' so the box dodges the curve and the selected-point star
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

@rank0
def update_chkpt_selection(spread_type):
    """Checkpoint selection, per criterion x eval group, for this coord/dataset: the ONE checkpoint
    index every one of its trials is scored at, argmaxed over the across-trial MEAN curve rather than
    per trial -- argmax(mean(...)), not mean(argmax(...)). Each criterion curves the comp score it
    selects on (BEST_CRITERIA: map -> comp.map.all, acc -> comp.acc.i2t) at every checkpoint of every
    completed trial (_chkpt_dpaths); index 0 is the base eval, plotted but never a candidate, so the
    winner is argmax over 1..n_chkpts with the earliest taking ties.

    The selection MOVES as trials land, so all of this is rewritten from scratch at each trial
    completion, for every completed trial of the coord/dataset -- not just the one that finished:
      - evals/_best/<criterion>/<group>.json in each trial: a copy of ITS eval<idx_best> file
      - coord_stats/<criterion>/<group>/chkpt_means.pkl ({'n_trials', 'chkpts', 'means', 'spreads',
        'idx_best'}) + chkpt_means.png (mean curve, mean +- spread band, selection marked)
      - coord_metadata.json's best_chkpt[<criterion>][<group>]
    """
    dpath_coord = ArtifactManager.dpath_coord
    best_chkpt = {criterion: {} for criterion in BEST_CRITERIA}
    for criterion, (score_key, metric) in BEST_CRITERIA.items():
        for group_key, group_name in _EVAL_GROUPS.items():
            trials = []  # (trial dir, its comp score at each checkpoint), completed trials only
            for dpath_trial in sorted(dpath_coord.iterdir()):
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

            dpath_group = dpath_coord / "coord_stats" / criterion / group_key
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
                f"{score_name} per Checkpoint -- {_coord_label(dpath_coord)}, "
                f"{DATASET_ALIAS2NAME[ArtifactManager.dataset]} ({group_name})"
            )
            _plot_chkpt_means(means, spreads, idx_best, n_trials, spread_type, score_name, title,
                              dpath_group / "chkpt_means.png")

    fpath_meta = dpath_coord / "coord_metadata.json"
    metadata = load_json(fpath_meta)
    metadata["best_chkpt"] = best_chkpt
    save_json(metadata, fpath_meta)

def _stats_table_grid(headers, labels, rows, spread_type):
    """Build a composite-score table's cell grid from [(row key, [score dict per completed trial]),
    ...]: a header row of the row-key column names in `headers` (e.g. ('Arm', 'Coord')) + one column
    per label in `labels`, then one row per entry -- its key's cells (the trial count appended to the
    last one: '<coord> (n_trials)') + each label's cell (read from the score dicts by its lowercased
    key): '-' (0 trials, or none carrying the score), 'XX.XX' (1 trial, mean), or 'XX.XX ± XX.XX'
    (>1 trial, mean ± spread)."""
    grid = [[*headers, *labels]]
    for row_key, score_maps in rows:
        row = [*row_key[:-1], f"{row_key[-1]} ({len(score_maps)})"]
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

def _collect_comps(arm_coords, datasets, criterion):
    """Per eval group, each (arm, coord) x dataset's completed-trial score maps, keyed by trial seed
    (the trial dir name; empty dict -> no trials yet): comps_all[group_key][((arm, coord), dataset)]
    [seed] is a {'map': ..., 'acc': ..., 'nshot': [...]} entry -- 'map'/'acc' flat label->score
    dicts merging the comp scores with the per-partition primitives ('id i2t' ... 'ood t2i') and
    the ID partition's n-shot bucket scores (keyed by lowercased bucket name), so table labels map
    to keys by lowercasing; 'nshot' the trial's bucket names in the file's (split) order. The eval
    writes the 'n-shot' dicts only for buckets with classes in the eval partition (the dev split
    drops some), so a bucket can be absent from a dataset's files. Each group reads its own
    best-checkpoint metrics file for the given selection criterion (evals/_best/<criterion>/),
    whose presence is also the completion signal, same as update_metric_stats."""
    comps_all = {group_key: {} for group_key in _EVAL_GROUPS}
    for arm, coord in arm_coords:
        for dataset in datasets:
            dpath_coord = _dpath_coord(dataset, arm, coord)
            comps = {group_key: {} for group_key in _EVAL_GROUPS}
            if dpath_coord.exists():
                for dpath_trial in sorted(dpath_coord.iterdir()):
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
                comps_all[group_key][((arm, coord), dataset)] = comps[group_key]
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

def _collect_hw(arm_coords, datasets):
    """Each (arm, coord) x dataset's completed-trial hardware/wall-clock readings, parsed from
    trial_metadata.json and keyed by trial seed (the trial dir name, like _collect_comps): one
    {_HW_LABELS label -> float} dict per trial -- runtime.trial / runtime.train.mean /
    runtime.eval.mean are float-seconds strings, memory.ram / memory.vram are 'used/total GB'
    strings (numerator taken). A written best-checkpoint (evals/_best/map/) metrics file is the
    completion signal, same as _collect_comps (native.json stands in for the set -- all per-group
    files are materialized together at trial end). Also each (arm, coord) x dataset's n_crashes
    ({'ram'/'vram'/'other' -> int}, summed across its seeds, completed or not) from its coord_metadata.json
    (datasets/<dataset>/arms/<arm>/coords/<coord>/; zeros for a dataset the coord never launched in),
    whose counters survive the trial-dir wipes that reset trial_metadata's."""
    hw_by, crashes_by = {}, {}
    for arm, coord in arm_coords:
        for dataset in datasets:
            dpath_coord = _dpath_coord(dataset, arm, coord)
            trials = {}
            if dpath_coord.exists():
                for dpath_trial in sorted(dpath_coord.iterdir()):
                    if (dpath_trial / "evals/_best/map/native.json").exists():
                        meta = load_json(dpath_trial / "trial_metadata.json")
                        trials[dpath_trial.name] = {
                            "Time Trial": float(meta["runtime"]["trial"]),
                            "Mean Time Train": float(meta["runtime"]["train"]["mean"]),
                            "Mean Time Eval": float(meta["runtime"]["eval"]["mean"]),
                            "Peak RAM": float(meta["memory"]["ram"].split("/")[0]),
                            "Peak VRAM": float(meta["memory"]["vram"].split("/")[0]),
                        }
            hw_by[((arm, coord), dataset)] = trials
            fpath_meta = dpath_coord / "coord_metadata.json"
            crashes_by[((arm, coord), dataset)] = (
                load_json(fpath_meta)["n_crashes"] if fpath_meta.exists() else {kind: 0 for kind in _CRASH_KINDS}
            )
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

def _cross_dataset_means(rows, datasets, comps_by, score_key, labels):
    """xmeans[(row, label)]: arithmetic mean, across datasets with completed trials, of that
    row/label's per-dataset mean comp score (percent), read from comp[score_key][label.lower()]
    (score_key: 'map' or 'acc'; comps_by keyed (row, dataset)); None when no dataset's trials carry
    the score."""

    def dataset_means(row, label):
        key = label.lower()
        means = []
        for dataset in datasets:
            # an n-shot bucket absent from a dataset's files (no classes there) leaves it out of the mean
            vals = [float(comp[score_key][key]) for comp in comps_by[(row, dataset)].values() if key in comp[score_key]]
            if vals:
                means.append(np.mean(vals) * 100)
        return means

    xmeans = {}
    for row in rows:
        for label in labels:
            vals = dataset_means(row, label)
            xmeans[(row, label)] = np.mean(vals) if vals else None
    return xmeans

def _order_rows(rows, xmeans, label):
    # order table rows by the mean for `label`, descending (ties keep campaign order); callers
    # filter out rows with no completed trials before ordering, so every mean is numeric
    return sorted(rows, key=lambda row: xmeans[(row, label)], reverse=True)

def _best_coords(arm_coords, datasets, comps_by, criterion):
    """best[(arm, dataset)]: the arm's best coord on that dataset for one criterion x eval group -- the
    coord with the highest across-trial mean of the criterion's comp score (BEST_CRITERIA: map ->
    comp.map.all, acc -> comp.acc.i2t; the same figure the convergence plots pick their winner by)
    among the arm's planned coords (arm_coords: (arm, coord) pairs in campaign order) with completed
    trials there (comps_by[((arm, coord), dataset)] non-empty), ties to the first; no entry when none
    has any."""
    score_key, metric = BEST_CRITERIA[criterion]
    means = {}  # (arm, dataset) -> [(coord, mean)], campaign order
    for arm, coord in arm_coords:
        for dataset in datasets:
            comps = comps_by[((arm, coord), dataset)]
            if comps:
                mean = np.mean([float(comp[score_key][metric]) for comp in comps.values()])
                means.setdefault((arm, dataset), []).append((coord, mean))
    return {key: max(cms, key=lambda cm: cm[1])[0] for key, cms in means.items()}  # max keeps the first of equals

def _arm_rows(arms, arm_coords, datasets, comps_by, criterion):
    """The best-coord-per-arm row set for one criterion x eval group: (rows, comps_arms, best) -- rows
    the (arm,) keys of the arms with a best coord (_best_coords over arm_coords) in some dataset, campaign
    order; comps_arms[((arm,), dataset)] the best coord's score maps there ({} where the arm has none); and
    best itself."""
    best = _best_coords(arm_coords, datasets, comps_by, criterion)
    rows = [(arm,) for arm in arms if any((arm, dataset) in best for dataset in datasets)]
    comps_arms = {
        ((arm,), dataset): comps_by[((arm, best[(arm, dataset)]), dataset)] if (arm, dataset) in best else {}
        for (arm,) in rows
        for dataset in datasets
    }
    return rows, comps_arms, best

def pick_best_coords():
    """{(arm, dataset): coord} over the current phase tree: each arm's best planned coord per dataset by Native
    mAP composite All -- criterion 'map', eval group 'native' (_best_coords: the highest across-trial mean at
    the selected checkpoint among the arm's coords with completed trials there, ties to the first in campaign
    order); no entry where none has any. The qual phase's selection (campaign_runner._qual_picks)."""
    metadata = _metadata()
    arm_coords = _matrix_arm_coords(metadata, metadata["datasets"])
    comps_by = _collect_comps(arm_coords, metadata["datasets"], "map")["native"]
    return _best_coords(arm_coords, metadata["datasets"], comps_by, "map")

def _curve(dataset, arm, coord, criterion, group_key):
    """(means, idx_best) of the coord's across-trial mean curve on `dataset` for one criterion x eval
    group -- update_chkpt_selection's coord_stats/<criterion>/<group>/chkpt_means.pkl, written for
    exactly the coords with completed trials there."""
    chkpt_means = load_pickle(_dpath_coord(dataset, arm, coord) / "coord_stats" / criterion / group_key / "chkpt_means.pkl")
    return chkpt_means["means"], chkpt_means["idx_best"]

def _col_styles(grid, bold_high, n_keys):
    """Per-column data-cell styling for one rendered table, shared by the png and xlsx tables:
    styles[c] for each score-label column c (the first n_keys columns are the row keys) -- row ->
    mean for numeric cells ('-' skipped) and the bold-winner rows (highest mean, ties included;
    empty unless bold_high)."""
    styles = {}
    for c in range(n_keys, len(grid[0])):
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

def _render_stats_table(grid, n_keys, title, fpath, bold_high, heatmap):
    fig, ax = plt.subplots(figsize=(1.2 + 1.5 * (len(grid[0]) - 1), 0.7 + 0.3 * len(grid)))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    table = ax.table(cellText=grid, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    table.auto_set_column_width(list(range(len(grid[0]))))
    styles = _col_styles(grid, bold_high, n_keys)
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col < n_keys:
            cell.set_text_props(fontweight="bold", ha="left" if col < n_keys and row > 0 else "center")  # row keys left-aligned
            cell.set_facecolor("#eaeaea")
            continue
        means, winners = styles[col]
        if row in winners:
            cell.set_text_props(fontweight="bold")
        if heatmap and row in means:
            cell.set_facecolor(f"#{_heat_hex(means[row])}")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _plot_convergence(curves, idx_win, score_name, title, fpath):
    """[(row key, mean curve, its selected index), ...] overlaid on one log-x axes: every row grey,
    curves[idx_win] redrawn in black on top (legend: its key's cells joined by '/', e.g. 'hp/LR-1.0e-5')
    with its selection marked by a diamond and a red dashed line across the plot at its score.
    Checkpoint 0 (the base eval) has no place on a log axis and is dropped -- it is not a selection
    candidate either way, so every curve starts at checkpoint 1."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")
    # plain checkpoint numbers on the log axis, not the default 10^k scientific labels; the minor
    # ticks carry most of them at these ranges (a campaign's n_chkpts is a couple of decades at most)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%g"))
    ax.set_xlim(1, max(len(means) - 1 for _, means, _ in curves))  # no margin below chkpt 1 (log axis)
    for i, (_, means, _) in enumerate(curves):  # every row grey, one legend entry for the pack
        ax.plot(np.arange(1, len(means)), means[1:], color="grey", alpha=0.6, linewidth=1,
                label=f"all ({len(curves)})" if i == 0 else None)

    row, means, idx_best = curves[idx_win]
    ax.axhline(means[idx_best], color="red", linestyle="--", linewidth=1)
    ax.plot(np.arange(1, len(means)), means[1:], color="black", linewidth=1, zorder=4, label="/".join(row))
    # diamond + value on the selected point itself
    ax.plot(idx_best, means[idx_best], marker="D", color="black", markersize=6, linestyle="none",
            zorder=5, label=f"selected ({idx_best})")
    ax.annotate(f"{means[idx_best]:.4f}", (idx_best, means[idx_best]), textcoords="offset points",
                xytext=(0, 9), ha="center", color="black", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Checkpoint", fontsize=10, fontweight="bold")
    ax.set_ylabel(score_name, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)  # 'best' so the box dodges the curves and the selected-point marker
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _render_stats_pngs(dpath_stats, headers, rowset_of, dataset, subject, labels, spread_type, bold_high, ordered, heatmap):
    """Render one score table + one convergence plot per selection criterion x eval group for ONE
    dataset: dpath_stats/{map,acc}/<group>/metrics.png -- map/ the comp mAP table (All/ID/OOD/I2T/I2I/T2I
    score columns), acc/ the comp I2T accuracy table (single I2T column), each plus the enabled
    supplemental columns (labels = (mAP labels, acc labels) per _score_labels) and each sourcing its
    own criterion's best checkpoints (map/ from evals/_best/map/, acc/ from evals/_best/acc/) -- and
    convergence.png. rowset_of(criterion, group_key) -> (rows, comps_by, curves): the table's row keys
    (tuples of len(headers) cells, e.g. (arm, coord) under headers ('Arm', 'Coord')), comps_by[(row,
    dataset)] their completed-trial score maps by seed (_collect_comps entries), and curves [(row,
    means, idx_best)] their across-trial mean curves (_curve) for the convergence plot.

    Tables: one '<key cells> (n_trials)' row per row key, stats aggregated across its completed
    trials, titled '<score name> -- <subject> (<group name>)'. bold_high bolds each score column's
    highest-mean cell (ties included; '-' cells ignored), ordered orders the rows by the table's own
    metric's mean over THIS dataset's completed trials (map tables by the mAP 'All' column, acc tables
    by the acc 'I2T' column) -- localized per dataset and per group, independent of the cross-dataset
    order used in the workbooks -- heatmap shades score cells white->#ff5533 over a fixed
    0.00->100.00 (as in update_campaign_stats). Convergence plots overlay every row's curve on one
    log-scaled checkpoint axis, all grey, with the winner -- the highest mean at its OWN selected
    checkpoint, ties to the first row -- black on top, its selection marked (_plot_convergence); no
    plot when there are no curves."""
    map_labels, acc_labels = labels
    for criterion in BEST_CRITERIA:
        score_labels = {"map": map_labels, "acc": acc_labels}[criterion]
        score_name = _CRITERION_SCORE_NAMES[criterion]
        for group_key, group_name in _EVAL_GROUPS.items():
            rows, comps_by, curves = rowset_of(criterion, group_key)
            if ordered:
                # localized order: this dataset's per-row trial means (single-dataset degenerate
                # case of _cross_dataset_means), not the cross-dataset means backing the workbook order
                means = _cross_dataset_means(rows, (dataset,), comps_by, criterion, (score_labels[0],))
                rows = _order_rows(rows, means, score_labels[0])
            dpath_group = dpath_stats / criterion / group_key
            dpath_group.mkdir(parents=True, exist_ok=True)
            title_suffix = f" -- {subject} ({group_name})"
            grid = _stats_table_grid(
                headers,
                score_labels,
                [(row, [comp[criterion] for comp in comps_by[(row, dataset)].values()]) for row in rows],
                spread_type,
            )
            _render_stats_table(grid, len(headers), f"{score_name}{title_suffix}", dpath_group / "metrics.png", bold_high, heatmap)
            if curves:
                idx_win = max(range(len(curves)), key=lambda i: curves[i][1][curves[i][2]])
                _plot_convergence(curves, idx_win, score_name, f"{score_name} Convergence{title_suffix}",
                                  dpath_group / "convergence.png")

@rank0
def update_arm_stats(dataset, arm, spread_type, bold_high, ordered, heatmap, supp_scores):
    """Render `arm`'s cross-coord tables/plots for `dataset`:
    datasets/<dataset>/arms/<arm>/arm_stats/{map,acc}/<group>/{metrics,convergence}.png (see
    _render_stats_pngs) -- one 'Coord' row per planned coord of the arm (the phase's matrix) with >= 1
    completed trial in this dataset (coords without local trials are omitted: no blank rows in the pngs), titled
    '<score name> -- <arm>, <dataset> (<group>)'. An arm with no dir on this dataset (no trial of it
    launched there) is skipped. Rendered at the end of the arm's seed cycle (train.py,
    arm_sweep_complete) and unconditionally by the runner on exit / tools.regen_stats."""
    dpath_arm = ArtifactManager.dpath_phase / "datasets" / dataset / "arms" / arm
    if not dpath_arm.exists():
        return
    coords = _metadata()["matrix"][dataset][arm]
    arm_coords = [(arm, coord) for coord in coords]
    comps_all = {criterion: _collect_comps(arm_coords, (dataset,), criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _best files, all criteria and groups
    # together, so row presence is criterion- and group-independent
    comps_ref = next(iter(comps_all["map"].values()))
    rows = [(coord,) for coord in coords if comps_ref[((arm, coord), dataset)]]

    def rowset(criterion, group_key):
        comps_by = {((coord,), dataset): comps_all[criterion][group_key][((arm, coord), dataset)] for (coord,) in rows}
        curves = [((coord,), *_curve(dataset, arm, coord, criterion, group_key)) for (coord,) in rows]
        return rows, comps_by, curves

    _render_stats_pngs(dpath_arm / "arm_stats", ("Coord",), rowset, dataset, f"{arm}, {DATASET_ALIAS2NAME[dataset]}",
                       _score_labels(supp_scores, _nshot_names(comps_all)), spread_type, bold_high, ordered, heatmap)

@rank0
def update_dataset_stats(dataset, spread_type, bold_high, ordered, heatmap, supp_scores):
    """Render `dataset`'s cross-arm tables/plots:
    datasets/<dataset>/dataset_stats/{arm_coords,arms}/{map,acc}/<group>/{metrics,convergence}.png (see
    _render_stats_pngs). arm_coords/ has one ('Arm', 'Coord') row per planned (arm, coord) (the phase's
    matrix) with >= 1 completed trial in this dataset. arms/ has one 'Arm' row per arm, each at its BEST coord for this
    dataset -- per criterion x group, the coord with the highest across-trial mean of the criterion's
    comp score among the arm's coords with completed trials here, ties to the first in campaign order
    (_best_coords) -- so both its table row and its convergence curve are that coord's; arms with no
    completed trial here are omitted. A dataset with no dir (no trial launched on it) is skipped.
    Rendered at the end of the dataset's seed cycle (train.py, dataset_sweep_complete) and
    unconditionally by the runner on exit / tools.regen_stats."""
    dpath_dataset = ArtifactManager.dpath_phase / "datasets" / dataset
    if not dpath_dataset.exists():
        return
    metadata = _metadata()
    arms = metadata["arms"]
    arm_coords = _matrix_arm_coords(metadata, (dataset,))
    comps_all = {criterion: _collect_comps(arm_coords, (dataset,), criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _best files, all criteria and groups
    # together, so row presence is criterion- and group-independent
    comps_ref = next(iter(comps_all["map"].values()))
    rows_ac = [row for row in arm_coords if comps_ref[(row, dataset)]]
    labels = _score_labels(supp_scores, _nshot_names(comps_all))
    subject = DATASET_ALIAS2NAME[dataset]
    dpath_stats = dpath_dataset / "dataset_stats"

    def rowset_arm_coords(criterion, group_key):
        curves = [(row, *_curve(dataset, *row, criterion, group_key)) for row in rows_ac]
        return rows_ac, comps_all[criterion][group_key], curves

    def rowset_arms(criterion, group_key):
        rows, comps_arms, best = _arm_rows(arms, arm_coords, (dataset,), comps_all[criterion][group_key], criterion)
        curves = [((arm,), *_curve(dataset, arm, best[(arm, dataset)], criterion, group_key)) for (arm,) in rows]
        return rows, comps_arms, curves

    _render_stats_pngs(dpath_stats / "arm_coords", ("Arm", "Coord"), rowset_arm_coords, dataset, subject, labels,
                       spread_type, bold_high, ordered, heatmap)
    _render_stats_pngs(dpath_stats / "arms", ("Arm",), rowset_arms, dataset, subject, labels,
                       spread_type, bold_high, ordered, heatmap)

@rank0
def update_campaign_stats(spread_type, bold_high, ordered, heatmap, supp_scores, overrides):
    """Write the campaign's workbooks, one per selection criterion x eval group under
    artifacts/<campaign>/<phase>/campaign_stats/{arm_coords,arms}/{map,acc}/<group>.xlsx. The arm_coords/
    workbooks have one row per planned (arm, coord) (the phase's matrix), keyed by two columns 'Arm' + 'Coord'; the arms/ workbooks
    one row per arm keyed by 'Arm' alone, each arm shown at its BEST coord per dataset -- the coord
    with the highest across-trial mean of the workbook's criterion's comp score among the arm's coords
    with completed trials in that dataset, ties to the first in campaign order (_best_coords; a
    per-(criterion, group) pick, so the workbook's mAP and accuracy sheets show the same coord).
    Every table below is laid out identically in both, over its own rows.

    Each workbook has three sheets: 'Composite mAP' (comp map scores, All/ID/OOD/I2T/I2I/T2I score
    columns), 'Composite I2T Accuracy' (comp acc, single I2T column) and 'Hardware Performance' (see
    below) -- the score sheets source the workbook's own criterion's best checkpoints
    (evals/_best/<criterion>/), so e.g. the map/ workbooks' accuracy sheet holds the acc scores at
    the best-mAP checkpoint and vice versa. supp_scores ({'primitive', 'n_shot'} -> bool) appends the
    enabled supplemental score columns (_score_labels: the per-partition primitive scores, then the
    ID-partition n-shot bucket scores -- one column per bucket, '-' where a dataset's files lack the
    bucket) to the right of both sheets' tables, and splits each mAP-sheet table banner into the
    title (over the key columns) plus grey merged 'Composite Scores' / 'Primitive Scores' / 'N-Shot
    Scores' group headers over their column groups (the accuracy sheet keeps full-width merged title
    banners). Each sheet opens with a bold '<repo-parent-dir> - <campaign> (<eval group name>;
    <selection name>)' title cell (e.g. 'bc_dev - dev (Native; mAP-selection)') and a blank row, then
    stacks one table per campaign dataset vertically -- a bold left-aligned title banner, then a
    table of header row (key columns + one column per score label) and one '<key cells> (n_trials)'
    row per row key, then a blank spacer row before the next dataset -- with the always-shown 'Mean'
    summary table at the bottom: one row per key, each cell the arithmetic mean, across datasets with
    completed trials, of that row/label's per-dataset mean (a point value, no spread). Cells are '-'
    (0 trials), 'XX.XX' (1 trial, mean) or 'XX.XX ± XX.XX' (>1 trial, mean ± spread), aggregated
    across the row's completed trials' scores.comp for the workbook's eval group. A row appears only
    once it has >= 1 completed trial in some dataset -- it then appears in every table of both sheets,
    with blank '-' rows in dataset tables lacking its trials; rows with no completed trials anywhere
    are omitted entirely. When bold_high is True, the highest-mean cell in each score column is
    bolded (ties included; '-' cells ignored). When ordered is True, each sheet's rows are ordered by
    its own metric's Mean-table first score column -- 'All' for mAP, 'I2T' for accuracy -- descending;
    when False, rows keep the fixed campaign_metadata order (arms, then coords within each arm).
    Within a sheet one row order is shared across all tables, but the two sheets' orders may differ.
    heatmap shades each score cell white->#ff5533 by value over a fixed 0.00->100.00 (False leaves
    cells unshaded). '-' cells are never shaded. To the right of this aggregate block sit per-seed
    blocks (one blank separator column apart): a 'seed <seed>' label in the campaign-banner row,
    then the per-dataset tables only (no Mean summary) built from that seed's trials alone,
    sitting in the same rows as the aggregate block's dataset tables -- plain key cells (no trial
    counts), single-trial 'XX.XX' cells, '-' where that seed's trial hasn't completed -- sharing the
    aggregate block's rows/order.

    overrides adds the config bands -- 'Arm Overrides' and, in the arm_coords/ workbooks only, 'Coord
    Overrides' (an arms/ row's coord differs per dataset, so it has no single coord config) -- to each
    sheet, in a left column band that the score blocks shift right past (one blank separator column
    between each), vertically aligned with the aggregate block's bottom Mean table so the Mean's key
    columns label their rows: one column per param declared on that side of the campaign matrix
    (ablation_arms for the arm band, hpo_coords for the coord band: the union of the rows' overrides.json
    'arm' / 'coord' keys, first-seen order), each cell the row's effective value resolved from its
    config.json (an arms/ row reads its arm's first (arm, coord) row's) -- '-' when the param is
    absent there, the signal that it is inert under that configuration (e.g. loss2.* with loss2.mix
    0.0). Params whose effective value is identical across every row of the workbook are omitted
    (they differentiate nothing); a band all of whose params are uniform is omitted entirely. These
    tables get no winner-bold/heatmap styling. The third sheet, 'Hardware Performance', mirrors the
    score sheets' layout (same campaign banner, aggregate block of per-dataset tables + bottom Mean
    table, per-seed blocks, overrides bands, and the mAP sheet's row order) with hardware readings in
    place of scores: Time Trial / Mean Time Train / Mean Time Eval (whole seconds) and Peak RAM /
    Peak VRAM (whole GB) columns, per-trial readings parsed from trial_metadata.json (float-seconds
    runtime strings; 'used/total GB' memory strings, numerator taken), every cell rounded to the
    nearest int. Each table aggregates the same trials as its score-sheet counterpart: dataset tables
    the mean across that dataset's completed trials ('<key cells> (n_trials)' labels, '-' rows where
    the row has none there), seed-block tables that seed's single-trial readings ('-' where its trial
    hasn't completed), and the Mean table the mean across datasets with completed trials of the row's
    per-dataset trial means -- plus Total Crashes RAM / VRAM / Other columns (Mean table only, since
    they don't decompose per dataset/seed), each cell the row's crash total of that cause across all
    its trials (seeds + datasets, completed or not; an arms/ row sums its best coords'), read from
    coord_metadata.json's n_crashes. Hardware cells get no winner-bold/heatmap styling. Column widths
    hug each column's longest header/data cell (banner/label text overflows); blank separator columns
    get a small ~square width. Regenerated at the end of each full seed sweep of the matrix (train.py,
    seed_sweep_complete) and unconditionally by the runner on exit / tools.regen_stats."""
    metadata = _metadata()
    arms, datasets = metadata["arms"], metadata["datasets"]
    arm_coords = _matrix_arm_coords(metadata, datasets)

    comps_all = {criterion: _collect_comps(arm_coords, datasets, criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _best files, all criteria and groups
    # together, so row presence and seeds are criterion- and group-independent. an (arm, coord) gets
    # rows only once it has >= 1 completed trial in some dataset; it then appears in every dataset
    # table (blank '-' row where that dataset has no trials for it yet)
    comps_ref = next(iter(comps_all["map"].values()))
    rows_ac = [row for row in arm_coords if any(comps_ref[(row, dataset)] for dataset in datasets)]
    rows_arms = [(arm,) for arm in arms if any(row[0] == arm for row in rows_ac)]
    seeds = sorted({seed for comps in comps_ref.values() for seed in comps}, key=int)
    hw_by, crashes_by = _collect_hw(arm_coords, datasets)

    if overrides:
        # an (arm, coord) row's overrides.json / config.json are written per dataset
        # (datasets/<dataset>/arms/<arm>/coords/<coord>/) but identical across them, so each row reads its
        # from the first dataset it has completed trials in; an (arm,) row of the arms workbooks reads its
        # arm's first (arm, coord) row's (the arm's params are the same in every coord of it)
        rep = {}
        for row in rows_ac:
            rep[row] = row
            rep.setdefault((row[0],), row)
        dpaths = {row: _dpath_coord(next(d for d in datasets if comps_ref[(row, d)]), *row) for row in rows_ac}
        declared = {row: load_json(dpaths[row] / "overrides.json") for row in rows_ac}  # {'arm': {...}, 'coord': {...}}
        config_by = {row: load_json(dpaths[row] / "config.json") for row in rows_ac}

    def override_value(row, key):
        # a param absent from the row's config.json is inert under that configuration -> '-'
        node = config_by[rep[row]]
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return "-"
            node = node[part]
        return str(node)

    def band_keys(side, rows):
        # the params declared on `side` ('arm' / 'coord') across `rows`, first-seen order (campaign
        # order); a param whose effective value is identical across every row differentiates nothing
        # -- drop the column (and with it the whole band when no column survives)
        keys = []
        for row in rows:
            for key in declared[rep[row]][side]:
                if key not in keys:
                    keys.append(key)
        return [key for key in keys if len({override_value(row, key) for row in rows}) > 1]

    def band_grids(band_specs, rows):
        # [(title, grid)] for the bands ([(title, keys)]) with surviving params: a header of param
        # names only -- no key columns; the rows align with (and are labeled by) the aggregate Mean
        # table's rows
        return [(title, [list(keys)] + [[override_value(row, key) for key in keys] for row in rows])
                for title, keys in band_specs if keys]

    def build_blocks(headers, rows, comps_by, score_key, labels):
        """A score sheet's blocks, left to right: (label, [(title, cell grid), ...]) -- the aggregate
        block (label None): one table per campaign dataset, then the always-shown 'Mean'
        cross-dataset summary table at the bottom; then one block per completed seed (label
        'seed <seed>'): the per-dataset tables only (no Mean summary), built from that seed's
        trials alone -- plain key cells (no trial counts), single-trial 'XX.XX' cells, '-'
        where that seed's trial hasn't completed. Rows are shared across all blocks -- when
        ordered, pinned to the aggregate Mean-table's first score column (labels[0]),
        descending. Also returns the sheet's row order."""
        xmeans = _cross_dataset_means(rows, datasets, comps_by, score_key, labels)
        rows = _order_rows(rows, xmeans, labels[0]) if ordered else rows

        tables = []
        for dataset in datasets:
            grid = _stats_table_grid(
                headers,
                labels,
                [(row, [comp[score_key] for comp in comps_by[(row, dataset)].values()]) for row in rows],
                spread_type,
            )
            tables.append((DATASET_ALIAS2NAME[dataset], grid))
        xgrid = [[*headers, *labels]]
        for row in rows:
            xgrid.append([*row] + ["-" if xmeans[(row, label)] is None else f"{xmeans[(row, label)]:.2f}" for label in labels])
        tables.append(("Mean", xgrid))
        blocks = [(None, tables)]

        for seed in seeds:
            stables = []
            for dataset in datasets:
                grid = [[*headers, *labels]]
                for row in rows:
                    comp = comps_by[(row, dataset)].get(seed)
                    grid.append([*row] + ["-" if comp is None or label.lower() not in comp[score_key]
                                          else f"{float(comp[score_key][label.lower()]) * 100:.2f}"
                                          for label in labels])
                stables.append((DATASET_ALIAS2NAME[dataset], grid))
            blocks.append((f"seed {seed}", stables))
        return blocks, rows

    def build_hw_blocks(headers, rows, hw_by, crash_totals):
        """The 'Hardware Performance' sheet's blocks, structured like build_blocks' (aggregate
        block of per-dataset tables + Mean table, then per-seed blocks) over the hw readings
        (hw_by[(row, dataset)][seed]): header key columns + _HW_LABELS (the Mean table appends the
        _HW_CRASH_LABELS crash totals, crash_totals[row]), cells the rounded mean of the row's
        per-trial readings ('-' when the row has none there); the Mean table means the per-dataset
        trial means across datasets."""

        def hw_row(cells, readings):
            if not readings:
                return [*cells] + ["-"] * len(_HW_LABELS)
            return [*cells] + [str(round(np.mean([r[hw_label] for r in readings]))) for hw_label in _HW_LABELS]

        tables = []
        for dataset in datasets:
            grid = [[*headers, *_HW_LABELS]]
            for row in rows:
                readings = list(hw_by[(row, dataset)].values())
                grid.append(hw_row([*row[:-1], f"{row[-1]} ({len(readings)})"], readings))
            tables.append((DATASET_ALIAS2NAME[dataset], grid))
        xgrid = [[*headers, *_HW_LABELS, *_HW_CRASH_LABELS]]
        for row in rows:
            dataset_means = [
                {hw_label: np.mean([r[hw_label] for r in hw_by[(row, dataset)].values()]) for hw_label in _HW_LABELS}
                for dataset in datasets if hw_by[(row, dataset)]
            ]
            xgrid.append(hw_row(row, dataset_means) + [str(crash_totals[row][kind]) for kind in _CRASH_KINDS])
        tables.append(("Mean", xgrid))
        blocks = [(None, tables)]

        for seed in seeds:
            stables = []
            for dataset in datasets:
                grid = [[*headers, *_HW_LABELS]]
                for row in rows:
                    trial = hw_by[(row, dataset)].get(seed)
                    grid.append(hw_row(row, [] if trial is None else [trial]))
                stables.append((DATASET_ALIAS2NAME[dataset], grid))
            blocks.append((f"seed {seed}", stables))
        return blocks

    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    header_fill = PatternFill("solid", fgColor="EAEAEA")
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def write_sheet(ws, blocks, groups, bands, group_name, n_keys, styled=True):
        """groups: None -> each table's banner is its title merged across the full table width;
        else [(group_title, n_group_cols), ...] -> the title sits over the block's n_keys key
        columns (merged across them when there are several), followed by one grey merged
        group-header cell per group (e.g. 'Composite Scores' over the composite columns,
        'Primitive Scores' over the primitive columns). bands ([(title, grid)], each grid a header
        + one-value-row-per-key grid in the sheet's row order; [] for none) render as the config
        column bands at the left, one blank separator column after each, with the score blocks all
        shifted right past them -- vertically aligned with the aggregate block's bottom Mean table
        so the Mean's key columns label their rows; band cells likewise get no winner-bold/heatmap
        styling. styled=False skips the winner-bold/heatmap styling of data cells altogether (the
        hardware sheet's readings aren't scores)."""
        widths = {}  # col idx -> longest header/data cell text (banner/label cells overflow instead)

        campaign = ws.cell(row=1, column=1, value=f"{paths['root'].parent.name} - {ArtifactManager.dpath_phase.parent.name} ({group_name})")  # <campaign>/<phase>/
        campaign.font = bold
        campaign.alignment = left

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
                n_cols = len(grid[0])  # key columns + one col per label (the hw Mean table carries extra crash columns)
                title = ws.cell(row=row, column=col0, value=title_text)
                title.font = bold
                title.alignment = left
                for c in range(col0, col0 + n_cols):  # border every cell of the banner so the merged ranges' edges all render
                    ws.cell(row=row, column=c).border = border
                if groups is None:
                    ws.merge_cells(start_row=row, start_column=col0, end_row=row, end_column=col0 + n_cols - 1)
                else:
                    widths[col0] = max(widths.get(col0, 0), len(title_text))  # the title must fit its (key) column(s)
                    if n_keys > 1:
                        ws.merge_cells(start_row=row, start_column=col0, end_row=row, end_column=col0 + n_keys - 1)
                    gcol = col0 + n_keys
                    for group_title, n_group in groups:
                        for c in range(gcol, gcol + n_group):  # fill every cell so the merged range renders grey
                            ws.cell(row=row, column=c).fill = header_fill
                        gcell = ws.cell(row=row, column=gcol, value=group_title)
                        gcell.font = bold
                        gcell.alignment = center
                        ws.merge_cells(start_row=row, start_column=gcol, end_row=row, end_column=gcol + n_group - 1)
                        gcol += n_group
                row += 1

                styles = _col_styles(grid, bold_high and styled, n_keys)
                for r, grid_row in enumerate(grid):
                    for c, val in enumerate(grid_row):
                        cell = ws.cell(row=row, column=col0 + c, value=val)
                        cell.alignment = left if c < n_keys and r > 0 else center  # key cells left-aligned
                        cell.border = border
                        widths[col0 + c] = max(widths.get(col0 + c, 0), len(val))
                        if r == 0 or c < n_keys:
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

    def write_workbook(fpath, headers, rows, comps_by, hw_by, crash_totals, band_specs, banner):
        wb = Workbook()
        ws_map = wb.active
        ws_map.title = "Composite mAP"
        map_blocks, map_rows = build_blocks(headers, rows, comps_by, "map", map_labels)
        write_sheet(ws_map, map_blocks, map_groups, band_grids(band_specs, map_rows), banner, len(headers))
        acc_blocks, acc_rows = build_blocks(headers, rows, comps_by, "acc", acc_labels)
        write_sheet(wb.create_sheet("Composite I2T Accuracy"), acc_blocks, None, band_grids(band_specs, acc_rows), banner,
                    len(headers))
        # the hardware sheet shares the mAP sheet's row order (and so its overrides bands)
        write_sheet(wb.create_sheet("Hardware Performance"), build_hw_blocks(headers, map_rows, hw_by, crash_totals), None,
                    band_grids(band_specs, map_rows), banner, len(headers), styled=False)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        wb.save(fpath)

    dpath_stats = ArtifactManager.dpath_phase / "campaign_stats"
    band_specs_ac = [("Arm Overrides", band_keys("arm", rows_ac)), ("Coord Overrides", band_keys("coord", rows_ac))] if overrides else []
    band_specs_arms = [("Arm Overrides", band_keys("arm", rows_arms))] if overrides else []
    crash_totals_ac = {
        row: {kind: sum(crashes_by[(row, dataset)][kind] for dataset in datasets) for kind in _CRASH_KINDS}
        for row in rows_ac
    }
    # one workbook set per selection criterion: every score in <kind>/<criterion>/ (both score sheets)
    # comes from that criterion's best checkpoints (e.g. the map/ workbooks' accuracy sheet holds the
    # acc scores at the best-mAP checkpoint), with the banner naming the selection
    for criterion, selection_name in _SELECTION_NAMES.items():
        for group_key, group_name in _EVAL_GROUPS.items():
            comps_by = comps_all[criterion][group_key]
            banner = f"{group_name}; {selection_name}"
            write_workbook(dpath_stats / "arm_coords" / criterion / f"{group_key}.xlsx", ("Arm", "Coord"), rows_ac, comps_by,
                           hw_by, crash_totals_ac, band_specs_ac, banner)
            # arms: each arm at its best coord per dataset under this criterion x group (_arm_rows' rows
            # are rows_arms: which arms have trials doesn't depend on the criterion or group)
            rows, comps_arms, best = _arm_rows(arms, arm_coords, datasets, comps_by, criterion)
            hw_arms = {
                ((arm,), dataset): hw_by[((arm, best[(arm, dataset)]), dataset)] if (arm, dataset) in best else {}
                for (arm,) in rows
                for dataset in datasets
            }
            crash_totals_arms = {
                (arm,): {kind: sum(crashes_by[((arm, best[(arm, dataset)]), dataset)][kind]
                                   for dataset in datasets if (arm, dataset) in best)
                         for kind in _CRASH_KINDS}
                for (arm,) in rows
            }
            write_workbook(dpath_stats / "arms" / criterion / f"{group_key}.xlsx", ("Arm",), rows, comps_arms,
                           hw_arms, crash_totals_arms, band_specs_arms, banner)


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
    title_prefix = f"{_coord_label(ArtifactManager.dpath_coord)}, {DATASET_ALIAS2NAME[ArtifactManager.dataset]}"

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
    # a dev.batch_diagnostics component that was off never recorded its series, and its panel(s) and
    # height slot are omitted outright: model grad norm, ||delta theta||, sim-grad sums, the S stats panel
    has_grad_norm = len(data_epoch["grad_norm_model"]) == len(x_train)
    has_delta_norm = len(data_epoch["delta_norm_model"]) == len(x_train)
    has_grad_sum_sim = len(data_epoch["grad_sum_sim1"]) == len(x_train)
    has_grad_sum_sim2 = len(data_epoch["grad_sum_sim2"]) == len(x_train)  # sim-grad sums + loss2 active
    has_sim_stats = len(data_epoch["sim1_min"]) == len(x_train)
    # loss2 active (mix != 0) -> per-loss subscripted labels, and its sim-grad sum gets its own strip
    # between the loss1 strip and the S panel, so each series keeps its own y-scale. Checked over every
    # loss2 series that can exist, since with diagnostics off only its logit scalars survive (the
    # subscripted labels still apply)
    has_loss2 = any(len(data_epoch[key]) == len(x_train) for key in ("grad_sum_sim2", "sim2_min", "temp2", "bias2"))
    # base slots 6-9 (grad/step/sim-grad-sum/S) are kept per enabled component, the loss2 sim-grad-sum
    # strip slotted after loss1's
    height_ratios = [
        *height_ratios[:6],
        *([height_ratios[6]] if has_grad_norm else []),
        *([height_ratios[7]] if has_delta_norm else []),
        *([height_ratios[8]] if has_grad_sum_sim else []),
        *([0.5] if has_grad_sum_sim2 else []),
        *([height_ratios[9]] if has_sim_stats else []),
        *height_ratios[10:],
    ]
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
        ax0.legend(loc="lower right", ncol=len(ax0.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
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
        ax1.legend(loc="lower right", ncol=len(ax1.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    id_mode_scores = data_eval["scores"][group_key]["id"] if has_eval else {}
    comp_nshot = id_mode_scores["map"].get("n-shot", {}) if has_eval else {}
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax2, x_eval, comp_nshot, key, key)
        if comp_nshot:
            ax2.legend(loc="lower right", ncol=len(ax2.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
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
        ax3.legend(loc="lower right", ncol=len(ax3.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax3.grid(True)
    ax3.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
    comp_nshot_acc = id_mode_scores["acc"].get("n-shot", {}) if has_eval else {}
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax4, x_eval, comp_nshot_acc, key, key)
        if comp_nshot_acc:
            ax4.legend(loc="lower right", ncol=len(ax4.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
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
    ax5.legend(loc="lower left", ncol=len(ax5.get_legend_handles_labels()[0]), fontsize=fontsize_legend)
    ax5.grid(True)
    ax5.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    axes = [ax0, ax1, ax2, ax3, ax4, ax5]

    if has_grad_norm:
        ax6 = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax6.plot(x_train, data_epoch["grad_norm_model"], color="tab:orange")
        ax6.set_ylabel(r"$\|\nabla_{\theta}\mathcal{L}\|$", fontsize=fontsize_axes + 4)
        # CM mathtext has no bold symbol fonts; a thin stroke outline fakes the bold
        ax6.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground="black")])
        ax6.set_yscale("log")
        ax6.minorticks_on()
        ax6.grid(which="minor", axis="y")
        ax6.grid(True)
        ax6.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax6)

    if has_delta_norm:
        # the step the optimizer actually took, directly under the gradient that produced it: Adam
        # rescales per parameter, so the two need not track each other
        ax6b = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax6b.plot(x_train, data_epoch["delta_norm_model"], color="tab:brown")
        ax6b.set_ylabel(r"$\|\Delta\theta\|$", fontsize=fontsize_axes + 4)
        ax6b.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground="black")])
        ax6b.set_yscale("log")
        ax6b.minorticks_on()
        ax6b.grid(which="minor", axis="y")
        ax6b.grid(True)
        ax6b.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax6b)

    if has_grad_sum_sim:
        ax7 = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax7.plot(x_train, data_epoch["grad_sum_sim1"], color="tab:orange", linewidth=1.0)
        ax7.axhline(0.0, color="gray", linewidth=0.5)
        ax7.set_ylabel(r"$\sum \nabla_S \mathcal{L}_1$" if has_loss2 else r"$\sum \nabla_S \mathcal{L}$", fontsize=fontsize_axes - 1)
        ax7.yaxis.label.set_path_effects([patheffects.withStroke(linewidth=0.6, foreground="black")])
        ax7.grid(True)
        ax7.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax7)

    if has_grad_sum_sim2:
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

    if has_sim_stats:
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
