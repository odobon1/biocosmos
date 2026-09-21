"""
Campaign reporting/presentation: per-coord metric-stats aggregation + checkpoint selection
(_datasets/<dataset>/_arms/<arm>/_coords/<coord>/coord_stats/), the per-eval-group composite-score summary
tables + convergence plots at every cross-coord level -- per arm (_arms/<arm>/arm_metrics/performance/), per dataset
(_datasets/<dataset>/dataset_metrics/{arm_coords,arms}/performance/) -- each {map,acc}/<group>/{scores,convergence}.png,
the phase workbooks (phase_metrics/{arm_coords,arms}/performance/{map,acc}/<group>/metrics.xlsx), the test workbooks
(artifacts/<campaign>/test/map/test_<group>.xlsx, from the score files test.py writes), per-trial
learning-curve plots, each arm's coords' logit-scale panels stacked into strip figures
(_arms/<arm>/arm_metrics/coord_strips/) and each finished arm's best coord's copy of its
learning curves (_arms/<arm>/best_coord/) -- all
under one phase dir of the campaign (artifacts/<campaign>/<phase>/,
ArtifactManager.dpath_phase). The phase's phase_metadata.json 'matrix' ({dataset: {arm: [coords]}}) is the
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


# Every entry point here takes `eval_groups` -- {group key: reported name} for the groups the campaign
# has in play (utils.config.eval_groups, off reporting.yaml's frozen `eval` block). The key is the scores
# group key and the stats artifact's file/dir name, the value the name titles and banners carry; every
# stats table/xlsx/plot artifact is rendered once per group. `native` is always in the map, so anything
# that just needs SOME group's file on disk reads native's by name.

# checkpoint-selection criterion (BEST_CRITERIA / evals/_selected/ subdir) -> banner display name
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

# Max heatmap columns in a P/Q strip before adjacent ones are folded together. Each recorded batch
# contributes one histogram column; once there are more than this, every 2 consecutive columns are
# averaged into one (then every 4, every 8, ...) so the strip keeps a readable cell width instead of
# collapsing into a smear. Columns therefore oscillate between this and half of it as training runs.
P_HEATMAP_HORIZONTAL_THRESHOLD = 128
# density colormaps: blues for the predictions and purples for the targets, both rising off the
# white page so empty bins vanish into it. Deeper and more saturated than matplotlib's Blues/Purples,
# whose muted tops leave a sparse strip washed out. The purple stops are the blue ones hue-rotated
# at the same saturation and lightness, so the two strips read as one style in two hues.
_P_CMAP = LinearSegmentedColormap.from_list("p_density", ["#FFFFFF", "#4C7FE8", "#0A1FB0"])
_Q_CMAP = LinearSegmentedColormap.from_list("q_density", ["#FFFFFF", "#9A4CE8", "#5D0AB0"])
# most of the mass sits in one bin (a BCE run starts with every pair near 0), so a linear ramp would
# leave the rest invisible -- sqrt scaling lifts the sparse bins into view
_HIST_NORM = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
# panel background for the learning curves' line plots (the heatmap strips paint over their own)
_BG_LINE_PANEL = "#FAF7F0"
# the dL/dalpha strips' attributions, in series order: the whole term (*), then its positive- (+) and
# negative- (-) target-mass shares -- black for the total, Okabe-Ito blue / vermillion for the pair
# (legible together, and under red-green color blindness)
_DALPHA_ATTRIBUTIONS = (("(*)", "black"), ("(+)", "#0072B2"), ("(-)", "#D55E00"))
# The residual family (the dalpha res / sres / ires strips, the KL figure's E_R / E_SR / E_IR) is
# exp(-2 alpha)-small against the order-one entries of p* and y it is built from, and two marks say what a
# batch's values are worth. Both key off PROVENANCE (resid_paths: the row fractions [hard closed form, feasible
# exact zero, plain p* - y subtraction]), never off the value alone -- an unvalidated residual can read as a
# noise floor, as an exact zero, or as a perfectly plausible number, so its magnitude certifies nothing.
#   - unvalidated (hatched): some of the batch's rows came off the subtraction, the general solver path, which
#     carries no validated error estimate. Marked on EVERY such batch, whatever the value reads.
#   - unresolved (solid, over the hatch): additionally, the term sits below _DALPHA_RES_FLOOR of the full
#     term's magnitude, where the difference is known to be arithmetic (a one-ulp artifact of p*_cap - y,
#     ~1e-42 or ~1e-15, flipping between the two with alpha for no physical reason). A ONE-WAY test: below it
#     the value is certainly not to be read, above it NOTHING is established -- A_full is a gradient magnitude,
#     not a bound on the subtraction's error, and it shrinks with the fit while that error (set by p* and y
#     themselves) does not, so on a well-fitted batch the floor drops under any noise. A fitted alpha-25 batch
#     read kl_ir = 0 against a true 2.3e-22 with the floor down at 6e-30.
# A batch whose rows all came off an exact path carries neither mark however small it reads. The curve is
# always drawn -- a mark says how far to trust it, not that it is missing
_DALPHA_RES_FLOOR = 100.0 * np.finfo(np.float64).eps  # ~2.2e-14
_DALPHA_RES_COMPS = ("res", "sres", "ires")
_RESID_MARKS = {  # mark -> its axvspan style, under the curves and the grid
    "unvalidated": dict(facecolor="none", edgecolor="#C4B99F", hatch="////", linewidth=0, zorder=0),
    "unresolved": dict(color="#E8E1D3", linewidth=0, zorder=0),  # a shade down from _BG_LINE_PANEL
}

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
def update_metric_stats(eval_groups, spread_type):
    dpath_coord = ArtifactManager.dpath_coord
    dpath_stats = dpath_coord / "coord_stats"
    for criterion in BEST_CRITERIA:
        for group_key in eval_groups:
            metric_dicts = []
            for dpath_trial in sorted((dpath_coord / "_seeds").iterdir()):
                # update_chkpt_selection runs first and (re)writes evals/_selected/<criterion>/ for
                # exactly the completed trials, so their presence still marks the set to aggregate
                fpath_metrics = dpath_trial / f"evals/_selected/{criterion}/{group_key}.json"
                if not fpath_metrics.exists():
                    continue
                metrics = load_json(fpath_metrics)
                for key in ("chkpt", "loss_raw", "sim", "targ", "killed"):  # non-score fields aren't aggregated
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
    """The coord dir holding (arm, coord)'s trials on `dataset`: _datasets/<dataset>/_arms/<arm>/_coords/<coord>."""
    return ArtifactManager.dpath_phase / "_datasets" / dataset / "_arms" / arm / "_coords" / coord

def _coord_label(dpath_coord):
    """'<arm>/<coord>' of a coord dir (_datasets/<dataset>/_arms/<arm>/_coords/<coord>), for plot titles."""
    return f"{dpath_coord.parent.parent.name}/{dpath_coord.name}"

def _chkpt_dpaths(dpath_trial):
    """[evals/base, evals/eval1, .., evals/eval<n_chkpts>] once the trial's FINAL eval is on disk,
    else None -- the trial-completion signal for every coord-level aggregation here. Each eval
    file's chkpt field carries 'k/n_chkpts', so the highest-numbered eval dir says whether k has
    reached n_chkpts without n_chkpts being threaded in from config. A killed trial (kill_thresh;
    trial_metadata.json's killed = the eval index it stopped at) is complete at that eval instead, its
    list ending at evals/eval<killed>. (The old signal, a written evals/_selected/, can't serve any
    more: _selected/ is now derived from the completed trials rather than written by each trial for
    itself.)"""
    dpath_evals = dpath_trial / "evals"
    dpaths_eval = sorted(dpath_evals.glob("eval*"), key=lambda dpath: int(dpath.name[len("eval"):]))
    if not dpaths_eval:
        return None
    chkpt = load_json(dpaths_eval[-1] / "native.json")["chkpt"]  # native is always in play
    idx_eval, n_chkpts = chkpt.split()[0].split("/")
    complete = idx_eval == n_chkpts or int(idx_eval) == load_json(dpath_trial / "trial_metadata.json")["killed"]
    return [dpath_evals / "base", *dpaths_eval] if complete else None

def _trial_complete(dataset, arm, coord, seed):
    return _chkpt_dpaths(_dpath_coord(dataset, arm, coord) / "_seeds" / str(seed)) is not None

def _matrix():
    """The phase's planned matrix, phase_metadata.json's 'matrix' ({dataset: {arm: [coords]}}, datasets and arms in
    campaign order): the (dataset, arm, coord) combos every sweep gate and table row here keys off (every arm x
    coord in the screening phase, each arm's picked coord(s) in the qual phase)."""
    return load_json(ArtifactManager.dpath_phase / "phase_metadata.json")["matrix"]

def _matrix_arms(matrix, datasets):
    """The planned arms over `datasets`, in campaign order."""
    return list(dict.fromkeys(arm for dataset in datasets for arm in matrix[dataset]))

def _matrix_arm_coords(matrix, datasets):
    """The planned (arm, coord) pairs over `datasets` (their union): arms in campaign order, each arm's coords in
    first-seen order across `datasets`' lists."""
    return [(arm, coord) for arm in _matrix_arms(matrix, datasets)
            for coord in dict.fromkeys(coord for dataset in datasets for coord in matrix[dataset][arm])]

def arm_sweep_complete(seed, dataset, arm):
    """True once `seed` has a completed trial in EVERY planned coord of `arm` on `dataset` -- the arm's cycle
    of the seed sweep. The arm's arm_metrics/ re-render only at these points (train.py): every trial
    completion reselects only its own coord's checkpoint, so mid-cycle the arm's tables would mix
    coords reselected against different trial counts."""
    return all(_trial_complete(dataset, arm, coord, seed) for coord in _matrix()[dataset][arm])

def arm_complete(dataset, arm):
    """True once EVERY planned trial of `arm` on `dataset` is complete -- all its coords (the phase's
    matrix) across all the phase's seeds (phase_metadata.json's 'seeds'), i.e. the whole arm rather than
    one seed's cycle of it (arm_sweep_complete). Gates the arm's best_coord/ mirror
    (update_best_coord_curves)."""
    metadata = load_json(ArtifactManager.dpath_phase / "phase_metadata.json")
    return all(
        _trial_complete(dataset, arm, coord, seed)
        for coord in metadata["matrix"][dataset][arm]
        for seed in metadata["seeds"]
    )

def dataset_sweep_complete(seed, dataset):
    """True once `seed` has a completed trial in EVERY planned (arm, coord) on `dataset` -- the dataset's
    cycle of the seed sweep; gates the dataset_metrics/ re-render the way arm_sweep_complete gates arm_metrics/."""
    return all(
        _trial_complete(dataset, arm, coord, seed)
        for arm, coords in _matrix()[dataset].items()
        for coord in coords
    )

def seed_sweep_complete(seed):
    """True once `seed` has a completed trial in EVERY planned (dataset, arm, coord) of the phase -- i.e. one
    full pass of the matrix; gates the phase_metrics/ workbooks' re-render."""
    return all(
        _trial_complete(dataset, arm, coord, seed)
        for dataset, arms in _matrix().items()
        for arm, coords in arms.items()
        for coord in coords
    )

def _plot_chkpt_means(means, spreads, idx_best, n_trials, spread_type, score_name, title, fpath):
    chkpts = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(chkpts, means, color="blue", label=f"mean (n={n_trials})")
    ax.fill_between(chkpts, means - spreads, means + spreads, color="blue", alpha=0.2, label=f"± {spread_type}")
    # diamond + value on the selected point itself, matching the convergence plots' marker
    ax.plot(idx_best, means[idx_best], marker="D", color="blue", markersize=3, linestyle="none",
            zorder=5, label=f"selected ({idx_best})")
    ax.annotate(f"{means[idx_best]:.4f}", (idx_best, means[idx_best]), textcoords="offset points",
                xytext=(0, 9), ha="center", color="blue", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Checkpoint", fontsize=10, fontweight="bold")
    ax.set_ylabel(score_name, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)  # 'best' so the box dodges the curve and the selected-point marker
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

@rank0
def update_chkpt_selection(eval_groups, spread_type):
    """Checkpoint selection, per criterion x eval group, for this coord/dataset: the ONE checkpoint
    index every one of its trials is scored at, argmaxed over the across-trial MEAN curve rather than
    per trial -- argmax(mean(...)), not mean(argmax(...)). Each criterion curves the comp score it
    selects on (BEST_CRITERIA: map -> comp.map.all, acc -> comp.acc.i2t) at every checkpoint of every
    completed trial (_chkpt_dpaths); index 0 is the base eval, plotted and a candidate only under
    the base eval (checkpoint 0) competes like any other, so the winner is argmax over 0..n_chkpts
    with it and over 1..n_chkpts without, the earliest taking ties either way. Killed trials
    (kill_thresh: stopped at an earlier eval, so their curves are shorter) shape the mean curve
    only when every trial of the coord/dataset was killed; either way each is scored at its OWN
    argmax, since it may have no eval at the coord's index.

    The selection MOVES as trials land, so all of this is rewritten from scratch at each trial
    completion, for every completed trial of the coord/dataset -- not just the one that finished:
      - evals/_selected/<criterion>/<group>.json in each trial: a copy of ITS eval<idx_best> file
        (a killed trial's eval at its own argmax) plus a 'killed' flag, which the tables shade by
      - evals/_best/<criterion>/<group>.json in each trial: a copy of its eval at the trial's OWN
        argmax (same base and tie rules) -- per-trial reference; nothing aggregates it
      - coord_stats/<criterion>/<group>/chkpt_means.pkl ({'n_trials', 'chkpts', 'means', 'spreads',
        'idx_best'}) + chkpt_means.png (mean curve, mean +- spread band, selection marked)
      - coord_metadata.json's best_chkpt[<criterion>][<group>]
    """
    dpath_coord = ArtifactManager.dpath_coord
    best_chkpt = {criterion: {} for criterion in BEST_CRITERIA}
    for criterion, (score_key, metric) in BEST_CRITERIA.items():
        for group_key, group_name in eval_groups.items():
            trials = []  # (trial dir, its eval dirs, its comp score at each checkpoint, killed), completed trials only
            for dpath_trial in sorted((dpath_coord / "_seeds").iterdir()):
                dpaths_chkpt = _chkpt_dpaths(dpath_trial)
                if dpaths_chkpt is None:
                    continue
                trials.append((dpath_trial, dpaths_chkpt, [
                    float(load_json(dpath / f"{group_key}.json")["scores"]["comp"][score_key][metric])
                    for dpath in dpaths_chkpt
                ], load_json(dpath_trial / "trial_metadata.json")["killed"] is not None))

            if not trials:
                return

            # killed trials stopped short, so their curves are shorter than a full run's: they shape the
            # coord's selection only when every trial was killed (then all are equally short)
            live = [trial for trial in trials if not trial[3]] or trials
            curves = np.array([curve for _, _, curve, _ in live])
            n_trials = len(curves)
            # a lone trial has no ddof=1 spread -> flat (invisible) band
            spreads = np.zeros(curves.shape[1]) if n_trials == 1 else np.array([_spread(col, spread_type) for col in curves.T])
            means = curves.mean(axis=0)
            idx_best = int(np.argmax(means))  # the base eval (index 0) competes like any checkpoint; argmax keeps the earliest tie

            for dpath_trial, dpaths_chkpt, curve, killed in trials:
                # _selected: the trial's eval at the coord's selected checkpoint (what the stats read)
                # -- a killed trial, which may have no eval there, at its own best instead -- with the
                # killed flag added for the tables; _best: at the trial's OWN argmax, for reading a
                # single trial on its own
                idx_trial_best = int(np.argmax(curve))  # the base eval competes here too
                for name, idx in (("_selected", idx_trial_best if killed else idx_best), ("_best", idx_trial_best)):
                    dpath_dest = dpath_trial / "evals" / name / criterion
                    dpath_dest.mkdir(parents=True, exist_ok=True)
                    metrics = load_json(dpaths_chkpt[idx] / f"{group_key}.json")
                    save_json({**metrics, "killed": killed} if name == "_selected" else metrics, dpath_dest / f"{group_key}.json")

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

def _score_cells(score_maps, labels, spread_type):
    """One data cell per label, aggregated across `score_maps` (one score dict per completed trial),
    each read by its label's lowercased key: '-' (0 trials, or none carrying the score), 'XX.XX'
    (1 trial, mean), or 'XX.XX ± XX.XX' (>1 trial, mean ± spread)."""
    cells = []
    for label in labels:
        key = label.lower()
        # an n-shot bucket with no classes in this dataset's eval partition is absent from its files
        nums = np.array([float(score_map[key]) for score_map in score_maps if key in score_map]) * 100
        if len(nums) == 0:
            cells.append("-")
        elif len(nums) == 1:
            cells.append(f"{nums[0]:.2f}")
        else:
            cells.append(f"{nums.mean():.2f} ± {_spread(nums, spread_type):.2f}")
    return cells

def _stats_table_grid(headers, labels, rows, score_key, spread_type):
    """Build a composite-score table's cell grid from [(row key, [_collect_comps entry per completed
    trial]), ...]: a header row of the row-key column names in `headers` (e.g. ('Arm', 'Coord')) + one
    column per label in `labels`, then one row per entry -- its key's cells (the trial count appended
    to the last one: '<coord> (n_trials)', with the killed count when any: '<coord> (n_trials, n
    killed)') + each label's _score_cells cell over the entries' `score_key` score maps. Also returns
    the indices of the grid rows holding a killed trial (kill_thresh), which the renderers shade
    yellow in place of the heatmap ramp."""
    grid = [[*headers, *labels]]
    killed_rows = set()
    for row_key, comps in rows:
        n_killed = sum(comp["killed"] for comp in comps)
        count = f"{len(comps)}, {n_killed} killed" if n_killed else f"{len(comps)}"
        grid.append([*row_key[:-1], f"{row_key[-1]} ({count})",
                     *_score_cells([comp[score_key] for comp in comps], labels, spread_type)])
        if n_killed:
            killed_rows.add(len(grid) - 1)
    return grid, killed_rows

def _comp_entry(scores_grp):
    """One trial's {'map': ..., 'acc': ..., 'nshot': [...]} entry from a metrics file's per-group
    scores subtree -- 'map'/'acc' flat label->score dicts merging the comp scores with the
    per-partition primitives ('id i2t' ... 'ood t2i') and the ID partition's n-shot bucket scores
    (keyed by lowercased bucket name), so table labels map to keys by lowercasing; 'nshot' the
    trial's bucket names in the file's (split) order. The eval writes the 'n-shot' dicts only for
    buckets with classes in the eval partition (the dev split drops some), so a bucket can be
    absent from a dataset's files."""
    nshot_map = scores_grp["id"]["map"].get("n-shot", {})
    nshot_acc = scores_grp["id"]["acc"].get("n-shot", {})
    return {
        "map": {**scores_grp["comp"]["map"],
                **{f"{p} {m}": scores_grp[p]["map"][m] for p in ("id", "ood") for m in ("i2t", "i2i", "t2i")},
                **{b.lower(): v for b, v in nshot_map.items()}},
        "acc": {**scores_grp["comp"]["acc"],
                **{f"{p} i2t": scores_grp[p]["acc"]["i2t"] for p in ("id", "ood")},
                **{b.lower(): v for b, v in nshot_acc.items()}},
        "nshot": [b.lower() for b in {**nshot_map, **nshot_acc}],
    }

def _collect_comps(group_keys, arm_coords, datasets, criterion):
    """Per eval group in `group_keys`, each (arm, coord) x dataset's completed-trial score maps, keyed
    by trial seed (the trial dir name; empty dict -> no trials yet):
    comps_all[group_key][((arm, coord), dataset)][seed] is a _comp_entry plus 'killed' (the file's flag:
    the trial was killed, kill_thresh). Each group reads its own best-checkpoint metrics file for the
    given selection criterion (evals/_selected/<criterion>/), whose presence is also the completion
    signal, same as update_metric_stats."""
    comps_all = {group_key: {} for group_key in group_keys}
    for arm, coord in arm_coords:
        for dataset in datasets:
            dpath_coord = _dpath_coord(dataset, arm, coord)
            comps = {group_key: {} for group_key in group_keys}
            if (dpath_coord / "_seeds").exists():
                for dpath_trial in sorted((dpath_coord / "_seeds").iterdir()):
                    for group_key in group_keys:
                        fpath_metrics = dpath_trial / f"evals/_selected/{criterion}/{group_key}.json"
                        if fpath_metrics.exists():
                            metrics = load_json(fpath_metrics)
                            comps[group_key][dpath_trial.name] = {**_comp_entry(metrics["scores"]), "killed": metrics["killed"]}
            for group_key in group_keys:
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
    strings (numerator taken) -- plus 'killed', whether the trial was killed (kill_thresh; the
    file's killed field is the eval index it stopped at, else null). A written best-checkpoint
    (evals/_selected/map/) metrics file is the completion signal, same as _collect_comps (native.json
    stands in for the set -- all per-group files are materialized together at trial end). Also each
    (arm, coord) x dataset's n_crashes ({'ram'/'vram'/'other' -> int}, summed across its seeds, completed
    or not) from its coord_metadata.json (_datasets/<dataset>/_arms/<arm>/_coords/<coord>/; zeros for a
    dataset the coord never launched in), whose counters survive the trial-dir wipes that reset
    trial_metadata's."""
    hw_by, crashes_by = {}, {}
    for arm, coord in arm_coords:
        for dataset in datasets:
            dpath_coord = _dpath_coord(dataset, arm, coord)
            trials = {}
            if (dpath_coord / "_seeds").exists():
                for dpath_trial in sorted((dpath_coord / "_seeds").iterdir()):
                    if (dpath_trial / "evals/_selected/map/native.json").exists():
                        meta = load_json(dpath_trial / "trial_metadata.json")
                        trials[dpath_trial.name] = {
                            "Time Trial": float(meta["runtime"]["trial"]),
                            "Mean Time Train": float(meta["runtime"]["train"]["mean"]),
                            "Mean Time Eval": float(meta["runtime"]["eval"]["mean"]),
                            "Peak RAM": float(meta["memory"]["ram"].split("/")[0]),
                            "Peak VRAM": float(meta["memory"]["vram"].split("/")[0]),
                            "killed": meta["killed"] is not None,
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
    matrix = _matrix()
    datasets = list(matrix)
    arm_coords = _matrix_arm_coords(matrix, datasets)
    comps_by = _collect_comps(("native",), arm_coords, datasets, "map")["native"]
    return _best_coords(arm_coords, datasets, comps_by, "map")

def _curve(dataset, arm, coord, criterion, group_key):
    """(means, idx_best) of the coord's across-trial mean curve on `dataset` for one criterion x eval
    group -- update_chkpt_selection's coord_stats/<criterion>/<group>/chkpt_means.pkl, written for
    exactly the coords with completed trials there."""
    chkpt_means = load_pickle(_dpath_coord(dataset, arm, coord) / "coord_stats" / criterion / group_key / "chkpt_means.pkl")
    return chkpt_means["means"], chkpt_means["idx_best"]

def _col_styles(grid, bold_high, n_keys):
    """Per-column data-cell styling for one rendered table, shared by the png and xlsx tables:
    styles[c] for each score-label column c (the first n_keys columns are the row keys) -- row ->
    mean for numeric cells ('-' and the hw sheet's 'X' skipped) and the bold-winner rows (highest
    mean, ties included; empty unless bold_high)."""
    styles = {}
    for c in range(n_keys, len(grid[0])):
        means = {r: float(grid[r][c].split(" ± ")[0]) for r in range(1, len(grid)) if grid[r][c] not in ("-", "X")}
        winners = set()
        if bold_high and means:
            top = max(means.values())
            winners = {r for r, m in means.items() if m == top}
        styles[c] = (means, winners)
    return styles

# heatmap shade of a table row holding a killed trial (kill_thresh): a sickly yellow in place of the
# white -> green score ramp, so the scores still read but visibly come from a run cut short
_KILLED_HEX = "DCE06E"

def _heat_hex(mean):
    """Heatmap cell color as 'RRGGBB': linear white (#ffffff) -> #4caf50 interpolation over a
    fixed 0.00 -> 100.00. The top of the ramp is a mid-tone leaf green rather than a saturated one,
    so a high-scoring cell reads as shaded paper and its black text keeps its contrast."""
    frac = max(0.0, min(1.0, mean / 100.0))
    return "".join(f"{round(255 - (255 - channel) * frac):02X}" for channel in (0x4C, 0xAF, 0x50))

def _render_stats_table(grid, n_keys, title, fpath, bold_high, heatmap, killed_rows):
    # matplotlib lays table rows out at a fixed 10pt * 1.2 = 1/6in apiece whatever the figure size
    # (1/4in after the 1.5x scale below), so the axes is made exactly that tall and fills the
    # figure: the table then spans it edge to edge for any row count and the title's pad is the
    # whole gap above it, instead of a tall table overflowing the axes and running over the title.
    fig, ax = plt.subplots(figsize=(1.2 + 1.5 * (len(grid[0]) - 1), 0.25 * len(grid)))
    fig.subplots_adjust(bottom=0, top=1)
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
            cell.set_facecolor(f"#{_KILLED_HEX if row in killed_rows else _heat_hex(means[row])}")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _plot_convergence(curves, idx_win, score_name, title, fpath):
    """[(row key, mean curve, its selected index), ...] overlaid on one log-x axes: every row grey,
    curves[idx_win] redrawn in black on top (legend: its key's cells joined by '/', e.g. 'hp/LR-1e-5')
    with its selection marked by a diamond and a red dashed line across the plot at its score.
    Checkpoint 0 (the base eval) is a selection candidate like any other but has no place on a log
    axis, so the axis is symlog: its linear stretch below 1 holds checkpoint 0, and the curves start
    there."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("symlog", linthresh=1, linscale=0.5, subs=np.arange(2, 10))
    # plain checkpoint numbers on the log axis, not the default 10^k scientific labels; the minor
    # ticks carry most of them at these ranges (a campaign's n_chkpts is a couple of decades at most)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%g"))
    ax.set_xlim(0, max(len(means) - 1 for _, means, _ in curves))  # no margin below checkpoint 0
    for i, (_, means, _) in enumerate(curves):  # every row grey, one legend entry for the pack
        ax.plot(np.arange(len(means)), means, color="grey", alpha=0.6, linewidth=1,
                label=f"all ({len(curves)})" if i == 0 else None)

    row, means, idx_best = curves[idx_win]
    ax.axhline(means[idx_best], color="red", linestyle="--", linewidth=1)
    ax.plot(np.arange(len(means)), means, color="black", linewidth=1, zorder=4, label="/".join(row))
    # diamond + value on the selected point itself
    ax.plot(idx_best, means[idx_best], marker="D", color="black", markersize=3, linestyle="none",
            zorder=5, label=f"selected ({idx_best})")
    ax.annotate(f"{means[idx_best]:.4f}", (idx_best, means[idx_best]), textcoords="offset points",
                xytext=(0, 9), ha="center", color="black", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Checkpoint", fontsize=10, fontweight="bold")
    ax.set_ylabel(score_name, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)  # 'best' so the box dodges the curves and the selected-point marker
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _render_stats_pngs(dpath_stats, headers, rowset_of, dataset, subject, eval_groups, labels, spread_type, bold_high,
                       ordered, heatmap):
    """Render one score table + one convergence plot per selection criterion x eval group for ONE
    dataset: dpath_stats/{map,acc}/<group>/scores.png -- map/ the comp mAP table (All/ID/OOD/I2T/I2I/T2I
    score columns), acc/ the comp I2T accuracy table (single I2T column), each plus the enabled
    supplemental columns (labels = (mAP labels, acc labels) per _score_labels) and each sourcing its
    own criterion's best checkpoints (map/ from evals/_selected/map/, acc/ from evals/_selected/acc/) -- and
    convergence.png. rowset_of(criterion, group_key) -> (rows, comps_by, curves): the table's row keys
    (tuples of len(headers) cells, e.g. (arm, coord) under headers ('Arm', 'Coord')), comps_by[(row,
    dataset)] their completed-trial score maps by seed (_collect_comps entries), and curves [(row,
    means, idx_best)] their across-trial mean curves (_curve) for the convergence plot.

    Tables: one '<key cells> (n_trials)' row per row key, stats aggregated across its completed
    trials, titled '<score name> -- <subject> (<group name>)'. bold_high bolds each score column's
    highest-mean cell (ties included; '-' cells ignored), ordered orders the rows by the table's own
    metric's mean over THIS dataset's completed trials (map tables by the mAP 'All' column, acc tables
    by the acc 'I2T' column) -- localized per dataset and per group, independent of the cross-dataset
    order used in the workbooks -- heatmap shades score cells white->#4caf50 over a fixed
    0.00->100.00 (as in update_phase_metrics), rows holding a killed trial (kill_thresh) in the
    killed yellow instead. Convergence plots overlay every row's curve on one
    log-scaled checkpoint axis, all grey, with the winner -- the highest mean at its OWN selected
    checkpoint, ties to the first row -- black on top, its selection marked (_plot_convergence, from
    checkpoint 0, the base eval); no plot when there are no curves."""
    map_labels, acc_labels = labels
    for criterion in BEST_CRITERIA:
        score_labels = {"map": map_labels, "acc": acc_labels}[criterion]
        score_name = _CRITERION_SCORE_NAMES[criterion]
        for group_key, group_name in eval_groups.items():
            rows, comps_by, curves = rowset_of(criterion, group_key)
            if ordered:
                # localized order: this dataset's per-row trial means (single-dataset degenerate
                # case of _cross_dataset_means), not the cross-dataset means backing the workbook order
                means = _cross_dataset_means(rows, (dataset,), comps_by, criterion, (score_labels[0],))
                rows = _order_rows(rows, means, score_labels[0])
            dpath_group = dpath_stats / criterion / group_key
            dpath_group.mkdir(parents=True, exist_ok=True)
            title_suffix = f" -- {subject} ({group_name})"
            grid, killed_rows = _stats_table_grid(
                headers,
                score_labels,
                [(row, list(comps_by[(row, dataset)].values())) for row in rows],
                criterion,
                spread_type,
            )
            _render_stats_table(grid, len(headers), f"{score_name}{title_suffix}", dpath_group / "scores.png", bold_high, heatmap,
                                killed_rows)
            if curves:
                idx_win = max(range(len(curves)), key=lambda i: curves[i][1][curves[i][2]])
                _plot_convergence(curves, idx_win, score_name, f"{score_name} Convergence{title_suffix}",
                                  dpath_group / "convergence.png")

@rank0
def update_arm_metrics(dataset, arm, eval_groups, spread_type, bold_high, ordered, heatmap, supp_scores):
    """Render `arm`'s cross-coord tables/plots for `dataset`:
    _datasets/<dataset>/_arms/<arm>/arm_metrics/performance/{map,acc}/<group>/{scores,convergence}.png (see
    _render_stats_pngs) -- one 'Coord' row per planned coord of the arm (the phase's matrix) with >= 1
    completed trial in this dataset (coords without local trials are omitted: no blank rows in the pngs), titled
    '<score name> -- <arm>, <dataset> (<group>)'. An arm with no dir on this dataset (no trial of it
    launched there) is skipped. Rendered at the end of the arm's seed cycle (train.py,
    arm_sweep_complete) and unconditionally by the runner on exit / tools.regen_stats. The arm's
    best_coord/ mirror rides along on the same calls (update_best_coord_curves)."""
    dpath_arm = ArtifactManager.dpath_phase / "_datasets" / dataset / "_arms" / arm
    if not dpath_arm.exists():
        return
    coords = _matrix()[dataset][arm]
    arm_coords = [(arm, coord) for coord in coords]
    comps_all = {criterion: _collect_comps(eval_groups, arm_coords, (dataset,), criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _selected files, all criteria and groups
    # together, so row presence is criterion- and group-independent
    comps_ref = next(iter(comps_all["map"].values()))
    rows = [(coord,) for coord in coords if comps_ref[((arm, coord), dataset)]]

    def rowset(criterion, group_key):
        comps_by = {((coord,), dataset): comps_all[criterion][group_key][((arm, coord), dataset)] for (coord,) in rows}
        curves = [((coord,), *_curve(dataset, arm, coord, criterion, group_key)) for (coord,) in rows]
        return rows, comps_by, curves

    _render_stats_pngs(dpath_arm / "arm_metrics" / "performance", ("Coord",), rowset, dataset,
                       f"{arm}, {DATASET_ALIAS2NAME[dataset]}", eval_groups,
                       _score_labels(supp_scores, _nshot_names(comps_all)), spread_type, bold_high, ordered, heatmap)
    update_best_coord_curves(dataset, arm)
    update_coord_strips(dataset, arm, spread_type)

@rank0
def update_best_coord_curves(dataset, arm):
    """Mirror the learning curves of `arm`'s best coord on `dataset` to
    _datasets/<dataset>/_arms/<arm>/best_coord/learning_curves/<seed>/ -- every trial of that coord,
    its learning_curves/ dir copied whole ({general,alpha,logalpha,KL}.png + scores/<group>.png) -- so the
    arm's answer reads off one dir without first working out which coord won.

    Best is the coord with the highest across-trial mean Native mAP composite All at its selected
    checkpoint (_best_coords on criterion 'map' x eval group 'native' -- the figure the qual phase picks
    its arms' coords by, pick_best_coords), ties to the first in campaign order.

    The dir stands for a FINISHED arm, so it exists only while every planned trial of the arm is in
    (arm_complete: all its coords x all the phase's seeds). It is rebuilt from scratch on each call and
    dropped whenever the arm is short of its plan again -- a coord or a seed added to it -- to be written
    once more when the arm completes over the new plan; the runner drops it at the moment such an edit is
    applied (campaign_runner._apply_plan) rather than at the arm's next completed trial. Screening phase
    only: the qual matrix reduces each arm to its pick(s), so there this would just duplicate the pick's
    own coord dir, and the trainval phase runs no evals to rank coords by."""
    if ArtifactManager.dpath_phase.name != "_screen":
        return
    dpath_best = ArtifactManager.dpath_phase / "_datasets" / dataset / "_arms" / arm / "best_coord"
    if dpath_best.exists():
        shutil.rmtree(dpath_best)
    if not arm_complete(dataset, arm):
        return
    arm_coords = [(arm, coord) for coord in _matrix()[dataset][arm]]
    comps_by = _collect_comps(("native",), arm_coords, (dataset,), "map")["native"]
    coord = _best_coords(arm_coords, (dataset,), comps_by, "map")[(arm, dataset)]
    for dpath_trial in sorted((_dpath_coord(dataset, arm, coord) / "_seeds").iterdir()):
        shutil.copytree(dpath_trial / "learning_curves", dpath_best / "learning_curves" / dpath_trial.name)

# the learning-curve strip figures (update_coord_strips), one per parameterization of the logit scale:
# (the series it plots, its target-implied bound trio, its symbol, the figure's name)
_STRIP_FIGURES = (("scale", "alpha_req", r"\alpha", "alpha"),
                  ("logit_scale", "log_alpha_req", r"\log \alpha", "logalpha"))

def _req_labels(reqs, sym):
    """{stat: legend label} for a scale panel's target-implied bound trio (`reqs`: {stat: values}),
    naming only the stats with something to draw -- a stat non-finite on every batch draws nothing, and
    an entry for a line that is nowhere on the panel is one the reader cannot place. The min and max
    bracket the band in the same solid style, so one entry covers the pair, naming whichever of the two
    draws and riding on that line (the other goes unlabelled, and an unlabelled line is left out of the
    legend); an empty map means the trio draws nothing at all. Shared by plot_alpha_curves' panels and
    the strip figures."""
    drawn = [stat for stat, vals in reqs.items() if np.isfinite(vals).any()]
    band = [stat for stat in ("min", "max") if stat in drawn]
    labels = {band[0]: rf"${sym}_{{\text{{req}}}}$ ({', '.join(band)})"} if band else {}
    if "mean" in drawn:
        labels["mean"] = rf"${sym}_{{\text{{req}}}}$ (mean)"
    return labels

def _strip_data(panels, scale_key, req_key, spread_type):
    """One coord's strip, off its `panels` [(x_train, data_epoch)] -- (x, scale, reqs): `scale` the
    (values, spread) of the logit-scale series and `reqs` {stat: (values, spread)} its target-implied
    bound trio (empty where the trials don't record it -- a BCE-family loss). With `spread_type` None
    (the per-seed figures, one trial apiece) every spread is None and the values are that trial's own;
    otherwise (the agg figure) every line is the mean over the coord's trials and every spread their
    _spread band, flat for a lone trial as the chkpt-mean curves' is. Series are truncated to the
    shortest, so a trial the kill threshold stopped early shortens its coord's strip rather than
    dropping out of it."""
    n = min(len(x) for x, _ in panels)

    def agg(key):
        vals = np.array([np.asarray(data[key], dtype=float)[:n] for _, data in panels])
        if spread_type is None:
            return vals[0], None
        spread = np.zeros(n) if len(vals) == 1 else np.array([_spread(col, spread_type) for col in vals.T])
        return vals.mean(axis=0), spread

    reqs = {}
    if all(len(data[f"{req_key}_max"]) == len(data[scale_key]) for _, data in panels):
        reqs = {stat: agg(f"{req_key}_{stat}") for stat in ("min", "mean", "max")}
    return panels[0][0][:n], agg(scale_key), reqs

def _render_strips(strips, fpath_plot, sym, plot_title, spread_type, fontsize_axes, fontsize_ticks,
                   fontsize_legend, subplot_border_width, fig_width, height_fixed, height_panel):
    """One figure of scale strips, `strips` [(coord, x, scale, reqs)] top to bottom (_strip_data): the
    scale panel plot_alpha_curves gives each trial -- the scale in purple over its bound trio in red,
    the min and max solid and the mean dashed -- one per coord of an arm, over one shared epoch axis. A
    (values, spread) pair with a spread shades it around its line, so on the agg figure every line carries
    its own across-trial band; one '± <spread_type>' entry names the convention for all of them.

    Each strip is named by its coord in bold, the names alternating left / right down the figure with each
    strip's y ticks on its name's side, so neither margin carries every label; the names on a side start
    at a common x (their pads make up the difference between each strip's tick labels and the widest
    block that side has to clear). The coords' top-level dimension -- combo-group names join with '_' and
    the first group varies slowest, so LR-5.0e-6_Alpha-{100,10,1} groups under its LR -- boxes each run of
    strips sharing it in a double-width border (_finish_curves' axes_blocks). The legend is the figure's,
    not a panel's (the strips draw the same series), pooled by label and laid out in one row above the top
    strip, right-aligned opposite the title."""
    fig = plt.figure(figsize=(fig_width, height_fixed + len(strips) * height_panel))
    gs = gridspec.GridSpec(len(strips), 1, hspace=0)
    axes = []

    for coord, x, (vals_scale, spread_scale), reqs in strips:
        ax = fig.add_subplot(gs[len(axes), 0], sharex=axes[0] if axes else None)
        # the scale rides above its bound lines (zorder): a softmax target mapping pinned to the scale
        # (utils.loss.InfoNCECriterion._tsm) makes the bound the scale itself wherever a row's targets
        # span the full similarity range, and the panel is about the scale -- drawn under, it would
        # vanish beneath the trio exactly where they coincide
        ax.plot(x, vals_scale, color="tab:purple", label=rf"${sym}$", zorder=3)
        if spread_scale is not None:
            ax.fill_between(x, vals_scale - spread_scale, vals_scale + spread_scale, color="tab:purple",
                            alpha=0.2, label=f"± {spread_type}")
        labels_req = _req_labels({stat: vals for stat, (vals, _) in reqs.items()}, sym)
        for stat, linestyle in (("min", "-"), ("mean", "--"), ("max", "-")):
            if stat in reqs:
                vals, spread = reqs[stat]
                ax.plot(x, vals, color="red", linestyle=linestyle, linewidth=1.0, label=labels_req.get(stat))
                if spread is not None:
                    ax.fill_between(x, vals - spread, vals + spread, color="red", alpha=0.15)
        # the coord names the strip, horizontal (a coord name reads badly rotated) and bold, on the side
        # its y ticks take -- alternating down the figure, so one margin doesn't carry every label
        side = "left" if len(axes) % 2 == 0 else "right"
        ax.set_ylabel(coord, rotation=0, ha="left", va="center", fontsize=fontsize_axes, fontweight="bold")
        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)
    axes[-1].set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")
    axes[-1].tick_params(labelbottom=True)

    # the names on a side start at a common x. A name sits labelpad points beyond its own tick labels, so
    # the pads make up each strip's difference from the widest block that side must clear -- the tick
    # labels alone on the right, where the name runs away from the panel, the tick labels plus the name on
    # the left, where it runs back toward one. The draw is what fills the tick labels in and lays them
    # out; widths are read off an explicit renderer and converted to points, labelpad's unit (so the save
    # dpi is moot).
    fig.canvas.draw()
    renderer, to_points = fig.canvas.get_renderer(), 72 / fig.dpi
    ticks = [max((t.get_window_extent(renderer).width for t in ax.get_yticklabels()), default=0.0) * to_points
             for ax in axes]
    names = [ax.yaxis.label.get_window_extent(renderer).width * to_points for ax in axes]
    for side, idxs in (("left", range(0, len(axes), 2)), ("right", range(1, len(axes), 2))):
        idxs = list(idxs)
        if idxs:
            span = max(ticks[i] + names[i] * (side == "left") for i in idxs)
            for i in idxs:
                axes[i].yaxis.labelpad = span - ticks[i] + 6

    handles = {}  # label -> handle, first strip to draw it winning: a coord whose loss records no bounds
    for ax in axes:                              # leaves them to a coord whose loss does
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    axes[0].legend(handles=list(handles.values()), loc="lower right", bbox_to_anchor=(1.0, 1.0),
                   ncol=len(handles), fontsize=fontsize_legend)
    _finish_curves(fig, axes, [], {}, plot_title, fpath_plot, fontsize_legend, subplot_border_width,
                   axes_blocks=_strip_blocks(strips, axes), box_blocks=True, alternate_sides=False)

def _strip_blocks(strips, axes):
    """The strip runs to box, grouped by the coords' top-level dimension: combo-group names join with '_'
    and the campaign crosses the first group slowest, so the component before the first '_' is the coarsest
    sweep the names carry -- LR-5.0e-6_Alpha-{100,10,1}, LR-5.0e-5_Alpha-{100,10,1} groups under its LR.
    A coord set with no such level (one combo group, so every name its own group) gets no boxes rather
    than a box per strip."""
    blocks, level = [], None
    for (coord, *_), ax in zip(strips, axes):
        if coord.split("_")[0] != level:
            level = coord.split("_")[0]
            blocks.append([])
        blocks[-1].append(ax)
    return [] if len(blocks) == len(axes) else blocks

@rank0
def update_coord_strips(dataset, arm, spread_type, fontsize_axes=12, fontsize_ticks=8, fontsize_legend=8,
                      subplot_border_width=1, fig_width=10, height_fixed=1.8, height_panel=0.8):
    """Render `arm`'s logit-scale strips for `dataset` --
    _datasets/<dataset>/_arms/<arm>/arm_metrics/coord_strips/{seeds/<seed>,agg}/{alpha,logalpha}.png:
    the scale panel of every planned coord of the arm (the phase's matrix) with a completed trial there,
    stacked into one figure per parameterization (_STRIP_FIGURES: alpha, the scale the logits carry;
    logalpha, the log-scale parameter the model learns), so an arm's coords read against each other on one
    epoch axis instead of one figure apiece. seeds/<seed>/ draws that seed's trial per coord, agg/ the mean
    over the coord's trials with a spread band on every line (_strip_data, _render_strips). The style
    defaults match plot_metrics', so a strip is the size its panel has in the trial's own figure.

    The epoch axis is rebuilt the way the trial figures' is (n_samps_seen / samps_per_epoch), reading
    samps_per_epoch off the recorded artifacts -- coord_metadata.json's horizon (the coord's sample volume)
    over trial_metadata.json's n_epochs -- rather than rebuilding a TrainConfig for it. A coord that doesn't
    track a figure's scale (a frozen logit scalar: the series is empty) is left out of that figure, and an
    arm with no coord tracking it gets no figure at all. Rendered with the arm's tables (update_arm_metrics)."""
    dpath_strips = (ArtifactManager.dpath_phase / "_datasets" / dataset / "_arms" / arm / "arm_metrics"
                    / "coord_strips")
    panels = {}  # coord -> {seed: (x_train, data_epoch)}, completed trials only, campaign coord order
    for coord in _matrix()[dataset][arm]:
        dpath_coord = _dpath_coord(dataset, arm, coord)
        if not (dpath_coord / "_seeds").exists():
            continue
        for dpath_trial in sorted((dpath_coord / "_seeds").iterdir()):
            if _chkpt_dpaths(dpath_trial) is None:
                continue
            # read off the trial's own completion, not the coord dir: that exists from the moment a trial
            # launches, whereas coord_metadata.json is the subprocess's to write and a crash can precede it
            n_samps = load_json(dpath_coord / "coord_metadata.json")["horizon"]["n_samps"]["total"]
            samps_per_epoch = n_samps / load_json(dpath_trial / "trial_metadata.json")["progress"]["n_epochs"]
            data_epoch = load_pickle(dpath_trial / "data_trial.pkl")["epoch"]
            # as plot_metrics builds it: each batch's metrics are measured on the pre-step model, so the
            # curve anchors at 0 and the last stamp is dropped
            x_train = np.array([0.0, *(v / samps_per_epoch for v in data_epoch["n_samps_seen"][:-1])])
            panels.setdefault(coord, {})[dpath_trial.name] = (x_train, data_epoch)

    for scale_key, req_key, sym, name in _STRIP_FIGURES:
        tracked = {
            coord: {seed: panel for seed, panel in by_seed.items() if len(panel[1][scale_key]) == len(panel[0])}
            for coord, by_seed in panels.items()
        }
        tracked = {coord: by_seed for coord, by_seed in tracked.items() if by_seed}
        if not tracked:
            continue
        plot_title = f"{arm}\n{DATASET_ALIAS2NAME[dataset]}"
        style = (fontsize_axes, fontsize_ticks, fontsize_legend, subplot_border_width, fig_width, height_fixed,
                 height_panel)
        for seed in sorted({seed for by_seed in tracked.values() for seed in by_seed}, key=int):
            strips = [(coord, *_strip_data([by_seed[seed]], scale_key, req_key, None))
                      for coord, by_seed in tracked.items() if seed in by_seed]
            _render_strips(strips, dpath_strips / "seeds" / seed / f"{name}.png", sym, plot_title, spread_type, *style)
        strips = [(coord, *_strip_data(list(by_seed.values()), scale_key, req_key, spread_type))
                  for coord, by_seed in tracked.items()]
        _render_strips(strips, dpath_strips / "agg" / f"{name}.png", sym, plot_title, spread_type, *style)

@rank0
def update_dataset_metrics(dataset, eval_groups, spread_type, bold_high, ordered, heatmap, supp_scores):
    """Render `dataset`'s cross-arm tables/plots:
    _datasets/<dataset>/dataset_metrics/{arm_coords,arms}/performance/{map,acc}/<group>/{scores,convergence}.png (see
    _render_stats_pngs). arm_coords/ has one ('Arm', 'Coord') row per planned (arm, coord) (the phase's
    matrix) with >= 1 completed trial in this dataset. arms/ has one 'Arm' row per arm, each at its BEST coord for this
    dataset -- per criterion x group, the coord with the highest across-trial mean of the criterion's
    comp score among the arm's coords with completed trials here, ties to the first in campaign order
    (_best_coords) -- so both its table row and its convergence curve are that coord's; arms with no
    completed trial here are omitted. arms/ is skipped in the qual phase, whose matrix reduces each
    arm to its pick(s) -- there it would just duplicate arm_coords/. A dataset with no dir (no trial launched on it) is skipped.
    Rendered at the end of the dataset's seed cycle (train.py, dataset_sweep_complete) and
    unconditionally by the runner on exit / tools.regen_stats."""
    dpath_dataset = ArtifactManager.dpath_phase / "_datasets" / dataset
    if not dpath_dataset.exists():
        return
    matrix = _matrix()
    arms = list(matrix[dataset])
    arm_coords = _matrix_arm_coords(matrix, (dataset,))
    comps_all = {criterion: _collect_comps(eval_groups, arm_coords, (dataset,), criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _selected files, all criteria and groups
    # together, so row presence is criterion- and group-independent
    comps_ref = next(iter(comps_all["map"].values()))
    rows_ac = [row for row in arm_coords if comps_ref[(row, dataset)]]
    labels = _score_labels(supp_scores, _nshot_names(comps_all))
    subject = DATASET_ALIAS2NAME[dataset]
    dpath_stats = dpath_dataset / "dataset_metrics"

    def rowset_arm_coords(criterion, group_key):
        curves = [(row, *_curve(dataset, *row, criterion, group_key)) for row in rows_ac]
        return rows_ac, comps_all[criterion][group_key], curves

    def rowset_arms(criterion, group_key):
        rows, comps_arms, best = _arm_rows(arms, arm_coords, (dataset,), comps_all[criterion][group_key], criterion)
        curves = [((arm,), *_curve(dataset, arm, best[(arm, dataset)], criterion, group_key)) for (arm,) in rows]
        return rows, comps_arms, curves

    _render_stats_pngs(dpath_stats / "arm_coords" / "performance", ("Arm", "Coord"), rowset_arm_coords, dataset,
                       subject, eval_groups, labels, spread_type, bold_high, ordered, heatmap)
    if ArtifactManager.dpath_phase.name != "qual":  # qual reduces each arm to its pick(s): arms/ would duplicate arm_coords/
        _render_stats_pngs(dpath_stats / "arms" / "performance", ("Arm",), rowset_arms, dataset, subject, eval_groups,
                           labels, spread_type, bold_high, ordered, heatmap)

def _override_value(config, key):
    """Effective value of dot-path `key` in a row's config.json dict; '-' when a segment is absent --
    the signal that the param is inert under that configuration (save_metadata_coord prunes it)."""
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return "-"
        node = node[part]
    return str(node)

def _band_keys(side, rows, declared, config_by):
    """The params declared on `side` ('arm' / 'coord') across `rows`' overrides.json dicts
    (declared[row]), first-seen order (campaign order); a param whose effective value
    (config_by[row], _override_value) is identical across every row differentiates nothing -- drop
    the column (and with it the whole band when no column survives)."""
    keys = []
    for row in rows:
        for key in declared[row][side]:
            if key not in keys:
                keys.append(key)
    return [key for key in keys if len({_override_value(config_by[row], key) for row in rows}) > 1]

def _band_grids(band_specs, rows, config_by):
    """[(title, grid)] for the bands ([(title, keys)]) with surviving params: a header of param
    names only -- no key columns; the rows align with (and are labeled by) the aggregate Mean
    table's rows."""
    return [(title, [list(keys)] + [[_override_value(config_by[row], key) for key in keys] for row in rows])
            for title, keys in band_specs if keys]

def _map_groups(supp_scores, nshot_names):
    """The mAP sheets' merged group headers, [(group_title, n_group_cols), ...]: with supplemental
    columns enabled, banners split into title + 'Composite Scores' + one group header per enabled
    supplemental group; None (composite columns only) keeps full-width title banners, as the
    accuracy sheet always does."""
    groups = [("Composite Scores", 6)]
    if supp_scores["primitive"]:
        groups.append(("Primitive Scores", 6))
    if supp_scores["n_shot"] and nshot_names:
        groups.append(("N-Shot Scores", len(nshot_names)))
    return groups if len(groups) > 1 else None

def _write_sheet(ws, blocks, groups, bands, banner, n_keys, bold_high, heatmap, styled=True):
    """Lay one workbook sheet out from its blocks ((label, [(title, cell grid, killed row indices),
    ...]) -- the aggregate block labeled None, then the per-seed blocks), side by side one blank separator
    column apart, under the campaign banner cell '<repo-parent-dir> - <campaign> (<banner>)'.
    groups: None -> each table's banner is its title merged across the full table width;
    else [(group_title, n_group_cols), ...] -> the title sits over the block's n_keys key
    columns (merged across them when there are several), followed by one grey merged
    group-header cell per group (e.g. 'Composite Scores' over the composite columns,
    'Primitive Scores' over the primitive columns). bands ([(title, grid)], each grid a header
    + one-value-row-per-key grid in the sheet's row order; [] for none) render as the config
    column bands at the left, one blank separator column after each, with the score blocks all
    shifted right past them -- vertically aligned with the aggregate block's bottom Mean table
    so the Mean's key columns label their rows; band cells likewise get no winner-bold/heatmap
    styling. styled=False skips the winner-bold/heatmap styling of data cells altogether (the
    hardware sheet's readings aren't scores). A table's killed rows (those holding a killed trial,
    kill_thresh) take the killed yellow in place of the heatmap ramp."""
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    header_fill = PatternFill("solid", fgColor="EAEAEA")
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    widths = {}  # col idx -> longest header/data cell text (banner/label cells overflow instead)

    campaign = ws.cell(row=1, column=1, value=f"{paths['root'].parent.name} - {ArtifactManager.dpath_phase.parent.name} ({banner})")  # <campaign>/<phase>/
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
        for title_text, grid, killed_rows in tables:
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
                        cell.fill = PatternFill("solid", fgColor=_KILLED_HEX if r in killed_rows else _heat_hex(means[r]))
                row += 1
            row += 1  # blank spacer row between tables
        # widest table decides the block's width (the hw Mean table carries extra crash columns)
        col0 += max(len(grid[0]) for _, grid, _ in tables) + 1

    # banner row of the aggregate block's bottom Mean table (dataset tables precede it)
    band_row = 3 + sum(len(grid) + 2 for _, grid, _ in blocks[0][1][:-1])
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

@rank0
def update_phase_metrics(eval_groups, spread_type, bold_high, ordered, heatmap, supp_scores, overrides):
    """Write the phase's workbooks, one per selection criterion x eval group under
    artifacts/<campaign>/<phase>/phase_metrics/{arm_coords,arms}/performance/{map,acc}/<group>/metrics.xlsx. The arm_coords/
    workbooks have one row per planned (arm, coord) (the phase's matrix), keyed by two columns 'Arm' + 'Coord'; the arms/ workbooks
    one row per arm, each arm shown at its BEST coord per dataset -- the coord with the highest
    across-trial mean of the workbook's criterion's comp score among the arm's coords with completed
    trials in that dataset, ties to the first in campaign order (_best_coords; a per-(criterion, group)
    pick, so the workbook's mAP and accuracy sheets show the same coord) -- keyed by 'Arm' + 'Coord'
    too, the Coord cell naming that pick: a per-(arm, dataset) value, so every dataset table (aggregate
    and seed blocks alike, all three sheets) carries it ('-' where the arm has no completed trial in
    that dataset) and the cross-dataset Mean table shows '-'. The arms/ workbooks are skipped in the
    qual phase, whose matrix reduces each arm to its pick(s) -- there they would just duplicate
    arm_coords/. Every table below is laid out identically in both, over its own rows.

    Each workbook has three sheets: 'Composite mAP' (comp map scores, All/ID/OOD/I2T/I2I/T2I score
    columns), 'Composite I2T Accuracy' (comp acc, single I2T column) and 'Hardware Performance' (see
    below) -- the score sheets source the workbook's own criterion's best checkpoints
    (evals/_selected/<criterion>/), so e.g. the map/ workbooks' accuracy sheet holds the acc scores at
    the best-mAP checkpoint and vice versa. supp_scores ({'primitive', 'n_shot'} -> bool) appends the
    enabled supplemental score columns (_score_labels: the per-partition primitive scores, then the
    ID-partition n-shot bucket scores -- one column per bucket, '-' where a dataset's files lack the
    bucket) to the right of both sheets' tables, and splits each mAP-sheet table banner into the
    title (over the key columns) plus grey merged 'Composite Scores' / 'Primitive Scores' / 'N-Shot
    Scores' group headers over their column groups (the accuracy sheet keeps full-width merged title
    banners). Each sheet opens with a bold '<repo-parent-dir> - <campaign> (<eval group name>;
    <selection name>)' title cell (e.g. 'bc_dev - dev (Standard; mAP-selection)') and a blank row, then
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
    when False, rows keep the fixed matrix order (arms, then coords within each arm).
    Within a sheet one row order is shared across all tables, but the two sheets' orders may differ.
    heatmap shades each score cell white->#4caf50 by value over a fixed 0.00->100.00 (False leaves
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
    absent there, the signal that it is inert under that configuration (e.g. loss2.* with loss.blend.lambda
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
    coord_metadata.json's n_crashes. A row holding a killed trial (kill_thresh; the score sheets'
    yellow rows) reads 'X' across its readings -- a run cut short, its wall-clock not comparable --
    in every table it is killed in (its seed's block, its dataset's table, the Mean table), the crash
    totals still counted. Hardware cells get no winner-bold/heatmap styling. Column widths
    hug each column's longest header/data cell (banner/label text overflows); blank separator columns
    get a small ~square width. Regenerated at the end of each full seed sweep of the matrix (train.py,
    seed_sweep_complete) and unconditionally by the runner on exit / tools.regen_stats."""
    matrix = _matrix()
    datasets = list(matrix)
    arms = _matrix_arms(matrix, datasets)
    arm_coords = _matrix_arm_coords(matrix, datasets)

    comps_all = {criterion: _collect_comps(eval_groups, arm_coords, datasets, criterion) for criterion in BEST_CRITERIA}
    # update_chkpt_selection materializes every completed trial's _selected files, all criteria and groups
    # together, so row presence and seeds are criterion- and group-independent. an (arm, coord) gets
    # rows only once it has >= 1 completed trial in some dataset; it then appears in every dataset
    # table (blank '-' row where that dataset has no trials for it yet)
    comps_ref = next(iter(comps_all["map"].values()))
    rows_ac = [row for row in arm_coords if any(comps_ref[(row, dataset)] for dataset in datasets)]
    rows_arms = [(arm,) for arm in arms if any(row[0] == arm for row in rows_ac)]
    seeds = sorted({seed for comps in comps_ref.values() for seed in comps}, key=int)
    hw_by, crashes_by = _collect_hw(arm_coords, datasets)

    config_ac, config_arms = {}, {}
    if overrides:
        # an (arm, coord) row's overrides.json / config.json are written per dataset
        # (_datasets/<dataset>/_arms/<arm>/_coords/<coord>/) but identical across them, so each row reads its
        # from the first dataset it has completed trials in; an (arm,) row of the arms workbooks reads its
        # arm's first (arm, coord) row's (the arm's params are the same in every coord of it)
        rep = {}
        for row in rows_ac:
            rep.setdefault((row[0],), row)
        dpaths = {row: _dpath_coord(next(d for d in datasets if comps_ref[(row, d)]), *row) for row in rows_ac}
        declared_ac = {row: load_json(dpaths[row] / "overrides.json") for row in rows_ac}  # {'arm': {...}, 'coord': {...}}
        config_ac = {row: load_json(dpaths[row] / "config.json") for row in rows_ac}
        declared_arms = {row: declared_ac[rep[row]] for row in rows_arms}
        config_arms = {row: config_ac[rep[row]] for row in rows_arms}

    def build_blocks(headers, rows, key_cells, comps_by, score_key, labels):
        """A score sheet's blocks, left to right: (label, [(title, cell grid), ...]) -- the aggregate
        block (label None): one table per campaign dataset, then the always-shown 'Mean'
        cross-dataset summary table at the bottom; then one block per completed seed (label
        'seed <seed>'): the per-dataset tables only (no Mean summary), built from that seed's
        trials alone -- plain key cells (no trial counts), single-trial 'XX.XX' cells, '-'
        where that seed's trial hasn't completed. key_cells(row, dataset) -> the row's key cells
        (one per header) in that dataset's tables, dataset None the Mean table's: the row itself in
        the arm_coords workbooks, the arm + its best coord there in the arms ones. Rows are shared
        across all blocks -- when ordered, pinned to the aggregate Mean-table's first score column
        (labels[0]), descending. Each table carries the indices of its rows holding a killed trial
        (kill_thresh; in the Mean table, one from any dataset) for _write_sheet's yellow
        shading. Also returns the sheet's row order."""
        xmeans = _cross_dataset_means(rows, datasets, comps_by, score_key, labels)
        rows = _order_rows(rows, xmeans, labels[0]) if ordered else rows

        tables = []
        for dataset in datasets:
            grid, killed_rows = _stats_table_grid(
                headers,
                labels,
                [(key_cells(row, dataset), list(comps_by[(row, dataset)].values())) for row in rows],
                score_key,
                spread_type,
            )
            tables.append((DATASET_ALIAS2NAME[dataset], grid, killed_rows))
        xgrid = [[*headers, *labels]]
        xkilled = set()
        for r, row in enumerate(rows, 1):
            xgrid.append([*key_cells(row, None)]
                         + ["-" if xmeans[(row, label)] is None else f"{xmeans[(row, label)]:.2f}" for label in labels])
            if any(comp["killed"] for dataset in datasets for comp in comps_by[(row, dataset)].values()):
                xkilled.add(r)
        tables.append(("Mean", xgrid, xkilled))
        blocks = [(None, tables)]

        for seed in seeds:
            stables = []
            for dataset in datasets:
                grid = [[*headers, *labels]]
                killed_rows = set()
                for r, row in enumerate(rows, 1):
                    comp = comps_by[(row, dataset)].get(seed)
                    grid.append([*key_cells(row, dataset)]
                                + ["-" if comp is None or label.lower() not in comp[score_key]
                                   else f"{float(comp[score_key][label.lower()]) * 100:.2f}"
                                   for label in labels])
                    if comp is not None and comp["killed"]:
                        killed_rows.add(r)
                stables.append((DATASET_ALIAS2NAME[dataset], grid, killed_rows))
            blocks.append((f"seed {seed}", stables))
        return blocks, rows

    def build_hw_blocks(headers, rows, key_cells, hw_by, crash_totals):
        """The 'Hardware Performance' sheet's blocks, structured like build_blocks' (aggregate
        block of per-dataset tables + Mean table, then per-seed blocks; key_cells as there) over
        the hw readings (hw_by[(row, dataset)][seed]): header key columns + _HW_LABELS (the Mean
        table appends the _HW_CRASH_LABELS crash totals, crash_totals[row]), cells the rounded mean
        of the row's per-trial readings ('-' when the row has none there; 'X' across the readings when
        the row holds a killed trial, kill_thresh -- the score sheets' yellow rows -- a run cut
        short, its readings not comparable; the Mean table's crash totals stand); the Mean table means
        the per-dataset trial means across datasets."""

        def hw_row(cells, readings, killed):
            if not readings:
                return [*cells] + ["-"] * len(_HW_LABELS)
            if killed:
                return [*cells] + ["X"] * len(_HW_LABELS)
            return [*cells] + [str(round(np.mean([r[hw_label] for r in readings]))) for hw_label in _HW_LABELS]

        tables = []
        for dataset in datasets:
            grid = [[*headers, *_HW_LABELS]]
            for row in rows:
                readings = list(hw_by[(row, dataset)].values())
                cells = key_cells(row, dataset)
                grid.append(hw_row([*cells[:-1], f"{cells[-1]} ({len(readings)})"], readings, any(r["killed"] for r in readings)))
            tables.append((DATASET_ALIAS2NAME[dataset], grid, set()))
        xgrid = [[*headers, *_HW_LABELS, *_HW_CRASH_LABELS]]
        for row in rows:
            dataset_means = [
                {hw_label: np.mean([r[hw_label] for r in hw_by[(row, dataset)].values()]) for hw_label in _HW_LABELS}
                for dataset in datasets if hw_by[(row, dataset)]
            ]
            killed = any(r["killed"] for dataset in datasets for r in hw_by[(row, dataset)].values())
            xgrid.append(hw_row(key_cells(row, None), dataset_means, killed) + [str(crash_totals[row][kind]) for kind in _CRASH_KINDS])
        tables.append(("Mean", xgrid, set()))
        blocks = [(None, tables)]

        for seed in seeds:
            stables = []
            for dataset in datasets:
                grid = [[*headers, *_HW_LABELS]]
                for row in rows:
                    trial = hw_by[(row, dataset)].get(seed)
                    grid.append(hw_row(key_cells(row, dataset), [] if trial is None else [trial], trial is not None and trial["killed"]))
                stables.append((DATASET_ALIAS2NAME[dataset], grid, set()))
            blocks.append((f"seed {seed}", stables))
        return blocks

    nshot_names = _nshot_names(comps_all)
    map_labels, acc_labels = _score_labels(supp_scores, nshot_names)
    map_groups = _map_groups(supp_scores, nshot_names)

    def write_workbook(fpath, headers, rows, key_cells, comps_by, hw_by, crash_totals, band_specs, config_by, banner):
        wb = Workbook()
        ws_map = wb.active
        ws_map.title = "Composite mAP"
        map_blocks, map_rows = build_blocks(headers, rows, key_cells, comps_by, "map", map_labels)
        _write_sheet(ws_map, map_blocks, map_groups, _band_grids(band_specs, map_rows, config_by), banner, len(headers),
                     bold_high, heatmap)
        acc_blocks, acc_rows = build_blocks(headers, rows, key_cells, comps_by, "acc", acc_labels)
        _write_sheet(wb.create_sheet("Composite I2T Accuracy"), acc_blocks, None, _band_grids(band_specs, acc_rows, config_by),
                     banner, len(headers), bold_high, heatmap)
        # the hardware sheet shares the mAP sheet's row order (and so its overrides bands)
        _write_sheet(wb.create_sheet("Hardware Performance"), build_hw_blocks(headers, map_rows, key_cells, hw_by, crash_totals),
                     None, _band_grids(band_specs, map_rows, config_by), banner, len(headers), bold_high, heatmap, styled=False)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        wb.save(fpath)

    dpath_stats = ArtifactManager.dpath_phase / "phase_metrics"
    write_arms = ArtifactManager.dpath_phase.name != "qual"  # qual reduces each arm to its pick(s): arms/ would duplicate arm_coords/
    band_specs_ac = [("Arm Overrides", _band_keys("arm", rows_ac, declared_ac, config_ac)),
                     ("Coord Overrides", _band_keys("coord", rows_ac, declared_ac, config_ac))] if overrides else []
    band_specs_arms = [("Arm Overrides", _band_keys("arm", rows_arms, declared_arms, config_arms))] if overrides else []
    crash_totals_ac = {
        row: {kind: sum(crashes_by[(row, dataset)][kind] for dataset in datasets) for kind in _CRASH_KINDS}
        for row in rows_ac
    }
    # one workbook set per selection criterion: every score in <kind>/<criterion>/ (both score sheets)
    # comes from that criterion's best checkpoints (e.g. the map/ workbooks' accuracy sheet holds the
    # acc scores at the best-mAP checkpoint), with the banner naming the selection
    for criterion, selection_name in _SELECTION_NAMES.items():
        for group_key, group_name in eval_groups.items():
            comps_by = comps_all[criterion][group_key]
            banner = f"{group_name}; {selection_name}"
            write_workbook(dpath_stats / "arm_coords" / "performance" / criterion / group_key / "metrics.xlsx",
                           ("Arm", "Coord"), rows_ac,
                           lambda row, dataset: row, comps_by, hw_by, crash_totals_ac, band_specs_ac, config_ac, banner)
            if not write_arms:
                continue
            # arms: each arm at its best coord per dataset under this criterion x group (_arm_rows' rows
            # are rows_arms: which arms have trials doesn't depend on the criterion or group), the pick
            # named in the row's Coord cell -- '-' where the arm has no trials in that dataset, and in the
            # Mean table (dataset None: no single pick there)
            rows, comps_arms, best = _arm_rows(arms, arm_coords, datasets, comps_by, criterion)

            def key_cells_arms(row, dataset):
                return (row[0], best[(row[0], dataset)] if (row[0], dataset) in best else "-")

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
            write_workbook(dpath_stats / "arms" / "performance" / criterion / group_key / "metrics.xlsx",
                           ("Arm", "Coord"), rows, key_cells_arms,
                           comps_arms, hw_arms, crash_totals_arms, band_specs_arms, config_arms, banner)


def _collect_test_scores(group_keys, arm_coords, datasets):
    """(comps_all, chkpts) over the test tree (ArtifactManager.dpath_phase): per eval group, each
    (arm, coord) x dataset's scored-trial score maps keyed by trial seed (the seed dir name; empty
    dict -> no scored trials yet) -- comps_all[group_key][((arm, coord), dataset)][seed] is a
    _comp_entry read from the trial's _seeds/<seed>/<group_key>.json (test.py writes all groups
    together, so any group's presence marks the trial scored) -- and chkpts[((arm, coord), dataset)]
    the files' 'chkpt' field: the checkpoint index the combo's trainval models were saved at
    (identical across its seeds); no entry where the combo has no scored trials."""
    comps_all = {group_key: {} for group_key in group_keys}
    chkpts = {}
    for arm, coord in arm_coords:
        for dataset in datasets:
            dpath_coord = _dpath_coord(dataset, arm, coord)
            comps = {group_key: {} for group_key in group_keys}
            if (dpath_coord / "_seeds").exists():
                for dpath_trial in sorted((dpath_coord / "_seeds").iterdir()):
                    for group_key in group_keys:
                        fpath_scores = dpath_trial / f"{group_key}.json"
                        if fpath_scores.exists():
                            data = load_json(fpath_scores)
                            comps[group_key][dpath_trial.name] = _comp_entry(data["scores"])
                            chkpts[((arm, coord), dataset)] = data["chkpt"]
            for group_key in group_keys:
                comps_all[group_key][((arm, coord), dataset)] = comps[group_key]
    return comps_all, chkpts

def _build_test_blocks(rows, datasets, seeds, comps_by, chkpts, score_key, labels, spread_type, ordered):
    """A test score sheet's blocks, laid out like update_phase_metrics' build_blocks (aggregate
    block of per-dataset tables + bottom 'Mean' cross-dataset summary, then one per-seed block of
    the per-dataset tables alone) with a third 'Chkpt' key column after 'Arm'/'Coord': the
    checkpoint index the combo's trainval models were saved at -- a per-(row, dataset) value, so the
    dataset tables (aggregate and seed blocks alike) carry it and the cross-dataset Mean table shows
    '-'. Rows are shared across all blocks -- when ordered, pinned to the aggregate Mean-table's
    first score column (labels[0]), descending. Also returns the sheet's row order."""
    xmeans = _cross_dataset_means(rows, datasets, comps_by, score_key, labels)
    rows = _order_rows(rows, xmeans, labels[0]) if ordered else rows
    headers = ("Arm", "Coord", "Chkpt")

    def chkpt_cell(row, dataset):
        return str(chkpts[(row, dataset)]) if (row, dataset) in chkpts else "-"

    tables = []
    for dataset in datasets:
        grid = [[*headers, *labels]]
        for row in rows:
            score_maps = [comp[score_key] for comp in comps_by[(row, dataset)].values()]
            grid.append([row[0], f"{row[1]} ({len(score_maps)})", chkpt_cell(row, dataset),
                         *_score_cells(score_maps, labels, spread_type)])
        tables.append((DATASET_ALIAS2NAME[dataset], grid, set()))
    xgrid = [[*headers, *labels]]
    for row in rows:
        xgrid.append([*row, "-"] + ["-" if xmeans[(row, label)] is None else f"{xmeans[(row, label)]:.2f}" for label in labels])
    tables.append(("Mean", xgrid, set()))
    blocks = [(None, tables)]

    for seed in seeds:
        stables = []
        for dataset in datasets:
            grid = [[*headers, *labels]]
            for row in rows:
                comp = comps_by[(row, dataset)].get(seed)
                grid.append([*row, chkpt_cell(row, dataset)]
                            + ["-" if comp is None or label.lower() not in comp[score_key]
                               else f"{float(comp[score_key][label.lower()]) * 100:.2f}"
                               for label in labels])
            stables.append((DATASET_ALIAS2NAME[dataset], grid, set()))
        blocks.append((f"seed {seed}", stables))
    return blocks, rows

@rank0
def update_test_stats(eval_groups, spread_type, bold_high, ordered, heatmap, supp_scores, overrides):
    """Write the test workbooks, one per eval group at artifacts/<campaign>/test/map/test_<group>.xlsx
    (ArtifactManager.dpath_phase = the test dir; the map/ folder mirrors the phase workbooks'
    <kind>/<criterion>/ layout for consistency -- test has no selection-criterion dimension, every
    trainval model already sitting at its qual-selected best-mAP checkpoint, so map/ is the only
    folder). Each workbook is laid out like the qual phase's phase_metrics/arm_coords/ ones --
    'Composite mAP' + 'Composite I2T Accuracy' sheets of stacked per-dataset tables + bottom Mean
    summary, per-seed blocks, and the 'Arm Overrides' / 'Coord Overrides' config bands (read from
    the test tree's coord config.json/overrides.json copies), with all of stats.yaml's styling
    (spread_type/bold_high/ordered/heatmap/supp_scores/overrides) applied the same way -- except
    that a third 'Chkpt' key column carries each (row, dataset)'s saved-checkpoint index
    (_build_test_blocks) and there is no 'Hardware Performance' sheet (the readings describe
    training trials; test runs none). Rows and scores come from the test tree's per-trial score
    files (_collect_test_scores) over the recorded matrix, same row-presence rule as the phase
    workbooks: a planned (arm, coord) appears once it has >= 1 scored trial in some dataset."""
    matrix = _matrix()
    datasets = list(matrix)
    arm_coords = _matrix_arm_coords(matrix, datasets)
    comps_all, chkpts = _collect_test_scores(eval_groups, arm_coords, datasets)
    comps_ref = next(iter(comps_all.values()))
    rows = [row for row in arm_coords if any(comps_ref[(row, dataset)] for dataset in datasets)]
    seeds = sorted({seed for comps in comps_ref.values() for seed in comps}, key=int)

    band_specs, config_by = [], {}
    if overrides:
        # a row's config.json/overrides.json copies are per dataset but identical across them, so
        # each row reads its from the first dataset it has scored trials in
        dpaths = {row: _dpath_coord(next(d for d in datasets if comps_ref[(row, d)]), *row) for row in rows}
        declared = {row: load_json(dpaths[row] / "overrides.json") for row in rows}  # {'arm': {...}, 'coord': {...}}
        config_by = {row: load_json(dpaths[row] / "config.json") for row in rows}
        band_specs = [("Arm Overrides", _band_keys("arm", rows, declared, config_by)),
                      ("Coord Overrides", _band_keys("coord", rows, declared, config_by))]

    nshot_names = _nshot_names({"test": comps_all})
    map_labels, acc_labels = _score_labels(supp_scores, nshot_names)
    map_groups = _map_groups(supp_scores, nshot_names)

    for group_key, group_name in eval_groups.items():
        comps_by = comps_all[group_key]
        banner = f"{group_name}; test"
        wb = Workbook()
        ws_map = wb.active
        ws_map.title = "Composite mAP"
        map_blocks, map_rows = _build_test_blocks(rows, datasets, seeds, comps_by, chkpts, "map", map_labels,
                                                  spread_type, ordered)
        _write_sheet(ws_map, map_blocks, map_groups, _band_grids(band_specs, map_rows, config_by), banner, 3,
                     bold_high, heatmap)
        acc_blocks, acc_rows = _build_test_blocks(rows, datasets, seeds, comps_by, chkpts, "acc", acc_labels,
                                                  spread_type, ordered)
        _write_sheet(wb.create_sheet("Composite I2T Accuracy"), acc_blocks, None, _band_grids(band_specs, acc_rows, config_by),
                     banner, 3, bold_high, heatmap)
        fpath = ArtifactManager.dpath_phase / "map" / f"test_{group_key}.xlsx"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        wb.save(fpath)


@rank0
def plot_metrics(
        data_tracker,
        dpath_trial,
        nshot_bucket_names,
        epoch_size,
        hpsm,
        eval_groups,
        fontsize_axes=12,
        fontsize_ticks=8,
        fontsize_legend=8,
        subplot_border_width=1,
        fig_width=10,
        height_panel_scores=1.83,
        height_panel_general=0.8,
    ):
    data = data_tracker.data
    data_epoch = data["epoch"]
    data_eval = data["eval"]
    # three title lines: arm, coord, then the dataset (plus, per eval group, the group's name)
    dpath_coord = ArtifactManager.dpath_coord
    title_prefix = f"{dpath_coord.parent.parent.name}\n{dpath_coord.name}\n{DATASET_ALIAS2NAME[ArtifactManager.dataset]}"

    # eval panels (retrieval / n-shot / accuracy) are drawn only when eval ran;
    # train panels (loss / grad norm / lr) plot whenever train data is present (e.g. train_pt=trainval).
    has_eval = "scores" in data_eval

    # tracked in samples under the hood; plotted in epoch units
    x_eval = [v / epoch_size for v in data_eval["n_samps_seen"]]
    # n_samps_seen is stamped post-batch, but each batch's metrics (loss, grads, stats, lr) are
    # measured on the pre-step model -- stamp at batch start so the train curves anchor at 0
    x_train = [0.0, *(v / epoch_size for v in data_epoch["n_samps_seen"][:-1])]

    # under learning_curves/: the five eval-score panels go to scores/<group>.png, once per eval group (a
    # trial with no eval data -- the trainval phase -- gets none), and the rest, the same for every group,
    # to general.png (loss, gradients, batch stats, LR), alpha.png / logalpha.png (the logit scale, as
    # the alpha the logits carry / as the log alpha parameter the model learns, over its InfoNCE
    # gradient decomposition) and KL.png (the InfoNCE KL decomposition). Within a figure every panel is
    # the same height (in; height_panel_scores for the scores, height_panel_general for the rest), and
    # the figure is as tall as its panels plus the ~1.8 in its title and x label take -- so a panel is
    # the same size however many the figure holds.
    height_fixed = 1.8
    if has_eval:
        for group_key, group_name in eval_groups.items():
            plot_score_curves(
                data_eval,
                x_eval,
                dpath_trial,
                nshot_bucket_names,
                fontsize_axes,
                fontsize_ticks,
                fontsize_legend,
                subplot_border_width,
                fig_width,
                height_fixed,
                height_panel_scores,
                group_key=group_key,
                plot_title=f"{title_prefix} ({group_name})",
                output_filename=f"learning_curves/scores/{group_key}.png",
            )
    plot_general_curves(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        has_eval,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        fig_width,
        height_fixed,
        height_panel_general,
        hpsm,
        plot_title=title_prefix,
        output_filename="learning_curves/general.png",
    )
    for scale_key, prefix, sym, name in (("scale", "dalpha", r"\alpha", "alpha"),
                                         ("logit_scale", "dlogalpha", r"\log \alpha", "logalpha")):
        plot_alpha_curves(
            data_epoch,
            x_train,
            dpath_trial,
            fontsize_axes,
            fontsize_ticks,
            fontsize_legend,
            subplot_border_width,
            fig_width,
            height_fixed,
            height_panel_general,
            scale_key,
            prefix,
            sym,
            plot_title=title_prefix,
            output_filename=f"learning_curves/{name}.png",
        )
    plot_kl_curves(
        data_epoch,
        x_train,
        dpath_trial,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        fig_width,
        height_fixed,
        height_panel_general,
        plot_title=title_prefix,
        output_filename="learning_curves/KL.png",
    )

def plot_score_curves(
    data_eval,
    x_eval,
    dpath_trial,
    bucket_comp_keys,
    fontsize_axes,
    fontsize_ticks,
    fontsize_legend,
    subplot_border_width,
    fig_width,
    height_fixed,
    height_panel,
    group_key,
    plot_title,
    output_filename,
):
    fig = plt.figure(figsize=(fig_width, height_fixed + 5 * height_panel))
    gs = gridspec.GridSpec(5, 1, hspace=0)
    legend_handles = {}  # panel -> its legend's handles, boxed outside the panel once the layout is settled

    ax0 = fig.add_subplot(gs[0, 0])

    retrieval_specs = (
        ("i2t", "I2T", _COLOR_I2T),
        ("i2i", "I2I", _COLOR_I2I),
        ("t2i", "T2I", _COLOR_T2I),
    )
    comp_scores = data_eval["scores"][group_key]["comp"]
    comp_map = comp_scores["map"]
    # the composite is the series checkpoint selection argmaxes -- heaviest and on top
    ax0.plot(x_eval, comp_map["all"], label="All", color=_COLOR_COMP, linewidth=_LW_COMP, zorder=4)
    ax0.plot(x_eval, comp_map["id"], label="ID", color=_COLOR_PARTITION, linewidth=_LW_PARTITION)
    ax0.plot(x_eval, comp_map["ood"], label="OOD", color=_COLOR_PARTITION, linestyle="--", linewidth=_LW_PARTITION)
    for metric_name, metric_label, color in retrieval_specs:
        ax0.plot(x_eval, comp_map[metric_name], label=metric_label, color=color, linewidth=_LW_MODALITY)
    _mark_best(ax0, x_eval, comp_map["all"], fontsize_legend)
    ax0.set_ylabel("mAP Composite", fontsize=fontsize_axes)
    ax0.set_ylim(0, 1)
    legend_handles[ax0] = ax0.get_legend_handles_labels()[0]
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
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

    ax1.set_ylabel("mAP Primitive", fontsize=fontsize_axes)
    ax1.set_ylim(0, 1)
    legend_handles[ax1] = ax1.get_legend_handles_labels()[0]
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    id_mode_scores = data_eval["scores"][group_key]["id"]
    comp_nshot = id_mode_scores["map"].get("n-shot", {})
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax2, x_eval, comp_nshot, key, key)
        if comp_nshot:
            legend_handles[ax2] = ax2.get_legend_handles_labels()[0]
    ax2.set_ylabel("n-shot mAP (ID)", fontsize=fontsize_axes)
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
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
    ax3.set_ylabel("I2T Acc.", fontsize=fontsize_axes)
    ax3.set_ylim(0, 1)
    legend_handles[ax3] = ax3.get_legend_handles_labels()[0]
    ax3.grid(True)
    ax3.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
    comp_nshot_acc = id_mode_scores["acc"].get("n-shot", {})
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax4, x_eval, comp_nshot_acc, key, key)
        if comp_nshot_acc:
            legend_handles[ax4] = ax4.get_legend_handles_labels()[0]
    ax4.set_ylabel("n-shot Acc.\n(ID I2T)", fontsize=fontsize_axes)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")
    ax4.grid(True)
    ax4.tick_params(labelsize=fontsize_ticks)

    _finish_curves(fig, [ax0, ax1, ax2, ax3, ax4], [], legend_handles, plot_title, dpath_trial / output_filename, fontsize_legend, subplot_border_width)

def plot_general_curves(
    data_epoch,
    data_eval,
    x_train,
    x_eval,
    dpath_trial,
    has_eval,
    fontsize_axes,
    fontsize_ticks,
    fontsize_legend,
    subplot_border_width,
    fig_width,
    height_fixed,
    height_panel,
    hpsm,
    plot_title,
    output_filename,
):
    # a reporting.batch_diagnostics component that was off never recorded its series, and its panel(s)
    # are omitted outright: model grad norm, ||delta theta||, sim-grad sums, the S stats panel
    has_grad_norm = len(data_epoch["grad_norm_model"]) == len(x_train)
    has_delta_norm = len(data_epoch["delta_norm_model"]) == len(x_train)
    has_grad_sum_sim = len(data_epoch["grad_sum_sim"]) == len(x_train)
    has_sim_stats = len(data_epoch["sim_min"]) == len(x_train)
    # the Q-stats panel, when the blended targets carry distributional signal -- TrainPipeline records
    # targ stats only for graded targets (a lone sp/mp target is a 0/1 indicator), so an empty series
    # gets no panel.
    targ_panels = [("targ_hist", "Q")] if len(data_epoch["targ_hist"]) == len(x_train) else []
    # the P strip (sigmoid(logits), the predicted pair probabilities) sits between S and Q, on Q's [0, 1]
    # axis so predictions and targets read against each other. Recorded only for a BCE-family loss,
    # so under InfoNCE there is no series and no panel.
    p_panels = [("p_hist", "P")] if len(data_epoch["p_hist"]) == len(x_train) else []
    # each tracked logit bias (TrialData bias series; empty when untracked) gets a strip between the
    # Q panels and LR -- bias2 is loss2's term's own under separate logit scalars (loss.logits.shared
    # false). The logit scales have their own figures (plot_alpha_curves)
    scalar_panels = [
        (key, rf"${sym}$")
        for key, sym in (("bias", "b"), ("bias2", "b_2"))
        if len(data_epoch[key]) == len(x_train)
    ]
    # the mean hard-pair similarity margin strips (sim_targ_stats on) sit directly above LR, panels
    # each drawing one line per reporting.learning_curves.hpsm.kappas value: with
    # hpsm.multimodal the I2T and T2I directions and then their mean, else the mean alone
    hpsm_kappas = hpsm["kappas"]
    margin_dirs = [("_i2t", "I2T "), ("_t2i", "T2I ")] if hpsm["multimodal"] else []
    margin_panels = [
        (f"sim_margin{suffix}", rf"{prefix}$\overline{{\Delta S}}_{{\kappa}}$")
        for suffix, prefix in (*margin_dirs, ("", ""))
        if len(data_epoch[f"sim_margin{suffix}"]) == len(x_train)
    ]
    # the effective-lambda strip (sim_targ_stats on; loss.unitless over a live loss blend only) sits right
    # above LR: loss2's term's share of the unitless blend coefficients, lambda L_1 /
    # (lambda L_1 + (1 - lambda) L_2) (utils.loss.Criterion.term_coeffs) -- where unitless moves the blend
    # off the nominal loss.blend.lambda, batch by batch
    has_lambda_eff = len(data_epoch["lambda_eff"]) == len(x_train)

    # every panel is the loss panel's height, one grid row each: the loss and LR panels, which are
    # always drawn, plus the optional ones above
    n_panels = (
        2 + has_grad_norm + has_delta_norm + has_grad_sum_sim + has_sim_stats + len(p_panels) + len(targ_panels)
        + len(scalar_panels) + len(margin_panels) + has_lambda_eff
    )
    fig = plt.figure(figsize=(fig_width, height_fixed + n_panels * height_panel))
    gs = gridspec.GridSpec(n_panels, 1, hspace=0)
    legend_handles = {}  # panel -> its legend's handles, boxed outside the panel once the layout is settled

    ax0 = fig.add_subplot(gs[0, 0])
    if len(data_epoch["loss_train"]) == len(x_train):
        ax0.plot(x_train, data_epoch["loss_train"], label="Train", color="tab:orange", zorder=3)
    if len(data_epoch["loss_raw_train"]) == len(x_train):
        ax0.plot(x_train, data_epoch["loss_raw_train"], label="Train (Raw)", color="tab:blue")
    if has_eval:
        for partition, partition_label, loss_color in (("id", "ID", "tab:green"), ("ood", "OOD", "tab:red")):
            ax0.plot(x_eval, data_eval["loss_raw"][partition], label=f"{partition_label} Val", color=loss_color)
    ax0.set_ylabel(r"$\mathcal{L}$", fontsize=fontsize_axes + 4)
    ax0.set_yscale("log")
    ax0.minorticks_on()
    ax0.grid(which="minor", axis="y")
    legend_handles[ax0] = ax0.get_legend_handles_labels()[0]
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    axes = [ax0]

    if has_grad_norm:
        ax6 = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax6.plot(x_train, data_epoch["grad_norm_model"], color="tab:orange")
        ax6.set_ylabel(r"$\|\nabla_{\theta}\mathcal{L}\|$", fontsize=fontsize_axes + 4)
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
        ax6b.set_yscale("log")
        ax6b.minorticks_on()
        ax6b.grid(which="minor", axis="y")
        ax6b.grid(True)
        ax6b.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax6b)

    if has_grad_sum_sim:
        ax7 = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax7.plot(x_train, data_epoch["grad_sum_sim"], color="tab:orange", linewidth=1.0)
        ax7.axhline(0.0, color="gray", linewidth=0.5)
        ax7.set_ylabel(r"$\sum \nabla_S \mathcal{L}$", fontsize=fontsize_axes - 1)
        ax7.grid(True)
        ax7.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax7)

    # the S panel: min/max solid,
    # mean dashed, median dotted; teal/rose is a dark, mutually contrasting pair that also stays
    # clear of the orange gradient panels above and the purple scale panels below.
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
        ax.set_ylabel(stat_ylabel, fontsize=fontsize_axes)
        ax.set_ylim(*stat_ylim)
        legend_handles[ax] = legend_styles
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    def add_hist_panel(hist_key, ylabel, cmap):
        """The per-batch distribution histograms as a density heatmap strip."""
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
        ax.set_ylabel(ylabel, fontsize=fontsize_axes)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    if has_sim_stats:
        add_stat_panel("sim", "S", (-1.0, 1.0), "#008080")
    for hist_key, label in p_panels:
        add_hist_panel(hist_key, label, _P_CMAP)
    for hist_key, label in targ_panels:
        add_hist_panel(hist_key, label, _Q_CMAP)

    for key, label in scalar_panels:
        ax = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax.plot(x_train, data_epoch[key], color="blue")
        ax.set_ylabel(label, fontsize=fontsize_axes + 4)
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    for key, label in margin_panels:
        # mean hard-pair similarity margin (hard_pair_similarity_margin): per row, the hardness-weighted
        # mean sim of the positives minus that of the negatives, averaged over rows; kappa 0 is the plain
        # mean-separation margin, larger kappas weight each side toward its hard pairs
        ax = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        margins = np.array(data_epoch[key])  # [batch, kappa]
        colors = plt.cm.viridis(np.linspace(0.0, 0.75, len(hpsm_kappas)))
        for idx_kappa in reversed(range(len(hpsm_kappas))):  # drawn last-to-first, so a smaller kappa sits on top
            ax.plot(x_train, margins[:, idx_kappa], color=colors[idx_kappa], linewidth=1.0, label=rf"$\kappa = {_kappa_label(hpsm_kappas[idx_kappa])}$")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_ylabel(label, fontsize=fontsize_axes + 4)
        legend_handles[ax] = ax.get_legend_handles_labels()[0][::-1]  # legend in config order
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    if has_lambda_eff:
        ax = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
        ax.plot(x_train, data_epoch["lambda_eff"], color="tab:olive", linewidth=1.0)
        ax.set_ylabel(r"$\lambda_{\mathrm{eff}}$", fontsize=fontsize_axes + 4)
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

    ax10 = fig.add_subplot(gs[len(axes), 0], sharex=ax0)
    if len(data_epoch["lr"]) == len(x_train):
        ax10.plot(x_train, data_epoch["lr"], color="red")
    ax10.set_ylabel("η", fontsize=fontsize_axes + 6)
    ax10.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax10.yaxis.set_offset_position("right")
    ax10.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax10.yaxis.get_offset_text().set_visible(False)
    ax10.set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")
    ax10.grid(True)
    ax10.tick_params(labelsize=fontsize_ticks)
    axes.append(ax10)

    _finish_curves(fig, axes, axes_hist, legend_handles, plot_title, dpath_trial / output_filename, fontsize_legend, subplot_border_width)

def _resid_unvalidated(data_epoch):
    """Per batch, whether any of its rows' residual family came off the plain p* - y subtraction
    (resid_paths' third fraction): numerically unvalidated, whatever the values read (_RESID_MARKS)."""
    return np.asarray(data_epoch["resid_paths"], dtype=float)[:, 2] > 0


def _dalpha_unresolved(data_epoch, comp):
    """Per batch, whether the residual dL/dalpha term `comp` is numerically unresolved -- the stronger of
    the two marks (_RESID_MARKS), and a one-way test: an unvalidated batch (_resid_unvalidated) whose term
    sits at or below _DALPHA_RES_FLOOR, its magnitude sum against the full term's. False establishes
    nothing about the value. The ratio is
    scale-free, so it is read off the dalpha family for both figures -- dlogalpha is alpha times dalpha,
    which cancels, and the dalpha one stays defined where the clamp zeroes the dlogalpha family."""
    A = np.asarray(data_epoch[f"dalpha_sum_abs_{comp}"], dtype=float)[:, 0]
    A_full = np.asarray(data_epoch["dalpha_sum_abs_full"], dtype=float)[:, 0]
    return _resid_unvalidated(data_epoch) & (A < _DALPHA_RES_FLOOR * A_full)


def _mark_batches(ax, x_train, mask, mark):
    """Mark a panel's `mask` batches as `mark` (_RESID_MARKS), one span per contiguous run, each
    carried half a sample either way so a lone batch still shows. Only the first span is labelled, the
    rest riding along out of the legend."""
    x = np.asarray(x_train, dtype=float)
    half = 0.5 * (x[-1] - x[0]) / max(1, len(x) - 1)
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(x[s] - half, x[e] + half, label=mark if i == 0 else "_nolegend_", **_RESID_MARKS[mark])


def plot_alpha_curves(
    data_epoch,
    x_train,
    dpath_trial,
    fontsize_axes,
    fontsize_ticks,
    fontsize_legend,
    subplot_border_width,
    fig_width,
    height_fixed,
    height_panel,
    scale_key,
    prefix,
    sym,
    plot_title,
    output_filename,
):
    # one figure per parameterization of the logit scale (`scale_key` / `prefix` its series, `sym` its
    # symbol): scale / dalpha*, the alpha the logits carry (pinned at 100 while logits.scale.clamp
    # holds), and logit_scale / dlogalpha*, the log alpha parameter the model learns, as the model
    # holds it. A trial recording none of a figure's series gets no figure.
    # On top, each tracked scale (TrialData; empty when untracked, e.g. frozen) gets a panel --
    # the "2" series is loss2's term's own under separate logit scalars (loss.logits.shared false)
    scale_panels = [
        (f"{scale_key}{suffix}", rf"${sym}{sub}$")
        for suffix, sub in (("", ""), ("2", "_2"))
        if len(data_epoch[f"{scale_key}{suffix}"]) == len(x_train)
    ]
    req_key = {"scale": "alpha_req", "logit_scale": "log_alpha_req"}[scale_key]  # the bounds in this figure's units
    # under each scale panel of the LOGALPHA figure -- log alpha being the parameter the model learns -- the
    # parameter's own signed .grad, as the optimizer consumed it (TrainPipeline._logit_scalar_values: read after
    # the backward, before the step). It is the measurement; the dlogalpha panels further down are an analytical
    # decomposition off the blended target distribution, which carries neither a term's blend coefficient nor
    # loss.unitless' 1 / L and so need not agree with it. The primary scale's panel also draws
    # loss.infonce.block_residuals' correction where recorded (dlogalpha_correction: the delta it made, so the
    # gradient without it is the black line minus the orange one)
    # The caption says .grad because the first dlogalpha panel below is captioned with the same nabla: that
    # one is the analytical full term, this one what the parameter got
    grad_panels = {
        f"{scale_key}{suffix}": (f"logit_scale{suffix}_grad", rf"$\nabla_{{{sym}{sub}}} \mathcal{{L}}$" + "\n(.grad)")
        for suffix, sub in (("", ""), ("2", "_2"))
        if scale_key == "logit_scale" and len(data_epoch[f"logit_scale{suffix}_grad"]) == len(x_train)
    }
    # below, the InfoNCE logit-scale gradient decomposition (sim_targ_stats on; an InfoNCE loss only,
    # since only it records the series), fifteen panels: the per-pair dL/dalpha terms summed, summed in
    # magnitude, and their coherence ratio C = |sum| / sum|.|, each for the full gradient, its
    # structural / residual parts and the residual's own structural / irreducible split
    # (utils.loss.infonce_batch_stats), every panel drawing the all /
    # positive-mass / negative-mass attributions -- bar the logalpha figure's five C panels, which
    # repeat the alpha figure's and are left blank. dlogalpha* is flat zero wherever logits.scale.clamp
    # holds the parameter above its cap -- its sums zero and its C NaN, there being no pressure on the
    # parameter to cancel -- dalpha* still carrying the pressure on the effective scale.
    # The three residual comps carry the provenance marks (_RESID_MARKS) on all three of their aggs: hatched
    # over every batch with rows off the unvalidated subtraction, solid where the term is also unresolved.

    def dalpha_label(agg, comp):
        sup = {"full": "", "struct": r"^{\text{S}}", "res": r"^{\text{R}}",
               "sres": r"^{\text{SR}}", "ires": r"^{\text{IR}}"}[comp]
        if agg == "sum":
            return rf"$\nabla_{{{sym}}}{sup} \mathcal{{L}}$"
        return rf"$\text{{{'A' if agg == 'sum_abs' else 'C'}}}_{{{sym}}}{sup}$"

    dalpha_panels = [
        (f"{prefix}_{agg}_{comp}", dalpha_label(agg, comp), agg, comp)
        for agg in ("sum", "sum_abs", "C")
        for comp in ("full", "struct", "res", "sres", "ires")
        if len(data_epoch[f"{prefix}_{agg}_{comp}"]) == len(x_train)
    ]
    # the residual terms' marks (_RESID_MARKS), on all three of a term's aggs -- where the magnitude is
    # unvalidated or arithmetic, so are the signed sum and the coherence ratio taken over it
    unvalidated = _resid_unvalidated(data_epoch) if len(data_epoch["resid_paths"]) == len(x_train) else None
    res_unresolved = {
        comp: _dalpha_unresolved(data_epoch, comp)
        for comp in _DALPHA_RES_COMPS
        if len(data_epoch[f"dalpha_sum_abs_{comp}"]) == len(data_epoch["dalpha_sum_abs_full"]) == len(x_train)
    }
    n_panels = len(scale_panels) + len(grad_panels) + len(dalpha_panels)
    if n_panels == 0:
        return

    fig = plt.figure(figsize=(fig_width, height_fixed + n_panels * height_panel))
    gs = gridspec.GridSpec(n_panels, 1, hspace=0)
    legend_handles = {}  # panel -> its legend's handles, boxed outside the panel once the layout is settled
    axes = []

    for key, label in scale_panels:
        ax = fig.add_subplot(gs[len(axes), 0], sharex=axes[0] if axes else None)
        # zorder: the bound lines below sit exactly on this one wherever the target mapping is a softmax
        # pinned to the scale (utils.loss.InfoNCECriterion._tsm -- the target is then always exactly
        # reachable at the current scale), and the panel is about the scale
        ax.plot(x_train, data_epoch[key], color="tab:purple", label=rf"${sym}$", zorder=3)
        if key == scale_key and len(data_epoch[f"{req_key}_max"]) == len(x_train):
            # per batch, the row-wise target-implied scale bound (utils.loss.infonce_batch_stats'
            # alpha_req; an InfoNCE loss only, so a BCE-family loss's panel gets no lines), read
            # against the scale this figure plots -- alpha_req on the alpha panel (the series sits at
            # 100 while logits.scale.clamp holds), log(alpha_req) on the logalpha one, each reduced
            # over rows in its own units: for row i, the smallest alpha whose logit range alpha * S
            # over S in [-1, 1] spans the blended
            # target distribution Y_i as optimal logits log(Y_i) (up to a constant), 0.5 * log(max_j Y_ij
            # / min_j Y_ij). Softmax feasibility is row-wise, so the batch's requirement is the max over
            # rows (solid), drawn with the min (solid) and the mean (dashed). A row holding a zero sits
            # at infinity and leaves a gap in the lines it reaches (the max and the mean): a stray one
            # under graded targets (a tax top-rank split, a bm-kernel pair meeting at the root), every
            # row under sp/mp targets with the linear tsm, which then draws nothing
            # the legend names only the stats the panel actually draws (_req_labels: the mean and the max
            # go non-finite wherever a stray zero row reaches them, all three under hard binary targets,
            # whose every row holds a zero). With none of the trio drawn the panel keeps no legend at all,
            # as a trial not recording the series does.
            vals_req = {stat: np.asarray(data_epoch[f"{req_key}_{stat}"], dtype=float)
                        for stat in ("min", "mean", "max")}
            labels_req = _req_labels(vals_req, sym)
            for stat, linestyle in (("min", "-"), ("mean", "--"), ("max", "-")):
                ax.plot(x_train, vals_req[stat], color="red", linestyle=linestyle, linewidth=1.0,
                        label=labels_req.get(stat))
            if labels_req:
                legend_handles[ax] = ax.get_legend_handles_labels()[0]
        ax.set_ylabel(label, fontsize=fontsize_axes + 4)
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)

        if key in grad_panels:
            key_grad, label_grad = grad_panels[key]
            ax = fig.add_subplot(gs[len(axes), 0], sharex=axes[0])
            ax.axhline(0.0, color="grey", linewidth=0.8, zorder=1)  # a signed series: which way it pushes
            ax.plot(x_train, data_epoch[key_grad], color="black", linewidth=1.0, label=".grad", zorder=3)
            if key == scale_key and len(data_epoch["dlogalpha_correction"]) == len(x_train):
                ax.plot(x_train, data_epoch["dlogalpha_correction"], color="#E69F00", linewidth=1.0,
                        label="correction", zorder=2)
                legend_handles[ax] = ax.get_legend_handles_labels()[0]
            ax.set_ylabel(label_grad, fontsize=fontsize_axes + 4)
            ax.grid(True)
            ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
            axes.append(ax)

    for key, label, agg, comp in dalpha_panels:
        ax = fig.add_subplot(gs[len(axes), 0], sharex=axes[0] if axes else None)
        # the logalpha figure's C panels repeat the alpha figure's exactly -- alpha cancels in the
        # ratio |sum| / sum|.| -- so they are drawn blank: the panel and its caption hold the row, so
        # the two figures stay aligned panel for panel, with the redundant curves, grid, y ticks and
        # legend all left off
        if prefix == "dlogalpha" and agg == "C":
            ax.set_yticks([])
        else:
            vals = np.array(data_epoch[key])  # [batch, (all, pos, neg)]
            for idx_attr, (attr_label, color) in enumerate(_DALPHA_ATTRIBUTIONS):
                ax.plot(x_train, vals[:, idx_attr], color=color, linewidth=1.0, label=attr_label)
            if agg == "C":
                ax.set_ylim(0.0, 1.0)  # a cancellation ratio
            else:
                ax.axhline(0.0, color="gray", linewidth=0.5)
            if comp in _DALPHA_RES_COMPS and unvalidated is not None:
                _mark_batches(ax, x_train, unvalidated, "unvalidated")
            if comp in res_unresolved:
                _mark_batches(ax, x_train, res_unresolved[comp], "unresolved")
            legend_handles[ax] = ax.get_legend_handles_labels()[0]
            ax.grid(True)
        ax.set_ylabel(label, fontsize=fontsize_axes)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)
    axes[-1].set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")
    axes[-1].tick_params(labelbottom=True)

    # the dalpha panels come agg-major, so each agg's five comps are a consecutive block: the gradient
    # sums, their magnitudes, their coherence ratios. Each block is ruled off top and bottom in a
    # double-width line (_finish_curves' axes_blocks) -- the three read as one quantity apiece across the
    # five decompositions, which the shared y scale of a block does not say on its own. The logalpha
    # figure's blank C block is ruled like the others, the two figures staying aligned block for block.
    axes_dalpha = axes[len(scale_panels) + len(grad_panels):]  # in dalpha_panels order
    axes_blocks = [
        [ax for ax, panel in zip(axes_dalpha, dalpha_panels) if panel[2] == agg]
        for agg in ("sum", "sum_abs", "C")
    ]
    _finish_curves(fig, axes, [], legend_handles, plot_title, dpath_trial / output_filename, fontsize_legend,
                   subplot_border_width, axes_blocks=[group for group in axes_blocks if group])

def plot_kl_curves(
    data_epoch,
    x_train,
    dpath_trial,
    fontsize_axes,
    fontsize_ticks,
    fontsize_legend,
    subplot_border_width,
    fig_width,
    height_fixed,
    height_panel,
    plot_title,
    output_filename,
):
    # the InfoNCE KL decomposition (sim_targ_stats on; an InfoNCE loss only, since only it records the
    # series -- any other trial gets no figure), five panels (utils.loss.infonce_kl_terms, both anchor
    # directions averaged, batch-meaned): D_KL(y || p) -- the raw loss less the targets' entropy -- then
    # its three parts, the structural E_S = D_KL(p* || p) (what the model could still remove at this
    # alpha), the irreducible E_IR = D_KL(y || p*) (the target outside the reachable set) and the cross
    # term E_SR, with the two representational parts also drawn summed as E_R = E_SR + E_IR. All are >= 0
    # (E_SR because p is itself reachable; bf16 logit rounding can dip it a hair below), so each panel
    # draws a zero reference line and autoscales -- the line hugs the bottom while the series stays
    # positive, and any dip below it shows.
    # The three representational panels are the residual family's (E_IR differences -H(y) against a
    # cross-entropy equal to it to within exp(-2 alpha), E_SR reads p* - y), so they carry the unvalidated
    # mark (_RESID_MARKS) over every batch with rows off the plain subtraction. There is no unresolved tier
    # here: that one is a test on the dalpha magnitudes, and nothing comparable certifies a KL term.
    kl_panels = [
        (vals, label, resid)
        for vals, label, resid in (
            (data_epoch["kl"], r"$D_{\mathrm{KL}}(\text{y}\|\text{p})$", False),
            (data_epoch["kl_s"], r"$\mathcal{E}^{\text{S}}$", False),
            (np.array(data_epoch["kl_sr"]) + np.array(data_epoch["kl_ir"]), r"$\mathcal{E}^{\text{R}}$", True),
            (data_epoch["kl_sr"], r"$\mathcal{E}^{\text{SR}}$", True),
            (data_epoch["kl_ir"], r"$\mathcal{E}^{\text{IR}}$", True),
        )
        if len(vals) == len(x_train)
    ]
    if not kl_panels:
        return
    unvalidated = _resid_unvalidated(data_epoch) if len(data_epoch["resid_paths"]) == len(x_train) else None

    fig = plt.figure(figsize=(fig_width, height_fixed + len(kl_panels) * height_panel))
    gs = gridspec.GridSpec(len(kl_panels), 1, hspace=0)
    legend_handles = {}  # panel -> its legend's handles, boxed outside the panel once the layout is settled
    axes = []

    for vals, label, resid in kl_panels:
        ax = fig.add_subplot(gs[len(axes), 0], sharex=axes[0] if axes else None)
        ax.plot(x_train, vals, color="darkmagenta", linewidth=1.0)
        ax.axhline(0.0, color="gray", linewidth=0.5)
        if resid and unvalidated is not None and unvalidated.any():
            _mark_batches(ax, x_train, unvalidated, "unvalidated")
            legend_handles[ax] = ax.get_legend_handles_labels()[0]
        ax.set_ylabel(label, fontsize=fontsize_axes)
        ax.grid(True)
        ax.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        axes.append(ax)
    axes[-1].set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")
    axes[-1].tick_params(labelbottom=True)

    _finish_curves(fig, axes, [], legend_handles, plot_title, dpath_trial / output_filename, fontsize_legend, subplot_border_width)

def _finish_curves(fig, axes, axes_hist, legend_handles, plot_title, fpath_plot, fontsize_legend,
                   subplot_border_width, axes_blocks=(), box_blocks=False, alternate_sides=True):
    """
    The pass every learning-curve figure (plot_score_curves / plot_general_curves / plot_alpha_curves /
    plot_kl_curves / _render_strips) ends on, over its top-to-bottom `axes`: panel styling (`axes_hist`,
    the heatmap strips, keep their own background), the title, the layout, the outside legends
    (`legend_handles`: panel -> handles), then the save. `axes_blocks` groups consecutive panels to rule
    off as blocks, `box_blocks` closing each block's sides too (see below). `alternate_sides` flips every
    other panel's y label and ticks to the right
    and boxes its legend opposite, so a stack of panels doesn't crowd one margin; False leaves every
    panel's sides as its figure set them and boxes every legend on the left (the strip figures, whose
    right margin carries the coord names).
    """
    for ax in axes:
        ax.label_outer()

    for idx_ax, ax in enumerate(axes):
        if ax not in axes_hist:
            ax.set_facecolor(_BG_LINE_PANEL)
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_border_width)
            spine.set_edgecolor("black")
        if alternate_sides and idx_ax % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    # each `axes_blocks` group (a run of consecutive panels, top to bottom -- plot_alpha_curves' three
    # dL/dalpha aggs, the strip figures' top-level coord groups) is ruled off as one block: the horizontal
    # spines bounding it -- the top of its first panel and the bottom of its last -- go to twice the panel
    # border's width, the edges shared between panels inside it staying at the panel width. The sides stay
    # too unless `box_blocks` closes them, which carries the same double width down every panel of the
    # block, boxing it. The panels sit flush (hspace=0), so a block's own bottom edge doubles as the next
    # block's top and the pair of thick spines draws as the one rule between them.
    for group in axes_blocks:
        group[0].spines["top"].set_linewidth(2 * subplot_border_width)
        group[-1].spines["bottom"].set_linewidth(2 * subplot_border_width)
        for ax in group if box_blocks else ():
            ax.spines["left"].set_linewidth(2 * subplot_border_width)
            ax.spines["right"].set_linewidth(2 * subplot_border_width)

    # the title is the top panel's own, left-aligned with its left edge: padded off the panel in points
    # rather than placed by figure fraction, so the spacing holds at any figure height
    axes[0].set_title(plot_title, loc="left", fontweight="bold", fontsize=20)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    # each y label is shrunk to its panel's height, and each legend goes in a box outside its panel, on
    # the side away from the panel's y label and tick labels, no taller than the panel: sized against
    # the panel heights the layout above settled, then a second pass makes room for the boxes
    for idx_ax, ax in enumerate(axes):
        _fit_ylabel(ax)
        if ax in legend_handles:
            side = "left" if not alternate_sides or idx_ax % 2 == 1 else "right"
            _place_legend_outside(ax, legend_handles[ax], side, fontsize_legend)
    plt.tight_layout()
    fpath_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath_plot, dpi=300)
    plt.close(fig)

def _fit_ylabel(ax):
    """
    Shrinks the axes' y label until it is no taller than the axes. The label is rotated, so its
    height is its longest line's length, and a multi-line label's lines shrink together.
    """
    label = ax.yaxis.label
    while label.get_window_extent().height > ax.bbox.height:
        label.set_fontsize(label.get_fontsize() * ax.bbox.height / label.get_window_extent().height)

def _place_legend_outside(ax, handles, side, fontsize):
    """
    Legend for `handles` in a box outside the axes on `side` ("left" / "right"), centered on the
    axes' height, with the fewest columns that keep the box no taller than the axes.
    """
    loc, anchor = ("center right", (0.0, 0.5)) if side == "left" else ("center left", (1.0, 0.5))
    for ncol in range(1, len(handles) + 1):
        legend = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=ncol, fontsize=fontsize)
        if legend.get_window_extent().height <= ax.bbox.height:
            break

def _kappa_label(kappa):
    """Mathtext for a kappa legend entry: a value of 1000 and over is abbreviated ('1k', '10k', .., '1M', ..)."""
    for unit, suffix in ((1e6, "M"), (1e3, "k")):
        if kappa >= unit:
            return rf"{kappa / unit:g}\mathrm{{{suffix}}}"
    return f"{kappa:g}"

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
