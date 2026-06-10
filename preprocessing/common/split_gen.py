import bisect
import copy
import os
import random
import shutil
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional, Any
from pathlib import Path

from utils.data import Split
from utils.utils import paths, save_pickle
from utils.config import GenSplitConfig

import pdb


def _process_rfpaths_parallel(rfpaths, dataset, desc):
    imgs_root = paths["imgs"][dataset]
    fpaths = [imgs_root / rfpath for rfpath in rfpaths]
    means = []
    vars_ = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        max_in_flight = max(64, (os.cpu_count() or 1) * 4)
        fpaths_iter = iter(fpaths)
        in_flight = set()
        for _ in range(min(max_in_flight, len(fpaths))):
            in_flight.add(ex.submit(process_image, next(fpaths_iter)))
        pbar = tqdm(total=len(fpaths), desc=desc, file=sys.stdout)
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                mean_img, var_img = fut.result()
                means.append(mean_img)
                vars_.append(var_img)
                pbar.update(1)
                try:
                    in_flight.add(ex.submit(process_image, next(fpaths_iter)))
                except StopIteration:
                    pass
        pbar.close()
    return means, vars_


def _snapshot_norm_stats(all_means, all_vars):
    means = np.stack(all_means)
    vars_ = np.stack(all_vars)
    mean_agg = means.mean(axis=0)
    var_agg = vars_.mean(axis=0) + means.var(axis=0)
    std_agg = np.sqrt(np.clip(var_agg, 0.0, None))
    return tuple(float(x) for x in mean_agg), tuple(float(x) for x in std_agg)


def get_norm_stats(data_indexes, dataset: str, cfg) -> Dict[str, Tuple[Tuple[float], Tuple[float]]]:
    if cfg.dev.get("skip_norm_stats", False):
        print("***** Skipping norm stats computation in dev mode; returning dummy values *****")
        return {pt: ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) for pt in ("train", "trainval", "whole")}
    return compute_rgb_norm_stats_by_partition(data_indexes, dataset)

def compute_rgb_norm_stats_by_partition(
    data_indexes: Dict[str, Any],
    dataset: str,
) -> Dict[str, Tuple[Tuple[float], Tuple[float]]]:
    """
    Norm stats accumulated incrementally: train → snapshot, +val → snapshot, 
    +test → snapshot; Each image contributes equally regardless of resolution.
    """
    groups = [
        ("train",    [d["rfpath"] for d in data_indexes["train"]]),
        ("trainval", [d["rfpath"] for d in data_indexes["val"]["id"]] +
                     [d["rfpath"] for d in data_indexes["val"]["ood"]]),
        ("whole",    [d["rfpath"] for d in data_indexes["test"]["id"]] +
                     [d["rfpath"] for d in data_indexes["test"]["ood"]]),
    ]

    all_means: List = []
    all_vars: List = []
    results = {}

    for pt_name, rfpaths in groups:
        if rfpaths:
            new_means, new_vars = _process_rfpaths_parallel(
                rfpaths, dataset, desc=f"Computing norm stats ({pt_name})"
            )
            all_means.extend(new_means)
            all_vars.extend(new_vars)

        results[pt_name] = _snapshot_norm_stats(all_means, all_vars)

    return results

# helper for _process_rfpaths_parallel()
def process_image(fpath_img):
    with Image.open(fpath_img) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    mean_img = arr.mean(axis=(0, 1), dtype=np.float64) / 255.0
    var_img = arr.var(axis=(0, 1), dtype=np.float64) / 255.0**2

    return mean_img, var_img

def truncate_subspecies(s: str) -> str:
    components = s.split("_", 2)
    if len(components) < 3:
        return s
    return components[0] + "_" + components[1]

def build_penult_2_cids(cids, cid_2_penult):
    penult_2_cids = defaultdict(list)
    for cid in cids:
        penult = cid_2_penult[cid]
        penult_2_cids[penult].append(cid)
    return penult_2_cids

def strat_sample_partition(
    n_classes: int,
    n_draws: int,
    n_insts_2_classes: Dict[int, List[str]],
    class_2_insts: Dict[str, List[str | Tuple[str, int]]],
    insts: Set[str | Tuple[str, int]],
    seed: int = None,
) -> Tuple[
    Set[str | Tuple[str, int]],
    Set[str | Tuple[str, int]],
]:

    def compute_class_hits(n_draws, n_classes):
        class_hits = [n_draws // n_classes] * n_classes
        plus_ones = n_draws % n_classes
        for i in range(plus_ones):
            class_hits[i] += 1
        return class_hits

    rng = random.Random(seed)

    insts_rem = copy.deepcopy(insts)
    insts_eval = []

    n_insts_rem = len(insts_rem)
    n_classes_rem = n_classes
    n_draws_rem = n_draws

    max_count_bucket = max(n_insts_2_classes.keys()) if len(n_insts_2_classes) > 0 else 0
    i = 0
    while True:
        i += 1
        classes_i = list(n_insts_2_classes[i])
        if classes_i:
            n_classes_i = len(classes_i)
            n_instances_i = n_classes_i * i
            n_draws_i = round(n_instances_i * n_draws_rem / n_insts_rem)

            rng.shuffle(classes_i)
            class_hits = compute_class_hits(n_draws_i, n_classes_i)

            for idx, k in enumerate(class_hits):
                c = classes_i[idx]
                inst_hits = rng.sample(class_2_insts[c], k)
                insts_eval += inst_hits

            n_insts_rem -= n_instances_i
            n_classes_rem -= n_classes_i
            n_draws_rem -= n_draws_i

        # bucket size at which proportional drawing yields exactly 1 instance per class
        # phase 1 exit threshold uses same live ratio as draw calculation
        if i >= (n_insts_rem / n_draws_rem) - 1 and (n_draws_rem >= n_classes_rem or n_classes_rem == 0):
            break

        if i >= max_count_bucket:
            break

    if n_draws_rem > 0 and n_classes_rem > 0:
        classes_rem = []
        insts_counts_rem = []

        for count in sorted(list(n_insts_2_classes.keys())):
            if count <= i:
                continue

            classes = n_insts_2_classes[count]
            for c in classes:
                classes_rem += [c] * count
                insts_c = class_2_insts[c]
                for inst in insts_c:
                    insts_counts_rem += [(inst, count)]

        _, insts_counts_strat2 = train_test_split(
            insts_counts_rem,
            stratify=classes_rem,
            test_size=n_draws_rem,
            shuffle=True,
            random_state=seed,
        )

        insts_counts_strat2.sort(key=lambda x: (x[1], x[0]))
        insts_strat2, _ = zip(*insts_counts_strat2)
        insts_eval += insts_strat2

    insts_pt = set(insts_eval)
    insts_rem -= insts_pt

    return insts_pt, insts_rem

def strat_sample_partition_ood(
    skeys_pool: Set[Tuple[str, int]],
    cid_2_penult: Dict[str, str],
    n_cids_whole: int,
    n_samps_whole: int,
    cfg: GenSplitConfig,
    ood_tol_flag: bool,
):
    """
    Pick exactly round(n_cids_whole * pct_partition) classes for OOD-val,
    stratified by penultimate class.
    """
    cid_2_skeys_pool = build_cid_2_skeys(skeys_pool)
    cids_pool = set(cid_2_skeys_pool.keys())
    n_cids_pt = round(n_cids_whole * cfg.pct_partition)

    penult_2_cids = defaultdict(list)
    for cid in sorted(cids_pool):
        penult_2_cids[cid_2_penult[cid]].append(cid)

    n_insts_2_classes_penult_lvl = defaultdict(list)
    for penult, cids in penult_2_cids.items():
        n_insts_2_classes_penult_lvl[len(cids)].append(penult)

    sample_once_more = True
    i = 0
    while sample_once_more:
        i += 1
        if i % 10_000 == 0 and i > 0:
            print(f"Warning: {i / 1_000}k seeds searched and no OOD partition found satisfying pct_ood_tol={cfg.pct_ood_tol}")
        
        cids_pt, cids_rem = strat_sample_partition(
            n_classes=len(penult_2_cids),
            n_draws=n_cids_pt,
            n_insts_2_classes=n_insts_2_classes_penult_lvl,
            class_2_insts=penult_2_cids,
            insts=set(cids_pool),
            seed=cfg.seed + i,
        )

        skeys_pt = {
            skey
            for cid in cids_pt
            for skey in cid_2_skeys_pool[cid]
        }
        pct_samps_pt = len(skeys_pt) / n_samps_whole
        close_enough = abs(pct_samps_pt - cfg.pct_partition) <= cfg.pct_ood_tol
        if close_enough or not ood_tol_flag:
            sample_once_more = False

    skeys_rem = {
        skey
        for cid in cids_rem
        for skey in cid_2_skeys_pool[cid]
    }

    return skeys_pt, skeys_rem

def strat_sample_partition_id(
    skeys_pool: Set[Tuple[str, int]], 
    n_samps_whole: int, 
    cfg: GenSplitConfig,
) -> Tuple[
    Set[Tuple[str, int]],
    Set[Tuple[str, int]],
]:

    cid_2_skeys_pool = build_cid_2_skeys(skeys_pool)
    cids_pool = set(cid_2_skeys_pool.keys())

    cids_pool_singles = {
        cid
        for cid in sorted(cids_pool)
        if len(cid_2_skeys_pool[cid]) == 1
    }
    cids_pool_multis = cids_pool - cids_pool_singles

    skeys_id_multis = {
        skey
        for cid in cids_pool_multis
        for skey in cid_2_skeys_pool[cid]
    }

    n_samps_id_val_target = round(n_samps_whole * cfg.pct_partition)

    n_insts_2_classes_leaf_lvl = defaultdict(list)
    cid_2_skeys_id_multis = defaultdict(list)

    for cid in sorted(cids_pool_multis):
        skeys_cid = list(sorted(cid_2_skeys_pool[cid]))
        cid_2_skeys_id_multis[cid] = skeys_cid
        n_insts_2_classes_leaf_lvl[len(skeys_cid)].append(cid)

    skeys_id_val, skeys_train_multis = strat_sample_partition(
        n_classes=len(cids_pool_multis),
        n_draws=n_samps_id_val_target,
        n_insts_2_classes=n_insts_2_classes_leaf_lvl,
        class_2_insts=cid_2_skeys_id_multis,
        insts=skeys_id_multis,
        seed=cfg.seed,
    )

    skeys_id_singles = {
        cid_2_skeys_pool[cid][0]
        for cid in sorted(cids_pool_singles)
    }
    skeys_train = skeys_train_multis | skeys_id_singles

    return skeys_id_val, skeys_train

def strat_sample_ood_id(
    skeys_pool: Set[Tuple[str, int]],
    n_cids_whole: int,
    n_samps_whole: int,
    cid_2_penult: Dict[str, str],
    cfg: GenSplitConfig,
    ood_tol_flag: bool = True,
) -> Tuple[
    Set[Tuple[str, int]],
    Set[Tuple[str, int]],
    Set[Tuple[str, int]],
]:
    skeys_pt_ood, skeys_pool = strat_sample_partition_ood(
        skeys_pool,
        cid_2_penult,
        n_cids_whole,
        n_samps_whole,
        cfg,
        ood_tol_flag,
    )
    skeys_pt_id, skeys_pool = strat_sample_partition_id(
        skeys_pool, 
        n_samps_whole, 
        cfg,
    )
    return skeys_pt_ood, skeys_pt_id, skeys_pool

def build_cid_2_skeys(skeys: Set[Tuple[str, int]]) -> Dict[str, List[Tuple[str, int]]]:
    cid_2_skeys = defaultdict(list)
    for cid, samp_idx in sorted(skeys):
        cid_2_skeys[cid].append((cid, samp_idx))
    return cid_2_skeys

def build_trainval_skeys_partition(skeys_pts):

    skeys_trainval = skeys_pts["train"] | skeys_pts["id_val"] | skeys_pts["ood_val"]
    skeys_id_test = skeys_pts.get("id_test", set())
    return skeys_trainval - skeys_id_test

def build_id_eval_nshot(skeys_pts, cfg):

    cid_2_skeys_trainval = build_cid_2_skeys(skeys_pts["trainval"])
    cid_2_skeys_id_test = build_cid_2_skeys(skeys_pts["id_test"])

    n_shot_tracker = []
    for _ in range(len(cfg.nst_names)):
        n_shot_tracker.append({"id_val": set(), "id_test": set()})

    for cid in sorted(cid_2_skeys_trainval):
        cid_skeys_val = set()
        cid_skeys_test = set(cid_2_skeys_id_test.get(cid, []))

        n_skeys_train = 0
        for skey in cid_2_skeys_trainval[cid]:
            if skey in skeys_pts["train"]:
                n_skeys_train += 1
            elif skey in skeys_pts["id_val"]:
                cid_skeys_val.add(skey)

        idx_id_val_bucket = bisect.bisect_left(cfg.nst_seps, n_skeys_train)
        n_shot_tracker[idx_id_val_bucket]["id_val"].update(cid_skeys_val)

        idx_trainval_bucket = bisect.bisect_left(cfg.nst_seps, len(cid_2_skeys_trainval[cid]))
        n_shot_tracker[idx_trainval_bucket]["id_test"].update(cid_skeys_test)

    # Second pass: OOD-val classes whose samples were borrowed into id_test.
    # These species are not in cid_2_skeys_trainval, so they were missed above.
    # We bucket their id_test samples using their trainval cardinality.
    cid_2_id_test_skeys_ood = defaultdict(set)
    for skey in skeys_pts["id_test"]:
        cid = skey[0]
        if cid not in cid_2_skeys_trainval:
            cid_2_id_test_skeys_ood[cid].add(skey)

    if cid_2_id_test_skeys_ood:
        for cid in sorted(cid_2_id_test_skeys_ood):
            n_skeys_trainval_ood = len(cid_2_skeys_trainval.get(cid, []))
            if n_skeys_trainval_ood == 0:
                continue
            idx_bucket = bisect.bisect_left(cfg.nst_seps, n_skeys_trainval_ood)
            n_shot_tracker[idx_bucket]["id_test"].update(cid_2_id_test_skeys_ood[cid])

    id_eval_nshot = {
        "names": cfg.nst_names,
        "buckets": {
            name: {
                "id_val": bucket["id_val"],
                "id_test": bucket["id_test"],
            }
            for name, bucket in zip(cfg.nst_names, n_shot_tracker)
        },
    }

    return id_eval_nshot

def build_global_cid2enc(skeys_pts):
    all_cids = sorted({cid for skeys in skeys_pts.values() for cid, _ in skeys})
    global_cid2enc = {cid: enc for enc, cid in enumerate(all_cids)}
    return global_cid2enc

def build_class_counts_by_partition(data_indexes, n_classes):
    results = {}
    for pt_name, pt_data in (
        ("train",    data_indexes["train"]),
        ("trainval", data_indexes["trainval"]),
        ("whole",    data_indexes["whole"]),
    ):
        counts = np.full(n_classes, np.nan)
        for datum in pt_data:
            enc = datum["class_enc"]
            counts[enc] = 1 if np.isnan(counts[enc]) else counts[enc] + 1
        results[pt_name] = counts
    return results

def build_dev_skeys_partitions(skeys_pts, size_dev):
    return {
        pt: set(sorted(skeys_partition)[:size_dev])
        for pt, skeys_partition in skeys_pts.items()
    }

def save_split(data_indexes, id_eval_nshot, class_counts, norm_stats, dpath_split, dpath_figs) -> None:
    norm_mean = {pt: norm_stats[pt][0] for pt in norm_stats}
    norm_std = {pt: norm_stats[pt][1] for pt in norm_stats}
    split = Split(data_indexes, id_eval_nshot, class_counts, norm_mean, norm_std)
    os.makedirs(dpath_split, exist_ok=True)
    if os.path.exists(dpath_figs):
        shutil.rmtree(dpath_figs)
    os.makedirs(dpath_figs)
    save_pickle(split, dpath_split / "split.pkl")

def plot_split_distribution(
    data: List[Tuple[int]],
    labels_data: List[str],
    colors: List[str],
    title: str,
    x_label: str,
    y_label: str,
    fpath: Path,
    ema: bool = False,
    scale: Optional[str] = None,
    markers: List[str] = ["", "", "", "", ""],
    markersizes: List[int] = [6, 6, 6, 6, 6],
    markeredgewidths: List[float] = [0.5, 0.5, 0.5, 0.5, 0.5],
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    def compute_ema(vals, alpha_ema=0.99):
        ema_vals = [vals[0]]
        for i in range(1, len(vals)):
            ema_i = alpha_ema * ema_vals[-1] + (1 - alpha_ema) * vals[i]
            ema_vals.append(ema_i)
        return ema_vals

    x = range(len(data[0]))
    plt.figure(figsize=(16, 6))

    if ema:
        for i in range(len(data)):
            data[i] = compute_ema(data[i])

    for i in range(len(data)):
        plt.plot(
            x,
            data[i],
            color=colors[i],
            label=labels_data[i],
            marker=markers[i],
            markersize=markersizes[i],
            markeredgewidth=markeredgewidths[i],
            linestyle=linestyle,
            alpha=alpha,
        )

    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel(x_label, fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")

    if scale == "log":
        plt.yscale("log")
        ax = plt.gca()
        ax.yaxis.set_major_locator(mticker.SymmetricalLogLocator(linthresh=1.5, base=10, subs=[1.0, 2.0, 5.0]))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    elif scale == "symlog":
        plt.yscale("symlog", linthresh=1.5)
        ax = plt.gca()
        ax.yaxis.set_major_locator(mticker.SymmetricalLogLocator(linthresh=1.5, base=10, subs=[1.0, 2.0, 5.0]))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath, dpi=300, bbox_inches="tight")

def gen_strat_sampling_dist_plots_ood(
    penult_2_cids: Dict[str, List[str]],
    skeys_pts: Dict[str, Set[Tuple[str, int]]],
    dpath_figs: Path,
) -> None:
    """
    Generate OOD class-distribution plots and save to disk.

    For each penultimate taxonomic level, counts how many of its classes fall
    into ID, OOD-val, and OOD-test. Penultimate levels are sorted by total
    class count (descending). Three plots are saved:

    - `ood_strat_sampling_dist.png`: linear-scale line plot.
    - `ood_strat_sampling_dist_log.png`: symlog-scale scatter (no smoothing).
    - `ood_strat_sampling_dist_log_smooth.png`: log-scale line plot with EMA smoothing.

    Args:
        penult_2_cids: Mapping from penultimate-level group name to its
            list of class IDs.
        skeys_pts: Dict with partition keys mapping to sets of sample 
            keys for each partition.
        dpath_figs: Directory where plot files are written.
    """

    cids_train = {cid for cid, _ in skeys_pts["train"]}
    cids_ood_val = {cid for cid, _ in skeys_pts["ood_val"]}
    cids_ood_test = {cid for cid, _ in skeys_pts["ood_test"]}

    penult_tups = [
        (
            len(cids),
            sum(cid in cids_train for cid in cids),
            sum(cid in cids_ood_val for cid in cids),
            sum(cid in cids_ood_test for cid in cids),
        )
        for cids in penult_2_cids.values()
    ]
    penult_tups.sort(key=lambda t: t[0], reverse=True)
    n_cids_penult, n_cids_penult_train, n_cids_penult_ood_val, n_cids_penult_ood_test = zip(*penult_tups)
    n_cids_penult_ood_eval = tuple(a + b for a, b in zip(n_cids_penult_ood_val, n_cids_penult_ood_test))

    data = [n_cids_penult, n_cids_penult_train, n_cids_penult_ood_eval, n_cids_penult_ood_val, n_cids_penult_ood_test]
    colors = ["crimson", "darkorange", "teal", "steelblue", "mediumpurple"]
    labels_data = ["Total", "Train", "OOD-Eval", "OOD-Val", "OOD-Test"]
    x_label = "Sorted Penultimate Classes"
    y_label = "Num. Classes"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Stratified Sampling Distribution",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "ood_strat_sampling_dist.png",
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Stratified Sampling Distribution (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "ood_strat_sampling_dist_log.png",
        scale="symlog",
        markers=["|", "|", "|", "|", "|"],
        markersizes=[7, 5, 5, 5, 5],
        markeredgewidths=[1.0, 1.0, 1.0, 1.0, 1.0],
        linestyle="",
        alpha=1.0,
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Stratified Sampling Distribution (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "ood_strat_sampling_dist_log_smooth.png",
        ema=True,
        scale="log",
    )

def gen_strat_sampling_dist_plots_id(
    cid_2_n_samps: Dict[str, int],
    skeys_pts: Dict[str, Set[Tuple[str, int]]],
    dpath_figs: Path,
) -> None:
    """
    Generate ID class-distribution plots and save to disk.

    For each ID class with multiple samples, counts how many of its samples fall
    into train, ID-val, and ID-test. Classes are sorted by total sample count
    (descending). Three plots are saved:

    - `id_strat_sampling_dist.png`: linear-scale line plot.
    - `id_strat_sampling_dist_log.png`: symlog-scale scatter (no smoothing).
    - `id_strat_sampling_dist_log_smooth.png`: log-scale line plot with EMA smoothing.
    
    Args:
        cid_2_n_samps: Mapping from class ID to total sample count.
        skeys_pts: Dict with partition keys mapping to sets of sample 
            keys for each partition.
        dpath_figs: Directory where plot files are written.
    """

    skeys_id = skeys_pts["train"] | skeys_pts["id_val"] | skeys_pts["id_test"]
    skeys_id_multis = {skey for skey in skeys_id if cid_2_n_samps[skey[0]] > 1}
    cid_2_skeys_id_multis = build_cid_2_skeys(skeys_id_multis)

    cid_tups = []
    for cid in cid_2_skeys_id_multis:
        n_skeys = cid_2_n_samps[cid]
        n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
        for skey in cid_2_skeys_id_multis[cid]:
            if skey in skeys_pts["train"]:
                n_skeys_train += 1
            elif skey in skeys_pts["id_val"]:
                n_skeys_id_val += 1
            elif skey in skeys_pts["id_test"]:
                n_skeys_id_test += 1

        cid_tups.append((cid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test))

    cid_tups.sort(key=lambda t: (t[1], t[0]), reverse=True)
    _, n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_val, n_skeys_ps_id_test = zip(*cid_tups)
    n_skeys_ps_id_eval = tuple(a + b for a, b in zip(n_skeys_ps_id_val, n_skeys_ps_id_test))

    data = [n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_eval, n_skeys_ps_id_val, n_skeys_ps_id_test]
    colors = ["crimson", "darkorange", "teal", "steelblue", "mediumpurple"]
    labels_data = ["Total", "Train (ID)", "Eval (ID)", "Val (ID)", "Test (ID)"]
    x_label = "Sorted Classes"
    y_label = "Num. Samples"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Stratified Sampling Distribution",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "id_strat_sampling_dist.png",
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Stratified Sampling Distribution (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "id_strat_sampling_dist_log.png",
        scale="symlog",
        markers=["|", "|", "|", "|", "|"],
        markersizes=[7, 5, 5, 5, 5],
        markeredgewidths=[0.5, 0.5, 0.5, 0.5, 0.5],
        linestyle="",
        alpha=1.0,
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Stratified Sampling Distribution (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "id_strat_sampling_dist_log_smooth.png",
        ema=True,
        scale="log",
    )

def generate_n_shot_table(id_eval_nshot, dpath_figs, col_width=0.20, fontsize_title=8, fontsize=5) -> None:
    n_shot_col_names = [name for name in id_eval_nshot["names"]]

    row_values_id_val = ["ID Val"]
    row_values_id_test = ["ID Test"]
    for name in id_eval_nshot["names"]:
        bucket_skeys_set_id_val = id_eval_nshot["buckets"][name]["id_val"]
        bucket_skeys_set_id_test = id_eval_nshot["buckets"][name]["id_test"]

        num_samps_val = len(bucket_skeys_set_id_val)
        if num_samps_val > 0:
            cids, _ = zip(*bucket_skeys_set_id_val)
            n_classes_val = len(set(cids))
        else:
            n_classes_val = 0
        row_values_id_val.append(f"{num_samps_val:,} ({n_classes_val})")

        num_samps_test = len(bucket_skeys_set_id_test)
        if num_samps_test > 0:
            cids, _ = zip(*bucket_skeys_set_id_test)
            n_classes_test = len(set(cids))
        else:
            n_classes_test = 0
        row_values_id_test.append(f"{num_samps_test:,} ({n_classes_test:,})")

    labels_cols = ["Partition"] + n_shot_col_names
    data = [row_values_id_val, row_values_id_test]

    _, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")

    col_widths = [col_width] * len(labels_cols)

    tbl = ax.table(
        cellText=data,
        colLabels=labels_cols,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    for col_idx, _ in enumerate(labels_cols):
        cell = tbl[0, col_idx]
        cell.get_text().set_fontweight("bold")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    for (_, _), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)

    plt.title("n-shot Bucket Sample (Class) Counts", fontsize=fontsize_title, fontweight="bold", y=0.70)
    plt.savefig(dpath_figs / "summary_nshot.png", dpi=300, bbox_inches="tight")

def count_unique_cids_from_skeys(skeys) -> int:
    return len({cid for cid, _ in skeys})

def count_total_samples_disjoint_partitions(skeys_pts) -> int:
    return len(
        skeys_pts["train"]
        | skeys_pts["id_val"]
        | skeys_pts["id_test"]
        | skeys_pts["ood_val"]
        | skeys_pts["ood_test"]
    )

def render_partition_summary_table(
    labels_cols,
    data,
    title,
    dpath_figs,
    figsize=(5, 2),
    pad=-5,
    dpi=150,
    fontsize=None,
    scale=None,
) -> None:
    _, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=data,
        colLabels=labels_cols,
        cellLoc="center",
        loc="center",
    )

    for col_idx, _ in enumerate(labels_cols):
        cell = tbl[0, col_idx]
        cell.get_text().set_fontweight("bold")

    if fontsize is not None:
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)

    if scale is not None:
        tbl.scale(*scale)

    plt.title(title, fontweight="bold", pad=pad)
    plt.savefig(dpath_figs / "summary_partitions.png", dpi=dpi, bbox_inches="tight")
    plt.close()

def generate_partition_summary_table(
    skeys_pts,
    dpath_figs,
    n_cids_total,
    title,
    labels_cols=None,
    figsize=(5, 2),
    pad=-5,
    dpi=150,
    fontsize=None,
    scale=None,
) -> None:
    if labels_cols is None:
        labels_cols = ["Partition", "Num. Classes", "Num. Samples"]

    n_samps_total = count_total_samples_disjoint_partitions(skeys_pts)
    row_specs = [
        ("OOD Test", "ood_test"),
        ("ID Test", "id_test"),
        ("OOD Val", "ood_val"),
        ("ID Val", "id_val"),
        ("Train", "train"),
        ("TrainVal", "trainval"),
        ("Whole", "whole"),
    ]

    data = []
    for row_name, partition in row_specs:
        skeys_partition = skeys_pts[partition]
        n_cids_partition = count_unique_cids_from_skeys(skeys_partition)
        n_samps_partition = len(skeys_partition)
        data.append([
            row_name,
            f"{n_cids_partition:,} ({n_cids_partition / n_cids_total:.2%})",
            f"{n_samps_partition:,} ({n_samps_partition / n_samps_total:.2%})",
        ])

    render_partition_summary_table(
        labels_cols=labels_cols,
        data=data,
        title=title,
        dpath_figs=dpath_figs,
        figsize=figsize,
        pad=pad,
        dpi=dpi,
        fontsize=fontsize,
        scale=scale,
    )
