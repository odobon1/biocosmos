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
import pandas as pd

from utils.data import Split
from utils.utils import paths, save_pickle, seed_libs, load_pickle
from utils.config import get_config_splits


DATASET2FANCY = {
    "bryo": "Bryozoa",
    "cub": "CUB",
    "lepid": "Lepidoptera",
    "nymph": "Nymphalidae",
}


class GenSplitDataManager:

    dataset = None

    class_data = None

    dpath_split = None
    dpath_figs = None
    dpath_split_dev = None
    dpath_figs_dev = None

    @staticmethod
    def setup(dataset):
        GenSplitDataManager.dataset = dataset
        GenSplitDataManager.cfg = cfg = get_config_splits()
        print(f"Generating split: '{cfg.split}'...")
        seed_libs(cfg.seed, seed_torch=False)
        GenSplitDataManager.class_data = load_pickle(paths["metadata"][dataset] / "class_data.pkl")

        GenSplitDataManager.dpath_split = paths["metadata"][dataset] / f"splits/{cfg.split}"
        GenSplitDataManager.dpath_figs = GenSplitDataManager.dpath_split / "figures"
        GenSplitDataManager.dpath_split_dev = paths["metadata"][dataset] / "splits/dev"
        GenSplitDataManager.dpath_figs_dev = GenSplitDataManager.dpath_split_dev / "figures"

    @staticmethod
    def get_cids():
        return sorted(GenSplitDataManager.class_data.keys())


def _process_rfpaths_parallel(rfpaths, desc):
    imgs_root = paths["imgs"][GenSplitDataManager.dataset]
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

def get_norm_stats(
    data_indexes, 
) -> Dict[str, Tuple[Tuple[float], Tuple[float]]]:
    dev_cfg = GenSplitDataManager.cfg.dev
    if dev_cfg.get("debug_mode", False) and dev_cfg["debug"].get("skip_norm_stats", False):
        print("***** Skipping norm stats computation in dev mode; returning dummy values *****")
        return {pt: ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) for pt in ("train", "trainval")}
    return compute_rgb_norm_stats_by_partition(data_indexes)

def compute_rgb_norm_stats_by_partition(
    data_indexes: Dict[str, Any],
) -> Dict[str, Tuple[Tuple[float], Tuple[float]]]:
    """
    Norm stats accumulated incrementally: train → snapshot, +val → snapshot, 
    +test → snapshot; Each image contributes equally regardless of resolution.
    """
    groups = [
        ("train",    [d["rfpath"] for d in data_indexes["train"]]),
        ("trainval", [d["rfpath"] for d in data_indexes["val"]["id"]] +
                     [d["rfpath"] for d in data_indexes["val"]["ood"]])
    ]

    all_means: List = []
    all_vars: List = []
    results = {}

    for pt_name, rfpaths in groups:
        if rfpaths:
            new_means, new_vars = _process_rfpaths_parallel(
                rfpaths, desc=f"Computing norm stats ({pt_name})"
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
    n_cids_all: int,
    n_samps_all: int,
    ood_tol_flag: bool,
):
    """
    Pick exactly round(n_cids_all * pct_partition) classes for OOD-val,
    stratified by penultimate class.
    """
    cid2sidxs_pool = build_cid2sidxs(skeys_pool)
    cids_pool = set(cid2sidxs_pool.keys())
    n_cids_pt = round(n_cids_all * GenSplitDataManager.cfg.pct_partition)

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
            print(f"Warning: {i / 1_000}k seeds searched and no OOD partition found satisfying pct_ood_tol={GenSplitDataManager.cfg.pct_ood_tol}")
        
        cids_pt, cids_rem = strat_sample_partition(
            n_classes=len(penult_2_cids),
            n_draws=n_cids_pt,
            n_insts_2_classes=n_insts_2_classes_penult_lvl,
            class_2_insts=penult_2_cids,
            insts=set(cids_pool),
            seed=GenSplitDataManager.cfg.seed + i,
        )

        skeys_pt = {
            (cid, sidx)
            for cid in cids_pt
            for sidx in cid2sidxs_pool[cid]
        }
        pct_samps_pt = len(skeys_pt) / n_samps_all
        close_enough = abs(pct_samps_pt - GenSplitDataManager.cfg.pct_partition) <= GenSplitDataManager.cfg.pct_ood_tol
        if close_enough or not ood_tol_flag:
            sample_once_more = False

    skeys_rem = {
        (cid, sidx)
        for cid in cids_rem
        for sidx in cid2sidxs_pool[cid]
    }

    return skeys_pt, skeys_rem

def strat_sample_partition_id(
    skeys_pool: Set[Tuple[str, int]], 
    n_samps_all: int,
) -> Tuple[
    Set[Tuple[str, int]],
    Set[Tuple[str, int]],
]:

    cid2sidxs_pool = build_cid2sidxs(skeys_pool)
    cids_pool = set(cid2sidxs_pool.keys())

    cids_pool_singles = {
        cid
        for cid in sorted(cids_pool)
        if len(cid2sidxs_pool[cid]) == 1
    }
    cids_pool_multis = cids_pool - cids_pool_singles

    skeys_pool_multis = {
        (cid, sidx)
        for cid in cids_pool_multis
        for sidx in cid2sidxs_pool[cid]
    }

    n_samps_pt = round(n_samps_all * GenSplitDataManager.cfg.pct_partition)

    n_insts_2_classes_leaf_lvl = defaultdict(list)
    cid_2_skeys_pool_multis = defaultdict(list)

    for cid in sorted(cids_pool_multis):
        skeys_pool_multis_cid = [(cid, sidx) for sidx in sorted(cid2sidxs_pool[cid])]
        cid_2_skeys_pool_multis[cid] = skeys_pool_multis_cid
        n_insts_2_classes_leaf_lvl[len(skeys_pool_multis_cid)].append(cid)

    skeys_pt, skeys_pool_multis = strat_sample_partition(
        n_classes=len(cids_pool_multis),
        n_draws=n_samps_pt,
        n_insts_2_classes=n_insts_2_classes_leaf_lvl,
        class_2_insts=cid_2_skeys_pool_multis,
        insts=skeys_pool_multis,
        seed=GenSplitDataManager.cfg.seed,
    )

    skeys_singles = {
        (cid, cid2sidxs_pool[cid][0])
        for cid in sorted(cids_pool_singles)
    }
    skeys_pool = skeys_pool_multis | skeys_singles

    return skeys_pt, skeys_pool

def strat_sample_ood_id(
    skeys_pool: Set[Tuple[str, int]],
    n_cids_all: int,
    n_samps_all: int,
    cid_2_penult: Dict[str, str],
    ood_tol_flag: bool = True,
) -> Tuple[
    Set[Tuple[str, int]],
    Set[Tuple[str, int]],
    Set[Tuple[str, int]],
]:
    skeys_pt_ood, skeys_pool = strat_sample_partition_ood(
        skeys_pool,
        cid_2_penult,
        n_cids_all,
        n_samps_all,
        ood_tol_flag,
    )
    skeys_pt_id, skeys_pool = strat_sample_partition_id(
        skeys_pool, 
        n_samps_all, 
    )
    return skeys_pt_ood, skeys_pt_id, skeys_pool

def build_cid2sidxs(skeys: Set[Tuple[str, int]]) -> Dict[str, List[int]]:
    cid2sidxs = defaultdict(list)
    for cid, samp_idx in sorted(skeys):
        cid2sidxs[cid].append(samp_idx)
    return cid2sidxs

def add_trainval(skeys_pts):
    skeys_pts["trainval"] = skeys_pts["train"] | skeys_pts["val_id"] | skeys_pts["val_ood"]

def build_nshot(skeys_pts, cfg=None):

    if cfg is None:
        cfg = GenSplitDataManager.cfg

    cid2sidxs_trainval = build_cid2sidxs(skeys_pts["trainval"])
    cid2sidxs_test_id = build_cid2sidxs(skeys_pts["test_id"])

    n_shot_tracker = []
    for _ in range(len(cfg.nst_names)):
        n_shot_tracker.append({"train/val": set(), "trainval/test": set()})

    for cid in sorted(cid2sidxs_trainval):
        n_skeys_train = 0
        in_val_id = False
        for sidx in cid2sidxs_trainval[cid]:
            if (cid, sidx) in skeys_pts["train"]:
                n_skeys_train += 1
            elif (cid, sidx) in skeys_pts["val_id"]:
                in_val_id = True

        if in_val_id:
            idx_val_id_bucket = bisect.bisect_left(cfg.nst_seps, n_skeys_train)
            n_shot_tracker[idx_val_id_bucket]["train/val"].add(cid)

        if cid in cid2sidxs_test_id:
            idx_trainval_bucket = bisect.bisect_left(cfg.nst_seps, len(cid2sidxs_trainval[cid]))
            n_shot_tracker[idx_trainval_bucket]["trainval/test"].add(cid)

    nshot = {
        "names": cfg.nst_names,
        "buckets": {
            "train/val": {name: bucket["train/val"]  for name, bucket in zip(cfg.nst_names, n_shot_tracker)},
            "trainval/test": {name: bucket["trainval/test"] for name, bucket in zip(cfg.nst_names, n_shot_tracker)},
        },
    }

    return nshot

def build_global_cid2enc(skeys_pts):
    all_cids = sorted({cid for skeys in skeys_pts.values() for cid, _ in skeys})
    global_cid2enc = {cid: enc for enc, cid in enumerate(all_cids)}
    return global_cid2enc

def build_class_counts_by_partition(data_indexes, n_classes):
    results = {}
    for pt_name, pt_data in (
        ("train",    data_indexes["train"]),
        ("trainval", data_indexes["trainval"]),
    ):
        counts = np.full(n_classes, np.nan)
        for datum in pt_data:
            enc = datum["class_enc"]
            counts[enc] = 1 if np.isnan(counts[enc]) else counts[enc] + 1
        results[pt_name] = counts
    return results

def build_skey2meta(
    skeys_pts,
    img_ptrs, 
) -> Dict[Tuple[str, int], Optional[Dict[str, Optional[str]]]]:
    skeys_all = {skey for pt in ("train", "val_id", "val_ood", "test_id", "test_ood") for skey in skeys_pts[pt]}
    if GenSplitDataManager.dataset in ("lepid", "nymph"):
        df = pd.read_csv(paths["csv"][GenSplitDataManager.dataset]["imgs"])
        lookup = df.set_index("mask_name")[["class_dv", "sex"]]
        result = {}
        for skey in skeys_all:
            cid, samp_idx = skey
            fname = img_ptrs[cid][samp_idx].split("/")[-1]
            pos = lookup["class_dv"].get(fname)
            sex = lookup["sex"].get(fname)
            result[(cid, samp_idx)] = {
                "pos": None if pd.isna(pos) else pos,
                "sex": None if pd.isna(sex) else sex,
            }
    else:
        result = {skey: None for skey in skeys_all}
    return result

def build_data_indexes(
    skeys_pts,
    img_ptrs,
    cid2enc,
    skey2meta: Dict[Tuple[str, int], Optional[Dict[str, Optional[str]]]],
):

    def build_partition_index(partition):
        data_index = []

        for skey in sorted(skeys_pts[partition]):
            cid, samp_idx = skey
            data_index.append(
                {
                    "class_enc": cid2enc[cid],
                    "rfpath": img_ptrs[cid][samp_idx],
                    "meta": skey2meta[skey],
                }
            )
        return data_index

    return {
        "train": build_partition_index("train"),
        "val": {
            "id": build_partition_index("val_id"),
            "ood": build_partition_index("val_ood"),
        },
        "trainval": build_partition_index("trainval"),
        "test": {
            "id": build_partition_index("test_id"),
            "ood": build_partition_index("test_ood"),
        },
    }

def build_dev_skeys_partitions(skeys_pts):
    size_dev = GenSplitDataManager.cfg.size_dev
    return {
        pt: set(sorted(skeys_partition)[:size_dev])
        for pt, skeys_partition in skeys_pts.items()
    }

def save_split(data_indexes, enc2cid, nshot, class_counts, norm_stats, dpath_split) -> None:
    norm_mean = {pt: norm_stats[pt][0] for pt in norm_stats}
    norm_std = {pt: norm_stats[pt][1] for pt in norm_stats}
    split = Split(data_indexes, enc2cid, nshot, class_counts, norm_mean, norm_std)
    os.makedirs(dpath_split, exist_ok=True)
    dpath_figs = dpath_split / "figures"
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
    """

    cids_train = {cid for cid, _ in skeys_pts["train"]}
    cids_val_ood = {cid for cid, _ in skeys_pts["val_ood"]}
    cids_test_ood = {cid for cid, _ in skeys_pts["test_ood"]}

    penult_tups = [
        (
            len(cids),
            sum(cid in cids_train for cid in cids),
            sum(cid in cids_val_ood for cid in cids),
            sum(cid in cids_test_ood for cid in cids),
        )
        for cids in penult_2_cids.values()
    ]
    penult_tups.sort(key=lambda t: t[0], reverse=True)
    n_cids_penult, n_cids_penult_train, n_cids_penult_val_ood, n_cids_penult_test_ood = zip(*penult_tups)
    n_cids_penult_ood_eval = tuple(a + b for a, b in zip(n_cids_penult_val_ood, n_cids_penult_test_ood))

    data = [n_cids_penult, n_cids_penult_train, n_cids_penult_ood_eval, n_cids_penult_val_ood, n_cids_penult_test_ood]
    colors = ["crimson", "darkorange", "teal", "steelblue", "mediumpurple"]
    labels_data = ["Total", "Train", "OOD-Eval", "OOD-Val", "OOD-Test"]
    x_label = "Sorted Penultimate Classes"
    y_label = "Num. Classes"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title=f"{DATASET2FANCY[GenSplitDataManager.dataset]} - OOD Partition Distributions",
        x_label=x_label,
        y_label=y_label,
        fpath=GenSplitDataManager.dpath_figs / "ood_strat_sampling_dist.png",
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title=f"{DATASET2FANCY[GenSplitDataManager.dataset]} - OOD Partition Distributions (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        fpath=GenSplitDataManager.dpath_figs / "ood_strat_sampling_dist_log.png",
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
        title=f"{DATASET2FANCY[GenSplitDataManager.dataset]} - OOD Partition Distributions (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        fpath=GenSplitDataManager.dpath_figs / "ood_strat_sampling_dist_log_smooth.png",
        ema=True,
        scale="log",
    )

def gen_strat_sampling_dist_plots_id(
    cid_2_n_samps: Dict[str, int],
    skeys_pts: Dict[str, Set[Tuple[str, int]]],
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

    cids_train = {cid for cid, _ in skeys_pts["train"]}
    skeys_id = skeys_pts["train"] | skeys_pts["val_id"] | skeys_pts["test_id"]
    # skeys that are in train/id-val/id-test with cids in train
    skeys_multis = {(cid, samp_key) for cid, samp_key in skeys_id if cid_2_n_samps[cid] > 1 and cid in cids_train}
    cid2sidxs_multis = build_cid2sidxs(skeys_multis)

    n_skeys_cids = []
    for cid in cid2sidxs_multis:
        n_skeys_all_cid = cid_2_n_samps[cid]
        n_skeys_train_cid, n_skeys_val_id_cid, n_skeys_test_id_cid = 0, 0, 0
        for sidx in cid2sidxs_multis[cid]:
            if (cid, sidx) in skeys_pts["train"]:
                n_skeys_train_cid += 1
            elif (cid, sidx) in skeys_pts["val_id"]:
                n_skeys_val_id_cid += 1
            elif (cid, sidx) in skeys_pts["test_id"]:
                n_skeys_test_id_cid += 1

        n_skeys_cids.append((cid, n_skeys_all_cid, n_skeys_train_cid, n_skeys_val_id_cid, n_skeys_test_id_cid))

    n_skeys_cids.sort(key=lambda t: (t[1], t[0]), reverse=True)
    _, n_skeys_all, n_skeys_train, n_skeys_val_id, n_skeys_test_id = zip(*n_skeys_cids)
    n_skeys_eval_id = tuple(a + b for a, b in zip(n_skeys_val_id, n_skeys_test_id))

    data = [n_skeys_all, n_skeys_train, n_skeys_eval_id, n_skeys_val_id, n_skeys_test_id]
    colors = ["crimson", "darkorange", "teal", "steelblue", "mediumpurple"]
    labels_data = ["Total", "Train", "ID Eval", "ID Val", "ID Test"]
    x_label = "Sorted Classes"
    y_label = "Num. Samples"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title=f"{DATASET2FANCY[GenSplitDataManager.dataset]} - ID Partition Distributions",
        x_label=x_label,
        y_label=y_label,
        fpath=GenSplitDataManager.dpath_figs / "id_strat_sampling_dist.png",
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title=f"{DATASET2FANCY[GenSplitDataManager.dataset]} - ID Partition Distributions (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        fpath=GenSplitDataManager.dpath_figs / "id_strat_sampling_dist_log.png",
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
        title=f"{DATASET2FANCY[GenSplitDataManager.dataset]} - ID Partition Distributions (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        fpath=GenSplitDataManager.dpath_figs / "id_strat_sampling_dist_log_smooth.png",
        ema=True,
        scale="log",
    )

def generate_n_shot_table(
    nshot, 
    skeys_pts, 
    col_widths=[0.30, 0.20], 
    fontsize_title=8, 
    fontsize=5,
) -> None:
    cid_2_n_val_id = defaultdict(int)
    for cid, _ in skeys_pts["val_id"]:
        cid_2_n_val_id[cid] += 1
    cid_2_n_test_id = defaultdict(int)
    for cid, _ in skeys_pts["test_id"]:
        cid_2_n_test_id[cid] += 1

    n_shot_col_names = list(nshot["names"])

    row_values_val = ["Train / ID Val"]
    row_values_test = ["TrainVal / ID Test"]
    for name in nshot["names"]:
        cids_val_id = nshot["buckets"]["train/val"][name]
        cids_test_id = nshot["buckets"]["trainval/test"][name]

        n_classes_val = len(cids_val_id)
        num_samps_val = sum(cid_2_n_val_id[cid] for cid in cids_val_id)
        row_values_val.append(f"{num_samps_val:,} ({n_classes_val})")

        n_classes_test = len(cids_test_id)
        num_samps_test = sum(cid_2_n_test_id[cid] for cid in cids_test_id)
        row_values_test.append(f"{num_samps_test:,} ({n_classes_test:,})")

    labels_cols = ["Partitions (Train / Eval)"] + n_shot_col_names
    data = [row_values_val, row_values_test]

    _, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")

    col_widths = [col_widths[0]] + [col_widths[1]] * (len(labels_cols) - 1)

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

    plt.title(
        f"{DATASET2FANCY[GenSplitDataManager.dataset]} - n-shot Bucket Sample (Class) Counts",
        fontsize=fontsize_title, 
        fontweight="bold", 
        y=0.70,
    )
    plt.savefig(GenSplitDataManager.dpath_figs / "summary_nshot.png", dpi=300, bbox_inches="tight")

def count_unique_cids_from_skeys(skeys) -> int:
    return len({cid for cid, _ in skeys})

def count_total_samples_disjoint_partitions(skeys_pts) -> int:
    return len(
        skeys_pts["train"]
        | skeys_pts["val_id"]
        | skeys_pts["test_id"]
        | skeys_pts["val_ood"]
        | skeys_pts["test_ood"]
    )

def render_partition_summary_table(
    labels_cols,
    data,
    title,
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
    plt.savefig(GenSplitDataManager.dpath_figs / "summary_partitions.png", dpi=dpi, bbox_inches="tight")
    plt.close()

def generate_partition_summary_table(
    skeys_pts,
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
        ("Train", "train"),
        ("TrainVal", "trainval"),
        ("ID Val", "val_id"),
        ("ID Test", "test_id"),
        ("OOD Val", "val_ood"),
        ("OOD Test", "test_ood"),
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
        figsize=figsize,
        pad=pad,
        dpi=dpi,
        fontsize=fontsize,
        scale=scale,
    )

def generate_splits(
    cids,
    skeys_pts,
    skeys_pts_dev,
    img_ptrs,
    penult_2_cids,
    cid_2_n_samps,
) -> None:

    # N-SHOT TRACKING

    print("Constructing n-shot tracking structures...")
    nshot = build_nshot(skeys_pts)
    print("n-shot tracking complete!")

    # GENERATE DATA INDEXES

    print("Generating data indexes...")
    cid2enc = build_global_cid2enc(skeys_pts)
    enc2cid = {enc: cid for cid, enc in cid2enc.items()}
    skey2meta = build_skey2meta(skeys_pts, img_ptrs)
    data_indexes = build_data_indexes(skeys_pts, img_ptrs, cid2enc, skey2meta)
    data_indexes_dev = build_data_indexes(skeys_pts_dev, img_ptrs, cid2enc, skey2meta)
    print("Data indexes complete!")

    # CLASS COUNTS

    print("Generating class counts for train partitions...")
    class_counts = build_class_counts_by_partition(data_indexes, len(cids))
    class_counts_dev = build_class_counts_by_partition(data_indexes_dev, len(cids))
    print("Class counts complete!")

    # TRAIN PARTITIONS NORMALIZATION STATS

    norm_stats = get_norm_stats(data_indexes)

    # SAVE SPLITS

    print("Saving splits...")
    save_split(
        data_indexes,
        enc2cid,
        nshot,
        class_counts,
        norm_stats,
        GenSplitDataManager.dpath_split,
    )
    save_split(
        data_indexes_dev,
        enc2cid,
        nshot,
        class_counts_dev,
        norm_stats,
        GenSplitDataManager.dpath_split_dev,
    )
    print("Splits saved!")

    # OOD STRATIFIED SAMPLING DISTRIBUTION PLOTTING

    print("Generating OOD stratified sampling distribution plots...")
    gen_strat_sampling_dist_plots_ood(
        penult_2_cids,
        skeys_pts,
    )
    print("OOD stratified sampling distribution plots complete!")

    # ID STRATIFIED SAMPLING DISTRIBUTION PLOTTING (singletons omitted)

    print("Generating ID stratified sampling distribution plots...")
    gen_strat_sampling_dist_plots_id(
        cid_2_n_samps,
        skeys_pts,
    )
    print("ID stratified sampling distribution plots complete!")

    # PARTITION SUMMARY TABLE

    print("Generating partition summary table...")
    generate_partition_summary_table(
        skeys_pts=skeys_pts,
        n_cids_total=len(cids),
        title=f"{DATASET2FANCY[GenSplitDataManager.dataset]} Partitions",
    )
    print("Partition summary table complete!")

    # N-SHOT BUCKET SUMMARY TABLE

    print("Generating n-shot bucket summary table...")
    generate_n_shot_table(nshot, skeys_pts)
    print("n-shot tracking table complete!")

    # SPLIT GENERATION COMPLETE

    print("Splits complete!")