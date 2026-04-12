import random
from sklearn.model_selection import train_test_split  # type: ignore[import]
import copy
import matplotlib.pyplot as plt  # type: ignore[import]
import bisect
import os
import numpy as np  # type: ignore[import]
import glob
from tqdm import tqdm  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from collections import Counter, defaultdict

from utils.data import assemble_indexes, Split, sid_to_genus
from utils.utils import paths, save_pickle

import pdb


def gen_genus_2_sids(sids):
    """
    `genus_2_sids` & `sid_2_skeys_id_multis` structure (class_2_insts):

    genus_2_sids:
    {
        genus0: [sid0, sid1, sid2, ...],
        genus1: [...],
        ...
    }

    sid_2_skeys_id_multis:
    {
        sid0: [skey0, skey1, skey2, ...],
        sid1: [...],
        ...
    }
    """
    genus_2_sids = defaultdict(list)
    for sid in sids:
        genus = sid_to_genus(sid)
        genus_2_sids[genus].append(sid)
    return genus_2_sids

def gen_n_insts_2_classes_g(sids):
    """
    `n_insts_2_classes_*` structure:

    n_insts_2_classes_g (OOD):
    {
        1: [genus0, genus1, genus2, ...],
        2: [...],
        4: [...],
        ...
    }

    n_insts_2_classes_s (ID):
    {
        1: [sid0, sid1, sid2, ...],
        2: [...],
        ...
    }
    """
    genera = [sid_to_genus(sid) for sid in sids]
    count_g = Counter(genera)
    n_insts_2_classes_g = defaultdict(list)
    for genus, count in count_g.items():
        n_insts_2_classes_g[count].append(genus)
    return n_insts_2_classes_g

def strat_split(n_classes, n_draws, pct_eval, n_insts_2_classes, class_2_insts, insts, seed=None):
    """
    Args:
    - n_classes -------------------------------
    - n_draws ---------------------------------
    - pct_eval -------------------------------- percentage for val/test, evenly distributed between both e.g. 10% yields 5% val, 5% test
    - n_insts_2_classes -----------------------
    - class_2_insts --------------------------- dictionary mapping classes [str] to lists of instances [List(str)]
    - insts ----------------------------------- set of instances

    Returns
    - [(set(insts), set(insts), set(insts)] --- Train, Val, Test
    """
    rng = random.Random(seed)  # local random number generator

    def compute_class_hits(n_draws, n_classes):
        """
        num_draws, num_classes --> list(class_hits) e.g. [1,1,1,0,0,0,0,0], [1,1,1,1,0,0], [2,1,1,1,1,1], etc.
        """

        class_hits = [n_draws // n_classes] * n_classes

        plus_ones = n_draws % n_classes
        for i in range(plus_ones):
            class_hits[i] += 1

        return class_hits

    insts_rem = copy.deepcopy(insts)
    insts_eval = []

    n_classes_rem = n_classes
    n_draws_rem   = n_draws

    count_min_strat2 = 1 / pct_eval
    i = 0
    while True:
        i += 1
        classes_i = list(n_insts_2_classes[i])
        if not classes_i:
            # n_insts_2_classes[i] is empty i.e. no classes at count i
            continue

        n_classes_i = len(classes_i)
        n_instances_i = n_classes_i * i
        n_draws_i = round(n_instances_i * pct_eval)

        rng.shuffle(classes_i)
        class_hits = compute_class_hits(n_draws_i, n_classes_i)

        for idx, k in enumerate(class_hits):
            c = classes_i[idx]

            inst_hits = rng.sample(class_2_insts[c], k)

            insts_eval += inst_hits

        n_classes_rem -= n_classes_i
        n_draws_rem -= n_draws_i

        if i >= count_min_strat2 - 1 and (n_draws_rem >= n_classes_rem or n_classes_rem == 0):
            break

    if n_draws_rem > 0 and n_classes_rem > 0:
        # construct classes_rem & instances_rem (instances_rem structured as tuples with inst_count for sorting, sorting is important for the zipper delegation between val/test)
        classes_rem = []
        insts_counts_rem = []  # List((instance, count))  ~ `count` is the number of instances in the corresponding class

        for count in sorted(list(n_insts_2_classes.keys())):
            if count <= i:
                continue
            else:
                classes = n_insts_2_classes[count]
                for c in classes:
                    classes_rem += [c] * count

                    insts_c = class_2_insts[c]  # list of instances
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

    insts_val = set()
    insts_test = set()

    # "zipper delegation"
    for idx, sid in enumerate(insts_eval):
        if idx % 2 == 0:
            insts_val.add(sid)
        else:
            insts_test.add(sid)

    insts_rem -= insts_val
    insts_rem -= insts_test

    return insts_rem, insts_val, insts_test

def gen_ood_partitions(
    n_genera,
    n_sids_ood_eval,
    pct_ood_eval,
    n_insts_2_classes_g,
    genus_2_sids,
    sids,
    cfg,
    sid_2_samp_idxs,
    n_samps_dict,
    n_samps_total,
):

    close_enough = False
    i = 0
    while not close_enough:
        i += 1
        sids_id, sids_ood_val, sids_ood_test = strat_split(
            n_classes=n_genera, 
            n_draws=n_sids_ood_eval, 
            pct_eval=pct_ood_eval, 
            n_insts_2_classes=n_insts_2_classes_g, 
            class_2_insts=genus_2_sids, 
            insts=sids,
            seed=cfg.seed+i,
        )

        # NUM SAMPLES CHECK

        n_samps_ood_val = 0
        for sid in sids_ood_val:
            n_samps_sid = n_samps_dict[sid]
            n_samps_ood_val += n_samps_sid
        pct_samps_ood_val = n_samps_ood_val / n_samps_total

        n_samps_ood_test = 0
        for sid in sids_ood_test:
            n_samps_sid = n_samps_dict[sid]
            n_samps_ood_test += n_samps_sid
        pct_samps_ood_test = n_samps_ood_test / n_samps_total

        if abs((pct_ood_eval / 2) - pct_samps_ood_val) < cfg.pct_ood_tol and abs((pct_ood_eval / 2) - pct_samps_ood_test) < cfg.pct_ood_tol:
            close_enough = True

    skeys_ood_val = set()
    for sid in sids_ood_val:
        for samp_idx in sid_2_samp_idxs[sid]:
            skey = (sid, samp_idx)
            skeys_ood_val.add(skey)

    skeys_ood_test = set()
    for sid in sids_ood_test:
        for samp_idx in sid_2_samp_idxs[sid]:
            skey = (sid, samp_idx)
            skeys_ood_test.add(skey)

    return sids_id, sids_ood_val, sids_ood_test, skeys_ood_val, skeys_ood_test

def gen_id_partitions(
    sids_id,
    sid_2_samp_idxs,
    n_samps_dict,
    n_samps_id_eval,
    pct_id_eval,
    cfg,
):

    sids_id_singles = set() # species id's with 1 sample i.e. singletons
    for sid in sorted(sids_id):
        if n_samps_dict[sid] == 1:
            sids_id_singles.add(sid)

    sids_id_multis = sids_id - sids_id_singles  # species id's with 2+ samples

    n_sids_id_multis = len(sids_id_multis)  # ID partitions: n_classes

    n_insts_2_classes_s = defaultdict(list)  # ID partitions: n_insts_2_classes
    for sid in sorted(sids_id_multis):
        count = n_samps_dict[sid]
        n_insts_2_classes_s[count].append(sid)

    pct_rem_id_eval = pct_id_eval / (1 - pct_id_eval)  # ID partitions: pct_eval (10 / 90)

    sid_2_skeys_id_multis = defaultdict(list)  # ID partitions: class_2_insts
    sid_2_skeys_id = defaultdict(list)  # used for n-shot tracking
    skeys_id_multis = set()  # ID partitions: insts

    for sid in sorted(sids_id):
        for samp_idx in sid_2_samp_idxs[sid]:
            skey = (sid, samp_idx)
            sid_2_skeys_id[sid].append(skey)
            if sid in sids_id_multis:
                sid_2_skeys_id_multis[sid].append(skey)
                skeys_id_multis.add(skey)

    skeys_train_multis, skeys_id_val, skeys_id_test = strat_split(
        n_classes=n_sids_id_multis, 
        n_draws=n_samps_id_eval, 
        pct_eval=pct_rem_id_eval, 
        n_insts_2_classes=n_insts_2_classes_s, 
        class_2_insts=sid_2_skeys_id_multis, 
        insts=skeys_id_multis,
        seed=cfg.seed,
    )

    skeys_id_singles = set((sid, sid_2_samp_idxs[sid][0]) for sid in sids_id_singles)
    skeys_train = skeys_train_multis.union(skeys_id_singles)

    return skeys_train, skeys_id_val, skeys_id_test, sid_2_skeys_id, sid_2_skeys_id_multis, sids_id_multis

def gen_id_eval_nshot(cfg, sids_id, skeys_partitions, sid_2_skeys_id):

    def find_range_index(nst_seps, n):
        assert isinstance(n, int), f"n must be an int, got {type(n).__name__}"
        if n <= 0:
            raise ValueError(f"find_range_index(): n = {n}")

        return bisect.bisect_left(nst_seps, n)

    """
    n-shot tracking data structures

    `n_shot_tracker` structure:
    ~ list-style ~
    (for assembly)

    n_shot_tracker = [
        {
            "train": set(skeys),
            "id_val": set(skeys),
            "id_test": set(skeys),
        },
        {...}.
        ...
    ]

    `id_eval_nshot` structure:
    ~ dict-style ~
    (for saving to file)

    id_eval_nshot = {
        "names": nst_names (it's a list),
        "buckets": {
            nst_name0: {
                "id_val": set(skeys),
                "id_test": set(skeys),
            }.
            nst_name1: {
                ...
            },
            ...
        },
    }

    """

    n_shot_tracker = []

    for _ in range(len(cfg.nst_names)):
        nst_bucket = {
            "train": set(),
            "id_val": set(),
            "id_test": set(),
        }
        n_shot_tracker.append(nst_bucket)

    for sid in sorted(sids_id):

        sid_skeys_train = set()
        sid_skeys_val   = set()
        sid_skeys_test  = set()

        n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
        for skey in sid_2_skeys_id[sid]:
            # check which set the sample landed in
            if skey in skeys_partitions["train"]:
                n_skeys_train += 1
                sid_skeys_train.add(skey)
            elif skey in skeys_partitions["id_val"]:
                n_skeys_id_val += 1
                sid_skeys_val.add(skey)
            elif skey in skeys_partitions["id_test"]:
                n_skeys_id_test += 1
                sid_skeys_test.add(skey)

        idx_nst_bucket = find_range_index(cfg.nst_seps, n_skeys_train)
        n_shot_tracker[idx_nst_bucket]["train"].update(sid_skeys_train)
        n_shot_tracker[idx_nst_bucket]["id_val"].update(sid_skeys_val)
        n_shot_tracker[idx_nst_bucket]["id_test"].update(sid_skeys_test)

    # list-style --> dict-style
    id_eval_nshot = {
        "names": cfg.nst_names,  # just to preserve ordering of the names if needed
        "buckets": {
            name: {
                "id_val": bucket["id_val"],
                "id_test": bucket["id_test"],
            }
            for name, bucket in zip(cfg.nst_names, n_shot_tracker)
        },
    }

    return id_eval_nshot

def gen_img_ptrs(sids):

    """
    `img_ptrs` structure:

    img_ptrs = {
        sid0: {
            0: fpath_img_s0_0,
            1: fpath_img_s0_1,
            ...
        },
        sid1: {
            ...
        },
        ...
    }
    """

    img_ptrs = {}

    # iterate through sids, fetch image filenames, assign to indexes, add to img_ptrs structure
    for sid in tqdm(sorted(sids)):
        
        img_ptrs[sid] = {}

        dpath_imgs_sid = paths["nymph_imgs"] / sid
        ffpaths_png    = sorted(glob.glob(f"{dpath_imgs_sid}/*.png"))
        rfpaths_png    = [png_file.split("images/", 1)[1] for png_file in ffpaths_png]  # full filepath --> relative filepath

        for i, rfpath in enumerate(rfpaths_png):
            img_ptrs[sid][i] = rfpath

    return img_ptrs

def gen_sid_2_samp_idxs(
    sids,
    pos_filter=None,
    img_ptrs=None,
    df_metadata=None,
):

    if pos_filter is None:
        if img_ptrs is None:
            img_ptrs = gen_img_ptrs(sids)
        return {
            sid: list(img_ptrs[sid].keys())
            for sid in sorted(sids)
        }

    if img_ptrs is None:
        img_ptrs = gen_img_ptrs(sids)
    if df_metadata is None:
        df_metadata = pd.read_csv(paths["nymph_metadata"])

    pos_lookup = df_metadata.set_index("mask_name")["class_dv"]

    sid_2_samp_idxs = {}
    for sid in sorted(sids):
        samp_idxs = []
        for samp_idx, rfpath in sorted(img_ptrs[sid].items()):
            fname_img = rfpath.split("/")[-1]
            if pos_lookup.get(fname_img) == pos_filter:
                samp_idxs.append(samp_idx)
        sid_2_samp_idxs[sid] = samp_idxs

    return sid_2_samp_idxs

def gen_data_indexes(sids, skeys_partitions):

    img_ptrs = gen_img_ptrs(sids)

    df_metadata = pd.read_csv(paths["nymph_metadata"])
    ordered_series_pos = df_metadata.set_index("mask_name")["class_dv"]
    ordered_series_sex = df_metadata.set_index("mask_name")["sex"]

    data_indexes = {}

    for partition_name in ["train", "id_val", "id_test", "ood_val", "ood_test"]:

        data_index = {
            "sids":    [],
            "rfpaths": [],
        }

        skeys_partition = skeys_partitions[partition_name]

        for skey in sorted(skeys_partition):
            sid = skey[0]
            samp_idx = skey[1]

            data_index["sids"].append(sid)

            rfpath = img_ptrs[sid][samp_idx]
            data_index["rfpaths"].append(rfpath)

        fname_imgs = [rfpath.split("/")[1] for rfpath in data_index["rfpaths"]]
        data_index["pos"] = ordered_series_pos.reindex(fname_imgs).astype(object).where(lambda x: x.notna(), None).tolist()
        data_index["sex"] = ordered_series_sex.reindex(fname_imgs).astype(object).where(lambda x: x.notna(), None).tolist()

        data_indexes[partition_name] = data_index

    return data_indexes

def gen_class_counts_train(data_indexes):

    # note: this all needs to get untangled....
    data_index, _    = assemble_indexes(data_indexes["train"])
    index_class_encs = data_index["class_encs"]

    n_classes          = len(set(index_class_encs))
    class_counts_train = np.bincount(index_class_encs, minlength=n_classes)  # counts[c] is number of samples with class encoding c

    return class_counts_train

def save_split(
    data_indexes, 
    id_eval_nshot, 
    class_counts_train, 
    dpath_split, 
    dpath_figs,
) -> None:
    split = Split(
        data_indexes,
        id_eval_nshot,
        class_counts_train,
    )
    # create dirs (after split has been generated so that dirs aren't created if the run is terminated early)
    os.makedirs(dpath_split, exist_ok=True)
    os.makedirs(dpath_figs, exist_ok=True)
    save_pickle(split, dpath_split / "split.pkl")

def plot_split_distribution(
    data, 
    labels_data, 
    colors, 
    title, 
    x_label, 
    y_label, 
    filepath, 
    ema=False, 
    scale=None, 
    marker="", 
    markersize=6, 
    markeredgewidth=0.5, 
    linestyle="-", 
    alpha=1.0,
) -> None:
    """
    Args:
    - filepath ---------- [str] ----------- Filepath to save plot to
    - ema --------------- [bool] ---------- Whether to apply exponential moving average
    - scale ------------- [None or str] --- None, "log", "symlog"
    """

    def compute_ema(vals, alpha_ema=0.99):
        ema = [vals[0]]
        for i in range(1, len(vals)):
            val_i = vals[i]
            ema_i = alpha_ema * ema[-1] + (1 - alpha_ema) * val_i
            ema.append(ema_i)

        return ema

    x  = range(len(data[0]))

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
            marker=marker, 
            markersize=markersize, 
            markeredgewidth=markeredgewidth, 
            linestyle=linestyle, 
            alpha=alpha,
        )

    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel(x_label, fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")

    if scale == "log":
        plt.yscale("log")
    elif scale == "symlog":
        plt.yscale("symlog", linthresh=1.5)

    plt.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

def gen_ood_distribution_plots(
    genus_2_sids, 
    sids_id, 
    sids_ood_val, 
    sids_ood_test, 
    dpath_figs,
) -> None:

    """
    `genus_tups` structure:

    (genus, n_sids, n_sids_id, n_sids_ood_val, n_sids_ood_test)

    ^ sorted on n_sids and then genus (alphabetical)
    """

    genus_tups = []
    for genus in genus_2_sids.keys():
        n_sids_g = len(genus_2_sids[genus])
        n_sids_g_id, n_sids_g_ood_val, n_sids_g_ood_test = 0, 0, 0
        for sid in genus_2_sids[genus]:
            if sid in sids_id:
                n_sids_g_id += 1
            elif sid in sids_ood_val:
                n_sids_g_ood_val += 1
            elif sid in sids_ood_test:
                n_sids_g_ood_test += 1

        genus_tups.append((genus, n_sids_g, n_sids_g_id, n_sids_g_ood_val, n_sids_g_ood_test))

    genus_tups.sort(key=lambda t: (t[1], t[0]), reverse=True)

    _, n_sids_pg, n_sids_pg_id, n_sids_pg_ood_val, n_sids_pg_ood_test = zip(*genus_tups)  # `_pg` = "per genus"
    n_sids_pg_ood_eval = [n_sids_pg_ood_val[i] + n_sids_pg_ood_test[i] for i in range(len(n_sids_pg))]

    data        = [n_sids_pg, n_sids_pg_id, n_sids_pg_ood_eval]
    colors      = ["crimson", "darkorange", "teal"]
    labels_data = ["Total", "ID-Train/Eval", "OOD-Eval"]
    x_label     = "Sorted Genera"
    y_label     = "Num. Species"

    title    = "OOD Sets Distribution"
    filepath = str(dpath_figs / "distribution_ood.png")
    plot_split_distribution(
        data, labels_data, colors, 
        title, x_label, y_label, 
        filepath,
    )

    title    = "OOD Sets Distribution (Log-Scale)"
    filepath = str(dpath_figs / "distribution_ood_log.png")
    plot_split_distribution(
        data, labels_data, colors, 
        title, x_label, y_label, 
        filepath, 
        ema=False, scale="symlog", 
        marker="|", markersize=6, markeredgewidth=1.0, linestyle="", alpha=1.0,
    )

    title = "OOD Sets Distribution (Log-Scale + Smoothed)"
    filepath = str(dpath_figs / "distribution_ood_log_smooth.png")
    plot_split_distribution(
        data, labels_data, colors, 
        title, x_label, y_label, 
        filepath, 
        ema=True, scale="log",
    )

def gen_id_distribution_plots(
    sids_id_multis, 
    sid_2_skeys_id_multis, 
    n_samps_dict, 
    skeys_partitions, 
    dpath_figs,
) -> None:

    """
    `sid_tups` structure:

    (sid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test)

    ^ sorted on n_skeys and then sid (alphabetical)
    """

    sid_tups = []
    for sid in sids_id_multis:  # ID singletons omitted from plotting (they are all in train)
        n_skeys = n_samps_dict[sid]
        n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
        for skey in sid_2_skeys_id_multis[sid]:
            # check which set the sample landed in
            if skey in skeys_partitions["train"]:
                n_skeys_train += 1
            elif skey in skeys_partitions["id_val"]:
                n_skeys_id_val += 1
            elif skey in skeys_partitions["id_test"]:
                n_skeys_id_test += 1

        sid_tups.append((sid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test))

    sid_tups.sort(key=lambda t: (t[1], t[0]), reverse=True)

    _, n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_val, n_skeys_ps_id_test = zip(*sid_tups)  # `_ps` = "per species"
    n_skeys_ps_id_eval = [n_skeys_ps_id_val[i] + n_skeys_ps_id_test[i] for i in range(len(n_skeys_ps))]

    data        = [n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_eval]
    colors      = ["crimson", "darkorange", "teal"]
    labels_data = ["Total", "Train (ID)", "Eval (ID)"]
    x_label     = "Sorted Species"
    y_label     = "Num. Samples"

    title    = "ID Sets Distribution"
    filepath = str(dpath_figs / "distribution_id.png")
    plot_split_distribution(
        data, labels_data, colors, 
        title, x_label, y_label, 
        filepath,
    )

    title    = "ID Sets Distribution (Log-Scale)"
    filepath = str(dpath_figs / "distribution_id_log.png")
    plot_split_distribution(
        data, labels_data, colors, 
        title, x_label, y_label, 
        filepath, 
        ema=False, scale="symlog", 
        marker="|", markersize=6, markeredgewidth=0.5, linestyle="", alpha=1.0,
    )

    title    = "ID Sets Distribution (Log-Scale + Smoothed)"
    filepath = str(dpath_figs / "distribution_id_log_smooth.png")
    plot_split_distribution(
        data, labels_data, colors, 
        title, x_label, y_label, 
        filepath, 
        ema=True, scale="log",
    )

def gen_split_stats_table(
    sids_id,
    sids_ood_val,
    sids_ood_test,
    skeys_partitions,
    dpath_figs,
    n_sids,
) -> None:

    n_sids_id = len(sids_id)
    n_sids_ood_val = len(sids_ood_val)
    n_sids_ood_test = len(sids_ood_test)

    n_samps_ood_val = len(skeys_partitions["ood_val"])
    n_samps_ood_test = len(skeys_partitions["ood_test"])
    n_samps_total = sum([len(skeys_partitions[pt]) for pt in skeys_partitions.keys()])

    sids_id_val_unrolled, _ = zip(*skeys_partitions["id_val"])
    n_sids_id_val = len(set(sids_id_val_unrolled))

    sids_id_test_unrolled, _ = zip(*skeys_partitions["id_test"])
    n_sids_id_test = len(set(sids_id_test_unrolled))

    n_skeys_train = len(skeys_partitions["train"])
    n_skeys_id_val = len(skeys_partitions["id_val"])
    n_skeys_id_test = len(skeys_partitions["id_test"])

    labels_cols = ["Set", "Num. Species", "Num. Samples"]
    data = [
        ["Train",         f"{n_sids_id:,} ({n_sids_id / n_sids:.2%})",             f"{n_skeys_train:,} ({n_skeys_train / n_samps_total:.2%})"],
        ["ID Val",        f"{n_sids_id_val:,} ({n_sids_id_val / n_sids:.2%})",     f"{n_skeys_id_val:,} ({n_skeys_id_val / n_samps_total:.2%})"],
        ["ID Test",       f"{n_sids_id_test:,} ({n_sids_id_test / n_sids:.2%})",   f"{n_skeys_id_test:,} ({n_skeys_id_test / n_samps_total:.2%})"],
        ["OOD Val",       f"{n_sids_ood_val:,} ({n_sids_ood_val / n_sids:.2%})",   f"{n_samps_ood_val:,} ({n_samps_ood_val / n_samps_total:.2%})"],
        ["OOD Test",      f"{n_sids_ood_test:,} ({n_sids_ood_test / n_sids:.2%})", f"{n_samps_ood_test:,} ({n_samps_ood_test / n_samps_total:.2%})"],
        ["Whole Dataset", f"{n_sids:,} (100.00%)",                                 f"{n_samps_total:,} (100.00%)"],
    ]

    _, ax = plt.subplots(figsize=(5, 2))
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

    plt.title("Split Stats", fontweight="bold", pad=-5)
    plt.savefig(str(dpath_figs / "stats_splits.png"), dpi=150, bbox_inches="tight")

def gen_n_shot_table(
    id_eval_nshot, 
    dpath_figs, 
    col_width=0.20, 
    fontsize_title=8, 
    fontsize=5,
) -> None:

    n_shot_col_names = [f"({name})-shot" for name in id_eval_nshot["names"]]

    # construct cell values
    row_values_id_val  = ["ID Val"]
    row_values_id_test = ["ID Test"]
    for name in id_eval_nshot["names"]:
        bucket_skeys_set_id_val  = id_eval_nshot["buckets"][name]["id_val"]
        bucket_skeys_set_id_test = id_eval_nshot["buckets"][name]["id_test"]

        num_samps_val = len(bucket_skeys_set_id_val)
        sids, _       = zip(*bucket_skeys_set_id_val)
        n_species_val = len(set(sids))
        row_values_id_val.append(f"{num_samps_val:,} ({n_species_val})")

        num_samps_test = len(bucket_skeys_set_id_test)
        sids, _        = zip(*bucket_skeys_set_id_test)
        n_species_test = len(set(sids))
        row_values_id_test.append(f"{num_samps_test:,} ({n_species_test:,})")

    labels_cols = ["Subset"] + n_shot_col_names

    data = [
        row_values_id_val,
        row_values_id_test,
    ]

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")

    col_widths = [col_width] * len(labels_cols)

    tbl = ax.table(
        cellText=data,
        colLabels=labels_cols,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    # tbl.scale(1.5, 1.0)

    for col_idx, _ in enumerate(labels_cols):
        cell = tbl[0, col_idx]
        cell.get_text().set_fontweight("bold")

    # fontsize
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    # thin out grid lines
    for (i, j), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)

    # plt.title("n-shot Bucket Stats: [Num. Samples (Num. Species)]", fontsize=10, fontweight="bold", pad=0)
    plt.title("n-shot Bucket Stats: [Num. Samples (Num. Species)]", fontsize=fontsize_title, fontweight="bold", y=0.70)
    plt.savefig(str(dpath_figs / "stats_nshot.png"), dpi=300, bbox_inches="tight")
