import bisect
import copy
import os
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np  # type: ignore[import]
from sklearn.model_selection import train_test_split  # type: ignore[import]


def sid_to_genus(sid: str) -> str:
    return sid.split("_")[0]

def truncate_subspecies(s: str) -> str:
    parts = s.split("_", 2)
    if len(parts) < 3:
        return s
    return parts[0] + "_" + parts[1]

def build_genus_2_sids(sids):
    genus_2_sids = defaultdict(list)
    for sid in sids:
        genus = sid_to_genus(sid)
        genus_2_sids[genus].append(sid)
    return genus_2_sids

def build_n_insts_2_classes_g(sids):
    genera = [sid_to_genus(sid) for sid in sids]
    count_g = Counter(genera)
    n_insts_2_classes_g = defaultdict(list)
    for genus, count in count_g.items():
        n_insts_2_classes_g[count].append(genus)
    return n_insts_2_classes_g

def strat_split(n_classes, n_draws, pct_eval, n_insts_2_classes, class_2_insts, insts, seed=None):
    rng = random.Random(seed)

    def compute_class_hits(n_draws, n_classes):
        class_hits = [n_draws // n_classes] * n_classes
        plus_ones = n_draws % n_classes
        for i in range(plus_ones):
            class_hits[i] += 1
        return class_hits

    insts_rem = copy.deepcopy(insts)
    insts_eval = []

    n_classes_rem = n_classes
    n_draws_rem = n_draws

    count_min_strat2 = 1 / pct_eval
    i = 0
    while True:
        i += 1
        classes_i = list(n_insts_2_classes[i])
        if not classes_i:
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

    insts_val = set()
    insts_test = set()
    for idx, sid in enumerate(insts_eval):
        if idx % 2 == 0:
            insts_val.add(sid)
        else:
            insts_test.add(sid)

    insts_rem -= insts_val
    insts_rem -= insts_test

    return insts_rem, insts_val, insts_test

def build_ood_partitions(
    n_insts_2_classes_g,
    genus_2_sids,
    sids,
    sid_2_samp_idxs,
    n_samps_dict,
    cfg,
    n_samps_total_target=None,
):
    n_sids = len(sids)
    n_samps_total = sum(n_samps_dict.values())
    n_samps_target = n_samps_total if n_samps_total_target is None else n_samps_total_target
    pct_eval_eff = cfg.pct_eval * (n_samps_target / n_samps_total)
    pct_eval_eff = min(max(pct_eval_eff, 0.0), 0.999)

    n_sids_ood_eval = round(n_sids * pct_eval_eff)
    n_genera = len(genus_2_sids)

    close_enough = False
    i = 0
    while not close_enough:
        i += 1
        sids_id, sids_ood_val, sids_ood_test = strat_split(
            n_classes=n_genera,
            n_draws=n_sids_ood_eval,
            pct_eval=pct_eval_eff,
            n_insts_2_classes=n_insts_2_classes_g,
            class_2_insts=genus_2_sids,
            insts=sids,
            seed=cfg.seed + i,
        )

        n_samps_ood_val = sum(n_samps_dict[sid] for sid in sids_ood_val)
        n_samps_ood_test = sum(n_samps_dict[sid] for sid in sids_ood_test)

        pct_samps_ood_val = n_samps_ood_val / n_samps_target
        pct_samps_ood_test = n_samps_ood_test / n_samps_target

        close_enough = (
            abs((cfg.pct_eval / 2) - pct_samps_ood_val) < cfg.pct_ood_tol
            and abs((cfg.pct_eval / 2) - pct_samps_ood_test) < cfg.pct_ood_tol
        )

    skeys_ood_val = set()
    for sid in sids_ood_val:
        for samp_idx in sid_2_samp_idxs[sid]:
            skeys_ood_val.add((sid, samp_idx))

    skeys_ood_test = set()
    for sid in sids_ood_test:
        for samp_idx in sid_2_samp_idxs[sid]:
            skeys_ood_test.add((sid, samp_idx))

    return sids_id, sids_ood_val, sids_ood_test, skeys_ood_val, skeys_ood_test

def build_id_partitions(
    sids_id,
    sid_2_samp_idxs,
    n_samps_dict,
    cfg,
    n_samps_total_target=None,
):
    n_samps_total = sum(n_samps_dict.values())
    n_samps_target = n_samps_total if n_samps_total_target is None else n_samps_total_target
    n_samps_id_eval = round(n_samps_target * cfg.pct_eval)

    sids_id_singles = set()
    for sid in sorted(sids_id):
        if n_samps_dict[sid] == 1:
            sids_id_singles.add(sid)

    sids_id_multis = sids_id - sids_id_singles
    n_sids_id_multis = len(sids_id_multis)

    n_insts_2_classes_s = defaultdict(list)
    for sid in sorted(sids_id_multis):
        count = n_samps_dict[sid]
        n_insts_2_classes_s[count].append(sid)

    pct_rem_id_eval = cfg.pct_eval / (1 - cfg.pct_eval)

    sid_2_skeys_id_multis = defaultdict(list)
    sid_2_skeys_id = defaultdict(list)
    skeys_id_multis = set()

    for sid in sorted(sids_id):
        for samp_idx in sid_2_samp_idxs[sid]:
            skey = (sid, samp_idx)
            sid_2_skeys_id[sid].append(skey)
            if sid in sids_id_multis:
                sid_2_skeys_id_multis[sid].append(skey)
                skeys_id_multis.add(skey)

    if n_samps_total_target is not None:
        n_samps_id_multis_total = len(skeys_id_multis)
        if n_samps_id_eval > n_samps_id_multis_total:
            raise ValueError(
                f"Requested ID eval samples ({n_samps_id_eval}) exceed available ID multis samples "
                f"({n_samps_id_multis_total})."
            )
        pct_rem_id_eval = n_samps_id_eval / n_samps_id_multis_total if n_samps_id_multis_total > 0 else 0.0

    skeys_train_multis, skeys_id_val, skeys_id_test = strat_split(
        n_classes=n_sids_id_multis,
        n_draws=n_samps_id_eval,
        pct_eval=pct_rem_id_eval,
        n_insts_2_classes=n_insts_2_classes_s,
        class_2_insts=sid_2_skeys_id_multis,
        insts=skeys_id_multis,
        seed=cfg.seed,
    )

    for sid in sorted(sids_id_multis):
        sid_skeys = set(sid_2_skeys_id_multis[sid])
        if len(sid_skeys.intersection(skeys_train_multis)) > 0:
            continue

        sid_skeys_val = sorted(sid_skeys.intersection(skeys_id_val))
        sid_skeys_test = sorted(sid_skeys.intersection(skeys_id_test))

        donor_partition = "val"
        if len(skeys_id_test) > len(skeys_id_val):
            donor_partition = "test"

        skey_move = None
        if donor_partition == "val" and sid_skeys_val:
            skey_move = sid_skeys_val[0]
            skeys_id_val.remove(skey_move)
        elif donor_partition == "test" and sid_skeys_test:
            skey_move = sid_skeys_test[0]
            skeys_id_test.remove(skey_move)
        elif sid_skeys_val:
            skey_move = sid_skeys_val[0]
            skeys_id_val.remove(skey_move)
        elif sid_skeys_test:
            skey_move = sid_skeys_test[0]
            skeys_id_test.remove(skey_move)

        if skey_move is not None:
            skeys_train_multis.add(skey_move)

    skeys_id_singles = set((sid, sid_2_samp_idxs[sid][0]) for sid in sids_id_singles)
    skeys_train = skeys_train_multis.union(skeys_id_singles)

    return skeys_train, skeys_id_val, skeys_id_test, sid_2_skeys_id, sid_2_skeys_id_multis, sids_id_multis

def build_id_eval_nshot(cfg, sids_id, skeys_partitions, sid_2_skeys_id):
    def find_range_index(nst_seps, n):
        assert isinstance(n, int), f"n must be an int, got {type(n).__name__}"
        if n <= 0:
            raise ValueError(f"find_range_index(): n = {n}")
        return bisect.bisect_left(nst_seps, n)

    n_shot_tracker = []
    for _ in range(len(cfg.nst_names)):
        n_shot_tracker.append({"train": set(), "id_val": set(), "id_test": set()})

    for sid in sorted(sids_id):
        sid_skeys_train = set()
        sid_skeys_val = set()
        sid_skeys_test = set()

        n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
        for skey in sid_2_skeys_id[sid]:
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

def build_class_counts_train(data_indexes):
    sid_2_class_enc = {}
    index_class_encs = []
    for sid in data_indexes["train"]["sids"]:
        if sid not in sid_2_class_enc:
            sid_2_class_enc[sid] = len(sid_2_class_enc)
        index_class_encs.append(sid_2_class_enc[sid])

    n_classes = len(sid_2_class_enc)
    class_counts_train = np.bincount(index_class_encs, minlength=n_classes)
    return class_counts_train

def build_dev_skeys_partitions(skeys_partitions, size_dev):
    if "train" not in skeys_partitions:
        raise KeyError("skeys_partitions must contain a 'train' partition")
    if size_dev <= 0:
        raise ValueError(f"size_dev must be greater than 0, got {size_dev}")

    skeys_train_sorted = sorted(skeys_partitions["train"])
    skeys_dev = set(skeys_train_sorted[:size_dev])

    return {
        partition_name: set(skeys_dev)
        for partition_name in skeys_partitions
    }

def save_split(data_indexes, id_eval_nshot, class_counts_train, dpath_split, dpath_figs) -> None:
    from utils.data import Split
    from utils.utils import save_pickle

    split = Split(data_indexes, id_eval_nshot, class_counts_train)
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
    def compute_ema(vals, alpha_ema=0.99):
        ema_vals = [vals[0]]
        for i in range(1, len(vals)):
            val_i = vals[i]
            ema_i = alpha_ema * ema_vals[-1] + (1 - alpha_ema) * val_i
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

def generate_ood_distribution_plots(genus_2_sids, sids_id, sids_ood_val, sids_ood_test, dpath_figs) -> None:
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
    _, n_sids_pg, n_sids_pg_id, n_sids_pg_ood_val, n_sids_pg_ood_test = zip(*genus_tups)
    n_sids_pg_ood_eval = [n_sids_pg_ood_val[i] + n_sids_pg_ood_test[i] for i in range(len(n_sids_pg))]

    data = [n_sids_pg, n_sids_pg_id, n_sids_pg_ood_eval]
    colors = ["crimson", "darkorange", "teal"]
    labels_data = ["Total", "ID-Train/Eval", "OOD-Eval"]
    x_label = "Sorted Genera"
    y_label = "Num. Species"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Sets Distribution",
        x_label=x_label,
        y_label=y_label,
        filepath=str(dpath_figs / "distribution_ood.png"),
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Sets Distribution (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        filepath=str(dpath_figs / "distribution_ood_log.png"),
        ema=False,
        scale="symlog",
        marker="|",
        markersize=6,
        markeredgewidth=1.0,
        linestyle="",
        alpha=1.0,
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Sets Distribution (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        filepath=str(dpath_figs / "distribution_ood_log_smooth.png"),
        ema=True,
        scale="log",
    )

def generate_id_distribution_plots(sids_id_multis, sid_2_skeys_id_multis, n_samps_dict, skeys_partitions, dpath_figs) -> None:
    sid_tups = []
    for sid in sids_id_multis:
        n_skeys = n_samps_dict[sid]
        n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
        for skey in sid_2_skeys_id_multis[sid]:
            if skey in skeys_partitions["train"]:
                n_skeys_train += 1
            elif skey in skeys_partitions["id_val"]:
                n_skeys_id_val += 1
            elif skey in skeys_partitions["id_test"]:
                n_skeys_id_test += 1

        sid_tups.append((sid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test))

    sid_tups.sort(key=lambda t: (t[1], t[0]), reverse=True)
    _, n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_val, n_skeys_ps_id_test = zip(*sid_tups)
    n_skeys_ps_id_eval = [n_skeys_ps_id_val[i] + n_skeys_ps_id_test[i] for i in range(len(n_skeys_ps))]

    data = [n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_eval]
    colors = ["crimson", "darkorange", "teal"]
    labels_data = ["Total", "Train (ID)", "Eval (ID)"]
    x_label = "Sorted Species"
    y_label = "Num. Samples"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Sets Distribution",
        x_label=x_label,
        y_label=y_label,
        filepath=str(dpath_figs / "distribution_id.png"),
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Sets Distribution (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        filepath=str(dpath_figs / "distribution_id_log.png"),
        ema=False,
        scale="symlog",
        marker="|",
        markersize=6,
        markeredgewidth=0.5,
        linestyle="",
        alpha=1.0,
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Sets Distribution (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        filepath=str(dpath_figs / "distribution_id_log_smooth.png"),
        ema=True,
        scale="log",
    )

def generate_n_shot_table(id_eval_nshot, dpath_figs, col_width=0.20, fontsize_title=8, fontsize=5) -> None:
    n_shot_col_names = [f"({name})-shot" for name in id_eval_nshot["names"]]

    row_values_id_val = ["ID Val"]
    row_values_id_test = ["ID Test"]
    for name in id_eval_nshot["names"]:
        bucket_skeys_set_id_val = id_eval_nshot["buckets"][name]["id_val"]
        bucket_skeys_set_id_test = id_eval_nshot["buckets"][name]["id_test"]

        num_samps_val = len(bucket_skeys_set_id_val)
        if num_samps_val > 0:
            sids, _ = zip(*bucket_skeys_set_id_val)
            n_species_val = len(set(sids))
        else:
            n_species_val = 0
        row_values_id_val.append(f"{num_samps_val:,} ({n_species_val})")

        num_samps_test = len(bucket_skeys_set_id_test)
        if num_samps_test > 0:
            sids, _ = zip(*bucket_skeys_set_id_test)
            n_species_test = len(set(sids))
        else:
            n_species_test = 0
        row_values_id_test.append(f"{num_samps_test:,} ({n_species_test:,})")

    labels_cols = ["Subset"] + n_shot_col_names
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

    plt.title("n-shot Bucket Stats: [Num. Samples (Num. Species)]", fontsize=fontsize_title, fontweight="bold", y=0.70)
    plt.savefig(str(dpath_figs / "stats_nshot.png"), dpi=300, bbox_inches="tight")
