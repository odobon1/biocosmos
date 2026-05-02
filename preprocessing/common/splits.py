import bisect
import copy
import os
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np  # type: ignore[import]
from sklearn.model_selection import train_test_split  # type: ignore[import]

from utils.data import species_to_genus


def truncate_subspecies(s: str) -> str:
    parts = s.split("_", 2)
    if len(parts) < 3:
        return s
    return parts[0] + "_" + parts[1]

def build_genus_2_cids(cids):
    genus_2_cids = defaultdict(list)
    for cid in cids:
        genus = species_to_genus(cid)
        genus_2_cids[genus].append(cid)
    return genus_2_cids

def build_n_insts_2_classes_g(cids):
    genera = [species_to_genus(cid) for cid in cids]
    count_g = Counter(genera)
    n_insts_2_classes_g = defaultdict(list)
    for genus, count in count_g.items():
        n_insts_2_classes_g[count].append(genus)
    return n_insts_2_classes_g

def strat_split(n_classes, n_draws, pct_eval, n_insts_2_classes, class_2_insts, insts, seed=None):
    rng = random.Random(seed)

    if pct_eval <= 0:
        raise ValueError(f"pct_eval must be > 0, got {pct_eval}")
    if n_classes <= 0:
        raise ValueError(f"n_classes must be > 0, got {n_classes}")

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
    max_count_bucket = max(n_insts_2_classes.keys()) if len(n_insts_2_classes) > 0 else 0
    i = 0
    while True:
        i += 1
        classes_i = list(n_insts_2_classes[i])
        if classes_i:
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

        # Guard against sparse/missing count buckets causing an unbounded loop.
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

    insts_val = set()
    insts_test = set()
    for idx, cid in enumerate(insts_eval):
        if idx % 2 == 0:
            insts_val.add(cid)
        else:
            insts_test.add(cid)

    insts_rem -= insts_val
    insts_rem -= insts_test

    return insts_rem, insts_val, insts_test

def build_class_index_maps(skeys_pool):
    cid_2_skeys_pool = defaultdict(list)
    for cid, samp_idx in sorted(skeys_pool):
        cid_2_skeys_pool[cid].append((cid, samp_idx))

    n_insts_2_classes_pool = defaultdict(list)
    for cid, cid_skeys in cid_2_skeys_pool.items():
        n_insts_2_classes_pool[len(cid_skeys)].append(cid)

    return cid_2_skeys_pool, n_insts_2_classes_pool

def draw_single_partition_from_pool(skeys_pool, n_target, choose_partition, seed):
    if choose_partition not in {"val", "test"}:
        raise ValueError(f"choose_partition must be one of {{'val', 'test'}}, got {choose_partition}")

    if n_target <= 0 or len(skeys_pool) == 0:
        return set(), set(skeys_pool)

    cid_2_skeys_pool, n_insts_2_classes_pool = build_class_index_maps(skeys_pool)
    n_classes_pool = len(cid_2_skeys_pool)
    if n_classes_pool == 0:
        return set(), set(skeys_pool)

    n_draws = min(len(skeys_pool), 2 * n_target)
    if n_draws <= 0:
        return set(), set(skeys_pool)

    pct_eval_pool = n_draws / len(skeys_pool)
    skeys_rem, skeys_val_tmp, skeys_test_tmp = strat_split(
        n_classes=n_classes_pool,
        n_draws=n_draws,
        pct_eval=pct_eval_pool,
        n_insts_2_classes=n_insts_2_classes_pool,
        class_2_insts=cid_2_skeys_pool,
        insts=skeys_pool,
        seed=seed,
    )

    if choose_partition == "test":
        skeys_chosen = skeys_test_tmp
        skeys_restored = skeys_val_tmp
    else:
        skeys_chosen = skeys_val_tmp
        skeys_restored = skeys_test_tmp

    skeys_rem_for_next = set(skeys_rem).union(set(skeys_restored))
    return set(skeys_chosen), skeys_rem_for_next

def sample_id_test_extra_taken(
    cids_id,
    cid_2_samp_idxs,
    n_samps_dict,
    cfg,
    skeys_id_test_extra,
    n_samps_total_target=None,
):
    if skeys_id_test_extra is None:
        skeys_id_test_extra = set()
    else:
        skeys_id_test_extra = set(skeys_id_test_extra)

    n_samps_total = sum(n_samps_dict.values())
    n_samps_target = n_samps_total if n_samps_total_target is None else n_samps_total_target
    n_samps_id_eval = round(n_samps_target * cfg.pct_eval)
    n_samps_id_test_target = n_samps_id_eval // 2

    cids_id_multis = {
        cid for cid in sorted(cids_id)
        if n_samps_dict[cid] > 1
    }
    skeys_id_multis = set()
    for cid in cids_id_multis:
        for samp_idx in cid_2_samp_idxs[cid]:
            skeys_id_multis.add((cid, samp_idx))

    if n_samps_total_target is not None:
        n_samps_id_test_pool_total = len(skeys_id_multis.union(skeys_id_test_extra))
        if n_samps_id_test_target > n_samps_id_test_pool_total:
            raise ValueError(
                f"Requested ID test samples ({n_samps_id_test_target}) exceed available ID test pool samples "
                f"({n_samps_id_test_pool_total})."
            )

    skeys_id_test_pool = skeys_id_multis.union(skeys_id_test_extra)
    skeys_id_test, _ = draw_single_partition_from_pool(
        skeys_pool=skeys_id_test_pool,
        n_target=n_samps_id_test_target,
        choose_partition="test",
        seed=cfg.seed,
    )
    return skeys_id_test.intersection(skeys_id_test_extra)

def build_ood_partitions(
    n_insts_2_classes_g,
    genus_2_cids,
    cids,
    cid_2_samp_idxs,
    n_samps_dict,
    cfg,
    n_samps_total_target=None,
):
    n_cids = len(cids)
    n_samps_total = sum(n_samps_dict.values())
    n_samps_target = n_samps_total if n_samps_total_target is None else n_samps_total_target
    pct_eval_eff = cfg.pct_eval * (n_samps_target / n_samps_total)
    pct_eval_eff = min(max(pct_eval_eff, 0.0), 0.999)

    n_cids_ood_eval = round(n_cids * pct_eval_eff)
    n_genera = len(genus_2_cids)

    max_tries = getattr(cfg, "ood_max_tries", 10_000)
    if max_tries <= 0:
        raise ValueError(f"ood_max_tries must be > 0, got {max_tries}")

    best_split = None
    best_error_score = (float("inf"), float("inf"))
    best_abs_error_val = None
    best_abs_error_test = None

    close_enough = False
    i = 0
    while not close_enough and i < max_tries:
        i += 1
        cids_id_i, cids_ood_val_i, cids_ood_test_i = strat_split(
            n_classes=n_genera,
            n_draws=n_cids_ood_eval,
            pct_eval=pct_eval_eff,
            n_insts_2_classes=n_insts_2_classes_g,
            class_2_insts=genus_2_cids,
            insts=cids,
            seed=cfg.seed + i,
        )

        n_samps_ood_val = sum(n_samps_dict[cid] for cid in cids_ood_val_i)
        n_samps_ood_test = sum(n_samps_dict[cid] for cid in cids_ood_test_i)

        skeys_ood_val_i = set()
        for cid in cids_ood_val_i:
            for samp_idx in cid_2_samp_idxs[cid]:
                skeys_ood_val_i.add((cid, samp_idx))

        skeys_id_test_extra_taken_i = sample_id_test_extra_taken(
            cids_id=cids_id_i,
            cid_2_samp_idxs=cid_2_samp_idxs,
            n_samps_dict=n_samps_dict,
            cfg=cfg,
            skeys_id_test_extra=skeys_ood_val_i,
            n_samps_total_target=n_samps_total_target,
        )
        n_samps_ood_val_after_id_test = len(skeys_ood_val_i - skeys_id_test_extra_taken_i)

        pct_samps_ood_val = n_samps_ood_val_after_id_test / n_samps_target
        pct_samps_ood_test = n_samps_ood_test / n_samps_target

        abs_error_val = abs((cfg.pct_partition) - pct_samps_ood_val)
        abs_error_test = abs((cfg.pct_partition) - pct_samps_ood_test)

        # Minimize the worst partition error first, then sum as tie-breaker.
        error_score = (max(abs_error_val, abs_error_test), abs_error_val + abs_error_test)
        if error_score < best_error_score:
            best_error_score = error_score
            best_split = (cids_id_i, cids_ood_val_i, cids_ood_test_i)
            best_abs_error_val = abs_error_val
            best_abs_error_test = abs_error_test

        close_enough = (
            abs((cfg.pct_partition) - pct_samps_ood_val) < cfg.pct_ood_tol
            and abs((cfg.pct_partition) - pct_samps_ood_test) < cfg.pct_ood_tol
        )

    if best_split is None:
        raise RuntimeError("Failed to construct any OOD split candidate.")

    if not close_enough:
        print(
            "Warning: OOD split tolerance was not met after "
            f"{max_tries} attempts. Using best candidate with "
            f"abs_error_val={best_abs_error_val:.6f}, abs_error_test={best_abs_error_test:.6f}, "
            f"target={cfg.pct_partition:.6f}, tol={cfg.pct_ood_tol:.6f}."
        )

    cids_id, cids_ood_val, cids_ood_test = best_split

    skeys_ood_val = set()
    for cid in cids_ood_val:
        for samp_idx in cid_2_samp_idxs[cid]:
            skeys_ood_val.add((cid, samp_idx))

    skeys_ood_test = set()
    for cid in cids_ood_test:
        for samp_idx in cid_2_samp_idxs[cid]:
            skeys_ood_test.add((cid, samp_idx))

    return cids_id, cids_ood_val, cids_ood_test, skeys_ood_val, skeys_ood_test

def build_id_partitions(
    cids_id,
    cid_2_samp_idxs,
    n_samps_dict,
    cfg,
    n_samps_total_target=None,
    skeys_id_test_extra=None,
):
    n_samps_total = sum(n_samps_dict.values())
    n_samps_target = n_samps_total if n_samps_total_target is None else n_samps_total_target
    n_samps_id_eval = round(n_samps_target * cfg.pct_eval)
    n_samps_id_test_target = n_samps_id_eval // 2
    n_samps_id_val_target = n_samps_id_eval - n_samps_id_test_target

    if skeys_id_test_extra is None:
        skeys_id_test_extra = set()
    else:
        skeys_id_test_extra = set(skeys_id_test_extra)

    cids_id_singles = set()
    for cid in sorted(cids_id):
        if n_samps_dict[cid] == 1:
            cids_id_singles.add(cid)

    cids_id_multis = cids_id - cids_id_singles
    n_cids_id_multis = len(cids_id_multis)

    cid_2_skeys_id_multis = defaultdict(list)
    cid_2_skeys_id = defaultdict(list)
    skeys_id_multis = set()

    for cid in sorted(cids_id):
        for samp_idx in cid_2_samp_idxs[cid]:
            skey = (cid, samp_idx)
            cid_2_skeys_id[cid].append(skey)
            if cid in cids_id_multis:
                cid_2_skeys_id_multis[cid].append(skey)
                skeys_id_multis.add(skey)

    if n_samps_total_target is not None:
        n_samps_id_test_pool_total = len(skeys_id_multis.union(skeys_id_test_extra))
        if n_samps_id_test_target > n_samps_id_test_pool_total:
            raise ValueError(
                f"Requested ID test samples ({n_samps_id_test_target}) exceed available ID test pool samples "
                f"({n_samps_id_test_pool_total})."
            )

    skeys_id_test_pool = skeys_id_multis.union(skeys_id_test_extra)
    skeys_id_test, skeys_id_test_pool_rem = draw_single_partition_from_pool(
        skeys_pool=skeys_id_test_pool,
        n_target=n_samps_id_test_target,
        choose_partition="test",
        seed=cfg.seed,
    )

    skeys_id_test_extra_taken = skeys_id_test.intersection(skeys_id_test_extra)

    skeys_id_multis_rem = skeys_id_test_pool_rem.intersection(skeys_id_multis)
    skeys_id_val, skeys_train_multis = draw_single_partition_from_pool(
        skeys_pool=skeys_id_multis_rem,
        n_target=n_samps_id_val_target,
        choose_partition="val",
        seed=cfg.seed + 1,
    )

    for cid in sorted(cids_id_multis):
        cid_skeys = set(cid_2_skeys_id_multis[cid])
        if len(cid_skeys.intersection(skeys_train_multis)) > 0:
            continue

        cid_skeys_val = sorted(cid_skeys.intersection(skeys_id_val))
        cid_skeys_test = sorted(cid_skeys.intersection(skeys_id_test))

        donor_partition = "val"
        if len(skeys_id_test) > len(skeys_id_val):
            donor_partition = "test"

        skey_move = None
        if donor_partition == "val" and cid_skeys_val:
            skey_move = cid_skeys_val[0]
            skeys_id_val.remove(skey_move)
        elif donor_partition == "test" and cid_skeys_test:
            skey_move = cid_skeys_test[0]
            skeys_id_test.remove(skey_move)
        elif cid_skeys_val:
            skey_move = cid_skeys_val[0]
            skeys_id_val.remove(skey_move)
        elif cid_skeys_test:
            skey_move = cid_skeys_test[0]
            skeys_id_test.remove(skey_move)

        if skey_move is not None:
            skeys_train_multis.add(skey_move)

    skeys_id_singles = set((cid, cid_2_samp_idxs[cid][0]) for cid in cids_id_singles)
    skeys_train = skeys_train_multis.union(skeys_id_singles)

    return (
        skeys_train,
        skeys_id_val,
        skeys_id_test,
        cid_2_skeys_id,
        cid_2_skeys_id_multis,
        cids_id_multis,
        skeys_id_test_extra_taken,
    )

def build_trainval_skeys_partition(skeys_partitions):
    skeys_trainval = set().union(
        skeys_partitions["train"],
        skeys_partitions["id_val"],
        skeys_partitions["ood_val"],
    )
    skeys_id_test = set(skeys_partitions.get("id_test", set()))
    return skeys_trainval - skeys_id_test

def build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id):
    def find_range_index(nst_seps, n):
        assert isinstance(n, int), f"n must be an int, got {type(n).__name__}"
        if n <= 0:
            raise ValueError(f"find_range_index(): n = {n}")
        return bisect.bisect_left(nst_seps, n)

    n_shot_tracker = []
    for _ in range(len(cfg.nst_names)):
        n_shot_tracker.append({"id_val": set(), "trainval": set(), "id_test": set()})

    for cid in sorted(cids_id):
        cid_skeys_val = set()
        cid_skeys_trainval = set()
        cid_skeys_test = set()

        n_skeys_train = 0
        n_skeys_trainval = 0
        for skey in cid_2_skeys_id[cid]:
            if skey in skeys_partitions["train"]:
                n_skeys_train += 1
                n_skeys_trainval += 1
                cid_skeys_trainval.add(skey)
            elif skey in skeys_partitions["id_val"]:
                n_skeys_trainval += 1
                cid_skeys_val.add(skey)
                cid_skeys_trainval.add(skey)
            elif skey in skeys_partitions["id_test"]:
                cid_skeys_test.add(skey)

        idx_id_val_bucket = find_range_index(cfg.nst_seps, n_skeys_train)
        n_shot_tracker[idx_id_val_bucket]["id_val"].update(cid_skeys_val)

        idx_trainval_bucket = find_range_index(cfg.nst_seps, n_skeys_trainval)
        n_shot_tracker[idx_trainval_bucket]["trainval"].update(cid_skeys_trainval)
        n_shot_tracker[idx_trainval_bucket]["id_test"].update(cid_skeys_test)

    # Second pass: OOD-val species whose samples were borrowed into id_test.
    # These species are not in cids_id, so they were missed above.
    # We bucket their id_test samples using their trainval cardinality.
    cid_2_id_test_skeys_ood = {}
    for skey in skeys_partitions["id_test"]:
        cid = skey[0]
        if cid not in cids_id:
            if cid not in cid_2_id_test_skeys_ood:
                cid_2_id_test_skeys_ood[cid] = set()
            cid_2_id_test_skeys_ood[cid].add(skey)

    if cid_2_id_test_skeys_ood:
        cid_2_n_trainval_ood = {}
        for skey in skeys_partitions["trainval"]:
            cid = skey[0]
            if cid in cid_2_id_test_skeys_ood:
                cid_2_n_trainval_ood[cid] = cid_2_n_trainval_ood.get(cid, 0) + 1

        for cid in sorted(cid_2_id_test_skeys_ood):
            n_skeys_trainval_ood = cid_2_n_trainval_ood.get(cid, 0)
            if n_skeys_trainval_ood == 0:
                continue
            idx_bucket = find_range_index(cfg.nst_seps, n_skeys_trainval_ood)
            n_shot_tracker[idx_bucket]["id_test"].update(cid_2_id_test_skeys_ood[cid])

    id_eval_nshot = {
        "names": cfg.nst_names,
        "buckets": {
            name: {
                "id_val": bucket["id_val"],
                "trainval": bucket["trainval"],
                "id_test": bucket["id_test"],
            }
            for name, bucket in zip(cfg.nst_names, n_shot_tracker)
        },
    }

    return id_eval_nshot

def build_class_counts_train(data_indexes):
    cid2enc = {}
    index_encs = []
    for cid in [datum["cid"] for datum in data_indexes["train"]]:
        if cid not in cid2enc:
            cid2enc[cid] = len(cid2enc)
        index_encs.append(cid2enc[cid])
    n_classes = len(cid2enc)
    class_counts_train = np.bincount(index_encs, minlength=n_classes)
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

def generate_ood_distribution_plots(genus_2_cids, cids_id, cids_ood_val, cids_ood_test, dpath_figs) -> None:
    genus_tups = []
    for genus in genus_2_cids.keys():
        n_cids_g = len(genus_2_cids[genus])
        n_cids_g_id, n_cids_g_ood_val, n_cids_g_ood_test = 0, 0, 0
        for cid in genus_2_cids[genus]:
            if cid in cids_id:
                n_cids_g_id += 1
            elif cid in cids_ood_val:
                n_cids_g_ood_val += 1
            elif cid in cids_ood_test:
                n_cids_g_ood_test += 1

        genus_tups.append((genus, n_cids_g, n_cids_g_id, n_cids_g_ood_val, n_cids_g_ood_test))

    genus_tups.sort(key=lambda t: (t[1], t[0]), reverse=True)
    _, n_cids_pg, n_cids_pg_id, n_cids_pg_ood_val, n_cids_pg_ood_test = zip(*genus_tups)
    n_cids_pg_ood_eval = [n_cids_pg_ood_val[i] + n_cids_pg_ood_test[i] for i in range(len(n_cids_pg))]

    data = [n_cids_pg, n_cids_pg_id, n_cids_pg_ood_eval]
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

def generate_id_distribution_plots(cids_id_multis, cid_2_skeys_id_multis, n_samps_dict, skeys_partitions, dpath_figs) -> None:
    cid_tups = []
    for cid in cids_id_multis:
        n_skeys = n_samps_dict[cid]
        n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
        for skey in cid_2_skeys_id_multis[cid]:
            if skey in skeys_partitions["train"]:
                n_skeys_train += 1
            elif skey in skeys_partitions["id_val"]:
                n_skeys_id_val += 1
            elif skey in skeys_partitions["id_test"]:
                n_skeys_id_test += 1

        cid_tups.append((cid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test))

    cid_tups.sort(key=lambda t: (t[1], t[0]), reverse=True)
    _, n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_val, n_skeys_ps_id_test = zip(*cid_tups)
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
    n_shot_col_names = [name for name in id_eval_nshot["names"]]

    row_values_id_val = ["ID Val"]
    row_values_trainval = ["TrainVal"]
    row_values_id_test = ["ID Test"]
    for name in id_eval_nshot["names"]:
        bucket_skeys_set_id_val = id_eval_nshot["buckets"][name]["id_val"]
        bucket_skeys_set_trainval = id_eval_nshot["buckets"][name]["trainval"]
        bucket_skeys_set_id_test = id_eval_nshot["buckets"][name]["id_test"]

        num_samps_val = len(bucket_skeys_set_id_val)
        if num_samps_val > 0:
            cids, _ = zip(*bucket_skeys_set_id_val)
            n_species_val = len(set(cids))
        else:
            n_species_val = 0
        row_values_id_val.append(f"{num_samps_val:,} ({n_species_val})")

        num_samps_trainval = len(bucket_skeys_set_trainval)
        if num_samps_trainval > 0:
            cids, _ = zip(*bucket_skeys_set_trainval)
            n_species_trainval = len(set(cids))
        else:
            n_species_trainval = 0
        row_values_trainval.append(f"{num_samps_trainval:,} ({n_species_trainval})")

        num_samps_test = len(bucket_skeys_set_id_test)
        if num_samps_test > 0:
            cids, _ = zip(*bucket_skeys_set_id_test)
            n_species_test = len(set(cids))
        else:
            n_species_test = 0
        row_values_id_test.append(f"{num_samps_test:,} ({n_species_test:,})")

    labels_cols = ["Subset"] + n_shot_col_names
    data = [row_values_id_val, row_values_trainval, row_values_id_test]

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

    plt.title("n-shot Bucket Stats: [Num. Samples (Num. Classes)]", fontsize=fontsize_title, fontweight="bold", y=0.70)
    plt.savefig(str(dpath_figs / "stats_nshot.png"), dpi=300, bbox_inches="tight")

def count_unique_cids_from_skeys(skeys) -> int:
    return len({cid for cid, _ in skeys})

def count_total_samples_disjoint_partitions(skeys_partitions) -> int:
    return len(
        skeys_partitions["train"]
        | skeys_partitions["id_val"]
        | skeys_partitions["id_test"]
        | skeys_partitions["ood_val"]
        | skeys_partitions["ood_test"]
    )

def render_stats_table(
    labels_cols,
    data,
    title,
    filepath,
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
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()

def generate_basic_split_stats_table(
    skeys_partitions,
    dpath_figs,
    n_cids_total,
    title,
    labels_cols=None,
    first_col_name="Set",
    ood_val_label="OOD Val",
    ood_test_label="OOD Test",
    figsize=(5, 2),
    pad=-5,
    dpi=150,
    fontsize=None,
    scale=None,
) -> None:
    if labels_cols is None:
        labels_cols = [first_col_name, "Num. Classes", "Num. Samples"]

    n_samps_total = count_total_samples_disjoint_partitions(skeys_partitions)
    row_specs = [
        ("Train", "train"),
        ("TrainVal", "trainval"),
        ("ID Val", "id_val"),
        ("ID Test", "id_test"),
        (ood_val_label, "ood_val"),
        (ood_test_label, "ood_test"),
    ]

    data = []
    for row_name, partition_name in row_specs:
        skeys_partition = skeys_partitions[partition_name]
        n_cids_partition = count_unique_cids_from_skeys(skeys_partition)
        n_samps_partition = len(skeys_partition)
        data.append([
            row_name,
            f"{n_cids_partition:,} ({n_cids_partition / n_cids_total:.2%})",
            f"{n_samps_partition:,} ({n_samps_partition / n_samps_total:.2%})",
        ])

    data.append([
        "Whole Dataset",
        f"{n_cids_total:,} (100.00%)",
        f"{n_samps_total:,} (100.00%)",
    ])

    render_stats_table(
        labels_cols=labels_cols,
        data=data,
        title=title,
        filepath=str(dpath_figs / "stats_splits.png"),
        figsize=figsize,
        pad=pad,
        dpi=dpi,
        fontsize=fontsize,
        scale=scale,
    )