import random
from sklearn.model_selection import train_test_split  # type: ignore[import]
import copy
import matplotlib.pyplot as plt  # type: ignore[import]
import bisect


def strat_split(n_classes, n_draws, pct_sets, n_insts_2_classes, class_2_insts, insts, seed=None):
    """
    Args:
    - n_classes -------------------------------
    - n_draws ---------------------------------
    - pct_sets -------------------------------- percentage for val/test, evenly distributed between both e.g. 10% yields 5% val, 5% test
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

    insts_rem  = copy.deepcopy(insts)
    insts_eval = []

    n_classes_rem = n_classes
    n_draws_rem   = n_draws

    count_min_strat2 = 1 / pct_sets
    i                = 0
    while True:
        i += 1
        classes_i = n_insts_2_classes[i]
        if not classes_i:
            # n_insts_2_classes[i] is empty i.e. no classes at count i
            continue

        n_classes_i   = len(classes_i)
        n_instances_i = n_classes_i * i
        n_draws_i     = round(n_instances_i * pct_sets)

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
        classes_rem      = []
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
            stratify    =classes_rem,
            test_size   =n_draws_rem,
            shuffle     =True,
            random_state=None,
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

    for sid in sids_id:

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
):
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