"""
Want to be able to run this one independently to generate more splits on the fly (i.e. not deeply integrated into the entire ingestion pipeline)
"""

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os
import bisect

from utils import paths, read_pickle, write_pickle
from utils_ingestion import strat_splits

import pdb


""" CONFIG PARAMS """

PCT_EVAL             = 0.05
PCT_OOD_CLOSE_ENOUGH = 0.001  # careful, this parameter dictates required accuracy as the stopping criterion for a random search, too low is no no (0.0001 is good)
SPLIT_NAME           = "dev"
ALLOW_OVERWRITES     = False
# NST_NAMES            = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51-100", "101+"]
# NST_UPPER_BOUNDS     = [1, 2, 3, 4, 5, 10, 20, 50, 100]  # divides n-shot space into len(NST_UPPER_BOUNDS) + 1 buckets, last bucket is if n is greater than the last upper bound
NST_NAMES            = ["1-19", "20-99", "100-499", "500+"]
NST_UPPER_BOUNDS     = [19, 99, 499]

assert len(NST_NAMES) == len(NST_UPPER_BOUNDS) + 1, f"len(NST_NAMES) ({len(NST_NAMES)}) != len(NST_UPPER_BOUNDS) + 1 ({len(NST_UPPER_BOUNDS)})"


dpath_splits = paths["metadata_o"] / f"splits/{SPLIT_NAME}"
dpath_figs   = dpath_splits / "figures"

if os.path.isdir(dpath_splits) and not ALLOW_OVERWRITES:
    error_msg = f"Split '{SPLIT_NAME}' already exists, choose a different SPLIT_NAME!"
    raise ValueError(error_msg)
print(F"SPLIT_NAME: '{SPLIT_NAME}'")

pct_ood_eval = pct_id_eval = PCT_EVAL / 2  # OOD splits: pct_splits

tax_nymph       = read_pickle(paths["metadata_o"] / "tax/nymph.pkl")
sids            = set(tax_nymph["found"].keys())  # OOD splits: insts
n_sids          = len(sids)
n_sids_ood_eval = round(n_sids * pct_ood_eval)

n_samps_dict  = {}
n_samps_total = 0
for sid in sids:
    n_samps_sid       = tax_nymph["found"][sid]["meta"]["num_imgs"]
    n_samps_dict[sid] = n_samps_sid
    n_samps_total += n_samps_sid

n_samps_eval = round(n_samps_total * PCT_EVAL)  # OOD splits: n_draws

genera       = []
genus_2_sids = defaultdict(list)  # OOD splits: class_2_insts
for sid in sids:
    genus = tax_nymph["found"][sid]["tax"]["genus"]
    genera.append(genus)
    genus_2_sids[genus].append(sid)

n_genera = len(set(genera))  # OOD splits: n_classes

"""
`genus_2_sids` & `sid_2_skeys_id_multis` structure (class_2_insts):

genus_2_sids:
{
    genus0 : [sid0, sid1, sid2, ...],
    genus1 : [...],
    ...
}

sid_2_skeys_id_multis:
{
    sid0 : [skey0, skey1, skey2, ...],
    sid1 : [...],
    ...
}
"""

count_g             = Counter(genera)
n_insts_2_classes_g = defaultdict(list)  # n_insts_2_classes OOD splits
for genus, count in count_g.items():
    n_insts_2_classes_g[count].append(genus)

"""
`n_insts_2_classes_*` structure:

n_insts_2_classes_g (OOD):
{
    1 : [genus0, genus1, genus2, ...],
    2 : [...],
    4 : [...],
    ...
}

n_insts_2_classes_s (ID):
{
    1 : [sid0, sid1, sid2, ...],
    2 : [...],
    ...
}
"""

# OOD SPLIT

print("Constructing OOD Split...")

close_enough = False
while not close_enough:

    sids_id, sids_ood_val, sids_ood_test = strat_splits(
        n_classes=n_genera, 
        n_draws=n_sids_ood_eval, 
        pct_splits=pct_ood_eval, 
        n_insts_2_classes=n_insts_2_classes_g, 
        class_2_insts=genus_2_sids, 
        insts=sids,
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

    if abs((pct_ood_eval / 2) - pct_samps_ood_val) < PCT_OOD_CLOSE_ENOUGH and abs((pct_ood_eval / 2) - pct_samps_ood_test) < PCT_OOD_CLOSE_ENOUGH:
        close_enough = True

skeys_ood_val = set()
for sid in sids_ood_val:
    for sidx in range(n_samps_dict[sid]):
        skey = (sid, sidx)
        skeys_ood_val.add(skey)

skeys_ood_test = set()
for sid in sids_ood_test:
    for sidx in range(n_samps_dict[sid]):
        skey = (sid, sidx)
        skeys_ood_test.add(skey)

print("OOD Split Complete!")

# ID SPLIT

print("Constructing ID Split...")

sids_id_singles = set() # species id's with 1 sample i.e. singletons
for sid in sids_id:
    if n_samps_dict[sid] == 1:
        sids_id_singles.add(sid)
        
sids_id_multis = sids_id - sids_id_singles  # species id's with 2+ samples

n_sids_id_multis = len(sids_id_multis)  # ID splits: n_classes
n_samps_id_eval  = n_samps_eval - (n_samps_ood_val + n_samps_ood_test)  # ID splits: n_draws

n_insts_2_classes_s = defaultdict(list)  # ID splits: n_insts_2_classes
for sid in sids_id_multis:
    count = n_samps_dict[sid]
    n_insts_2_classes_s[count].append(sid)

pct_rem_id_eval = pct_id_eval / (1 - pct_id_eval)  # ID splits: pct_splits (10 / 90)

sid_2_skeys_id_multis = defaultdict(list)  # ID splits: class_2_insts
sid_2_skeys_id        = defaultdict(list)  # used for n-shot tracking
skeys_id_multis       = set()  # ID splits: insts

for sid in sids_id:
    for sidx in range(n_samps_dict[sid]):
        skey = (sid, sidx)
        sid_2_skeys_id[sid].append(skey)
        if sid in sids_id_multis:
            sid_2_skeys_id_multis[sid].append(skey)
            skeys_id_multis.add(skey)

skeys_train_multis, skeys_id_val, skeys_id_test = strat_splits(
    n_classes=n_sids_id_multis, 
    n_draws=n_samps_id_eval, 
    pct_splits=pct_rem_id_eval, 
    n_insts_2_classes=n_insts_2_classes_s, 
    class_2_insts=sid_2_skeys_id_multis, 
    insts=skeys_id_multis,
)

skeys_id_singles = set((sid, 0) for sid in sids_id_singles)
skeys_train      = skeys_train_multis.union(skeys_id_singles)

print("ID Split Complete!")

# N-SHOT TRACKING

print("Constructing n-shot tracking structures...")

skeys_id = set()
skeys_id |= skeys_train
skeys_id |= skeys_id_val
skeys_id |= skeys_id_test

def find_range_index(upper_bounds, n):
    assert isinstance(n, int), f"n must be an int, got {type(n).__name__}"
    if n <= 0:
        raise ValueError(f"find_range_index(): n = {n}")

    return bisect.bisect_left(upper_bounds, n)

"""
n-shot tracking data structures

`n_shot_tracker` structure:
~ [List] Edition ~
(for assembly)

n_shot_tracker = [
    {
        "train" : set(skeys),
        "id_val" : set(skeys),
        "id_test" : set(skeys),
    },
    {...}.
    ...
]

`id_eval_nshot` structure:
~ [Dict] Edition ~
(for saving to file)

id_eval_nshot = {
    "names" : NST_NAMES (it's a list),
    "buckets" : {
        nst_name0 : {
            "id_val" : set(skeys),
            "id_test" : set(skeys),
        }.
        nst_name1 : {
            ...
        },
        ...
    },
}

"""

n_shot_tracker = []

for _ in range(len(NST_NAMES)):
    nst_bucket = {
        "train" : set(),
        "id_val" : set(),
        "id_test" : set(),
    }
    n_shot_tracker.append(nst_bucket)

for sid in sids_id:

    sid_skeys_train = set()
    sid_skeys_val   = set()
    sid_skeys_test  = set()

    n_skeys = n_samps_dict[sid]
    n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
    for skey in sid_2_skeys_id[sid]:
        # check which split the sample landed in
        if skey in skeys_train:
            n_skeys_train += 1
            sid_skeys_train.add(skey)
        elif skey in skeys_id_val:
            n_skeys_id_val += 1
            sid_skeys_val.add(skey)
        elif skey in skeys_id_test:
            n_skeys_id_test += 1
            sid_skeys_test.add(skey)

    if n_skeys != n_skeys_train + n_skeys_id_val + n_skeys_id_test:
        print("PROBLEM")
        print(f"{n_skeys} {n_skeys_train} {n_skeys_id_val} {n_skeys_id_test}")

    idx_nst_bucket = find_range_index(NST_UPPER_BOUNDS, n_skeys_train)
    n_shot_tracker[idx_nst_bucket]["train"].update(sid_skeys_train)
    n_shot_tracker[idx_nst_bucket]["id_val"].update(sid_skeys_val)
    n_shot_tracker[idx_nst_bucket]["id_test"].update(sid_skeys_test)

print("n-shot tracking complete!")

# SAVE SPLITS AND N-SHOT TRACKING TO FILE (all are sets of skeys)

print("Saving Stuff...")

# create dirs (after splits have been generated so that dirs aren't created if the run is terminated early)
os.makedirs(dpath_splits, exist_ok=True)
os.makedirs(dpath_figs, exist_ok=True)

write_pickle(skeys_train, dpath_splits / "train.pkl")
write_pickle(skeys_id_val, dpath_splits / "id_val.pkl")
write_pickle(skeys_id_test, dpath_splits / "id_test.pkl")
write_pickle(skeys_ood_val, dpath_splits / "ood_val.pkl")
write_pickle(skeys_ood_test, dpath_splits / "ood_test.pkl")

# [List] Edition --> [Dict] Edition
id_eval_nshot = {
    "names" : NST_NAMES,  # just to preserve ordering of the names if needed
    "buckets" : {
        name : {
            "id_val" : bucket["id_val"],
            "id_test" : bucket["id_test"],
        }
        for name, bucket in zip(NST_NAMES, n_shot_tracker)
    },
}

write_pickle(id_eval_nshot, dpath_splits / f"id_eval_nshot.pkl")

print("Saved!")

# ANALYSIS + PLOTTING
# maybe isolate plotting / table generation to another file -- might be a little harder with the n-shot tracking, but still probably possible

print("Plotting stuff...")

def plot_split_distribution(data, labels_data, colors, title, x_label, y_label, filepath, ema=False, scale=None, marker="", markersize=6, markeredgewidth=0.5, linestyle="-", alpha=1.0):
    """
    Args:
    - n_insts_pc -------- [List(int)] ----- Number of instances per class (sorted?)
    - n_insts_pc_rem ---- [List(int)] ----- Number of instances per class remaining after split (e.g. train in train/eval split)
    - n_insts_pc_eval --- [List(int)] ----- Number of instances per class split out for eval
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

# OOD SPLIT DISTRIBUTION PLOTTING

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

title    = "OOD Split Distribution"
filepath = str(dpath_figs / "distribution_ood.png")
plot_split_distribution(
    data, labels_data, colors, 
    title, x_label, y_label, 
    filepath,
)

title    = "OOD Split Distribution (Log-Scale)"
filepath = str(dpath_figs / "distribution_ood_log.png")
plot_split_distribution(
    data, labels_data, colors, 
    title, x_label, y_label, 
    filepath, 
    ema=False, scale="symlog", 
    marker="|", markersize=6, markeredgewidth=1.0, linestyle="", alpha=1.0,
)

title = "OOD Split Distribution (Log-Scale + Smoothed)"
filepath = str(dpath_figs / "distribution_ood_log_smooth.png")
plot_split_distribution(
    data, labels_data, colors, 
    title, x_label, y_label, 
    filepath, 
    ema=True, scale="log",
)

# ID SPLIT DISTRIBUTION PLOTTING (singletons omitted)

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
        # check which split the sample landed in
        if skey in skeys_train:
            n_skeys_train += 1
        elif skey in skeys_id_val:
            n_skeys_id_val += 1
        elif skey in skeys_id_test:
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

title    = "ID Split Distribution"
filepath = str(dpath_figs / "distribution_id.png")
plot_split_distribution(
    data, labels_data, colors, 
    title, x_label, y_label, 
    filepath,
)

title    = "ID Split Distribution (Log-Scale)"
filepath = str(dpath_figs / "distribution_id_log.png")
plot_split_distribution(
    data, labels_data, colors, 
    title, x_label, y_label, 
    filepath, 
    ema=False, scale="symlog", 
    marker="|", markersize=6, markeredgewidth=0.5, linestyle="", alpha=1.0,
)

title    = "ID Split Distribution (Log-Scale + Smoothed)"
filepath = str(dpath_figs / "distribution_id_log_smooth.png")
plot_split_distribution(
    data, labels_data, colors, 
    title, x_label, y_label, 
    filepath, 
    ema=True, scale="log",
)

print("Plotting Complete!")

# SPLIT STATS TABLE

print("Generating Stats Tables...")

n_sids_id       = len(sids_id)
n_sids_ood_val  = len(sids_ood_val)
n_sids_ood_test = len(sids_ood_test)

sids_id_val_unrolled, _ = zip(*skeys_id_val)
n_sids_id_val           = len(set(sids_id_val_unrolled))

sids_id_test_unrolled, _ = zip(*skeys_id_test)
n_sids_id_test           = len(set(sids_id_test_unrolled))

n_skeys_train   = len(skeys_train)
n_skeys_id_val  = len(skeys_id_val)
n_skeys_id_test = len(skeys_id_test)

labels_cols = ["Split", "Num. Species", "Num. Samples"]
data = [
    ["Train",         f"{n_sids_id:,} ({n_sids_id / n_sids:.2%})",             f"{n_skeys_train:,} ({n_skeys_train / n_samps_total:.2%})"],
    ["ID Val",        f"{n_sids_id_val:,} ({n_sids_id_val / n_sids:.2%})",     f"{n_skeys_id_val:,} ({n_skeys_id_val / n_samps_total:.2%})"],
    ["ID Test",       f"{n_sids_id_test:,} ({n_sids_id_test / n_sids:.2%})",   f"{n_skeys_id_test:,} ({n_skeys_id_test / n_samps_total:.2%})"],
    ["OOD Val",       f"{n_sids_ood_val:,} ({n_sids_ood_val / n_sids:.2%})",   f"{n_samps_ood_val:,} ({n_samps_ood_val / n_samps_total:.2%})"],
    ["OOD Test",      f"{n_sids_ood_test:,} ({n_sids_ood_test / n_sids:.2%})", f"{n_samps_ood_test:,} ({n_samps_ood_test / n_samps_total:.2%})"],
    ["Whole Dataset", f"{n_sids:,} (100.00%)",                                 f"{n_samps_total:,} (100.00%)"],
]

fig, ax = plt.subplots(figsize=(5, 2))
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

# N-SHOT TRACKING STATS TABLES

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

labels_cols = ["Split"] + n_shot_col_names

data = [
    row_values_id_val,
    row_values_id_test,
]

fig, ax = plt.subplots(figsize=(5, 2))
ax.axis("off")

col_width  = 0.20  # config param
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
tbl.set_fontsize(5)  # config param

# thin out grid lines
for (i, j), cell in tbl.get_celld().items():
    cell.set_linewidth(0.5)

fontsize_title = 8  # config param

# plt.title("n-shot Bucket Stats: [Num. Samples (Num. Species)]", fontsize=10, fontweight="bold", pad=0)
plt.title("n-shot Bucket Stats: [Num. Samples (Num. Species)]", fontsize=fontsize_title, fontweight="bold", y=0.70)
plt.savefig(str(dpath_figs / "stats_nshot.png"), dpi=300, bbox_inches="tight")

print("Stats Tables Complete! Have a nice day!")
