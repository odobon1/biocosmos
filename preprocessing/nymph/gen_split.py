"""
python -m preprocessing.nymph.gen_split
"""

from collections import Counter, defaultdict
import matplotlib.pyplot as plt  # type: ignore[import]
import os
import glob
from tqdm import tqdm  # type: ignore[import]
import random
import numpy as np  # type: ignore[import]
from sklearn.model_selection import train_test_split  # type: ignore[import]
import copy
import pandas as pd  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from utils.data import Split, assemble_indexes, sid_to_genus
from utils.config import get_config_gen_split
from preprocessing.nymph.species_ids import get_sids_phylo_nymph
from utils.gen_split import strat_split, plot_split_distribution, gen_id_eval_nshot

import pdb


cfg = get_config_gen_split()

random.seed(cfg.seed)
np.random.seed(cfg.seed)

assert len(cfg.nst_names) == len(cfg.nst_seps) + 1, \
    f"len(nst_names) ({len(cfg.nst_names)}) != len(nst_seps) + 1 ({len(cfg.nst_seps)})"

dpath_split = paths["metadata"]["nymph"] / f"splits/{cfg.split_name}"
dpath_figs  = dpath_split / "figures"

if os.path.isdir(dpath_split) and not cfg.allow_overwrite:
    error_msg = f"Split '{cfg.split_name}' already exists, choose different split_name!"
    raise ValueError(error_msg)
print(F"Split Name: '{cfg.split_name}'")

pct_ood_eval = pct_id_eval = cfg.pct_eval / 2  # OOD per-set percentage

class_data = load_pickle(paths["metadata"]["nymph"] / "class_data.pkl")

# SET SIDS
sids = set(get_sids_phylo_nymph())  # OOD sets: insts

n_sids          = len(sids)
n_sids_ood_eval = round(n_sids * pct_ood_eval)

n_samps_dict  = {}
n_samps_total = 0
for sid in sids:
    n_samps_sid       = class_data[sid]["n_imgs"]
    n_samps_dict[sid] = n_samps_sid
    n_samps_total += n_samps_sid

n_samps_eval = round(n_samps_total * cfg.pct_eval)  # OOD sets: n_draws

genera       = []
genus_2_sids = defaultdict(list)  # OOD sets: class_2_insts
for sid in sorted(sids):
    genus = sid_to_genus(sid)
    genera.append(genus)
    genus_2_sids[genus].append(sid)

n_genera = len(set(genera))  # OOD sets: n_classes

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

count_g             = Counter(genera)
n_insts_2_classes_g = defaultdict(list)  # n_insts_2_classes OOD sets
for genus, count in count_g.items():
    n_insts_2_classes_g[count].append(genus)

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

# OOD SETS

print("Constructing OOD Set...")

close_enough = False
i            = 0
while not close_enough:
    i += 1
    sids_id, sids_ood_val, sids_ood_test = strat_split(
        n_classes        =n_genera, 
        n_draws          =n_sids_ood_eval, 
        pct_sets         =pct_ood_eval, 
        n_insts_2_classes=n_insts_2_classes_g, 
        class_2_insts    =genus_2_sids, 
        insts            =sids,
        seed             =cfg.seed+i,
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
    for samp_idx in range(n_samps_dict[sid]):
        skey = (sid, samp_idx)
        skeys_ood_val.add(skey)

skeys_ood_test = set()
for sid in sids_ood_test:
    for samp_idx in range(n_samps_dict[sid]):
        skey = (sid, samp_idx)
        skeys_ood_test.add(skey)

print("OOD Complete!")

# ID SETS

print("Constructing ID Sets...")

sids_id_singles = set() # species id's with 1 sample i.e. singletons
for sid in sids_id:
    if n_samps_dict[sid] == 1:
        sids_id_singles.add(sid)

sids_id_multis = sids_id - sids_id_singles  # species id's with 2+ samples

n_sids_id_multis = len(sids_id_multis)  # ID sets: n_classes
n_samps_id_eval  = n_samps_eval - (n_samps_ood_val + n_samps_ood_test)  # ID sets: n_draws

n_insts_2_classes_s = defaultdict(list)  # ID sets: n_insts_2_classes
for sid in sids_id_multis:
    count = n_samps_dict[sid]
    n_insts_2_classes_s[count].append(sid)

pct_rem_id_eval = pct_id_eval / (1 - pct_id_eval)  # ID sets: pct_sets (10 / 90)

sid_2_skeys_id_multis = defaultdict(list)  # ID sets: class_2_insts
sid_2_skeys_id        = defaultdict(list)  # used for n-shot tracking
skeys_id_multis       = set()  # ID sets: insts

for sid in sids_id:
    for samp_idx in range(n_samps_dict[sid]):
        skey = (sid, samp_idx)
        sid_2_skeys_id[sid].append(skey)
        if sid in sids_id_multis:
            sid_2_skeys_id_multis[sid].append(skey)
            skeys_id_multis.add(skey)

skeys_train_multis, skeys_id_val, skeys_id_test = strat_split(
    n_classes        =n_sids_id_multis, 
    n_draws          =n_samps_id_eval, 
    pct_sets         =pct_rem_id_eval, 
    n_insts_2_classes=n_insts_2_classes_s, 
    class_2_insts    =sid_2_skeys_id_multis, 
    insts            =skeys_id_multis,
    seed             =cfg.seed,
)

skeys_id_singles = set((sid, 0) for sid in sids_id_singles)
skeys_train      = skeys_train_multis.union(skeys_id_singles)

print("ID Complete!")

skeys_partitions = {
    "train":    skeys_train,
    "id_val":   skeys_id_val,
    "id_test":  skeys_id_test,
    "ood_val":  skeys_ood_val,
    "ood_test": skeys_ood_test,
}

# N-SHOT TRACKING

print("Constructing n-shot tracking structures...")
id_eval_nshot = gen_id_eval_nshot(cfg, sids_id, skeys_partitions, sid_2_skeys_id)
print("n-shot tracking complete!")

# GENERATE DATA INDEXES

"""
`img_ptrs` structure:

img_ptrs = {
    sid0: {
        0: fpath_img_s0_0,
        1: fpath_img_s0_1,
        ...,
    },
    sid1: {
        ...,
    },
    ...
}
"""

img_ptrs = {}

# iterate through sids, fetch image filenames, assign to indexes, add to img_ptrs structure
for sid in tqdm(sids):
    
    img_ptrs[sid] = {}

    dpath_imgs_sid = paths["nymph_imgs"] / sid
    ffpaths_png    = glob.glob(f"{dpath_imgs_sid}/*.png")
    rfpaths_png    = [png_file.split("images/", 1)[1] for png_file in ffpaths_png]  # full filepath --> relative filepath

    for i, rfpath in enumerate(rfpaths_png):
        img_ptrs[sid][i] = rfpath


df_metadata = pd.read_csv(paths["nymph_metadata"])

data_indexes = {}

for partition_name in ["train", "id_val", "id_test", "ood_val", "ood_test"]:

    data_index = {
        "sids":    [],
        "rfpaths": [],
    }

    skeys_partition = skeys_partitions[partition_name]

    for skey in skeys_partition:
        sid = skey[0]
        samp_idx = skey[1]

        data_index["sids"].append(sid)

        rfpath = img_ptrs[sid][samp_idx]
        data_index["rfpaths"].append(rfpath)

        fname_img = rfpath.split("/")[1]

    fname_imgs = [rfpath.split("/")[1] for rfpath in data_index["rfpaths"]]
    ordered_series_pos = df_metadata.set_index("mask_name")["class_dv"]
    ordered_series_sex = df_metadata.set_index("mask_name")["sex"]
    data_index["pos"] = ordered_series_pos.reindex(fname_imgs).astype(object).where(lambda x: x.notna(), None).tolist()
    data_index["sex"] = ordered_series_sex.reindex(fname_imgs).astype(object).where(lambda x: x.notna(), None).tolist()

    data_indexes[partition_name] = data_index

# CLASS COUNTS (FOR CLASS IMBALANCE)

# note: this all needs to get untangled....
data_index, _    = assemble_indexes(data_indexes["train"])
index_class_encs = data_index["class_encs"]

n_classes          = len(set(index_class_encs))
class_counts_train = np.bincount(index_class_encs, minlength=n_classes)  # counts[c] is number of samples with class encoding c

# SAVE SPLIT

print("Saving Split...")

split = Split(
    data_indexes,
    id_eval_nshot,
    class_counts_train,
)

# create dirs (after split has been generated so that dirs aren't created if the run is terminated early)
os.makedirs(dpath_split, exist_ok=True)
os.makedirs(dpath_figs, exist_ok=True)

save_pickle(split, dpath_split / "split.pkl")

print("Saved!")

# ANALYSIS + PLOTTING
# maybe isolate plotting / table generation to another file -- might be a little harder with the n-shot tracking, but still probably possible

print("Plotting stuff...")

# OOD SET DISTRIBUTION PLOTTING

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

# ID SETS DISTRIBUTION PLOTTING (singletons omitted)

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

labels_cols = ["Set", "Num. Species", "Num. Samples"]
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

labels_cols = ["Set"] + n_shot_col_names

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
