from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os

from utils import paths, read_pickle, write_pickle
from utils_ingestion import strat_splits

import pdb


# config params
PCT_EVAL = 0.2
PCT_OOD_CLOSE_ENOUGH = 0.001  # careful, this parameter dictates required accuracy as the stopping criterion for a random search, too low is no no (0.0001 is good)
VERBOSE_OOD_SAMPLE_CHECK = False
SPLIT_NAME = "A"
ALLOW_OVERWRITES = True

dirpath_splits = paths["metadata_o"] / f"splits/{SPLIT_NAME}"
dirpath_figs = paths["repo_o"] / f"figures/splits/{SPLIT_NAME}"

if os.path.isdir(dirpath_splits) and not ALLOW_OVERWRITES:
    error_msg = f"Split '{SPLIT_NAME}' already exists, choose a different SPLIT_NAME!"
    raise ValueError(error_msg)
print(F"SPLIT_NAME: {SPLIT_NAME}")

pct_ood_eval = pct_id_eval = PCT_EVAL / 2  # OOD splits: pct_splits

tax_nymph = read_pickle(paths["metadata_o"] / "tax/nymph.pkl")
sids = set(tax_nymph["found"].keys())  # OOD splits: insts
n_sids = len(sids)
n_sids_ood_eval = round(n_sids * pct_ood_eval)

n_samps_dict = {}
n_samps_total = 0
for sid in sids:
    n_samps_sid = tax_nymph["found"][sid]["meta"]["num_imgs"]
    n_samps_dict[sid] = n_samps_sid
    n_samps_total += n_samps_sid

n_samps_eval = round(n_samps_total * PCT_EVAL)  # OOD splits: n_draws

genera = []
genus_2_sids = defaultdict(list)  # OOD splits: class_2_insts
for sid in sids:
    genus = tax_nymph["found"][sid]["tax"]["genus"]
    genera.append(genus)
    genus_2_sids[genus].append(sid)

n_genera = len(set(genera))  # OOD splits: n_classes

"""
`genus_2_sids` & `sid_2_skeys` structure (class_2_insts):

genus_2_sids:
{
    genus0 : [sid0, sid1, sid2, ...],
    genus1 : [...],
    ...
}

sid_2_skeys:
{
    sid0 : [skey0, skey1, skey2, ...],
    sid1 : [...],
    ...
}
"""

count_g = Counter(genera)

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

    if VERBOSE_OOD_SAMPLE_CHECK:
        print("------------------------")
        print(f"{round(pct_samps_ood_val * 100, 3)}%")
        print(f"{round(pct_samps_ood_test * 100, 3)}%")

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
n_samps_id_eval = n_samps_eval - (n_samps_ood_val + n_samps_ood_test)  # ID splits: n_draws

n_insts_2_classes_s = defaultdict(list)  # ID splits: n_insts_2_classes
for sid in sids_id_multis:
    count = n_samps_dict[sid]
    n_insts_2_classes_s[count].append(sid)

pct_rem_id_eval = pct_id_eval / (1 - pct_id_eval)  # ID splits: pct_splits (10 / 90)

sid_2_skeys = defaultdict(list)  # ID splits: class_2_insts
skeys = set()  # ID splits: insts
for sid in sids_id_multis:
    for sidx in range(n_samps_dict[sid]):
        skey = (sid, sidx)
        sid_2_skeys[sid].append(skey)
        skeys.add(skey)

skeys_train, skeys_id_val, skeys_id_test = strat_splits(
    n_classes=n_sids_id_multis, 
    n_draws=n_samps_id_eval, 
    pct_splits=pct_rem_id_eval, 
    n_insts_2_classes=n_insts_2_classes_s, 
    class_2_insts=sid_2_skeys, 
    insts=skeys,
)

skeys_id_singles = set((sid, 0) for sid in sids_id_singles)
skeys_train.update(skeys_id_singles)

print("ID Split Complete!")

# SAVE SPLITS TO FILE

# create dirs (after splits have been generated so that dirs aren't created if the run is terminated early)
os.makedirs(dirpath_splits, exist_ok=True)
os.makedirs(dirpath_figs, exist_ok=True)

write_pickle(skeys_train, dirpath_splits / "train.pkl")
write_pickle(skeys_id_val, dirpath_splits / "id_val.pkl")
write_pickle(skeys_id_test, dirpath_splits / "id_test.pkl")
write_pickle(skeys_ood_val, dirpath_splits / "ood_val.pkl")
write_pickle(skeys_ood_test, dirpath_splits / "ood_test.pkl")

# ANALYSIS + PLOTTING
# maybe isolate plotting / table generation to another file

def plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath, ema=False, scale=None, marker="", markersize=6, markeredgewidth=0.5, linestyle="-", alpha=1.0):
    """
    Args:
    - n_insts_pc [List(int)] -------- Number of instances per class (sorted?)
    - n_insts_pc_rem [List(int)] ---- Number of instances per class remaining after split (e.g. train in train/eval split)
    - n_insts_pc_eval [List(int)] --- Number of instances per class split out for eval
    - filepath [str] ---------------- Filepath to save plot to
    - ema [bool] -------------------- Whether to apply exponential moving average
    - scale [None or str] ----------- None, "log", "symlog"
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
            label=labels[i], 
            marker=marker, 
            markersize=markersize, 
            markeredgewidth=markeredgewidth, 
            linestyle=linestyle, 
            alpha=alpha
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

data = [n_sids_pg, n_sids_pg_id, n_sids_pg_ood_eval]
colors = ["crimson", "darkorange", "teal"]
labels = ["Total", "ID-Train/Eval", "OOD-Eval"]
title = "OOD Split Distribution"
x_label = "Sorted Genera"
y_label = "Num. Species"
filepath = str(dirpath_figs / "distribution_ood.png")
plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath)

title = "OOD Split Distribution (Log-Scale)"
filepath = str(dirpath_figs / "distribution_ood_log.png")
plot_split_distribution(
    data, labels, colors, 
    title, x_label, y_label, 
    filepath, 
    ema=False, scale="symlog", 
    marker="|", markersize=6, markeredgewidth=1.0, linestyle="", alpha=1.0,
)

title = "OOD Split Distribution (Log-Scale + Smoothed)"
filepath = str(dirpath_figs / "distribution_ood_log_smooth.png")
plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath, ema=True, scale="log")

# ID SPLIT DISTRIBUTION PLOTTING (singletons omitted)

"""
`sid_tups` structure:

(sid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test)

^ sorted on n_skeys and then sid (alphabetical)

***** Think this is all I need for the n-shot analysis / data structure organization *****
"""

sid_tups = []
for sid in sids_id_multis:
    n_skeys = n_samps_dict[sid]
    n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
    for skey in sid_2_skeys[sid]:
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

data = [n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_eval]
colors = ["crimson", "darkorange", "teal"]
labels = ["Total", "Train (ID)", "Eval (ID)"]
title = "ID Split Distribution"
x_label = "Sorted Species"
y_label = "Num. Samples"
filepath = str(dirpath_figs / "distribution_id.png")
plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath)

title = "ID Split Distribution (Log-Scale)"
filepath = str(dirpath_figs / "distribution_id_log.png")
plot_split_distribution(
    data, labels, colors, 
    title, x_label, y_label, 
    filepath, 
    ema=False, scale="symlog", 
    marker="|", markersize=6, markeredgewidth=0.5, linestyle="", alpha=1.0)

title = "ID Split Distribution (Log-Scale + Smoothed)"
filepath = str(dirpath_figs / "distribution_id_log_smooth.png")
plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath, ema=True, scale="log")

# STATS TABLE

n_sids_id = len(sids_id)
n_sids_ood_val = len(sids_ood_val)
n_sids_ood_test = len(sids_ood_test)

sids_id_val, _ = zip(*skeys_id_val)
n_sids_id_val = len(set(sids_id_val))

sids_id_test, _ = zip(*skeys_id_test)
n_sids_id_test = len(set(sids_id_test))

n_skeys_train = len(skeys_train)
n_skeys_id_val = len(skeys_id_val)
n_skeys_id_test = len(skeys_id_test)

col_labels = [
    "Split", 
    "Num. Species", 
    "Num. Samples",
]
data = [
    [
        "Train", 
        f"{n_sids_id:,} ({100 * n_sids_id / n_sids:.2f}%)", 
        f"{len(skeys_train):,} ({100 * n_skeys_train / n_samps_total:.2f}%)",
    ],
    [
        "ID Val", 
        f"{n_sids_id_val:,} ({100 * n_sids_id_val / n_sids:.2f}%)", 
        f"{len(skeys_id_val):,} ({100 * n_skeys_id_val / n_samps_total:.2f}%)"
    ],
    [
        "ID Test", 
        f"{n_sids_id_test:,} ({100 * n_sids_id_test / n_sids:.2f}%)", 
        f"{len(skeys_id_test):,} ({100 * n_skeys_id_test / n_samps_total:.2f}%)"
    ],
    [
        "OOD Val", 
        f"{n_sids_ood_val:,} ({100 * n_sids_ood_val / n_sids:.2f}%)", 
        f"{n_samps_ood_val:,} ({100 * n_samps_ood_val / n_samps_total:.2f}%)"
    ],
    [
        "OOD Test", 
        f"{n_sids_ood_test:,} ({100 * n_sids_ood_test / n_sids:.2f}%)", 
        f"{n_samps_ood_test:,} ({100 * n_samps_ood_test / n_samps_total:.2f}%)"
    ],
    [
        "Whole Dataset", 
        f"{n_sids:,} (100.00%)", 
        f"{n_samps_total:,} (100.00%)"
    ],
]

fig, ax = plt.subplots(figsize=(5, 2))
ax.axis("off")
tbl = ax.table(
    cellText=data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center"
)

for col_idx, _ in enumerate(col_labels):
    cell = tbl[0, col_idx]
    cell.get_text().set_fontweight("bold")

plt.title("Split Stats", fontweight="bold", pad=-5)
plt.savefig(str(dirpath_figs / "split_stats.png"), dpi=150, bbox_inches="tight")
