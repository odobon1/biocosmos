from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os

from utils import dirpaths, read_pickle, write_pickle
from utils_ingestion import strat_splits


# config params
PCT_EVAL = 0.2
PCT_OOD_CLOSE_ENOUGH = 0.001
VERBOSE_OOD_SAMPLE_CHECK = False
SPLIT_NAME = "D"

dirpath_splits = dirpaths["repo_oli"] / f"metadata/splits/{SPLIT_NAME}"
dirpath_figs = dirpaths["repo_oli"] / f"figures/splits/{SPLIT_NAME}"

if os.path.isdir(dirpath_splits):
    error_msg = f"Split '{SPLIT_NAME}' already exists, choose a different SPLIT_NAME!"
    raise ValueError(error_msg)

pct_ood_eval = pct_id_eval = PCT_EVAL / 2

tax_nymph = read_pickle(dirpaths["repo_oli"] / "metadata/tax/nymph.pkl")
sids = set(tax_nymph["found"].keys())
n_sids = len(sids)
n_sids_ood_eval = round(n_sids * pct_ood_eval)

n_samps_dict = {}
n_samps = 0
for sid in sids:
    n_samps_s = tax_nymph["found"][sid]["meta"]["num_imgs"]
    n_samps_dict[sid] = n_samps_s
    n_samps += n_samps_s

n_samps_eval = round(n_samps * PCT_EVAL)

genera = []
genus_2_sids = defaultdict(list)
for sid in sids:
    genus = tax_nymph["found"][sid]["tax"]["genus"]
    genera.append(genus)
    genus_2_sids[genus].append(sid)

n_genera = len(set(genera))

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

count_2_classes_g = defaultdict(list)
for genus, count in count_g.items():
    count_2_classes_g[count].append(genus)

"""
`count_2_classes_*` structure:

count_2_classes_g (OOD):
{
    1 : [genus0, genus1, genus2, ...],
    2 : [...],
    4 : [...],
    ...
}

count_2_classes_s (ID):
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
        count_2_classes=count_2_classes_g, 
        pct_splits=pct_ood_eval, 
        class_2_insts=genus_2_sids, 
        insts=sids,
    )

    # NUM SAMPLES CHECK

    n_samps_ood_val = 0
    for sid in sids_ood_val:
        n_samps_s = n_samps_dict[sid]
        n_samps_ood_val += n_samps_s
    pct_samps_ood_val = n_samps_ood_val / n_samps

    n_samps_ood_test = 0
    for sid in sids_ood_test:
        n_samps_s = n_samps_dict[sid]
        n_samps_ood_test += n_samps_s
    pct_samps_ood_test = n_samps_ood_test / n_samps

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

sids_id_singles = set()
for sid in sids_id:
    if n_samps_dict[sid] == 1:
        sids_id_singles.add(sid)
        
sids_rem = sids_id - sids_id_singles

n_sids_rem = len(sids_rem)  # n_classes
n_samps_id_eval = n_samps_eval - (n_samps_ood_val + n_samps_ood_test)  # n_draws

count_2_classes_s = defaultdict(list)  # count_2_classes
for sid in sids_rem:
    count = n_samps_dict[sid]
    count_2_classes_s[count].append(sid)

pct_rem_id_eval = pct_id_eval / (1 - pct_id_eval)  # pct_splits (10 / 90)

sid_2_skeys = defaultdict(list)  # class_2_insts
skeys = set()  # insts
for sid in sids_rem:
    for sidx in range(n_samps_dict[sid]):
        skey = (sid, sidx)
        sid_2_skeys[sid].append(skey)
        skeys.add(skey)

skeys_train, skeys_id_val, skeys_id_test = strat_splits(
    n_classes=n_sids_rem, 
    n_draws=n_samps_id_eval, 
    count_2_classes=count_2_classes_s, 
    pct_splits=pct_rem_id_eval, 
    class_2_insts=sid_2_skeys, 
    insts=skeys,
)

skeys_singles = set((sid, 0) for sid in sids_id_singles)
skeys_train.update(skeys_singles)

print("ID Split Complete!")

# SAVE SPLITS TO FILE

# create dirs (after splits have been generated so that dirs aren't created if the run is terminated early)
os.makedirs(dirpath_splits)
os.makedirs(dirpath_figs)

write_pickle(skeys_train, dirpath_splits / "train.pkl")
write_pickle(skeys_id_val, dirpath_splits / "id_val.pkl")
write_pickle(skeys_id_test, dirpath_splits / "id_test.pkl")
write_pickle(skeys_ood_val, dirpath_splits / "ood_val.pkl")
write_pickle(skeys_ood_test, dirpath_splits / "ood_test.pkl")

# ANALYSIS + PLOTTING

def plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath, ema=False, log=False):
    """
    Args:
    - n_insts_pc [List(int)] -------- Number of instances per class (sorted?)
    - n_insts_pc_rem [List(int)] ---- Number of instances per class remaining after split (e.g. train in train/eval split)
    - n_insts_pc_eval [List(int)] --- Number of instances per class split out for eval
    - filepath [str] ---------------- Filepath to save plot to
    - ema [bool] -------------------- Whether to apply exponential moving average
    - log [bool] -------------------- Whether to plot y-axis on log-scale
    """

    def compute_ema(vals, alpha_ema=0.99):
        ema = [vals[0]]
        for i in range(1, len(vals)):
            val_i = vals[i]
            ema_i = alpha_ema * ema[-1] + (1 - alpha_ema) * val_i
            ema.append(ema_i)

        return ema

    x  = range(len(data[0]))

    plt.figure(figsize=(8, 6))

    if ema:
        for i in range(len(data)):
            data[i] = compute_ema(data[i])

    for i in range(len(data)):
        plt.plot(x, data[i], color=colors[i], label=labels[i])

    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel(x_label, fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")

    if log:
        plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")

# OOD PLOTTING

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

title = "OOD Split Distribution (Log-Scale + Smoothed)"
filepath = str(dirpath_figs / "distribution_ood_log.png")
plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath, ema=True, log=True)

# ID PLOTTING

"""
`sid_tups` structure:

(sid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test)

^ sorted on n_skeys and then sid (alphabetical)
"""

sid_tups = []
for sid in sid_2_skeys.keys():
    n_skeys_s = len(sid_2_skeys[sid])
    n_skeys_s_train, n_skeys_s_id_val, n_skeys_s_id_test = 0, 0, 0
    for skey in sid_2_skeys[sid]:
        if skey in skeys_train:
            n_skeys_s_train += 1
        elif skey in skeys_id_val:
            n_skeys_s_id_val += 1
        elif skey in skeys_id_test:
            n_skeys_s_id_test += 1

    sid_tups.append((sid, n_skeys_s, n_skeys_s_train, n_skeys_s_id_val, n_skeys_s_id_test))

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

title = "ID Split Distribution (Log-Scale + Smoothed)"
filepath = str(dirpath_figs / "distribution_id_log.png")
plot_split_distribution(data, labels, colors, title, x_label, y_label, filepath, ema=True, log=True)

# STATS TABLE

col_labels = ["Split", "Num. Species", "Num. Samples"]
data = [
    ["Train", f"{len(sids_id):,} ({round(100 * len(sids_id) / n_sids, 3)}%)", f"{len(skeys_train):,} ({round(100 * len(skeys_train) / n_samps, 3)}%)"],
    ["ID Val", f"{len(sids_id):,} ({round(100 * len(sids_id) / n_sids, 3)}%)", f"{len(skeys_id_val):,} ({round(100 * len(skeys_id_val) / n_samps, 3)}%)"],
    ["ID Test", f"{len(sids_id):,} ({round(100 * len(sids_id) / n_sids, 3)}%)", f"{len(skeys_id_test):,} ({round(100 * len(skeys_id_test) / n_samps, 3)}%)"],
    ["OOD Val", f"{len(sids_ood_val):,} ({round(100 * len(sids_ood_val) / n_sids, 3)}%)", f"{n_samps_ood_val:,} ({round(100 * n_samps_ood_val / n_samps, 3)}%)"],
    ["OOD Test", f"{len(sids_ood_test):,} ({round(100 * len(sids_ood_test) / n_sids, 3)}%)", f"{n_samps_ood_test:,} ({round(100 * n_samps_ood_test / n_samps, 3)}%)"],
    ["Whole Dataset", f"{n_sids:,} (100.0%)", f"{n_samps:,} (100.0%)"],
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
