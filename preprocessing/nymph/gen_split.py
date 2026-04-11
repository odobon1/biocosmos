"""
python -m preprocessing.nymph.gen_split
"""

from collections import Counter, defaultdict
import os
import random
import numpy as np  # type: ignore[import]

from utils.utils import paths, load_pickle, seed_libs
from utils.data import sid_to_genus
from utils.config import get_config_gen_split
from preprocessing.nymph.species_ids import get_sids_phylo_nymph
from utils.gen_split import (
    gen_ood_partitions,
    gen_id_partitions,
    gen_id_eval_nshot,
    gen_data_indexes,
    gen_class_counts_train,
    save_split,
    gen_ood_distribution_plots,
    gen_id_distribution_plots,
    gen_split_stats_table,
    gen_n_shot_table,
)

import pdb


def gen_split():
    cfg = get_config_gen_split()
    seed_libs(cfg.seed, seed_torch=False)

    dpath_split = paths["metadata"]["nymph"] / f"splits/{cfg.split_name}"
    dpath_figs = dpath_split / "figures"

    if os.path.isdir(dpath_split) and not cfg.allow_overwrite:
        error_msg = f"Split '{cfg.split_name}' already exists, choose different split_name!"
        raise ValueError(error_msg)
    print(f"Generating split: '{cfg.split_name}'")

    pct_ood_eval = pct_id_eval = cfg.pct_eval / 2  # OOD per-set percentage

    class_data = load_pickle(paths["metadata"]["nymph"] / "class_data.pkl")

    sids = sorted(set(get_sids_phylo_nymph()))  # OOD sets: insts  # OOD sets: insts

    n_sids = len(sids)
    n_sids_ood_eval = round(n_sids * pct_ood_eval)

    n_samps_dict = {}
    n_samps_total = 0
    for sid in sids:
        n_samps_sid = class_data[sid]["n_imgs"]
        n_samps_dict[sid] = n_samps_sid
        n_samps_total += n_samps_sid

    n_samps_eval = round(n_samps_total * cfg.pct_eval)  # OOD sets: n_draws

    genera = []
    genus_2_sids = defaultdict(list)  # OOD sets: class_2_insts
    for sid in sids:
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

    count_g = Counter(genera)
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

    # OOD PARTITIONS

    print("Constructing OOD partitions...")
    sids_id, sids_ood_val, sids_ood_test, skeys_ood_val, skeys_ood_test, n_samps_ood_val, n_samps_ood_test = gen_ood_partitions(
        n_genera,
        n_sids_ood_eval,
        pct_ood_eval,
        n_insts_2_classes_g,
        genus_2_sids,
        set(sids),
        cfg,
        n_samps_dict,
        n_samps_total,
    )
    print("OOD partitions complete!")

    # ID PARTITIONS

    print("Constructing ID partitions...")
    skeys_train, skeys_id_val, skeys_id_test, sid_2_skeys_id, sid_2_skeys_id_multis, sids_id_multis = gen_id_partitions(
        sids_id,
        n_samps_dict,
        n_samps_eval,
        n_samps_ood_val,
        n_samps_ood_test,
        pct_id_eval,
        cfg,
    )
    print("ID partitions complete!")

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }

    # N-SHOT TRACKING

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = gen_id_eval_nshot(cfg, sids_id, skeys_partitions, sid_2_skeys_id)
    print("n-shot tracking complete!")

    # GENERATE DATA INDEXES

    print("Generating data indexes...")
    data_indexes = gen_data_indexes(sids, skeys_partitions)
    print("Data indexes complete!")

    # CLASS COUNTS (FOR CLASS IMBALANCE)

    print("Generating class counts for train partition...")
    # class_counts_train = Counter(data_indexes["train"]["sids"])
    class_counts_train = gen_class_counts_train(data_indexes)
    print("Class counts complete!")

    # SAVE SPLIT

    print("Saving Split...")
    save_split(
        data_indexes, 
        id_eval_nshot, 
        class_counts_train, 
        dpath_split, 
        dpath_figs,
    )
    print("Split saved!")

    # OOD DISTRIBUTION PLOTTING

    print("Generating OOD distribution plots...")
    gen_ood_distribution_plots(
        genus_2_sids, 
        sids_id, 
        sids_ood_val, 
        sids_ood_test, 
        dpath_figs,
    )
    print("OOD distribution plots complete!")

    # ID DISTRIBUTION PLOTTING (singletons omitted)

    print("Generating ID distribution plots...")
    gen_id_distribution_plots(
        sids_id_multis, 
        sid_2_skeys_id_multis, 
        n_samps_dict, 
        skeys_partitions, 
        dpath_figs,
    )
    print("ID distribution plots complete!")

    # SPLIT STATS TABLE

    print("Generating split stats table...")
    gen_split_stats_table(
        sids_id,
        sids_ood_val,
        sids_ood_test,
        skeys_partitions,
        dpath_figs,
        n_sids,
    )
    print("Split stats table complete!")

    # N-SHOT TRACKING STATS TABLE

    print("Generating n-shot tracking stats table...")
    gen_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")

def main():
    print("Generating split...")
    gen_split()
    print("Split complete!")


if __name__ == "__main__":
    main()