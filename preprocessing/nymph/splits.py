"""
python -m preprocessing.nymph.splits
"""

from utils.utils import paths, seed_libs
from utils.config import get_config_splits
from preprocessing.common.splits import (
    build_genus_2_sids,
    build_n_insts_2_classes_g,
    build_ood_partitions,
    build_id_partitions,
    build_id_eval_nshot,
    build_class_counts_train,
    save_split,
    generate_ood_distribution_plots,
    generate_id_distribution_plots,
    generate_n_shot_table,
)
from preprocessing.nymph.splits_utils import (
    build_data_indexes,
    build_img_ptrs,
    build_sid_2_samp_idxs,
    generate_split_stats_table,
)
from utils.phylo import PhyloVCV

import pdb


DATASET = "nymph"


def build_splits():
    cfg = get_config_splits()
    seed_libs(cfg.seed, seed_torch=False)
    dpath_split = paths["metadata"][DATASET] / f"splits/{cfg.split_name}"
    dpath_figs = dpath_split / "figures"
    print(f"Generating split: '{cfg.split_name}'")

    pvcv = PhyloVCV(dataset=DATASET)
    sids = pvcv.get_sids()  # OOD partitions: insts

    img_ptrs_all = build_img_ptrs(sids)
    sid_2_samp_idxs = build_sid_2_samp_idxs(sids, pos_filter=cfg.pos_filter, img_ptrs=img_ptrs_all)

    sids_dropped = [sid for sid in sorted(sids) if len(sid_2_samp_idxs[sid]) == 0]
    if sids_dropped:
        print(f"Dropping {len(sids_dropped)} species with no samples matching pos_filter={cfg.pos_filter!r}.")

    sids = [sid for sid in sorted(sids) if len(sid_2_samp_idxs[sid]) > 0]
    n_sids = len(sids)
    if not sids:
        raise ValueError(f"No samples available after applying pos_filter={cfg.pos_filter!r}.")

    n_samps_dict = {sid: len(sid_2_samp_idxs[sid]) for sid in sids}
    n_samps_total = sum(n_samps_dict.values())

    if cfg.pos_filter is not None:
        n_samps_total_raw = sum(len(v) for v in img_ptrs_all.values())
        print(
            f"Retained {n_sids:,}/{len(pvcv.get_sids()):,} species and {n_samps_total:,}/{n_samps_total_raw:,} samples "
            f"after pos_filter={cfg.pos_filter!r}."
        )

    genus_2_sids = build_genus_2_sids(sids)  # OOD partitions: class_2_insts
    n_insts_2_classes_g = build_n_insts_2_classes_g(sids)  # OOD partitions: n_insts_2_classes

    # OOD PARTITIONS

    print("Constructing OOD partitions...")
    sids_id, sids_ood_val, sids_ood_test, skeys_ood_val, skeys_ood_test = build_ood_partitions(
        n_insts_2_classes_g,
        genus_2_sids,
        set(sids),
        sid_2_samp_idxs,
        n_samps_dict,
        cfg,
    )
    print("OOD partitions complete!")

    # ID PARTITIONS

    print("Constructing ID partitions...")
    skeys_train, skeys_id_val, skeys_id_test, sid_2_skeys_id, sid_2_skeys_id_multis, sids_id_multis = build_id_partitions(
        sids_id,
        sid_2_samp_idxs,
        n_samps_dict,
        cfg,
    )
    print("ID partitions complete!")

    # PARTITION SKEYS (SAMPLE-KEYS)

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }

    # N-SHOT TRACKING

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, sids_id, skeys_partitions, sid_2_skeys_id)
    print("n-shot tracking complete!")

    # GENERATE DATA INDEXES

    print("Generating data indexes...")
    data_indexes = build_data_indexes(sids, skeys_partitions)
    if cfg.pos_filter is not None:
        partition_indexes = {
            "train": data_indexes["train"],
            "validation/id": data_indexes["validation"]["id"],
            "validation/ood": data_indexes["validation"]["ood"],
            "test/id": data_indexes["test"]["id"],
            "test/ood": data_indexes["test"]["ood"],
        }
        for partition_name, data_index in partition_indexes.items():
            invalid_pos = sorted({pos for pos in data_index["pos"] if pos != cfg.pos_filter})
            if invalid_pos:
                raise ValueError(
                    f"Partition '{partition_name}' contains positions outside pos_filter={cfg.pos_filter!r}: {invalid_pos}"
                )
    print("Data indexes complete!")

    # CLASS COUNTS (FOR CLASS IMBALANCE)

    print("Generating class counts for train partition...")
    # class_counts_train = Counter(data_indexes["train"]["sids"])
    class_counts_train = build_class_counts_train(data_indexes)
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
    generate_ood_distribution_plots(
        genus_2_sids, 
        sids_id, 
        sids_ood_val, 
        sids_ood_test, 
        dpath_figs,
    )
    print("OOD distribution plots complete!")

    # ID DISTRIBUTION PLOTTING (singletons omitted)

    print("Generating ID distribution plots...")
    generate_id_distribution_plots(
        sids_id_multis, 
        sid_2_skeys_id_multis, 
        n_samps_dict, 
        skeys_partitions, 
        dpath_figs,
    )
    print("ID distribution plots complete!")

    # SPLIT STATS TABLE

    print("Generating split stats table...")
    generate_split_stats_table(
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
    generate_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")

def main():
    print("Generating split...")
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()