"""
python -m preprocessing.bryo.splits
"""

from utils.utils import paths, seed_libs
from utils.config import get_config_splits
from preprocessing.common.splits import (
    build_genus_2_cids,
    build_n_insts_2_classes_g,
    build_ood_partitions,
    build_id_partitions,
    build_trainval_skeys_partition,
    build_id_eval_nshot,
    build_class_counts_train,
    build_dev_skeys_partitions,
    save_split,
    generate_ood_distribution_plots,
    generate_id_distribution_plots,
    generate_n_shot_table,
    generate_basic_split_stats_table,
)
from preprocessing.bryo.splits_utils import (
    build_data_indexes_bryo,
    build_img_ptrs_bryo,
)
from utils.phylo import PhyloVCV


DATASET = "bryo"


def build_splits() -> None:
    cfg = get_config_splits()
    seed_libs(cfg.seed, seed_torch=False)
    dpath_split = paths["metadata"][DATASET] / f"splits/{cfg.split_name}"
    dpath_figs = dpath_split / "figures"
    dpath_split_dev = paths["metadata"][DATASET] / "splits/dev"
    dpath_figs_dev = dpath_split_dev / "figures"
    print(f"Generating split: '{cfg.split_name}'")

    pvcv = PhyloVCV(dataset=DATASET)
    cids = pvcv.get_cids()  # genera for bryo

    img_ptrs_all = build_img_ptrs_bryo(cids)

    cids_dropped = [cid for cid in sorted(cids) if cid not in img_ptrs_all or len(img_ptrs_all[cid]) == 0]
    if cids_dropped:
        print(f"Dropping {len(cids_dropped)} genera with no images: {cids_dropped}")

    cids = [cid for cid in sorted(cids) if cid in img_ptrs_all and len(img_ptrs_all[cid]) > 0]
    n_cids = len(cids)
    if not cids:
        raise ValueError("No genera with images found.")

    cid_2_samp_idxs = {cid: list(sorted(img_ptrs_all[cid].keys())) for cid in cids}
    n_samps_dict = {cid: len(cid_2_samp_idxs[cid]) for cid in cids}

    genus_2_cids = build_genus_2_cids(cids)
    n_insts_2_classes_g = build_n_insts_2_classes_g(cids)

    # OOD PARTITIONS

    print("Constructing OOD partitions...")
    cids_id, cids_ood_val, cids_ood_test, skeys_ood_val, skeys_ood_test = build_ood_partitions(
        n_insts_2_classes_g,
        genus_2_cids,
        set(cids),
        cid_2_samp_idxs,
        n_samps_dict,
        cfg,
    )
    print("OOD partitions complete!")

    # ID PARTITIONS

    n_samps_dict_id = {cid: n_samps_dict[cid] for cid in sorted(cids_id)}

    print("Constructing ID partitions...")
    (
        skeys_train,
        skeys_id_val,
        skeys_id_test,
        cid_2_skeys_id,
        cid_2_skeys_id_multis,
        cids_id_multis,
        skeys_id_test_extra_taken,
    ) = build_id_partitions(
        cids_id,
        cid_2_samp_idxs,
        n_samps_dict,
        cfg,
        skeys_id_test_extra=skeys_ood_val,
    )
    skeys_ood_val = skeys_ood_val - skeys_id_test_extra_taken
    print("ID partitions complete!")

    # PARTITION SKEYS (SAMPLE-KEYS)

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }
    skeys_partitions["trainval"] = build_trainval_skeys_partition(skeys_partitions)
    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, cfg.size_dev)

    # N-SHOT TRACKING

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id)
    print("n-shot tracking complete!")

    # GENERATE DATA INDEXES

    print("Generating data indexes...")
    data_indexes = build_data_indexes_bryo(cids, skeys_partitions, img_ptrs=img_ptrs_all)
    data_indexes_dev = build_data_indexes_bryo(cids, skeys_partitions_dev, img_ptrs=img_ptrs_all)
    print("Data indexes complete!")

    # CLASS COUNTS (FOR CLASS IMBALANCE)

    print("Generating class counts for train partition...")
    class_counts_train = build_class_counts_train(data_indexes)
    class_counts_train_dev = build_class_counts_train(data_indexes_dev)
    print("Class counts complete!")

    # SAVE SPLIT

    print("Saving split...")
    save_split(
        data_indexes,
        id_eval_nshot,
        class_counts_train,
        dpath_split,
        dpath_figs,
    )
    save_split(
        data_indexes_dev,
        id_eval_nshot,
        class_counts_train_dev,
        dpath_split_dev,
        dpath_figs_dev,
    )
    print("Primary and dev splits saved!")

    # OOD DISTRIBUTION PLOTTING

    print("Generating OOD distribution plots...")
    generate_ood_distribution_plots(
        genus_2_cids,
        cids_id,
        cids_ood_val,
        cids_ood_test,
        dpath_figs,
    )
    print("OOD distribution plots complete!")

    # ID DISTRIBUTION PLOTTING (singletons omitted)

    print("Generating ID distribution plots...")
    generate_id_distribution_plots(
        cids_id_multis,
        cid_2_skeys_id_multis,
        n_samps_dict_id,
        skeys_partitions,
        dpath_figs,
    )
    print("ID distribution plots complete!")

    # SPLIT STATS TABLE

    print("Generating split stats table...")
    generate_basic_split_stats_table(
        skeys_partitions=skeys_partitions,
        dpath_figs=dpath_figs,
        n_cids_total=n_cids,
        title="Split Stats (Bryozoa)",
    )
    print("Split stats table complete!")

    # N-SHOT TRACKING STATS TABLE

    print("Generating n-shot tracking stats table...")
    generate_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")

def main() -> None:
    print("Generating split...")
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()