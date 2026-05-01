"""
python -m preprocessing.lepid.splits
"""

from utils.utils import paths, seed_libs, load_pickle
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
    generate_id_distribution_plots,
    generate_n_shot_table,
    generate_basic_split_stats_table,
    generate_ood_distribution_plots,
)
from preprocessing.lepid.splits_utils import (
    build_data_indexes_lepid,
    build_img_ptrs_lepid,
    build_cid_2_samp_idxs_lepid,
)
from utils.phylo import PhyloVCV


DATASET = "lepid"


def build_splits():
    cfg = get_config_splits()
    seed_libs(cfg.seed, seed_torch=False)
    dpath_split = paths["metadata"][DATASET] / f"splits/{cfg.split_name}"
    dpath_figs = dpath_split / "figures"
    dpath_split_dev = paths["metadata"][DATASET] / "splits/dev"
    dpath_figs_dev = dpath_split_dev / "figures"
    print(f"Generating split: '{cfg.split_name}'")

    print("Loading phylogenetic covariance structure...")
    pvcv = PhyloVCV(dataset=DATASET)
    cids = pvcv.get_cids()
    print(f"Loaded phylogeny with {len(cids):,} species.")

    print("Loading class data...")
    class_data = load_pickle(paths["metadata"][DATASET] / "class_data.pkl")
    cid_2_family = {
        cid: class_data[cid]["family"]
        for cid in cids
    }
    print("Class data loaded.")

    print("Indexing Lepid image paths...")
    img_ptrs_all = build_img_ptrs_lepid(cids, cid_2_family)
    print("Image path indexing complete.")

    print("Applying sample-level position filter...")
    cid_2_samp_idxs = build_cid_2_samp_idxs_lepid(
        cids,
        cid_2_family,
        pos_filter=cfg.pos_filter,
        img_ptrs=img_ptrs_all,
    )
    print("Position filtering complete.")

    cids_dropped = [cid for cid in sorted(cids) if len(cid_2_samp_idxs[cid]) == 0]
    if cids_dropped:
        print(f"Dropping {len(cids_dropped)} species with no samples matching pos_filter={cfg.pos_filter!r}.")

    cids = [cid for cid in sorted(cids) if len(cid_2_samp_idxs[cid]) > 0]
    if not cids:
        raise ValueError(f"No samples available after applying pos_filter={cfg.pos_filter!r}.")

    cid_2_family = {
        cid: cid_2_family[cid]
        for cid in cids
    }
    n_samps_dict = {cid: len(cid_2_samp_idxs[cid]) for cid in cids}
    genus_2_cids = build_genus_2_cids(cids)
    n_insts_2_classes_g = build_n_insts_2_classes_g(cids)

    print("Constructing OOD species partitions...")
    cids_id, cids_ood_val, cids_ood_test, skeys_ood_val, skeys_ood_test = build_ood_partitions(
        n_insts_2_classes_g,
        genus_2_cids,
        set(cids),
        cid_2_samp_idxs,
        n_samps_dict,
        cfg,
    )
    print("OOD species partitions complete!")

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

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }
    skeys_partitions["trainval"] = build_trainval_skeys_partition(skeys_partitions)
    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, cfg.size_dev)

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id)
    print("n-shot tracking complete!")

    print("Generating data indexes...")
    data_indexes = build_data_indexes_lepid(
        cids,
        skeys_partitions,
        cid_2_family,
        img_ptrs=img_ptrs_all,
    )
    data_indexes_dev = build_data_indexes_lepid(
        cids,
        skeys_partitions_dev,
        cid_2_family,
        img_ptrs=img_ptrs_all,
    )
    print("Data indexes complete!")

    print("Generating class counts for train partition...")
    class_counts_train = build_class_counts_train(data_indexes)
    class_counts_train_dev = build_class_counts_train(data_indexes_dev)
    print("Class counts complete!")

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

    print("Generating OOD distribution plots...")
    generate_ood_distribution_plots(
        genus_2_cids,
        cids_id,
        cids_ood_val,
        cids_ood_test,
        dpath_figs,
    )
    print("OOD distribution plots complete!")

    print("Generating ID distribution plots...")
    generate_id_distribution_plots(
        cids_id_multis,
        cid_2_skeys_id_multis,
        n_samps_dict,
        skeys_partitions,
        dpath_figs,
    )
    print("ID distribution plots complete!")

    print("Generating split stats table...")
    generate_basic_split_stats_table(
        skeys_partitions=skeys_partitions,
        dpath_figs=dpath_figs,
        n_cids_total=len(cids),
        title="Split Stats (Lepidoptera)",
    )
    print("Split stats table complete!")

    print("Generating n-shot tracking stats table...")
    generate_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")


def main():
    print("Generating split...")
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()