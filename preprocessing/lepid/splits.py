"""
python -m preprocessing.lepid.splits
"""

from utils.utils import paths, seed_libs, load_pickle
from utils.config import get_config_splits
from preprocessing.common.splits import (
    build_genus_2_sids,
    build_n_insts_2_classes_g,
    build_ood_partitions,
    build_id_partitions,
    build_id_eval_nshot,
    build_class_counts_train,
    build_dev_skeys_partitions,
    save_split,
    generate_id_distribution_plots,
    generate_n_shot_table,
)
from preprocessing.lepid.splits_utils import (
    build_data_indexes_lepid,
    build_img_ptrs_lepid,
    build_ood_family_partitions,
    build_ood_genus_partitions,
    build_sid_2_samp_idxs_lepid,
    generate_ood_distribution_plots_lepid,
    generate_split_stats_table_lepid,
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

    if cfg.ood_family_name is None:
        raise ValueError("config.splits.ood_family_name must be set for Lepid split generation")

    print("Loading phylogenetic covariance structure...")
    pvcv = PhyloVCV(dataset=DATASET)
    sids = pvcv.get_sids()
    print(f"Loaded phylogeny with {len(sids):,} species.")

    print("Loading class data...")
    class_data = load_pickle(paths["metadata"][DATASET] / "class_data.pkl")
    sid_2_family = {
        sid: class_data[sid]["family"]
        for sid in sids
    }
    print("Class data loaded.")

    print("Indexing Lepid image paths...")
    img_ptrs_all = build_img_ptrs_lepid(sids, sid_2_family)
    print("Image path indexing complete.")

    print("Applying sample-level position filter...")
    sid_2_samp_idxs = build_sid_2_samp_idxs_lepid(
        sids,
        sid_2_family,
        pos_filter=cfg.pos_filter,
        img_ptrs=img_ptrs_all,
    )
    print("Position filtering complete.")

    sids_dropped = [sid for sid in sorted(sids) if len(sid_2_samp_idxs[sid]) == 0]
    if sids_dropped:
        print(f"Dropping {len(sids_dropped)} species with no samples matching pos_filter={cfg.pos_filter!r}.")

    sids = [sid for sid in sorted(sids) if len(sid_2_samp_idxs[sid]) > 0]
    if not sids:
        raise ValueError(f"No samples available after applying pos_filter={cfg.pos_filter!r}.")

    sid_2_family = {
        sid: sid_2_family[sid]
        for sid in sids
    }
    n_samps_dict = {sid: len(sid_2_samp_idxs[sid]) for sid in sids}
    n_samps_total_all = sum(n_samps_dict.values())

    print("Constructing OOD family partitions...")
    sids_after_family, _, _, skeys_ood_family_val, skeys_ood_family_test = build_ood_family_partitions(
        sids,
        sid_2_family,
        sid_2_samp_idxs,
        cfg,
    )
    print("OOD family partitions complete!")

    print("Constructing OOD genus partitions...")
    n_samps_dict_after_family = {sid: n_samps_dict[sid] for sid in sids_after_family}
    sids_after_genus, sids_ood_genus_val, sids_ood_genus_test, skeys_ood_genus_val, skeys_ood_genus_test = build_ood_genus_partitions(
        sids_after_family,
        sid_2_family,
        sid_2_samp_idxs,
        n_samps_dict_after_family,
        cfg,
        n_samps_total_target=n_samps_total_all,
    )
    print("OOD genus partitions complete!")

    genus_2_sids = build_genus_2_sids(sids_after_genus)
    n_insts_2_classes_g = build_n_insts_2_classes_g(sids_after_genus)

    print("Constructing OOD species partitions...")
    n_samps_dict_after_genus = {sid: n_samps_dict[sid] for sid in sids_after_genus}
    sids_id, sids_ood_val, sids_ood_test, skeys_ood_val, skeys_ood_test = build_ood_partitions(
        n_insts_2_classes_g,
        genus_2_sids,
        set(sids_after_genus),
        sid_2_samp_idxs,
        n_samps_dict_after_genus,
        cfg,
        n_samps_total_target=n_samps_total_all,
    )
    print("OOD species partitions complete!")

    print("Constructing ID partitions...")
    skeys_train, skeys_id_val, skeys_id_test, sid_2_skeys_id, sid_2_skeys_id_multis, sids_id_multis = build_id_partitions(
        sids_id,
        sid_2_samp_idxs,
        n_samps_dict_after_genus,
        cfg,
        n_samps_total_target=n_samps_total_all,
    )
    print("ID partitions complete!")

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
        "ood_genus_val": skeys_ood_genus_val,
        "ood_genus_test": skeys_ood_genus_test,
        "ood_family_val": skeys_ood_family_val,
        "ood_family_test": skeys_ood_family_test,
    }
    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, cfg.size_dev)

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, sids_id, skeys_partitions, sid_2_skeys_id)
    print("n-shot tracking complete!")

    print("Generating data indexes...")
    data_indexes = build_data_indexes_lepid(
        sids,
        skeys_partitions,
        sid_2_family,
        img_ptrs=img_ptrs_all,
    )
    data_indexes_dev = build_data_indexes_lepid(
        sids,
        skeys_partitions_dev,
        sid_2_family,
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
    generate_ood_distribution_plots_lepid(
        sid_2_family,
        sids_after_family,
        sids_after_genus,
        genus_2_sids,
        sids_id,
        sids_ood_val,
        sids_ood_test,
        sids_ood_genus_val,
        sids_ood_genus_test,
        dpath_figs,
    )
    print("OOD distribution plots complete!")

    print("Generating ID distribution plots...")
    generate_id_distribution_plots(
        sids_id_multis,
        sid_2_skeys_id_multis,
        n_samps_dict_after_genus,
        skeys_partitions,
        dpath_figs,
    )
    print("ID distribution plots complete!")

    print("Generating split stats table...")
    generate_split_stats_table_lepid(
        sid_2_family,
        skeys_partitions,
        dpath_figs,
        len(sids),
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