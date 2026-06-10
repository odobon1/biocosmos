"""
python -m preprocessing.bryo.split_gen
"""

from utils.utils import paths, seed_libs, load_pickle
from utils.config import get_config_splits
from preprocessing.common.split_gen import (
    build_penult_2_cids,
    build_n_insts_2_classes_penult,
    build_ood_partitions,
    build_id_partitions,
    build_trainval_skeys_partition,
    build_id_eval_nshot,
    build_class_counts_by_partition,
    build_dev_skeys_partitions,
    build_global_cid2enc,
    save_split,
    gen_strat_sampling_dist_plots_ood,
    gen_strat_sampling_dist_plots_id,
    generate_n_shot_table,
    generate_partition_summary_table,
    get_norm_stats,
)
from preprocessing.bryo.split_gen_utils import (
    build_data_indexes_bryo,
    build_img_ptrs_bryo,
)
from utils.phylo import PhyloVCV

import pdb


DATASET_NAME = "bryo"


def build_splits() -> None:
    cfg = get_config_splits()
    seed_libs(cfg.seed, seed_torch=False)
    dpath_split = paths["metadata"][DATASET_NAME] / f"splits/{cfg.split}"
    dpath_figs = dpath_split / "figures"
    dpath_split_dev = paths["metadata"][DATASET_NAME] / "splits/dev"
    dpath_figs_dev = dpath_split_dev / "figures"
    print(f"Generating split: '{cfg.split}'")

    pvcv = PhyloVCV(dataset=DATASET_NAME)
    cids = pvcv.get_cids()  # genera for bryo

    img_ptrs_all = build_img_ptrs_bryo(cids)

    cids = [cid for cid in sorted(cids) if cid in img_ptrs_all and len(img_ptrs_all[cid]) > 0]
    n_cids = len(cids)

    cid_2_samp_idxs = {cid: list(sorted(img_ptrs_all[cid].keys())) for cid in cids}
    cid_2_n_samps = {cid: len(cid_2_samp_idxs[cid]) for cid in cids}
    class_data = load_pickle(paths["metadata"][DATASET_NAME] / "class_data.pkl")
    cid_2_penult = {cid: class_data[cid]["family"] for cid in cids}
    penult_2_cids = build_penult_2_cids(cids, cid_2_penult)
    n_insts_2_classes_penult = build_n_insts_2_classes_penult(cids, cid_2_penult)

    #####################################################################################################################

    

    #####################################################################################################################

    # OOD PARTITIONS

    print("Constructing OOD partitions...")
    cids_id, cids_ood_val, cids_ood_test, skeys_ood_val, skeys_ood_test = build_ood_partitions(
        n_insts_2_classes_penult,
        penult_2_cids,
        set(cids),
        cid_2_samp_idxs,
        cid_2_n_samps,
        cfg,
    )
    print("OOD partitions complete!")

    # ID PARTITIONS

    cid_2_n_samps_id = {cid: cid_2_n_samps[cid] for cid in sorted(cids_id)}

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
        cid_2_n_samps,  # !
        cfg,
        skeys_id_test_extra=skeys_ood_val,
    )
    skeys_ood_val = skeys_ood_val - skeys_id_test_extra_taken
    print("ID partitions complete!")


    #####################################################################################################################


    # PARTITION SKEYS (SAMPLE-KEYS)

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }
    skeys_partitions["trainval"] = build_trainval_skeys_partition(skeys_partitions)
    skeys_partitions["whole"] = (
        skeys_partitions["train"]
        | skeys_partitions["id_val"]
        | skeys_partitions["id_test"]
        | skeys_partitions["ood_val"]
        | skeys_partitions["ood_test"]
    )
    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, cfg.size_dev)

    # N-SHOT TRACKING

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id)
    print("n-shot tracking complete!")

    # GENERATE DATA INDEXES

    print("Generating data indexes...")
    cid2enc = build_global_cid2enc(skeys_partitions)
    data_indexes = build_data_indexes_bryo(cids, skeys_partitions, cid2enc, img_ptrs=img_ptrs_all)
    data_indexes_dev = build_data_indexes_bryo(cids, skeys_partitions_dev, cid2enc, img_ptrs=img_ptrs_all)
    print("Data indexes complete!")

    # CLASS COUNTS (FOR CLASS IMBALANCE)

    print("Generating class counts by partition...")
    class_counts = build_class_counts_by_partition(data_indexes, len(cid2enc))
    class_counts_dev = build_class_counts_by_partition(data_indexes_dev, len(cid2enc))
    print("Class counts complete!")

    # COMPUTE NORMALIZATION STATS BY PARTITION

    norm_stats = get_norm_stats(data_indexes, dataset=DATASET_NAME, cfg=cfg)

    # SAVE SPLIT

    print("Saving split...")
    save_split(
        data_indexes,
        id_eval_nshot,
        class_counts,
        norm_stats,
        dpath_split,
        dpath_figs,
    )
    save_split(
        data_indexes_dev,
        id_eval_nshot,
        class_counts_dev,
        norm_stats,
        dpath_split_dev,
        dpath_figs_dev,
    )
    print("Primary and dev splits saved!")

    # OOD STRATIFIED SAMPLING DISTRIBUTION PLOTTING

    print("Generating OOD stratified sampling distribution plots...")
    gen_strat_sampling_dist_plots_ood(
        penult_2_cids,
        cids_id,
        cids_ood_val,
        cids_ood_test,
        dpath_figs,
    )
    print("OOD stratified sampling distribution plots complete!")

    # ID STRATIFIED SAMPLING DISTRIBUTION PLOTTING (singletons omitted)

    print("Generating ID stratified sampling distribution plots...")
    gen_strat_sampling_dist_plots_id(
        cids_id_multis,
        cid_2_skeys_id_multis,
        cid_2_n_samps_id,
        skeys_partitions,
        dpath_figs,
    )
    print("ID stratified sampling distribution plots complete!")

    # PARTITION SUMMARY TABLE

    print("Generating partition summary table...")
    generate_partition_summary_table(
        skeys_partitions=skeys_partitions,
        dpath_figs=dpath_figs,
        n_cids_total=n_cids,
        title="Bryozoa Partition Summary",
    )
    print("Partition summary table complete!")

    # N-SHOT BUCKET SUMMARY TABLE

    print("Generating n-shot bucket summary table...")
    generate_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")

def main() -> None:
    print("Generating split...")
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()