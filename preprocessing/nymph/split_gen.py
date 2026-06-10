"""
python -m preprocessing.nymph.split_gen
"""

from utils.utils import paths, seed_libs, load_pickle
from utils.config import get_config_splits
from preprocessing.common.split_gen import (
    build_penult_2_cids,
    strat_sample_ood_id,
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
from preprocessing.nymph.split_gen_utils import (
    build_data_indexes,
    build_img_ptrs,
    build_cid_2_samp_idxs,
)
from utils.phylo import PhyloVCV

import pdb


DATASET_NAME = "nymph"


def build_splits():
    cfg = get_config_splits()
    print(f"Generating split: '{cfg.split}'...")
    seed_libs(cfg.seed, seed_torch=False)
    class_data = load_pickle(paths["metadata"][DATASET_NAME] / "class_data.pkl")

    dpath_split = paths["metadata"][DATASET_NAME] / f"splits/{cfg.split}"
    dpath_figs = dpath_split / "figures"
    dpath_split_dev = paths["metadata"][DATASET_NAME] / "splits/dev"
    dpath_figs_dev = dpath_split_dev / "figures"

    pvcv = PhyloVCV(dataset=DATASET_NAME)
    cids = pvcv.get_cids()  # OOD partitions: insts

    img_ptrs = build_img_ptrs(cids)
    cid_2_samp_idxs = build_cid_2_samp_idxs(
        cids,
        img_ptrs,
        pos_filter=cfg.pos_filter,
    )

    # NYMPH-SPECIFIC
    cids_dropped = [cid for cid in sorted(cids) if len(cid_2_samp_idxs[cid]) == 0]
    if cids_dropped:
        print(f"Dropping {len(cids_dropped)} species with no samples matching pos_filter={cfg.pos_filter!r}.")
    cids = [cid for cid in sorted(cids) if len(cid_2_samp_idxs[cid]) > 0]  # filter cids

    cid_2_n_samps = {cid: len(cid_2_samp_idxs[cid]) for cid in cids}
    
    cid_2_penult = {cid: class_data[cid]["genus"] for cid in cids}
    penult_2_cids = build_penult_2_cids(cids, cid_2_penult)  # OOD partitions: class_2_insts

    skeys_pool = {(cid, samp_idx) for cid, samp_idxs in cid_2_samp_idxs.items() for samp_idx in samp_idxs}
    n_cids_whole = len(cids)
    n_samps_whole = len(skeys_pool)

    # TEST PARTITIONS

    print("Constructing OOD + ID test partitions...")
    skeys_ood_test, skeys_id_test, skeys_pool = strat_sample_ood_id(
        skeys_pool,
        n_cids_whole,
        n_samps_whole,
        cid_2_penult,
        cfg,
    )
    print("OOD + ID test partitions complete!")

    # VALIDATION PARTITIONS

    print("Constructing OOD + ID validation partitions...")
    skeys_ood_val, skeys_id_val, skeys_train = strat_sample_ood_id(
        skeys_pool,
        n_cids_whole,
        n_samps_whole,
        cid_2_penult,
        cfg,
    )
    print("OOD + ID validation partitions complete!")

    # PARTITION SKEYS (SAMPLE-KEYS)

    skeys_pts = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }
    skeys_pts["trainval"] = build_trainval_skeys_partition(skeys_pts)
    skeys_pts["whole"] = (
        skeys_pts["train"]
        | skeys_pts["id_val"]
        | skeys_pts["id_test"]
        | skeys_pts["ood_val"]
        | skeys_pts["ood_test"]
    )
    skeys_pts_dev = build_dev_skeys_partitions(skeys_pts, cfg.size_dev)

    # N-SHOT TRACKING

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(skeys_pts, cfg)
    print("n-shot tracking complete!")

    # GENERATE DATA INDEXES

    print("Generating data indexes...")
    cid2enc = build_global_cid2enc(skeys_pts)
    data_indexes = build_data_indexes(cids, skeys_pts, cid2enc)
    data_indexes_dev = build_data_indexes(cids, skeys_pts_dev, cid2enc)
    print("Data indexes complete!")

    # CLASS COUNTS

    print("Generating class counts for train partitions...")
    class_counts = build_class_counts_by_partition(data_indexes, len(cid2enc))
    class_counts_dev = build_class_counts_by_partition(data_indexes_dev, len(cid2enc))
    print("Class counts complete!")

    # TRAIN PARTITIONS NORMALIZATION STATS

    norm_stats = get_norm_stats(data_indexes, dataset=DATASET_NAME, cfg=cfg)

    # SAVE SPLIT

    print("Saving Split...")
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
        skeys_pts,
        dpath_figs,
    )
    print("OOD stratified sampling distribution plots complete!")

    # ID STRATIFIED SAMPLING DISTRIBUTION PLOTTING (singletons omitted)

    print("Generating ID stratified sampling distribution plots...")
    gen_strat_sampling_dist_plots_id(
        cid_2_n_samps, 
        skeys_pts, 
        dpath_figs,
    )
    print("ID stratified sampling distribution plots complete!")

    # PARTITION SUMMARY TABLE

    print("Generating partition summary table...")
    generate_partition_summary_table(
        skeys_pts=skeys_pts,
        dpath_figs=dpath_figs,
        n_cids_total=len(cids),
        title="Nymphalidae Partition Summary",
    )
    print("Partition summary table complete!")

    # N-SHOT BUCKET SUMMARY TABLE

    print("Generating n-shot bucket summary table...")
    generate_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")

def main():
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()