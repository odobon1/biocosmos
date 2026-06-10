"""
python -m preprocessing.cub.split_gen
"""

from collections import defaultdict
from scipy.io import loadmat
import numpy as np

from preprocessing.common.split_gen import (
    build_class_counts_by_partition,
    build_dev_skeys_partitions,
    build_global_cid2enc,
    build_id_eval_nshot,
    build_penult_2_cids,
    build_trainval_skeys_partition,
    gen_strat_sampling_dist_plots_id,
    gen_strat_sampling_dist_plots_ood,
    generate_n_shot_table,
    save_split,
    strat_sample_ood_id,
    generate_partition_summary_table,
    get_norm_stats,
)
from preprocessing.cub.split_gen_utils import build_data_indexes_cub
from utils.config import get_config_splits
from utils.utils import load_pickle, paths, seed_libs
from utils.phylo import PhyloVCV

import pdb


DATASET_NAME = "cub"


def _normalize_cub_rfpath(raw_path: str) -> str:
    raw_path = str(raw_path)
    idx_images = raw_path.find("images/")
    return raw_path[idx_images + len("images/"):]

def _class_dir_to_common_name(class_dir: str) -> str:
    _, raw_name = class_dir.split(".", 1)
    return raw_name.lower()

def _build_classdir_to_cid(class_data):
    classdir_to_cid = {}
    for cid, cid_data in class_data.items():
        species = cid_data.get("species")
        common_name = cid_data.get("common_name", cid)
        classdir_to_cid[common_name] = species
    return classdir_to_cid

def _build_img_ptrs(rfpaths_whole, class_data):
    classdir_to_cid = _build_classdir_to_cid(class_data)

    img_ptrs = defaultdict(dict)
    cid_2_samp_idxs = defaultdict(list)
    rfpath_2_skey = {}
    cid_offsets = defaultdict(int)

    for rfpath in rfpaths_whole:
        parts = rfpath.split("/")
        class_dir = parts[0]
        class_name = _class_dir_to_common_name(class_dir)

        cid = classdir_to_cid[class_name]
        samp_idx = cid_offsets[cid]
        cid_offsets[cid] += 1

        img_ptrs[cid][samp_idx] = rfpath
        cid_2_samp_idxs[cid].append(samp_idx)
        rfpath_2_skey[rfpath] = (cid, samp_idx)

    cids = sorted(img_ptrs.keys())
    cid_2_n_samps = {
        cid: len(cid_2_samp_idxs[cid])
        for cid in cids
    }

    return img_ptrs, rfpath_2_skey, cid_2_n_samps

def build_splits() -> None:
    cfg = get_config_splits()
    print(f"Generating split: '{cfg.split}'...")
    seed_libs(cfg.seed, seed_torch=False)
    class_data = load_pickle(paths["metadata"][DATASET_NAME] / "class_data.pkl")

    dpath_split = paths["metadata"][DATASET_NAME] / f"splits/{cfg.split}"
    dpath_figs = dpath_split / "figures"
    dpath_split_dev = paths["metadata"][DATASET_NAME] / "splits/dev"
    dpath_figs_dev = dpath_split_dev / "figures"

    pvcv = PhyloVCV(dataset=DATASET_NAME)
    cids = pvcv.get_cids()

    fpath_att_splits = paths["data"][DATASET_NAME] / "xlsa17/data/CUB/att_splits.mat"
    split_sets = loadmat(fpath_att_splits)

    idxs_pool = (split_sets["trainval_loc"] - 1).squeeze()
    idxs_test_id = (split_sets["test_seen_loc"] - 1).squeeze()
    idxs_test_ood = (split_sets["test_unseen_loc"] - 1).squeeze()

    fpath_res = paths["data"][DATASET_NAME] / "xlsa17/data/CUB/res101.mat"
    data = loadmat(fpath_res)
    rfpaths_whole = np.array(
        [_normalize_cub_rfpath(item[0][0]) for item in data["image_files"]],
        dtype=str,
    )

    rfpaths_pool = rfpaths_whole[idxs_pool]
    rfpaths_test_id = rfpaths_whole[idxs_test_id]
    rfpaths_test_ood = rfpaths_whole[idxs_test_ood]

    img_ptrs, rfpath_2_skey, cid_2_n_samps = _build_img_ptrs(rfpaths_whole, class_data)

    n_cids_whole = len(cids)
    n_samps_whole = sum(cid_2_n_samps.values())

    cid_2_penult = {cid: class_data[cid]["genus"] for cid in cids}

    # TEST PARTITIONS

    skeys_pool = {rfpath_2_skey[rfpath] for rfpath in rfpaths_pool}  # trainval skeys
    skeys_id_test = {rfpath_2_skey[rfpath] for rfpath in rfpaths_test_id}
    skeys_ood_test = {rfpath_2_skey[rfpath] for rfpath in rfpaths_test_ood}

    # VALIDATION PARTITIONS

    print("Constructing OOD + ID validation partitions...")
    skeys_ood_val, skeys_id_val, skeys_train = strat_sample_ood_id(
        skeys_pool,
        n_cids_whole,
        n_samps_whole,
        cid_2_penult,
        cfg,
        ood_tol_flag=False,
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

    penult_2_cids = build_penult_2_cids(cids, cid_2_penult)

    # N-SHOT TRACKING

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(skeys_pts, cfg)
    print("n-shot tracking complete!")

    # GENERATE DATA INDEXES

    print("Generating data indexes...")
    cid2enc = build_global_cid2enc(skeys_pts)
    data_indexes = build_data_indexes_cub(skeys_pts, img_ptrs, cid2enc)
    data_indexes_dev = build_data_indexes_cub(skeys_pts_dev, img_ptrs, cid2enc)
    print("Data indexes complete!")

    # CLASS COUNTS

    print("Generating class counts for train partitions...")
    class_counts = build_class_counts_by_partition(data_indexes, len(cid2enc))
    class_counts_dev = build_class_counts_by_partition(data_indexes_dev, len(cid2enc))
    print("Class counts complete!")

    # TRAIN PARTITIONS NORMALIZATION STATS

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
        title="CUB Partition Summary",
    )
    print("Partition summary table complete!")

    # N-SHOT BUCKET SUMMARY TABLE

    print("Generating n-shot bucket summary table...")
    generate_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")

def main() -> None:
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()