"""
python -m preprocessing.cub.split_gen
"""

from scipy.io import loadmat
import numpy as np

from preprocessing.common.split_gen import (
    GenSplitDataManager,
    build_dev_skeys_partitions,
    build_penult_2_cids,
    add_trainval_whole,
    strat_sample_ood_id,
    generate_splits,
)
from preprocessing.cub.split_gen_utils import build_img_ptrs, normalize_cub_rfpath
from utils.utils import paths

import pdb


def build_splits() -> None:
    GenSplitDataManager.setup("cub")
    cids = GenSplitDataManager.get_cids()

    fpath_att_splits = paths["data"]["cub"] / "xlsa17/data/CUB/att_splits.mat"
    split_sets = loadmat(fpath_att_splits)

    idxs_pool = (split_sets["trainval_loc"] - 1).squeeze()
    idxs_test_id = (split_sets["test_seen_loc"] - 1).squeeze()
    idxs_test_ood = (split_sets["test_unseen_loc"] - 1).squeeze()

    fpath_res = paths["data"]["cub"] / "xlsa17/data/CUB/res101.mat"
    data = loadmat(fpath_res)
    rfpaths_whole = np.array(
        [normalize_cub_rfpath(item[0][0]) for item in data["image_files"]],
        dtype=str,
    )

    rfpaths_pool = rfpaths_whole[idxs_pool]
    rfpaths_test_id = rfpaths_whole[idxs_test_id]
    rfpaths_test_ood = rfpaths_whole[idxs_test_ood]

    img_ptrs, rfpath_2_skey = build_img_ptrs(rfpaths_whole)
    
    cid_2_n_samps = {cid: len(img_ptrs[cid]) for cid in cids}

    n_cids_whole = len(cids)
    n_samps_whole = sum(cid_2_n_samps.values())

    cid_2_penult = {cid: GenSplitDataManager.class_data[cid]["genus"] for cid in cids}
    penult_2_cids = build_penult_2_cids(cids, cid_2_penult)

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
        ood_tol_flag=False,
    )
    print("OOD + ID validation partitions complete!")

    #############################################################################

    # PARTITION SKEYS (SAMPLE-KEYS)

    skeys_pts = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }
    add_trainval_whole(skeys_pts)
    skeys_pts_dev = build_dev_skeys_partitions(skeys_pts)

    generate_splits(
        cids,
        skeys_pts,
        skeys_pts_dev,
        img_ptrs,
        penult_2_cids,
        cid_2_n_samps,
    )

def main():
    build_splits()


if __name__ == "__main__":
    main()