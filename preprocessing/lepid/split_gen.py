"""
python -m preprocessing.lepid.split_gen
"""

from preprocessing.common.split_gen import (
    GenSplitDataManager,
    build_penult_2_cids,
    strat_sample_ood_id,
    add_trainval,
    build_dev_skeys_partitions,
    generate_splits,
)
from preprocessing.lepid.split_gen_utils import build_img_ptrs


def build_splits():
    GenSplitDataManager.setup("lepid")
    cids = GenSplitDataManager.get_cids()

    img_ptrs = build_img_ptrs(cids)
    cids = [cid for cid in sorted(cids) if len(img_ptrs[cid]) > 0]  # filter cids

    cid_2_n_samps = {cid: len(img_ptrs[cid]) for cid in cids}

    cid_2_penult = {cid: GenSplitDataManager.class_data[cid]["genus"] for cid in cids}
    penult_2_cids = build_penult_2_cids(cids, cid_2_penult)

    skeys_pool = {(cid, samp_idx) for cid in cids for samp_idx in img_ptrs[cid]}
    n_cids_all = len(cids)
    n_samps_all = len(skeys_pool)

    # TEST PARTITIONS

    print("Constructing OOD + ID test partitions...")
    skeys_test_ood, skeys_test_id, skeys_pool = strat_sample_ood_id(
        skeys_pool,
        n_cids_all,
        n_samps_all,
        cid_2_penult,
    )
    print("OOD + ID test partitions complete!")

    # VALIDATION PARTITIONS

    print("Constructing OOD + ID validation partitions...")
    skeys_val_ood, skeys_val_id, skeys_train = strat_sample_ood_id(
        skeys_pool,
        n_cids_all,
        n_samps_all,
        cid_2_penult,
    )
    print("OOD + ID validation partitions complete!")

    #############################################################################

    # PARTITION SKEYS (SAMPLE-KEYS)

    skeys_pts = {
        "train": skeys_train,
        "val_id": skeys_val_id,
        "test_id": skeys_test_id,
        "val_ood": skeys_val_ood,
        "test_ood": skeys_test_ood,
    }
    add_trainval(skeys_pts)
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
