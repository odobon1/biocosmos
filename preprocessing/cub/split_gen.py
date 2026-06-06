"""
python -m preprocessing.cub.split_gen
"""

from collections import defaultdict
from scipy.io import loadmat
import numpy as np

from preprocessing.common.split_gen import (
    build_class_counts_by_partition,
    build_dev_skeys_partitions,
    build_id_eval_nshot,
    build_trainval_skeys_partition,
    generate_n_shot_table,
    save_split,
    strat_split,
    generate_partition_summary_table,
    get_norm_stats,
)
from preprocessing.cub.split_gen_utils import build_data_indexes_cub
from utils.config import get_config_splits
from utils.utils import load_pickle, paths, seed_libs


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

def _build_img_ptrs(index_rfpaths_all, class_data):
    classdir_to_cid = _build_classdir_to_cid(class_data)

    img_ptrs = defaultdict(dict)
    cid_2_samp_idxs = defaultdict(list)
    rfpath_2_skey = {}
    cid_offsets = defaultdict(int)

    for rfpath in index_rfpaths_all:
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

    return img_ptrs, cid_2_samp_idxs, rfpath_2_skey, cids, cid_2_n_samps

def _choose_ood_val_cids(
    cids_train,
    n_cids_total_target,
    cfg,
):
    """Pick exactly round(n_cids_total_target * pct_partition) species for OOD-val."""
    n_cids_ood_val_target = round(n_cids_total_target * cfg.pct_partition)

    if n_cids_ood_val_target == 0:
        return set()

    rng = np.random.default_rng(cfg.seed)
    cids_train_sorted = sorted(cids_train)
    return set(rng.choice(cids_train_sorted, size=n_cids_ood_val_target, replace=False).tolist())

def _split_train_into_train_idval_oodval(
    skeys_train_pool,
    n_cids_total_target,
    n_samps_total_target,
    cfg,
):
    cid_2_skeys_train = defaultdict(list)
    for cid, samp_idx in sorted(skeys_train_pool):
        cid_2_skeys_train[cid].append((cid, samp_idx))

    cids_train = set(cid_2_skeys_train.keys())

    cids_ood_val = _choose_ood_val_cids(
        cids_train=cids_train,
        n_cids_total_target=n_cids_total_target,
        cfg=cfg,
    )

    skeys_ood_val = {
        skey
        for cid in cids_ood_val
        for skey in cid_2_skeys_train[cid]
    }

    skeys_id_pool = skeys_train_pool - skeys_ood_val
    cid_2_skeys_id_pool = defaultdict(list)
    for cid, samp_idx in sorted(skeys_id_pool):
        cid_2_skeys_id_pool[cid].append((cid, samp_idx))

    cids_id_pool = set(cid_2_skeys_id_pool.keys())
    cids_id_singles = {
        cid
        for cid in sorted(cids_id_pool)
        if len(cid_2_skeys_id_pool[cid]) == 1
    }
    cids_id_multis = cids_id_pool - cids_id_singles

    skeys_id_multis = {
        skey
        for cid in cids_id_multis
        for skey in cid_2_skeys_id_pool[cid]
    }

    n_samps_id_val_target = round(n_samps_total_target * cfg.pct_partition)

    if n_samps_id_val_target == 0:
        skeys_id_val = set()
        skeys_train_multis = skeys_id_multis
    else:
        n_insts_2_classes_s = defaultdict(list)
        cid_2_skeys_id_multis = defaultdict(list)

        for cid in sorted(cids_id_multis):
            cid_skeys = list(sorted(cid_2_skeys_id_pool[cid]))
            cid_2_skeys_id_multis[cid] = cid_skeys
            n_insts_2_classes_s[len(cid_skeys)].append(cid)

        pct_id_val = n_samps_id_val_target / len(skeys_id_multis)

        skeys_train_multis, skeys_id_val_a, skeys_id_val_b = strat_split(
            n_classes=len(cids_id_multis),
            n_draws=n_samps_id_val_target,
            pct_eval=pct_id_val,
            n_insts_2_classes=n_insts_2_classes_s,
            class_2_insts=cid_2_skeys_id_multis,
            insts=skeys_id_multis,
            seed=cfg.seed,
        )
        skeys_id_val = skeys_id_val_a | skeys_id_val_b

        # Keep at least one train sample for each multi-instance ID cid.
        for cid in sorted(cids_id_multis):
            cid_skeys = set(cid_2_skeys_id_multis[cid])
            if cid_skeys & skeys_train_multis:
                continue

            cid_skeys_val = sorted(cid_skeys & skeys_id_val)

            skey_move = cid_skeys_val[0]
            skeys_id_val.remove(skey_move)
            skeys_train_multis.add(skey_move)

        # !
        assert len(skeys_id_val) == n_samps_id_val_target

    skeys_id_singles = {
        cid_2_skeys_id_pool[cid][0]
        for cid in sorted(cids_id_singles)
    }
    skeys_train = skeys_train_multis | skeys_id_singles

    # !
    assert skeys_train | skeys_id_val | skeys_ood_val == skeys_train_pool
    assert skeys_train & skeys_id_val == set()
    assert skeys_train & skeys_ood_val == set()
    assert skeys_id_val & skeys_ood_val == set()

    return skeys_train, skeys_id_val, skeys_ood_val

def _build_cid_2_skeys_id(skeys_train, skeys_id_val, skeys_id_test):
    cid_2_skeys_id = defaultdict(list)
    skeys_id_all = skeys_train | skeys_id_val | skeys_id_test
    for cid, samp_idx in sorted(skeys_id_all):
        cid_2_skeys_id[cid].append((cid, samp_idx))
    return cid_2_skeys_id

def build_splits() -> None:
    cfg = get_config_splits()
    seed_libs(cfg.seed, seed_torch=False)

    dpath_split = paths["metadata"][DATASET_NAME] / f"splits/{cfg.split}"
    dpath_figs = dpath_split / "figures"
    dpath_split_dev = paths["metadata"][DATASET_NAME] / "splits/dev"
    dpath_figs_dev = dpath_split_dev / "figures"

    print(f"Generating split: '{cfg.split}'")

    class_data = load_pickle(paths["metadata"][DATASET_NAME] / "class_data.pkl")

    fpath_att_splits = paths["data"][DATASET_NAME] / "xlsa17/data/CUB/att_splits.mat"
    split_sets = loadmat(fpath_att_splits)

    idxs_train = (split_sets["trainval_loc"] - 1).squeeze()
    idxs_test_id = (split_sets["test_seen_loc"] - 1).squeeze()
    idxs_test_ood = (split_sets["test_unseen_loc"] - 1).squeeze()

    fpath_res = paths["data"][DATASET_NAME] / "xlsa17/data/CUB/res101.mat"
    data = loadmat(fpath_res)
    index_rfpaths_all = np.array(
        [_normalize_cub_rfpath(item[0][0]) for item in data["image_files"]],
        dtype=str,
    )

    index_rfpaths_train = index_rfpaths_all[idxs_train]
    index_rfpaths_test_id = index_rfpaths_all[idxs_test_id]
    index_rfpaths_test_ood = index_rfpaths_all[idxs_test_ood]

    img_ptrs, _, rfpath_2_skey, cids, cid_2_n_samps = _build_img_ptrs(index_rfpaths_all, class_data)

    skeys_train_pool = {rfpath_2_skey[rfpath] for rfpath in index_rfpaths_train}
    skeys_id_test = {rfpath_2_skey[rfpath] for rfpath in index_rfpaths_test_id}
    skeys_ood_test = {rfpath_2_skey[rfpath] for rfpath in index_rfpaths_test_ood}

    # !
    assert skeys_train_pool & skeys_id_test == set()
    assert skeys_train_pool & skeys_ood_test == set()
    assert skeys_id_test & skeys_ood_test == set()

    n_cids_total_target = len(cids)
    n_samps_total_target = sum(cid_2_n_samps.values())

    print("Constructing train/ID-val/OOD-val from train pool...")
    skeys_train, skeys_id_val, skeys_ood_val = _split_train_into_train_idval_oodval(
        skeys_train_pool=skeys_train_pool,
        n_cids_total_target=n_cids_total_target,
        n_samps_total_target=n_samps_total_target,
        cfg=cfg,
    )
    print("Train/validation partitioning complete!")

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }
    skeys_partitions["trainval"] = build_trainval_skeys_partition(skeys_partitions)

    all_skeys = set()
    for partition, skeys in skeys_partitions.items():
        if partition == "trainval":
            continue
        
        # !
        assert all_skeys & skeys == set()

        all_skeys.update(skeys)

    # !
    skeys_expected = set(rfpath_2_skey.values())
    assert all_skeys == skeys_expected

    skeys_partitions["whole"] = (
        skeys_partitions["train"]
        | skeys_partitions["id_val"]
        | skeys_partitions["id_test"]
        | skeys_partitions["ood_val"]
        | skeys_partitions["ood_test"]
    )
    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, cfg.size_dev)

    cid_2_skeys_id = _build_cid_2_skeys_id(skeys_train, skeys_id_val, skeys_id_test)
    cids_id = {cid for cid, _ in skeys_train}

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id)
    print("n-shot tracking complete!")

    print("Generating data indexes...")
    data_indexes = build_data_indexes_cub(skeys_partitions, img_ptrs)
    data_indexes_dev = build_data_indexes_cub(skeys_partitions_dev, img_ptrs)
    print("Data indexes complete!")

    print("Generating class counts by partition...")
    class_counts = build_class_counts_by_partition(data_indexes)
    class_counts_dev = build_class_counts_by_partition(data_indexes_dev)
    print("Class counts complete!")

    norm_stats = get_norm_stats(data_indexes, dataset=DATASET_NAME, cfg=cfg)

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

    # PARTITION SUMMARY TABLE

    print("Generating partition summary table...")
    generate_partition_summary_table(
        skeys_partitions=skeys_partitions,
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
    print("Generating split...")
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()