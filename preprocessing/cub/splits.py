"""
python -m preprocessing.cub.splits
"""

from collections import defaultdict
from scipy.io import loadmat
import numpy as np

from preprocessing.common.splits import (
    build_class_counts_train,
    build_dev_skeys_partitions,
    build_id_eval_nshot,
    build_trainval_skeys_partition,
    generate_n_shot_table,
    save_split,
    strat_split,
    generate_basic_split_stats_table,
)
from preprocessing.cub.splits_utils import build_data_indexes_cub
from utils.config import get_config_splits
from utils.utils import load_pickle, paths, seed_libs


DATASET = "cub"


def _normalize_cub_rfpath(raw_path: str) -> str:
    raw_path = str(raw_path)
    idx_images = raw_path.find("images/")
    if idx_images == -1:
        raise ValueError(f"Could not parse CUB rfpath from raw image path: {raw_path}")
    return raw_path[idx_images + len("images/"):]

def _class_dir_to_common_name(class_dir: str) -> str:
    if not isinstance(class_dir, str) or "." not in class_dir:
        raise ValueError(f"Invalid CUB class directory: {class_dir}")
    _, raw_name = class_dir.split(".", 1)
    return raw_name.lower()

def _build_classdir_to_cid(class_data):
    classdir_to_cid = {}
    for cid, cid_data in class_data.items():
        species = cid_data.get("species")
        if not isinstance(species, str) or not species:
            raise ValueError(f"Invalid species for cid='{cid}': {species}")
        common_name = cid_data.get("common_name", cid)
        if not isinstance(common_name, str) or not common_name:
            raise ValueError(f"Invalid common_name for cid='{cid}': {common_name}")
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
        if len(parts) < 2:
            raise ValueError(f"Unexpected CUB rfpath format: {rfpath}")
        class_dir = parts[0]
        class_name = _class_dir_to_common_name(class_dir)
        if class_name not in classdir_to_cid:
            raise KeyError(f"CUB class directory '{class_dir}' missing from class_data mapping")

        cid = classdir_to_cid[class_name]
        samp_idx = cid_offsets[cid]
        cid_offsets[cid] += 1

        img_ptrs[cid][samp_idx] = rfpath
        cid_2_samp_idxs[cid].append(samp_idx)
        rfpath_2_skey[rfpath] = (cid, samp_idx)

    cids = sorted(img_ptrs.keys())
    n_samps_dict = {
        cid: len(cid_2_samp_idxs[cid])
        for cid in cids
    }

    return img_ptrs, cid_2_samp_idxs, rfpath_2_skey, cids, n_samps_dict

def _build_skeys_from_rfpaths(index_rfpaths, rfpath_2_skey, partition_name: str):
    skeys = set()
    for rfpath in index_rfpaths:
        if rfpath not in rfpath_2_skey:
            raise KeyError(f"rfpath '{rfpath}' from partition '{partition_name}' not found in global lookup")
        skeys.add(rfpath_2_skey[rfpath])

    if len(skeys) != len(index_rfpaths):
        raise ValueError(f"Duplicate rfpaths detected in partition '{partition_name}'")

    return skeys

def _choose_ood_val_cids(
    cids_train,
    n_cids_total_target,
    cfg,
):
    """Pick exactly round(n_cids_total_target * pct_partition) species for OOD-val."""
    n_cids_ood_val_target = round(n_cids_total_target * cfg.pct_partition)

    if n_cids_ood_val_target > len(cids_train):
        raise ValueError(
            f"Target OOD-val cid count ({n_cids_ood_val_target}) exceeds available train cid count ({len(cids_train)})"
        )

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
    if n_samps_id_val_target > len(skeys_id_multis):
        raise ValueError(
            f"Target ID-val sample count ({n_samps_id_val_target}) exceeds available ID multis pool ({len(skeys_id_multis)})"
        )

    if n_samps_id_val_target == 0:
        skeys_id_val = set()
        skeys_train_multis = set(skeys_id_multis)
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
            insts=set(skeys_id_multis),
            seed=cfg.seed,
        )
        skeys_id_val = set(skeys_id_val_a).union(skeys_id_val_b)

        # Keep at least one train sample for each multi-instance ID cid.
        for cid in sorted(cids_id_multis):
            cid_skeys = set(cid_2_skeys_id_multis[cid])
            if cid_skeys.intersection(skeys_train_multis):
                continue

            cid_skeys_val = sorted(cid_skeys.intersection(skeys_id_val))
            if not cid_skeys_val:
                raise ValueError(f"Could not preserve train coverage for cid '{cid}'")

            skey_move = cid_skeys_val[0]
            skeys_id_val.remove(skey_move)
            skeys_train_multis.add(skey_move)

        if len(skeys_id_val) != n_samps_id_val_target:
            raise ValueError(
                f"ID-val sample count drifted after coverage adjustment: got {len(skeys_id_val)}, "
                f"expected {n_samps_id_val_target}"
            )

    skeys_id_singles = {
        cid_2_skeys_id_pool[cid][0]
        for cid in sorted(cids_id_singles)
    }
    skeys_train = set(skeys_train_multis).union(skeys_id_singles)

    if skeys_train.union(skeys_id_val).union(skeys_ood_val) != set(skeys_train_pool):
        raise ValueError("train/id_val/ood_val partitions do not fully cover train pool")

    if skeys_train.intersection(skeys_id_val) or skeys_train.intersection(skeys_ood_val) or skeys_id_val.intersection(skeys_ood_val):
        raise ValueError("train/id_val/ood_val partitions are not mutually disjoint")

    return skeys_train, skeys_id_val, skeys_ood_val

def _build_cid_2_skeys_id(skeys_train, skeys_id_val, skeys_id_test):
    cid_2_skeys_id = defaultdict(list)
    skeys_id_all = set(skeys_train).union(skeys_id_val).union(skeys_id_test)
    for cid, samp_idx in sorted(skeys_id_all):
        cid_2_skeys_id[cid].append((cid, samp_idx))
    return cid_2_skeys_id

def build_splits() -> None:
    cfg = get_config_splits()
    seed_libs(cfg.seed, seed_torch=False)

    dpath_split = paths["metadata"][DATASET] / f"splits/{cfg.split_name}"
    dpath_figs = dpath_split / "figures"
    dpath_split_dev = paths["metadata"][DATASET] / "splits/dev"
    dpath_figs_dev = dpath_split_dev / "figures"

    print(f"Generating split: '{cfg.split_name}'")

    class_data = load_pickle(paths["metadata"][DATASET] / "class_data.pkl")

    fpath_att_splits = paths["data"][DATASET] / "xlsa17/data/CUB/att_splits.mat"
    split_sets = loadmat(fpath_att_splits)

    idxs_train = (split_sets["trainval_loc"] - 1).squeeze()
    idxs_test_id = (split_sets["test_seen_loc"] - 1).squeeze()
    idxs_test_ood = (split_sets["test_unseen_loc"] - 1).squeeze()

    fpath_res = paths["data"][DATASET] / "xlsa17/data/CUB/res101.mat"
    data = loadmat(fpath_res)
    index_rfpaths_all = np.array(
        [_normalize_cub_rfpath(item[0][0]) for item in data["image_files"]],
        dtype=str,
    )

    index_rfpaths_train = index_rfpaths_all[idxs_train]
    index_rfpaths_test_id = index_rfpaths_all[idxs_test_id]
    index_rfpaths_test_ood = index_rfpaths_all[idxs_test_ood]

    img_ptrs, _, rfpath_2_skey, cids, n_samps_dict = _build_img_ptrs(index_rfpaths_all, class_data)

    skeys_train_pool = _build_skeys_from_rfpaths(index_rfpaths_train, rfpath_2_skey, "train_pool")
    skeys_id_test = _build_skeys_from_rfpaths(index_rfpaths_test_id, rfpath_2_skey, "id_test")
    skeys_ood_test = _build_skeys_from_rfpaths(index_rfpaths_test_ood, rfpath_2_skey, "ood_test")

    if not skeys_train_pool or not skeys_id_test or not skeys_ood_test:
        raise ValueError("One or more required fixed partitions are empty")

    if skeys_train_pool.intersection(skeys_id_test):
        raise ValueError("train_pool and id_test partitions overlap")
    if skeys_train_pool.intersection(skeys_ood_test):
        raise ValueError("train_pool and ood_test partitions overlap")
    if skeys_id_test.intersection(skeys_ood_test):
        raise ValueError("id_test and ood_test partitions overlap")

    n_cids_total_target = len(cids)
    n_samps_total_target = sum(n_samps_dict.values())

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
    for partition_name, skeys in skeys_partitions.items():
        if partition_name == "trainval":
            continue
        intersection = all_skeys.intersection(skeys)
        if intersection:
            raise ValueError(f"Partition '{partition_name}' overlaps previous partitions")
        all_skeys.update(skeys)

    skeys_expected = set(rfpath_2_skey.values())
    if all_skeys != skeys_expected:
        missing = len(skeys_expected - all_skeys)
        extra = len(all_skeys - skeys_expected)
        raise ValueError(f"Split skeys coverage mismatch: missing={missing}, extra={extra}")

    skeys_partitions_dev = build_dev_skeys_partitions(skeys_partitions, cfg.size_dev)

    cid_2_skeys_id = _build_cid_2_skeys_id(skeys_train, skeys_id_val, skeys_id_test)
    # Only species with at least one training sample are valid for n-shot tracking.
    # Species selected for OOD-val may still appear in the fixed id_test (allowed by
    # design), but they have 0 train samples and must be excluded here.
    cids_id = {cid for cid, _ in skeys_train}

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id)
    print("n-shot tracking complete!")

    print("Generating data indexes...")
    data_indexes = build_data_indexes_cub(cids, skeys_partitions, img_ptrs)
    data_indexes_dev = build_data_indexes_cub(cids, skeys_partitions_dev, img_ptrs)
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

    # SPLIT STATS TABLE

    print("Generating split stats table...")
    generate_basic_split_stats_table(
        skeys_partitions=skeys_partitions,
        dpath_figs=dpath_figs,
        n_cids_total=len(cids),
        title="Split Stats (CUB)",
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