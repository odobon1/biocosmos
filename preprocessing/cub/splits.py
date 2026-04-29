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
    generate_n_shot_table,
    save_split,
    strat_split,
)
from preprocessing.cub.splits_utils import build_data_indexes_cub, generate_split_stats_table
from utils.config import get_config_splits
from utils.utils import load_pickle, paths, seed_libs


DATASET = "cub"


def _normalize_cub_rfpath(raw_path: str) -> str:
    raw_path = str(raw_path)
    idx_images = raw_path.find("images/")
    if idx_images == -1:
        raise ValueError(f"Could not parse CUB rfpath from raw image path: {raw_path}")
    return raw_path[idx_images + len("images/"):]

def _build_classdir_to_cid(class_data):
    classdir_to_cid = {}
    for cid, cid_data in class_data.items():
        rdpath_imgs = cid_data["rdpath_imgs"]
        if not isinstance(rdpath_imgs, str) or not rdpath_imgs.startswith("images/"):
            raise ValueError(f"Invalid rdpath_imgs for cid='{cid}': {rdpath_imgs}")
        class_dir = rdpath_imgs.split("/", 1)[1]
        classdir_to_cid[class_dir] = cid
    return classdir_to_cid

def _build_img_ptrs(index_rfpaths_all, class_data):
    classdir_to_cid = _build_classdir_to_cid(class_data)

    img_ptrs = defaultdict(dict)
    sid_2_samp_idxs = defaultdict(list)
    rfpath_2_skey = {}
    sid_offsets = defaultdict(int)

    for rfpath in index_rfpaths_all:
        parts = rfpath.split("/")
        if len(parts) < 2:
            raise ValueError(f"Unexpected CUB rfpath format: {rfpath}")
        class_dir = parts[0]
        if class_dir not in classdir_to_cid:
            raise KeyError(f"CUB class directory '{class_dir}' missing from class_data mapping")

        sid = classdir_to_cid[class_dir]
        samp_idx = sid_offsets[sid]
        sid_offsets[sid] += 1

        img_ptrs[sid][samp_idx] = rfpath
        sid_2_samp_idxs[sid].append(samp_idx)
        rfpath_2_skey[rfpath] = (sid, samp_idx)

    sids = sorted(img_ptrs.keys())
    n_samps_dict = {
        sid: len(sid_2_samp_idxs[sid])
        for sid in sids
    }

    return img_ptrs, sid_2_samp_idxs, rfpath_2_skey, sids, n_samps_dict

def _build_skeys_from_rfpaths(index_rfpaths, rfpath_2_skey, partition_name: str):
    skeys = set()
    for rfpath in index_rfpaths:
        if rfpath not in rfpath_2_skey:
            raise KeyError(f"rfpath '{rfpath}' from partition '{partition_name}' not found in global lookup")
        skeys.add(rfpath_2_skey[rfpath])

    if len(skeys) != len(index_rfpaths):
        raise ValueError(f"Duplicate rfpaths detected in partition '{partition_name}'")

    return skeys

def _choose_ood_val_sids(
    sids_train,
    n_sids_total_target,
    cfg,
):
    """Pick exactly round(n_sids_total_target * pct_partition) species for OOD-val."""
    n_sids_ood_val_target = round(n_sids_total_target * cfg.pct_partition)

    if n_sids_ood_val_target > len(sids_train):
        raise ValueError(
            f"Target OOD-val sid count ({n_sids_ood_val_target}) exceeds available train sid count ({len(sids_train)})"
        )

    if n_sids_ood_val_target == 0:
        return set()

    rng = np.random.default_rng(cfg.seed)
    sids_train_sorted = sorted(sids_train)
    return set(rng.choice(sids_train_sorted, size=n_sids_ood_val_target, replace=False).tolist())

def _split_train_into_train_idval_oodval(
    skeys_train_pool,
    n_sids_total_target,
    n_samps_total_target,
    cfg,
):
    sid_2_skeys_train = defaultdict(list)
    for sid, samp_idx in sorted(skeys_train_pool):
        sid_2_skeys_train[sid].append((sid, samp_idx))

    sids_train = set(sid_2_skeys_train.keys())

    sids_ood_val = _choose_ood_val_sids(
        sids_train=sids_train,
        n_sids_total_target=n_sids_total_target,
        cfg=cfg,
    )

    skeys_ood_val = {
        skey
        for sid in sids_ood_val
        for skey in sid_2_skeys_train[sid]
    }

    skeys_id_pool = skeys_train_pool - skeys_ood_val
    sid_2_skeys_id_pool = defaultdict(list)
    for sid, samp_idx in sorted(skeys_id_pool):
        sid_2_skeys_id_pool[sid].append((sid, samp_idx))

    sids_id_pool = set(sid_2_skeys_id_pool.keys())
    sids_id_singles = {
        sid
        for sid in sorted(sids_id_pool)
        if len(sid_2_skeys_id_pool[sid]) == 1
    }
    sids_id_multis = sids_id_pool - sids_id_singles

    skeys_id_multis = {
        skey
        for sid in sids_id_multis
        for skey in sid_2_skeys_id_pool[sid]
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
        sid_2_skeys_id_multis = defaultdict(list)

        for sid in sorted(sids_id_multis):
            sid_skeys = list(sorted(sid_2_skeys_id_pool[sid]))
            sid_2_skeys_id_multis[sid] = sid_skeys
            n_insts_2_classes_s[len(sid_skeys)].append(sid)

        pct_id_val = n_samps_id_val_target / len(skeys_id_multis)

        skeys_train_multis, skeys_id_val_a, skeys_id_val_b = strat_split(
            n_classes=len(sids_id_multis),
            n_draws=n_samps_id_val_target,
            pct_eval=pct_id_val,
            n_insts_2_classes=n_insts_2_classes_s,
            class_2_insts=sid_2_skeys_id_multis,
            insts=set(skeys_id_multis),
            seed=cfg.seed,
        )
        skeys_id_val = set(skeys_id_val_a).union(skeys_id_val_b)

        # Keep at least one train sample for each multi-instance ID sid.
        for sid in sorted(sids_id_multis):
            sid_skeys = set(sid_2_skeys_id_multis[sid])
            if sid_skeys.intersection(skeys_train_multis):
                continue

            sid_skeys_val = sorted(sid_skeys.intersection(skeys_id_val))
            if not sid_skeys_val:
                raise ValueError(f"Could not preserve train coverage for sid '{sid}'")

            skey_move = sid_skeys_val[0]
            skeys_id_val.remove(skey_move)
            skeys_train_multis.add(skey_move)

        if len(skeys_id_val) != n_samps_id_val_target:
            raise ValueError(
                f"ID-val sample count drifted after coverage adjustment: got {len(skeys_id_val)}, "
                f"expected {n_samps_id_val_target}"
            )

    skeys_id_singles = {
        sid_2_skeys_id_pool[sid][0]
        for sid in sorted(sids_id_singles)
    }
    skeys_train = set(skeys_train_multis).union(skeys_id_singles)

    if skeys_train.union(skeys_id_val).union(skeys_ood_val) != set(skeys_train_pool):
        raise ValueError("train/id_val/ood_val partitions do not fully cover train pool")

    if skeys_train.intersection(skeys_id_val) or skeys_train.intersection(skeys_ood_val) or skeys_id_val.intersection(skeys_ood_val):
        raise ValueError("train/id_val/ood_val partitions are not mutually disjoint")

    return skeys_train, skeys_id_val, skeys_ood_val

def _build_sid_2_skeys_id(skeys_train, skeys_id_val, skeys_id_test):
    sid_2_skeys_id = defaultdict(list)
    skeys_id_all = set(skeys_train).union(skeys_id_val).union(skeys_id_test)
    for sid, samp_idx in sorted(skeys_id_all):
        sid_2_skeys_id[sid].append((sid, samp_idx))
    return sid_2_skeys_id

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

    img_ptrs, _, rfpath_2_skey, sids, n_samps_dict = _build_img_ptrs(index_rfpaths_all, class_data)

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

    n_sids_total_target = len(sids)
    n_samps_total_target = sum(n_samps_dict.values())

    print("Constructing train/ID-val/OOD-val from train pool...")
    skeys_train, skeys_id_val, skeys_ood_val = _split_train_into_train_idval_oodval(
        skeys_train_pool=skeys_train_pool,
        n_sids_total_target=n_sids_total_target,
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

    all_skeys = set()
    for partition_name, skeys in skeys_partitions.items():
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

    sid_2_skeys_id = _build_sid_2_skeys_id(skeys_train, skeys_id_val, skeys_id_test)
    # Only species with at least one training sample are valid for n-shot tracking.
    # Species selected for OOD-val may still appear in the fixed id_test (allowed by
    # design), but they have 0 train samples and must be excluded here.
    sids_id = {sid for sid, _ in skeys_train}

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, sids_id, skeys_partitions, sid_2_skeys_id)
    print("n-shot tracking complete!")

    print("Generating data indexes...")
    data_indexes = build_data_indexes_cub(sids, skeys_partitions, img_ptrs)
    data_indexes_dev = build_data_indexes_cub(sids, skeys_partitions_dev, img_ptrs)
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
    sids_ood_val = {sid for sid, _ in skeys_ood_val}
    sids_id_test = {sid for sid, _ in skeys_id_test}
    sids_ood_test = {sid for sid, _ in skeys_ood_test}
    generate_split_stats_table(
        sids_id,
        sids_ood_val,
        sids_id_test,
        sids_ood_test,
        skeys_partitions,
        dpath_figs,
        len(sids),
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