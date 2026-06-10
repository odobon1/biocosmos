import bisect
import copy
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional, Any
from pathlib import Path

from utils.data import Split
from utils.utils import paths, save_pickle
from utils.config import GenSplitConfig

import pdb


def _process_rfpaths_parallel(rfpaths, dataset, desc):
    imgs_root = paths["imgs"][dataset]
    fpaths = [imgs_root / rfpath for rfpath in rfpaths]
    means = []
    vars_ = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        max_in_flight = max(64, (os.cpu_count() or 1) * 4)
        fpaths_iter = iter(fpaths)
        in_flight = set()
        for _ in range(min(max_in_flight, len(fpaths))):
            in_flight.add(ex.submit(process_image, next(fpaths_iter)))
        pbar = tqdm(total=len(fpaths), desc=desc, file=sys.stdout)
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                mean_img, var_img = fut.result()
                means.append(mean_img)
                vars_.append(var_img)
                pbar.update(1)
                try:
                    in_flight.add(ex.submit(process_image, next(fpaths_iter)))
                except StopIteration:
                    pass
        pbar.close()
    return means, vars_


def _snapshot_norm_stats(all_means, all_vars):
    means = np.stack(all_means)
    vars_ = np.stack(all_vars)
    mean_agg = means.mean(axis=0)
    var_agg = vars_.mean(axis=0) + means.var(axis=0)
    std_agg = np.sqrt(np.clip(var_agg, 0.0, None))
    return tuple(float(x) for x in mean_agg), tuple(float(x) for x in std_agg)


def get_norm_stats(data_indexes, dataset: str, cfg) -> Dict[str, Tuple[Tuple[float], Tuple[float]]]:
    if cfg.dev.get("skip_norm_stats", False):
        print("***** Skipping norm stats computation in dev mode; returning dummy values *****")
        return {pt: ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) for pt in ("train", "trainval", "whole")}
    return compute_rgb_norm_stats_by_partition(data_indexes, dataset)

def compute_rgb_norm_stats_by_partition(
    data_indexes: Dict[str, Any],
    dataset: str,
) -> Dict[str, Tuple[Tuple[float], Tuple[float]]]:
    """
    Norm stats accumulated incrementally: train → snapshot, +val → snapshot, 
    +test → snapshot; Each image contributes equally regardless of resolution.
    """
    groups = [
        ("train",    [d["rfpath"] for d in data_indexes["train"]]),
        ("trainval", [d["rfpath"] for d in data_indexes["val"]["id"]] +
                     [d["rfpath"] for d in data_indexes["val"]["ood"]]),
        ("whole",    [d["rfpath"] for d in data_indexes["test"]["id"]] +
                     [d["rfpath"] for d in data_indexes["test"]["ood"]]),
    ]

    all_means: List = []
    all_vars: List = []
    results = {}

    for pt_name, rfpaths in groups:
        if rfpaths:
            new_means, new_vars = _process_rfpaths_parallel(
                rfpaths, dataset, desc=f"Computing norm stats ({pt_name})"
            )
            all_means.extend(new_means)
            all_vars.extend(new_vars)

        results[pt_name] = _snapshot_norm_stats(all_means, all_vars)

    return results

# helper for _process_rfpaths_parallel()
def process_image(fpath_img):
    with Image.open(fpath_img) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    mean_img = arr.mean(axis=(0, 1), dtype=np.float64) / 255.0
    var_img = arr.var(axis=(0, 1), dtype=np.float64) / 255.0**2

    return mean_img, var_img

def truncate_subspecies(s: str) -> str:
    parts = s.split("_", 2)
    if len(parts) < 3:
        return s
    return parts[0] + "_" + parts[1]

def build_penult_2_cids(cids, cid_2_penult):
    penult_2_cids = defaultdict(list)
    for cid in cids:
        penult = cid_2_penult[cid]
        penult_2_cids[penult].append(cid)
    return penult_2_cids

def build_n_insts_2_classes_penult(cids, cid_2_penult):
    penults = [cid_2_penult[cid] for cid in cids]
    count_penult = Counter(penults)
    n_insts_2_classes_penult = defaultdict(list)
    for penult, count in count_penult.items():
        n_insts_2_classes_penult[count].append(penult)
    return n_insts_2_classes_penult

def strat_split(
    n_classes: int,
    n_draws: int,
    pct_eval: float,
    n_insts_2_classes: Dict[int, List[str]],
    class_2_insts: Dict[str, List[str | Tuple[str, int]]],
    insts: Set[str | Tuple[str, int]],
    seed: int = None,
) -> Tuple[
    Set[str | Tuple[str, int]], 
    Set[str | Tuple[str, int]], 
    Set[str | Tuple[str, int]]
]:
    """
    Stratified split of instances into remainder, val, and test sets.

    Draws `n_draws` instances for evaluation, stratified by class size so that
    each size bucket contributes proportionally. The draw proceeds in two
    phases:

    Phase 1 — low-count buckets: iterates count buckets from 1 upward, drawing
    `round(bucket_size * pct_eval)` instances per bucket. Stops once remaining
    draws can be handled without per-bucket stratification (i.e. when
    `n_draws_rem >= n_classes_rem`).

    Phase 2 — high-count buckets: if draws remain after phase 1, uses
    sklearn `train_test_split` with stratification on class to draw from the
    remaining higher-count buckets.

    Eval instances are interleaved into val/test by alternating index (even →
    val, odd → test).

    Args:
        n_classes: Total number of classes across all buckets.
        n_draws: Total number of instances to draw for evaluation
            (val + test combined).
        pct_eval: Fraction of instances to hold out; must be > 0.
            Controls when phase 1 stops (`count >= 1/pct_eval - 1`).
        n_insts_2_classes: Mapping from instance count to
            the list of classes with that count.
        class_2_insts: Mapping from class ID to its list of
            instances.
        insts: Full set of instances to split.
        seed: Random seed for reproducibility.

    Returns:
        insts_rem: Instances not drawn for evaluation (train pool).
        insts_val: Instances drawn for validation.
        insts_test: Instances drawn for test.
    """

    def compute_class_hits(n_draws, n_classes):
        class_hits = [n_draws // n_classes] * n_classes
        plus_ones = n_draws % n_classes
        for i in range(plus_ones):
            class_hits[i] += 1
        return class_hits

    rng = random.Random(seed)

    insts_rem = copy.deepcopy(insts)
    insts_eval = []

    n_classes_rem = n_classes
    n_draws_rem = n_draws

    count_min_strat2 = 1 / pct_eval
    max_count_bucket = max(n_insts_2_classes.keys()) if len(n_insts_2_classes) > 0 else 0
    i = 0
    while True:
        i += 1
        classes_i = list(n_insts_2_classes[i])
        if classes_i:
            n_classes_i = len(classes_i)
            n_instances_i = n_classes_i * i
            n_draws_i = round(n_instances_i * pct_eval)

            rng.shuffle(classes_i)
            class_hits = compute_class_hits(n_draws_i, n_classes_i)

            for idx, k in enumerate(class_hits):
                c = classes_i[idx]
                inst_hits = rng.sample(class_2_insts[c], k)
                insts_eval += inst_hits

            n_classes_rem -= n_classes_i
            n_draws_rem -= n_draws_i

        if i >= count_min_strat2 - 1 and (n_draws_rem >= n_classes_rem or n_classes_rem == 0):
            break

        # Guard against sparse/missing count buckets causing an unbounded loop.
        if i >= max_count_bucket:
            break

    if n_draws_rem > 0 and n_classes_rem > 0:
        classes_rem = []
        insts_counts_rem = []

        for count in sorted(list(n_insts_2_classes.keys())):
            if count <= i:
                continue

            classes = n_insts_2_classes[count]
            for c in classes:
                classes_rem += [c] * count
                insts_c = class_2_insts[c]
                for inst in insts_c:
                    insts_counts_rem += [(inst, count)]

        _, insts_counts_strat2 = train_test_split(
            insts_counts_rem,
            stratify=classes_rem,
            test_size=n_draws_rem,
            shuffle=True,
            random_state=seed,
        )

        insts_counts_strat2.sort(key=lambda x: (x[1], x[0]))
        insts_strat2, _ = zip(*insts_counts_strat2)
        insts_eval += insts_strat2

    insts_val = set()
    insts_test = set()
    for idx, cid in enumerate(insts_eval):
        if idx % 2 == 0:
            insts_val.add(cid)
        else:
            insts_test.add(cid)

    insts_rem -= insts_val
    insts_rem -= insts_test
    
    return insts_rem, insts_val, insts_test

def build_class_index_maps(skeys_pool):
    cid_2_skeys_pool = defaultdict(list)
    for cid, samp_idx in sorted(skeys_pool):
        cid_2_skeys_pool[cid].append((cid, samp_idx))

    n_insts_2_classes_pool = defaultdict(list)
    for cid, cid_skeys in cid_2_skeys_pool.items():
        n_insts_2_classes_pool[len(cid_skeys)].append(cid)

    return cid_2_skeys_pool, n_insts_2_classes_pool

def draw_single_partition_from_pool(
    skeys_pool: Set[Tuple[str, int]],
    n_target: int,
    choose_partition: str,
    seed: int,
) -> Tuple[
    Set[Tuple[str, int]], 
    Set[Tuple[str, int]],
]:
    """
    Draw one eval partition (val or test) from a pool of sample keys.

    Calls `strat_split` with `n_draws = 2 * n_target` to produce a balanced
    val/test split, then returns only the half named by `choose_partition` and
    folds the other half back into the remainder.

    Args:
        skeys_pool: Pool of (cid, samp_idx) keys to draw from.
        n_target: Desired number of keys in the returned partition.
        choose_partition: Which half to return — `"val"` or `"test"`.
        seed: Random seed passed to `strat_split`.

    Returns:
        skeys_chosen: The drawn partition of size ~n_target.
        skeys_rem: Remaining keys (pool minus chosen partition).
    """
    if choose_partition not in {"val", "test"}:
        raise ValueError(f"choose_partition must be one of {{'val', 'test'}}, got {choose_partition}")

    if n_target <= 0 or len(skeys_pool) == 0:
        return set(), skeys_pool

    cid_2_skeys_pool, n_insts_2_classes_pool = build_class_index_maps(skeys_pool)
    n_classes_pool = len(cid_2_skeys_pool)
    if n_classes_pool == 0:
        return set(), skeys_pool

    n_draws = min(len(skeys_pool), 2 * n_target)
    if n_draws <= 0:
        return set(), skeys_pool

    pct_eval_pool = n_draws / len(skeys_pool)
    skeys_rem, skeys_val_tmp, skeys_test_tmp = strat_split(
        n_classes=n_classes_pool,
        n_draws=n_draws,
        pct_eval=pct_eval_pool,
        n_insts_2_classes=n_insts_2_classes_pool,
        class_2_insts=cid_2_skeys_pool,
        insts=skeys_pool,
        seed=seed,
    )

    if choose_partition == "test":
        skeys_chosen = skeys_test_tmp
        skeys_restored = skeys_val_tmp
    else:
        skeys_chosen = skeys_val_tmp
        skeys_restored = skeys_test_tmp

    skeys_rem_for_next = skeys_rem | skeys_restored

    return skeys_chosen, skeys_rem_for_next

def sample_id_test_extra_taken(
    cids_id,
    cid_2_samp_idxs,
    cid_2_n_samps,
    cfg,
    skeys_id_test_extra,
):
    if skeys_id_test_extra is None:
        skeys_id_test_extra = set()

    n_samps_total = sum(cid_2_n_samps.values())
    n_samps_id_eval = round(n_samps_total * cfg.pct_eval)
    n_samps_id_test_target = n_samps_id_eval // 2

    cids_id_multis = {cid for cid in sorted(cids_id) if cid_2_n_samps[cid] > 1}
    skeys_id_multis = {(cid, samp_idx) for cid in cids_id_multis for samp_idx in cid_2_samp_idxs[cid]}

    skeys_id_test_pool = skeys_id_multis | skeys_id_test_extra
    skeys_id_test, _ = draw_single_partition_from_pool(
        skeys_pool=skeys_id_test_pool,
        n_target=n_samps_id_test_target,
        choose_partition="test",
        seed=cfg.seed,
    )
    return skeys_id_test & skeys_id_test_extra

def build_ood_partitions(
    n_insts_2_classes_penult: Dict[int, List[str]],
    penult_2_cids: Dict[str, List[str]],
    cids: Set[str],
    cid_2_samp_idxs: Dict[str, List[int]],
    cid_2_n_samps: Dict[str, int],
    cfg: GenSplitConfig,
) -> Tuple[
    set[str], 
    set[str], 
    set[str], 
    set[Tuple[str, int]], 
    set[Tuple[str, int]],
]:
    """
    Assign classes to ID, OOD-val, and OOD-test partitions.

    Classes are split at the penultimate taxonomic level so that OOD partitions
    contain entire unseen classes. Trials repeat (incrementing the seed each
    time) until both OOD partitions are within `cfg.pct_ood_tol` of
    `cfg.pct_partition` in sample-count fraction.

    Args:
        n_insts_2_classes_penult: Mapping from instance count to the list
            of penultimate-level groups with that count; used by strat_split
            for stratified sampling.
        penult_2_cids: Mapping from each penultimate-level group to its
            member class IDs.
        cids: Full set of class IDs to partition.
        cid_2_samp_idxs: Mapping from class ID to list of
            sample indices in that class.
        cid_2_n_samps: Mapping from class ID to number of
            samples.
        cfg: Config object with fields:
            - pct_eval (float): fraction of classes to hold out for OOD eval.
            - pct_partition (float): target fraction of total samples per eval
              partition.
            - pct_ood_tol (float): max acceptable absolute error between actual
              and target sample fraction; loop exits when both partitions
              satisfy this.
            - seed (int): base random seed (incremented by trial index each
              iteration).

    Returns:
        cids_id: Class IDs assigned to in-distribution.
        cids_ood_val: Class IDs assigned to OOD validation.
        cids_ood_test: Class IDs assigned to OOD test.
        skeys_ood_val: (cid, samp_idx) sample keys in OOD val.
        skeys_ood_test: (cid, samp_idx) sample keys in OOD test.
    """
    n_cids = len(cids)
    n_samps_total = sum(cid_2_n_samps.values())

    n_cids_ood_eval = round(n_cids * cfg.pct_eval)
    n_penults = len(penult_2_cids)

    close_enough = False
    i = 0
    while not close_enough:
        i += 1
        if i % 10_000 == 0 and i > 0:
            print(f"Warning: {i / 1_000}k seeds searched and no OOD partition found satisfying pct_ood_tol={cfg.pct_ood_tol}.")

        cids_id, cids_ood_val, cids_ood_test = strat_split(
            n_classes=n_penults,
            n_draws=n_cids_ood_eval,
            pct_eval=cfg.pct_eval,
            n_insts_2_classes=n_insts_2_classes_penult,
            class_2_insts=penult_2_cids,
            insts=cids,
            seed=cfg.seed + i,
        )

        n_samps_ood_test = sum(cid_2_n_samps[cid] for cid in cids_ood_test)

        skeys_ood_val = {(cid, samp_idx) for cid in cids_ood_val for samp_idx in cid_2_samp_idxs[cid]}

        skeys_id_test_extra_taken = sample_id_test_extra_taken(
            cids_id=cids_id,
            cid_2_samp_idxs=cid_2_samp_idxs,
            cid_2_n_samps=cid_2_n_samps,
            cfg=cfg,
            skeys_id_test_extra=skeys_ood_val,
        )
        n_samps_ood_val_after_id_test = len(skeys_ood_val - skeys_id_test_extra_taken)

        pct_samps_ood_val = n_samps_ood_val_after_id_test / n_samps_total
        pct_samps_ood_test = n_samps_ood_test / n_samps_total

        close_enough = (
            abs((cfg.pct_partition) - pct_samps_ood_val) < cfg.pct_ood_tol
            and abs((cfg.pct_partition) - pct_samps_ood_test) < cfg.pct_ood_tol
        )

    skeys_ood_val = {(cid, samp_idx) for cid in cids_ood_val for samp_idx in cid_2_samp_idxs[cid]}
    skeys_ood_test = {(cid, samp_idx) for cid in cids_ood_test for samp_idx in cid_2_samp_idxs[cid]}

    return cids_id, cids_ood_val, cids_ood_test, skeys_ood_val, skeys_ood_test

def build_id_partitions(
    cids_id: Set[str],
    cid_2_samp_idxs: Dict[str, List[int]],
    cid_2_n_samps: Dict[str, int],
    cfg: GenSplitConfig,
    skeys_id_test_extra: Optional[Set[Tuple[str, int]]] = None,
) -> Tuple[
    Set[Tuple[str, int]], 
    Set[Tuple[str, int]], 
    Set[Tuple[str, int]], 
    Dict[str, List[Tuple[str, int]]], 
    Dict[str, List[Tuple[str, int]]], 
    Set[str], 
    Set[Tuple[str, int]],
]:
    """
    Split in-distribution classes into train, ID-val, and ID-test sample sets.

    Draws eval samples from multi-sample classes only (singles are always
    assigned to train). Test is drawn first from the pool of multi-sample ID
    keys plus any `skeys_id_test_extra` keys; val is drawn from what remains
    after test. A fixup pass then ensures every multi-sample class has at least
    one sample in train: if a class has no train samples, one sample is moved
    back from whichever eval partition (val or test) is currently larger.

    Args:
        cids_id: Class IDs assigned to in-distribution.
        cid_2_samp_idxs: Mapping from class ID to its list of sample indices.
        cid_2_n_samps: Mapping from class ID to number of samples.
        cfg: Config object with fields:
            - pct_eval (float): fraction of total samples to allocate to ID
              eval (split evenly between val and test).
            - seed (int): random seed; test uses seed, val uses seed+1.
        skeys_id_test_extra: Additional (cid, samp_idx) keys eligible for the
            test draw (e.g. OOD-val keys shared with ID test); excluded from
            the val pool.

    Returns:
        skeys_train: (cid, samp_idx) keys assigned to train.
        skeys_id_val: (cid, samp_idx) keys assigned to ID val.
        skeys_id_test: (cid, samp_idx) keys assigned to ID test.
        cid_2_skeys_id: All (cid, samp_idx) keys per ID class.
        cid_2_skeys_id_multis: Same, restricted to multi-sample ID classes.
        cids_id_multis: ID class IDs that have more than one sample.
        skeys_id_test_extra_taken: Subset of `skeys_id_test_extra` that was
            actually drawn into ID test.
    """
    n_samps_total = sum(cid_2_n_samps.values())
    n_samps_id_eval = round(n_samps_total * cfg.pct_eval)
    n_samps_id_test_target = n_samps_id_eval // 2
    n_samps_id_val_target = n_samps_id_eval - n_samps_id_test_target

    if skeys_id_test_extra is None:
        skeys_id_test_extra = set()

    cids_id_singles = {cid for cid in cids_id if cid_2_n_samps[cid] == 1}
    cids_id_multis = cids_id - cids_id_singles

    cid_2_skeys_id = {
        cid: [(cid, samp_idx) for samp_idx in cid_2_samp_idxs[cid]]
        for cid in sorted(cids_id)
    }
    cid_2_skeys_id_multis = {cid: cid_2_skeys_id[cid] for cid in cids_id_multis}
    skeys_id_multis = {skey for skeys in cid_2_skeys_id_multis.values() for skey in skeys}  # flatten all skeys in cid_2_skeys_id_multis into one set

    skeys_id_test_pool = skeys_id_multis | skeys_id_test_extra
    skeys_id_test, skeys_id_test_pool_rem = draw_single_partition_from_pool(
        skeys_pool=skeys_id_test_pool,
        n_target=n_samps_id_test_target,
        choose_partition="test",
        seed=cfg.seed,
    )

    skeys_id_test_extra_taken = skeys_id_test & skeys_id_test_extra

    skeys_id_multis_rem = skeys_id_test_pool_rem & skeys_id_multis
    skeys_id_val, skeys_train_multis = draw_single_partition_from_pool(
        skeys_pool=skeys_id_multis_rem,
        n_target=n_samps_id_val_target,
        choose_partition="val",
        seed=cfg.seed + 1,
    )

    for cid in sorted(cids_id_multis):
        cid_skeys = set(cid_2_skeys_id_multis[cid])
        if len(cid_skeys & skeys_train_multis) > 0:
            continue

        cid_skeys_val = sorted(cid_skeys & skeys_id_val)
        cid_skeys_test = sorted(cid_skeys & skeys_id_test)

        donor_partition = "val"
        if len(skeys_id_test) > len(skeys_id_val):
            donor_partition = "test"

        skey_move = None
        if donor_partition == "val" and cid_skeys_val:
            skey_move = cid_skeys_val[0]
            skeys_id_val.remove(skey_move)
        elif donor_partition == "test" and cid_skeys_test:
            skey_move = cid_skeys_test[0]
            skeys_id_test.remove(skey_move)
        elif cid_skeys_val:
            skey_move = cid_skeys_val[0]
            skeys_id_val.remove(skey_move)
        elif cid_skeys_test:
            skey_move = cid_skeys_test[0]
            skeys_id_test.remove(skey_move)

        if skey_move is not None:
            skeys_train_multis.add(skey_move)

    skeys_id_singles = set((cid, cid_2_samp_idxs[cid][0]) for cid in cids_id_singles)
    skeys_train = skeys_train_multis | skeys_id_singles

    return (
        skeys_train,
        skeys_id_val,
        skeys_id_test,
        cid_2_skeys_id,
        cid_2_skeys_id_multis,
        cids_id_multis,
        skeys_id_test_extra_taken,
    )

def build_trainval_skeys_partition(skeys_partitions):

    skeys_trainval = skeys_partitions["train"] | skeys_partitions["id_val"] | skeys_partitions["ood_val"]
    skeys_id_test = skeys_partitions.get("id_test", set())
    return skeys_trainval - skeys_id_test

def build_id_eval_nshot(cfg, cids_id, skeys_partitions, cid_2_skeys_id):

    n_shot_tracker = []
    for _ in range(len(cfg.nst_names)):
        n_shot_tracker.append({"id_val": set(), "trainval": set(), "id_test": set()})

    for cid in sorted(cids_id):
        cid_skeys_val = set()
        cid_skeys_trainval = set()
        cid_skeys_test = set()

        n_skeys_train = 0
        n_skeys_trainval = 0
        for skey in cid_2_skeys_id[cid]:
            if skey in skeys_partitions["train"]:
                n_skeys_train += 1
                n_skeys_trainval += 1
                cid_skeys_trainval.add(skey)
            elif skey in skeys_partitions["id_val"]:
                n_skeys_trainval += 1
                cid_skeys_val.add(skey)
                cid_skeys_trainval.add(skey)
            elif skey in skeys_partitions["id_test"]:
                cid_skeys_test.add(skey)

        idx_id_val_bucket = bisect.bisect_left(cfg.nst_seps, n_skeys_train)
        n_shot_tracker[idx_id_val_bucket]["id_val"].update(cid_skeys_val)

        idx_trainval_bucket = bisect.bisect_left(cfg.nst_seps, n_skeys_trainval)
        n_shot_tracker[idx_trainval_bucket]["trainval"].update(cid_skeys_trainval)
        n_shot_tracker[idx_trainval_bucket]["id_test"].update(cid_skeys_test)

    # Second pass: OOD-val classes whose samples were borrowed into id_test.
    # These species are not in cids_id, so they were missed above.
    # We bucket their id_test samples using their trainval cardinality.
    cid_2_id_test_skeys_ood = defaultdict(set)
    for skey in skeys_partitions["id_test"]:
        cid = skey[0]
        if cid not in cids_id:
            cid_2_id_test_skeys_ood[cid].add(skey)

    if cid_2_id_test_skeys_ood:
        cid_2_n_trainval_ood = {}
        for skey in skeys_partitions["trainval"]:
            cid = skey[0]
            if cid in cid_2_id_test_skeys_ood:
                cid_2_n_trainval_ood[cid] = cid_2_n_trainval_ood.get(cid, 0) + 1

        for cid in sorted(cid_2_id_test_skeys_ood):
            n_skeys_trainval_ood = cid_2_n_trainval_ood.get(cid, 0)
            if n_skeys_trainval_ood == 0:
                continue
            idx_bucket = bisect.bisect_left(cfg.nst_seps, n_skeys_trainval_ood)
            n_shot_tracker[idx_bucket]["id_test"].update(cid_2_id_test_skeys_ood[cid])

    id_eval_nshot = {
        "names": cfg.nst_names,
        "buckets": {
            name: {
                "id_val": bucket["id_val"],
                "trainval": bucket["trainval"],
                "id_test": bucket["id_test"],
            }
            for name, bucket in zip(cfg.nst_names, n_shot_tracker)
        },
    }

    return id_eval_nshot

def build_global_cid2enc(skeys_partitions):
    all_cids = sorted({cid for skeys in skeys_partitions.values() for cid, _ in skeys})
    global_cid2enc = {cid: enc for enc, cid in enumerate(all_cids)}
    return global_cid2enc

def build_class_counts_by_partition(data_indexes, n_classes):
    results = {}
    for pt_name, pt_data in (
        ("train",    data_indexes["train"]),
        ("trainval", data_indexes["trainval"]),
        ("whole",    data_indexes["whole"]),
    ):
        counts = np.full(n_classes, np.nan)
        for datum in pt_data:
            enc = datum["class_enc"]
            counts[enc] = 1 if np.isnan(counts[enc]) else counts[enc] + 1
        results[pt_name] = counts
    return results

def build_dev_skeys_partitions(skeys_partitions, size_dev):
    return {
        partition: set(sorted(skeys_partition)[:size_dev])
        for partition, skeys_partition in skeys_partitions.items()
    }

def save_split(data_indexes, id_eval_nshot, class_counts, norm_stats, dpath_split, dpath_figs) -> None:
    norm_mean = {pt: norm_stats[pt][0] for pt in norm_stats}
    norm_std = {pt: norm_stats[pt][1] for pt in norm_stats}
    split = Split(data_indexes, id_eval_nshot, class_counts, norm_mean, norm_std)
    os.makedirs(dpath_split, exist_ok=True)
    if os.path.exists(dpath_figs):
        shutil.rmtree(dpath_figs)
    os.makedirs(dpath_figs)
    save_pickle(split, dpath_split / "split.pkl")

def plot_split_distribution(
    data: List[Tuple[int]],
    labels_data: List[str],
    colors: List[str],
    title: str,
    x_label: str,
    y_label: str,
    fpath: Path,
    ema: bool = False,
    scale: Optional[str] = None,
    markers: List[str] = ["", "", ""],
    markersizes: List[int] = [6, 6, 6],
    markeredgewidths: List[float] = [0.5, 0.5, 0.5],
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    def compute_ema(vals, alpha_ema=0.99):
        ema_vals = [vals[0]]
        for i in range(1, len(vals)):
            val_i = vals[i]
            ema_i = alpha_ema * ema_vals[-1] + (1 - alpha_ema) * val_i
            ema_vals.append(ema_i)
        return ema_vals

    x = range(len(data[0]))
    plt.figure(figsize=(16, 6))

    if ema:
        for i in range(len(data)):
            data[i] = compute_ema(data[i])

    for i in range(len(data)):
        plt.plot(
            x,
            data[i],
            color=colors[i],
            label=labels_data[i],
            marker=markers[i],
            markersize=markersizes[i],
            markeredgewidth=markeredgewidths[i],
            linestyle=linestyle,
            alpha=alpha,
        )

    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel(x_label, fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")

    if scale == "log":
        plt.yscale("log")
    elif scale == "symlog":
        plt.yscale("symlog", linthresh=1.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath, dpi=300, bbox_inches="tight")

def gen_strat_sampling_dist_plots_ood(
    penult_2_cids: Dict[str, List[str]],
    cids_id: Set[str], 
    cids_ood_val: Set[str], 
    cids_ood_test: Set[str], 
    dpath_figs: Path,
) -> None:
    """
    Generate OOD class-distribution plots and save to disk.

    For each penultimate taxonomic level, counts how many of its classes fall
    into ID, OOD-val, and OOD-test. Penultimate levels are sorted by total
    class count (descending). Three plots are saved:

    - `ood_strat_sampling_dist.png`: linear-scale line plot.
    - `ood_strat_sampling_dist_log.png`: symlog-scale scatter (no smoothing).
    - `ood_strat_sampling_dist_log_smooth.png`: log-scale line plot with EMA smoothing.

    Args:
        penult_2_cids: Mapping from penultimate-level group name to its
            list of class IDs.
        cids_id: Class IDs assigned to in-distribution.
        cids_ood_val: Class IDs assigned to OOD validation.
        cids_ood_test: Class IDs assigned to OOD test.
        dpath_figs: Directory where plot files are written.
    """

    penult_tups = [
        (
            len(cids),
            sum(cid in cids_id for cid in cids),
            sum(cid in cids_ood_val for cid in cids),
            sum(cid in cids_ood_test for cid in cids),
        )
        for cids in penult_2_cids.values()
    ]
    penult_tups.sort(key=lambda t: t[0], reverse=True)
    n_cids_penult, n_cids_penult_id, n_cids_penult_ood_val, n_cids_penult_ood_test = zip(*penult_tups)
    n_cids_penult_ood_eval = tuple(a + b for a, b in zip(n_cids_penult_ood_val, n_cids_penult_ood_test))

    data = [n_cids_penult, n_cids_penult_id, n_cids_penult_ood_eval]
    colors = ["crimson", "darkorange", "teal"]
    labels_data = ["Total", "ID-Train/Eval", "OOD-Eval"]
    x_label = "Sorted Penultimate Classes"
    y_label = "Num. Classes"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Stratified Sampling Distribution",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "ood_strat_sampling_dist.png",
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Stratified Sampling Distribution (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "ood_strat_sampling_dist_log.png",
        ema=False,
        scale="symlog",
        markers=["|", "|", "|"],
        markersizes=[7, 5, 5],
        markeredgewidths=[1.0, 1.0, 1.0],
        linestyle="",
        alpha=1.0,
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="OOD Stratified Sampling Distribution (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "ood_strat_sampling_dist_log_smooth.png",
        ema=True,
        scale="log",
    )

def gen_strat_sampling_dist_plots_id(
    cids_id_multis: Set[str], 
    cid_2_skeys_id_multis: Dict[str, List[Tuple[str, int]]],
    cid_2_n_samps: Dict[str, int],
    skeys_partitions: Dict[str, Set[Tuple[str, int]]],
    dpath_figs: Path,
) -> None:
    """
    Generate ID class-distribution plots and save to disk.

    For each ID class with multiple samples, counts how many of its samples fall
    into train, ID-val, and ID-test. Classes are sorted by total sample count
    (descending). Three plots are saved:

    - `id_strat_sampling_dist.png`: linear-scale line plot.
    - `id_strat_sampling_dist_log.png`: symlog-scale scatter (no smoothing).
    - `id_strat_sampling_dist_log_smooth.png`: log-scale line plot with EMA smoothing.

    Args:
        cids_id_multis: Class IDs in the ID split that have multiple samples.
        cid_2_skeys_id_multis: Mapping from class ID to its sample keys (for
            multi-sample ID classes).
        cid_2_n_samps: Mapping from class ID to total sample count.
        skeys_partitions: Dict with partition keys mapping to sets of sample 
            keys for each partition.
        dpath_figs: Directory where plot files are written.
    """
    cid_tups = []
    for cid in cids_id_multis:
        n_skeys = cid_2_n_samps[cid]
        n_skeys_train, n_skeys_id_val, n_skeys_id_test = 0, 0, 0
        for skey in cid_2_skeys_id_multis[cid]:
            if skey in skeys_partitions["train"]:
                n_skeys_train += 1
            elif skey in skeys_partitions["id_val"]:
                n_skeys_id_val += 1
            elif skey in skeys_partitions["id_test"]:
                n_skeys_id_test += 1

        cid_tups.append((cid, n_skeys, n_skeys_train, n_skeys_id_val, n_skeys_id_test))

    cid_tups.sort(key=lambda t: (t[1], t[0]), reverse=True)
    _, n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_val, n_skeys_ps_id_test = zip(*cid_tups)
    n_skeys_ps_id_eval = tuple(a + b for a, b in zip(n_skeys_ps_id_val, n_skeys_ps_id_test))

    data = [n_skeys_ps, n_skeys_ps_train, n_skeys_ps_id_eval]
    colors = ["crimson", "darkorange", "teal"]
    labels_data = ["Total", "Train (ID)", "Eval (ID)"]
    x_label = "Sorted Classes"
    y_label = "Num. Samples"

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Stratified Sampling Distribution",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "id_strat_sampling_dist.png",
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Stratified Sampling Distribution (Log-Scale)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "id_strat_sampling_dist_log.png",
        ema=False,
        scale="symlog",
        markers=["|", "|", "|"],
        markersizes=[7, 5, 5],
        markeredgewidths=[0.5, 0.5, 0.5],
        linestyle="",
        alpha=1.0,
    )

    plot_split_distribution(
        data,
        labels_data,
        colors,
        title="ID Stratified Sampling Distribution (Log-Scale + Smoothed)",
        x_label=x_label,
        y_label=y_label,
        fpath=dpath_figs / "id_strat_sampling_dist_log_smooth.png",
        ema=True,
        scale="log",
    )

def generate_n_shot_table(id_eval_nshot, dpath_figs, col_width=0.20, fontsize_title=8, fontsize=5) -> None:
    n_shot_col_names = [name for name in id_eval_nshot["names"]]

    row_values_id_val = ["ID Val"]
    row_values_trainval = ["TrainVal"]
    row_values_id_test = ["ID Test"]
    for name in id_eval_nshot["names"]:
        bucket_skeys_set_id_val = id_eval_nshot["buckets"][name]["id_val"]
        bucket_skeys_set_trainval = id_eval_nshot["buckets"][name]["trainval"]
        bucket_skeys_set_id_test = id_eval_nshot["buckets"][name]["id_test"]

        num_samps_val = len(bucket_skeys_set_id_val)
        if num_samps_val > 0:
            cids, _ = zip(*bucket_skeys_set_id_val)
            n_classes_val = len(set(cids))
        else:
            n_classes_val = 0
        row_values_id_val.append(f"{num_samps_val:,} ({n_classes_val})")

        num_samps_trainval = len(bucket_skeys_set_trainval)
        if num_samps_trainval > 0:
            cids, _ = zip(*bucket_skeys_set_trainval)
            n_classes_trainval = len(set(cids))
        else:
            n_classes_trainval = 0
        row_values_trainval.append(f"{num_samps_trainval:,} ({n_classes_trainval})")

        num_samps_test = len(bucket_skeys_set_id_test)
        if num_samps_test > 0:
            cids, _ = zip(*bucket_skeys_set_id_test)
            n_classes_test = len(set(cids))
        else:
            n_classes_test = 0
        row_values_id_test.append(f"{num_samps_test:,} ({n_classes_test:,})")

    labels_cols = ["Partition"] + n_shot_col_names
    data = [row_values_id_val, row_values_trainval, row_values_id_test]

    _, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")

    col_widths = [col_width] * len(labels_cols)

    tbl = ax.table(
        cellText=data,
        colLabels=labels_cols,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    for col_idx, _ in enumerate(labels_cols):
        cell = tbl[0, col_idx]
        cell.get_text().set_fontweight("bold")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    for (_, _), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)

    plt.title("n-shot Bucket Sample (Class) Counts", fontsize=fontsize_title, fontweight="bold", y=0.70)
    plt.savefig(dpath_figs / "summary_nshot.png", dpi=300, bbox_inches="tight")

def count_unique_cids_from_skeys(skeys) -> int:
    return len({cid for cid, _ in skeys})

def count_total_samples_disjoint_partitions(skeys_partitions) -> int:
    return len(
        skeys_partitions["train"]
        | skeys_partitions["id_val"]
        | skeys_partitions["id_test"]
        | skeys_partitions["ood_val"]
        | skeys_partitions["ood_test"]
    )

def render_partition_summary_table(
    labels_cols,
    data,
    title,
    dpath_figs,
    figsize=(5, 2),
    pad=-5,
    dpi=150,
    fontsize=None,
    scale=None,
) -> None:
    _, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=data,
        colLabels=labels_cols,
        cellLoc="center",
        loc="center",
    )

    for col_idx, _ in enumerate(labels_cols):
        cell = tbl[0, col_idx]
        cell.get_text().set_fontweight("bold")

    if fontsize is not None:
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)

    if scale is not None:
        tbl.scale(*scale)

    plt.title(title, fontweight="bold", pad=pad)
    plt.savefig(dpath_figs / "summary_partitions.png", dpi=dpi, bbox_inches="tight")
    plt.close()

def generate_partition_summary_table(
    skeys_partitions,
    dpath_figs,
    n_cids_total,
    title,
    labels_cols=None,
    figsize=(5, 2),
    pad=-5,
    dpi=150,
    fontsize=None,
    scale=None,
) -> None:
    if labels_cols is None:
        labels_cols = ["Partition", "Num. Classes", "Num. Samples"]

    n_samps_total = count_total_samples_disjoint_partitions(skeys_partitions)
    row_specs = [
        ("OOD Test", "ood_test"),
        ("ID Test", "id_test"),
        ("OOD Val", "ood_val"),
        ("ID Val", "id_val"),
        ("Train", "train"),
        ("TrainVal", "trainval"),
        ("Whole", "whole"),
    ]

    data = []
    for row_name, partition in row_specs:
        skeys_partition = skeys_partitions[partition]
        n_cids_partition = count_unique_cids_from_skeys(skeys_partition)
        n_samps_partition = len(skeys_partition)
        data.append([
            row_name,
            f"{n_cids_partition:,} ({n_cids_partition / n_cids_total:.2%})",
            f"{n_samps_partition:,} ({n_samps_partition / n_samps_total:.2%})",
        ])

    render_partition_summary_table(
        labels_cols=labels_cols,
        data=data,
        title=title,
        dpath_figs=dpath_figs,
        figsize=figsize,
        pad=pad,
        dpi=dpi,
        fontsize=fontsize,
        scale=scale,
    )
