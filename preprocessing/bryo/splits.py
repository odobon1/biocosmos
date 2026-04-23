

"""
python -m preprocessing.bryo.splits
"""

from preprocessing.common.splits import (
    build_class_counts_train,
    build_id_eval_nshot,
    build_id_partitions,
    generate_id_distribution_plots,
    generate_n_shot_table,
    save_split,
)
from preprocessing.bryo.splits_utils import (
    build_data_indexes_bryo,
    build_img_ptrs_bryo,
    build_ood_skeys,
    generate_split_stats_table_bryo,
    split_ood_genera_val_test,
)
from utils.config import get_config_splits
from utils.utils import get_subdirectory_names, load_pickle, paths, seed_libs


DATASET = "bryo"


def build_splits() -> None:
    cfg = get_config_splits()
    seed_libs(cfg.seed, seed_torch=False)
    dpath_split = paths["metadata"][DATASET] / f"splits/{cfg.split_name}"
    dpath_figs = dpath_split / "figures"
    print(f"Generating split: '{cfg.split_name}'")

    class_data = load_pickle(paths["metadata"][DATASET] / "class_data.pkl")

    genera_imgs = set(get_subdirectory_names(paths["bryo_imgs"]))
    genera_cd = set(class_data.keys())
    genera_ood = genera_imgs - genera_cd
    if not genera_ood:
        raise ValueError("genera_ood is empty from genera_imgs - genera_cd; cannot build OOD split.")

    img_ptrs_all = build_img_ptrs_bryo(sorted(genera_imgs))

    genera_available = {
        genus
        for genus, ptrs in img_ptrs_all.items()
        if len(ptrs) > 0
    }
    if not genera_available:
        raise ValueError("No Bryo .jpg images were found for any configured genus.")

    genera_ood = genera_ood.intersection(genera_available)
    if not genera_ood:
        raise ValueError("OOD genus pool is empty after filtering to genera with .jpg images.")

    genera_id = genera_available - genera_ood
    if not genera_id:
        raise ValueError("No ID genera remain after assigning OOD genera.")

    sid_2_samp_idxs = {
        sid: list(sorted(img_ptrs_all[sid].keys()))
        for sid in sorted(genera_available)
    }
    n_samps_dict = {
        sid: len(sid_2_samp_idxs[sid])
        for sid in sorted(genera_available)
    }
    n_samps_dict_id = {
        sid: n_samps_dict[sid]
        for sid in sorted(genera_id)
    }

    print("Constructing fixed OOD partitions from configured genera...")
    genera_ood_val, genera_ood_test = split_ood_genera_val_test(genera_ood, n_samps_dict, cfg.seed)
    skeys_ood_val = build_ood_skeys(genera_ood_val, sid_2_samp_idxs)
    skeys_ood_test = build_ood_skeys(genera_ood_test, sid_2_samp_idxs)
    if len(skeys_ood_val) == 0 or len(skeys_ood_test) == 0:
        raise ValueError(
            "OOD split generation produced an empty partition. "
            "Check configured OOD genera and available .jpg files."
        )
    print("OOD partitions complete!")

    print("Constructing ID partitions...")
    skeys_train, skeys_id_val, skeys_id_test, sid_2_skeys_id, sid_2_skeys_id_multis, sids_id_multis = build_id_partitions(
        set(genera_id),
        sid_2_samp_idxs,
        n_samps_dict_id,
        cfg,
    )
    print("ID partitions complete!")

    skeys_partitions = {
        "train": skeys_train,
        "id_val": skeys_id_val,
        "id_test": skeys_id_test,
        "ood_val": skeys_ood_val,
        "ood_test": skeys_ood_test,
    }

    print("Constructing n-shot tracking structures...")
    id_eval_nshot = build_id_eval_nshot(cfg, set(genera_id), skeys_partitions, sid_2_skeys_id)
    print("n-shot tracking complete!")

    print("Generating data indexes...")
    data_indexes = build_data_indexes_bryo(sorted(genera_available), skeys_partitions, img_ptrs=img_ptrs_all)
    print("Data indexes complete!")

    print("Generating class counts for train partition...")
    class_counts_train = build_class_counts_train(data_indexes)
    print("Class counts complete!")

    print("Saving split...")
    save_split(
        data_indexes,
        id_eval_nshot,
        class_counts_train,
        dpath_split,
        dpath_figs,
    )
    print("Split saved!")

    print("Generating ID distribution plots...")
    generate_id_distribution_plots(
        sids_id_multis,
        sid_2_skeys_id_multis,
        n_samps_dict_id,
        skeys_partitions,
        dpath_figs,
    )
    print("ID distribution plots complete!")

    print("Generating split stats table...")
    generate_split_stats_table_bryo(
        set(genera_id),
        genera_ood_val,
        genera_ood_test,
        skeys_partitions,
        dpath_figs,
        len(genera_available),
    )
    print("Split stats table complete!")

    print("Generating n-shot tracking stats table...")
    generate_n_shot_table(id_eval_nshot, dpath_figs)
    print("n-shot tracking table complete!")


def main() -> None:
    print("Generating split...")
    build_splits()
    print("Split complete!")


if __name__ == "__main__":
    main()