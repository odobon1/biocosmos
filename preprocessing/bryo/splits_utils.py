import random

import matplotlib.pyplot as plt  # type: ignore[import]


def build_img_ptrs_bryo(genera):
    from utils.utils import paths

    img_ptrs = {}
    for genus in sorted(genera):
        dpath_imgs_genus = paths["bryo_imgs"] / genus
        if not dpath_imgs_genus.is_dir():
            continue

        ffpaths_jpg = sorted(
            fpath
            for fpath in dpath_imgs_genus.iterdir()
            if fpath.is_file() and fpath.suffix.lower() == ".jpg"
        )

        img_ptrs[genus] = {
            idx: f"{genus}/{fpath.name}"
            for idx, fpath in enumerate(ffpaths_jpg)
        }

    return img_ptrs


def split_ood_genera_val_test(genera_ood, n_samps_dict, seed):
    genera_sorted = sorted(
        list(genera_ood),
        key=lambda genus: (-n_samps_dict[genus], genus),
    )
    rng = random.Random(seed)
    rng.shuffle(genera_sorted)

    genera_ood_val = set()
    genera_ood_test = set()
    n_samps_val = 0
    n_samps_test = 0

    for genus in genera_sorted:
        n_samps_g = n_samps_dict[genus]
        if n_samps_val <= n_samps_test:
            genera_ood_val.add(genus)
            n_samps_val += n_samps_g
        else:
            genera_ood_test.add(genus)
            n_samps_test += n_samps_g

    # Ensure both partitions are populated when possible.
    if len(genera_sorted) > 1 and len(genera_ood_test) == 0:
        genus_move = sorted(genera_ood_val, key=lambda g: n_samps_dict[g], reverse=True)[0]
        genera_ood_val.remove(genus_move)
        genera_ood_test.add(genus_move)

    return genera_ood_val, genera_ood_test


def build_ood_skeys(genera_ood, sid_2_samp_idxs):
    skeys_ood = set()
    for genus in sorted(genera_ood):
        for samp_idx in sid_2_samp_idxs[genus]:
            skeys_ood.add((genus, samp_idx))
    return skeys_ood


def build_data_indexes_bryo(genera, skeys_partitions, img_ptrs=None):
    if img_ptrs is None:
        img_ptrs = build_img_ptrs_bryo(genera)

    def build_partition_index(partition_name):
        data_index = {
            "sids": [],
            "rfpaths": [],
            "pos": [],
            "sex": [],
        }

        for sid, samp_idx in sorted(skeys_partitions[partition_name]):
            data_index["sids"].append(sid)
            data_index["rfpaths"].append(img_ptrs[sid][samp_idx])
            data_index["pos"].append(None)
            data_index["sex"].append(None)

        return data_index

    return {
        "train": build_partition_index("train"),
        "validation": {
            "id": build_partition_index("id_val"),
            "ood": build_partition_index("ood_val"),
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood": build_partition_index("ood_test"),
        },
    }


def generate_split_stats_table_bryo(
    sids_id,
    sids_ood_val,
    sids_ood_test,
    skeys_partitions,
    dpath_figs,
    n_sids,
) -> None:
    n_sids_id = len(sids_id)
    n_sids_ood_val = len(sids_ood_val)
    n_sids_ood_test = len(sids_ood_test)

    n_samps_ood_val = len(skeys_partitions["ood_val"])
    n_samps_ood_test = len(skeys_partitions["ood_test"])
    n_samps_total = sum(len(skeys_partitions[partition]) for partition in skeys_partitions)

    sids_id_val_unrolled, _ = zip(*skeys_partitions["id_val"])
    n_sids_id_val = len(set(sids_id_val_unrolled))

    sids_id_test_unrolled, _ = zip(*skeys_partitions["id_test"])
    n_sids_id_test = len(set(sids_id_test_unrolled))

    n_skeys_train = len(skeys_partitions["train"])
    n_skeys_id_val = len(skeys_partitions["id_val"])
    n_skeys_id_test = len(skeys_partitions["id_test"])

    labels_cols = ["Set", "Num. Genera", "Num. Samples"]
    data = [
        ["Train", f"{n_sids_id:,} ({n_sids_id / n_sids:.2%})", f"{n_skeys_train:,} ({n_skeys_train / n_samps_total:.2%})"],
        ["ID Val", f"{n_sids_id_val:,} ({n_sids_id_val / n_sids:.2%})", f"{n_skeys_id_val:,} ({n_skeys_id_val / n_samps_total:.2%})"],
        ["ID Test", f"{n_sids_id_test:,} ({n_sids_id_test / n_sids:.2%})", f"{n_skeys_id_test:,} ({n_skeys_id_test / n_samps_total:.2%})"],
        ["OOD Val", f"{n_sids_ood_val:,} ({n_sids_ood_val / n_sids:.2%})", f"{n_samps_ood_val:,} ({n_samps_ood_val / n_samps_total:.2%})"],
        ["OOD Test", f"{n_sids_ood_test:,} ({n_sids_ood_test / n_sids:.2%})", f"{n_samps_ood_test:,} ({n_samps_ood_test / n_samps_total:.2%})"],
        ["Whole Dataset", f"{n_sids:,} (100.00%)", f"{n_samps_total:,} (100.00%)"],
    ]

    _, ax = plt.subplots(figsize=(5, 2))
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

    plt.title("Split Stats (Bryozoa)", fontweight="bold", pad=-5)
    plt.savefig(str(dpath_figs / "stats_splits.png"), dpi=150, bbox_inches="tight")