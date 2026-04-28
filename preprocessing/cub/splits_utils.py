from typing import Iterable
import matplotlib.pyplot as plt  # type: ignore[import]


def build_data_indexes_cub(
    sids: Iterable[str],
    skeys_partitions,
    img_ptrs,
):
    sid_set = set(sids)
    missing = sid_set - set(img_ptrs.keys())
    if missing:
        raise KeyError(f"Image pointers missing for {len(missing)} sids")

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


def generate_split_stats_table(
    sids_id,
    sids_ood_val,
    sids_id_test,
    sids_ood_test,
    skeys_partitions,
    dpath_figs,
    n_sids,
) -> None:
    """Generate a stats table showing split distribution for CUB dataset."""
    n_sids_id = len(sids_id)
    n_sids_ood_val = len(sids_ood_val)
    n_sids_id_test = len(sids_id_test)
    n_sids_ood_test = len(sids_ood_test)

    n_samps_ood_val = len(skeys_partitions["ood_val"])
    n_samps_ood_test = len(skeys_partitions["ood_test"])
    n_samps_total = sum([len(skeys_partitions[pt]) for pt in skeys_partitions.keys()])

    sids_id_val_unrolled, _ = zip(*skeys_partitions["id_val"])
    n_sids_id_val = len(set(sids_id_val_unrolled))

    n_skeys_train = len(skeys_partitions["train"])
    n_skeys_id_val = len(skeys_partitions["id_val"])
    n_skeys_id_test = len(skeys_partitions["id_test"])

    labels_cols = ["Set", "Num. Species", "Num. Samples"]
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

    plt.title("Split Stats (Caltech-UCSD Birds)", fontweight="bold", pad=-5)
    plt.savefig(str(dpath_figs / "stats_splits.png"), dpi=150, bbox_inches="tight")
    plt.close()