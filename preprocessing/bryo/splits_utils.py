import matplotlib.pyplot as plt  # type: ignore[import]


def build_img_ptrs_bryo(genera):
    from utils.utils import paths

    img_ptrs = {}
    for genus in sorted(genera):
        dpath_imgs_genus = paths["imgs"]["bryo"] / genus
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

def build_data_indexes_bryo(genera, skeys_partitions, img_ptrs=None):
    if img_ptrs is None:
        img_ptrs = build_img_ptrs_bryo(genera)

    def build_partition_index(partition_name):
        data_index = []
        cid2enc = {}

        for cid, samp_idx in sorted(skeys_partitions[partition_name]):
            if cid not in cid2enc:
                cid2enc[cid] = len(cid2enc)

            data_index.append(
                {
                    "cid": cid,
                    "class_enc": cid2enc[cid],
                    "rfpath": img_ptrs[cid][samp_idx],
                    "meta": None,
                }
            )
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
    cids_id,
    cids_ood_val,
    cids_ood_test,
    skeys_partitions,
    dpath_figs,
    n_cids,
) -> None:
    n_cids_id = len(cids_id)
    n_cids_ood_val = len(cids_ood_val)
    n_cids_ood_test = len(cids_ood_test)

    n_samps_ood_val = len(skeys_partitions["ood_val"])
    n_samps_ood_test = len(skeys_partitions["ood_test"])
    n_samps_total = sum(len(skeys_partitions[partition]) for partition in skeys_partitions)

    cids_id_val_unrolled, _ = zip(*skeys_partitions["id_val"])
    n_cids_id_val = len(set(cids_id_val_unrolled))

    cids_id_test_unrolled, _ = zip(*skeys_partitions["id_test"])
    n_cids_id_test = len(set(cids_id_test_unrolled))

    n_skeys_train = len(skeys_partitions["train"])
    n_skeys_id_val = len(skeys_partitions["id_val"])
    n_skeys_id_test = len(skeys_partitions["id_test"])

    labels_cols = ["Set", "Num. Classes", "Num. Samples"]
    data = [
        ["Train", f"{n_cids_id:,} ({n_cids_id / n_cids:.2%})", f"{n_skeys_train:,} ({n_skeys_train / n_samps_total:.2%})"],
        ["ID Val", f"{n_cids_id_val:,} ({n_cids_id_val / n_cids:.2%})", f"{n_skeys_id_val:,} ({n_skeys_id_val / n_samps_total:.2%})"],
        ["ID Test", f"{n_cids_id_test:,} ({n_cids_id_test / n_cids:.2%})", f"{n_skeys_id_test:,} ({n_skeys_id_test / n_samps_total:.2%})"],
        ["OOD Val", f"{n_cids_ood_val:,} ({n_cids_ood_val / n_cids:.2%})", f"{n_samps_ood_val:,} ({n_samps_ood_val / n_samps_total:.2%})"],
        ["OOD Test", f"{n_cids_ood_test:,} ({n_cids_ood_test / n_cids:.2%})", f"{n_samps_ood_test:,} ({n_samps_ood_test / n_samps_total:.2%})"],
        ["Whole Dataset", f"{n_cids:,} (100.00%)", f"{n_samps_total:,} (100.00%)"],
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