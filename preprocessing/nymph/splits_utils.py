import glob
import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]


def build_img_ptrs(sids):
    from utils.utils import paths

    img_ptrs = {}

    for sid in tqdm(sorted(sids)):
        img_ptrs[sid] = {}

        dpath_imgs_sid = paths["nymph_imgs"] / sid
        ffpaths_png = sorted(glob.glob(f"{dpath_imgs_sid}/*.png"))
        rfpaths_png = [png_file.split("images/", 1)[1] for png_file in ffpaths_png]

        for i, rfpath in enumerate(rfpaths_png):
            img_ptrs[sid][i] = rfpath

    return img_ptrs

def build_sid_2_samp_idxs(
    sids,
    pos_filter=None,
    img_ptrs=None,
    df_metadata=None,
):
    if pos_filter is None:
        if img_ptrs is None:
            img_ptrs = build_img_ptrs(sids)
        return {
            sid: list(img_ptrs[sid].keys())
            for sid in sorted(sids)
        }

    if img_ptrs is None:
        img_ptrs = build_img_ptrs(sids)
    if df_metadata is None:
        from utils.utils import paths

        df_metadata = pd.read_csv(paths["nymph_metadata"])

    pos_lookup = df_metadata.set_index("mask_name")["class_dv"]

    sid_2_samp_idxs = {}
    for sid in sorted(sids):
        samp_idxs = []
        for samp_idx, rfpath in sorted(img_ptrs[sid].items()):
            fname_img = rfpath.split("/")[-1]
            if pos_lookup.get(fname_img) == pos_filter:
                samp_idxs.append(samp_idx)
        sid_2_samp_idxs[sid] = samp_idxs

    return sid_2_samp_idxs

def build_data_indexes(sids, skeys_partitions):
    from utils.utils import paths

    img_ptrs = build_img_ptrs(sids)

    df_metadata = pd.read_csv(paths["nymph_metadata"])
    metadata_lookup = df_metadata.set_index("mask_name")[["class_dv", "sex"]]

    def build_partition_index(partition_name):
        data_index = {
            "sids": [],
            "rfpaths": [],
        }

        for sid, samp_idx in sorted(skeys_partitions[partition_name]):
            data_index["sids"].append(sid)
            data_index["rfpaths"].append(img_ptrs[sid][samp_idx])

        fname_imgs = [rfpath.split("/")[1] for rfpath in data_index["rfpaths"]]
        metadata_rows = metadata_lookup.reindex(fname_imgs).astype(object).where(lambda x: x.notna(), None)
        data_index["pos"] = metadata_rows["class_dv"].tolist()
        data_index["sex"] = metadata_rows["sex"].tolist()
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
    n_samps_total = sum([len(skeys_partitions[pt]) for pt in skeys_partitions.keys()])

    sids_id_val_unrolled, _ = zip(*skeys_partitions["id_val"])
    n_sids_id_val = len(set(sids_id_val_unrolled))

    sids_id_test_unrolled, _ = zip(*skeys_partitions["id_test"])
    n_sids_id_test = len(set(sids_id_test_unrolled))

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

    plt.title("Split Stats (Nymphalidae)", fontweight="bold", pad=-5)
    plt.savefig(str(dpath_figs / "stats_splits.png"), dpi=150, bbox_inches="tight")
