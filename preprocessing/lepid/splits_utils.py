import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from preprocessing.common.splits import generate_ood_distribution_plots, truncate_subspecies
from utils.data import species_to_genus
from utils.utils import paths


def build_img_ptrs_lepid(sids, sid_2_family):

    img_ptrs = {
        sid: {}
        for sid in sorted(sids)
    }
    sid_set = set(sids)
    sid_offsets = {
        sid: 0
        for sid in sorted(sids)
    }
    df_metadata = pd.read_csv(paths["lepid_metadata_imgs"], usecols=["mask_path", "mask_name"])
    for row in tqdm(df_metadata.itertuples(index=False), total=len(df_metadata), desc="Indexing Lepid images"):
        mask_path = row.mask_path
        mask_name = row.mask_name

        parts = str(mask_path).strip().split("/")
        if len(parts) < 3:
            continue

        family = parts[-3]
        subdir = parts[-2]
        sid = truncate_subspecies(subdir)

        if sid not in sid_set:
            continue
        if sid_2_family[sid] != family:
            continue

        rfpath = f"{family}/{subdir}/{mask_name}"
        idx = sid_offsets[sid]
        img_ptrs[sid][idx] = rfpath
        sid_offsets[sid] += 1

    return img_ptrs

def build_sid_2_samp_idxs_lepid(
    sids,
    sid_2_family,
    pos_filter=None,
    img_ptrs=None,
    df_metadata=None,
):
    from utils.utils import paths

    if img_ptrs is None:
        img_ptrs = build_img_ptrs_lepid(sids, sid_2_family)

    if pos_filter is None:
        return {
            sid: list(img_ptrs[sid].keys())
            for sid in sorted(sids)
        }

    if df_metadata is None:
        df_metadata = pd.read_csv(paths["lepid_metadata_imgs"])

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

def build_data_indexes_lepid(
    sids,
    skeys_partitions,
    sid_2_family,
    img_ptrs=None,
    df_metadata=None,
):
    from utils.utils import paths

    if img_ptrs is None:
        img_ptrs = build_img_ptrs_lepid(sids, sid_2_family)

    if df_metadata is None:
        df_metadata = pd.read_csv(paths["lepid_metadata_imgs"])

    metadata_lookup = df_metadata.set_index("mask_name")[["class_dv", "sex"]]

    def build_partition_index(partition_name):
        data_index = []
        cid2enc = {}

        for sid, samp_idx in sorted(skeys_partitions[partition_name]):
            if sid not in cid2enc:
                cid2enc[sid] = len(cid2enc)

            rfpath = img_ptrs[sid][samp_idx]
            fname = rfpath.split("/")[-1]
            pos = metadata_lookup["class_dv"].get(fname)
            sex = metadata_lookup["sex"].get(fname)
            data_index.append(
                {
                    "cid": sid,
                    "class_enc": cid2enc[sid],
                    "rfpath": rfpath,
                    "meta": {
                        "pos": None if pd.isna(pos) else pos,
                        "sex": None if pd.isna(sex) else sex,
                    },
                }
            )
        return data_index

    validation_ood_species = build_partition_index("ood_val")
    test_ood_species = build_partition_index("ood_test")

    return {
        "train": build_partition_index("train"),
        "validation": {
            "id": build_partition_index("id_val"),
            "ood_species": validation_ood_species,
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood_species": test_ood_species,
        },
    }

def generate_ood_distribution_plots_lepid(
    genus_2_sids_species,
    sids_id,
    sids_ood_val,
    sids_ood_test,
    dpath_figs,
) -> None:
    generate_ood_distribution_plots(
        genus_2_sids_species,
        sids_id,
        sids_ood_val,
        sids_ood_test,
        dpath_figs,
    )

def generate_split_stats_table_lepid(
    sid_2_family,
    skeys_partitions,
    dpath_figs,
    n_sids,
) -> None:
    def count_ranks_from_skeys(skeys):
        sids = {sid for sid, _ in skeys}
        genera = {species_to_genus(sid) for sid in sids}
        families = {sid_2_family[sid] for sid in sids}
        return len(sids), len(genera), len(families), len(skeys)

    rows = [
        ("Train", skeys_partitions["train"]),
        ("ID Val", skeys_partitions["id_val"]),
        ("ID Test", skeys_partitions["id_test"]),
        ("OOD Species Val", skeys_partitions["ood_val"]),
        ("OOD Species Test", skeys_partitions["ood_test"]),
    ]

    n_samps_total = sum(len(skeys_partitions[name]) for name in skeys_partitions)
    n_genera_total = len({species_to_genus(sid) for sid in sid_2_family})
    n_families_total = len(set(sid_2_family.values()))

    data = []
    for row_name, skeys in rows:
        n_species, n_genera, n_families, n_samps = count_ranks_from_skeys(skeys)
        data.append([
            row_name,
            f"{n_families:,} ({n_families / n_families_total:.2%})",
            f"{n_genera:,} ({n_genera / n_genera_total:.2%})",
            f"{n_species:,} ({n_species / n_sids:.2%})",
            f"{n_samps:,} ({n_samps / n_samps_total:.2%})",
        ])

    data.append([
        "Whole Dataset",
        f"{n_families_total:,} (100.00%)",
        f"{n_genera_total:,} (100.00%)",
        f"{n_sids:,} (100.00%)",
        f"{n_samps_total:,} (100.00%)",
    ])

    labels_cols = ["Partition", "Num. Families", "Num. Genera", "Num. Classes", "Num. Samples"]

    fig_height = max(3.0, 0.42 * (len(data) + 1))
    _, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis("off")
    tbl = ax.table(
        cellText=data,
        colLabels=labels_cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)

    for col_idx, _ in enumerate(labels_cols):
        cell = tbl[0, col_idx]
        cell.get_text().set_fontweight("bold")

    plt.title("Split Stats (Lepidoptera)", fontweight="bold", pad=6)
    plt.savefig(str(dpath_figs / "stats_splits.png"), dpi=150, bbox_inches="tight")