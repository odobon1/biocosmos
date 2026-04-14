from collections import defaultdict
import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from preprocessing.common.splits import (
    generate_ood_distribution_plots,
    plot_split_distribution,
    sid_to_genus,
    strat_split,
    truncate_subspecies,
)


def build_img_ptrs_lepid(sids, sid_2_family):
    from utils.utils import paths

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

def build_ood_family_partitions(
    sids,
    sid_2_family,
    sid_2_samp_idxs,
    cfg,
):
    family_2_sids = defaultdict(list)
    for sid in sorted(sids):
        family_2_sids[sid_2_family[sid]].append(sid)

    if cfg.ood_family_name is None:
        raise ValueError("cfg.ood_family_name must be provided for Lepid split generation")

    family_name = cfg.ood_family_name.strip().lower()
    if family_name not in family_2_sids:
        families = ", ".join(sorted(family_2_sids))
        raise ValueError(f"Unknown ood_family_name='{cfg.ood_family_name}'. Available families: {families}")

    sids_ood_family = list(sorted(family_2_sids[family_name]))
    if len(sids_ood_family) < 2:
        raise ValueError(
            f"OOD family '{family_name}' must contain at least 2 species, found {len(sids_ood_family)}"
        )

    sids_ood_family = sorted(
        sids_ood_family,
        key=lambda sid: (-len(sid_2_samp_idxs[sid]), sid),
    )

    sids_ood_family_val = set()
    sids_ood_family_test = set()
    for idx, sid in enumerate(sids_ood_family):
        if idx % 2 == 0:
            sids_ood_family_val.add(sid)
        else:
            sids_ood_family_test.add(sid)

    skeys_ood_family_val = set()
    for sid in sorted(sids_ood_family_val):
        for samp_idx in sid_2_samp_idxs[sid]:
            skeys_ood_family_val.add((sid, samp_idx))

    skeys_ood_family_test = set()
    for sid in sorted(sids_ood_family_test):
        for samp_idx in sid_2_samp_idxs[sid]:
            skeys_ood_family_test.add((sid, samp_idx))

    sids_remaining = set(sids) - sids_ood_family_val - sids_ood_family_test

    return (
        sids_remaining,
        sids_ood_family_val,
        sids_ood_family_test,
        skeys_ood_family_val,
        skeys_ood_family_test,
    )

def build_ood_genus_partitions(
    sids,
    sid_2_family,
    sid_2_samp_idxs,
    n_samps_dict,
    cfg,
    n_samps_total_target=None,
):
    genus_2_sids = defaultdict(list)
    family_2_genera = defaultdict(list)
    for sid in sorted(sids):
        genus = sid_to_genus(sid)
        family = sid_2_family[sid]
        genus_2_sids[genus].append(sid)
        if genus not in family_2_genera[family]:
            family_2_genera[family].append(genus)

    n_insts_2_classes_f = defaultdict(list)
    for family in sorted(family_2_genera):
        n_insts_2_classes_f[len(family_2_genera[family])].append(family)

    n_genera = len(genus_2_sids)
    n_families = len(family_2_genera)
    n_samps_total = sum(n_samps_dict.values())
    n_samps_target = n_samps_total if n_samps_total_target is None else n_samps_total_target
    pct_eval_eff = cfg.pct_eval * (n_samps_target / n_samps_total)
    pct_eval_eff = min(max(pct_eval_eff, 0.0), 0.999)
    n_genera_ood_eval = round(n_genera * pct_eval_eff)

    close_enough = False
    i = 0
    while not close_enough:
        i += 1
        families_id, families_ood_val, families_ood_test = strat_split(
            n_classes=n_families,
            n_draws=n_genera_ood_eval,
            pct_eval=pct_eval_eff,
            n_insts_2_classes=n_insts_2_classes_f,
            class_2_insts=family_2_genera,
            insts=set(genus_2_sids.keys()),
            seed=cfg.seed + i,
        )

        genera_ood_val = set(families_ood_val)
        genera_ood_test = set(families_ood_test)

        sids_ood_genus_val = {
            sid
            for genus in genera_ood_val
            for sid in genus_2_sids[genus]
        }
        sids_ood_genus_test = {
            sid
            for genus in genera_ood_test
            for sid in genus_2_sids[genus]
        }

        n_samps_ood_genus_val = sum(n_samps_dict[sid] for sid in sids_ood_genus_val)
        n_samps_ood_genus_test = sum(n_samps_dict[sid] for sid in sids_ood_genus_test)

        pct_samps_ood_genus_val = n_samps_ood_genus_val / n_samps_target
        pct_samps_ood_genus_test = n_samps_ood_genus_test / n_samps_target

        close_enough = (
            abs((cfg.pct_eval / 2) - pct_samps_ood_genus_val) < cfg.pct_ood_tol
            and abs((cfg.pct_eval / 2) - pct_samps_ood_genus_test) < cfg.pct_ood_tol
        )

    genera_id = set(families_id)
    sids_id = {
        sid
        for genus in genera_id
        for sid in genus_2_sids[genus]
    }

    skeys_ood_genus_val = set()
    for sid in sorted(sids_ood_genus_val):
        for samp_idx in sid_2_samp_idxs[sid]:
            skeys_ood_genus_val.add((sid, samp_idx))

    skeys_ood_genus_test = set()
    for sid in sorted(sids_ood_genus_test):
        for samp_idx in sid_2_samp_idxs[sid]:
            skeys_ood_genus_test.add((sid, samp_idx))

    return sids_id, sids_ood_genus_val, sids_ood_genus_test, skeys_ood_genus_val, skeys_ood_genus_test

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
        data_index = {
            "sids": [],
            "rfpaths": [],
        }

        for sid, samp_idx in sorted(skeys_partitions[partition_name]):
            data_index["sids"].append(sid)
            data_index["rfpaths"].append(img_ptrs[sid][samp_idx])

        fname_imgs = [rfpath.split("/")[2] for rfpath in data_index["rfpaths"]]
        metadata_rows = metadata_lookup.reindex(fname_imgs).astype(object).where(lambda x: x.notna(), None)
        data_index["pos"] = metadata_rows["class_dv"].tolist()
        data_index["sex"] = metadata_rows["sex"].tolist()
        return data_index

    validation_ood_species = build_partition_index("ood_val")
    test_ood_species = build_partition_index("ood_test")

    return {
        "train": build_partition_index("train"),
        "validation": {
            "id": build_partition_index("id_val"),
            "ood": validation_ood_species,
            "ood_species": validation_ood_species,
            "ood_genus": build_partition_index("ood_genus_val"),
            "ood_family": build_partition_index("ood_family_val"),
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood": test_ood_species,
            "ood_species": test_ood_species,
            "ood_genus": build_partition_index("ood_genus_test"),
            "ood_family": build_partition_index("ood_family_test"),
        },
    }

def generate_ood_distribution_plots_lepid(
    sid_2_family,
    sids_after_family,
    sids_after_genus,
    genus_2_sids_species,
    sids_id,
    sids_ood_val,
    sids_ood_test,
    sids_ood_genus_val,
    sids_ood_genus_test,
    dpath_figs,
) -> None:
    def plot_rank_distribution(entries, labels_data, x_label, y_label, title_prefix, filepath_prefix):
        entries.sort(key=lambda t: (t[1], t[0]), reverse=True)
        _, totals, kept, held_out = zip(*entries)
        data = [totals, kept, held_out]
        colors = ["crimson", "darkorange", "teal"]

        plot_split_distribution(
            data,
            labels_data,
            colors,
            title=f"{title_prefix}",
            x_label=x_label,
            y_label=y_label,
            filepath=str(dpath_figs / f"{filepath_prefix}.png"),
        )
        plot_split_distribution(
            data,
            labels_data,
            colors,
            title=f"{title_prefix} (Log-Scale)",
            x_label=x_label,
            y_label=y_label,
            filepath=str(dpath_figs / f"{filepath_prefix}_log.png"),
            ema=False,
            scale="symlog",
            marker="|",
            markersize=6,
            markeredgewidth=1.0,
            linestyle="",
            alpha=1.0,
        )
        plot_split_distribution(
            data,
            labels_data,
            colors,
            title=f"{title_prefix} (Log-Scale + Smoothed)",
            x_label=x_label,
            y_label=y_label,
            filepath=str(dpath_figs / f"{filepath_prefix}_log_smooth.png"),
            ema=True,
            scale="log",
        )

    for suffix in ("", "_log", "_log_smooth"):
        fpath = dpath_figs / f"distribution_ood_family{suffix}.png"
        if fpath.exists():
            fpath.unlink()

    families_after_family = {
        sid_2_family[sid]
        for sid in sids_after_family
    }

    family_2_genera_total = defaultdict(set)
    family_2_genera_kept = defaultdict(set)
    family_2_genera_ood = defaultdict(set)
    for sid, family in sid_2_family.items():
        if family not in families_after_family:
            continue
        genus = sid_to_genus(sid)
        family_2_genera_total[family].add(genus)
        if sid in sids_after_genus:
            family_2_genera_kept[family].add(genus)
        if sid in sids_ood_genus_val or sid in sids_ood_genus_test:
            family_2_genera_ood[family].add(genus)

    genus_entries = []
    for family in sorted(family_2_genera_total):
        genus_entries.append((
            family,
            len(family_2_genera_total[family]),
            len(family_2_genera_kept[family]),
            len(family_2_genera_ood[family]),
        ))

    plot_rank_distribution(
        genus_entries,
        labels_data=["Total", "Post Genus Holdout", "OOD-Genus Eval"],
        x_label="Sorted Families",
        y_label="Num. Genera",
        title_prefix="OOD Genus Distribution",
        filepath_prefix="distribution_ood_genus",
    )

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
        genera = {sid_to_genus(sid) for sid in sids}
        families = {sid_2_family[sid] for sid in sids}
        return len(sids), len(genera), len(families), len(skeys)

    rows = [
        ("Train", skeys_partitions["train"]),
        ("ID Val", skeys_partitions["id_val"]),
        ("ID Test", skeys_partitions["id_test"]),
        ("OOD Species Val", skeys_partitions["ood_val"]),
        ("OOD Species Test", skeys_partitions["ood_test"]),
        ("OOD Genus Val", skeys_partitions["ood_genus_val"]),
        ("OOD Genus Test", skeys_partitions["ood_genus_test"]),
        ("OOD Family Val", skeys_partitions["ood_family_val"]),
        ("OOD Family Test", skeys_partitions["ood_family_test"]),
    ]

    n_samps_total = sum(len(skeys_partitions[name]) for name in skeys_partitions)
    n_genera_total = len({sid_to_genus(sid) for sid in sid_2_family})
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

    labels_cols = ["Partition", "Num. Families", "Num. Genera", "Num. Species", "Num. Samples"]

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
