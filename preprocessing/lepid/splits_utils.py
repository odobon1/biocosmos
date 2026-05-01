import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from preprocessing.common.splits import truncate_subspecies
from utils.data import species_to_genus
from utils.utils import paths


def build_img_ptrs_lepid(cids, cid_2_family):

    img_ptrs = {
        cid: {}
        for cid in sorted(cids)
    }
    cid_set = set(cids)
    cid_offsets = {
        cid: 0
        for cid in sorted(cids)
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
        cid = truncate_subspecies(subdir)

        if cid not in cid_set:
            continue
        if cid_2_family[cid] != family:
            continue

        rfpath = f"{family}/{subdir}/{mask_name}"
        idx = cid_offsets[cid]
        img_ptrs[cid][idx] = rfpath
        cid_offsets[cid] += 1

    return img_ptrs

def build_cid_2_samp_idxs_lepid(
    cids,
    cid_2_family,
    pos_filter=None,
    img_ptrs=None,
    df_metadata=None,
):
    from utils.utils import paths

    if img_ptrs is None:
        img_ptrs = build_img_ptrs_lepid(cids, cid_2_family)

    if pos_filter is None:
        return {
            cid: list(img_ptrs[cid].keys())
            for cid in sorted(cids)
        }

    if df_metadata is None:
        df_metadata = pd.read_csv(paths["lepid_metadata_imgs"])

    pos_lookup = df_metadata.set_index("mask_name")["class_dv"]

    cid_2_samp_idxs = {}
    for cid in sorted(cids):
        samp_idxs = []
        for samp_idx, rfpath in sorted(img_ptrs[cid].items()):
            fname_img = rfpath.split("/")[-1]
            if pos_lookup.get(fname_img) == pos_filter:
                samp_idxs.append(samp_idx)
        cid_2_samp_idxs[cid] = samp_idxs

    return cid_2_samp_idxs

def build_data_indexes_lepid(
    cids,
    skeys_partitions,
    cid_2_family,
    img_ptrs=None,
    df_metadata=None,
):
    from utils.utils import paths

    if img_ptrs is None:
        img_ptrs = build_img_ptrs_lepid(cids, cid_2_family)

    if df_metadata is None:
        df_metadata = pd.read_csv(paths["lepid_metadata_imgs"])

    metadata_lookup = df_metadata.set_index("mask_name")[["class_dv", "sex"]]

    def build_partition_index(partition_name):
        data_index = []
        cid2enc = {}

        for cid, samp_idx in sorted(skeys_partitions[partition_name]):
            if cid not in cid2enc:
                cid2enc[cid] = len(cid2enc)

            rfpath = img_ptrs[cid][samp_idx]
            fname = rfpath.split("/")[-1]
            pos = metadata_lookup["class_dv"].get(fname)
            sex = metadata_lookup["sex"].get(fname)
            data_index.append(
                {
                    "cid": cid,
                    "class_enc": cid2enc[cid],
                    "rfpath": rfpath,
                    "meta": {
                        "pos": None if pd.isna(pos) else pos,
                        "sex": None if pd.isna(sex) else sex,
                    },
                }
            )
        return data_index

    validation_ood = build_partition_index("ood_val")
    test_ood = build_partition_index("ood_test")

    return {
        "train": build_partition_index("train"),
        "trainval": build_partition_index("trainval"),
        "validation": {
            "id": build_partition_index("id_val"),
            "ood": validation_ood,
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood": test_ood,
        },
    }