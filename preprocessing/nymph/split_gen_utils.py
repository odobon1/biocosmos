import glob
import pandas as pd
from tqdm import tqdm

from utils.utils import paths


def build_img_ptrs(cids):

    img_ptrs = {}

    for cid in tqdm(sorted(cids)):
        img_ptrs[cid] = {}

        dpath_imgs_cid = paths["imgs"]["nymph"] / cid
        ffpaths_png = sorted(glob.glob(f"{dpath_imgs_cid}/*.png"))
        rfpaths_png = [png_file.split("images/", 1)[1] for png_file in ffpaths_png]

        for i, rfpath in enumerate(rfpaths_png):
            img_ptrs[cid][i] = rfpath

    return img_ptrs

def build_cid_2_samp_idxs(
    cids,
    img_ptrs,
    pos_filter=None,
    df_metadata=None,
):
    if pos_filter is None:
        if img_ptrs is None:
            img_ptrs = build_img_ptrs(cids)
        return {
            cid: list(img_ptrs[cid].keys())
            for cid in sorted(cids)
        }

    if img_ptrs is None:
        img_ptrs = build_img_ptrs(cids)
    if df_metadata is None:
        df_metadata = pd.read_csv(paths["nymph_metadata"])

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

def build_data_indexes(cids, skeys_pts, cid2enc, img_ptrs=None, df_metadata=None):

    if img_ptrs is None:
        img_ptrs = build_img_ptrs(cids)

    if df_metadata is None:
        df_metadata = pd.read_csv(paths["nymph_metadata"])
    metadata_lookup = df_metadata.set_index("mask_name")[["class_dv", "sex"]]

    def build_partition_index(partition):
        data_index = []
        for cid, samp_idx in sorted(skeys_pts[partition]):
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

    return {
        "train": build_partition_index("train"),
        "trainval": build_partition_index("trainval"),
        "val": {
            "id": build_partition_index("id_val"),
            "ood": build_partition_index("ood_val"),
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood": build_partition_index("ood_test"),
        },
        "whole": build_partition_index("whole"),
    }