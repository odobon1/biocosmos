import glob
import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]


def build_img_ptrs(cids):
    from utils.utils import paths

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
    pos_filter=None,
    img_ptrs=None,
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
        from utils.utils import paths

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

def build_data_indexes(cids, skeys_partitions):
    from utils.utils import paths

    img_ptrs = build_img_ptrs(cids)

    df_metadata = pd.read_csv(paths["nymph_metadata"])
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

    return {
        "train": build_partition_index("train"),
        "trainval": build_partition_index("trainval"),
        "validation": {
            "id": build_partition_index("id_val"),
            "ood": build_partition_index("ood_val"),
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood": build_partition_index("ood_test"),
        },
    }