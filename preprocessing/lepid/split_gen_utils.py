import pandas as pd
from tqdm import tqdm

from preprocessing.common.split_gen import truncate_subspecies, GenSplitDataManager
from utils.utils import paths


def build_img_ptrs(cids):

    img_ptrs = {cid: {} for cid in sorted(cids)}
    cid_set = set(cids)
    cid_offsets = {cid: 0 for cid in sorted(cids)}

    cid_2_family = {cid: GenSplitDataManager.class_data[cid]["family"] for cid in cids}
    pos_filter = GenSplitDataManager.cfg.pos_filter

    usecols = ["mask_path", "mask_name"] + (["class_dv"] if pos_filter is not None else [])
    df_imgs = pd.read_csv(paths["csv"]["lepid"]["imgs"], usecols=usecols)

    for row in tqdm(df_imgs.itertuples(index=False), total=len(df_imgs), desc="Indexing Lepid images"):
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
        if pos_filter is not None and row.class_dv != pos_filter:
            continue

        rfpath = f"{family}/{subdir}/{mask_name}"
        idx = cid_offsets[cid]
        img_ptrs[cid][idx] = rfpath
        cid_offsets[cid] += 1

    return img_ptrs
