import glob
import pandas as pd
from tqdm import tqdm

from utils.utils import paths
from preprocessing.common.split_gen import GenSplitDataManager


def build_img_ptrs(cids):

    pos_filter = GenSplitDataManager.cfg.pos_filter

    if pos_filter is not None:
        df_metadata = pd.read_csv(paths["csv"]["nymph"]["imgs"])
        pos_lookup = df_metadata.set_index("mask_name")["class_dv"]

    img_ptrs = {}
    for cid in tqdm(sorted(cids)):
        dpath_imgs_cid = paths["imgs"]["nymph"] / cid
        ffpaths_png = sorted(glob.glob(f"{dpath_imgs_cid}/*.png"))
        rfpaths_png = [png_file.split("images/", 1)[1] for png_file in ffpaths_png]

        idx = 0
        img_ptrs[cid] = {}
        for rfpath in rfpaths_png:
            if pos_filter is not None:
                fname_img = rfpath.split("/")[-1]
                if pos_lookup.get(fname_img) != pos_filter:
                    continue
            img_ptrs[cid][idx] = rfpath
            idx += 1

    return img_ptrs
