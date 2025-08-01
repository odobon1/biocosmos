import glob
from tqdm import tqdm
import os

from utils import paths, load_pickle, save_pickle

import pdb


""" CONFIG PARAMS """

SPLIT_NAME = "S29-0"


if not os.path.isdir(paths["metadata_o"] / f"splits/{SPLIT_NAME}"):
    error_msg = f"Split '{SPLIT_NAME}' doesn't exist!"
    raise ValueError(error_msg)
else:
    dpath_data_indexes = paths["metadata_o"] / f"data_indexes/{SPLIT_NAME}"
    os.makedirs(dpath_data_indexes, exist_ok=True)

# load species id's
sids = load_pickle(paths["metadata_o"] / "species_ids/known.pkl")

"""
`img_ptrs` structure:

img_ptrs = {
    sid0: {
        0: fpath_img_s0_0,
        1: fpath_img_s0_1,
        ...,
    },
    sid1: {
        ...,
    },
    ...
}
"""

img_ptrs = {}

# iterate through sids, fetch image filenames, assign to indexes, add to img_ptrs structure
for sid in tqdm(sids):
    
    img_ptrs[sid] = {}

    dpath_imgs_sid = dpath_imgs = paths["nymph_imgs"] / sid
    ffpaths_png = glob.glob(f"{dpath_imgs_sid}/*.png")
    rfpaths_png = [png_file.split("images/", 1)[1] for png_file in ffpaths_png]  # full filepath --> relative filepath

    for i, rfpath in enumerate(rfpaths_png):
        img_ptrs[sid][i] = rfpath


for split in ["train", "id_val", "id_test", "ood_val", "ood_test"]:

    data_index = {
        "sids": [],
        "rfpaths": [],
    }

    skeys_split = load_pickle(paths["metadata_o"] / f"splits/{SPLIT_NAME}/{split}.pkl")

    for skey in skeys_split:
        sid = skey[0]
        sidx = skey[1]

        data_index["sids"].append(sid)

        rfpath = img_ptrs[sid][sidx]
        data_index["rfpaths"].append(rfpath)

    save_pickle(data_index, paths["metadata_o"] / f"data_indexes/{SPLIT_NAME}/{split}.pkl")
