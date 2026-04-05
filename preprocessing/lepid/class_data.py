"""
python -m preprocessing.lepid.class_data

Creates:
metadata/lepid/class_data.pkl

Structure:
class_data = {
    sid: {
        "subfamily": "<subfamily>",
        "genus": "<genus>",
        "common_name": "<common_name>",
        "n_imgs": <number_of_images_in_directory>,
    },
    ...
}
"""

import glob
import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]
from typing import List

from utils.utils import paths, load_pickle, save_pickle
from preprocessing.lepid.species_ids import get_sids_lepid

import pdb


sids: List[str] = get_sids_lepid()
genera: List[str] = sorted(set([sid.split("_")[0] for sid in sids]))

df_metadata_lepid = pd.read_csv(paths["lepid_metadata_tax"])
sids2commons = load_pickle(paths["preproc"]["lepid"] / "intermediaries/sids2commons.pkl")

class_data = {}
for genus in tqdm(genera, desc="Generating class data"):
    df_metadata_sid = df_metadata_lepid[df_metadata_lepid["genus"] == genus]  # metadata subset on species
    if df_metadata_sid.empty:
        continue

    family = df_metadata_sid["family"].iloc[0]
    subfamily = df_metadata_sid["subfamily"].iloc[0]
    tribe = df_metadata_sid["tribe"].iloc[0]

    sids_genus = [sid for sid in sids if sid.startswith(genus + "_")]
    for sid in sids_genus:
        class_data[sid] = {
            "family": family,
            "subfamily": subfamily,
            "tribe": tribe,
            "genus": genus,
            "common_name": sids2commons[sid],
        # "n_imgs": n_imgs,  # needs to be computed after dorsal have been extracted (also need to account for subspecies dirs!)
    }

save_pickle(class_data, paths["metadata"]["lepid"] / "class_data.pkl")