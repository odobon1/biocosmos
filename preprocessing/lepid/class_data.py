"""
python -m preprocessing.lepid.class_data

Creates:
metadata/lepid/class_data.pkl

Structure:
class_data = {
    sid: {
        "family": "<family>",
        "subfamily": "<subfamily>",
        "tribe": "<tribe>",
        "genus": "<genus>",
        "common_name": "<common_name>",
        # "n_imgs": <number_of_images_in_directory>,
    },
    ...
}
"""

from tqdm import tqdm  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from preprocessing.lepid.species_ids import get_sids_lepid

import pdb


sids = get_sids_lepid()
df_metadata_tax = pd.read_csv(paths["lepid_metadata_tax"])
sids2commons = load_pickle(paths["preproc"]["lepid"] / "intermediaries/sids2commons.pkl")

class_data = {}
for family in tqdm(sids.keys(), desc="Generating class data"):
    genera_family = sorted(set([sid.split("_")[0] for sid in sids[family]]))
    for genus in genera_family:
        df_metadata_genus = df_metadata_tax[df_metadata_tax["genus"] == genus]
        if df_metadata_genus.empty:
            subfamily = None
            tribe = None
        else:
            subfamily = df_metadata_genus["subfamily"].iloc[0]
            if subfamily == "x":
                subfamily = None
            tribe = df_metadata_genus["tribe"].iloc[0]
            if tribe == "x":
                tribe = None

        sids_genus = sorted(set([sid for sid in sids[family] if sid.split("_")[0] == genus]))
        for sid in sids_genus:
            class_data[sid] = {
                "family": family,
                "subfamily": subfamily,
                "tribe": tribe,
                "genus": genus,
                "common_name": sids2commons[sid],
                # "n_imgs": n_imgs,  # needs to be computed after ventral images have been removed (also need to account for subspecies dirs!)
            }

save_pickle(class_data, paths["metadata"]["lepid"] / "class_data.pkl")