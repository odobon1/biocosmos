"""
python -m preprocessing.lepid.class_data

Creates:
metadata/lepid/class_data.pkl

Structure:
class_data = {
    sid: {
        "family": "<family>",
        "genus": "<genus>",
        "common_name": "<common_name>",
        # "n_imgs": <number_of_images_in_directory>,
    },
    ...
}
"""

from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from preprocessing.lepid.species_ids import get_sids_lepid

import pdb


sids = get_sids_lepid()
sids2commons = load_pickle(paths["preproc"]["lepid"] / "intermediaries/sids2commons.pkl")

class_data = {}
for family in tqdm(sids.keys(), desc="Generating class data"):
    for sids_family in sids[family]:
        class_data[sids_family] = {
            "family": family,
            "genus": sids_family.split("_")[0],
            "common_name": sids2commons[sids_family],
            # "n_imgs": n_imgs,  # needs to be computed after ventral images have been removed (also need to account for subspecies dirs!)
        }

save_pickle(class_data, paths["metadata"]["lepid"] / "class_data.pkl")