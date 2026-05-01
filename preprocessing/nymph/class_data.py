"""
python -m preprocessing.nymph.class_data

Creates:
metadata/nymph/class_data.pkl

Structure:
class_data = {
    cid: {
        "subfamily": "<subfamily>",
        "genus": "<genus>",
        "species": "<species>",
        "common_name": "<common_name>",
    },
    ...
}
"""

import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from preprocessing.nymph.species_ids import get_cids_nymph

import pdb


def generate_class_data() -> None:
    cids = get_cids_nymph()
    df_metadata = pd.read_csv(paths["nymph_metadata"])
    species_2_subfamily = (
        df_metadata
        .drop_duplicates(subset=["species"])
        .set_index("species")["subfamily"]
        .to_dict()
    )
    cids2commons = load_pickle(paths["preproc"]["nymph"] / "intermediaries/cids2commons.pkl")

    class_data = {}
    for cid in tqdm(sorted(cids), desc="Generating class data"):

        subfamily = species_2_subfamily[cid]
        if subfamily == "moth":
            subfamily = None

        class_data[cid] = {
            "subfamily": subfamily,
            "genus": cid.split("_")[0],
            "species": cid,
            "common_name": cids2commons[cid],
        }

    save_pickle(class_data, paths["metadata"]["nymph"] / "class_data.pkl")

def main() -> None:
    print("Building class data...")
    generate_class_data()
    print("Class data complete")


if __name__ == "__main__":
    main()