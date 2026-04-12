"""
python -m preprocessing.nymph.class_data

Creates:
metadata/nymph/class_data.pkl

Structure:
class_data = {
    sid: {
        "subfamily": "<subfamily>",
        "genus": "<genus>",
        "common_name": "<common_name>",
    },
    ...
}
"""

import pandas as pd  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from preprocessing.nymph.species_ids import get_sids_nymph

import pdb


def build_class_data() -> None:
    sids = get_sids_nymph()
    df_metadata = pd.read_csv(paths["nymph_metadata"])
    sids2commons = load_pickle(paths["preproc"]["nymph"] / "intermediaries/sids2commons.pkl")

    class_data = {}
    for sid in tqdm(sids, desc="Generating class data"):

        df_metadata_sid = df_metadata[df_metadata["species"] == sid]  # metadata subset on species

        subfamily = df_metadata_sid["subfamily"].iloc[0]
        if subfamily == "moth":
            subfamily = None

        class_data[sid] = {
            "subfamily": subfamily,
            "genus": sid.split("_")[0],
            "common_name": sids2commons[sid],
        }

    save_pickle(class_data, paths["metadata"]["nymph"] / "class_data.pkl")

def main() -> None:
    print("Building class data...")
    build_class_data()
    print("Class data complete")


if __name__ == "__main__":
    main()