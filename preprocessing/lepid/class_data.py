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
        "species": "<species>",
        "common_name": "<common_name>",
    },
    ...
}
"""

from tqdm import tqdm  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from preprocessing.lepid.species_ids import get_sids_lepid

import pdb


def generate_class_data() -> None:
    sids = get_sids_lepid()
    df_metadata_tax = pd.read_csv(paths["lepid_metadata_tax"])
    genus_2_tax = (
        df_metadata_tax
        .drop_duplicates(subset=["genus"])
        .set_index("genus")[["subfamily", "tribe"]]
        .to_dict(orient="index")
    )
    sids2commons = load_pickle(paths["preproc"]["lepid"] / "intermediaries/sids2commons.pkl")

    class_data = {}
    for family in tqdm(sids.keys(), desc="Generating class data"):
        genus_2_sids = {}
        for sid in sids[family]:
            genus = sid.split("_")[0]
            genus_2_sids.setdefault(genus, []).append(sid)

        for genus in sorted(genus_2_sids.keys()):
            tax_genus = genus_2_tax.get(genus)
            if tax_genus is None:
                subfamily = None
                tribe = None
            else:
                subfamily = tax_genus["subfamily"]
                if subfamily == "x":
                    subfamily = None
                tribe = tax_genus["tribe"]
                if tribe == "x":
                    tribe = None

            for sid in sorted(genus_2_sids[genus]):
                class_data[sid] = {
                    "family": family,
                    "subfamily": subfamily,
                    "tribe": tribe,
                    "genus": genus,
                    "species": sid,
                    "common_name": sids2commons[sid],
                }

    save_pickle(class_data, paths["metadata"]["lepid"] / "class_data.pkl")

def main() -> None:
    print("Building class data...")
    generate_class_data()
    print("Class data complete")


if __name__ == "__main__":
    main()