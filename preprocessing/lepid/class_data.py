"""
python -m preprocessing.lepid.class_data

Creates:
metadata/lepid/class_data.pkl

Structure:
class_data = {
    cid: {
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
from preprocessing.lepid.species_ids import get_cids_lepid

import pdb


def generate_class_data() -> None:
    cids = get_cids_lepid()
    df_metadata_tax = pd.read_csv(paths["lepid_metadata_tax"])
    genus_2_tax = (
        df_metadata_tax
        .drop_duplicates(subset=["genus"])
        .set_index("genus")[["subfamily", "tribe"]]
        .to_dict(orient="index")
    )
    cids2commons = load_pickle(paths["preproc"]["lepid"] / "intermediaries/cids2commons.pkl")

    class_data = {}
    for family in tqdm(cids.keys(), desc="Generating class data"):
        genus_2_cids = {}
        for cid in cids[family]:
            genus = cid.split("_")[0]
            genus_2_cids.setdefault(genus, []).append(cid)

        for genus in sorted(genus_2_cids.keys()):
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

            for cid in sorted(genus_2_cids[genus]):
                class_data[cid] = {
                    "family": family,
                    "subfamily": subfamily,
                    "tribe": tribe,
                    "genus": genus,
                    "species": cid,
                    "common_name": cids2commons[cid],
                }

    save_pickle(class_data, paths["metadata"]["lepid"] / "class_data.pkl")

def main() -> None:
    print("Building class data...")
    generate_class_data()
    print("Class data complete")


if __name__ == "__main__":
    main()