"""
Takes metadata/nymph/class_data.pkl structure and produces metadata/nymph/rank_keys.pkl structure
"""

from bidict import bidict  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle

import pdb


def build_rank_keys():

    rank_keys_nymph = {
        "genus": bidict(),
        "species": bidict(),
    }

    """
    `rank_keys_nymph` Structure:

    rank_keys_nymph = {
        "species": bidict(
            sid0: species_rank_key0,
            sid1: species_rank_key1,
            sid2: ...,
            ...
        ),
        "genus": bidict(
            genus0: genus_rank_key0,
            genus1: genus_rank_key1,
            genus2: ...,
            ...
        ),
    }
    """

    class_data = load_pickle(paths["metadata"]["nymph"] / "class_data.pkl")

    for rkey_species, sid in enumerate(class_data.keys()):

        rank_keys_nymph["species"][sid] = rkey_species  # species uses sid bc over 10% of the species have shared epithets (i.e. different genus, same species epithet) i.e. different rkey for each sid at the species level
        
        genus_str = class_data[sid]["genus"]
        if genus_str not in rank_keys_nymph["genus"].keys():

            rkey_genus = len(rank_keys_nymph["genus"].keys())
            rank_keys_nymph["genus"][genus_str] = rkey_genus

    save_pickle(rank_keys_nymph, paths["metadata"]["nymph"] / "rank_keys.pkl")

def main() -> None:
    print("Building rank keys...")
    build_rank_keys()
    print("Rank keys complete")


if __name__ == "__main__":
    main()