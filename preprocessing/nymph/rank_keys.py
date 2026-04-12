"""
Takes metadata/nymph/class_data.pkl structure and produces metadata/nymph/rank_keys.pkl structure
"""

from bidict import bidict  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle


def generate_rank_keys():
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
    rank_keys_nymph = {
        "genus": bidict(),
        "species": bidict(),
    }
    class_data = load_pickle(paths["metadata"]["nymph"] / "class_data.pkl")

    for rkey_species, sid in enumerate(sorted(class_data.keys())):

        rank_keys_nymph["species"][sid] = rkey_species  # different rkey for each sid at the species level

    for rkey_genus, genus_str in enumerate(sorted({class_data[sid]["genus"] for sid in class_data.keys()})):

        rank_keys_nymph["genus"][genus_str] = rkey_genus

    save_pickle(rank_keys_nymph, paths["metadata"]["nymph"] / "rank_keys.pkl")

def main() -> None:
    print("Building rank keys...")
    generate_rank_keys()
    print("Rank keys complete")


if __name__ == "__main__":
    main()