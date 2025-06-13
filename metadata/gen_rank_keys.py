"""
Takes metadata/tax/nymph structure and produces metadata/rank_keys/nymph structure
"""

from bidict import bidict

from utils import dirpaths, read_pickle, write_pickle


rank_keys_nymph = {
    "genus" : bidict(),
    "species" : bidict(),
}

tax_nymph = read_pickle(dirpaths["repo_oli"] / "metadata/tax/nymph.pkl")

genus_strs = []
species_strs = []

for s in tax_nymph["found"].keys():

    genus = tax_nymph["found"][s]["tax"]["genus"]
    species = tax_nymph["found"][s]["tax"]["species"]

    if genus not in genus_strs:
        genus_strs.append(genus)
    if species not in species_strs:
        species_strs.append(species)

genus_strs.sort()
species_strs.sort()

def generate_rank_keys(rank_keys, rank, rank_strs):

    for rank_idx, rank_str in enumerate(rank_strs):
        
        rank_keys[rank][rank_str] = rank_idx

generate_rank_keys(rank_keys_nymph, "genus", genus_strs)
generate_rank_keys(rank_keys_nymph, "species", species_strs)

write_pickle(rank_keys_nymph, dirpaths["repo_oli"] / "metadata/rank_keys/nymph.pkl")
