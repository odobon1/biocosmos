"""
Takes metadata/tax/nymph structure and produces metadata/rank_keys/nymph structure
"""

from bidict import bidict

from utils import paths, read_pickle, write_pickle

import pdb


rank_keys_nymph = {
    "genus" : bidict(),
    "species" : bidict(),
}

"""
`rank_keys_nymph` Structure:

rank_keys_nymph = {
    "species" : bidict(
        sid0 : species_rank_key0,
        sid1 : species_rank_key1,
        sid2 : ...,
        ...
    ),
    "genus" : bidict(
        genus0 : genus_rank_key0,
        genus1 : genus_rank_key1,
        genus2 : ...,
        ...
    ),
}
"""

tax_nymph = read_pickle(paths["metadata_o"] / "tax/nymph.pkl")

for rkey_species, sid in enumerate(tax_nymph["found"].keys()):

    rank_keys_nymph["species"][sid] = rkey_species  # species uses sid bc over 10% of the species have shared epithets (i.e. different genus, same species epithet) i.e. different rkey for each sid at the species level
    
    genus_str = tax_nymph["found"][sid]["tax"]["genus"]
    if genus_str not in rank_keys_nymph["genus"].keys():

        rkey_genus = len(rank_keys_nymph["genus"].keys())
        rank_keys_nymph["genus"][genus_str] = rkey_genus

write_pickle(rank_keys_nymph, paths["metadata_o"] / "rank_keys/nymph.pkl")
