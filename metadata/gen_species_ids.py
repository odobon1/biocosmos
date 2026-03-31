import os
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths, save_pickle
from utils.data import gbif_common_name

import pdb


non_alpha_valids = ["polygonia_c-aureum", "polygonia_c-album", "nymphalis_l-album"]

def is_odd(sid):
    
    genus, species_epithet = sid.split("_", 1)
    
    if not genus or not species_epithet:
        return True
        
    for ch in sid:
        if not (ch.isalpha() or ch == "_"):
            return True
        
    return False

sids = [sid for sid in os.listdir(paths["nymph_imgs"])]

sids_known = []
sids_unknown = []

for sid in sids:
    odd_name = is_odd(sid)
    
    if odd_name and sid not in non_alpha_valids:
        sids_unknown.append(sid)
    else:
        sids_known.append(sid)

dpath_species_ids = paths["metadata"] / "species_ids"
save_pickle(sids, dpath_species_ids / "all.pkl")
save_pickle(sids_known, dpath_species_ids / "known.pkl")
save_pickle(sids_unknown, dpath_species_ids / "unknown.pkl")


################################## PHYLO #################################

from utils.phylo import PhyloVCV

pvcv       = PhyloVCV()
sids_phylo = pvcv.get_sids()

sids_phylo_present = []
sids_phylo_absent  = []
for sid_p in sids_phylo:
    if sid_p in sids_known:
        sids_phylo_present.append(sid_p)
    else:
        sids_phylo_absent.append(sid_p)

"""
sids_phylo_present --- sIDs present in both the phylogenetic tree and the known species list
sids_phylo_absent  --- sIDs present in the phylogenetic tree but absent from the known species list
"""
save_pickle(sids_phylo_present, paths["metadata"] / "species_ids/phylo_present.pkl")
save_pickle(sids_phylo_absent,  paths["metadata"] / "species_ids/phylo_absent.pkl")