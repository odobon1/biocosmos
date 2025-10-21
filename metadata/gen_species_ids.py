import os
from tqdm import tqdm

from utils import paths, load_pickle, save_pickle
from utils_data import gbif_common_name

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

sids = [sid for sid in os.listdir(paths["nymph"] / "images")]

sids_known = []
sids_unknown = []

for sid in sids:
    odd_name = is_odd(sid)
    
    if odd_name and sid not in non_alpha_valids:
        sids_unknown.append(sid)
    else:
        sids_known.append(sid)
        
sids2commons = {}
for sid in tqdm(sids_known, desc="Retrieving Common Names"):
    common = gbif_common_name(sid.replace("_", " "))
    sids2commons[sid] = common

dpath_species_ids = paths["metadata"] / "species_ids"
save_pickle(sids, dpath_species_ids / "all.pkl")
save_pickle(sids_known, dpath_species_ids / "known.pkl")
save_pickle(sids_unknown, dpath_species_ids / "unknown.pkl")
save_pickle(sids2commons, dpath_species_ids / "sids2commons.pkl")
