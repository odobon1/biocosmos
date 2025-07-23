"""
Must be run on HiPerGator
"""

import os

from utils import paths, load_pickle, save_pickle


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
        
save_pickle(sids, paths["metadata_o"] / "species_ids/all.pkl")
save_pickle(sids_known, paths["metadata_o"] / "species_ids/known.pkl")
save_pickle(sids_unknown, paths["metadata_o"] / "species_ids/unknown.pkl")
