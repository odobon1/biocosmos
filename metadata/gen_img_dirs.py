"""
Must be run on HiPerGator
"""

import os

from utils import dirpaths, read_pickle, write_pickle


non_alpha_valids = ["polygonia_c-aureum", "polygonia_c-album", "nymphalis_l-album"]

def is_odd(img_dir):
    
    genus, species_epithet = img_dir.split("_", 1)
    
    if not genus or not species_epithet:
        return True
        
    for ch in img_dir:
        if not (ch.isalpha() or ch == "_"):
            return True
        
    return False

dirpath_img_dirs = dirpaths["biocosmos"] / "data/datasets/nymphalidae_whole_specimen-v240606/images"
img_dirs = [img_dir for img_dir in os.listdir(dirpath_img_dirs)]

img_dirs_known = []
img_dirs_unknown = []

for img_dir in img_dirs:
    odd_name = is_odd(img_dir)
    
    if odd_name and img_dir not in non_alpha_valids:
        img_dirs_unknown.append(img_dir)
    else:
        img_dirs_known.append(img_dir)
        
write_pickle(img_dirs, dirpaths["biocosmos"] / "odobon3.gatech/biocosmos/metadata/img_dirs/all.pkl")
write_pickle(img_dirs_known, dirpaths["biocosmos"] / "odobon3.gatech/biocosmos/metadata/img_dirs/known.pkl")
write_pickle(img_dirs_unknown, dirpaths["biocosmos"] / "odobon3.gatech/biocosmos/metadata/img_dirs/unknown.pkl")
