import os
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths
from utils.data import truncate_subspecies

import pdb


def get_cids_lepid():

    cids = {
        "hedylidae": set(),
        "hesperiidae": set(),
        "lycaenidae": set(),
        "nymphalidae": set(),
        "papilionidae": set(),
        "pieridae": set(),
        "riodinidae": set(),
    }
    
    for family in cids.keys():
        for cid in os.listdir(paths["imgs"]["lepid"] / family):
            cid = truncate_subspecies(cid)
            cids[family].add(cid)

        cids[family] = sorted(cids[family])
    
    return cids