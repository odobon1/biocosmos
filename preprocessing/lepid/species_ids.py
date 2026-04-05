import os
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths
from utils.data import before_second_underscore

import pdb


def get_sids_lepid():

    sids = {
        "hedylidae": set(),
        "hesperiidae": set(),
        "lycaenidae": set(),
        "nymphalidae": set(),
        "papilionidae": set(),
        "pieridae": set(),
        "riodinidae": set(),
    }
    
    for family in sids.keys():
        for sid in os.listdir(paths["lepid_imgs"] / family):
            sid = before_second_underscore(sid)
            sids[family].add(sid)

        sids[family] = sorted(sids[family])
    
    return sids