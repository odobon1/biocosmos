import os
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths
from utils.phylo import PhyloVCV
from utils.data import before_second_underscore

import pdb


def get_sids_lepid(truncate_subspecies=True):

    FAMILIES = ["hedylidae", "hesperiidae", "lycaenidae", "nymphalidae", "papilionidae", "pieridae", "riodinidae"]

    sids = set()
    for family in FAMILIES:
        for sid in os.listdir(paths["lepid_imgs"] / family):
            if truncate_subspecies:
                sid = before_second_underscore(sid)
            sids.add(sid)
    return list(sids)