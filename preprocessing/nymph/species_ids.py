import os
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths
from utils.phylo import PhyloVCV

import pdb


def get_sids_nymph():

    NON_ALPHA_EXCEPTIONS_NYMPH = ["polygonia_c-aureum", "polygonia_c-album", "nymphalis_l-album"]

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

    for sid in sids:
        if not is_odd(sid) or sid in NON_ALPHA_EXCEPTIONS_NYMPH:
            sids_known.append(sid)

    return sids_known

def get_sids_phylo_nymph():

    sids_known = get_sids_nymph()

    pvcv = PhyloVCV(dataset="nymph")
    sids_phylo_tree = pvcv.get_sids()

    sids_phylo = []  # sids on the phylogenetic tree that are also in the known species list
    for sid_p in sids_phylo_tree:
        if sid_p in sids_known:
            sids_phylo.append(sid_p)

    return sids_phylo