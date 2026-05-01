import os

from utils.utils import paths

import pdb


def get_cids_nymph():

    NON_ALPHA_EXCEPTIONS_NYMPH = ["polygonia_c-aureum", "polygonia_c-album", "nymphalis_l-album"]

    def is_odd(cid):
        genus, species_epithet = cid.split("_", 1)
        if not genus or not species_epithet:
            return True
        for ch in cid:
            if not (ch.isalpha() or ch == "_"):
                return True
        return False

    cids = sorted(os.listdir(paths["imgs"]["nymph"]))

    cids_known = []

    for cid in cids:
        if not is_odd(cid) or cid in NON_ALPHA_EXCEPTIONS_NYMPH:
            cids_known.append(cid)

    return sorted(cids_known)
