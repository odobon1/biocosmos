import os

from utils.utils import paths

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

    sids = sorted(os.listdir(paths["nymph_imgs"]))

    sids_known = []

    for sid in sids:
        if not is_odd(sid) or sid in NON_ALPHA_EXCEPTIONS_NYMPH:
            sids_known.append(sid)

    return sorted(sids_known)
