"""
python -m preprocessing.nymph.sids2commons
"""

from utils.utils import paths, save_pickle
from preprocessing.common.sid2commons import build_sids2commons
from preprocessing.nymph.species_ids import get_sids_nymph


MAX_WORKERS = 16

def generate_sids2commons() -> None:
    sids = get_sids_nymph()
    fpath_sids2commons = paths["preproc"]["nymph"] / "intermediaries/sids2commons.pkl"
    sids2commons = build_sids2commons(sids, max_workers=MAX_WORKERS)
    save_pickle(sids2commons, fpath_sids2commons)
    
def main() -> None:
    print("Building sids2commons mapping...")
    generate_sids2commons()
    print("sids2commons mapping complete")


if __name__ == "__main__":
    main()