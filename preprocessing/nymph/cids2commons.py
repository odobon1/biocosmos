"""
python -m preprocessing.nymph.cids2commons
"""

from utils.utils import paths, save_pickle
from preprocessing.common.cid2commons import build_cids2commons
from preprocessing.nymph.species_ids import get_cids_nymph


MAX_WORKERS = 16

def generate_cids2commons() -> None:
    cids = get_cids_nymph()
    fpath_cids2commons = paths["preproc"]["nymph"] / "intermediaries/cids2commons.pkl"
    cids2commons = build_cids2commons(cids, max_workers=MAX_WORKERS)
    save_pickle(cids2commons, fpath_cids2commons)
    
def main() -> None:
    print("Building cids2commons mapping...")
    generate_cids2commons()
    print("cids2commons mapping complete")


if __name__ == "__main__":
    main()