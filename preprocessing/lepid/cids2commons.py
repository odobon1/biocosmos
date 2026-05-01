"""
python -m preprocessing.lepid.cids2commons
"""

from itertools import chain

from utils.utils import paths, save_pickle
from preprocessing.common.cid2commons import build_cids2commons
from preprocessing.lepid.species_ids import get_cids_lepid


MAX_WORKERS = 16

def generate_cids2commons() -> None:
    cids = get_cids_lepid()
    cids = list(chain.from_iterable(cids.values()))  # convert dict of lists to flat list
    fpath_cids2commons = paths["preproc"]["lepid"] / "intermediaries/cids2commons.pkl"
    cids2commons = build_cids2commons(cids, max_workers=MAX_WORKERS)
    save_pickle(cids2commons, fpath_cids2commons)

def main() -> None:
    print("Building cids2commons mapping...")
    generate_cids2commons()
    print("cids2commons mapping complete")


if __name__ == "__main__":
    main()