"""
python -m preprocessing.lepid.sids2commons
"""

from itertools import chain

from utils.utils import paths, save_pickle
from preprocessing.common.sid2commons import build_sids2commons
from preprocessing.lepid.species_ids import get_sids_lepid


MAX_WORKERS = 16

def generate_sids2commons() -> None:
    sids = get_sids_lepid()
    sids = list(chain.from_iterable(sids.values()))  # convert dict of lists to flat list
    fpath_sids2commons = paths["preproc"]["lepid"] / "intermediaries/sids2commons.pkl"
    sids2commons = build_sids2commons(sids, max_workers=MAX_WORKERS)
    save_pickle(sids2commons, fpath_sids2commons)

def main() -> None:
    print("Building sids2commons mapping...")
    generate_sids2commons()
    print("sids2commons mapping complete")


if __name__ == "__main__":
    main()