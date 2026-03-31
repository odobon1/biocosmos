"""
python -m tools.gen_sids2commons
"""

from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle
from utils.data import gbif_common_name


# Nymphalidae
FPATH_SIDS = paths["metadata"] / "species_ids/known.pkl"
FPATH_SIDS2COMMONS = paths["metadata"] / "sids2commons/sids2commons.pkl"


sids = load_pickle(FPATH_SIDS)

sids2commons = {}
for sid in tqdm(sids, desc="Retrieving Common Names"):
    common = gbif_common_name(sid.replace("_", " "))
    sids2commons[sid] = common

save_pickle(sids2commons, FPATH_SIDS2COMMONS)