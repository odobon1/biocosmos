import pickle
from pathlib import Path


# config params
# MACHINE = "pace"
MACHINE = "hpg"

if MACHINE == "pace":
    paths = {
        "repo_o" : Path("/home/hice1/odobon3/Documents/biocosmos"),
        "vlm4bio" : Path("VLM4Bio/datasets")
    }
elif MACHINE == "hpg":

    dpath_biocosmos = Path("/blue/arthur.porto-biocosmos")
    dpath_repo_o = dpath_biocosmos / "odobon3.gatech/biocosmos"
    dpath_vlm4bio = dpath_biocosmos / "data/datasets/VLM4Bio"
    dpath_nymph = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v240606"
    fpath_nymph_metadata = dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v240606.csv"
    # dpath_nymph = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v250613"
    # fpath_nymph_metadata = dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv"
    dpath_nymph_1 = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v240606"
    fpath_nymph_1_metadata = dpath_nymph_1 / "metadata/data_meta-nymphalidae_whole_specimen-v240606.csv"
    dpath_nymph_2 = dpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v250613"
    fpath_nymph_2_metadata = dpath_nymph_2 / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv"

    paths = {
        "biocosmos" : dpath_biocosmos,
        "repo_o" : dpath_repo_o,
        "metadata_o" : dpath_repo_o / "metadata_o",
        "vlm4bio" : dpath_vlm4bio,
        "nymph" : dpath_nymph_1,
        "nymph_metadata" : fpath_nymph_1_metadata,
        "nymph_1" : dpath_nymph_1,
        "nymph_1_metadata" : fpath_nymph_1_metadata,
        "nymph_2" : dpath_nymph_2,
        "nymph_2_metadata" : fpath_nymph_2_metadata,
    }

def write_pickle(obj, picklepath):
    with open(picklepath, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(picklepath):
    with open(picklepath, "rb") as f:
        obj = pickle.load(f)
    return obj
