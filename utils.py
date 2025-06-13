import pickle
from pathlib import Path


# config params
# MACHINE = "pace"
MACHINE = "hpg"

if MACHINE == "pace":
    dirpaths = {
        "repo_oli" : Path("/home/hice1/odobon3/Documents/biocosmos"),
        "vlm4bio" : Path("VLM4Bio/datasets")
    }
elif MACHINE == "hpg":
    dirpaths = {
        "biocosmos" : Path("/blue/arthur.porto-biocosmos"),
        "repo_oli" : Path("/blue/arthur.porto-biocosmos/odobon3.gatech/biocosmos"),
        "vlm4bio" : Path("/blue/arthur.porto-biocosmos/data/datasets/VLM4Bio"),
    }

def write_pickle(obj, picklepath):
    with open(picklepath, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(picklepath):
    with open(picklepath, "rb") as f:
        obj = pickle.load(f)
    return obj
