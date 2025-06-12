import pickle
from pathlib import Path


# config
machine = "pace"
# machine = "hpg"

if machine == "pace":
    dirpath_repo_oli = Path("/home/hice1/odobon3/Documents/biocosmos")
elif machine == "hpg":
    dirpath_biocosmos = Path("/blue/arthur.porto-biocosmos")
    # dirpath_repo_oli = 

def write_pickle(obj, picklepath):
    with open(picklepath, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(picklepath):
    with open(picklepath, "rb") as f:
        obj = pickle.load(f)
    return obj
