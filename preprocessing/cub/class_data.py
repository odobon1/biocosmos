"""
python -m preprocessing.cub.class_data

Creates:
metadata/cub/class_data.pkl

Structure:
class_data = {
    cid: {
        "order": "<order>",
        "family": "<family>",
        "genus": "<genus>",
        "species": "<species>",
        "common_name": "<common_name>",
        "rdpath_imgs" : "<image_path>",
        "split": "<test, train, or val>",
    },
    ...
}
"""
import pandas as pd  # type: ignore[import]
from scipy.io import loadmat
from tqdm import tqdm  # type: ignore[import]
import requests  # type: ignore[import]

from utils.utils import paths, save_pickle

import pdb


COMMON_NAME_CORRECTIONS = {
    "Chuck will Widow": "Chuck-will's-widow",
    "Heermann Gull": "Heermann's Gull",
    "Anna Hummingbird": "Anna's Hummingbird",
    "Clark Nutcracker": "Clark's Nutcracker",
    "Geococcyx": "Greater Roadrunner",
    "Baird Sparrow": "Baird's Sparrow",
    "Brewer Sparrow": "Brewer's Sparrow",
    "Henslow Sparrow": "Henslow's Sparrow",
    "Le Conte Sparrow": "Le Conte's Sparrow",
    "Nelson Sharp tailed Sparrow": "Nelson's Sparrow",
    "Barn Swallow": "Barn Swallow",
    "Swainson Warbler": "Swainson's Warbler",
    "Bewick Wren": "Bewick's Wren",
    "Brewer Blackbird": "Brewer's Blackbird",
    "Brandt Cormorant": "Brandt's Cormorant",
    "Scott Oriole": "Scott's Oriole",
    "Sayornis": "Eastern phoebe", # --> issue child
    "Wilson Warbler": "Wilson's Warbler",
}

SKIP_NAMES = {
    "Sayornis",
    "Geococcyx" # we can decide this
}

SPLITS_FILES = {
    "test": "testclasses.txt",
    "train": "trainclasses1.txt",
    "val": "valclasses1.txt"
}

INAT_GBIF_BACKBONE = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c"


def load_common_names(fpath_att_splits: str) -> list[str]:
    """Load and parse class names from the .mat splits file."""
    split_sets = loadmat(fpath_att_splits)
    raw_names = [s[0][0] for s in split_sets['allclasses_names']]

    return pd.DataFrame({
            "class_name": raw_names,
            "common_name": [parse_class_name(r) for r in raw_names],
        })

# load_common_names() helper
def parse_class_name(raw: str) -> str:
    """Convert '001.Black_footed_Albatross' -> 'Black footed Albatross'"""
    return raw.split('.')[1].replace('_', ' ')

def build_df_class_data(df_common_names: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Query GBIF for each name and return a DataFrame and list of failed lookups."""
    common_names = df_common_names["common_name"]
    class_names = df_common_names["class_name"]
    rows = []
    failed = []

    for i in tqdm(range(len(common_names)), desc="Querying GBIF"):
        cname, clname = common_names[i], class_names[i]
        # if cname in SKIP_NAMES: # skipping names which point to genus-level (i.e. have multiple species)
        #     continue
        queried_name = COMMON_NAME_CORRECTIONS.get(cname, cname)
        result = query_gbif(queried_name, INAT_GBIF_BACKBONE)
        # if any of the results are None, query general GBIF API
        if any(v is None for v in result.values()):
            result = query_gbif(queried_name)
        rows.append({"common_name": cname, "class_name": clname, **result})

        if result["species"] is None:
            failed.append(cname)

    return pd.DataFrame(rows), failed

# build_df_class_data() helper
def query_gbif(common_name: str,  dataset_key: str|None = None) -> dict:
    """Query GBIF for taxonomy of a single bird common name."""
    url = "https://api.gbif.org/v1/species/search"
    params = {
        "q": common_name,
        "qField": "VERNACULAR",
        "rank": "SPECIES",
        "status": "ACCEPTED",
        "datasetKey": dataset_key,
        "limit": 5,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    results = r.json().get("results", [])

    birds = [r for r in results if r.get("class") == "Aves"]
    candidates = birds if birds else results

    if candidates:
        hit = candidates[0]
        sci = hit.get("canonicalName")
        genus, species = sci.split(' ')[:2] if sci else (None, None)
        return {
            "order":  hit.get("order"),
            "family": hit.get("family"),
            "genus":  genus,
            "species": species,
        }
    return {"order": None, "family": None, "genus": None, "species": None}

def assign_split(df_class_data: pd.DataFrame, txt_path: str, split: str) -> pd.DataFrame:
    """Assign a split label to rows in df whose class_name appears in the txt file."""
    with open(txt_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    mask = df_class_data['class_name'].isin(classes)
    df_class_data.loc[mask, 'split'] = split
    
    return df_class_data

def generate_class_data(df_class_data: pd.DataFrame) -> None:
    class_data = {}
    for _, row in df_class_data.iterrows():
        order = row['order'].lower()
        family = row['family'].lower()
        genus = row['genus'].lower()
        species = f"{genus}_{row['species'].lower()}"
        common_name = row['common_name'].lower().replace(" ", "_")
        rdpath_imgs = f"images/{row['class_name']}"
        split = row['split']

        cid = common_name

        class_data[cid] = {
            "order": order,
            "family": family,
            "genus": genus,
            "species": species,
            "common_name": common_name,
            "rdpath_imgs": rdpath_imgs,
            "split": split,
        }
    save_pickle(class_data, paths["metadata"]["cub"] / "class_data.pkl")

def main() -> None:
    print("Building class data...")

    fpath_att_splits = paths["data"]["cub"] / "xlsa17/data/CUB/att_splits.mat"
    df_common_names = load_common_names(fpath_att_splits)
    df_class_data, failed = build_df_class_data(df_common_names)
    if len(failed): 
        print("There are some failed datasets")    

    for split in ['train', 'val', 'test']:
        txt_path = paths["data"]["cub"] / "xlsa17/data/CUB" / SPLITS_FILES[split]
        assign_split(df_class_data, txt_path, split)

    generate_class_data(df_class_data)

    print("Class data complete")


if __name__ == "__main__":
    main()