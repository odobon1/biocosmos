import requests
import os
import pandas as pd
from scipy.io import loadmat
from mapping import COMMON_NAME_CORRECTIONS, SKIP_NAMES

from utils.utils import paths


def parse_class_name(raw: str) -> str:
    """Convert '001.Black_footed_Albatross' -> 'Black footed Albatross'"""
    return raw.split('.')[1].replace('_', ' ')


def load_common_names(mat_path: str) -> list[str]:
    """Load and parse class names from the .mat splits file."""
    split_sets = loadmat(mat_path)
    raw_names = [s[0][0] for s in split_sets['allclasses_names']]

    return pd.DataFrame({
            "class_name": raw_names,
            "common_name": [parse_class_name(r) for r in raw_names],
        })

def query_gbif(common_name: str) -> dict:
    """Query GBIF for taxonomy of a single bird common name."""
    url = "https://api.gbif.org/v1/species/search"
    params = {
        "q": common_name,
        "qField": "VERNACULAR",
        "rank": "SPECIES",
        "status": "ACCEPTED",
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

def build_taxonomy_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Query GBIF for each name and return a DataFrame and list of failed lookups."""
    common_names = df['common_name']
    class_names = df['class_name']
    rows = []
    failed = []

    for i in range(len(common_names)):
        cname, clname = common_names[i], class_names[i]
        if cname in SKIP_NAMES: # skipping names which point to genus-level (i.e. have multiple species)
            continue
        queried_name = COMMON_NAME_CORRECTIONS.get(cname, cname)
        result = query_gbif(queried_name)
        rows.append({"common_name": cname, 'class_name': clname, **result})

        if result["species"] is None:
            failed.append(cname)

    return pd.DataFrame(rows), failed

def assign_split(df: pd.DataFrame, txt_path: str, split: str) -> pd.DataFrame:
    """Assign a split label to rows in df whose class_name appears in the txt file."""
    with open(txt_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    mask = df['class_name'].isin(classes)
    df.loc[mask, 'split'] = split
    
    return df

# def main():
#     # TODO: need to test the full download + generation of class_data
#     print("[INFO] Downloading Attribute Splits...")
#     download_splits(SPLITS_URL, paths['dpath_cub'])

#     print("[INFO] Extracting Attribute Splits...")
#     extract_splits(paths['dpath_cub'], paths['dpath_cub'])

#     mat_path = os.path.join(paths['dpath_cub'], "xlsa17", "data", "CUB", "att_splits.mat")
#     output_path = paths["cub_metadata_gen"]

#     print("[INFO] Loading class names...")
#     common_names = load_common_names(mat_path)

#     print(f"[INFO] Querying GBIF for {len(common_names)} species...")
#     df, failed = build_taxonomy_df(common_names)

#     report_failures(failed)

#     print(f"[INFO] Saving to {output_path}...")
#     df.to_csv(output_path, index=False)
#     print("[INFO] Done.")


# if __name__ == "__main__":
#     main()
