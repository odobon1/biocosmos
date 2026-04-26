"""
python -m preprocessing.cub.class_data

Creates:
metadata/cub/class_data.pkl

Structure:
class_data = {
    sid: {
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
import os

from utils.utils import paths, save_pickle
from preprocessing.cub.splits_utils import download_splits, extract_splits, SPLITS_URL
from preprocessing.cub.metadata import build_taxonomy_df, load_common_names, assign_split
from preprocessing.cub.mapping import SPLITS_FILES

def generate_class_data(df_metadata: pd.DataFrame) -> None:

    class_data = {}

    for index, row in df_metadata.iterrows():
        sid = f"{row['genus']} {row['species'].lower()}"
        if sid not in class_data: 
            sid_data = {}
            sid_data['order'] = row['order']
            sid_data['family'] = row['family']
            sid_data['genus'] = row['genus']
            sid_data['species'] = row['species']
            
            sid_data['common_name'] = row['common_name']
            sid_data['rdpath_imgs'] = f"images/{row['class_name']}"
            
            sid_data['split'] = row['split']
            class_data[sid] = sid_data

    save_pickle(class_data, paths["metadata"]["cub"] / "class_data.pkl")

def main() -> None:
    print("Downloading Attribute Splits...")
    download_splits(SPLITS_URL, paths['dpath_cub'] / "xlsa17.zip")

    print("Extracting Attribute Splits...")
    extract_splits(paths['dpath_cub'] / "xlsa17.zip", paths['dpath_cub'])

    mat_path = os.path.join(paths['dpath_cub'], "xlsa17", "data", "CUB", "att_splits.mat")

    print("Loading class names...")
    common_names = load_common_names(mat_path)

    print(f"Querying GBIF for {len(common_names)} species...")
    taxa_df, failed = build_taxonomy_df(common_names)

    if len(failed): 
        print("There are some failed datasets")    

    for split in ['train', 'val', 'test']:
        assign_split(taxa_df, os.path.join('cub_test', "xlsa17", "data", "CUB", SPLITS_FILES[split]), split)

    print("Building class data...")

    generate_class_data(taxa_df)
    print("Class data complete")


if __name__ == "__main__":
    main()