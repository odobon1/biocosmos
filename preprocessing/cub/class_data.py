"""
python -m preprocessing.cub.class_data
# NOTE: must generate metatadata via preprocessing.cub.metadata before running class_data 

# TODO: skip the metadata, generate directly

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
        "image_path" : "<image_path>",
    },
    ...
}
"""
import pandas as pd  # type: ignore[import]

from utils.utils import paths, save_pickle

def generate_class_data() -> None:

    df_metadata = pd.read_csv(paths["cub_metadata_gen"])
    class_data = {}

    for index, row in df_metadata.iterrows():
        sid = f"{row['genus']} {row['species'].lower()}"
        if sid not in class_data: 
            sid_data = {}
            sid_data['order'] = row['order']
            sid_data['family'] = row['family']
            sid_data['genus'] = row['genus']
            sid_data['species'] = row['species']
            
            image_path = f"images/{row['image_path'].split('/')[0]}"
            sid_data['common_name'] = row['commonName']
            sid_data['rdpath_imgs'] = image_path
            
            sid_data['split'] = row['split']
            class_data[sid] = sid_data

    save_pickle(class_data, paths["metadata"]["cub"] / "class_data.pkl")

def main() -> None:
    print("Building class data...")
    generate_class_data()
    print("Class data complete")


if __name__ == "__main__":
    main()