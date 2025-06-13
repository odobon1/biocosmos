import os
import glob
import pandas as pd
from tqdm import tqdm

from utils import dirpath_biocosmos, read_pickle, write_pickle


img_dirs = read_pickle(dirpath_biocosmos / "odobon3.gatech/biocosmos/tax_tree/metadata/img_dirs/known.pkl")

dirpath_img_dirs = dirpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v240606/images"

filepath_metadata_og = dirpath_biocosmos / "data/datasets/nymphalidae_whole_specimen-v240606/metadata/data_meta-nymphalidae_whole_specimen-v240606.csv"
df_metadata_og = pd.read_csv(filepath_metadata_og)

species_metadata_og = df_metadata_og["species"].unique().tolist()

metadata = {
    "found" : {},
    "missing" : set(),
}

for s in tqdm(img_dirs):
    
    if s in species_metadata_og:
    
        dirpath_s = dirpath_img_dirs / s

        # get number of images in directory
        png_files = glob.glob(f"{dirpath_s}/*.png")
        num_imgs = len(png_files)
        
        png_files = [png_file.rsplit("/", 1)[-1] for png_file in png_files]  # full filepath --> filename

        df_metadata_s = df_metadata_og[df_metadata_og["species"] == s]  # metadata subset on species
        num_rows_metadata_s = len(df_metadata_s)
        
        png_files_metadata = df_metadata_s["mask_name"].tolist()  # png filenames from "mask_name" field extracted as list
        
        num_imgs_matched = 0
        for png_file in png_files:
            if png_file in png_files_metadata:
                num_imgs_matched += 1
        
        metadata["found"][s] = {
            "tax" : {
                "subfamily" : df_metadata_s["subfamily"].iloc[0],
                "genus" : df_metadata_s["genus"].iloc[0],
                "species" : df_metadata_s["species"].iloc[0].split("_")[-1],
            },
            "meta" : {
                "num_imgs" : num_imgs,  # number of species images in the directory
                "num_rows_metadata" : num_rows_metadata_s,  # number of rows in metadata corresponding to species
                "num_imgs_matched" : num_imgs_matched,  # number of images in directory that are found in metadata (as per *.png filename)
            },
        }
        
    else:
        
        metadata["missing"].add(s)

write_pickle(metadata, dirpath_biocosmos / "odobon3.gatech/biocosmos/tax_tree/metadata/tax/butterflies_test.pkl")
