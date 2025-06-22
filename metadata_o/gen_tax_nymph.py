"""
Must be run on HiPerGator
"""

import os
import glob
import pandas as pd
from tqdm import tqdm

from utils import paths, read_pickle, write_pickle

import pdb


sids = read_pickle(paths["metadata_o"] / "species_ids/known.pkl")
nymph_metadata = pd.read_csv(paths["nymph_metadata"])
unique_species = nymph_metadata["species"].unique().tolist()

metadata = {
    "found" : {},
    "missing" : set(),
}

dirpath_sids = paths["nymph"] / "images"
for sid in tqdm(sids):
    
    if sid in unique_species:
    
        dirpath_sid = dirpath_sids / sid

        # get number of images in directory
        png_files = glob.glob(f"{dirpath_sid}/*.png")
        num_imgs = len(png_files)
        
        png_files = [png_file.rsplit("/", 1)[-1] for png_file in png_files]  # full filepath --> filename

        df_metadata_sid = nymph_metadata[nymph_metadata["species"] == sid]  # metadata subset on species
        n_rows_metadata_sid = len(df_metadata_sid)
        
        png_files_metadata = df_metadata_sid["mask_name"].tolist()  # png filenames from "mask_name" field extracted as list
        
        num_imgs_matched = 0
        for png_file in png_files:
            if png_file in png_files_metadata:
                num_imgs_matched += 1

        metadata["found"][sid] = {
            "tax" : {
                "subfamily" : df_metadata_sid["subfamily"].iloc[0],
                "genus" : df_metadata_sid["species"].iloc[0].split("_")[0],
                "species" : df_metadata_sid["species"].iloc[0].split("_")[-1],
            },
            "meta" : {
                "num_imgs" : num_imgs,  # number of species images in the directory
                "num_rows_metadata" : n_rows_metadata_sid,  # number of rows in metadata corresponding to species
                "num_imgs_matched" : num_imgs_matched,  # number of images in directory that are found in metadata (as per *.png filename)
            },
        }
        
    else:
        
        metadata["missing"].add(sid)

write_pickle(metadata, paths["metadata_o"] / "tax/nymph.pkl")
