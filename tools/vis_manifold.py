"""
python -m tools.vis_manifold
"""

import torch  # type: ignore[import]
from torch.amp import autocast  # type: ignore[import]
import numpy as np  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from sklearn.manifold import TSNE  # type: ignore[import]
from sklearn.decomposition import PCA  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]
import random

from models import VLMWrapper
from utils_config import get_config_eval
from utils_data import spawn_dataloader, spawn_indexes, sid_to_genus
from utils import paths, get_text_preps

def get_embs_and_labels(modelw, dataloader, device, mixed_prec):
    """
    Iterate through dataloader to generate image embeddings and retrieve labels (genera + species IDs)
    """
    modelw.model.eval()
    
    embs_all   = []
    genera_all = []
    sids_all   = []

    print("Generating embeddings...")
    with torch.no_grad():
        for imgs_b, _, _, targ_data_b in tqdm(dataloader):
            imgs_b = imgs_b.to(device, non_blocking=True)
            
            # generate embeddings
            if mixed_prec:
                with autocast(device_type=device.type):
                    embs_img_b = modelw.embed_images(imgs_b)
            else:
                embs_img_b = modelw.embed_images(imgs_b)
            
            embs_all.append(embs_img_b.cpu().numpy())
            for item in targ_data_b:
                sid = item['sid']
                sids_all.append(sid)
                genera_all.append(sid_to_genus(sid))

    embs_all = np.concatenate(embs_all, axis=0)

    return embs_all, genera_all, sids_all

def compute_tsne(embeddings, perplexity=30, random_state=42):
    """
    Reduce embedding dimensionality to 2D via t-SNE
    """
    print(f"Running t-SNE on {embeddings.shape[0]} samples (dim={embeddings.shape[1]})...")
    tsne = TSNE(
        perplexity  =perplexity, 
        random_state=random_state, 
    )
    embs_2d = tsne.fit_transform(embeddings)
    return embs_2d

def compute_pca(embeddings):
    """
    Reduce embedding dimensionality to 2D via PCA
    """
    print(f"Running PCA on {embeddings.shape[0]} samples (dim={embeddings.shape[1]})...")
    pca     = PCA(n_components=2)
    embs_2d = pca.fit_transform(embeddings)
    return embs_2d

def plot_projection(embs_2d, labels, title, fpath_plot, method):
    print(f"Plotting {method} projection...")
    
    # convert labels to numpy array for indexing convenience
    labels_np     = np.array(labels)
    unique_labels = sorted(list(set(labels)))
    n_classes     = len(unique_labels)
    
    plt.figure(figsize=(16, 12))

    colors = list(sns.color_palette("husl", n_classes))
    random.seed(42)
    random.shuffle(colors)  # shuffle colors to mitigate similar colors for nearby classes

    for i, label in enumerate(unique_labels):
        mask = (labels_np == label)  # boolean mask for current label
        
        plt.scatter(
            embs_2d[mask, 0], 
            embs_2d[mask, 1], 
            c         =[colors[i]], 
            label     =label,
            s         =60,
            alpha     =0.8,
            edgecolors='w',  # white edge around dots
            linewidths=0.5
        )
    
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel(f"{method} Dimension 1", fontsize=14)
    plt.ylabel(f"{method} Dimension 2", fontsize=14)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    print(f"Saving plot to {fpath_plot}")
    plt.savefig(fpath_plot, dpi=300, bbox_inches="tight")
    plt.close()

def get_dataloader(cfg, splitset_name, modelw):
    text_preps    = get_text_preps(cfg.text_preps)
    index_data, _ = spawn_indexes(split_name=cfg.split_name, splitset_name=splitset_name)
    dataloader, _ = spawn_dataloader(
        index_data    =index_data,
        text_preps    =text_preps,
        config        =cfg,
        shuffle       =False,
        drop_last     =False,
        img_pp        =modelw.img_pp_val,
        use_dv_sampler=False,
    )
    return dataloader

def main():

    # component of plot title that appears in parentheses, set to None for no tag
    # TAG = "base"
    TAG = None

    cfg    = get_config_eval(verbose=True)
    modelw = VLMWrapper.build(cfg, verbose=True)
    
    print(f"Preparing ID data...")
    dataloader_id = get_dataloader(cfg, "id_val", modelw)
    print(f"Preparing OOD data...")
    dataloader_ood = get_dataloader(cfg, "ood_val", modelw)

    # get embeddings and labels
    embs_id,  genera_id,  sids_id  = get_embs_and_labels(modelw, dataloader_id,  cfg.device, cfg.hw.mixed_prec)
    embs_ood, genera_ood, sids_ood = get_embs_and_labels(modelw, dataloader_ood, cfg.device, cfg.hw.mixed_prec)

    # compute t-SNE projections
    proj_tsne_id  = compute_tsne(embs_id)
    proj_tsne_ood = compute_tsne(embs_ood)
    # compute PCA projections
    proj_pca_id  = compute_pca(embs_id)
    proj_pca_ood = compute_pca(embs_ood)
    
    dpath_plots            = paths["root"] / cfg.rdpath_trial / "plots"
    fpath_tsne_id_genera   = dpath_plots / f"tsne_id_genera.png"
    fpath_tsne_id_species  = dpath_plots / f"tsne_id_species.png"
    fpath_tsne_ood_genera  = dpath_plots / f"tsne_ood_genera.png"
    fpath_tsne_ood_species = dpath_plots / f"tsne_ood_species.png"
    fpath_pca_id_genera    = dpath_plots / f"pca_id_genera.png"
    fpath_pca_id_species   = dpath_plots / f"pca_id_species.png"
    fpath_pca_ood_genera   = dpath_plots / f"pca_ood_genera.png"
    fpath_pca_ood_species  = dpath_plots / f"pca_ood_species.png"
    
    if TAG is not None:
        tag = f" ({TAG})"
    else:
        tag = ""

    title_tsne_id_genera   = f"t-SNE: ID validation, colored by genus{tag}"
    title_tsne_id_species  = f"t-SNE: ID validation, colored by species{tag}"
    title_tsne_ood_genera  = f"t-SNE: OOD validation, colored by genus{tag}"
    title_tsne_ood_species = f"t-SNE: OOD validation, colored by species{tag}"
    title_pca_id_genera    = f"PCA: ID validation, colored by genus{tag}"
    title_pca_id_species   = f"PCA: ID validation, colored by species{tag}"
    title_pca_ood_genera   = f"PCA: OOD validation, colored by genus{tag}"
    title_pca_ood_species  = f"PCA: OOD validation, colored by species{tag}"

    # plot t-SNE projections
    plot_projection(proj_tsne_id,  genera_id,  title_tsne_id_genera,   fpath_tsne_id_genera,   "t-SNE")
    plot_projection(proj_tsne_id,  sids_id,    title_tsne_id_species,  fpath_tsne_id_species,  "t-SNE")
    plot_projection(proj_tsne_ood, genera_ood, title_tsne_ood_genera,  fpath_tsne_ood_genera,  "t-SNE")
    plot_projection(proj_tsne_ood, sids_ood,   title_tsne_ood_species, fpath_tsne_ood_species, "t-SNE")
    # plot PCA projections
    plot_projection(proj_pca_id,  genera_id,  title_pca_id_genera,   fpath_pca_id_genera,   "PCA")
    plot_projection(proj_pca_id,  sids_id,    title_pca_id_species,  fpath_pca_id_species,  "PCA")
    plot_projection(proj_pca_ood, genera_ood, title_pca_ood_genera,  fpath_pca_ood_genera,  "PCA")
    plot_projection(proj_pca_ood, sids_ood,   title_pca_ood_species, fpath_pca_ood_species, "PCA")

if __name__ == "__main__":
    main()