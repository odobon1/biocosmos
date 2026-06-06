"""
`rfpath_model` must be specified in eval.yaml

torchrun --standalone --nproc-per-node=auto -m tools.vis_manifold
"""

import torch
from torch.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

from models import VLMWrapper
from utils.config import get_config_eval
from utils.data import spawn_dataloader, spawn_partition_data, load_cid_2_penult
from utils.utils import paths, get_text_template
from utils.ddp import setup_ddp, cleanup_ddp

def get_embs_and_labels(modelw, dataloader, device, mixed_prec, cid_2_penult):
    """
    Iterate through dataloader to generate image embeddings and retrieve labels
    (penultimate-level groups + leaf-level class IDs)
    """
    modelw.model.eval()

    embs_all, penults_all, cids_all = [], [], []

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
                cid = item["cid"]
                cids_all.append(cid)
                penults_all.append(cid_2_penult[cid])

    embs_all = np.concatenate(embs_all, axis=0)

    return embs_all, penults_all, cids_all

def compute_tsne(embeddings, perplexity=30, random_state=42):
    """
    Reduce embedding dimensionality to 2D via t-SNE
    """
    print(f"Running t-SNE on {embeddings.shape[0]} samples (dim={embeddings.shape[1]})...")
    tsne = TSNE(perplexity=perplexity, random_state=random_state)
    embs_2d = tsne.fit_transform(embeddings)
    return embs_2d

def compute_pca(embeddings):
    """
    Reduce embedding dimensionality to 2D via PCA
    """
    print(f"Running PCA on {embeddings.shape[0]} samples (dim={embeddings.shape[1]})...")
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(embeddings)
    return embs_2d

def plot_projection(embs_2d, labels, title, fpath_plot, method):
    print(f"Plotting {method} projection...")

    # convert labels to numpy array for indexing convenience
    labels_np = np.array(labels)
    unique_labels = sorted(list(set(labels)))
    n_classes = len(unique_labels)

    plt.figure(figsize=(16, 12))

    colors = list(sns.color_palette("husl", n_classes))
    random.seed(42)
    random.shuffle(colors)  # shuffle colors to mitigate similar colors for nearby classes

    for i, label in enumerate(unique_labels):
        mask = (labels_np == label)  # boolean mask for current label

        plt.scatter(
            embs_2d[mask, 0],
            embs_2d[mask, 1],
            c=[colors[i]],
            label=label,
            s=60,
            alpha=0.8,
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

def get_dataloader(cfg, partition, modelw):
    text_template = get_text_template(cfg.text_template, dataset=cfg.dataset)
    index_data, _ = spawn_partition_data(config=cfg, partition=partition)
    dataloader = spawn_dataloader(
        index_data=index_data,
        text_template=text_template,
        config=cfg,
        shuffle=False,
        drop_last=False,
        img_pp=modelw.img_pp_inf,
        use_dv_sampler=False,
    )
    return dataloader

def main():

    setup_ddp()

    # component of plot title that appears in parentheses, set to None for no tag
    # TAG = "base"
    TAG = None

    cfg    = get_config_eval(verbose=True)
    modelw = VLMWrapper.build(cfg, verbose=True)

    cid_2_penult = load_cid_2_penult(cfg.dataset)

    print(f"Preparing ID data...")
    dataloader_id = get_dataloader(cfg, "id", modelw)
    print(f"Preparing OOD data...")
    dataloader_ood = get_dataloader(cfg, "ood", modelw)

    # get embeddings and labels
    embs_id, penults_id, cids_id = get_embs_and_labels(modelw, dataloader_id, cfg.device, cfg.hw.mixed_prec, cid_2_penult)
    embs_ood, penults_ood, cids_ood = get_embs_and_labels(modelw, dataloader_ood, cfg.device, cfg.hw.mixed_prec, cid_2_penult)

    # compute t-SNE projections
    proj_tsne_id  = compute_tsne(embs_id)
    proj_tsne_ood = compute_tsne(embs_ood)
    # compute PCA projections
    proj_pca_id  = compute_pca(embs_id)
    proj_pca_ood = compute_pca(embs_ood)

    dpath_plots = (paths["root"] / cfg.rfpath_model).parent / "../../plots"
    fpath_tsne_id_penult = dpath_plots / "tsne_id_penult.png"
    fpath_tsne_id_class = dpath_plots / "tsne_id_class.png"
    fpath_tsne_ood_penult = dpath_plots / "tsne_ood_penult.png"
    fpath_tsne_ood_class = dpath_plots / "tsne_ood_class.png"
    fpath_pca_id_penult = dpath_plots / "pca_id_penult.png"
    fpath_pca_id_class = dpath_plots / "pca_id_class.png"
    fpath_pca_ood_penult = dpath_plots / "pca_ood_penult.png"
    fpath_pca_ood_class = dpath_plots / "pca_ood_class.png"

    if TAG is not None:
        tag = f" ({TAG})"
    else:
        tag = ""

    title_tsne_id_penult = f"t-SNE: ID validation, colored by penult. lvl{tag}"
    title_tsne_id_class = f"t-SNE: ID validation, colored by class{tag}"
    title_tsne_ood_penult = f"t-SNE: OOD validation, colored by penult. lvl{tag}"
    title_tsne_ood_class = f"t-SNE: OOD validation, colored by class{tag}"
    title_pca_id_penult = f"PCA: ID validation, colored by penult. lvl{tag}"
    title_pca_id_class = f"PCA: ID validation, colored by class{tag}"
    title_pca_ood_penult = f"PCA: OOD validation, colored by penult. lvl{tag}"
    title_pca_ood_class = f"PCA: OOD validation, colored by class{tag}"

    # plot t-SNE projections
    plot_projection(proj_tsne_id, penults_id, title_tsne_id_penult, fpath_tsne_id_penult, "t-SNE")
    plot_projection(proj_tsne_id, cids_id, title_tsne_id_class, fpath_tsne_id_class, "t-SNE")
    plot_projection(proj_tsne_ood, penults_ood, title_tsne_ood_penult, fpath_tsne_ood_penult, "t-SNE")
    plot_projection(proj_tsne_ood, cids_ood, title_tsne_ood_class, fpath_tsne_ood_class, "t-SNE")
    # plot PCA projections
    plot_projection(proj_pca_id, penults_id, title_pca_id_penult, fpath_pca_id_penult, "PCA")
    plot_projection(proj_pca_id, cids_id, title_pca_id_class, fpath_pca_id_class, "PCA")
    plot_projection(proj_pca_ood, penults_ood, title_pca_ood_penult, fpath_pca_ood_penult, "PCA")
    plot_projection(proj_pca_ood, cids_ood, title_pca_ood_class, fpath_pca_ood_class, "PCA")

    cleanup_ddp()

if __name__ == "__main__":
    main()
