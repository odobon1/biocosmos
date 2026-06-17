import torch
import torch.distributed as dist
from torch.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

from utils.data import spawn_dataloader, spawn_partition_data, load_cid_2_penult
from utils.utils import get_text_template
from utils.eval import gather_variable_rows, gather_object_list
from utils.ddp import rank0


def get_embs_and_labels(modelw, dataloader, device, mixed_prec, cid_2_penult):
    """
    Iterate through dataloader to generate image embeddings and retrieve labels
    (penultimate-level groups + leaf-level class IDs), all-gathered across ranks.
    """
    modelw.model.eval()

    embs_all, penults_all, cids_all = [], [], []

    if dist.get_rank() == 0:
        print("Generating embeddings...")
    with torch.no_grad():
        for imgs_b, _, _, targ_data_b in tqdm(dataloader, disable=(dist.get_rank() != 0)):
            imgs_b = imgs_b.to(device, non_blocking=True)

            # generate embeddings
            if mixed_prec:
                with autocast(device_type=device.type):
                    embs_img_b = modelw.embed_images(imgs_b)
            else:
                embs_img_b = modelw.embed_images(imgs_b)

            embs_all.append(embs_img_b)
            for item in targ_data_b:
                cid = item["cid"]
                cids_all.append(cid)
                penults_all.append(cid_2_penult[cid])

    # under DDP each rank only sees its shard; all-gather so the full partition is plotted
    embs_local = torch.cat(embs_all, dim=0)
    embs_all = gather_variable_rows(embs_local).cpu().numpy()
    penults_all = gather_object_list(penults_all)
    cids_all = gather_object_list(cids_all)

    return embs_all, penults_all, cids_all

def compute_tsne(embeddings, perplexity=30, n_iter=1000, init="pca", random_state=42):
    """
    Reduce embedding dimensionality to 2D via t-SNE
    """
    print(f"Running t-SNE on {embeddings.shape[0]} samples (dim={embeddings.shape[1]})...")
    tsne = TSNE(perplexity=perplexity, max_iter=n_iter, init=init, random_state=random_state)
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
    index_data, _, enc2cid = spawn_partition_data(config=cfg, partition=partition)
    dataloader = spawn_dataloader(
        index_data=index_data,
        enc2cid=enc2cid,
        text_template=text_template,
        config=cfg,
        shuffle=False,
        drop_last=False,
        img_pp=modelw.img_pp_inf,
        use_dv_sampler=False,
        exact_distributed=True,
    )
    return dataloader

@rank0
def generate_projection_plots(
    embs_id, penults_id, cids_id,
    embs_ood, penults_ood, cids_ood,
    dpath_vis,
    tag=None,
    tsne_cfg=None,
):
    """
    Compute t-SNE/PCA projections of the ID/OOD embeddings and write the 8 plots
    under dpath_vis/{tsne,pca}/. Rank-0 only.
    """
    tsne_cfg = tsne_cfg or {}
    # compute t-SNE projections
    proj_tsne_id  = compute_tsne(embs_id, **tsne_cfg)
    proj_tsne_ood = compute_tsne(embs_ood, **tsne_cfg)
    # compute PCA projections
    proj_pca_id  = compute_pca(embs_id)
    proj_pca_ood = compute_pca(embs_ood)

    dpath_tsne = dpath_vis / "tsne"
    dpath_pca = dpath_vis / "pca"
    dpath_tsne.mkdir(parents=True, exist_ok=True)
    dpath_pca.mkdir(parents=True, exist_ok=True)
    fpath_tsne_id_penult = dpath_tsne / "id_penult.png"
    fpath_tsne_id_class = dpath_tsne / "id_leaf.png"
    fpath_tsne_ood_penult = dpath_tsne / "ood_penult.png"
    fpath_tsne_ood_class = dpath_tsne / "ood_leaf.png"
    fpath_pca_id_penult = dpath_pca / "id_penult.png"
    fpath_pca_id_class = dpath_pca / "id_leaf.png"
    fpath_pca_ood_penult = dpath_pca / "ood_penult.png"
    fpath_pca_ood_class = dpath_pca / "ood_leaf.png"

    if tag is not None:
        tag = f" ({tag})"
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

def generate_manifold_viz(cfg, modelw, dataloader_id, dataloader_ood, dpath_vis, tag=None, tsne_cfg=None):
    """
    Build ID/OOD image embeddings from `modelw` over the given dataloaders and write the
    t-SNE/PCA plots under `dpath_vis`. Every rank participates in the embedding all-gather;
    the projection + plotting is rank-0 only.
    """
    cid_2_penult = load_cid_2_penult(cfg.dataset)
    embs_id, penults_id, cids_id = get_embs_and_labels(modelw, dataloader_id, cfg.device, cfg.hw.mixed_prec, cid_2_penult)
    embs_ood, penults_ood, cids_ood = get_embs_and_labels(modelw, dataloader_ood, cfg.device, cfg.hw.mixed_prec, cid_2_penult)
    generate_projection_plots(
        embs_id, penults_id, cids_id,
        embs_ood, penults_ood, cids_ood,
        dpath_vis,
        tag=tag,
        tsne_cfg=tsne_cfg,
    )
