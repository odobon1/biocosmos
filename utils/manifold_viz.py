import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random

from utils.data import load_cid_2_penult
from utils.ddp import rank0


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

def generate_projection_plots(
    embs_id,
    embs_ood,
    cids_id,
    cids_ood,
    penults_id,
    penults_ood,
    dpath_vis,
    tag=None,
    cfg_tsne=None,
):
    """
    Compute t-SNE/PCA projections of the ID/OOD embeddings and write the 8 plots
    under dpath_vis/{tsne,pca}/.
    """
    cfg_tsne = cfg_tsne or {}
    # compute t-SNE projections
    proj_tsne_id  = compute_tsne(embs_id, **cfg_tsne)
    proj_tsne_ood = compute_tsne(embs_ood, **cfg_tsne)
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

@rank0
def generate_manifold_viz(dataset, eval_bundle_id, eval_bundle_ood, dpath_vis, tag=None, cfg_tsne=None):
    """
    Write the t-SNE/PCA plots under `dpath_vis`, reusing the ID/OOD image embeddings collected
    during eval (already all-gathered to every rank). Rank-0 only.
    """
    cid_2_penult = load_cid_2_penult(dataset)
    embs_id = eval_bundle_id["embs_img"].cpu().numpy()
    embs_ood = eval_bundle_ood["embs_img"].cpu().numpy()
    cids_id = eval_bundle_id["cids_img"]
    cids_ood = eval_bundle_ood["cids_img"]
    penults_id = [cid_2_penult[cid] for cid in cids_id]
    penults_ood = [cid_2_penult[cid] for cid in cids_ood]
    generate_projection_plots(
        embs_id,
        embs_ood,
        cids_id,
        cids_ood,
        penults_id,
        penults_ood,
        dpath_vis,
        tag=tag,
        cfg_tsne=cfg_tsne,
    )
