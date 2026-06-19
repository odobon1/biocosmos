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

def build_color_map(labels):
    """
    Deterministic label -> color over the full label set, so a class keeps the same color
    across every plot (rounds, t-SNE/PCA, ID/OOD). Palette is shuffled (fixed seed) to avoid
    similar colors landing on adjacent classes.
    """
    labels_sorted = sorted(set(labels))
    colors = list(sns.color_palette("husl", len(labels_sorted)))
    random.seed(42)
    random.shuffle(colors)
    return dict(zip(labels_sorted, colors))

def plot_projection(embs_2d, labels, title, fpath_plot, method, color_map, stoch_layer, alpha=1.0):
    print(f"Plotting {method} projection...")

    labels_arr = np.asarray(labels)
    point_colors = np.array([color_map[label] for label in labels])  # per-point RGB

    # draw order: shuffled (stochastic layering, so no single class fully occludes another)
    # or class-grouped in sorted order (original layering); both are deterministic
    if stoch_layer:
        order = np.random.default_rng(42).permutation(len(labels_arr))
    else:
        order = np.argsort(labels_arr, kind="stable")

    point_alpha = alpha[order] if isinstance(alpha, np.ndarray) else alpha  # per-point alpha masks a partition

    plt.figure(figsize=(16, 12))

    plt.scatter(
        embs_2d[order, 0],
        embs_2d[order, 1],
        c=point_colors[order],
        s=15,
        alpha=point_alpha,
    )

    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel(f"{method} Dimension 1", fontsize=14)
    plt.ylabel(f"{method} Dimension 2", fontsize=14)

    if method == "t-SNE":
        plt.gca().set_aspect("equal", adjustable="box")  # t-SNE dims are unitless; render 1:1 so structure isn't stretched

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
    color_leaf,
    color_penult,
    dpath_vis,
    tag=None,
    cfg_tsne=None,
    stoch_layer=True,
):
    """
    Compute t-SNE/PCA projections of the ID/OOD embeddings and write 20 plots under
    dpath_vis/{tsne,pca}/: ID-only, OOD-only, the combined ID+OOD projection, and the
    combined projection with each partition masked (alpha 0) in turn, each colored by
    penultimate-level group and by leaf class. The combined plots are a joint projection
    over the stacked ID+OOD embeddings (the separate ID/OOD projections live in different
    coordinate spaces and cannot be overlaid); the masked variants share that exact
    geometry, only hiding the other partition's points.
    """
    cfg_tsne = cfg_tsne or {}
    embs_fullset = np.concatenate([embs_id, embs_ood], axis=0)
    cids_fullset = list(cids_id) + list(cids_ood)
    penults_fullset = list(penults_id) + list(penults_ood)

    # per-point alpha over the fullset (ID rows first, then OOD) to mask one partition
    is_id = np.arange(len(cids_fullset)) < len(cids_id)
    alpha_id = is_id.astype(float)
    alpha_ood = (~is_id).astype(float)

    tag = f" ({tag})" if tag is not None else ""

    for method, project in (("t-SNE", lambda e: compute_tsne(e, **cfg_tsne)), ("PCA", compute_pca)):
        dpath_method = dpath_vis / ("tsne" if method == "t-SNE" else "pca")
        dpath_method.mkdir(parents=True, exist_ok=True)

        proj_id = project(embs_id)
        proj_ood = project(embs_ood)
        proj_fullset = project(embs_fullset)

        # (projection, labels, color_map, filename, subject, label_kind, alpha)
        panels = [
            (proj_id,      penults_id,      color_penult, "id_penult.png",          "ID",            "penult. lvl", 1.0),
            (proj_id,      cids_id,         color_leaf,   "id_leaf.png",            "ID",            "class",       1.0),
            (proj_ood,     penults_ood,     color_penult, "ood_penult.png",         "OOD",           "penult. lvl", 1.0),
            (proj_ood,     cids_ood,        color_leaf,   "ood_leaf.png",           "OOD",           "class",       1.0),
            (proj_fullset, penults_fullset, color_penult, "fullset_penult.png",     "ID+OOD",        "penult. lvl", 1.0),
            (proj_fullset, cids_fullset,    color_leaf,   "fullset_leaf.png",       "ID+OOD",        "class",       1.0),
            (proj_fullset, penults_fullset, color_penult, "fullset_id_penult.png",  "ID (in ID+OOD)",  "penult. lvl", alpha_id),
            (proj_fullset, cids_fullset,    color_leaf,   "fullset_id_leaf.png",    "ID (in ID+OOD)",  "class",       alpha_id),
            (proj_fullset, penults_fullset, color_penult, "fullset_ood_penult.png", "OOD (in ID+OOD)", "penult. lvl", alpha_ood),
            (proj_fullset, cids_fullset,    color_leaf,   "fullset_ood_leaf.png",   "OOD (in ID+OOD)", "class",       alpha_ood),
        ]
        for proj, labels, color_map, fname, subject, kind, alpha in panels:
            title = f"{method}: {subject} validation, colored by {kind}{tag}"
            plot_projection(proj, labels, title, dpath_method / fname, method, color_map, stoch_layer, alpha)

@rank0
def generate_manifold_viz(dataset, eval_bundle_id, eval_bundle_ood, dpath_vis, cfg_manifold_viz, tag=None):
    """
    Write the t-SNE/PCA plots under `dpath_vis`, reusing the ID/OOD image embeddings collected
    during eval (already all-gathered to every rank). `cfg_manifold_viz` is the manifold_viz.yaml
    contents (t-SNE params + stochastic-layering flag). Rank-0 only.
    """
    cid_2_penult = load_cid_2_penult(dataset)
    embs_id = eval_bundle_id["embs_img"].cpu().numpy()
    embs_ood = eval_bundle_ood["embs_img"].cpu().numpy()
    cids_id = eval_bundle_id["cids_img"]
    cids_ood = eval_bundle_ood["cids_img"]
    penults_id = [cid_2_penult[cid] for cid in cids_id]
    penults_ood = [cid_2_penult[cid] for cid in cids_ood]
    # color maps span the whole dataset so a class is colored identically in every plot
    color_leaf = build_color_map(cid_2_penult.keys())
    color_penult = build_color_map(cid_2_penult.values())
    generate_projection_plots(
        embs_id,
        embs_ood,
        cids_id,
        cids_ood,
        penults_id,
        penults_ood,
        color_leaf,
        color_penult,
        dpath_vis,
        tag=tag,
        cfg_tsne=cfg_manifold_viz["tsne"],
        stoch_layer=cfg_manifold_viz["stoch_layer"],
    )
