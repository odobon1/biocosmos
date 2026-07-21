import torch

from utils.utils import load_split


def build_wting(cfg_wting, dataset, split, train_pt, dim):
    """
    Startup half of a weighting scheme: per-class counts, and the scalar that normalizes its
    weights to mean 1.0.

    The full weight vector (1D) / matrix (2D) is built here once, reduced to that scalar, and
    discarded -- training recomputes weights per batch from the counts via `compute_batch_wts`,
    so nothing of size n_classes^2 is held for the run.

    Args:
    - cfg_wting --- `wting` block of a loss config
    - dim --------- 1 for per-class weights, 2 for per-class-pair weights

    Returns:
    - counts --- Per-class sample counts for the train partition, NaN for absent classes; pt[n_classes]
    - norm ----- Mean weight over all classes (1D) / class pairs (2D)
    """
    counts = torch.tensor(load_split(dataset, split).class_counts[train_pt], dtype=torch.float64)
    class_encs = torch.arange(counts.numel())

    if dim == 1:
        wts = _compute_wts(cfg_wting, counts)
        norm = wts.nanmean()
    elif dim == 2:
        wts = _compute_wts(cfg_wting, _pair_counts(counts, class_encs, cfg_wting["cp_type"]))
        if cfg_wting["cp_type"] == 1:
            norm = wts.nanmean()
        elif cfg_wting["cp_type"] == 2:
            # symmetric weight values are considered to be of the same class (i.e. symmetrical values are considered duplicates), structured like so for convenient indexing i.e. although (i, j) and (j, i) key into different elements of the matrix, we are treating these as being the same
            norm = wts[torch.triu(torch.ones_like(wts, dtype=torch.bool))].nanmean()  # mean over the upper triangle (1D)

    return counts, norm.item()

def compute_batch_wts(cfg_wting, counts, class_encs_b, dim, norm):
    """
    Class-balancing weights for a batch, rebuilt from counts and normalized by the startup scalar.

    Returns pt[B] for dim 1, pt[B, B] for dim 2. Computed in float64 (`class_bal` evaluates
    1 - beta^n, which cancels catastrophically for small counts) and cast to float32, since a
    float64 result would promote the whole weighted loss reduction to float64.
    """
    if dim == 1:
        counts_b = counts[class_encs_b]
    elif dim == 2:
        counts_b = _pair_counts(counts, class_encs_b, cfg_wting["cp_type"])

    return (_compute_wts(cfg_wting, counts_b) / norm).float()

def _pair_counts(counts, class_encs, cp_type):
    """
    Class-pair counts for the given class encodings; pt[K, K] for pt[K] encodings.

    cp_type 2 counts negative pairs double. "Negative" is by class identity, not position -- two
    entries of the same class are a positive pair wherever they land in the matrix, which for a
    batch means anywhere two samples share a class, not just the diagonal.
    """
    counts_k = counts[class_encs]
    pair = counts_k.unsqueeze(1) * counts_k.unsqueeze(0)

    if cp_type == 2:
        neg = class_encs.unsqueeze(1) != class_encs.unsqueeze(0)
        pair = torch.where(neg, pair * 2, pair)

    return pair

def _compute_wts(cfg_wting, counts):
    """
    Weighting formula, unnormalized; dimension-agnostic -- applied to per-class counts (1D) or
    class-pair counts (2D) alike.
    """
    if cfg_wting["type"] is None:
        wts = torch.ones_like(counts)
    elif cfg_wting["type"] == "inv_freq":
        gamma = cfg_wting["inv_freq"]["gamma"]
        wts = 1.0 / counts.pow(gamma)
    elif cfg_wting["type"] == "class_bal":
        beta = cfg_wting["class_bal"]["beta"]
        eps = 1e-8
        wts = (1.0 - beta) / (1.0 - torch.pow(beta, counts)).clamp_min(eps)  # (1 - β) / (1 - β^n_c)

    return wts
