import torch

from utils.utils import load_split


def build_wting(cfg_wting, dataset, split, train_pt, dim, batch_size):
    """
    Startup half of a weighting scheme: per-class counts, and the scalar that normalizes its
    weights to mean 1.0.

    The full weight vector (1D) / matrix (2D) is built here once, reduced to that scalar, and
    discarded -- training recomputes weights per batch from the counts via `compute_cls_imb_wts`,
    so nothing of size n_classes^2 is held for the run.

    Args:
    - cfg_wting ---- `wting.cls_imb` block of a loss config
    - dim ---------- 1 for per-class weights, 2 for per-class-pair weights
    - batch_size --- Global train batch size; only consumed by `freq_type_2d = pair_prob`

    Returns:
    - counts ---- Per-class sample counts for the train partition, NaN for absent classes; pt[n_classes]
    - ds_norm --- Mean weight over all classes (1D) / class pairs (2D); dataset-level constant, computed once at startup
    """
    counts = torch.tensor(load_split(dataset, split).class_counts[train_pt], dtype=torch.float64)
    class_encs = torch.arange(counts.numel())

    if dim == 1:
        wts = _compute_wts(cfg_wting, counts)
        ds_norm = wts.nanmean()
    elif dim == 2:
        wts = _compute_wts(cfg_wting, _pair_freqs(counts, class_encs, cfg_wting["freq_type_2d"], batch_size))
        if cfg_wting["freq_type_2d"] == "naive":
            ds_norm = wts.nanmean()
        elif cfg_wting["freq_type_2d"] == "cmx2" or cfg_wting["freq_type_2d"] == "pair_prob":
            ds_norm = wts[torch.triu(torch.ones_like(wts, dtype=torch.bool))].nanmean()  # mean over upper triangle

    return counts, ds_norm.item()

def compute_cls_imb_wts(cfg_wting, counts, class_encs_b, dim, ds_norm, batch_size):
    """
    Class-balancing weights for a batch, rebuilt from counts and normalized by the startup scalar.

    Returns pt[B] for dim 1, pt[B, B] for dim 2. Computed in float64 (`class_bal` evaluates
    1 - beta^n, which cancels catastrophically for small counts) and cast to float32, since a
    float64 result would promote the whole weighted loss reduction to float64.
    """
    if dim == 1:
        freqs_b = counts[class_encs_b]
    elif dim == 2:
        freqs_b = _pair_freqs(counts, class_encs_b, cfg_wting["freq_type_2d"], batch_size)

    return (_compute_wts(cfg_wting, freqs_b) / ds_norm).float()

def _pair_freqs(counts, class_encs, freq_type_2d, batch_size):
    """
    Class-pair counts for the given class encodings; pt[K, K] for pt[K] encodings.

    `freq_type_2d = cmx2` counts negative pairs double. "Negative" is by class identity, not position -- two
    entries of the same class are a positive pair wherever they land in the matrix, which for a
    batch means anywhere two samples share a class, not just the diagonal.
    """
    counts_k = counts[class_encs]
    pair_freqs = torch.outer(counts_k, counts_k)

    if freq_type_2d == "cmx2":
        matches = class_encs.unsqueeze(1) == class_encs.unsqueeze(0)
        pair_freqs = torch.where(matches, pair_freqs, pair_freqs * 2)
    elif freq_type_2d == "pair_prob":
        total_count = counts.nansum()
        c1 = counts[class_encs].unsqueeze(1)
        c2 = counts[class_encs].unsqueeze(0)
        homog = (c1 / total_count) * (1 + (batch_size - 1) * (c2 - 1) / (total_count - 1))
        heterog = (c1 / total_count) * (batch_size - 1) * (c2 / (total_count - 1))
        matches = class_encs.unsqueeze(1) == class_encs.unsqueeze(0)
        pair_freqs = torch.where(matches, homog, heterog)  # pair-probability matrix

    return pair_freqs

def _compute_wts(cfg_wting, freqs):
    """
    Weighting formula, unnormalized; dimension-agnostic -- applied to per-class counts (1D) or
    class-pair counts (2D) alike.
    """
    if cfg_wting["type"] is None:
        wts = torch.ones_like(freqs)
    elif cfg_wting["type"] == "inv_freq":
        gamma = cfg_wting["inv_freq"]["gamma"]
        wts = 1.0 / freqs.pow(gamma)
    elif cfg_wting["type"] == "class_bal":
        beta = cfg_wting["class_bal"]["beta"]
        eps = 1e-8
        wts = (1.0 - beta) / (1.0 - torch.pow(beta, freqs)).clamp_min(eps)  # (1 - β) / (1 - β^n_c)

    return wts
