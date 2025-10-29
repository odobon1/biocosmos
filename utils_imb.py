import torch  # type: ignore[import]
import numpy as np  # type: ignore[import]

from utils import load_split


def compute_class_wts(cfg):

    # note: needs to be untangled....
    split       = load_split(cfg.split_name)
    counts      = split.class_counts_train
    pair_counts = np.outer(counts, counts)
    n_classes   = len(counts)

    cfg_cw = cfg.cfg_loss["class_weighting"]

    if cfg_cw["cp_type"] == 2:
        neg_mult2 = np.full((n_classes, n_classes), 2)
        np.fill_diagonal(neg_mult2, 1)
        pair_counts = pair_counts * neg_mult2

    if cfg_cw["type"] is None:
        class_wts      = np.ones_like(counts)
        class_pair_wts = np.ones_like(pair_counts)
    elif cfg_cw["type"] == "inv_freq":
        gamma          = cfg_cw.get("if_gamma", 0.0)
        class_wts      = 1.0 / np.power(counts, gamma)
        class_pair_wts = 1.0 / np.power(pair_counts, gamma)
    elif cfg_cw["type"] == "class_balanced":
        beta           = cfg_cw.get("cb_beta", 0.0)
        eps            = 1e-8
        class_wts      = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), eps)  # (1 - β) / (1 - β^n_c)
        class_pair_wts = (1.0 - beta) / np.maximum(1.0 - np.power(beta, pair_counts), eps)

    # normalize s.t. mean(wts) == 1.0
    # class_wts /= class_wts.mean()
    class_wts = class_wts / class_wts.mean()

    if cfg_cw["cp_type"] == 1:
        class_pair_wts = class_pair_wts / class_pair_wts.mean()
    elif cfg_cw["cp_type"] == 2:
        mask = np.triu(np.ones_like(class_pair_wts, dtype=bool))
        # values of the upper triangle (1D)
        tri_vals = class_pair_wts[mask]
        # symmetric weight values are considered to be of the same class (i.e. symmetrical values are considered duplicates), structured like so for convenient indexing i.e. although (i, j) and (j, i) key into different elements of the matrix, we are treating these as being the same
        class_pair_wts = class_pair_wts / tri_vals.mean()
    
    # class weight clipping (currently not used)
    class_wt_clip = cfg_cw.get("class_wt_clip", None)
    if class_wt_clip is not None:
        class_wts      = np.minimum(class_wts, class_wt_clip)
        class_pair_wts = np.minimum(class_pair_wts, class_wt_clip)
        # renormalize after clipping
        class_wts      = class_wts / class_wts.mean()
        class_pair_wts = class_pair_wts / class_pair_wts.mean()

    class_wts      = torch.tensor(class_wts)
    class_pair_wts = torch.tensor(class_pair_wts)

    return class_wts, class_pair_wts