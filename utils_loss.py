import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

from utils_pp import compute_rank_dists
from utils_phylo import PhyloVCV

import pdb


phylo_vcv = PhyloVCV()

def compute_targets(targ_type, batch_size, class_encs_b, targ_data_b, device):
    if targ_type == "aligned":
        targs = compute_targs_aligned(batch_size)
    elif targ_type == "multipos":
        targs = compute_targs_multipos(class_encs_b)
    elif targ_type == "hierarchical":
        targs = compute_targs_hierarchical(targ_data_b)
    elif targ_type == "phylogenetic":
        targs = compute_targs_phylogenetic(targ_data_b)
    targs = targs.to(device)  # --- Tensor(B, B)

    return targs

def compute_targs_aligned(batch_size):
    targs = torch.eye(batch_size)
    return targs

def compute_targs_multipos(class_encs_b):
    targs = (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float()
    return targs

def compute_targs_hierarchical(targ_data_b):
    rank_dists = compute_rank_dists(targ_data_b)
    targs      = 1 - 0.5 * rank_dists
    return targs

def compute_targs_phylogenetic(targ_data_b):
    targs = phylo_vcv.get_targs_batch(targ_data_b)
    return targs

def compute_loss(config_loss, logits, class_encs_b, targ_data_b, class_wts, class_pair_wts, device):
    if config_loss['type'] == "infonce1":
        loss = compute_loss_infonce(
            config_loss, 
            logits, 
            class_encs_b,
            targ_data_b,
            class_wts, 
            device,
        )
    elif config_loss['type'] == "infonce2":
        loss = compute_loss_infonce_2Dwtd(
            config_loss, 
            logits, 
            class_encs_b,
            targ_data_b,
            class_pair_wts, 
            device,
        )
    elif config_loss['type'] == "bce":
        loss = compute_loss_bce(
            config_loss, 
            logits, 
            class_encs_b,
            targ_data_b,
            class_pair_wts, 
            device,
        )
    elif config_loss['type'] in ("mse", "huber"):
        loss = compute_loss_regression(
            config_loss, 
            logits, 
            class_encs_b,
            targ_data_b,
            class_pair_wts, 
            device,
        )
    return loss

def compute_loss_infonce(config_loss, logits, class_encs_b, targ_data_b, class_wts, device):
    """
    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """
    B     = logits.size(0)
    targs = compute_targets(config_loss['targ'], B, class_encs_b, targ_data_b, device)
    targs = targs / targs.sum(dim=1, keepdim=True)

    rw_wts = class_wts[class_encs_b]  # "re-weighting" weights

    loss_i2t_b = F.cross_entropy(logits,   targs,   reduction="none")  # --- Tensor(B)
    loss_t2i_b = F.cross_entropy(logits.T, targs.T, reduction="none")  # --- Tensor(B)

    if config_loss['focal']:
        """
        p_t = exp(-CE)
        focal factor: (1 - p_t)^gamma
        expm1 has better precision when p_t ~ 1 (CE ~ 0)
        expm1(x) = e^x - 1

        think this may be the variant we want to study this variant actually (the numerically stable form) ~ check DL materials
        only supports 0/1 targets
        """
        gamma   = config_loss["cfg"]["focal"]["gamma"]
        foc_i2t = (-torch.expm1(-loss_i2t_b)).clamp_min(1e-12).pow(gamma)
        foc_t2i = (-torch.expm1(-loss_t2i_b)).clamp_min(1e-12).pow(gamma)

        w_i2t = foc_i2t * rw_wts
        w_t2i = foc_t2i * rw_wts
    else:
        w_i2t = rw_wts
        w_t2i = rw_wts

    num_i2t = (w_i2t * loss_i2t_b).sum()
    num_t2i = (w_t2i * loss_t2i_b).sum()
    den_i2t = w_i2t.detach().sum()
    den_t2i = w_t2i.detach().sum()

    loss_batch = 0.5 * (num_i2t / den_i2t + num_t2i / den_t2i)

    return loss_batch

def compute_loss_infonce_2Dwtd(config_loss, logits, class_encs_b, targ_data_b, class_pair_wts, device):
    """
    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """
    B     = logits.size(0)
    targs = compute_targets(config_loss['targ'], B, class_encs_b, targ_data_b, device)
    targs = targs / targs.sum(dim=1, keepdim=True)

    rw_wts = class_pair_wts[class_encs_b][:, class_encs_b]  # --- Tensor(B, B); "re-weighting" weights

    log_p_i2t = F.log_softmax(logits,   dim=-1)
    log_p_t2i = F.log_softmax(logits.T, dim=-1)

    loss_i2t = -(targs * log_p_i2t)
    loss_t2i = -(targs.T * log_p_t2i)

    if config_loss['focal']:

        preds_i2t = log_p_i2t.exp()
        preds_t2i = log_p_t2i.exp()

        foc_i2t = focal_2d(preds_i2t, targs,   config_loss["cfg"]["focal"])
        foc_t2i = focal_2d(preds_t2i, targs.T, config_loss["cfg"]["focal"])

        w_i2t = foc_i2t * rw_wts
        w_t2i = foc_t2i * rw_wts
    else:
        w_i2t = rw_wts
        w_t2i = rw_wts

    num_i2t = (w_i2t * loss_i2t).sum()
    num_t2i = (w_t2i * loss_t2i).sum()

    den_i2t = w_i2t.detach().sum() / B  # divided by batch size for apples-to-apples w/ infonce1 ~ this needs to be investigated
    den_t2i = w_t2i.detach().sum() / B

    loss_batch = 0.5 * (num_i2t / den_i2t + num_t2i / den_t2i)

    return loss_batch

def compute_loss_bce(config_loss, logits, class_encs_b, targ_data_b, class_pair_wts, device):

    dyn_posneg = config_loss["cfg"].get("dyn_posneg", False)

    B     = logits.size(0)
    targs = compute_targets(config_loss['targ'], B, class_encs_b, targ_data_b, device)

    # batch class-pair weight matrix; advanced indexing used to extract submatrix as per class_enc indices (row/col selection)
    rw_wts = class_pair_wts[class_encs_b][:, class_encs_b]  # ----------- Tensor(B, B); "re-weighting" weights

    if config_loss['focal']:
        preds   = torch.sigmoid(logits)
        foc_wts = focal_2d(preds, targs, config_loss["cfg"]["focal"])  # --- Tensor(B, B)
    else:
        foc_wts = torch.ones_like(targs)

    if dyn_posneg:
        num_pos = torch.sum(targs).item()
        num_neg = B**2 - num_pos
        # scaling (numerical stability measure)
        wt_neg = num_pos / (B**2 / 2)  # (_ / (B^2 / 2)) equivalent to dividing by mean of num_pos and num_neg
        wt_pos = num_neg / (B**2 / 2)
        posneg_wts = targs * wt_pos + (1 - targs) * wt_neg
    else:
        posneg_wts = torch.ones_like(targs)

    W = rw_wts * posneg_wts * foc_wts

    loss_raw   = F.binary_cross_entropy_with_logits(logits, targs, reduction="none")  # --- Tensor(B, B); unweighted loss matrix
    loss_batch = (W * loss_raw).sum() / W.detach().sum()  # weighted mean loss -- the norm here is irrelevant with the subsequent loss norm (may be some numerical considerations here though, might even want to prenorm the individual terms)

    # used to render total batch loss the same regardless of reweighting (i.e. individual loss components are adjusted with reweighting, but the amount of "total learning" stays the same for apples-to-apples comparison with baselines)
    with torch.no_grad():
        norm = loss_raw.mean() / loss_batch
    loss_batch = norm * loss_batch

    return loss_batch

def compute_loss_regression(config_loss, logits, class_encs_b, targ_data_b, class_pair_wts, device):
    cfg_regr  = config_loss["cfg"]["regression"]

    B     = logits.size(0)
    targs = compute_targets(config_loss['targ'], B, class_encs_b, targ_data_b, device)

    # compute pop/neg balancing weights before casting targets to range [-1, 1] (discrete targs only)
    dyn_posneg = config_loss["cfg"].get("dyn_posneg", False)
    if dyn_posneg:
        num_pos = torch.sum(targs).item()
        num_neg = B**2 - num_pos
        # scaling (numerical stability measure)
        wt_neg = num_pos / (B**2 / 2)  # (equivalent to dividing by mean of num_pos and num_neg)
        wt_pos = num_neg / (B**2 / 2)
        posneg_wts = targs * wt_pos + (1 - targs) * wt_neg
    else:
        posneg_wts = torch.ones_like(targs)

    targs = targs * 2.0 - 1.0  # range [0, 1] --> [-1, 1]
    preds = logits

    rw_wts = class_pair_wts[class_encs_b][:, class_encs_b]  # -------------- Tensor(B, B); "re-weighting" weights

    if config_loss['focal']:
        foc_wts = focal_2d(preds, targs, config_loss["cfg"]["focal"])  # --- Tensor(B, B)
    else:
        foc_wts = torch.ones_like(targs)

    W = rw_wts * posneg_wts * foc_wts

    if config_loss['type'] == "mse":
        loss_raw = F.mse_loss(preds, targs, reduction="none")
    elif config_loss['type'] == "huber":
        loss_raw = F.smooth_l1_loss(preds, targs, beta=cfg_regr["huber_beta"], reduction="none")

    loss_batch = (W * loss_raw).sum() / W.detach().sum().clamp_min(1e-12)  # weighted mean loss

    with torch.no_grad():
        norm = loss_raw.mean() / loss_batch
    loss_batch = norm * loss_batch

    return loss_batch

def focal_2d(preds, targs, cfg_focal):

    gamma     = cfg_focal["gamma"]
    comp_type = cfg_focal["comp_type"]

    if comp_type == 1:
        p_t = (1 - preds) + targs * (2 * preds - 1)
    elif comp_type == 2:
        p_t = 1 - torch.abs(targs - preds)
    
    # p_t = p_t.clamp(1e-12, 1 - 1e-12)

    foc = (1 - p_t).pow(gamma)
    
    return foc