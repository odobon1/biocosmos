import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

from utils_pp import compute_rank_dists

import pdb


def compute_targets(targ_type, batch_size, class_encs_b, rank_keys_b, device):
    if targ_type == "pairwise":
        targs = torch.eye(batch_size, device=device)  # ------------------------------- Tensor(B, B)
    elif targ_type == "multipos":
        targs = (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float()  # --- Tensor(B, B)
        targs = targs.to(device)
    elif targ_type == "hierarchical":
        rank_dists = compute_rank_dists(rank_keys_b)
        targs      = 1 - 0.5 * rank_dists
        targs      = targs.to(device)  # ---------------------------------------------- Tensor(B, B)
    return targs

def compute_loss(targ_type, loss_type, logits, class_encs_b, rank_keys_b, class_wts, class_pair_wts, cfg_focal, cfg_regr, alpha_pos, dyn_posneg, device):
    class_encs_b = torch.tensor(class_encs_b).to(device)
    if loss_type == "infonce1":
        loss = compute_loss_infonce(
            targ_type, 
            logits, 
            class_encs_b, 
            rank_keys_b, 
            class_wts, 
            cfg_focal, 
            device,
        )
        return loss
    if loss_type == "infonce2":
        loss = compute_loss_infonce_2Dwtd(
            targ_type, 
            logits, 
            class_encs_b, 
            rank_keys_b, 
            class_pair_wts, 
            cfg_focal, 
            device,
        )
        return loss
    elif loss_type == "sigmoid":
        loss = compute_loss_sigmoid(
            targ_type, 
            logits, 
            class_encs_b, 
            rank_keys_b, 
            class_pair_wts, 
            cfg_focal, 
            alpha_pos, 
            dyn_posneg,
            device,
        )
        return loss
    elif loss_type in ("mse", "huber"):
        loss = compute_loss_regression(
            targ_type, 
            loss_type, 
            logits, 
            class_encs_b, 
            rank_keys_b, 
            class_pair_wts, 
            cfg_focal, 
            cfg_regr, 
            alpha_pos, 
            device,
        )
        return loss

def compute_loss_infonce(targ_type, logits, class_encs_b, rank_keys_b, class_wts, cfg_focal, device):
    """
    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """
    B     = logits.size(0)
    targs = compute_targets(targ_type, B, class_encs_b, rank_keys_b, device)
    targs = targs / targs.sum(dim=1, keepdim=True)

    rw_wts = class_wts[class_encs_b]  # "re-weighting" weights

    loss_i2t_b = F.cross_entropy(logits,   targs,   reduction="none")  # --- Tensor(B)
    loss_t2i_b = F.cross_entropy(logits.T, targs.T, reduction="none")  # --- Tensor(B)

    if cfg_focal["enabled"]:
        """
        p_t = exp(-CE)
        focal factor: (1 - p_t)^gamma
        expm1 has better precision when p_t ~ 1 (CE ~ 0)
        expm1(x) = e^x - 1

        AX THIS STYLE
        only supports 0/1 targets
        """
        gamma   = cfg_focal["gamma"]
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

def compute_loss_infonce_2Dwtd(targ_type, logits, class_encs_b, rank_keys_b, class_pair_wts, cfg_focal, device):
    """
    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """
    B     = logits.size(0)
    targs = compute_targets(targ_type, B, class_encs_b, rank_keys_b, device)
    targs = targs / targs.sum(dim=1, keepdim=True)

    rw_wts = class_pair_wts[class_encs_b][:, class_encs_b]  # --- Tensor(B, B); "re-weighting" weights

    log_p_i2t = F.log_softmax(logits,   dim=-1)
    log_p_t2i = F.log_softmax(logits.T, dim=-1)

    loss_i2t = -(targs * log_p_i2t)
    loss_t2i = -(targs.T * log_p_t2i)

    if cfg_focal["enabled"]:

        preds_i2t = log_p_i2t.exp()
        preds_t2i = log_p_t2i.exp()

        foc_i2t = focal_2d(preds_i2t, targs, cfg_focal)
        foc_t2i = focal_2d(preds_t2i, targs.T, cfg_focal)

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

def compute_loss_sigmoid(targ_type, logits, class_encs_b, rank_keys_b, class_pair_wts, cfg_focal, alpha_pos, dyn_posneg, device):
    B     = logits.size(0)
    targs = compute_targets(targ_type, B, class_encs_b, rank_keys_b, device)

    # batch class-pair weight matrix; advanced indexing used to extract submatrix as per class_enc indices (row/col selection)
    rw_wts = class_pair_wts[class_encs_b][:, class_encs_b]  # ----------------------------- Tensor(B, B); "re-weighting" weights

    if cfg_focal["enabled"]:
        preds   = torch.sigmoid(logits)
        foc_wts = focal_2d(preds, targs, cfg_focal)  # ------------------------------------ Tensor(B, B)
    else:
        foc_wts = torch.ones_like(targs)

    if dyn_posneg:
        num_pos = torch.sum(targs).item()
        num_neg = B**2 - num_pos
        # scaling (numerical stability measure)
        wt_neg = num_pos / (B**2 / 2)  # (equivalent to dividing by mean of num_pos and num_neg)
        wt_pos = num_neg / (B**2 / 2)
        posneg_wts = targs * wt_pos + (1 - targs) * wt_neg
    else:
        posneg_wts = targs * alpha_pos + (1 - targs) * (1 - alpha_pos)  # continuous i.e. compatible with hierarchical ~  review this

    W = rw_wts * posneg_wts * foc_wts

    loss_raw   = F.binary_cross_entropy_with_logits(logits, targs, reduction="none")  # --- Tensor(B, B); unweighted loss matrix
    loss_batch = (W * loss_raw).sum() / W.detach().sum()  # weighted mean loss -- the norm here is irrelevant with the subsequent loss norm (may be some numerical considerations here though, might even want to prenorm the individual terms)

    # used to render total batch loss the same regardless of reweighting (i.e. individual loss components are adjusted with reweighting, but the amount of "total learning" stays the same for apples-to-apples comparison with baselines)
    with torch.no_grad():
        norm = loss_raw.mean() / loss_batch
    loss_batch = norm * loss_batch

    return loss_batch

def compute_loss_regression(targ_type, loss_type, logits, class_encs_b, rank_keys_b, class_pair_wts, cfg_focal, cfg_regr, alpha_pos, device):
    B     = logits.size(0)
    targs = compute_targets(targ_type, B, class_encs_b, rank_keys_b, device)

    if cfg_regr["scale_type"] == 1:
        if cfg_regr["temp"] or cfg_regr["bias"]:
            logits = logits.tanh()
        preds = (logits + 1.0) * 0.5
        preds = preds.clamp(0.0, 1.0)
    elif cfg_regr["scale_type"] == 2:
        preds = logits.sigmoid()

    # batch class-pair weight matrix; advanced indexing used to extract submatrix as per class_enc indices (row/col selection)
    rw_wts = class_pair_wts[class_encs_b][:, class_encs_b]  # --- Tensor(B, B); "re-weighting" weights

    if cfg_focal["enabled"]:
        foc_wts = focal_2d(preds, targs, cfg_focal)  # ---------- Tensor(B, B)
    else:
        foc_wts = torch.ones_like(targs)
        
    posneg_wts = targs * alpha_pos + (1 - targs) * (1 - alpha_pos)  # continuous i.e. compatible with hierarchical (review this)

    W = rw_wts * posneg_wts * foc_wts

    if loss_type == "mse":
        loss_raw = F.mse_loss(preds, targs, reduction="none")
    elif loss_type == "huber":
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