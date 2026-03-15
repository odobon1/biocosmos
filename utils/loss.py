import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

from utils.pp import compute_rank_dists
from utils.phylo import PhyloVCV

import pdb


phylo_vcv = PhyloVCV()

def compute_targets(targ_type, batch_size, class_encs_b, targ_data_b, device):
    if targ_type == "aligned":
        targs = compute_targs_aligned(batch_size)
    elif targ_type == "multipos":
        targs = compute_targs_multipos(class_encs_b)
    elif targ_type == "tax":
        targs = compute_targs_tax(targ_data_b)
    elif targ_type == "phylo":
        targs = compute_targs_phylo(targ_data_b)
    targs = targs.to(device)  # pt[B, B]

    return targs

def compute_targs_aligned(batch_size):
    targs = torch.eye(batch_size)
    return targs

def compute_targs_multipos(class_encs_b):
    targs = (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float()
    return targs

def compute_targs_tax(targ_data_b):
    rank_dists = compute_rank_dists(targ_data_b)
    targs      = 1 - 0.5 * rank_dists
    return targs

def compute_targs_phylo(targ_data_b):
    targs = phylo_vcv.get_targs_batch(targ_data_b)
    return targs

def compute_loss(config_loss, logits, class_encs_b, targ_data_b, class_wts, class_pair_wts, device, train):
    if config_loss['type'] == "infonce1":
        loss, loss_raw = compute_loss_infonce(
            config_loss, 
            logits, 
            class_encs_b,
            targ_data_b,
            class_wts, 
            device,
            train,
        )
    elif config_loss['type'] == "infonce2":
        loss, loss_raw = compute_loss_infonce_2Dwtd(
            config_loss, 
            logits, 
            class_encs_b,
            targ_data_b,
            class_pair_wts, 
            device,
            train,
        )
    elif config_loss['type'] == "bce":
        loss, loss_raw = compute_loss_bce(
            config_loss, 
            logits, 
            class_encs_b,
            targ_data_b,
            class_pair_wts, 
            device,
            train,
        )
    return loss, loss_raw

def compute_loss_infonce(config_loss, logits, class_encs_b, targ_data_b, class_wts, device, train):
    """
    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """
    B     = logits.size(0)
    targs = compute_targets(config_loss['targ'], B, class_encs_b, targ_data_b, device)
    targs = targs / targs.sum(dim=1, keepdim=True)

    loss_i2t_raw_b = F.cross_entropy(logits,   targs,   reduction="none")  # pt[B]
    loss_t2i_raw_b = F.cross_entropy(logits.T, targs.T, reduction="none")  # pt[B]
    loss_raw = 0.5 * (loss_i2t_raw_b.mean() + loss_t2i_raw_b.mean())

    if not train:
        return loss_raw, loss_raw

    W_cb = class_wts[class_encs_b]  # class-balancing weights

    if config_loss['focal']:
        """
        p_t = exp(-CE)
        focal factor: (1 - p_t)^gamma
        expm1 has better precision when p_t ~ 1 (CE ~ 0)
        expm1(x) = e^x - 1

        only supports 0/1 targets
        """
        gamma = config_loss["cfg"]["focal"]["gamma"]
        W_foc_i2t = (-torch.expm1(-loss_i2t_raw_b)).clamp_min(1e-12).pow(gamma)
        W_foc_t2i = (-torch.expm1(-loss_t2i_raw_b)).clamp_min(1e-12).pow(gamma)

    else:
        W_foc_i2t = torch.ones_like(targs)
        W_foc_t2i = torch.ones_like(targs.T)

    W_i2t = W_foc_i2t * W_cb
    W_t2i = W_foc_t2i * W_cb

    num_i2t = (W_i2t * loss_i2t_raw_b).sum()
    num_t2i = (W_t2i * loss_t2i_raw_b).sum()
    den_i2t = W_i2t.detach().sum().clamp_min(1e-12)
    den_t2i = W_t2i.detach().sum().clamp_min(1e-12)

    loss = 0.5 * (num_i2t / den_i2t + num_t2i / den_t2i)

    return loss, loss_raw

def compute_loss_infonce_2Dwtd(config_loss, logits, class_encs_b, targ_data_b, class_pair_wts, device, train):
    """
    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """
    B     = logits.size(0)
    targs = compute_targets(config_loss['targ'], B, class_encs_b, targ_data_b, device)
    targs = targs / targs.sum(dim=1, keepdim=True)

    log_p_i2t = F.log_softmax(logits,   dim=-1)
    log_p_t2i = F.log_softmax(logits.T, dim=-1)

    loss_i2t_raw = -(targs   * log_p_i2t)
    loss_t2i_raw = -(targs.T * log_p_t2i)

    loss_raw = 0.5 * (loss_i2t_raw.sum(dim=1).mean() + loss_t2i_raw.sum(dim=1).mean())

    if not train:
        return loss_raw, loss_raw

    W_cb = class_pair_wts[class_encs_b][:, class_encs_b]  # class-balancing weights; pt[B, B]

    if config_loss['focal']:
        preds_i2t = log_p_i2t.exp()
        preds_t2i = log_p_t2i.exp()
        W_foc_i2t = focal_2d(preds_i2t, targs,   config_loss["cfg"]["focal"])
        W_foc_t2i = focal_2d(preds_t2i, targs.T, config_loss["cfg"]["focal"])
    else:
        W_foc_i2t = torch.ones_like(targs)
        W_foc_t2i = torch.ones_like(targs.T)

    W_i2t = W_foc_i2t * W_cb
    W_t2i = W_foc_t2i * W_cb

    num_i2t = (W_i2t * loss_i2t_raw).sum()
    num_t2i = (W_t2i * loss_t2i_raw).sum()

    den_i2t = W_i2t.detach().sum().clamp_min(1e-12)
    den_t2i = W_t2i.detach().sum().clamp_min(1e-12)

    loss = 0.5 * (num_i2t / den_i2t + num_t2i / den_t2i)

    return loss, loss_raw

def compute_loss_bce(config_loss, logits, class_encs_b, targ_data_b, class_pair_wts, device, train):

    B     = logits.size(0)
    targs = compute_targets(config_loss["targ"], B, class_encs_b, targ_data_b, device)

    if train:
        # batch class-pair weight matrix; advanced indexing used to extract submatrix as per class_enc indices (row/col selection)
        W_cb = class_pair_wts[class_encs_b][:, class_encs_b]  # class-balancing weights; pt[B, B]

        if config_loss['focal']:
            preds = torch.sigmoid(logits)
            W_foc = focal_2d(preds, targs, config_loss["cfg"]["focal"])  # pt[B, B]
        else:
            W_foc = torch.ones_like(targs)

        dyn_smr = config_loss["cfg"].get("dyn_smr", False)
        if dyn_smr:
            num_pos = torch.sum(targs).item()
            num_neg = B**2 - num_pos
            # scaling (numerical stability measure)
            wt_neg  = num_pos / (B**2 / 2)  # (_ / (B^2 / 2)) equivalent to dividing by mean of num_pos and num_neg
            wt_pos  = num_neg / (B**2 / 2)
            smr_wts = targs * wt_pos + (1 - targs) * wt_neg
        else:
            smr_wts = torch.ones_like(targs)

        W = W_cb * smr_wts * W_foc

    else:

        W = torch.ones_like(targs)

    loss_raw_matrix = F.binary_cross_entropy_with_logits(logits, targs, reduction="none")  # unweighted loss matrix; pt[B, B]
    loss = (W * loss_raw_matrix).sum() / W.detach().sum()  # weighted mean loss -- the norm here is irrelevant with the subsequent loss norm (may be some numerical considerations here though, might even want to prenorm the individual terms)

    loss_raw = loss_raw_matrix.mean()

    # used to render total batch loss the same regardless of reweighting (i.e. individual loss components are adjusted with reweighting, but the amount of "total learning" stays the same for apples-to-apples comparison with baselines)
    with torch.no_grad():
        norm = loss_raw / loss
    loss = norm * loss

    return loss, loss_raw

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