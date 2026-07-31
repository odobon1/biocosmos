import torch
import torch.nn.functional as F
import abc

from utils.rank_encs import compute_rank_dists
from utils.phylo import PhyloVCV
from utils.imb import build_wting, compute_cls_imb_wts

import pdb


_phylo_vcv_cache: dict[tuple, PhyloVCV] = {}
_htarg_shuf: bool = False
_phylo_seed: int | None = None


def configure_htarg_shuf(htarg_shuf: bool, seed: int | None) -> None:
    """Set phylo-target shuffling for this run; call once at setup before any loss is computed."""
    global _htarg_shuf, _phylo_seed
    _htarg_shuf = htarg_shuf
    _phylo_seed = seed

def get_phylo_vcv(dataset: str) -> PhyloVCV:
    key = (dataset, _htarg_shuf, _phylo_seed)
    if key not in _phylo_vcv_cache:
        _phylo_vcv_cache[key] = PhyloVCV(dataset=dataset, htarg_shuf=_htarg_shuf, seed=_phylo_seed)
    return _phylo_vcv_cache[key]

def compute_targets(targ_type, batch_size, class_encs_b, targ_data_b, device):
    if targ_type == "iw":
        targs = compute_targs_iw(batch_size)
    elif targ_type == "sw":
        targs = compute_targs_sw(class_encs_b)
    elif targ_type == "tax":
        targs = compute_targs_tax(targ_data_b)
    elif targ_type == "phylo":
        targs = compute_targs_phylo(targ_data_b)
    targs = targs.to(device)  # pt[B, B]

    return targs

def compute_targs_iw(batch_size):
    targs = torch.eye(batch_size)
    return targs

def compute_targs_sw(class_encs_b):
    targs = (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float()
    return targs

def compute_targs_tax(targ_data_b):
    rank_dists = compute_rank_dists(targ_data_b)
    R = len(targ_data_b[0]["rank_encs"])  # tree depth (max rank_dist)
    targs = 1 - rank_dists / R
    return targs

def compute_targs_phylo(targ_data_b):
    dataset = targ_data_b[0]["dataset"]
    targs = get_phylo_vcv(dataset).get_targs_batch(targ_data_b)
    return targs

class Criterion(abc.ABC):
    """
    A loss paired with the class-imbalance weighting it consumes. The weighting dimensionality is a
    property of the loss (`wting_dim`) -- 1D per-class weights for InfoNCE1, 2D per-class-pair
    weights for InfoNCE2/BCE.

    Only the class counts and the normalization scalar are held; batch weights are computed from
    them on the fly, so no n_classes (1D) / n_classes^2 (2D) weight buffer persists for the run.
    """

    wting_dim: int

    def __init__(self, cfg_loss, dataset, split, train_pt, device, batch_size):
        self.cfg = cfg_loss
        self.device = device
        self.batch_size = batch_size
        counts, self.wt_mean = build_wting(cfg_loss["wting"]["cls_imb"], dataset, split, train_pt, self.wting_dim, batch_size)
        self.counts = counts.to(device)

    @staticmethod
    def build(cfg_loss, dataset, split, train_pt, device, batch_size):
        crit_cls = {
            "infonce1": InfoNCE1Criterion,
            "infonce2": InfoNCE2Criterion,
            "bce":      BCECriterion,
        }[cfg_loss["type"]]

        return crit_cls(cfg_loss, dataset, split, train_pt, device, batch_size)

    def _targets(self, batch_size, class_encs_b, targ_data_b):
        return compute_targets(self.cfg["targ"], batch_size, class_encs_b, targ_data_b, self.device)

    def _cls_imb_wts(self, class_encs_b):
        return compute_cls_imb_wts(self.cfg["wting"]["cls_imb"], self.counts, class_encs_b, self.wting_dim, self.wt_mean, self.batch_size)

    @abc.abstractmethod
    def __call__(self, logits, class_encs_b, targ_data_b, train):
        """
        Computes loss for a batch given logits and target data.

        Returns:
        - loss ------- Weighted scalar loss (== loss_raw when not training)
        - loss_raw --- Unweighted scalar loss
        - targs ------ Target matrix; pt[B, B]
        """
        raise NotImplementedError

class InfoNCE1Criterion(Criterion):
    """
    InfoNCE weighted by 1D per-class weights, applied to per-sample cross-entropy terms.

    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """

    wting_dim = 1

    def __call__(self, logits, class_encs_b, targ_data_b, train):
        B = logits.size(0)
        targs_raw = self._targets(B, class_encs_b, targ_data_b)
        targs = targs_raw / targs_raw.sum(dim=1, keepdim=True)

        loss_i2t_raw_b = F.cross_entropy(logits, targs, reduction="none")  # pt[B]
        loss_t2i_raw_b = F.cross_entropy(logits.T, targs.T, reduction="none")  # pt[B]
        loss_raw = 0.5 * (loss_i2t_raw_b.mean() + loss_t2i_raw_b.mean())

        if not train:
            return loss_raw, loss_raw, targs_raw

        W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B]

        if self.cfg["wting"]["focal"]["gamma"] > 0.0:
            """
            p_t = exp(-CE)
            focal factor: (1 - p_t)^gamma
            expm1 has better precision when p_t ~ 1 (CE ~ 0)
            expm1(x) = e^x - 1

            only supports 0/1 targets
            """
            gamma = self.cfg["wting"]["focal"]["gamma"]
            W_foc_i2t = (-torch.expm1(-loss_i2t_raw_b)).clamp_min(1e-12).pow(gamma)
            W_foc_t2i = (-torch.expm1(-loss_t2i_raw_b)).clamp_min(1e-12).pow(gamma)

        else:
            W_foc_i2t = torch.ones_like(targs)
            W_foc_t2i = torch.ones_like(targs.T)

        W_i2t = W_foc_i2t * W_ci
        W_t2i = W_foc_t2i * W_ci

        num_i2t = (W_i2t * loss_i2t_raw_b).sum()
        num_t2i = (W_t2i * loss_t2i_raw_b).sum()
        den_i2t = W_i2t.detach().sum().clamp_min(1e-12)
        den_t2i = W_t2i.detach().sum().clamp_min(1e-12)

        loss = 0.5 * (num_i2t / den_i2t + num_t2i / den_t2i)

        return loss, loss_raw, targs_raw

class InfoNCE2Criterion(Criterion):
    """
    InfoNCE weighted by 2D per-class-pair weights, applied elementwise to the loss matrix.

    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """

    wting_dim = 2

    def __call__(self, logits, class_encs_b, targ_data_b, train):
        B = logits.size(0)
        targs_raw = self._targets(B, class_encs_b, targ_data_b)
        targs = targs_raw / targs_raw.sum(dim=1, keepdim=True)

        log_p_i2t = F.log_softmax(logits,   dim=-1)
        log_p_t2i = F.log_softmax(logits.T, dim=-1)

        loss_i2t_raw = -(targs   * log_p_i2t)
        loss_t2i_raw = -(targs.T * log_p_t2i)

        loss_raw = 0.5 * (loss_i2t_raw.sum(dim=1).mean() + loss_t2i_raw.sum(dim=1).mean())

        if not train:
            return loss_raw, loss_raw, targs_raw

        W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B, B]

        if self.cfg["wting"]["focal"]["gamma"] > 0.0:
            preds_i2t = log_p_i2t.exp()
            preds_t2i = log_p_t2i.exp()
            W_foc_i2t = focal_2d(preds_i2t, targs, self.cfg["wting"]["focal"])
            W_foc_t2i = focal_2d(preds_t2i, targs.T, self.cfg["wting"]["focal"])
        else:
            W_foc_i2t = torch.ones_like(targs)
            W_foc_t2i = torch.ones_like(targs.T)

        W_i2t = W_foc_i2t * W_ci
        W_t2i = W_foc_t2i * W_ci

        num_i2t = (W_i2t * loss_i2t_raw).sum()
        num_t2i = (W_t2i * loss_t2i_raw).sum()

        den_i2t = W_i2t.detach().sum().clamp_min(1e-12)
        den_t2i = W_t2i.detach().sum().clamp_min(1e-12)

        loss = 0.5 * (num_i2t / den_i2t + num_t2i / den_t2i)

        return loss, loss_raw, targs_raw

class BCECriterion(Criterion):
    """
    Sigmoid BCE weighted by 2D per-class-pair weights.
    """

    wting_dim = 2

    def __call__(self, logits, class_encs_b, targ_data_b, train):
        B = logits.size(0)
        targs = self._targets(B, class_encs_b, targ_data_b)
        # fp32: cos-path logits are bf16 under autocast, where sigmoid saturates to exactly 1.0 at
        # |logit| >~ 6 (zeroing focal weights on the easy set, quantizing the rest); upcast once and
        # reuse for both focal preds and the BCE loss. No-op when logits are already fp32 (geo sim).
        logits_f = logits.float()

        if train:

            W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B, B]
            if self.cfg["wting"]["norm"]["cls_imb"]:
                W_ci = W_ci / W_ci.detach().mean()
            
            if self.cfg["wting"]["focal"]["gamma"] > 0.0:
                preds = torch.sigmoid(logits_f)
                W_foc = focal_2d(preds, targs, self.cfg["wting"]["focal"], clamp_base=True)  # pt[B, B]
            else:
                W_foc = torch.ones_like(targs)

            if self.cfg["wting"]["dsmr"]:
                mass_pos = torch.sum(targs)
                mass_neg = torch.sum(1.0 - targs)
                scale = B**2 / (2 * mass_pos * mass_neg)  # mass_neg == 0 --> inf here, but masked below (guard against div-by-zero for all-positive batch)
                wt_neg = scale * mass_pos
                wt_pos = scale * mass_neg
                W_dsmr = torch.where(
                    mass_neg == 0,
                    torch.ones_like(targs),
                    targs * wt_pos + (1 - targs) * wt_neg,
                )
            # else:
            #     W_dsmr = torch.ones_like(targs)

            agg = self.cfg["wting"]["agg"]
            if self.cfg["wting"]["dsmr"]:
                if agg == "prod":
                    W = W_ci * W_foc * W_dsmr
                elif agg == "mean":
                    W = (W_ci + W_foc + W_dsmr) / 3
                elif agg == "geo_mean":
                    eps = torch.finfo(W_ci.dtype).tiny  # per-factor log floor; engages only when a factor is already ~0 (e.g. focal underflow)
                    W = torch.exp((W_ci.clamp_min(eps).log() + W_foc.clamp_min(eps).log() + W_dsmr.clamp_min(eps).log()) / 3.0)  # log-space: no product underflow / clamp-floor artifact
                elif agg == "harm_mean":
                    W = 3.0 / (1.0 / W_ci + 1.0 / W_foc.clamp_min(1e-8) + 1.0 / W_dsmr)
            else:
                if agg == "prod":
                    W = W_ci * W_foc
                elif agg == "mean":
                    W = (W_ci + W_foc) / 2
                elif agg == "geo_mean":
                    eps = torch.finfo(W_ci.dtype).tiny  # per-factor log floor; engages only when a factor is already ~0 (e.g. focal underflow)
                    W = torch.exp((W_ci.clamp_min(eps).log() + W_foc.clamp_min(eps).log()) / 2.0)  # log-space: no product underflow / clamp-floor artifact
                elif agg == "harm_mean":
                    W = 2.0 / (1.0 / W_ci + 1.0 / W_foc.clamp_min(1e-8))

            if self.cfg["wting"]["norm"]["agg"]:
                W = W / W.detach().mean().clamp_min(1e-12)

        else:

            W = torch.ones_like(targs)

        loss_raw_matrix = F.binary_cross_entropy_with_logits(logits_f, targs, reduction="none")  # unweighted loss matrix; pt[B, B]
        loss = (W * loss_raw_matrix).sum() / B
        loss_raw = loss_raw_matrix.sum() / B

        return loss, loss_raw, targs

def focal_2d(preds, targs, cfg_focal, clamp_base=False):

    gamma = cfg_focal["gamma"]
    comp_type = cfg_focal["comp_type"]

    if comp_type == 1:
        p_t = (1 - preds) + targs * (2 * preds - 1)
    elif comp_type == 2:
        p_t = 1 - torch.abs(targs - preds)

    base = 1 - p_t
    if clamp_base:
        base = base.clamp_min(1e-12)  # pow backward at base 0 is inf for gamma < 1

    foc = base.pow(gamma)

    return foc
