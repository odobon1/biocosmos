import torch
import torch.nn.functional as F
import torch.distributed as dist
import abc
from contextlib import nullcontext
import math

from utils.rank_encs import compute_rank_dists
from utils.phylo import PhyloVCV
from utils.imb import build_wting, compute_cls_imb_wts
from utils.head import compute_sim

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
    property of the loss (`wting_dim`) -- 1D per-class weights for InfoNCE, 2D per-class-pair
    weights for BCE.

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
            "infonce": InfoNCECriterion,
            "bce":     BCECriterion,
        }[cfg_loss["crit"]]

        return crit_cls(cfg_loss, dataset, split, train_pt, device, batch_size)

    def _targets(self, batch_size, class_encs_b, targ_data_b):
        return compute_targets(self.cfg["targ"], batch_size, class_encs_b, targ_data_b, self.device)

    def _cls_imb_wts(self, class_encs_b):
        return compute_cls_imb_wts(self.cfg["wting"]["cls_imb"], self.counts, class_encs_b, self.wting_dim, self.wt_mean, self.batch_size)

    def _focal_2d(self, Z, Y):
        if "focal" not in self.cfg["wting"]:
            return torch.ones_like(Y)
        gamma = self.cfg["wting"]["focal"]["gamma"]
        return torch.abs(Y - self._preds(Z)).clamp_min(1e-12).pow(gamma)

    @abc.abstractmethod
    def _preds(self, Z):
        """Logits -> prediction probabilities under this criterion's link (for focal weighting)."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        """
        Computes loss for a batch given logits and target data. `logit_scale` is this criterion's
        learnable log logit scale param (model.logit_scale for loss1, model.logit_scale2 for loss2),
        raw (pre-clamp).

        Returns:
        - loss ------- Weighted scalar loss (== loss_raw when not training)
        - loss_raw --- Unweighted scalar loss
        - targs ------ Target matrix; pt[B, B]
        """
        raise NotImplementedError

class InfoNCECriterion(Criterion):
    """
    InfoNCE weighted by 1D per-class weights, applied to per-sample cross-entropy terms.

    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """

    wting_dim = 1

    def _preds(self, Z):
        return F.softmax(Z, dim=1)

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        B = logits.size(0)
        
        Y = self._targets(B, class_encs_b, targ_data_b)  # pt[B, B]
        Y_mass = Y.sum(dim=1)

        if self.cfg["infonce"]["tsm"]["type"] == "linear":
            Y_scaled = Y / Y_mass[:, None]  # pt[B, B]; for MP + HCon (note: symmetrical for MP, non-symmetrical for HCon)
        elif self.cfg["infonce"]["tsm"]["type"] == "softmax":
            tau_Y = self.cfg["infonce"]["tsm"]["sm_temp"]
            if tau_Y == "pinned":
                if self.cfg["logits"]["temperature"]["clamp"]:
                    logit_scale = logit_scale.clamp(max=math.log(100))
                Y_scaled = F.softmax(2 * Y * torch.exp(logit_scale), dim=1)  # pt[B, B]; for HCon (note: symmetrical for MP, non-symmetrical for HCon)
            else:
                Y_scaled = F.softmax(2 * Y / tau_Y, dim=1)  # pt[B, B]; for MP + HCon (note: symmetrical for MP, non-symmetrical for HCon)

        loss_i2t_raw = -Y_scaled * F.log_softmax(logits,   dim=1)  # pt[B, B]
        loss_t2i_raw = -Y_scaled * F.log_softmax(logits.T, dim=1)  # pt[B, B]

        # per-anchor CE (row sum) averaged over anchors -- CLIP's /B, the scale the weighted loss carries;
        # mass correction is a per-anchor weight, so it stays out of the weighting-free reference
        loss_raw = 0.5 * (loss_i2t_raw.sum(dim=1).mean() + loss_t2i_raw.sum(dim=1).mean())

        if not train:
            return loss_raw, loss_raw, Y

        if self.cfg["infonce"]["targ_mass_preservation"]:
            loss_i2t_raw = loss_i2t_raw * Y_mass[:, None]
            loss_t2i_raw = loss_t2i_raw * Y_mass[:, None]

        W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B]
        if self.cfg["wting"]["cls_imb"]["norm"]:
            W_ci = W_ci / W_ci.mean()  # pt[B]

        # Note: 2D-focal is still used despite 1D class-imbalance weighting (reduces to standard focal loss in the SP setting)
        W_foc_i2t = self._focal_2d(logits,   Y_scaled)  # pt[B, B]
        W_foc_t2i = self._focal_2d(logits.T, Y_scaled)  # pt[B, B]

        W_i2t = W_foc_i2t * W_ci[:, None]  # pt[B, B]
        W_t2i = W_foc_t2i * W_ci[:, None]  # pt[B, B]

        loss_i2t = (W_i2t * loss_i2t_raw).sum(dim=1)
        loss_t2i = (W_t2i * loss_t2i_raw).sum(dim=1)

        loss = 0.5 * (loss_i2t.mean() + loss_t2i.mean())

        return loss, loss_raw, Y

class BCECriterion(Criterion):
    """
    Sigmoid BCE weighted by 2D per-class-pair weights.
    """

    wting_dim = 2

    def _preds(self, Z):
        return torch.sigmoid(Z)

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        B = logits.size(0)
        
        Y = self._targets(B, class_encs_b, targ_data_b)
        # fp32: cos-path logits are bf16 under autocast, where sigmoid saturates to exactly 1.0 at
        # |logit| >~ 6 (zeroing focal weights on the easy set, quantizing the rest); upcast once and
        # reuse for both focal preds and the BCE loss. No-op when logits are already fp32 (geo sim).
        logits_f = logits.float()

        if train:
            W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B, B]
            if self.cfg["wting"]["cls_imb"]["norm"]:
                W_ci = W_ci / W_ci.mean()
            W_foc = self._focal_2d(logits_f, Y)  # pt[B, B]
            W = W_ci * W_foc
            if self.cfg["wting"]["bce"]["dsmr"]:
                mass_pos = torch.sum(Y)
                mass_neg = torch.sum(1.0 - Y)
                W = W * _dsmr_weight(Y, mass_pos, mass_neg, B)
        else:
            W = torch.ones_like(Y)

        loss_raw_matrix = F.binary_cross_entropy_with_logits(logits_f, Y, reduction="none")  # unweighted loss matrix; pt[B, B]
        loss = (W * loss_raw_matrix).sum() / B
        loss_raw = loss_raw_matrix.sum() / B

        return loss, loss_raw, Y

def _dsmr_weight(targs, mass_pos, mass_neg, B):
    """
    DSMR (dynamic same-class mass reweighting) pair weight from the global pos/neg target mass. Shared
    by the full-batch (BCECriterion.__call__) and tiled (chunked) paths -- for a tile, `targs` is the
    [C, B] block and (mass_pos, mass_neg) are the full-BxB masses. mass_neg == 0 (all-positive batch)
    is masked to 1.0 (scale -> inf otherwise).
    """
    scale = B**2 / (2 * mass_pos * mass_neg)  # mass_neg == 0 --> inf here, but masked below (guard against div-by-zero for all-positive batch)
    wt_neg = scale * mass_pos
    wt_pos = scale * mass_neg
    W_dsmr = torch.where(
        mass_neg == 0, 
        torch.ones_like(targs), 
        targs * wt_pos + (1 - targs) * wt_neg
    )
    return W_dsmr

# ------------------------------------------------------------------------------------------------
# Tiled / chunked global-batch loss (hardware.loss_chunk_size)
#
# The full-batch contrastive loss materializes several BxB matrices (sim, logits, weights, loss) and
# their autograd graph -- O(B^2) VRAM, the wall that OOMs bs32k. The chunked path computes the exact
# same weighted-BCE loss and gradients while never holding the full BxB matrices, and no rank computes
# more than its share of them: the BxB rows are sharded across ranks into world_size equal row-bands of
# b = B/world_size (SigLIP-style decomposition -- the pairwise-independent BCE loss has no batch-global
# normalizer, so each rank sweeps only its own band instead of redundantly recomputing the full matrix),
# and each rank sums its band over b/C row-blocks of C x B (C = loss_chunk_size rows, b an exact multiple
# of C), backpropagating each block into the detached embedding leaves (GradCache-style representation
# gradients). Peak VRAM is O(C*B); per-rank loss compute is O(B^2/world_size). Cross-band couplings --
# the loss/raw totals, the batch stats, and the precomputed constants below -- are folded with small
# all-reduces so every rank returns identical full-batch values; the leaves' band-partial dL/dembs sum
# to the full gradient across ranks (completed by batch_step_chunked's grad all-reduce).
#
# Supports the full BCE config space (incl. sw/iw/tax/phylo targets, cls_imb.norm, a BCE+BCE
# secondary-loss mix, and mix_unit_scale) -- only InfoNCE is excluded (chunking_supported).
# The reductions that couple across the whole BxB matrix -- the cls_imb.norm weight-mean
# normalizers, the DSMR mass, and the per-loss mix_unit_scale scalar -- are all DETACHED constants, so
# they are precomputed (cheap embedding-free closed forms + no_grad band sweeps, all-reduced to
# rank-identical values) before the single grad-carrying backward sweep applies them as constants.
# See _precompute_crit_consts.
#
# Centering (logits.bce.center) is reproduced exactly, never per-tile: "sim" recovers the full-batch
# sim mean in-graph per block via the cos bilinearity mean(sim) = mean(img) . mean(txt) (geo sims have
# no closed form -- TrainConfig rejects center: sim + chunking + geo, and this module asserts it);
# grad_proj/grad_proj2 precompute the full-batch mean of the incoming gradient at the projection node
# (_crit_center_grad_mean: a band sweep with tile-local autograd, all-reduced) and every tile subtracts
# that constant (_ZeroSumGradConst via compute_logits's center_global) in place of the full path's g.mean().
# ------------------------------------------------------------------------------------------------

def chunking_supported(cfg_loss, cfg_loss2):
    """
    The tiled loss reproduces every BCE config but not InfoNCE (its row/column softmax couples along
    columns, which a row-block cannot tile). Config treats hardware.loss_chunk_size as inert (full
    BxB path) when this returns False.
    """
    if cfg_loss["crit"] != "bce":
        return False
    if cfg_loss2["mix"] != 0.0 and cfg_loss2["crit"] != "bce":
        return False
    return True

def make_targ_block_fn(targ_type, class_encs_b, targ_data_b, B, device):
    """
    Build a closure (rs, re) -> [re-rs, B] target row-block (rows rs:re vs all B columns) matching the
    full-batch compute_targets for the given targ_type. Reusable per-tile inputs (tax rank vectors,
    phylo correlation lookups) are precomputed once here so the sweeps only slice per block.
    """
    if targ_type == "sw":
        return lambda rs, re: (class_encs_b[rs:re].unsqueeze(1) == class_encs_b.unsqueeze(0)).float()
    if targ_type == "iw":
        cols = torch.arange(B, device=device)
        return lambda rs, re: (torch.arange(rs, re, device=device).unsqueeze(1) == cols.unsqueeze(0)).float()
    if targ_type == "tax":
        tax_vecs = torch.tensor([td["rank_encs"] for td in targ_data_b], device=device)  # [B, R]
        R = tax_vecs.size(1)
        def targ_block(rs, re):
            neq = (tax_vecs[rs:re].unsqueeze(1) != tax_vecs.unsqueeze(0)).int()  # [C, B, R]
            # first differing rank (deterministic, no argmax tie-break); all-equal -> R (rank_dist 0)
            div = R - (neq.cumsum(dim=2) >= 1).sum(dim=2)  # divergence level; [C, B]
            rank_dists = R - div
            return (1.0 - rank_dists / R).float()
        return targ_block
    if targ_type == "phylo":
        return get_phylo_vcv(targ_data_b[0]["dataset"]).make_targ_block_fn(targ_data_b, device)

def bce_dsmr_mass(targ_type, targ_block_fn, class_encs_b, B, chunk_size, lo, hi, world_size):
    """
    Global DSMR mass over the full BxB target matrix: mass_pos = sum(targs), mass_neg = B^2 - sum(targs)
    (== sum(1 - targs) for targets in [0, 1]). For sw/iw (0/1 targets) mass_pos is the O(B) closed form
    sum_k count_k^2 / B (rank-identical, no collective); for soft tax/phylo targets it is summed over
    this rank's band [lo, hi) of target tiles (embedding-free) and all-reduced across bands.
    Matches torch.sum(targs) / torch.sum(1 - targs) in BCECriterion.__call__.
    """
    device = class_encs_b.device
    if targ_type == "sw":
        counts = torch.bincount(class_encs_b).to(torch.float64)
        mass_pos = (counts * counts).sum()
    elif targ_type == "iw":
        mass_pos = torch.tensor(float(B), dtype=torch.float64, device=device)
    else:  # tax, phylo -- soft targets: sum over this rank's band of tiles, fold across bands
        mass_pos = torch.zeros((), dtype=torch.float64, device=device)
        for rs in range(lo, hi, chunk_size):
            mass_pos += targ_block_fn(rs, rs + chunk_size).double().sum()
        if world_size > 1:
            dist.all_reduce(mass_pos)
    mass_pos = mass_pos.to(device=device, dtype=torch.float32)
    mass_neg = torch.tensor(float(B) * float(B), dtype=torch.float32, device=device) - mass_pos
    return mass_pos, mass_neg


class _SimTargStatsAccum:
    """
    Streams the per-batch sim/target distribution stats over the loss tiles so the chunked path can
    report the same batch_stats dict as sim_targ_batch_stats without holding the full BxB matrices.
    min/max/mean are exact; the median is over a strided subsample of each tile (an exact BxB median
    would need the whole matrix). Targets are rescaled [0,1]->[-1,1] to share sim's range, as there.
    """
    def __init__(self, device):
        self.sim_min = torch.tensor(float("inf"), device=device)
        self.sim_max = torch.tensor(float("-inf"), device=device)
        self.sim_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.targ_min = torch.tensor(float("inf"), device=device)
        self.targ_max = torch.tensor(float("-inf"), device=device)
        self.targ_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0
        self.sim_samp = []
        self.targ_samp = []

    def update(self, sim_tile, targs_tile):
        s = sim_tile.reshape(-1).float()
        t = targs_tile.reshape(-1).float()
        self.sim_min = torch.minimum(self.sim_min, s.min())
        self.sim_max = torch.maximum(self.sim_max, s.max())
        self.sim_sum += s.double().sum()
        self.targ_min = torch.minimum(self.targ_min, t.min())
        self.targ_max = torch.maximum(self.targ_max, t.max())
        self.targ_sum += t.double().sum()
        self.count += s.numel()
        stride = max(1, s.numel() // 4096)  # bound the median subsample per tile
        self.sim_samp.append(s[::stride])
        self.targ_samp.append(t[::stride])

    def finalize(self, world_size):
        sim_samp = torch.cat(self.sim_samp)
        targ_samp = torch.cat(self.targ_samp)
        sim_min, sim_max, sim_sum = self.sim_min, self.sim_max, self.sim_sum
        targ_min, targ_max, targ_sum = self.targ_min, self.targ_max, self.targ_sum
        count = self.count
        if world_size > 1:  # fold per-band partials; the bands partition the BxB rows exactly
            ext = torch.stack([-sim_min, -targ_min, sim_max, targ_max])
            dist.all_reduce(ext, op=dist.ReduceOp.MAX)
            sim_min, targ_min, sim_max, targ_max = -ext[0], -ext[1], ext[2], ext[3]
            sums = torch.stack([sim_sum, targ_sum])
            dist.all_reduce(sums)
            sim_sum, targ_sum = sums[0], sums[1]
            count *= world_size  # equal bands -> equal per-rank counts
            # median subsamples: equal bands + equal tile sizes -> equal lengths on every rank, so a
            # plain all_gather reassembles the exact same subsample pool a single full sweep produces
            samp = torch.stack([sim_samp, targ_samp])
            parts = [torch.empty_like(samp) for _ in range(world_size)]
            dist.all_gather(parts, samp)
            sim_samp = torch.cat([p[0] for p in parts])
            targ_samp = torch.cat([p[1] for p in parts])
        sim_median = sim_samp.median()
        targ_median = targ_samp.median()
        return {
            "sim_min":     sim_min.item(),
            "sim_max":     sim_max.item(),
            "sim_median":  sim_median.item(),
            "sim_mean":    (sim_sum / count).item(),
            "targ_min":    (2.0 * targ_min - 1.0).item(),
            "targ_max":    (2.0 * targ_max - 1.0).item(),
            "targ_median": (2.0 * targ_median - 1.0).item(),
            "targ_mean":   (2.0 * targ_sum / count - 1.0).item(),
        }


def _crit_block_weight_bce(crit, logits_f, targs, class_encs_rows, class_encs_cols, B, consts):
    """
    For one criterion and one [C, B] row-block: the aggregated per-pair weight W (differentiable via the
    focal factor) and the raw BCE matrix. `consts` carries the precomputed detached global scalars
    (cls_imb_mean, dsmr_mass); a None entry means that normalizer is off. Mirrors the
    train-mode weighting of BCECriterion.__call__ tile-by-tile via the shared _dsmr_weight.
    """
    cfg_wting = crit.cfg["wting"]

    W_ci = compute_cls_imb_wts(
        cfg_wting["cls_imb"], 
        crit.counts, 
        class_encs_rows, 
        crit.wting_dim,
        crit.wt_mean, 
        crit.batch_size, 
        class_encs_cols=class_encs_cols
    )
    if consts["cls_imb_mean"] is not None:
        W_ci = W_ci / consts["cls_imb_mean"]

    W_foc = crit._focal_2d(logits_f, targs)

    W = W_ci * W_foc
    if cfg_wting["bce"]["dsmr"]:
        W = W * _dsmr_weight(targs, *consts["dsmr_mass"], B)

    bce = F.binary_cross_entropy_with_logits(logits_f, targs, reduction="none")

    return W, bce

def _crit_block_logits_f(crit, secondary, img_rows, txt, compute_logits, center, center_global=None):
    """
    [C, B] similarity tile and its float32 logits tile for a criterion (its sim_type + logit scale/bias).

    `center`/`center_global` are passed through to compute_logits: the chunked sweeps supply the
    full-batch quantity (in-graph global sim mean for "sim"; detached full-batch incoming-grad mean
    for grad_proj/grad_proj2) so tiles reproduce full-batch centering exactly, never the per-tile mean.
    """
    sim_block = compute_sim(img_rows, txt, crit.cfg["sim"])
    logits_block = compute_logits(sim_block, crit.cfg["logits"]["temperature"]["clamp"], center, secondary=secondary, center_global=center_global)
    return sim_block, logits_block.float()

def _precompute_crit_consts(crit, secondary, img, txt, targ_block_fn, class_encs_b, B,
                            compute_logits, chunk_size, mixed_prec, device, need_L, autocast_ctx,
                            lo, hi, world_size, center, center_global_det):
    """
    Detached global constants for one criterion (see module header). cls_imb_mean (mean of W_ci) and
    dsmr_mass are embedding-free; L_value (the criterion's full weighted loss, needed for
    mix_unit_scale) requires a no_grad tile sweep, run only when mix_unit_scale is active. All
    sweeps cover only this rank's row-band [lo, hi);
    the partial sums are all-reduced so every rank derives identical constants.
    `center`/`center_global_det` reproduce the criterion's centered forward in the L sweep (for
    "sim" the detached global sim mean; grad_proj* leave the forward untouched).
    Returns (consts dict for _crit_block_weight_bce, L_value|None).
    """
    cfg_w = crit.cfg["wting"]
    targ_type = crit.cfg["targ"]

    cls_imb_mean = None
    if cfg_w["cls_imb"]["norm"]:  # mean of W_ci over BxB -- embedding-free, tiled to stay O(C*B)
        s = torch.zeros((), dtype=torch.float64, device=device)
        for rs in range(lo, hi, chunk_size):
            W_ci = compute_cls_imb_wts(cfg_w["cls_imb"], crit.counts, class_encs_b[rs:rs + chunk_size],
                                       crit.wting_dim, crit.wt_mean, crit.batch_size, class_encs_cols=class_encs_b)
            s += W_ci.double().sum()
        if world_size > 1:
            dist.all_reduce(s)
        cls_imb_mean = (s / (B * B)).float()

    dsmr_mass = bce_dsmr_mass(targ_type, targ_block_fn, class_encs_b, B, chunk_size, lo, hi, world_size) if cfg_w["bce"]["dsmr"] else None

    L_value = None
    if need_L:
        consts_raw = {"cls_imb_mean": cls_imb_mean, "dsmr_mass": dsmr_mass}
        sum_Wbce = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for rs in range(lo, hi, chunk_size):
                re = rs + chunk_size
                with autocast_ctx():
                    _, logits_f = _crit_block_logits_f(crit, secondary, img[rs:re], txt, compute_logits, center, center_global_det)
                    targs_block = targ_block_fn(rs, re)
                    W, bce = _crit_block_weight_bce(crit, logits_f, targs_block, class_encs_b[rs:re], class_encs_b, B, consts_raw)
                sum_Wbce += (W * bce).double().sum()
        if world_size > 1:
            dist.all_reduce(sum_Wbce)
        L_value = (sum_Wbce / B).float()

    return {"cls_imb_mean": cls_imb_mean, "dsmr_mass": dsmr_mass}, L_value

def _crit_center_grad_mean(crit, secondary, img, txt, targ_fn, class_encs_b, B, consts, coeff,
                           compute_logits, chunk_size, autocast_ctx, lo, hi, world_size, device):
    """
    grad_proj/grad_proj2: the detached full-batch mean of the incoming gradient at the criterion's
    projection node, so every tile of the grad sweep subtracts the same constant the full-batch
    _ZeroSumGrad projection would (a per-tile mean would be wrong). Sweeps this rank's band with
    tile-local autograd (the sim tile as leaf, embeddings detached: graphs stay O(C*B), no param
    .grad touched), which captures the focal-weight gradient terms exactly; band partials are
    all-reduced. Returns the constant at the node the mode projects: the sim node for grad_proj
    (e^t folded in by autograd), the scaled-sim node for grad_proj2 (the post-clamp e^t divided
    back out, recovered as a probe gradient through compute_logits).
    """
    clamp = crit.cfg["logits"]["temperature"]["clamp"]
    g_sum = torch.zeros((), dtype=torch.float64, device=device)
    for rs in range(lo, hi, chunk_size):
        re = rs + chunk_size
        with autocast_ctx():
            sim_leaf = compute_sim(img[rs:re].detach(), txt.detach(), crit.cfg["sim"]).requires_grad_(True)
            logits_f = compute_logits(sim_leaf, clamp, None, secondary=secondary).float()
            targs_block = targ_fn(rs, re)
            W, bce = _crit_block_weight_bce(crit, logits_f, targs_block, class_encs_b[rs:re], class_encs_b, B, consts)
            tile_loss = (W * bce).sum() / B
        g_sum += torch.autograd.grad(tile_loss, sim_leaf)[0].double().sum()
    if world_size > 1:
        dist.all_reduce(g_sum)
    c = (coeff * g_sum / (B * B)).float()  # mean incoming grad at the sim node, loss-mix coefficient folded in
    if crit.cfg["logits"]["bce"]["center"] == "grad_proj2":
        # the scaled-sim node's grad = (sim node's grad) / e^t; recover the post-clamp e^t as a probe gradient
        probe = torch.zeros(1, 1, device=device, requires_grad=True)
        e_det = torch.autograd.grad(compute_logits(probe, clamp, None, secondary=secondary).sum(), probe)[0].reshape(()).detach()
        c = c / e_det
    return c.detach()

def _gsum_hook(acc):
    """Tensor backward hook accumulating the incoming grad's sum into `acc`. Returns None so the
    gradient itself passes through untouched (a non-None return would replace it)."""
    def hook(g):
        acc.add_(g.double().sum())
    return hook

def chunked_bce_loss_backward(img, txt, class_encs_b, targ_data_b, crit1, crit2, mix, mix_unit_scale,
                              compute_logits, chunk_size, mixed_prec, device, rank, world_size):
    """
    Tiled + row-band-sharded global-batch BCE loss + backward (GradCache-style representation gradients).
    Computes the exact same weighted BCE loss and gradients as the full-batch path (BCECriterion.__call__
    blended by _global_batch_loss) over the full BxB matrix, but never materializes it and shares the work
    across ranks: the BxB rows split into world_size equal bands (SigLIP-style decomposition), this rank
    sums only its band [rank*b, (rank+1)*b) over row-blocks of C x B (b an exact multiple of C =
    chunk_size), and each block's gradient is backpropagated into the embedding leaves as computed, so
    peak VRAM is O(C*B) and per-rank compute is O(B^2/world_size). The loss/raw totals and batch stats
    are all-reduced, so every rank returns identical full-batch values; the leaves' .grad hold this
    band's PARTIAL dL/dembs, which sum to the full gradient across ranks (the caller completes them --
    see batch_step_chunked). Exact up to floating-point summation order.

    Supports a BCE+BCE loss mix: loss = (1 - mix)*s1*L1 + mix*s2*L2, where Lk is criterion k's weighted
    loss and sk = 1/Lk.detach() if mix_unit_scale else 1 (mix == 0 -> just crit1). All cross-tile-coupled
    normalizers are precomputed detached constants (_precompute_crit_consts), so the backward is single-pass.

    - img, txt --------- detached [B, D] embedding leaves (requires_grad); receive band-partial dL/dembs
                         in their .grad.
    - crit1, crit2 ----- primary / secondary BCECriterion (crit2 None when mix == 0).
    - compute_logits --- VLMWrapper.compute_logits(sim, clamp, center, secondary, center_global) -> logits
                         tile; center_global carries the precomputed full-batch centering quantity
                         (module header) so tiled centering is exact.
    - rank, world_size - this rank's band index / number of bands (1 -> unsharded full sweep).

    Returns (loss, loss_raw, batch_stats, grad_sum_sims), all detached; gradients left in the leaves' /
    params' .grad. grad_sum_sims = (sum(dL/dsim1), sum(dL/dsim2)|None), the full-batch sums accumulated
    tile-by-tile via backward hooks (all-reduced across bands) -- the same values the full-batch path
    reads off the retained sim grads.
    """
    B = img.size(0)
    b = B // world_size
    # checking that the BxB rows split into world_size equal bands of whole chunk_size-row blocks:
    # ragged bands would silently double-count rows across ranks (wrong gradients, no error)
    if b * world_size != B or b % chunk_size != 0:
        raise ValueError(
            f"global batch ({B}) must split into world_size ({world_size}) equal row-bands, each an exact "
            f"multiple of hardware.loss_chunk_size ({chunk_size}); got band size {b}"
        )
    lo, hi = rank * b, (rank + 1) * b
    crits = [(crit1, False)] + ([(crit2, True)] if mix != 0.0 else [])

    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_prec else nullcontext()

    centers = [crit.cfg["logits"]["bce"]["center"] for crit, _ in crits]
    for k, (crit, _) in enumerate(crits):
        # config rejects this combo too (TrainConfig); guard direct callers against a silently-wrong tile mean
        assert not (centers[k] == "sim" and crit.cfg["sim"] != "cos"), "center: sim under chunking requires cos sim"
    m_det = None
    if "sim" in centers:
        # full-batch sim-matrix mean via the cos bilinearity: mean_ij(img_i . txt_j) = mean(img) . mean(txt)
        with torch.no_grad():
            m_det = torch.dot(img.mean(0), txt.mean(0))

    need_L = mix != 0.0 and mix_unit_scale
    targ_fns, consts_list, L_values = [], [], []
    for k, (crit, secondary) in enumerate(crits):
        targ_fn = make_targ_block_fn(crit.cfg["targ"], class_encs_b, targ_data_b, B, device)
        consts, L_val = _precompute_crit_consts(crit, secondary, img, txt, targ_fn, class_encs_b, B,
                                                compute_logits, chunk_size, mixed_prec, device, need_L, autocast_ctx,
                                                lo, hi, world_size,
                                                centers[k], m_det if centers[k] == "sim" else None)
        targ_fns.append(targ_fn); consts_list.append(consts); L_values.append(L_val)

    mix_w = [1.0] if mix == 0.0 else [1.0 - mix, mix]
    if need_L:
        coeffs = [mix_w[k] / L_values[k].clamp_min(1e-12) for k in range(len(crits))]  # mix_unit_scale: /Lk.detach()
    else:
        coeffs = list(mix_w)

    # grad_proj*: the detached full-batch incoming-grad mean per criterion -- the constant every tile
    # of the grad sweep subtracts in place of the full path's g.mean() (_ZeroSumGradConst)
    center_consts = [
        _crit_center_grad_mean(crit, secondary, img, txt, targ_fns[k], class_encs_b, B, consts_list[k],
                               coeffs[k], compute_logits, chunk_size, autocast_ctx, lo, hi, world_size, device)
        if centers[k] in ("grad_proj", "grad_proj2") else None
        for k, (crit, secondary) in enumerate(crits)
    ]

    # sim-grad-sum metric (learning-curve strip): accumulated tile-by-tile via backward hooks. Under
    # "sim" centering part of the full-path dL/dsim routes through the global mean, which here lives
    # on the leaves and bypasses the tiles -- a hook on the in-graph mean captures it (dm/dsim sums
    # to exactly 1), so the folded totals match the full-batch sim.grad.sum().
    grad_sums = [torch.zeros((), dtype=torch.float64, device=device) for _ in crits]

    wbce_tot = [torch.zeros((), dtype=torch.float64, device=device) for _ in crits]
    raw_tot = [torch.zeros((), dtype=torch.float64, device=device) for _ in crits]
    stats = _SimTargStatsAccum(device)

    for rs in range(lo, hi, chunk_size):
        re = rs + chunk_size  # the band is an exact multiple of chunk_size (checked above)
        targ_blocks = []
        with autocast_ctx():
            block_loss = 0.0
            sim1_block = None
            for k, (crit, secondary) in enumerate(crits):
                cg = center_consts[k]
                if centers[k] == "sim":
                    # in-graph per block: the centering's backward routes through the mean embeddings
                    # into every leaf row, completing the exact full-batch projection across blocks
                    cg = torch.dot(img.mean(0), txt.mean(0))
                    if cg.requires_grad:
                        cg.register_hook(_gsum_hook(grad_sums[k]))
                sim_block, logits_f = _crit_block_logits_f(crit, secondary, img[rs:re], txt, compute_logits, centers[k], cg)
                if sim_block.requires_grad:
                    sim_block.register_hook(_gsum_hook(grad_sums[k]))
                targs_block = targ_fns[k](rs, re)
                W, bce = _crit_block_weight_bce(crit, logits_f, targs_block, class_encs_b[rs:re], class_encs_b, B, consts_list[k])
                num = (W * bce).sum()
                block_loss = block_loss + coeffs[k] * num / B
                wbce_tot[k] += num.detach().double()
                raw_tot[k] += bce.sum().detach().double()
                targ_blocks.append(targs_block.detach())
                if k == 0:
                    sim1_block = sim_block.detach()
        block_loss.backward()
        targ_stat = targ_blocks[0] if mix == 0.0 else (1.0 - mix) * targ_blocks[0] + mix * targ_blocks[1]
        stats.update(sim1_block, targ_stat)

    if world_size > 1:  # fold the band-partial loss totals; the leaves' .grad stay band-partial
        packed = torch.stack(wbce_tot + raw_tot + grad_sums)
        dist.all_reduce(packed)
        wbce_tot = [packed[k] for k in range(len(crits))]
        raw_tot = [packed[len(crits) + k] for k in range(len(crits))]
        grad_sums = [packed[2 * len(crits) + k] for k in range(len(crits))]

    loss = torch.zeros((), dtype=torch.float64, device=device)
    loss_raw = torch.zeros((), dtype=torch.float64, device=device)
    for k in range(len(crits)):
        loss += coeffs[k] * (wbce_tot[k] / B)
        loss_raw += mix_w[k] * (raw_tot[k] / B)
    grad_sum_sims = (grad_sums[0].item(), grad_sums[1].item() if len(crits) == 2 else None)
    return loss.float(), loss_raw.float(), stats.finalize(world_size), grad_sum_sims
