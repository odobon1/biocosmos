import torch
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import open_clip
from open_clip.pretrained import get_pretrained_cfg, download_pretrained
import abc
import math
from contextlib import nullcontext
from typing import List, Tuple, Any, Dict, Optional

from utils.utils import paths
from utils.loss import (
    Criterion,
    chunked_bce_loss_backward,
    hard_pair_similarity_margin,
    infonce_batch_stats,
    HIST_BINS,
    pos_prevalence,
    sep_logit_scalars,
)
from utils.head import compute_sim
from utils.data import make_image_preprocessor_inference, make_image_preprocessor_train, normalize_imgs_u8
from utils.config import TrainConfig

import pdb


def resolve_bias_init(config):
    """
    loss.logits.bce.bias.init as a float (or None): pos_prevalence -> logit of the loss's expected weighted
    positive prevalence over its blended target matrix (utils.loss.pos_prevalence), so the initial sigmoid
    matches the target prior where the scaled sims average to zero.
    """
    init = config.loss["logits"]["bce"]["bias"]["init"]
    if init == "pos_prevalence":
        p = pos_prevalence(config.loss, config.loss["loss1"], config.loss["loss2"], config.dataset, config.split, config.train_pt, config.batch_size)
        init = math.log(p / (1 - p))
        if dist.get_rank() == 0:
            print(f"logit bias init from positive prevalence: p = {p:.4g} -> bias = {init:.4f}")
    return init


#                            open_clip model name            pretrain  quick-gelu
CLIP_MODELS = {
    "bioclip":               ("hf-hub:imageomics/bioclip",   None,     False),
    "bioclip2":              ("hf-hub:imageomics/bioclip-2", None,     False),
    "clip_vitb16":           ("ViT-B-16",                    "openai", True),
    "clip_vitb32":           ("ViT-B-32",                    "openai", True),
    "clip_vitl14":           ("ViT-L-14",                    "openai", True),
    "clip_vitl14_336":       ("ViT-L-14-336",                "openai", True),
}
SIGLIP_MODELS = {
    "siglip_vitb16":         ("ViT-B-16-SigLIP",             "webli",  False),
    "siglip_vitb16_256":     ("ViT-B-16-SigLIP-256",         "webli",  False),
    "siglip_vitb16_384":     ("ViT-B-16-SigLIP-384",         "webli",  False),
    "siglip_vitl16_256":     ("ViT-L-16-SigLIP-256",         "webli",  False),
    "siglip_vitl16_384":     ("ViT-L-16-SigLIP-384",         "webli",  False),
    "siglip_vitso400m14":    ("ViT-SO400M-14-SigLIP",        "webli",  False),
    "siglip2_vitb16":        ("ViT-B-16-SigLIP2",            "webli",  False),
    "siglip2_vitb16_384":    ("ViT-B-16-SigLIP2-384",        "webli",  False),
    "siglip2_vitl16_384":    ("ViT-L-16-SigLIP2-384",        "webli",  False),
    "siglip2_vitso400m14":   ("ViT-SO400M-14-SigLIP2",       "webli",  False),
    "siglip2_vitgopt16_384": ("ViT-gopt-16-SigLIP2-384",     "webli",  False),
}

class _AllGather(torch.autograd.Function):
    """
    Autograd-friendly all_gather for embeddings.
    Forward: gathers tensors from all ranks.
    Backward: routes and all-reduces the gradient for the local shard.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        world_size = dist.get_world_size()
        ctx.world_size = world_size
        out = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(out, x)
        return tuple(out)

    @staticmethod
    def backward(ctx: Any, *grads: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if ctx.world_size == 1:
            return grads[0]
        grad_stack = torch.stack(grads, dim=0)
        dist.all_reduce(grad_stack)
        grad_input = grad_stack[dist.get_rank()]
        return grad_input

class _ZeroSumGrad(torch.autograd.Function):
    """
    Identity forward; backward projects the incoming gradient to zero-sum (g - g.mean()), the
    minimal L2 modification satisfying sum(g) = 0 (logits.bce.center: grad_proj/grad_proj2).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> torch.Tensor:
        return g - g.mean()

class _ZeroSumGradConst(torch.autograd.Function):
    """
    Identity forward; backward subtracts a precomputed constant from the incoming gradient. The
    chunked path's _ZeroSumGrad stand-in: over a [C, B] tile the per-tile g.mean() is wrong, so the
    tiled sweep precomputes the full-batch incoming-grad mean and every tile subtracts that constant,
    reproducing the full-batch projection exactly (see chunked_bce_loss_backward).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ctx.c = c
        return x

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return g - ctx.c, None

def sim_targ_batch_stats(sim: torch.Tensor, targs: torch.Tensor, hpsm_kappas: List[float],
                         logits: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Per-batch distribution stats over the full BxB similarity and target matrices; drives
    the batch-level learning-curve strips, sim_targ.log, and the eval sim/targ sections
    (per-chunk, averaged across chunks).

    - sim ----- similarity matrix, already on [-1, 1] (cosine / geodesic-mapped)
    - targs --- the blended target matrix on [0, 1] (Criterion.targ_memb)
    - logits -- the logits (scale/bias applied to sim), or None to skip the p_hist entry. Pass
                them only for a BCE-family loss, where sigmoid(logits) is the
                predicted pair probability -- on [0, 1] like the targets, so the P and Q strips
                are directly comparable. Under InfoNCE the row-softmax carries no such reading.
    - hpsm_kappas -- the kappa values the mean hard-pair similarity margins are reported at
                (reporting.learning_curves.hpsm.kappas; one curve-strip line each).

    Returns min/max/median/mean for sim/targ as flat keys (the batch logs read those), the mean
    hard-pair similarity margins (hard_pair_similarity_margin averaged over anchors, each a list
    with one entry per hpsm_kappas value): sim_margin_i2t over S (image anchors), sim_margin_t2i
    over S.T (text anchors) and sim_margin, their mean -- the bidirectional reading; each its own
    curve strip -- plus targ_hist and -- when logits are given -- p_hist: the targets and predicted
    probabilities as HIST_BINS
    fractions over [0, 1] (distributions, not point stats; the curve strips render them as heatmap
    columns). Reductions are stacked so the device->host transfer is a single .cpu() sync.
    """
    with torch.no_grad():
        s = sim.detach()
        t = targs.detach()
        reductions = [
            s.min(), s.max(), s.median(), s.mean(),
            t.min(), t.max(), t.median(), t.mean(),
        ]
        S, Q = s.float(), t.float()
        # per kappa, the I2T (image-anchored rows of S) and T2I (text-anchored rows of S.T) means
        reductions += [hard_pair_similarity_margin(M, N, kappa).mean() for kappa in hpsm_kappas for M, N in ((S, Q), (S.T, Q.T))]
        packed = torch.stack([r.float() for r in reductions])
        packed = torch.cat([packed, torch.histc(t.float(), bins=HIST_BINS, min=0.0, max=1.0) / t.numel()])
        if logits is not None:
            p = logits.detach().float().sigmoid()
            packed = torch.cat([packed, torch.histc(p, bins=HIST_BINS, min=0.0, max=1.0) / p.numel()])
        vals = packed.cpu().tolist()
    n_kappas = len(hpsm_kappas)
    margin_i2t = vals[8:8 + 2 * n_kappas:2]
    margin_t2i = vals[9:8 + 2 * n_kappas:2]
    stats = {
        "sim_min":     vals[0],
        "sim_max":     vals[1],
        "sim_median":  vals[2],
        "sim_mean":    vals[3],
        "targ_min":    vals[4],
        "targ_max":    vals[5],
        "targ_median": vals[6],
        "targ_mean":   vals[7],
        "sim_margin_i2t": margin_i2t,
        "sim_margin_t2i": margin_t2i,
        "sim_margin":     [0.5 * (a + b) for a, b in zip(margin_i2t, margin_t2i)],
        "targ_hist":      vals[8 + 2 * n_kappas:8 + 2 * n_kappas + HIST_BINS],
    }
    if logits is not None:
        stats["p_hist"] = vals[8 + 2 * n_kappas + HIST_BINS:]
    return stats

def stat_logits(logits, cfg_loss):
    """The logits sim_targ_batch_stats should derive p_hist from: the branch tuple's first entry
    (identical values across branches) for a BCE-family loss, else None -- sigmoid is only the
    model's pair probability under the sigmoid-BCE path."""
    return logits[0] if cfg_loss["crit"] in ("bce", "bif_bce") else None

class VLMWrapper(abc.ABC):
    """
    Base class for all vision-language model wrappers. Dispatches to an appropriate subclass based on model_type and
    centralizes common initialization logic (e.g. device, tokenizer, checkpoint, etc.)
    """
    def __init__(
        self, 
        config: TrainConfig, 
        model_name: str, 
        pretrained: str, 
        quick_gelu: bool
    ) -> None:
        """
        Args:
        - config ------- Configuration object
        - model_name --- open_clip model identifier
        - pretrained --- Pretraining dataset
        - quick_gelu --- Whether to use the QuickGELU activations
        """
        self.cfg = config

        # Vision-tower overrides, merged into one vision_cfg passed via model_kwargs (which replaces vision_cfg
        # wholesale, so every override is folded into a copy of the base config).
        #   - vis_proj_head -> timm_proj: an ARCHITECTURE choice, applied for train AND eval so a checkpoint's
        #     projection head reloads with a matching module.
        #   - patch/head/stochastic-depth dropout: parameterless and train-only (applied only when the model is in
        #     train mode; eval runs in eval mode regardless).
        # Native CLIP takes only patch_dropout, via open_clip's force_patch_dropout.
        force_patch_dropout = None
        vision_cfg_extra = {}
        is_siglip = config.arch["model_type"] in SIGLIP_MODELS

        if is_siglip and config.arch["siglip"]["vis_proj_head"] is not None:
            vision_cfg_extra["timm_proj"] = config.arch["siglip"]["vis_proj_head"]

        if hasattr(config, "dropout"):
            dropout = config.dropout
            if is_siglip:
                vision_cfg_extra["patch_dropout"] = dropout["patch_dropout"]
                vision_cfg_extra["timm_drop"] = dropout["siglip"]["proj_head"]
                if dropout["siglip"]["stoch_depth"] is not None:
                    vision_cfg_extra["timm_drop_path"] = dropout["siglip"]["stoch_depth"]
            else:
                force_patch_dropout = dropout["patch_dropout"]

        model_kwargs = {}
        if vision_cfg_extra:
            model_kwargs["vision_cfg"] = {
                **open_clip.get_model_config(model_name)["vision_cfg"],
                **vision_cfg_extra,
            }

        # timm_proj adds head params absent from the released checkpoint, which open_clip strict-loads with no
        # opt-out -- skip its weight load and reload non-strict, allowing missing keys only under visual.head.
        load_weights = "timm_proj" not in vision_cfg_extra

        model, img_pp_train, img_pp_inf = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            load_weights=load_weights,
            force_quick_gelu=quick_gelu,
            force_patch_dropout=force_patch_dropout,
            cache_dir=paths["hf_cache"],
            **model_kwargs,
        )

        if not load_weights:
            ckpt_path = download_pretrained(get_pretrained_cfg(model_name, pretrained), cache_dir=paths["hf_cache"])
            incompatible = open_clip.load_checkpoint(model, ckpt_path, strict=False)
            bad_missing = [k for k in incompatible.missing_keys if not k.startswith("visual.head.")]
            if bad_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Pretrained load mismatch beyond the vision head -- "
                    f"missing: {bad_missing}, unexpected: {incompatible.unexpected_keys}"
                )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.device = config.device
        self.type = config.arch["model_type"]
        self.model = model.to(self.device).eval()
        self.img_pp_train = img_pp_train
        self.img_pp_inf = img_pp_inf
        # base-model default normalization stats, from the released inference pipeline's trailing Normalize
        self.norm_mean = img_pp_inf.transforms[-1].mean
        self.norm_std = img_pp_inf.transforms[-1].std

        tokenizer = open_clip.get_tokenizer(model_name)
        self.txt_pp = lambda txts: tokenizer(txts).to(self.device)

        if config.hw.act_chkpt:
            self.model.set_grad_checkpointing(True)

        if hasattr(config, "loss"):
            cfg_logits = config.loss["logits"]
            if cfg_logits["scale"]["init"] is not None:  # scale.init set in config
                if hasattr(self.model, "logit_scale"):  # logit_scale attribute exists
                    with torch.no_grad():
                        self.model.logit_scale.fill_(math.log(cfg_logits["scale"]["init"]))  # alpha -> log(alpha)
            # the logit bias is the sigmoid / BCE path's, so it goes with the CRITERION, not the model family
            # (either family may train under either: a CLIP model under bce, a SigLIP model under infonce):
            # - InfoNCE carries none, whichever the family -- logit_bias = None, as open_clip itself spells a
            #   bias-free model, and compute_logits then adds nothing. A shared scalar bias cannot move a row
            #   softmax, so nothing is lost; and it is dropped outright rather than left in unused, which
            #   would leave a SigLIP model's trainable parameter for DDP to wait on a gradient for (no
            #   find_unused_parameters)
            # - a BCE-family loss always carries one: a SigLIP model's own, and for a CLIP model (none of its
            #   own) a fixed 0.0 buffer under bias.init: null, a learnable parameter under a set one
            if config.loss["crit"] == "infonce":
                self.model.logit_bias = None
            else:
                bias_init = resolve_bias_init(config)
                if bias_init is None:  # (bias.init: null) in config
                    if self.model.logit_bias is None:  # logit bias attribute is None (CLIP default)
                        delattr(self.model, "logit_bias")
                        self.model.register_buffer("logit_bias", torch.tensor(0.0, device=self.device))
                else:  # bias.init set in config
                    if isinstance(self.model.logit_bias, nn.Parameter):  # logit_bias attribute is a nn.Parameter
                        with torch.no_grad():
                            self.model.logit_bias.fill_(bias_init)
                    else:  # logit_bias attribute is not a nn.Parameter
                        delattr(self.model, "logit_bias")
                        self.model.register_parameter("logit_bias", nn.Parameter(torch.tensor(bias_init, device=self.device)))
            if cfg_logits["scale"]["freeze"] and isinstance(self.model.logit_scale, nn.Parameter):
                self.model.logit_scale.requires_grad_(False)
            if cfg_logits["bce"]["bias"]["freeze"] and isinstance(self.model.logit_bias, nn.Parameter):
                self.model.logit_bias.requires_grad_(False)

            # separate logit scalars (loss.logits.shared false under a live loss blend): loss2's term runs on
            # its own pair, logit_scale2 / logit_bias2 (compute_logits' secondary) -- a copy of the first
            # pair as just initialized, so the two share loss.logits.* (init, freeze) and part ways in training.
            # Under InfoNCE there is no bias to copy, so the second pair has none either (logit_bias2 = None)
            if sep_logit_scalars(config.loss):
                for attr in ("logit_scale", "logit_bias"):
                    scalar = getattr(self.model, attr)
                    if scalar is None:
                        setattr(self.model, f"{attr}2", None)
                    elif isinstance(scalar, nn.Parameter):
                        self.model.register_parameter(f"{attr}2", nn.Parameter(scalar.detach().clone(), requires_grad=scalar.requires_grad))
                    else:
                        self.model.register_buffer(f"{attr}2", scalar.detach().clone())

    @classmethod
    def build(cls, config: TrainConfig, verbose: bool) -> Any:
        """
        Factory method to construct the appropriate VLM wrapper based on configuration.

        Args:
        - config ---- Configuration object containing architecture and loss settings
        - verbose --- Whether to print loading status

        Returns:
        - Initialized VLMWrapper subclass instance
        """
        if verbose:
            print("Loading base model...")

        if config.arch["model_type"] in CLIP_MODELS:
            modelw = CLIPWrapper(config)
        elif config.arch["model_type"] in SIGLIP_MODELS:
            modelw = SigLIPWrapper(config)
        else:
            raise ValueError(f"Unknown model_type: '{config.arch['model_type']}'")

        modelw.set_image_preprocessors()

        if config.arch["clip"]["non_causal"]:
            modelw.disable_causal_mask_text()

        if verbose:
            print("Model loaded!\n")

        return modelw

    @property
    def _unwrapped_model(self) -> nn.Module:
        """
        Helper to get the underlying model, handling DDP wrapping.
        """
        return self.model.module if hasattr(self.model, "module") else self.model

    @property
    def embed_dim(self) -> int:
        model = self._unwrapped_model
        if isinstance(self, CLIPWrapper):
            return model.text_projection.weight.shape[0]
        if isinstance(self, SigLIPWrapper):
            return model.text.text_projection.weight.shape[0]
        raise TypeError(f"Unsupported wrapper type: {type(self).__name__}")

    def set_image_preprocessors(self) -> None:
        self.img_pp_inf = make_image_preprocessor_inference(self.img_res)
        if hasattr(self.cfg, "aug_cfg"):
            self.img_pp_train = make_image_preprocessor_train(
                self.img_res,
                aug_cfg=self.cfg.aug_cfg,
            )
        # normalization runs on-device (prep_imgs), not in the preprocessors: [C, 1, 1] fp32 tensors
        self._norm_mean_t = torch.as_tensor(self.norm_mean, dtype=torch.float32, device=self.device).view(-1, 1, 1)
        self._norm_std_t = torch.as_tensor(self.norm_std, dtype=torch.float32, device=self.device).view(-1, 1, 1)

    def prep_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        uint8 image batch from a dataloader (CPU or device) -> normalized fp32 on device: the
        second half of the preprocessing contract (the Compose emits raw uint8 so workers/queues
        carry 1 byte per element and the H2D copy is 4x smaller).
        """
        imgs = imgs.to(self.device, non_blocking=True)
        return normalize_imgs_u8(imgs, self._norm_mean_t, self._norm_std_t)

    def embed_images(self, imgs_b: torch.Tensor) -> torch.Tensor:
        """
        Runs batch of images through image encoder and returns batch of unit-length embeddings.

        Args:
        - imgs_b --- Batch of preprocessor-emitted uint8 images (CPU or device); Tensor(B, C, H, W)

        Returns:
        - Normalized image embeddings; Tensor(B, D)
        """
        model = self._unwrapped_model

        imgs_b = self.prep_imgs(imgs_b)
        with torch.set_grad_enabled(self.model.training):  # enable autograd if model in train mode, disable if in eval mode
            embs_imgs_b = model.encode_image(imgs_b)

        embs_imgs_b = F.normalize(embs_imgs_b, dim=1)

        return embs_imgs_b

    def embed_texts(self, txts: List[str]) -> torch.Tensor:
        """
        Runs texts through text encoder and returns unit-length embeddings.

        Args:
        - txts --- List of texts of length L

        Returns:
        - Normalized text embeddings; Tensor(L, D)
        """
        model = self._unwrapped_model

        toks_txts = self.txt_pp(txts)  # --------------- Tensor(L, T)
        with torch.set_grad_enabled(self.model.training):
            embs_txts = model.encode_text(toks_txts)

        embs_txts = F.normalize(embs_txts, dim=1)  # --- Tensor(L, D)

        return embs_txts

    def compute_logits(
        self, 
        sim: torch.Tensor,
        clamp_scale: bool,
        center: Optional[str],
        center_global: Optional[torch.Tensor] = None,
        half_live: bool = False,
        secondary: bool = False,
    ) -> torch.Tensor:
        """
        Scales similarity matrix by exp(learnable logit scale) (alpha = 1 / tau) and adds the logit bias where the
        model carries one: under a BCE-family loss, never under InfoNCE (logit_bias is None there -- see __init__).

        The head runs in float32 whatever the sims came in as: under bf16 autocast the cos sims arrive in bf16,
        and scaling and biasing them THERE rounds the result to bf16's grid at the LOGITS' exponent, which near
        the sigmoid's decision threshold -- alpha * sim + bias ~ 0, the two terms cancelling -- is far coarser
        than the pairs' own differences. At SigLIP's own scale (alpha ~ 117, bias -12.93) the bf16 sims
        0.11035 and 0.11084 both come out at logit 0.0, where float32 arithmetic on the SAME bf16 sims gives
        -0.0213 and +0.0359. So the cast comes first: upcasting already-rounded logits afterwards (as the BCE
        criteria do) restores nothing. A large alpha is no protection -- it is the cancellation that sets the
        exponent -- and the sims' own bf16 quantization is untouched by this, being upstream of the head.

        `secondary` selects the second logit-scalar pair (logit_scale2 / logit_bias2), loss2's term's own
        under separate logit scalars (loss.logits.shared false; utils.loss.sep_logit_scalars) -- the pair
        exists only then.

        `half_live` (bifurcated branches): uses 0.5*p + 0.5*p.detach() for the logit scale/bias, so
        each of the two un-halved branch calls contributes exactly half their grad -- the branch sum
        matches the non-bifurcated 1x (the towers, living in one branch each, already get 1x).
        Values are unchanged.

        `clamp_scale` caps the logit scale at ln(100) before exp() (scale multiplier <= 100, CLIP's stability
        cap); otherwise exp() is unbounded and can overflow to +inf and amplify the bf16 quantization of sim.

        `center` (loss.logits.bce.center) makes dL/dsim zero-sum:
        - None ---------- no centering (plain scale + bias).
        - "sim" --------- centers the scaled sims about their mean before the bias; changes the forward
          (batch-dependent operating-point shift), and the zero-sum backward follows from it.
        - "grad_proj" --- forward untouched; backward-only global-mean projection of the gradient
          entering sim (_ZeroSumGrad), so logit scale AND bias learn from the raw BCE gradient.
        - "grad_proj2" -- forward untouched; projection applied at the scaled sims instead: bias keeps
          its raw gradient, logit scale receives the projected gradient.
        The encoder gradient is identical for both grad_proj variants.

        `center_global` (chunked path only) carries the precomputed full-batch quantity that makes tiled
        centering exact instead of per-tile: for "sim" the IN-GRAPH global sim mean (from the cos mean
        factorization mean(sim) = mean(img) . mean(txt), computed from the embedding leaves -- config
        rejects center: sim + chunking for geo sims); for grad_proj/grad_proj2 the DETACHED full-batch
        mean of the incoming gradient at the projection node (_ZeroSumGradConst). None (full-batch path)
        -> the full BxB reductions computed here.
        """
        model = self._unwrapped_model
        sim = sim.float()
        if secondary:
            logit_scale, logit_bias = model.logit_scale2, model.logit_bias2
        else:
            logit_scale, logit_bias = model.logit_scale, model.logit_bias
        if half_live:
            logit_scale = 0.5 * logit_scale + 0.5 * logit_scale.detach()
            logit_bias = 0.5 * logit_bias + 0.5 * logit_bias.detach()
        if clamp_scale:
            logit_scale = logit_scale.clamp(max=math.log(100))

        if center == "grad_proj":
            # scale + bias see raw grads
            sim = _ZeroSumGrad.apply(sim) if center_global is None else _ZeroSumGradConst.apply(sim, center_global)

        sim_scaled = sim * logit_scale.exp()

        if center == "grad_proj2":
            # bias raw; scale sees projected grad
            sim_scaled = _ZeroSumGrad.apply(sim_scaled) if center_global is None else _ZeroSumGradConst.apply(sim_scaled, center_global)

        if center == "sim":
            sim_scaled = sim_scaled - (sim_scaled.mean() if center_global is None else center_global * logit_scale.exp())

        if logit_bias is None:  # InfoNCE: no bias (see __init__), the row softmax scores the scaled sims
            return sim_scaled

        logits = sim_scaled + logit_bias

        return logits

    def freeze(self, freeze_txt: bool, freeze_img: bool) -> None:
        """
        Freezes parameters of text and/or image encoders.
        """
        if freeze_txt: self.freeze_text_encoder()
        if freeze_img: self.freeze_image_encoder()

    def freeze_image_encoder(self) -> None:
        """
        Freezes vision encoder parameters.
        """
        for name, param in self.model.named_parameters():
            if name.startswith("visual."):
                param.requires_grad = False

    @abc.abstractmethod
    def freeze_text_encoder(self) -> None:
        """
        Abstract method to freeze text encoder parameters. Must be implemented in subclasses.
        """
        raise NotImplementedError

    def _loss_full_batch(
        self,
        embs_img_all: torch.Tensor,
        embs_txt_all: torch.Tensor,
        class_encs_all: torch.Tensor,
        targ_data_all: List[Any],
    ) -> torch.Tensor:
        """
        Computes the loss for the full global batch under the criterion (self.crit).

        The similarity/logit matrices are returned as branch tuples, every branch [img-row,
        txt-col], so downstream grad logging sums branches elementwise regardless of variant:

        - non-bifurcated crits: 1-tuples (sim,) / (logits,).
        - bifurcated crits: 2-tuples (i2t, t2i). The i2t branch detaches the text embeddings, the
          t2i branch the image embeddings, so each branch's gradient flows into one tower only;
          the branches rejoin only at the final scalar loss (the criterion consumes the t2i
          branch as its transposed [txt-row, img-col] view). The un-halved branch sum makes the
          loss value (and loss_raw) 2x the non-bifurcated reading, but every gradient matches
          non-bifurcated 1x: towers live in one branch each, and the logit scale/bias are
          half-live (see compute_logits) so their two branch contributions sum to 1x. Branch
          values are identical up to fp; grads bifurcate.

        Under separate logit scalars (Criterion.sep_scalars) the sims are scaled once per loss term, each
        under its own scalar pair (compute_logits' secondary), and the criterion takes the terms' logits and
        log-scales; the returned `logits` are the primary term's (loss1's), which the logit grad-norm log and
        the logit-dependent batch stats then read.

        Returns (loss, loss_raw, logits, sims, targs, y): targs the blended target matrix Q the batch
        stats read (Criterion.targ_memb), y the blended target distribution the criterion trained
        against (Criterion.targ_dist; under loss.unitless a training InfoNCE loss returns the terms'
        distributions under their normalized blend coefficients -- the distribution its gradient follows;
        under separate logit scalars the primary term's distribution).
        """
        model = self._unwrapped_model
        crit = self.crit
        clamp = crit.cfg["logits"]["scale"]["clamp"]
        # loss.logits.bce.* is the sigmoid / BCE path's and declared inert under InfoNCE (utils.config.inert_params),
        # so it is not read there. It is a no-op there in exact arithmetic -- a row softmax is shift-invariant, so
        # dL/dsim has zero row (i2t) and column (t2i) sums under any weighting that reads the logits through p:
        # grad_proj* would subtract a mean that is zero, `sim` shift every logit alike -- but not a BIT-exact one
        # (the mean comes out ~1e-8), and an inert setting should leave a run bit-identical
        center = crit.cfg["logits"]["bce"]["center"] if crit.cfg["crit"] != "infonce" else None
        secondaries = (False, True) if crit.sep_scalars else (False,)

        if crit.bifurcated:
            sims = (
                compute_sim(embs_img_all, embs_txt_all.detach(), crit.cfg["sim"]),
                compute_sim(embs_img_all.detach(), embs_txt_all, crit.cfg["sim"]),
            )
            logits_terms = [tuple(self.compute_logits(sim, clamp, center, half_live=True, secondary=secondary) for sim in sims) for secondary in secondaries]
            crit_logits = logits_terms
        else:
            sims = (compute_sim(embs_img_all, embs_txt_all, crit.cfg["sim"]),)
            logits_terms = [(self.compute_logits(sims[0], clamp, center, secondary=secondary),) for secondary in secondaries]
            crit_logits = [logits[0] for logits in logits_terms]
        logits = logits_terms[0]

        if crit.sep_scalars:
            logit_scale = (model.logit_scale, model.logit_scale2)
        else:
            crit_logits, logit_scale = crit_logits[0], model.logit_scale

        # sims[0]: branch values are identical, and an InfoNCE loss (the only reader) is never bifurcated
        loss, loss_raw, targs, y = crit(crit_logits, class_encs_all, targ_data_all, self.model.training, logit_scale, sims[0])

        return loss, loss_raw, logits, sims, targs, y

    def _batch_stats(
        self,
        sims: Tuple[torch.Tensor, ...],
        targs: torch.Tensor,
        y: torch.Tensor,
        logits: Tuple[torch.Tensor, ...],
    ) -> Dict[str, Any]:
        """
        The batch's stats: sim_targ_batch_stats (sims[0] / logits[0]: branch values are identical, so
        the first branch carries them) plus, for an InfoNCE loss, the reachable-optimum diagnostics
        (infonce_batch_stats: the logit-scale gradient and the KL decompositions, and the row-wise
        target-implied scale bounds), which take the raw log-scale parameter and its clamp flag: from
        them the alpha the logits carry (post-clamp, as compute_logits applies it) and, for the
        parameter's own gradient, whether the clamp holds it.
        """
        cfg_loss = self.cfg.loss
        kappas = self.cfg.reporting["learning_curves"]["hpsm"]["kappas"]
        stats = sim_targ_batch_stats(sims[0], targs, kappas, logits=stat_logits(logits, cfg_loss))
        if cfg_loss["crit"] == "infonce":
            logit_scale = self._unwrapped_model.logit_scale
            stats.update(infonce_batch_stats(sims[0], targs, y, logits[0], logit_scale.detach(), cfg_loss["logits"]["scale"]["clamp"]))
        if self.crit.lambda_eff is not None:  # a unitless loss blend's effective lambda (Criterion.term_coeffs)
            stats["lambda_eff"] = self.crit.lambda_eff.item()
        if self.crit.dlogalpha_correction is not None:  # block_residuals' delta to logit_scale's gradient
            stats["dlogalpha_correction"] = self.crit.dlogalpha_correction.item()
        return stats

    def _global_batch_loss(
        self,
        embs_img_sb: torch.Tensor,
        embs_txt_sb: torch.Tensor,
        class_encs_sb: torch.Tensor,
        targ_data_sb: List[Any],
        loss_flag: bool = True,
    ) -> torch.Tensor:
        """
        Compute loss using the full global batch:
        - gathers embeddings + targets across GPUs,
        - applies the criterion over its blended targets.
        Works for single GPU as well (gather is a no-op).
        """
        embs_img_b, embs_txt_b, class_encs_b, targ_data_b = self._gather_batch(
            embs_img_sb, embs_txt_sb, class_encs_sb, targ_data_sb
        )
        # the gathered embeddings are what batch_step returns to the grad-norm logger; retain so
        # .grad carries the full-batch dL/dembs after backward (same quantity the chunked path
        # all-reduces into its returned leaves). Each reporting.batch_diagnostics component gates only its
        # own retains/stats (emb_logit_grads the embedding+logit retains, sim_grad_sums the sim
        # retains, sim_targ_stats the batch stats) -- the loss/gradient path is untouched
        diag = self.cfg.reporting["batch_diagnostics"]
        if diag["emb_logit_grads"]:
            if embs_img_b.requires_grad:
                embs_img_b.retain_grad()
            if embs_txt_b.requires_grad:
                embs_txt_b.retain_grad()

        if not loss_flag:
            return None, None, embs_img_b, embs_txt_b, None, class_encs_b, None, None

        loss, loss_raw, logits, sims, targs, y = self._loss_full_batch(embs_img_b, embs_txt_b, class_encs_b, targ_data_b)
        # retain every branch's grad for the aggregate (branch-summed) grad logging: logits for the
        # logit= log field, sims for the sim-grad-sum curve strip
        for t in (*(logits if diag["emb_logit_grads"] else ()), *(sims if diag["sim_grad_sums"] else ())):
            if t.requires_grad:
                t.retain_grad()

        batch_stats = self._batch_stats(sims, targs, y, logits) if diag["sim_targ_stats"] else None
        return loss, loss_raw, embs_img_b, embs_txt_b, logits, class_encs_b, batch_stats, sims

    def _gather_batch(
        self,
        embs_img_sb: torch.Tensor,
        embs_txt_sb: torch.Tensor,
        class_encs_sb: torch.Tensor,
        targ_data_sb: List[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
        """
        Turn the local batch on this rank into a global batch across all ranks.
        Works for both single GPU (no-op) and DDP.
        Supports uneven local batch sizes by padding to the max local size,
        gathering, then trimming back to the true sizes.
        """
        if self.world_size == 1:
            return embs_img_sb, embs_txt_sb, class_encs_sb, targ_data_sb

        device = embs_img_sb.device
        SB = embs_img_sb.size(0)

        # gather true local batch sizes from all ranks
        sizes_t = torch.tensor([SB], device=device, dtype=torch.long)
        sizes_list = [torch.zeros_like(sizes_t) for _ in range(self.world_size)]
        dist.all_gather(sizes_list, sizes_t)
        sizes = [int(x.item()) for x in sizes_list]

        SB_max = max(sizes)

        def pad_rows(x: torch.Tensor, SB_target: int) -> torch.Tensor:
            SB = x.size(0)
            if SB == SB_target:
                return x
            pad_shape = (SB_target - SB, *x.shape[1:])
            pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        # pad tensors so all_gather can handle the last uneven batch
        embs_img_pad = pad_rows(embs_img_sb, SB_max)
        embs_txt_pad = pad_rows(embs_txt_sb, SB_max)
        class_encs_pad = pad_rows(class_encs_sb, SB_max)

        # embeddings (need gradients)
        img_parts_pad = _AllGather.apply(embs_img_pad)
        txt_parts_pad = _AllGather.apply(embs_txt_pad)

        # class encodings (no grad)
        class_parts_pad = [torch.empty_like(class_encs_pad) for _ in range(self.world_size)]
        dist.all_gather(class_parts_pad, class_encs_pad)

        # target metadata (Python objects) already supports variable-length lists
        targ_data_sb_list = [None] * self.world_size
        dist.all_gather_object(targ_data_sb_list, list(targ_data_sb))

        # trim each gathered shard back to its true local size
        img_parts = [part[:sizes[r]] for r, part in enumerate(img_parts_pad)]
        txt_parts = [part[:sizes[r]] for r, part in enumerate(txt_parts_pad)]
        class_parts = [part[:sizes[r]] for r, part in enumerate(class_parts_pad)]

        embs_img_b = torch.cat(img_parts, dim=0)
        embs_txt_b = torch.cat(txt_parts, dim=0)
        class_encs_b = torch.cat(class_parts, dim=0)

        targ_data_b = []
        for td_sb in targ_data_sb_list:
            targ_data_b += td_sb

        return embs_img_b, embs_txt_b, class_encs_b, targ_data_b

    def batch_step(
        self,
        imgs_sb: torch.Tensor,
        txts_sb: Tuple[str],
        class_encs_sb: torch.Tensor,
        targ_data_sb: Tuple[Any],
        loss_flag: bool = True,
    ) -> Tuple[Any]:
        """
        Performs a single forward pass step. Encodes images and text, computes loss if flag is set.

        Args:
        - imgs_sb --------- Sub-batch of images; pt[SB, C, H, W]
        - txts_sb --------- Sub-batch of texts
        - class_encs_sb --- Sub-batch of class encodings; pt[SB]
        - targ_data_sb ---- Sub-batch of target data
        - loss_flag ------- Whether to compute loss

        Returns:
        - loss ----------- Scalar loss (or None)
        - embs_img_b ----- Batch of normalized image embeddings; pt[B, D]
        - embs_txt_b ----- Batch of normalized text embeddings; pt[B, D]
        - logits --------- Branch tuple of [img-row, txt-col] logit matrices ((logits,) non-bifurcated,
                           (i2t, t2i) bifurcated -- see _loss_full_batch), grads retained for the
                           logit-grad-norm log; None if loss_flag is False
        - class_encs_b --- Batch of class encodings; pt[B]
        - batch_stats ---- Per-batch sim/target distribution stats (flat dict), or None if loss_flag is False
        - sims ----------- Branch tuple mirroring `logits`, grads retained for the sim-grad-sum strip
        """
        toks_sb = self.txt_pp(txts_sb)
        output = self.model(imgs_sb, toks_sb)

        embs_img_sb = F.normalize(output[0], dim=1)
        embs_txt_sb = F.normalize(output[1], dim=1)

        loss, loss_raw, embs_img_b, embs_txt_b, logits, class_encs_b, batch_stats, sims = self._global_batch_loss(
            embs_img_sb,
            embs_txt_sb,
            class_encs_sb,
            targ_data_sb,
            loss_flag=loss_flag,
        )

        return loss, loss_raw, embs_img_b, embs_txt_b, logits, class_encs_b, batch_stats, sims

    def batch_step_chunked(self, imgs_sb, txts_sb, class_encs_sb, targ_data_sb):
        """
        Memory-tiled training step for the global-batch BCE-family loss (bce/bif_bce;
        hardware.loss_chunk_size), used in place of batch_step + loss.backward() when chunking is on. Does the
        encoder forward, the tiled loss, AND the full backward internally, so the caller must NOT call
        loss.backward() afterwards.

        The BxB loss is never materialized, and no rank computes more than its share of it: after
        gathering the global-batch embeddings, they are detached into leaves and the BxB rows are
        sharded across ranks into equal row-bands (SigLIP-style decomposition of the pairwise-
        independent BCE loss; per-rank loss compute is O(B^2/world_size)). Each rank sums its band
        over C x B row-blocks, backpropagating each block into those leaves as it is computed
        (chunked_bce_loss_backward, which all-reduces the loss/stats so every rank returns full-batch
        values). A single representation-gradient backward then pushes the band-partial dL/dembs into
        the encoder. DDP grad sync is suppressed during these multiple backwards and replaced with one
        manual all-reduce of every parameter's grad, which completes the exact full-batch gradient:
        each rank's param grads -- encoder params via the _AllGather-routed band partials, post-gather
        logit scale/bias params via their direct band partials -- are disjoint-band contributions that
        SUM to the full-batch value, so there is no /world_size (unlike DDP's replicated-loss averaging).

        Returns batch_step's tuple shape, with loss/loss_raw detached, logits None (no full logit
        matrix is formed, so its grad-norm diagnostic is unavailable), the embedding leaves
        (carrying full-batch dL/dembs in .grad after a post-backward all-reduce) in place of
        embs_img_b / embs_txt_b for grad-norm logging, and -- since the backward already ran and the
        sim matrices are gone -- the sims slot carries the grad_sum_sim float accumulated
        tile-by-tile by chunked_bce_loss_backward. With reporting.batch_diagnostics.sim_targ_stats
        off batch_stats is None; with .sim_grad_sums off the sims slot carries None.
        """
        chunk = self.cfg.hw.loss_chunk_size
        mixed_prec = self.cfg.hw.mixed_prec
        device = self.cfg.device
        diag = self.cfg.reporting["batch_diagnostics"]
        kappas = self.cfg.reporting["learning_curves"]["hpsm"]["kappas"]

        # DDP.forward must run under no_sync too, so the reducer is never armed for this step (we sync
        # gradients manually below); otherwise DDP would expect a matching synced backward. The reducer
        # arms regardless of world size, so no_sync is required even at world_size 1 -- without it the
        # multiple tile backwards trip DDP's "marked ready only once" check.
        with self.model.no_sync():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_prec else nullcontext():
                toks_sb = self.txt_pp(txts_sb)
                output = self.model(imgs_sb, toks_sb)
                embs_img_sb = F.normalize(output[0], dim=1)
                embs_txt_sb = F.normalize(output[1], dim=1)
                embs_img_b, embs_txt_b, class_encs_b, targ_data_b = self._gather_batch(
                    embs_img_sb, embs_txt_sb, class_encs_sb, targ_data_sb
                )

            img = embs_img_b.detach().requires_grad_(True)
            txt = embs_txt_b.detach().requires_grad_(True)

            rank = dist.get_rank() if self.world_size > 1 else 0
            loss, loss_raw, batch_stats, grad_sum_sim = chunked_bce_loss_backward(
                img, txt, class_encs_b, targ_data_b, self.crit, self.compute_logits, chunk, mixed_prec, device,
                rank, self.world_size, sim_grad_sums=diag["sim_grad_sums"],
                sim_targ_stats=diag["sim_targ_stats"], hpsm_kappas=kappas
            )

            # representation gradient: push the accumulated band-partial dL/dembs into the encoder (one
            # backward, only for sides whose encoder is trainable -- a frozen tower's embeddings do not
            # require grad); _AllGather.backward sums the partials, so each rank's encoder grads land as
            # exact disjoint-band contributions
            reps, grads = [], []
            for emb, leaf in ((embs_img_b, img), (embs_txt_b, txt)):
                if emb.requires_grad:
                    reps.append(emb)
                    grads.append(leaf.grad)
            if reps:
                torch.autograd.backward(reps, grads)

        if self.world_size > 1:
            # disjoint per-rank contributions (encoder grads by sample shard, logit scale/bias by band)
            # sum to the full-batch gradient -- plain sum, no /world_size. NCCL matches collectives by
            # issue order, so the p.grad-is-None pattern must be rank-identical: true while participation
            # is config-driven (freeze flags), broken (silent hang/corrupt) by any data-dependent module skip
            for p in self.model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
            # the representation backward has consumed the leaves' band-partial grads; fold them so the
            # returned leaves carry full-batch dL/dembs for grad-norm logging (diagnostics-only, so
            # skipped when reporting.batch_diagnostics.emb_logit_grads is off)
            if diag["emb_logit_grads"]:
                dist.all_reduce(img.grad)
                dist.all_reduce(txt.grad)

        return loss, loss_raw, img, txt, None, class_encs_b, batch_stats, grad_sum_sim

    def batch_step_local(
        self,
        imgs_sb: torch.Tensor,
        txts_sb: Tuple[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eval-oriented forward pass that keeps the batch local to the current rank.
        Returns normalized image and text embeddings for the local sub-batch.
        """
        toks_sb = self.txt_pp(txts_sb)
        output = self.model(imgs_sb, toks_sb)

        embs_img_sb = F.normalize(output[0], dim=1)
        embs_txt_sb = F.normalize(output[1], dim=1)

        return embs_img_sb, embs_txt_sb

    def eval_loss_chunked(
        self,
        embs_img: torch.Tensor,
        embs_txt: torch.Tensor,
        class_encs: torch.Tensor,
        targ_data: List[Any],
        chunk_size_loss: int,
    ) -> Tuple[Optional[float], List[Dict[str, float]]]:
        """
        Raw eval loss over precomputed paired (image, text) embeddings, using
        fixed-size chunks as the contrastive negative pool so the difficulty
        matches the global training batch (rather than a single rank's local
        sub-batch). The final partial chunk is dropped so every chunk poses an
        equal-size negative pool; returns (None, []) when there isn't one full chunk.

        Inputs are the full partition gathered across ranks, so this runs
        identically (and redundantly) on every rank with no further
        collectives.

        Args:
        - embs_img ---------- Gathered, normalized image embeddings for the full partition; pt[N, D]
        - embs_txt ---------- Gathered, normalized (paired) text embeddings for the full partition; pt[N, D]
        - class_encs -------- Gathered class encodings for the full partition; pt[N]
        - targ_data --------- Gathered target data for the full partition (len N)
        - chunk_size_loss --- Contrastive negative-pool size per chunk; set to the global batch size
                              (per-rank sub-batch x world_size) so the eval loss is apples-to-apples
                              with train batch loss

        Returns:
        - Mean raw eval loss across full chunks, or None if N < chunk_size_loss
        - Per-chunk sim/targ distribution stats (sim_targ_batch_stats per full chunk;
          the dropped partial chunk contributes none), or [] if N < chunk_size_loss
        """
        N = embs_img.size(0)
        if N < chunk_size_loss:
            return None, []

        # deterministic shuffle (identical on every rank) so chunks are class-mixed
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)).to(embs_img.device)
        embs_img = embs_img[perm]
        embs_txt = embs_txt[perm]
        class_encs = class_encs[perm]
        targ_data = [targ_data[i] for i in perm.tolist()]

        loss_total = 0.0
        chunk_stats = []
        kappas = self.cfg.reporting["learning_curves"]["hpsm"]["kappas"]
        for i in range(0, N - chunk_size_loss + 1, chunk_size_loss):
            sl = slice(i, i + chunk_size_loss)
            _, loss_raw, _, sims, targs, _ = self._loss_full_batch(embs_img[sl], embs_txt[sl], class_encs[sl], targ_data[sl])
            chunk_stats.append(sim_targ_batch_stats(sims[0], targs, kappas))
            loss_total += loss_raw.item()

        return loss_total / len(chunk_stats), chunk_stats

class CLIPWrapper(VLMWrapper):
    def __init__(self, config: TrainConfig) -> None:
        model_name, pretrained, quick_gelu = CLIP_MODELS[config.arch["model_type"]]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_inf.transforms[1].size[0]

    def freeze_text_encoder(self) -> None:
        """
        Freezes CLIP text encoder parameters.
        """
        for name, param in self.model.named_parameters():
            if (
                name.startswith("token_embedding.")
                or name == "positional_embedding"
                or name.startswith("transformer.")
                or name.startswith("ln_final.")
                or name == "text_projection"
            ):
                param.requires_grad = False

    def disable_causal_mask_text(self) -> None:
        """
        Converts CLIP text encoder's causal attention mask to non-causal.
        """
        self._unwrapped_model.attn_mask.zero_()  # convert causal attention mask to non-causal

class SigLIPWrapper(VLMWrapper):
    def __init__(self, config: TrainConfig) -> None:
        model_name, pretrained, quick_gelu = SIGLIP_MODELS[config.arch["model_type"]]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_inf.transforms[0].size[0]

    def freeze_text_encoder(self) -> None:
        """
        Freezes SigLIP text encoder parameters.
        """
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False
