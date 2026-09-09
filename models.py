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
from utils.loss import Criterion, chunked_bce_loss_backward, HIST_BINS, pos_prevalence
from utils.head import compute_sim
from utils.data import make_image_preprocessor_inference, make_image_preprocessor_train, normalize_imgs_u8
from utils.config import TrainConfig

import pdb


def resolve_bias_init(cfg_loss, config, tag):
    """
    A loss's logits.bce.bias.init as a float (or None): pos_prevalence -> logit of the loss's expected
    weighted positive prevalence (utils.loss.pos_prevalence), so the initial sigmoid matches the target
    prior where the scaled sims average to zero.
    """
    init = cfg_loss["logits"]["bce"]["bias"]["init"]
    if init == "pos_prevalence":
        p = pos_prevalence(cfg_loss, config.dataset, config.split, config.train_pt, config.batch_size)
        init = math.log(p / (1 - p))
        if dist.get_rank() == 0:
            print(f"loss{tag} logit bias init from positive prevalence: p = {p:.4g} -> bias = {init:.4f}")
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

def sim_targ_batch_stats(sim: torch.Tensor, targs: torch.Tensor, idx: Optional[int] = None,
                         logits: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Per-batch distribution stats over the full BxB similarity and target matrices; drives
    the batch-level learning-curve strips, sim_targ.log, and the eval sim/targ sections
    (per-chunk, averaged across chunks).

    - sim ----- similarity matrix, already on [-1, 1] (cosine / geodesic-mapped)
    - targs --- target matrix on [0, 1]
    - idx ----- loss-branch index; suffixes the key prefixes (sim1_*/targ1_*) so the two
                branches' stats coexist in one flat dict. None (eval) keeps sim_*/targ_*.
    - logits -- the branch's logits (temp/bias applied to sim), or None to skip the p{tag}_hist
                entry. Pass them only for BCE-family branches, where sigmoid(logits) is the
                predicted pair probability -- on [0, 1] like the targets, so the P and Y strips
                are directly comparable. Under InfoNCE the row-softmax carries no such reading.

    Returns min/max/median/mean for sim/targ as flat keys (the batch logs read those), plus
    targ{tag}_hist and -- when logits are given -- p{tag}_hist: the targets and predicted
    probabilities as HIST_BINS fractions over [0, 1] (distributions, not point stats; the curve
    strips render them as heatmap columns). Reductions are stacked so the device->host transfer is
    a single .cpu() sync.
    """
    with torch.no_grad():
        s = sim.detach()
        t = targs.detach()
        reductions = [
            s.min(), s.max(), s.median(), s.mean(),
            t.min(), t.max(), t.median(), t.mean(),
        ]
        packed = torch.stack([r.float() for r in reductions])
        packed = torch.cat([packed, torch.histc(t.float(), bins=HIST_BINS, min=0.0, max=1.0) / t.numel()])
        if logits is not None:
            p = logits.detach().float().sigmoid()
            packed = torch.cat([packed, torch.histc(p, bins=HIST_BINS, min=0.0, max=1.0) / p.numel()])
        vals = packed.cpu().tolist()
    tag = "" if idx is None else str(idx)
    stats = {
        f"sim{tag}_min":     vals[0],
        f"sim{tag}_max":     vals[1],
        f"sim{tag}_median":  vals[2],
        f"sim{tag}_mean":    vals[3],
        f"targ{tag}_min":    vals[4],
        f"targ{tag}_max":    vals[5],
        f"targ{tag}_median": vals[6],
        f"targ{tag}_mean":   vals[7],
        f"targ{tag}_hist":   vals[8:8 + HIST_BINS],
    }
    if logits is not None:
        stats[f"p{tag}_hist"] = vals[8 + HIST_BINS:]
    return stats

def stat_logits(logits, cfg_loss):
    """The logits sim_targ_batch_stats should derive p{tag}_* from: the branch tuple's first entry
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

        if hasattr(config, "loss1"):
            cfg_logits = config.loss1["logits"]
            if cfg_logits["temp"]["init"] is not None:  # temp.init set in config
                if hasattr(self.model, "logit_scale"):  # logit_scale attribute exists
                    with torch.no_grad():
                        self.model.logit_scale.fill_(-math.log(cfg_logits["temp"]["init"]))  # tau -> log(1/tau)
            bias_init = resolve_bias_init(config.loss1, config, "1")
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
            if cfg_logits["temp"]["freeze"] and isinstance(self.model.logit_scale, nn.Parameter):
                self.model.logit_scale.requires_grad_(False)
            if cfg_logits["bce"]["bias"]["freeze"] and isinstance(self.model.logit_bias, nn.Parameter):
                self.model.logit_bias.requires_grad_(False)

        if hasattr(config, "loss") and config.loss["mix"] != 0.0:
            cfg_logits2 = config.loss2["logits"]
            if cfg_logits2["temp"]["init"] is None:  # (temp.init: null) in config
                self.model.register_parameter("logit_scale2", nn.Parameter(torch.tensor(self.model.logit_scale.detach().item(), device=self.device)))
            else:  # temp.init set in config
                self.model.register_parameter("logit_scale2", nn.Parameter(torch.tensor(-math.log(cfg_logits2["temp"]["init"]), device=self.device)))  # tau -> log(1/tau)
            bias_init2 = resolve_bias_init(config.loss2, config, "2")
            if bias_init2 is None:
                self.model.register_parameter("logit_bias2", nn.Parameter(torch.tensor(self.model.logit_bias.detach().item(), device=self.device)))
            else:
                self.model.register_parameter("logit_bias2", nn.Parameter(torch.tensor(bias_init2, device=self.device)))
            if cfg_logits2["temp"]["freeze"]:
                self.model.logit_scale2.requires_grad_(False)
            if cfg_logits2["bce"]["bias"]["freeze"]:
                self.model.logit_bias2.requires_grad_(False)

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
        if hasattr(self.cfg, "aug"):
            self.img_pp_train = make_image_preprocessor_train(
                self.img_res,
                aug_cfg=self.cfg.aug,
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
        secondary: bool = False,
        center_global: Optional[torch.Tensor] = None,
        half_live: bool = False
    ) -> torch.Tensor:
        """
        Scales similarity matrix by exp(learnable logit scale) (1 / tau) and adds logit bias if applicable (BCE).

        `half_live` (bifurcated branches): uses 0.5*p + 0.5*p.detach() for the logit scale/bias, so
        each of the two un-halved branch calls contributes exactly half their grad -- the branch sum
        matches the non-bifurcated 1x (the towers, living in one branch each, already get 1x).
        Values are unchanged.

        `clamp_scale` caps the logit scale at ln(100) before exp() (scale multiplier <= 100, CLIP's stability
        cap); otherwise exp() is unbounded and can overflow to +inf and amplify the bf16 quantization of sim.

        `center` (the criterion's logits.bce.center) makes dL/dsim zero-sum:
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
        if not secondary:
            logit_scale, logit_bias = model.logit_scale, model.logit_bias
        else:
            logit_scale, logit_bias = model.logit_scale2, model.logit_bias2
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

    def _loss_for_crit_full_batch(
        self,
        embs_img_all: torch.Tensor,
        embs_txt_all: torch.Tensor,
        class_encs_all: torch.Tensor,
        targ_data_all: List[Any],
        crit: Criterion,
        secondary: bool = False,
    ) -> torch.Tensor:
        """
        Computes loss for the full global batch under a given criterion (primary or secondary).

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
        """
        model = self._unwrapped_model
        logit_scale = model.logit_scale2 if secondary else model.logit_scale
        clamp = crit.cfg["logits"]["temp"]["clamp"]
        center = crit.cfg["logits"]["bce"]["center"]

        if crit.bifurcated:
            sims = (
                compute_sim(embs_img_all, embs_txt_all.detach(), crit.cfg["sim"]),
                compute_sim(embs_img_all.detach(), embs_txt_all, crit.cfg["sim"]),
            )
            logits = tuple(self.compute_logits(sim, clamp, center, secondary=secondary, half_live=True) for sim in sims)
            crit_logits = logits
        else:
            sims = (compute_sim(embs_img_all, embs_txt_all, crit.cfg["sim"]),)
            logits = (self.compute_logits(sims[0], clamp, center, secondary=secondary),)
            crit_logits = logits[0]

        loss, loss_raw, targs = crit(crit_logits, class_encs_all, targ_data_all, self.model.training, logit_scale)

        return loss, loss_raw, logits, sims, targs

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
        - applies primary + secondary loss configs.
        Works for single GPU as well (gather is a no-op).
        """
        embs_img_b, embs_txt_b, class_encs_b, targ_data_b = self._gather_batch(
            embs_img_sb, embs_txt_sb, class_encs_sb, targ_data_sb
        )
        # the gathered embeddings are what batch_step returns to the grad-norm logger; retain so
        # .grad carries the full-batch dL/dembs after backward (same quantity the chunked path
        # all-reduces into its returned leaves). Each dev.batch_diagnostics component gates only its
        # own retains/stats (emb_logit_grads the embedding+logit retains, sim_grad_sums the sim
        # retains, sim_targ_stats the batch stats) -- the loss/gradient path is untouched
        diag = self.cfg.dev["batch_diagnostics"]
        if diag["emb_logit_grads"]:
            if embs_img_b.requires_grad:
                embs_img_b.retain_grad()
            if embs_txt_b.requires_grad:
                embs_txt_b.retain_grad()

        if not loss_flag:
            return None, None, embs_img_b, embs_txt_b, (None, None), class_encs_b, None, (None, None)

        loss1, loss1_raw, logits1, sims1, targs1 = self._loss_for_crit_full_batch(
            embs_img_b,
            embs_txt_b,
            class_encs_b,
            targ_data_b,
            self.crit1,
            secondary=False,
        )
        # retain every branch's grad for the aggregate (branch-summed) grad logging: logits for the
        # logit= log field, sims for the sim-grad-sum curve strips
        for t in (*(logits1 if diag["emb_logit_grads"] else ()), *(sims1 if diag["sim_grad_sums"] else ())):
            if t.requires_grad:
                t.retain_grad()

        mix = self.cfg.loss["mix"]
        unitless = self.cfg.loss["unitless"]
        if unitless:
            # rescale each loss by its detached magnitude (loss / loss.detach()). Under a blend this
            # equalizes the two losses' magnitudes so `mix` controls their true gradient-contribution
            # ratio: Adam cancels a global loss scale but NOT the relative scale between the blended
            # losses, which can differ by orders of magnitude and drift over training. A lone loss just
            # reads a constant 1.0. A bifurcated loss reads 2x its gradient scale (un-halved branch sum,
            # 1x grads), so its normalizer is L/2 -- the gradient-scale-equivalent value -- keeping the
            # ratio true across bif/non-bif blends (a lone bif loss reads 2.0).
            loss1 = loss1 / (loss1.detach() / (2.0 if self.crit1.bifurcated else 1.0)).clamp_min(1e-12)
        if mix != 0.0:
            loss2, loss2_raw, logits2, sims2, targs2 = self._loss_for_crit_full_batch(
                embs_img_b,
                embs_txt_b,
                class_encs_b,
                targ_data_b,
                self.crit2,
                secondary=True,
            )
            for t in (*(logits2 if diag["emb_logit_grads"] else ()), *(sims2 if diag["sim_grad_sums"] else ())):
                if t.requires_grad:
                    t.retain_grad()

            if unitless:
                loss2 = loss2 / (loss2.detach() / (2.0 if self.crit2.bifurcated else 1.0)).clamp_min(1e-12)

            loss = (1.0 - mix) * loss1 + mix * loss2
            loss_raw = (1.0 - mix) * loss1_raw + mix * loss2_raw

            # each branch's sim/target matrices are tracked separately (learning-curve strips split
            # the two losses' stats); sims{1,2}[0]/logits{1,2}[0]: branch values are identical, so
            # the first branch carries the stats
            batch_stats = {
                **sim_targ_batch_stats(sims1[0], targs1, idx=1, logits=stat_logits(logits1, self.cfg.loss1)),
                **sim_targ_batch_stats(sims2[0], targs2, idx=2, logits=stat_logits(logits2, self.cfg.loss2)),
            } if diag["sim_targ_stats"] else None

            return loss, loss_raw, embs_img_b, embs_txt_b, (logits1, logits2), class_encs_b, batch_stats, (sims1, sims2)

        # sims1[0]: branch values are identical, so the first branch carries the sim stats
        batch_stats = sim_targ_batch_stats(sims1[0], targs1, idx=1, logits=stat_logits(logits1, self.cfg.loss1)) if diag["sim_targ_stats"] else None
        return loss1, loss1_raw, embs_img_b, embs_txt_b, (logits1, None), class_encs_b, batch_stats, (sims1, None)

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
        - logits --------- (logits1, logits2); each a per-loss branch tuple of [img-row, txt-col]
                           logit matrices ((logits,) non-bifurcated, (i2t, t2i) bifurcated -- see
                           _loss_for_crit_full_batch), grads retained for the logit-grad-norm log;
                           logits2 None at mix 0
        - class_encs_b --- Batch of class encodings; pt[B]
        - batch_stats ---- Per-batch sim/target distribution stats (flat dict), or None if loss_flag is False
        - sims ----------- (sims1, sims2); branch tuples mirroring `logits`, grads retained for
                           the sim-grad-sum strip
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
        hardware.loss_chunk_size), incl. a BCE-family loss2 mix, used in place of batch_step +
        loss.backward() when chunking is on. Does the
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

        Returns batch_step's tuple shape, with loss/loss_raw detached, logits = (None, None) (no full
        logit matrix is formed, so its grad-norm diagnostic is unavailable), the embedding leaves
        (carrying full-batch dL/dembs in .grad after a post-backward all-reduce) in place of
        embs_img_b / embs_txt_b for grad-norm logging, and -- since the backward already ran and the
        sim matrices are gone -- the sims slot carries the (grad_sum_sim1, grad_sum_sim2) floats
        accumulated tile-by-tile by chunked_bce_loss_backward. With dev.batch_diagnostics.sim_targ_stats
        off batch_stats is None; with .sim_grad_sums off the sims slot carries (None, None).
        """
        chunk = self.cfg.hw.loss_chunk_size
        mixed_prec = self.cfg.hw.mixed_prec
        device = self.cfg.device
        diag = self.cfg.dev["batch_diagnostics"]

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
            loss, loss_raw, batch_stats, grad_sum_sims = chunked_bce_loss_backward(
                img, txt, class_encs_b, targ_data_b, self.crit1, self.crit2, self.cfg.loss["mix"],
                self.cfg.loss["unitless"], self.compute_logits, chunk, mixed_prec, device,
                rank, self.world_size, sim_grad_sums=diag["sim_grad_sums"],
                sim_targ_stats=diag["sim_targ_stats"]
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
            # skipped when dev.batch_diagnostics.emb_logit_grads is off)
            if diag["emb_logit_grads"]:
                dist.all_reduce(img.grad)
                dist.all_reduce(txt.grad)

        return loss, loss_raw, img, txt, (None, None), class_encs_b, batch_stats, grad_sum_sims

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
        for i in range(0, N - chunk_size_loss + 1, chunk_size_loss):
            sl = slice(i, i + chunk_size_loss)
            _, loss_raw, _, sims1, targs1 = self._loss_for_crit_full_batch(
                embs_img[sl], embs_txt[sl], class_encs[sl], targ_data[sl], self.crit1, secondary=False,
            )
            mix = self.cfg.loss["mix"]
            if mix != 0.0:
                _, loss2_raw, _, _, targs2 = self._loss_for_crit_full_batch(
                    embs_img[sl], embs_txt[sl], class_encs[sl], targ_data[sl], self.crit2, secondary=True,
                )
                loss_raw = (1.0 - mix) * loss_raw + mix * loss2_raw
                # the eval sim/targ section stays a single summary: the primary config's sim and
                # the mix-blend of the two configs' targets (unlike the train learning curves,
                # which track the two branches' stats separately)
                targs_stat = (1.0 - mix) * targs1 + mix * targs2
            else:
                targs_stat = targs1
            chunk_stats.append(sim_targ_batch_stats(sims1[0], targs_stat))
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
