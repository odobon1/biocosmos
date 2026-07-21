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
from typing import List, Tuple, Any, Dict, Union, Optional
from pathlib import Path

from utils.utils import paths, load_split
from utils.loss import compute_loss
from utils.head import compute_sim
from utils.imb import compute_class_wts
from utils.data import make_image_preprocessor_inference, make_image_preprocessor_train
from utils.ddp import rank0
from utils.config import TrainConfig, EvalConfig


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

def sim_targ_batch_stats(sim: torch.Tensor, targs: torch.Tensor) -> Dict[str, float]:
    """
    Per-batch distribution stats over the full BxB similarity and (mix-blended) target
    matrices; drives the batch-level learning-curve strip, sim_targ.log, and the eval
    sim/targ sections (per-chunk, averaged across chunks).

    - sim ---- similarity matrix, already on [-1, 1] (cosine / geodesic-mapped)
    - targs -- target matrix on [0, 1]; rescaled to [-1, 1] so it shares sim's range

    Returns min/max/median/mean for each as flat keys. Reductions are stacked so the
    device->host transfer is a single .cpu() sync. The [0, 1] -> [-1, 1] rescale is applied
    to the four target scalars (affine, so it commutes with min/max/median/mean).
    """
    with torch.no_grad():
        s = sim.detach()
        t = targs.detach()
        reductions = [
            s.min(), s.max(), s.median(), s.mean(),
            t.min(), t.max(), t.median(), t.mean(),
        ]
        vals = torch.stack([r.float() for r in reductions]).cpu().tolist()
    sim_vals = vals[:4]
    targ_vals = [2.0 * v - 1.0 for v in vals[4:]]  # [0, 1] -> [-1, 1]
    return {
        "sim_min":     sim_vals[0],
        "sim_max":     sim_vals[1],
        "sim_median":  sim_vals[2],
        "sim_mean":    sim_vals[3],
        "targ_min":    targ_vals[0],
        "targ_max":    targ_vals[1],
        "targ_median": targ_vals[2],
        "targ_mean":   targ_vals[3],
    }

class VLMWrapper(abc.ABC):
    """
    Base class for all vision-language model wrappers. Dispatches to an appropriate subclass based on model_type and
    centralizes common initialization logic (e.g. device, tokenizer, checkpoint, etc.)
    """
    def __init__(
        self, 
        config: Union[TrainConfig, EvalConfig], 
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
        #   - vis_proj -> timm_proj: an ARCHITECTURE choice, applied for train AND eval so a checkpoint's
        #     projection head reloads with a matching module.
        #   - patch/head/stochastic-depth dropout: parameterless and train-only (EvalConfig carries no `dropout`,
        #     and eval runs in eval mode regardless).
        # Native CLIP takes only patch_dropout, via open_clip's force_patch_dropout.
        force_patch_dropout = None
        vision_cfg_extra = {}
        is_siglip = config.arch["model_type"] in SIGLIP_MODELS

        if is_siglip and config.arch["siglip"]["vis_proj"] is not None:
            vision_cfg_extra["timm_proj"] = config.arch["siglip"]["vis_proj"]

        if hasattr(config, "dropout"):
            dropout = config.dropout
            if is_siglip:
                vision_cfg_extra["patch_dropout"] = dropout["patch_dropout"]
                vision_cfg_extra["timm_drop"] = dropout["siglip"]["vis_proj"]
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
        self.type = config.arch['model_type']
        self.model = model.to(self.device).eval()
        self.img_pp_train = img_pp_train
        self.img_pp_inf = img_pp_inf

        tokenizer = open_clip.get_tokenizer(model_name)
        self.txt_pp = lambda txts: tokenizer(txts).to(self.device)

        if config.hw.act_chkpt:
            self.model.set_grad_checkpointing(True)

        if hasattr(config, 'loss'):
            cfg_logits = config.loss["logits"]
            if cfg_logits["scale_init"] is not None:  # scale_init set in config
                if hasattr(self.model, "logit_scale"):  # logit_scale attribute exists
                    with torch.no_grad():
                        self.model.logit_scale.fill_(cfg_logits["scale_init"])
            if cfg_logits["bias_init"] is None:  # (bias_init: null) in config
                if self.model.logit_bias is None:  # logit bias attribute is None (CLIP default)
                    delattr(self.model, "logit_bias")
                    self.model.register_buffer("logit_bias", torch.tensor(0.0, device=self.device))
            else:  # bias_init set in config
                if isinstance(self.model.logit_bias, nn.Parameter):  # logit_bias attribute is a nn.Parameter
                    with torch.no_grad():
                        self.model.logit_bias.fill_(cfg_logits["bias_init"])
                else:  # logit_bias attribute is not a nn.Parameter
                    delattr(self.model, "logit_bias")
                    self.model.register_parameter("logit_bias", nn.Parameter(torch.tensor(cfg_logits["bias_init"], device=self.device)))
            if cfg_logits["freeze_scale"] and isinstance(self.model.logit_scale, nn.Parameter):
                self.model.logit_scale.requires_grad_(False)
            if cfg_logits["freeze_bias"] and isinstance(self.model.logit_bias, nn.Parameter):
                self.model.logit_bias.requires_grad_(False)

        if hasattr(config, 'loss2') and config.loss2["mix"] != 0.0:
            cfg_logits2 = config.loss2["logits"]
            if cfg_logits2["scale_init"] is None:  # (scale_init: null) in config
                self.model.register_parameter("logit_scale2", nn.Parameter(torch.tensor(self.model.logit_scale.detach().item(), device=self.device)))
            else:  # scale_init set in config
                self.model.register_parameter("logit_scale2", nn.Parameter(torch.tensor(cfg_logits2["scale_init"], device=self.device)))
            if cfg_logits2["bias_init"] is None:
                self.model.register_parameter("logit_bias2", nn.Parameter(torch.tensor(self.model.logit_bias.detach().item(), device=self.device)))
            else:
                self.model.register_parameter("logit_bias2", nn.Parameter(torch.tensor(cfg_logits2["bias_init"], device=self.device)))
            if cfg_logits2["freeze_scale"]:
                self.model.logit_scale2.requires_grad_(False)
            if cfg_logits2["freeze_bias"]:
                self.model.logit_bias2.requires_grad_(False)

    @classmethod
    def build(cls, config: Union[TrainConfig, EvalConfig], verbose: bool) -> Any:
        """
        Factory method to construct the appropriate VLM wrapper based on configuration.
        Handles loading from checkpoint if `rdpath_model` is specified.

        Args:
        - config ---- Configuration object containing architecture and loss settings
        - verbose --- Whether to print loading status

        Returns:
        - Initialized VLMWrapper subclass instance
        """
        checkpoint = None
        if getattr(config, "rdpath_model", None) is not None:
            if verbose:
                print(f"Loading '{config.rdpath_model}'/model.pt...")
            fpath_model = paths["root"] / config.rdpath_model / "model.pt"
            checkpoint = torch.load(
                fpath_model, 
                map_location="cpu",  # map_location="cpu" avoids loading two copies of the entire state dict into VRAM at once
            )
        else:
            if verbose:
                print("Loading base model...")

        if config.arch['model_type'] in CLIP_MODELS:
            modelw = CLIPWrapper(config)
        elif config.arch['model_type'] in SIGLIP_MODELS:
            modelw = SigLIPWrapper(config)
        else:
            raise ValueError(f"Unknown model_type: '{config.arch['model_type']}'")

        if hasattr(config, 'loss'):
            modelw.loss_type = config.loss['type']
        if checkpoint is not None:
            modelw._unwrapped_model.load_state_dict(checkpoint["model"], strict=False)
            for key in ("logit_scale2", "logit_bias2"):
                if key in checkpoint["model"]:
                    modelw._unwrapped_model.register_parameter(key, nn.Parameter(checkpoint["model"][key]))
            modelw.norm_mean = checkpoint["norm_mean"]
            modelw.norm_std = checkpoint["norm_std"]
        else:
            modelw.set_image_norms(config)

        modelw.set_image_preprocessors()

        if config.arch['clip']['non_causal']:
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

    def set_image_norms(self, config: Union[TrainConfig, EvalConfig]) -> None:
        """
        Sets normalization mean and std for image preprocessors based on config (dataset-specific or default).
        """
        if config.img_norm == "default":
            self.norm_mean = self.img_pp_inf.transforms[-1].mean
            self.norm_std = self.img_pp_inf.transforms[-1].std
        elif config.img_norm == "dataset":
            split = load_split(config.dataset, config.split)
            self.norm_mean = split.norm_mean[config.train_pt]
            self.norm_std = split.norm_std[config.train_pt]

    def set_image_preprocessors(self) -> None:
        self.img_pp_inf = make_image_preprocessor_inference(self.img_res, norm_mean=self.norm_mean, norm_std=self.norm_std)
        if hasattr(self.cfg, "aug"):
            self.img_pp_train = make_image_preprocessor_train(
                self.img_res,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                aug_cfg=self.cfg.aug,
            )

    @rank0
    def save(self, dpath: Path) -> None:
        dpath.mkdir(parents=True, exist_ok=True)
        fpath = dpath / "model.pt"
        state_dict_model = self._unwrapped_model.state_dict()
        torch.save(
            {
                "model": state_dict_model,
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
            }, 
            fpath
        )

    def set_class_wts(self, config: TrainConfig, secondary: bool = False) -> None:
        cfg_loss = config.loss if not secondary else config.loss2
        cw, cpw = compute_class_wts(config.dataset, config.split, cfg_loss, train_pt=config.train_pt)
        if not secondary:
            self.class_wts = cw.to(self.device)
            self.class_pair_wts = cpw.to(self.device)
        else:
            self.class_wts2 = cw.to(self.device)
            self.class_pair_wts2 = cpw.to(self.device)

    def embed_images(self, imgs_b: torch.Tensor) -> torch.Tensor:
        """
        Runs batch of images through image encoder and returns batch of unit-length embeddings.

        Args:
        - imgs_b --- Batch of images; Tensor(B, C, H, W)

        Returns:
        - Normalized image embeddings; Tensor(B, D)
        """
        model = self._unwrapped_model
        
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

    def compute_logits(self, sim: torch.Tensor, secondary: bool = False) -> torch.Tensor:
        """
        Scales similarity matrix by learnable logit scale (temperature) and adds logit bias if applicable (e.g. SigLIP).
        """
        model = self._unwrapped_model
        if not secondary:
            return sim * model.logit_scale.exp() + model.logit_bias
        else:
            return sim * model.logit_scale2.exp() + model.logit_bias2

    def compute_batch_loss(
        self, 
        logits:       torch.Tensor, 
        class_encs_b: torch.Tensor, 
        targ_data_b:  List[Any], 
        cfg_loss:     Dict[str, Any], 
        secondary:    bool = False,
    ) -> torch.Tensor:
        """
        Computes loss for a batch given logits and target data.
        """
        cw, cpw = (self.class_wts, self.class_pair_wts) if not secondary else (self.class_wts2, self.class_pair_wts2)
        loss, loss_raw, targs = compute_loss(
            cfg_loss,
            logits,
            class_encs_b,
            targ_data_b,
            cw,
            cpw,
            self.device,
            train=self.model.training,
        )
        return loss, loss_raw, targs

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

    def _loss_for_cfg_full_batch(
        self,
        embs_img_all: torch.Tensor,
        embs_txt_all: torch.Tensor,
        class_encs_all: torch.Tensor,
        targ_data_all: List[Any],
        cfg_loss: Dict[str, Any],
        secondary: bool = False,
    ) -> torch.Tensor:
        """
        Computes loss for the full global batch according to a given loss configuration (primary or secondary).
        """
        sim = compute_sim(embs_img_all, embs_txt_all, cfg_loss["sim"])
        logits = self.compute_logits(sim, secondary=secondary)
        loss, loss_raw, targs = self.compute_batch_loss(logits, class_encs_all, targ_data_all, cfg_loss, secondary=secondary)

        return loss, loss_raw, logits, sim, targs

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

        if not loss_flag:
            return None, None, embs_img_b, embs_txt_b, (None, None), class_encs_b, None

        loss1, loss1_raw, logits1, sim1, targs1 = self._loss_for_cfg_full_batch(
            embs_img_b,
            embs_txt_b,
            class_encs_b,
            targ_data_b,
            self.cfg.loss,
            secondary=False,
        )
        if logits1.requires_grad:
            logits1.retain_grad()

        mix = self.cfg.loss2["mix"]
        if mix != 0.0:
            loss2, loss2_raw, logits2, sim2, targs2 = self._loss_for_cfg_full_batch(
                embs_img_b,
                embs_txt_b,
                class_encs_b,
                targ_data_b,
                self.cfg.loss2,
                secondary=True,
            )
            if logits2.requires_grad:
                logits2.retain_grad()  # for batch-level logging of logit-level gradient norm

            loss = (1.0 - mix) * loss1 + mix * loss2
            loss_raw = (1.0 - mix) * loss1_raw + mix * loss2_raw

            # similarity is shared across loss configs (same embeddings, same sim_type in practice);
            # the tracked target matrix is the mix-blend of the two configs' targets
            targs_stat = (1.0 - mix) * targs1 + mix * targs2
            batch_stats = sim_targ_batch_stats(sim1, targs_stat)

            return loss, loss_raw, embs_img_b, embs_txt_b, (logits1, logits2), class_encs_b, batch_stats

        batch_stats = sim_targ_batch_stats(sim1, targs1)
        return loss1, loss1_raw, embs_img_b, embs_txt_b, (logits1, None), class_encs_b, batch_stats

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
        - logits --------- (logits1, logits2); Logits computed for the batch; Tuple[pt[B, B], pt[B, B] | None]
        - class_encs_b --- Batch of class encodings; pt[B]
        - batch_stats ---- Per-batch sim/target distribution stats (flat dict), or None if loss_flag is False
        """
        toks_sb = self.txt_pp(txts_sb)
        output = self.model(imgs_sb, toks_sb)

        embs_img_sb = F.normalize(output[0], dim=1)
        embs_txt_sb = F.normalize(output[1], dim=1)

        if embs_img_sb.requires_grad:
            embs_img_sb.retain_grad()  # for batch-level logging of image embeddings gradient norm
        if embs_txt_sb.requires_grad:
            embs_txt_sb.retain_grad()  # for batch-level logging of text embeddings gradient norm

        loss, loss_raw, embs_img_b, embs_txt_b, logits, class_encs_b, batch_stats = self._global_batch_loss(
            embs_img_sb,
            embs_txt_sb,
            class_encs_sb,
            targ_data_sb,
            loss_flag=loss_flag,
        )

        return loss, loss_raw, embs_img_b, embs_txt_b, logits, class_encs_b, batch_stats

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
        n_samps = 0
        chunk_stats = []
        for i in range(0, N - chunk_size_loss + 1, chunk_size_loss):
            sl = slice(i, i + chunk_size_loss)
            _, loss_raw, _, sim1, targs1 = self._loss_for_cfg_full_batch(
                embs_img[sl], embs_txt[sl], class_encs[sl], targ_data[sl], self.cfg.loss, secondary=False,
            )
            mix = self.cfg.loss2["mix"]
            if mix != 0.0:
                _, loss2_raw, _, _, targs2 = self._loss_for_cfg_full_batch(
                    embs_img[sl], embs_txt[sl], class_encs[sl], targ_data[sl], self.cfg.loss2, secondary=True,
                )
                loss_raw = (1.0 - mix) * loss_raw + mix * loss2_raw
                # similarity is shared across loss configs (same embeddings, same sim_type in
                # practice) so stats track the primary config's sim only; the tracked target
                # matrix is the mix-blend of the two configs' targets (mirrors _global_batch_loss)
                targs_stat = (1.0 - mix) * targs1 + mix * targs2
            else:
                targs_stat = targs1
            chunk_stats.append(sim_targ_batch_stats(sim1, targs_stat))
            loss_total += loss_raw.item() * chunk_size_loss
            n_samps += chunk_size_loss

        return loss_total / n_samps, chunk_stats

class CLIPWrapper(VLMWrapper):
    def __init__(self, config: Union[TrainConfig, EvalConfig]) -> None:
        model_name, pretrained, quick_gelu = CLIP_MODELS[config.arch['model_type']]
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
    def __init__(self, config: Union[TrainConfig, EvalConfig]) -> None:
        model_name, pretrained, quick_gelu = SIGLIP_MODELS[config.arch['model_type']]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_inf.transforms[0].size[0]

    def freeze_text_encoder(self) -> None:
        """
        Freezes SigLIP text encoder parameters.
        """
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False
