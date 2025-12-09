import torch  # type: ignore[import]
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu  # type: ignore[import]
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import open_clip  # type: ignore[import]
import abc
from types import SimpleNamespace
from typing import List, Tuple, Any, Dict
from pathlib import Path

from utils import paths
from utils_loss import compute_loss
from utils_head import compute_sim
from utils_imb import compute_class_wts

import pdb


CLIP_MODELS = {
    "bioclip":               ("hf-hub:imageomics/bioclip",   None,         False),
    "bioclip2":              ("hf-hub:imageomics/bioclip-2", None,         False),
    "clip_vitb16":           ("ViT-B-16",                    "openai",     True),
    "clip_vitb32":           ("ViT-B-32",                    "openai",     True),
    "clip_vitl14":           ("ViT-L-14",                    "openai",     True),
    "clip_vitl14_336":       ("ViT-L-14-336",                "openai",     True),
    "clip_rn50":             ("RN50",                        "openai",     True),
    "clip_rn101":            ("RN101",                       "openai",     True),
    "clip_rn50x4":           ("RN50x4",                      "openai",     True),
    "clip_rn50x16":          ("RN50x16",                     "openai",     True),
    "clip_rn50x64":          ("RN50x64",                     "openai",     True),
}
SIGLIP_MODELS = {
    "siglip_vitb16":         ("ViT-B-16-SigLIP",             "webli",      False),
    "siglip_vitb16_384":     ("ViT-B-16-SigLIP-384",         "webli",      False),
    "siglip_vitl16_384":     ("ViT-L-16-SigLIP-384",         "webli",      False),
    "siglip_vitso400m14":    ("ViT-SO400M-14-SigLIP",        "webli",      False),
    "siglip2_vitb16":        ("ViT-B-16-SigLIP2",            "webli",      False),
    "siglip2_vitb16_384":    ("ViT-B-16-SigLIP2-384",        "webli",      False),
    "siglip2_vitl16_384":    ("ViT-L-16-SigLIP2-384",        "webli",      False),
    "siglip2_vitso400m14":   ("ViT-SO400M-14-SigLIP2",       "webli",      False),
    "siglip2_vitgopt16_384": ("ViT-gopt-16-SigLIP2-384",     "webli",      False),
}
VITAMIN_MODELS = {
    "vitamin_s":             ("ViTamin-S",                   "datacomp1b", False),
    "vitamin_s_ltt":         ("ViTamin-S-LTT",               "datacomp1b", False),
    "vitamin_b":             ("ViTamin-B",                   "datacomp1b", False),
    "vitamin_b_ltt":         ("ViTamin-B-LTT",               "datacomp1b", False),
    "vitamin_l":             ("ViTamin-L",                   "datacomp1b", False),
    "vitamin_l_256":         ("ViTamin-L-256",               "datacomp1b", False),
    "vitamin_l_336":         ("ViTamin-L-336",               "datacomp1b", False),
    "vitamin_l_384":         ("ViTamin-L-384",               "datacomp1b", False),
    "vitamin_l2":            ("ViTamin-L2",                  "datacomp1b", False),
    "vitamin_l2_384":        ("ViTamin-L2-384",              "datacomp1b", False),
    "vitamin_xl_384":        ("ViTamin-XL-384",              "datacomp1b", False),
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

class VLMWrapper(abc.ABC):
    """
    Base class for all vision-language model wrappers. Dispatches to an appropriate subclass based on model_type and
    centralizes common initialization logic (e.g. device, tokenizer, checkpoint, etc.)
    """
    def __init__(
        self, 
        config:     Any, 
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

        model, img_pp_train, img_pp_val = open_clip.create_model_and_transforms(
            model_name, 
            pretrained      =pretrained, 
            force_quick_gelu=quick_gelu,
            cache_dir       =paths["hf_cache"],
        )

        self.rank       = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.device       = config.device
        self.type         = config.arch['model_type']
        self.model        = model.to(self.device).eval()
        self.img_pp_train = img_pp_train
        self.img_pp_val   = img_pp_val

        tokenizer   = open_clip.get_tokenizer(model_name)
        self.txt_pp = lambda txts: tokenizer(txts).to(self.device)

        if config.hw.act_chkpt:
            self.model.set_grad_checkpointing(True)

        cfg_logits = config.loss["cfg"]["logits"]
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

        if config.loss2["mix"] != 0.0:
            cfg_logits2 = config.loss2["cfg"]["logits"]
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

        if config.loss['type'] == "huber":
            cfg_regr        = getattr(config, "regression", SimpleNamespace())
            self.huber_beta = getattr(cfg_regr, "huber_beta", 0.1)

        self.sim_type = config.loss["sim"]

    @classmethod
    def build(cls, config: Any, verbose: bool) -> Any:
        """
        Factory method to construct the appropriate VLM wrapper based on configuration.
        Handles loading from checkpoints if `rdpath_trial` is specified.

        Args:
        - config ---- Configuration object containing architecture and loss settings
        - verbose --- Whether to print loading status

        Returns:
        - Initialized VLMWrapper subclass instance
        """
        model_state_dict = None
        if config.has_field("rdpath_trial") and config.rdpath_trial is not None:
            if verbose: print(f"Loading '{config.rdpath_trial}' ({config.save_crit})...")
            fpath_state_dict_model = paths["root"] / config.rdpath_trial / f"models/best_{config.save_crit}/state_dict_model.pt"
            model_state_dict       = torch.load(
                fpath_state_dict_model, 
                map_location="cpu",  # map_location="cpu" avoids loading two copies of the entire state dict into VRAM at once
            )
        else:
            if verbose: print("Loading base model...")

        if config.arch['model_type'] in CLIP_MODELS:
            modelw = CLIPWrapper(config)
        elif config.arch['model_type'] in SIGLIP_MODELS:
            modelw = SigLIPWrapper(config)
        elif config.arch['model_type'] in VITAMIN_MODELS:
            modelw = ViTaminWrapper(config)
        else:
            raise ValueError(f"Unknown model_type: '{config.arch['model_type']}'")

        modelw.loss_type = config.loss['type']
        if model_state_dict is not None:
            modelw._unwrapped_model.load_state_dict(model_state_dict)

        if config.arch['non_causal']:
            modelw.disable_causal_mask_text()

        if verbose: print("Model loaded!\n")

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
        if isinstance(self, ViTaminWrapper):
            return model.text.text_projection.shape[0]

    def save(self, dpath: Path) -> None:
        fpath            = dpath / "state_dict_model.pt"
        state_dict_model = self._unwrapped_model.state_dict()
        torch.save(state_dict_model, fpath)

    def set_class_wts(self, config: Any, secondary: bool = False) -> None:
        cfg_loss = config.loss if not secondary else config.loss2
        cw, cpw  = compute_class_wts(config.split_name, cfg_loss)
        if not secondary:
            self.class_wts      = cw.to(self.device)
            self.class_pair_wts = cpw.to(self.device)
        else:
            self.class_wts2      = cw.to(self.device)
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

    def img2txt_classify(
        self, 
        embs_imgs_b:    torch.Tensor,  # --- Tensor(B, D)
        embs_txts:      torch.Tensor,  # --- Tensor(L, D)
        class_encs_txt: List[int],
    ) -> List[int]:
        """
        Perform image-to-text classification by computing similarities between image and text embeddings and selecting 
        the text class with the highest similarity for each image.

        Args:
        - embs_imgs_b ------ Batch of image embeddings (D for dim. embeddings)
        - embs_txts -------- Text embeddings
        - class_encs_txt --- Text class encodings

        Returns:
        - Predicted text class encodings for each image in the batch
        """

        sim       = compute_sim(embs_imgs_b, embs_txts, "cos")
        idxs_pred = sim.argmax(dim=-1)

        class_encs_txt_pred = idxs_pred.tolist()
        # class_encs_txt_pred = [class_encs_txt[i] for i in idxs_pred.tolist()]  # may want to use this if split-set indexing schema is globalized

        return class_encs_txt_pred

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
        loss = compute_loss(
            cfg_loss,
            logits, 
            class_encs_b,
            targ_data_b,
            cw, 
            cpw, 
            self.device,
        )
        return loss

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
        embs_img_all:   torch.Tensor,
        embs_txt_all:   torch.Tensor,
        class_encs_all: torch.Tensor,
        targ_data_all:  List[Any],
        cfg_loss:       Dict[str, Any],
        secondary:      bool = False,
    ) -> torch.Tensor:
        """
        Computes loss for the full global batch according to a given loss configuration (primary or secondary).
        """
        sim    = compute_sim(embs_img_all, embs_txt_all, cfg_loss["sim"])
        logits = self.compute_logits(sim, secondary=secondary)
        loss   = self.compute_batch_loss(logits, class_encs_all, targ_data_all, cfg_loss, secondary=secondary)

        return loss

    def _global_batch_loss(
        self,
        embs_img_b:   torch.Tensor,
        embs_txt_b:   torch.Tensor,
        class_encs_b: torch.Tensor,
        targ_data_b:  List[Any],
    ) -> torch.Tensor:
        """
        Compute loss using the full global batch:
        - gathers embeddings + targets across GPUs,
        - applies primary + secondary loss configs.
        Works for single GPU as well (gather is a no-op).
        """
        embs_img_all, embs_txt_all, class_encs_all, targ_data_all = self._gather_batch(
            embs_img_b, embs_txt_b, class_encs_b, targ_data_b
        )

        loss1 = self._loss_for_cfg_full_batch(
            embs_img_all,
            embs_txt_all,
            class_encs_all,
            targ_data_all,
            self.cfg.loss,
            secondary=False,
        )

        mix = self.cfg.loss2["mix"]
        if mix != 0.0:
            loss2 = self._loss_for_cfg_full_batch(
                embs_img_all,
                embs_txt_all,
                class_encs_all,
                targ_data_all,
                self.cfg.loss2,
                secondary=True,
            )
            return (1.0 - mix) * loss1 + mix * loss2

        return loss1

    def _gather_batch(
        self,
        embs_img_b:   torch.Tensor,
        embs_txt_b:   torch.Tensor,
        class_encs_b: torch.Tensor,
        targ_data_b:  List[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
        """
        Turn the local batch on this rank into a global batch across all ranks.
        Works for both single GPU (no-op) and DDP.
        """
        if self.world_size == 1:
            return embs_img_b, embs_txt_b, class_encs_b, targ_data_b  # single GPU no-op

        # Embeddings (need gradients)
        img_parts    = _AllGather.apply(embs_img_b)  # Tuple(G)
        txt_parts    = _AllGather.apply(embs_txt_b)
        embs_img_all = torch.cat(img_parts, dim=0)
        embs_txt_all = torch.cat(txt_parts, dim=0)

        # Class encodings (no grad)
        class_list = [torch.empty_like(class_encs_b) for _ in range(self.world_size)]
        dist.all_gather(class_list, class_encs_b)
        class_encs_all = torch.cat(class_list, dim=0)

        # Target metadata (Python objects)
        obj_list = [None] * self.world_size
        dist.all_gather_object(obj_list, targ_data_b)
        targ_data_all = []
        for td in obj_list:
            targ_data_all.extend(td)

        return embs_img_all, embs_txt_all, class_encs_all, targ_data_all

    def batch_step(
        self,
        imgs_b:       torch.Tensor,
        txts_b:       Tuple[str],
        class_encs_b: torch.Tensor,
        targ_data_b:  Tuple[Any],
        loss_flag:    bool = True,
    ) -> Tuple[Any]:
        """
        Performs a single forward pass step. Encodes images and text, computes loss if flag is set.

        Args:
        - imgs_b --------- Batch of images; Tensor(B, C, H, W)
        - txts_b --------- Batch of texts
        - class_encs_b --- Batch of class encodings; Tensor(B)
        - targ_data_b ---- Batch of target data
        - loss_flag ------ Whether to compute loss

        Returns:
        - loss ----------- Scalar loss (or None)
        - embs_img_b ----- Batch of normalized image embeddings; Tensor(B, D)
        """
        toks_b = self.txt_pp(txts_b)
        output = self.model(imgs_b, toks_b)

        embs_img_b = F.normalize(output[0], dim=1)
        embs_txt_b = F.normalize(output[1], dim=1)

        loss = None
        if loss_flag:
            loss = self._global_batch_loss(
                embs_img_b,
                embs_txt_b,
                class_encs_b,
                targ_data_b,
            )

        return loss, embs_img_b

class CLIPWrapper(VLMWrapper):
    def __init__(self, config: Any) -> None:
        model_name, pretrained, quick_gelu = CLIP_MODELS[config.arch['model_type']]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_val.transforms[1].size[0]

    def freeze_text_encoder(self) -> None:
        """
        Freezes CLIP text encoder parameters.
        """
        for name, param in self.model.named_parameters():
            if (
                name.startswith("token_embedding.") or
                name == "positional_embedding"      or
                name.startswith("transformer.")     or
                name.startswith("ln_final.")        or
                name == "text_projection"
            ):
                param.requires_grad = False

    def disable_causal_mask_text(self) -> None:
        """
        Converts CLIP text encoder's causal attention mask to non-causal.
        """
        self._unwrapped_model.attn_mask.zero_()  # convert causal attention mask to non-causal

class SigLIPWrapper(VLMWrapper):
    def __init__(self, config: Any) -> None:
        model_name, pretrained, quick_gelu = SIGLIP_MODELS[config.arch['model_type']]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_val.transforms[0].size[0]

    def freeze_text_encoder(self) -> None:
        """
        Freezes SigLIP text encoder parameters.
        """
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False

class ViTaminWrapper(VLMWrapper):
    def __init__(self, config: Any) -> None:
        model_name, pretrained, quick_gelu = VITAMIN_MODELS[config.arch['model_type']]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_val.transforms[1].size[0]

    def freeze_text_encoder(self) -> None:
        """
        Freezes ViTamin text encoder parameters.
        """
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False

    def disable_causal_mask_text(self) -> None:
        """
        Converts ViTamin text encoder's causal attention mask to non-causal.
        """
        self._unwrapped_model.text.attn_mask.zero_()