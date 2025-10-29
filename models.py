import torch  # type: ignore[import]
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu  # type: ignore[import]
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
import open_clip  # type: ignore[import]
import abc
from types import SimpleNamespace

from utils import paths
from utils_loss import compute_loss

import pdb


CLIP_MODELS = {
    "bioclip":               ("hf-hub:imageomics/bioclip",   None,         False),
    "bioclip2":              ("hf-hub:imageomics/bioclip-2", None,         False),  # note: bioclip2 quick_gelu=False different from quick_gelu=None, unlike bioclip and all the others (which are the same)
    "clip_vitb16":           ("ViT-B-16",                    "openai",     True),  # Note: there are models like ('ViT-B-32-quickgelu', 'openai'), which suggests that model names not styled as "*-quickgelu" weren't trained with quickgelu ~ look into this
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

class VLMWrapper(abc.ABC):
    """
    Base class for all vision-language model wrappers. Dispatches to an appropriate subclass based on model_type and
    centralizes common initialization logic (e.g. device, tokenizer, checkpoint, etc.)
    """
    def __init__(self, config, model_name, pretrained, quick_gelu):
        """
        maybe add logic supporting quick_gelu = None and evaluate BioCLIP 2 once more, also enables systematically 
        checking through model types once the experimental infrastructure is more fleshed out

        Args:
        - config ------- [TrainConfig or EvaLConfig] --- configuration object
        - model_name --- [str] ------------------------- open_clip model identifier
        - pretrained --- [str] ------------------------- 
        - quick_gelu --- [bool] ------------------------ 
        """

        self.cfg = config

        model, img_pp_train, img_pp_val = open_clip.create_model_and_transforms(
            model_name, 
            pretrained      =pretrained, 
            force_quick_gelu=quick_gelu,
            cache_dir       =paths["hf_cache"],
        )

        self.device       = config.device
        self.type         = config.model_type
        self.model        = model.to(self.device).eval()
        self.img_pp_train = img_pp_train
        self.img_pp_val   = img_pp_val

        tokenizer   = open_clip.get_tokenizer(model_name)
        self.txt_pp = lambda txts: tokenizer(txts).to(self.device)

        if config.act_chkpt:
            self.model.set_grad_checkpointing(True)

        if config.loss_type in ("mse", "huber"):
            if hasattr(self.model, "regr_temp"):
                raise ValueError("regr_temp already exists!")
            if hasattr(self.model, "regr_bias"):
                raise ValueError("regr_bias already exists!")
            cfg_loss = config.cfg_loss
            if cfg_loss["regression"]["temp"]:
                self.model.regr_temp = nn.Parameter(torch.tensor(1.0))
            else:
                self.model.regr_temp = 1.0
            if cfg_loss["regression"]["bias"]:
                self.model.regr_bias = nn.Parameter(torch.tensor(0.0))
            else:
                self.model.regr_bias = 0.0
            if config.loss_type == "huber":
                cfg_regr        = getattr(config, "regression", SimpleNamespace())
                self.huber_beta = getattr(cfg_regr, "huber_beta", 0.1)

    def set_class_wts(self, class_wts, class_pair_wts):
        self.class_wts      = class_wts.to(self.device)
        self.class_pair_wts = class_pair_wts.to(self.device)

    def set_targ_type(self, targ_type):
        self.targ_type = targ_type

    @classmethod
    def build(cls, config):

        model_state_dict = None
        if config.has_field("rdpath_trial") and config.rdpath_trial is not None:
            print(f"Loading '{config.rdpath_trial}' ({config.save_crit})...")
            fpath_model_state_dict = paths["root"] / config.rdpath_trial / f"models/best_{config.save_crit}/state_dict_model.pt"
            model_state_dict       = torch.load(
                fpath_model_state_dict, 
                # map_location=device,
                map_location="cpu",  # map_location = "cpu" avoids loading two copies of the entire state dict into VRAM at once
            )
        else:
            print("Loading fresh model...")

        if config.model_type in CLIP_MODELS:
            modelw = CLIPWrapper(config)
        elif config.model_type in SIGLIP_MODELS:
            modelw = SigLIPWrapper(config)
        elif config.model_type in VITAMIN_MODELS:
            modelw = ViTaminWrapper(config)
        else:
            raise ValueError(f"Unknown model_type: '{config.model_type}'")

        modelw.loss_type = config.loss_type
        if model_state_dict is not None:
            modelw.model.load_state_dict(model_state_dict)

        if config.non_causal:
            modelw.disable_causal_mask_text()

        print("Model loaded!\n")

        return modelw

    @property
    def embed_dim(self):
        if isinstance(self, CLIPWrapper):
            return self.model.text_projection.weight.shape[0]
        if isinstance(self, SigLIPWrapper):
            return self.model.text.text_projection.weight.shape[0]
        if isinstance(self, ViTaminWrapper):
            return self.model.text.text_projection.shape[0]

    def save(self, dpath):
        fpath            = dpath / "state_dict_model.pt"
        state_dict_model = self.model.state_dict()
        torch.save(state_dict_model, fpath)

    def embed_images(self, imgs_b):
        """
        Runs batch of images through image encoder and returns batch of unit-length embeddings.

        Args:
        - imgs_b --- [Tensor(B, C, H, W)] --- Batch of images
        """

        with torch.set_grad_enabled(self.model.training):
            embs_imgs_b = self.model.encode_image(imgs_b)

        embs_imgs_b = F.normalize(embs_imgs_b, dim=1)

        return embs_imgs_b

    def embed_texts(self, txts):
        """
        Runs texts through text encoder and returns unit-length embeddings.

        Args:
        - txts --- [np.array(str)] --- Array of texts of length L

        Returns:
        - [Tensor(L, D)] ------------- Text embeddings
        """

        tokens_txts = self.txt_pp(txts)  # ------------- Tensor(L, T) ~ L for num. classes, T for num. tokens i.e. context length

        with torch.set_grad_enabled(self.model.training):
            embs_txts = self.model.encode_text(tokens_txts)

        embs_txts = F.normalize(embs_txts, dim=1)  # --- Tensor(L, D)

        return embs_txts

    def img2txt_classify(self, embs_imgs_b, embs_txts, classes_enc_txts):
        """

        Args:
        - embs_imgs_b -------- [Tensor(B, D)] --- Batch of image embeddings (D for dim. embeddings)
        - embs_txts ---------- [Tensor(L, D)] --- Text embeddings
        - classes_enc_txts --- [list(int)] ------ Text class encodings
        """

        logits = embs_imgs_b @ embs_txts.T
        probs  = logits.softmax(dim=-1)

        pred_idxs             = probs.argmax(dim=-1)
        scores                = probs[torch.arange(len(pred_idxs)), pred_idxs].tolist()
        pred_classes_enc_txts = [classes_enc_txts[i] for i in pred_idxs.tolist()]

        return pred_classes_enc_txts, scores

    def compute_logits(self, sim):
        if self.loss_type in ("infonce1", "infonce2"):
            return self.compute_logits_infonce(sim)
        elif self.loss_type == "sigmoid":
            return self.compute_logits_sigmoid(sim)
        elif self.loss_type in ("mse", "huber"):
            return self.compute_logits_regression(sim)

    def compute_logits_infonce(self, sim):
        logits = sim * self.model.logit_scale.exp()  # need to clamp this to 100? (probably need to do this yourself, see where temp param winds up for existing models)
        return logits

    def compute_logits_sigmoid(self, sim):
        logits = sim * self.model.logit_scale.exp() + self.model.logit_bias
        return logits

    def compute_logits_regression(self, sim):
        logits = sim * self.model.regr_temp + self.model.regr_bias
        return logits

    def compute_batch_loss(self, logits, class_encs_b, rank_keys_b):

        return compute_loss(
            self.targ_type, 
            self.loss_type, 
            logits, 
            class_encs_b, 
            rank_keys_b, 
            self.class_wts, 
            self.class_pair_wts, 
            self.device,
            self.cfg,
        )

    def freeze(self, freeze_txt, freeze_img):
        if freeze_txt: self.freeze_text_encoder()
        if freeze_img: self.freeze_image_encoder()

    def freeze_image_encoder(self):
        for name, param in self.model.named_parameters():
            if name.startswith("visual."):
                param.requires_grad = False

    # implemented in sub-classes
    @abc.abstractmethod
    def freeze_text_encoder(self):
        raise NotImplementedError

class CLIPWrapper(VLMWrapper):
    def __init__(self, config):
        model_name, pretrained, quick_gelu = CLIP_MODELS[config.model_type]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_val.transforms[1].size[0]

    def freeze_text_encoder(self):
        for name, param in self.model.named_parameters():
            if (
                name.startswith("token_embedding.") or
                name == "positional_embedding"      or
                name.startswith("transformer.")     or
                name.startswith("ln_final.")        or
                name == "text_projection"
            ):
                param.requires_grad = False

    def disable_causal_mask_text(self):
        self.model.attn_mask.zero_()  # convert causal attention mask to non-causal

class SigLIPWrapper(VLMWrapper):
    def __init__(self, config):
        model_name, pretrained, quick_gelu = SIGLIP_MODELS[config.model_type]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_val.transforms[0].size[0]

    def freeze_text_encoder(self):
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False

class ViTaminWrapper(VLMWrapper):
    def __init__(self, config):
        model_name, pretrained, quick_gelu = VITAMIN_MODELS[config.model_type]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.img_res = self.img_pp_val.transforms[1].size[0]

    def freeze_text_encoder(self):
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False

    def disable_causal_mask_text(self):
        self.model.text.attn_mask.zero_()  # convert causal attention mask to non-causal
