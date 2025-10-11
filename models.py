import torch  # type: ignore[import]
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu  # type: ignore[import]
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F  # type: ignore[import]
import open_clip  # type: ignore[import]
import abc

from utils import paths
from utils_pp import compute_rank_dists

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
    "clip_rn101_yfcc15m":    ("RN101",                       "yfcc15m",    True),
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

        model, img_pp_train, img_pp_val = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained, 
            quick_gelu=quick_gelu,
            cache_dir=paths["hf_cache"],
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

    def set_class_wts(self, class_wts, class_pair_wts):
        self.class_wts      = class_wts.to(self.device)
        self.class_pair_wts = class_pair_wts.to(self.device)

    def set_focal(self, cfg_focal):
        self.focal           = cfg_focal["enabled"]
        self.focal_gamma     = cfg_focal["gamma"]
        self.focal_alpha_pos = cfg_focal["alpha_pos"]
        self.focal_comp_type = cfg_focal["comp_type"]

    def set_sigmoid_targ_type(self, sig_targ_type):
        self.sig_targ_type = sig_targ_type

    def set_hierarchical(self, hierarchical):
        self.hierarchical = hierarchical

    @classmethod
    def build(cls, config):

        model_state_dict = None
        if config.has_field("rdpath_trial") and config.rdpath_trial is not None:
            print(f"Loading '{config.rdpath_trial}' ({config.save_crit})...")
            fpath_model_state_dict = paths["repo_o"] / config.rdpath_trial / f"models/best_{config.save_crit}/state_dict_model.pt"
            model_state_dict       = torch.load(
                fpath_model_state_dict, 
                # map_location=device,
                map_location="cpu",  # map_location = "cpu" avoids loading two copies of the entire state dict into VRAM at once
            )
        else:
            print("Loading fresh model...")

        if config.model_type in CLIP_MODELS:
            inst = CLIPWrapper(config)
        elif config.model_type in SIGLIP_MODELS:
            inst = SigLIPWrapper(config)
        elif config.model_type in VITAMIN_MODELS:
            inst = ViTaminWrapper(config)
        else:
            raise ValueError(f"Unknown model_type: '{config.model_type}'")

        inst.loss_type = config.loss_type
        if model_state_dict is not None:
            inst.model.load_state_dict(model_state_dict)

        print("Model loaded!\n")

        return inst

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
        if self.loss_type == "infonce":
            return self.compute_logits_clip(sim)
        elif self.loss_type == "sigmoid":
            return self.compute_logits_siglip(sim)
        else:
            raise ValueError(f"Unknown loss_type: '{self.loss_type}', must be one of {{infonce, sigmoid}}")

    def compute_logits_clip(self, sim):
        logits = sim * self.model.logit_scale.exp()
        return logits

    def compute_logits_siglip(self, sim):
        logits = sim * self.model.logit_scale.exp() + self.model.logit_bias
        return logits

    def compute_loss(self, logits, class_encs_b, rank_keys):
        class_encs_b = torch.tensor(class_encs_b).to(self.device)
        if self.loss_type == "infonce":
            return self.compute_loss_infonce(logits, class_encs_b, rank_keys)
        elif self.loss_type == "sigmoid":
            return self.compute_loss_sigmoid(logits, class_encs_b, rank_keys)

    def compute_loss_infonce(self, logits, class_encs_b, rank_keys):

        """
        Note: may need to be adjusted for multiple GPUs (wrt reduction)
        """

        B = logits.size(0)
        if self.hierarchical:
            rank_dists = compute_rank_dists(rank_keys)
            targs_raw  = 1 - 0.5 * rank_dists
            targs      = targs_raw / targs_raw.sum(dim=1, keepdim=True)
            targs      = targs.to(self.device)
        else:
            targs = torch.eye(B, device=self.device)  # --- Tensor(B, B)
        rw_wts = self.class_wts[class_encs_b]  # "re-weighting" weights

        loss_i2t_b = F.cross_entropy(logits,   targs,   reduction="none")
        loss_t2i_b = F.cross_entropy(logits.T, targs.T, reduction="none")

        if self.focal:
            """
            p_t = exp(-CE)
            focal factor: (1 - p_t)^gamma
            expm1 has better precision when p_t ~ 1 (CE ~ 0)
            expm1(x) = e^x - 1
            """
            g       = self.focal_gamma
            foc_i2t = (-torch.expm1(-loss_i2t_b)).clamp_min(1e-12).pow(g)
            foc_t2i = (-torch.expm1(-loss_t2i_b)).clamp_min(1e-12).pow(g)

            w_i2t = foc_i2t * rw_wts
            w_t2i = foc_t2i * rw_wts
        else:
            w_i2t = rw_wts
            w_t2i = rw_wts

        num_i2t = (w_i2t * loss_i2t_b).sum()
        num_t2i = (w_t2i * loss_t2i_b).sum()
        den_i2t = w_i2t.detach().sum().clamp_min(1e-12)
        den_t2i = w_t2i.detach().sum().clamp_min(1e-12)

        loss_batch = 0.5 * (num_i2t / den_i2t + num_t2i / den_t2i)

        return loss_batch

    def _sigmoid_focal_modulate(self, logits, targs):
        p = torch.sigmoid(logits)
        if self.focal_comp_type == 1:
            p_t = torch.where(targs > 0, p, 1 - p)
        elif self.focal_comp_type == 2:
            p_t = (1 - p) + targs * (2 * p - 1)
        elif self.focal_comp_type == 3:
            p_t = 1 - torch.abs(targs - p)
        else:
            raise ValueError("Focal loss comp_type must be one of: {1, 2, 3}")
        foc = (1 - p_t).pow(self.focal_gamma)
        return foc

    def compute_loss_sigmoid(self, logits, class_encs_b, rank_keys):
        """
        targ_type options: {pairwise, multipos}
        """

        if self.hierarchical:
            rank_dists = compute_rank_dists(rank_keys)
            targs      = 1 - 0.5 * rank_dists
            targs      = targs.to(self.device)
        else:
            if self.sig_targ_type == "pairwise":
                B     = logits.size(0)
                targs = torch.eye(B, device=self.device)  # --------------------------------------------------- Tensor(B, B)
            elif self.sig_targ_type == "multipos":
                targs = (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float().to(self.device)  # --- Tensor(B, B)
            else:
                raise ValueError("targ_type must be one of: {pairwise, multipos}")

        # batch class-pair weight matrix; advanced indexing used to extract submatrix as per class_enc indices (row/col selection)
        rw_wts = self.class_pair_wts[class_encs_b][:, class_encs_b].to(self.device)  # -------------------- Tensor(B, B); "re-weighting" weights

        if self.focal:
            foc_wts = self._sigmoid_focal_modulate(logits, targs)  # -------------------------------------- Tensor(B, B)
        else:
            foc_wts = torch.ones_like(targs)

        alpha_pos  = self.focal_alpha_pos
        posneg_wts = targs * alpha_pos + (1 - targs) * (1 - alpha_pos)  # continuous i.e. compatible with hierarchical

        W = rw_wts * posneg_wts * foc_wts

        loss_raw   = F.binary_cross_entropy_with_logits(logits, targs, reduction="none")  # --------------- Tensor(B, B); unweighted loss matrix
        loss_batch = (W * loss_raw).sum() / W.detach().sum()  # weighted mean loss -- the norm here is irrelevant with the subsequent loss norm (may be some numerical considerations here though, might even want to prenorm the individual terms)

        # used to render total batch loss the same regardless of reweighting (i.e. individual loss components are adjusted with reweighting, but the amount of "total learning" stays the same for apples-to-apples comparison with baselines)
        with torch.no_grad():
            norm = loss_raw.mean() / loss_batch
        loss_batch = norm * loss_batch

        return loss_batch

    def freeze(self, freeze_txt, freeze_img):
        if freeze_txt: self.freeze_text_encoder()
        if freeze_img: self.freeze_image_encoder()

    def freeze_image_encoder(self):
        for name, param in self.model.named_parameters():
            if name.startswith("visual."):
                param.requires_grad = False

    @abc.abstractmethod
    def freeze_text_encoder(self):
        raise NotImplementedError

class CLIPWrapper(VLMWrapper):
    def __init__(self, config):
        model_name, pretrained, quick_gelu = CLIP_MODELS[config.model_type]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.input_res = self.img_pp_val.transforms[1].size[0]

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

class SigLIPWrapper(VLMWrapper):
    def __init__(self, config):
        model_name, pretrained, quick_gelu = SIGLIP_MODELS[config.model_type]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.input_res = self.img_pp_val.transforms[0].size[0]

    def freeze_text_encoder(self):
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False

class ViTaminWrapper(VLMWrapper):
    def __init__(self, config):
        model_name, pretrained, quick_gelu = VITAMIN_MODELS[config.model_type]
        super().__init__(config, model_name, pretrained, quick_gelu)

        self.input_res = self.img_pp_val.transforms[1].size[0]

    def freeze_text_encoder(self):
        for name, param in self.model.named_parameters():
            if name.startswith("text."):
                param.requires_grad = False
