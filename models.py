import torch
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import open_clip

from utils import paths

import pdb


class VisionLanguageModelWrapper:

    def __init__(self, model_type, device, run_name=None, chkpt_crit=None):

        self.device = device
        self.type   = model_type

        # add emb dim var (if it doesn't already exist in model)

        self.criterion_clip = torch.nn.CrossEntropyLoss()

        if model_type == "bioclip":
            model_name, pretrained, quick_gelu = "hf-hub:imageomics/bioclip",   None,         False
        elif model_type == "bioclip2":
            model_name, pretrained, quick_gelu = "hf-hub:imageomics/bioclip-2", None,         False  # note: bioclip2 quick_gelu=False different from no quick_gelu specified at all (unlike bioclip which is the same)

        elif model_type == "clip_vitb16":
            model_name, pretrained, quick_gelu = "ViT-B-16",                    "openai",     True
        elif model_type == "clip_vitb32":
            model_name, pretrained, quick_gelu = "ViT-B-32",                    "openai",     True
        elif model_type == "clip_vitl14":
            model_name, pretrained, quick_gelu = "ViT-L-14",                    "openai",     True
        elif model_type == "clip_vitl14_336":
            model_name, pretrained, quick_gelu = "ViT-L-14-336",                "openai",     True
        
        elif model_type == "clip_rn50":
            model_name, pretrained, quick_gelu = "RN50",                        "openai",     True
        elif model_type == "clip_rn101":
            model_name, pretrained, quick_gelu = "RN101",                       "openai",     True
        elif model_type == "clip_rn101_yfcc15m":
            model_name, pretrained, quick_gelu = "RN101",                       "yfcc15m",    True
        elif model_type == "clip_rn50x4":
            model_name, pretrained, quick_gelu = "RN50x4",                      "openai",     True
        elif model_type == "clip_rn50x16":
            model_name, pretrained, quick_gelu = "RN50x16",                     "openai",     True
        elif model_type == "clip_rn50x64":
            model_name, pretrained, quick_gelu = "RN50x64",                     "openai",     True

        elif model_type == "siglip_vitb16":
            model_name, pretrained, quick_gelu = "ViT-B-16-SigLIP",             "webli",      False
        elif model_type == "siglip_vitb16_384":
            model_name, pretrained, quick_gelu = "ViT-B-16-SigLIP-384",         "webli",      False
        elif model_type == "siglip_vitl16_384":
            model_name, pretrained, quick_gelu = "ViT-L-16-SigLIP-384",         "webli",      False
        elif model_type == "siglip_vitso400m14":
            model_name, pretrained, quick_gelu = "ViT-SO400M-14-SigLIP",        "webli",      False
        elif model_type == "siglip2_vitb16":
            model_name, pretrained, quick_gelu = "ViT-B-16-SigLIP2",            "webli",      False
        elif model_type == "siglip2_vitb16_384":
            model_name, pretrained, quick_gelu = "ViT-B-16-SigLIP2-384",        "webli",      False
        elif model_type == "siglip2_vitl16_384":
            model_name, pretrained, quick_gelu = "ViT-L-16-SigLIP2-384",        "webli",      False
        elif model_type == "siglip2_vitso400m14":
            model_name, pretrained, quick_gelu = "ViT-SO400M-14-SigLIP2",       "webli",      False
        elif model_type == "siglip2_vitgopt16_384":
            model_name, pretrained, quick_gelu = "ViT-gopt-16-SigLIP2-384",     "webli",      False

        elif model_type == "vitamin_s":
            model_name, pretrained, quick_gelu = "ViTamin-S",                   "datacomp1b", False
        elif model_type == "vitamin_s_ltt":
            model_name, pretrained, quick_gelu = "ViTamin-S-LTT",               "datacomp1b", False
        elif model_type == "vitamin_b":
            model_name, pretrained, quick_gelu = "ViTamin-B",                   "datacomp1b", False
        elif model_type == "vitamin_b_ltt":
            model_name, pretrained, quick_gelu = "ViTamin-B-LTT",               "datacomp1b", False
        elif model_type == "vitamin_l":
            model_name, pretrained, quick_gelu = "ViTamin-L",                   "datacomp1b", False
        elif model_type == "vitamin_l_256":
            model_name, pretrained, quick_gelu = "ViTamin-L-256",               "datacomp1b", False
        elif model_type == "vitamin_l_336":
            model_name, pretrained, quick_gelu = "ViTamin-L-336",               "datacomp1b", False
        elif model_type == "vitamin_l_384":
            model_name, pretrained, quick_gelu = "ViTamin-L-384",               "datacomp1b", False
        elif model_type == "vitamin_l2":
            model_name, pretrained, quick_gelu = "ViTamin-L2",                  "datacomp1b", False
        elif model_type == "vitamin_l2_384":
            model_name, pretrained, quick_gelu = "ViTamin-L2-384",              "datacomp1b", False
        elif model_type == "vitamin_xl_384":
            model_name, pretrained, quick_gelu = "ViTamin-XL-384",              "datacomp1b", False

        # self.model, _, self.img_pp = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model, _, self.img_pp = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, quick_gelu=quick_gelu)
        self.model                 = self.model.to(self.device).eval()
        if run_name is not None:
            self.load_checkpoint(run_name, chkpt_crit)

        tokenizer   = open_clip.get_tokenizer(model_name)
        self.txt_pp = lambda txts: tokenizer(txts).to(self.device)

    def load_checkpoint(self, run_name, chkpt_crit):

        assert chkpt_crit == "comp" or chkpt_crit == "img2img", "VisionLanguageModelWrapper(): run_name specified but chkpt_crit != ('comp' or 'img2img')"
        
        dpath_model = paths["artifacts"] / run_name / "models" / f"best_{chkpt_crit}.pt"
        checkpoint = torch.load(
            dpath_model, 
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict)

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

    def compute_loss_clip(self, logits):
        B       = logits.size(0)
        targets = torch.arange(B, device=self.device, dtype=torch.long)

        loss_i2t_b = self.criterion_clip(logits, targets)
        loss_t2i_b = self.criterion_clip(logits.T, targets)
        loss_b     = 0.5 * (loss_i2t_b + loss_t2i_b)

        return loss_b
    
    def compute_logits_clip(self, sim):
        logits = sim * self.model.logit_scale.exp()
        return logits

    def compute_loss_siglip_aligned(self, logits):
        B       = logits.size(0)
        targets = torch.eye(B, device=self.device)

        loss_b  = F.binary_cross_entropy_with_logits(logits, targets)

        return loss_b

    def compute_logits_siglip(self, sim):
        logits = sim * self.model.logit_scale.exp() + self.model.logit_bias
        return logits
