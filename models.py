import torch
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import open_clip

from utils import paths

import pdb


OPEN_CLIP_TYPES = (
    "bioclip",
    "bioclip2",
    "openai_vitb32",
    "openai_vitb32qg",
    "openai_vitb16",
    "openai_vitl14",
    "openai_vitl14_336",
    "openai_rn50",
    "openai_rn101",
    "openai_rn101_yfcc15m",
    "openai_rn50x4",
    "openai_rn50x16",
    "openai_rn50x64",
)

class CLIPWrapper:

    def __init__(self, clip_type, device, run_name=None, chkpt_crit=None):

        self.device = device
        self.type   = clip_type

        # hugging face
        if clip_type == "openai_vitb32_hf":

            model_name   = "openai/clip-vit-base-patch32"
            preprocessor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
            self.model   = CLIPModel.from_pretrained(model_name)
            self.model   = self.model.to(self.device).eval()
            if run_name is not None:
                self.load_checkpoint(run_name, chkpt_crit)

            # per-sample image transform - runs in each DataLoader worker (when num_workers > 0) so I/O (disk read) and CPU work (resize, normalize, to-tensor) 
            # happen in parallel across multiple processes -- enables fully parallelized transforms, overlap with GPU compute, easy to cache in RAM if desired, 
            # at the expense of a little more boilerplate in Dataset and needing to manage transforms
            self.img_pp = lambda imgs: preprocessor.image_processor(
                    images=[imgs],
                    return_tensors="pt",
            )["pixel_values"][0]

            # batch text tokenization
            self.txt_pp = lambda txts: preprocessor.tokenizer(
                text=txts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

        # open_clip
        elif clip_type in OPEN_CLIP_TYPES:
            if clip_type == "bioclip":
                model_name, pretrained, quick_gelu = "hf-hub:imageomics/bioclip",   None,      False
            elif clip_type == "bioclip2":
                model_name, pretrained, quick_gelu = "hf-hub:imageomics/bioclip-2", None,      False  # note: bioclip2 quick_gelu=False different from no quick_gelu specified at all (unlike bioclip which is the same)
            elif clip_type == "openai_vitb32":
                model_name, pretrained, quick_gelu = "ViT-B-32",                    "openai",  True
            elif clip_type == "openai_vitb16":
                model_name, pretrained, quick_gelu = "ViT-B-16",                    "openai",  True
            elif clip_type == "openai_vitl14":
                model_name, pretrained, quick_gelu = "ViT-L-14",                    "openai",  True
            elif clip_type == "openai_vitl14_336":
                model_name, pretrained, quick_gelu = "ViT-L-14-336",                "openai",  True
            elif clip_type == "openai_rn50":
                model_name, pretrained, quick_gelu = "RN50",                        "openai",  True
            elif clip_type == "openai_rn101":
                model_name, pretrained, quick_gelu = "RN101",                       "openai",  True
            elif clip_type == "openai_rn101_yfcc15m":
                model_name, pretrained, quick_gelu = "RN101",                       "yfcc15m", True
            elif clip_type == "openai_rn50x4":
                model_name, pretrained, quick_gelu = "RN50x4",                      "openai",  True
            elif clip_type == "openai_rn50x16":
                model_name, pretrained, quick_gelu = "RN50x16",                     "openai",  True
            elif clip_type == "openai_rn50x64":
                model_name, pretrained, quick_gelu = "RN50x64",                     "openai",  True

            # self.model, _, self.img_pp = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.model, _, self.img_pp = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, quick_gelu=quick_gelu)
            self.model                 = self.model.to(self.device).eval()
            if run_name is not None:
                self.load_checkpoint(run_name, chkpt_crit)

            tokenizer   = open_clip.get_tokenizer(model_name)
            self.txt_pp = lambda txts: tokenizer(txts).to(self.device)

        else:

            raise ValueError(f"Invalid clip_type specified: '{clip_type}'")

    def load_checkpoint(self, run_name, chkpt_crit):

        assert chkpt_crit == "comp" or chkpt_crit == "img2img", "CLIPWrapper(): run_name specified but chkpt_crit != ('comp' or 'img2img')"
        
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
            if self.type == "openai_vitb32_hf":
                embs_imgs_b = self.model.get_image_features(pixel_values=imgs_b)
            elif self.type in OPEN_CLIP_TYPES:
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
            if self.type == "openai_vitb32_hf":
                embs_txts = self.model.get_text_features(**tokens_txts)
            elif self.type in OPEN_CLIP_TYPES:
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
