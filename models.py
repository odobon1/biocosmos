import torch
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import open_clip

import pdb


class CLIPWrapper:

    def __init__(self, clip_type, device):

        self.device = device
        self.type   = clip_type

        if clip_type == "openai":

            model_name   = "openai/clip-vit-base-patch32"
            preprocessor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
            self.clip    = CLIPModel.from_pretrained(model_name).to(device)
            self.clip.eval()

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
            ).to(device)

        elif clip_type == "bioclip":

            self.clip, _, self.img_pp = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            self.clip.to(device)
            self.clip.eval()

            tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

            # batch text tokenization
            self.txt_pp = lambda txts: tokenizer(txts).to(device)

        else:

            raise ValueError(f"'{clip_type}' specified for clip_type, must be 'openai' or 'bioclip'")

    def embed_images(self, imgs_b):
        """
        Args:
        - imgs_b --- [Tensor(B, C, H, W)] --- Batch of images
        """

        imgs_b = imgs_b.to(self.device)
        with torch.no_grad():
            if self.type == "openai":
                embs_imgs_b = self.clip.get_image_features(pixel_values=imgs_b)
            elif self.type == "bioclip":
                embs_imgs_b = self.clip.encode_image(imgs_b)

        return embs_imgs_b

    def embed_texts(self, txts_b):
        """
        Args:
        - txts_b --- [list(str)] --- Batch of texts
        """

        tokens_txts_b = self.txt_pp(txts_b)  # --------- Tensor(L, T) ~ L for num. classes, T for num. tokens i.e. context length

        with torch.no_grad():
            if self.type == "openai":
                embs_txts_b = self.clip.get_text_features(**tokens_txts_b)
            elif self.type == "bioclip":
                embs_txts_b = self.clip.encode_text(tokens_txts_b)

        return embs_txts_b

    def img2txt_classify(self, embs_imgs_b, embs_txts, classes_enc_txts):
        """
        Args:
        - embs_imgs_b -------- [Tensor(B, D)] --- Batch of image embeddings (D for dim. embeddings)
        - embs_txts ---------- [Tensor(L, D)] --- Text embeddings
        - classes_enc_txts --- [list(int)] ------ Text class encodings
        """

        # cosine similarity + softmax
        embs_imgs = F.normalize(embs_imgs_b, dim=-1)  # --- Tensor(B, D) ~ D for dim. embeddings
        embs_txts = F.normalize(embs_txts, dim=-1)  # ----- Tensor(L, D)
        logits    = embs_imgs @ embs_txts.T
        probs     = logits.softmax(dim=-1)

        pred_idxs             = probs.argmax(dim=-1)
        scores                = probs[torch.arange(len(pred_idxs)), pred_idxs].tolist()
        pred_classes_enc_txts = [classes_enc_txts[i] for i in pred_idxs.tolist()]

        return pred_classes_enc_txts, scores
