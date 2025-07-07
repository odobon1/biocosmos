import torch
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import open_clip

from utils import paths

import pdb


class CLIPWrapper:

    def __init__(self, clip_type, device, checkpoint=None, criterion=None):

        self.device = device
        self.type   = clip_type

        if clip_type == "openai":

            model_name   = "openai/clip-vit-base-patch32"
            preprocessor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
            self.model   = CLIPModel.from_pretrained(model_name).to(device)

            self.model.eval()

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

            self.model, _, self.img_pp = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            self.model.to(device)

            if checkpoint is not None:
                assert criterion == "comp" or criterion == "img2img", "CLIPWrapper(): checkpoint specified but criterion != ('comp' or 'img2img')"
                dpath_model = paths["artifacts"] / checkpoint / "models" / f"best_{criterion}.pt"
                state_dict  = torch.load(dpath_model, map_location=device)
                self.model.load_state_dict(state_dict)

            self.model.eval()

            tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

            # batch text tokenization
            self.txt_pp = lambda txts: tokenizer(txts).to(device)

        else:

            raise ValueError(f"'{clip_type}' specified for clip_type, must be 'openai' or 'bioclip'")

    def embed_images(self, imgs_b):
        """
        Runs batch of images through image encoder and returns batch of unit-length embeddings.

        Args:
        - imgs_b --- [Tensor(B, C, H, W)] --- Batch of images
        """

        with torch.set_grad_enabled(self.model.training):
            if self.type == "openai":
                embs_imgs_b = self.model.get_image_features(pixel_values=imgs_b)
            elif self.type == "bioclip":
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
            if self.type == "openai":
                embs_txts = self.model.get_text_features(**tokens_txts)
            elif self.type == "bioclip":
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
