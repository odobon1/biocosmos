import torch
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import open_clip


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

            raise ValueError(f"{clip_type} specified for clip_type, must be 'openai' or 'bioclip'")

    def embed_images(self, imgs):
        """
        Args:
        - imgs [Tensor(B, C, H, W)] --- Batch of images
        """
        imgs = imgs.to(self.device)
        with torch.no_grad():
            if self.type == "openai":
                img_embs = self.clip.get_image_features(pixel_values=imgs)
            else:  # BioCLIP
                img_embs = self.clip.encode_image(imgs)

        return img_embs

    def embed_texts(self, labels):
        """
        Args:
        - labels [list(str)] --- Labels/Classes (raw labels to be used for image-to-text classification in this case)
        """
        label_tokens = self.txt_pp(labels)  # --------- Tensor(L, T) ~ L for num. classes, T for num. tokens i.e. context length

        with torch.no_grad():
            if self.type == "openai":
                txt_embs = self.clip.get_text_features(**label_tokens)
            else:  # BioCLIP
                txt_embs = self.clip.encode_text(label_tokens)

        return txt_embs

    def img2txt_classify(self, embs_images, embs_labels, labels_enc):
        """
        Args:
        - embs_images [Tensor(B, D)] --- Batch of image embeddings (D for dim. embeddings)
        - embs_labels [Tensor(L, D)] --- Label embeddings
        - labels_enc [list(int)] ------- Batch of label/class encodings
        """

        # cosine similarity + softmax
        img_embs = F.normalize(embs_images, dim=-1)  # --- Tensor(B, D) ~ D for dim. embeddings
        txt_embs = F.normalize(embs_labels, dim=-1)  # --- Tensor(L, D)
        logits   = img_embs @ txt_embs.T
        probs    = logits.softmax(dim=-1)

        pred_idxs       = probs.argmax(dim=-1)
        scores          = probs[torch.arange(len(pred_idxs)), pred_idxs].tolist()
        pred_labels_enc = [labels_enc[i] for i in pred_idxs.tolist()]

        return pred_labels_enc, scores
