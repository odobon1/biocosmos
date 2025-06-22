import torch
# monkey-patch modeling_utils safety check
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import open_clip
from collections import Counter

from utils import paths, read_pickle, write_pickle

import pdb


torch.set_printoptions(profile="full")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


# config params
CLIP_TYPE       = "bioclip"  # "openai" / "bioclip"
CACHED_IMGS     = False  # preload, preprocess, cache all images into memory -- be careful with RAM usage, if you exceed available memory you'll hit swapping and slow everything down
BATCH_SIZE      = 512
NUM_WORKERS     = 4  # adjust to CPU cores
SPLIT_NAME      = "D"
LABEL_TYPE      = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
LABEL_BASE_TYPE = "tax"  # "tax" / "sci"

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

    def img2txt_classify(self, imgs, labels):
        """
        Args:
        - imgs [Tensor(B, C, H, W)] --- The images
        - labels [list(str)] ---------- Raw labels to be used for image-to-text classification
        """

        imgs = imgs.to(self.device)
        label_tokens = self.txt_pp(labels)

        with torch.no_grad():
            if self.type == "openai":
                img_embs = self.clip.get_image_features(pixel_values=imgs)
                txt_embs = self.clip.get_text_features(**label_tokens)
            else:  # BioCLIP
                img_embs = self.clip.encode_image(imgs)
                txt_embs = self.clip.encode_text(label_tokens)

        # cosine similarity + softmax
        img_embs = F.normalize(img_embs, dim=-1)  # --- Tensor(B, D) ~ D for dim. embeddings
        txt_embs = F.normalize(txt_embs, dim=-1)  # --- Tensor(L, D) ~ L for number of labels/classes
        logits   = img_embs @ txt_embs.T
        probs    = logits.softmax(dim=-1)

        idxs_pred   = probs.argmax(dim=-1)
        scores      = probs[torch.arange(len(idxs_pred)), idxs_pred].tolist()
        label_preds = [labels[i] for i in idxs_pred.tolist()]

        return label_preds, scores

class ImageTextDataset(Dataset):
    """
    PyTorch requirements for custom Dataset:
    - Inheritance from torch.utils.data.Dataset
    - Implementations of:
        - __len__(self) --> int
        - __getitem__(self, idx) --> sample
    - Everything else is up to you!
    """

    def __init__(self, index_labels, index_rfpaths_imgs, dpath_imgs, img_pp, cached_imgs=False):
        
        self.index_labels       = index_labels
        self.index_rfpaths_imgs = index_rfpaths_imgs
        self.dpath_imgs         = dpath_imgs
        self.img_pp             = img_pp
        self.cached_imgs        = cached_imgs

        self.n_samples = len(self.index_labels)

        if self.cached_imgs:
            # load all images into memory (as preprocessed tensors)
            self.imgs_mem = []
            for rfpath in tqdm(self.index_rfpaths_imgs, desc="Preloading, Preprocessing, Caching Images"):
                img   = Image.open(self.dpath_imgs / rfpath).convert("RGB")
                img_t = self.img_pp(img)
                self.imgs_mem.append(img_t)

    def __len__(self):
        return self.n_samples
    
    # gets called in the background on indices of batch N+1 while GPU (and main process) are busy running img2txt_classify() on batch N
    def __getitem__(self, idx):
        """
        idx --> sample (preprocessed image, label)
        Returns transformed image and label
        """
        label = self.index_labels[idx]

        if self.cached_imgs:
            img_t = self.imgs_mem[idx]
        else:
            # load + preprocess image
            img   = Image.open(self.dpath_imgs / self.index_rfpaths_imgs[idx]).convert("RGB")
            img_t = self.img_pp(img)
        
        return img_t, label

def collate_fn(batch):
    """
    collate_fn takes list of individual samples from Dataset and merges them into a single batch
    augmentation can be done here methinks
    """
    imgs, txts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # --- Tensor(B, C, H, W)
    return imgs, list(txts)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"({device})",
        f"Split ------------- {SPLIT_NAME}",
        f"CLIP-type --------- {CLIP_TYPE}",
        f"Label Type -------- {LABEL_TYPE}",
        f"Label Base Type --- {LABEL_BASE_TYPE}",
        sep="\n"
    )
    print()

    model = CLIPWrapper(CLIP_TYPE, device)

    # GET DATASET INDEXES (LABELS + RELATIVE FILEPATHS TO IMAGES)

    data_index = read_pickle(paths["metadata_o"] / f"data_indexes/{SPLIT_NAME}/id_val.pkl")

    if LABEL_TYPE == "bioclip":
        label_prepending = "a photo of "  # BioCLIP-style prepending
    elif LABEL_TYPE == "openai":
        label_prepending = "a photo of a "  # OpenAI CLIP-style prepending
    elif LABEL_TYPE == "base":
        label_prepending = ""  # no prepending

    if LABEL_BASE_TYPE == "sci":
        labels_base_index = data_index["base_labels_sci"]
    elif LABEL_BASE_TYPE == "tax":
        labels_base_index = data_index["base_labels_tax"]

    # reminder: make sure this prepending step doesn't happen every val cycle
    index_labels = [label_prepending + labels_base_index[i] for i in range(len(labels_base_index))]
    index_rfpaths_imgs = data_index["rfpaths"]

    dataset = ImageTextDataset(
        index_labels=index_labels,
        index_rfpaths_imgs=index_rfpaths_imgs,
        dpath_imgs=paths["nymph"] / "images",
        img_pp=model.img_pp,
        cached_imgs=CACHED_IMGS,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # speeds up host --> GPU copies (True)
        collate_fn=collate_fn,
    )

    # reminder: make sure this sort doesn't happen every validation run
    labels = sorted(set(index_labels))

    n_labels = len(labels)
    n_samps = len(dataset)
    counter_labels = Counter(index_labels)
    _, n_maj = counter_labels.most_common(1)[0]
    print(
        f"Num. Labels ------------------------ {n_labels:,}",
        f"Expected Prec@1 Random Selection --- {100 * 1 / n_labels:.2f}% ({n_samps / n_labels:.2f}/{n_samps:,})",
        f"Prec@1 Majority Selection ---------- {100 * n_maj / n_samps:.2f}% ({n_maj}/{n_samps:,})",
        sep="\n"
    )
    print()

    # validation loop
    n_correct = 0
    for idx_b, (imgs, labels_targ) in enumerate(tqdm(loader, desc="Image-to-Text Eval (ID)", leave=False), start=1):

        labels_pred, _ = model.img2txt_classify(imgs, labels)
        n_correct_b = sum(p == t for p, t in zip(labels_pred, labels_targ))
        n_correct += n_correct_b
        
        prec1_b = n_correct_b / BATCH_SIZE
        tqdm.write(f" Batch {idx_b:3d} --- Prec@1: {prec1_b:.2%} ({n_correct_b}/{BATCH_SIZE})")


    # performance computation
    prec1 = n_correct / n_samps
    print(f"\nOverall Prec@1: {prec1:.2%} ({n_correct}/{n_samps})")

if __name__ == "__main__":
    main()
