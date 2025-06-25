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
import time

from utils import paths, read_pickle, write_pickle
from models import CLIPWrapper
from utils_eval import compute_map_img2img

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
SPLIT_NAME      = "A"
LABEL_TYPE      = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
LABEL_BASE_TYPE = "tax"  # "tax" / "sci"

class ImageTextDataset(Dataset):
    """
    PyTorch requirements for custom Dataset:
    - Inheritance from torch.utils.data.Dataset
    - Implementations of:
        - __len__(self) --> int
        - __getitem__(self, idx) --> sample
    - Everything else is up to you!
    """

    def __init__(self, index_labels_enc, index_rfpaths_imgs, dpath_imgs, img_pp, cached_imgs=False):
        
        self.index_labels_enc   = index_labels_enc
        self.index_rfpaths_imgs = index_rfpaths_imgs
        self.dpath_imgs         = dpath_imgs
        self.img_pp             = img_pp
        self.cached_imgs        = cached_imgs

        self.n_samples = len(self.index_labels_enc)

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
        idx --> sample (preprocessed image, label/class encoding)
        Returns transformed image and label
        """
        label_enc = self.index_labels_enc[idx]

        if self.cached_imgs:
            img_t = self.imgs_mem[idx]
        else:
            # load + preprocess image
            img   = Image.open(self.dpath_imgs / self.index_rfpaths_imgs[idx]).convert("RGB")
            img_t = self.img_pp(img)
        
        return img_t, label_enc

def collate_fn(batch):
    """
    collate_fn takes list of individual samples from Dataset and merges them into a single batch
    augmentation can be done here methinks
    """
    imgs, labels_enc = zip(*batch)

    imgs = torch.stack(imgs, dim=0)  # --- Tensor(B, C, H, W)
    return imgs, list(labels_enc)

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

    # GET DATASET INDEXES (LABELS + LABEL ENCODINGS + RELATIVE FILEPATHS TO IMAGES)

    data_index      = read_pickle(paths["metadata_o"] / f"data_indexes/{SPLIT_NAME}/id_val.pkl")
    rank_keys_nymph = read_pickle(paths["metadata_o"] / "rank_keys/nymph.pkl")

    index_rfpaths_imgs = data_index["rfpaths"]
    index_sids         = data_index["sids"]
    index_labels_enc   = [rank_keys_nymph["species"][sid] for sid in index_sids]

    if LABEL_BASE_TYPE == "sci":
        labels_base = read_pickle(paths["metadata_o"] / f"base_labels/nymph_sci.pkl")
    elif LABEL_BASE_TYPE == "tax":
        labels_base = read_pickle(paths["metadata_o"] / f"base_labels/nymph_tax.pkl")

    if LABEL_TYPE == "bioclip":
        label_prepending = "a photo of "  # BioCLIP-style prepending
    elif LABEL_TYPE == "openai":
        label_prepending = "a photo of a "  # OpenAI CLIP-style prepending
    elif LABEL_TYPE == "base":
        label_prepending = ""  # no prepending

    dataset = ImageTextDataset(
        index_labels_enc=index_labels_enc,
        index_rfpaths_imgs=index_rfpaths_imgs,
        dpath_imgs=paths["nymph"] / "images",
        img_pp=model.img_pp,
        cached_imgs=CACHED_IMGS,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # (True) speeds up host --> GPU copies, higher RAM cost
        prefetch_factor=2,  # how many batches each worker will load in advance -- higher prefetch_factor increases throughput, higher RAM cost ~ only takes effect when num_workers > 0
        collate_fn=collate_fn,
    )

    # reminder: make sure this sort doesn't happen every validation run
    sids = sorted(set(index_sids))  # i.e. "classes"
    labels = [label_prepending + labels_base[sid] for sid in sids]
    labels_enc = [rank_keys_nymph["species"][sid] for sid in sids]

    n_labels = len(labels)
    n_samps = len(dataset)
    counter_labels = Counter(index_labels_enc)
    _, n_maj = counter_labels.most_common(1)[0]
    print(
        f"Num. Labels ------------------------ {n_labels:,}",
        f"Expected Prec@1 Random Selection --- {100 * 1 / n_labels:.2f}% ({n_samps / n_labels:.2f}/{n_samps:,})",
        f"Prec@1 Majority Selection ---------- {100 * n_maj / n_samps:.2f}% ({n_maj}/{n_samps:,})",
        sep="\n"
    )
    print()

    time_start = time.time()

    # for img2img eval
    img_embs_all = []
    labels_all = []

    # eval loop
    n_correct = 0
    for idx_b, (imgs, targ_labels_enc) in enumerate(tqdm(loader, desc="Image-to-Text Eval (ID)", leave=False), start=1):

        embs_imgs = model.embed_images(imgs)  # ---- Tensor(B, D)
        embs_lbls = model.embed_texts(labels)  # --- Tensor(L, D)  ***** EMBED LABELS ONCE AND INDEX INTO THAT TENSOR IF POSSIBLE, WILL BE FASTER *****
        pred_labels_enc, _ = model.img2txt_classify(embs_imgs, embs_lbls, labels_enc)

        img_embs_all.append(embs_imgs.cpu())
        labels_all.append(torch.tensor(targ_labels_enc, dtype=torch.long))

        n_correct_b = sum(p == t for p, t in zip(pred_labels_enc, targ_labels_enc))
        n_correct += n_correct_b

        B = len(imgs)
        prec1_b = n_correct_b / B
        tqdm.write(f" Batch {idx_b:3d} --- Prec@1: {prec1_b:.2%} ({n_correct_b}/{B:,})")

    # img2txt precision@1 computation
    prec1 = n_correct / n_samps
    print(f"\nImage-to-text Classification Precision@1 --- {prec1:.2%} ({n_correct}/{n_samps})")

    # img2img mAP computation
    img_embs_all = torch.cat(img_embs_all, dim=0)  # --- Tensor(Q, D)
    labels_all = torch.cat(labels_all, dim=0)  # ------- Tensor(Q)
    map_img2img = compute_map_img2img(img_embs_all, labels_all)
    print(f"Image-to-image Retrieval mAP --------------- {map_img2img:.4f}")

    time_end = time.time()
    print(f"Time Elapsed: {time_end - time_start:.2f} s")

if __name__ == "__main__":
    main()
