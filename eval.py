import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
import time

from utils import paths, read_pickle, write_pickle
from models import CLIPWrapper
from utils_eval import compute_map_img2img, compute_map_txt2img

import pdb


torch.set_printoptions(profile="full")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


# config params
CLIP_TYPE       = "bioclip"  # "openai" / "bioclip"
CACHED_IMGS     = False  # preload, preprocess, cache all images into memory
BATCH_SIZE      = 512
NUM_WORKERS     = 4  # adjust to CPU cores
SPLIT_NAME      = "D"
TEXT_PREPENDING = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
TEXT_BASE       = "tax"  # "tax" / "sci"

class ImageDataset(Dataset):
    """
    PyTorch requirements for custom Dataset:
    - Inheritance from torch.utils.data.Dataset
    - Implementations of:
        - __len__(self) --> int
        - __getitem__(self, idx) --> sample
    - Everything else is up to you!
    """

    def __init__(self, index_imgs_class_enc, index_imgs_rfpaths, dpath_imgs, img_pp, cached_imgs=False):
        
        self.index_imgs_class_enc = index_imgs_class_enc
        self.index_imgs_rfpaths   = index_imgs_rfpaths
        self.dpath_imgs           = dpath_imgs
        self.img_pp               = img_pp
        self.cached_imgs          = cached_imgs

        self.n_samples = len(self.index_imgs_class_enc)

        if self.cached_imgs:
            # load all images into memory (as preprocessed tensors)
            self.imgs_mem = []
            for rfpath in tqdm(self.index_imgs_rfpaths, desc="Preloading, Preprocessing, Caching Images"):
                img   = Image.open(self.dpath_imgs / rfpath).convert("RGB")
                img_t = self.img_pp(img)
                self.imgs_mem.append(img_t)

    def __len__(self):
        return self.n_samples
    
    # gets called in the background on indices of batch N+1 while GPU (and main process) are busy running img2txt_classify() on batch N
    def __getitem__(self, idx):
        """
        Returns transformed image and class encoding.
        idx --> sample (preprocessed image, class encoding)
        """
        class_enc = self.index_imgs_class_enc[idx]

        if self.cached_imgs:
            img_t = self.imgs_mem[idx]
        else:
            # load + preprocess image
            img   = Image.open(self.dpath_imgs / self.index_imgs_rfpaths[idx]).convert("RGB")
            img_t = self.img_pp(img)
        
        return img_t, class_enc

def collate_fn(batch):
    """
    collate_fn takes list of individual samples from Dataset and merges them into a single batch
    augmentation can be done here methinks
    """
    imgs_b, classes_enc_imgs_b = zip(*batch)

    imgs_b = torch.stack(imgs_b, dim=0)  # --- Tensor(B, C, H, W)
    return imgs_b, list(classes_enc_imgs_b)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"({device})",
        f"Split ------------ {SPLIT_NAME}",
        f"CLIP-type -------- {CLIP_TYPE}",
        f"Text Type -------- {TEXT_PREPENDING}",
        f"Text Base Type --- {TEXT_BASE}",
        sep="\n"
    )
    print()

    model = CLIPWrapper(CLIP_TYPE, device)

    # GET DATASET INDEXES (RELATIVE FILEPATHS TO IMAGES + CLASS ENCODINGS)

    data_index      = read_pickle(paths["metadata_o"] / f"data_indexes/{SPLIT_NAME}/id_val.pkl")
    rank_keys_nymph = read_pickle(paths["metadata_o"] / "rank_keys/nymph.pkl")

    index_imgs_rfpaths   = data_index["rfpaths"]
    index_imgs_sids      = data_index["sids"]
    index_imgs_class_enc = [rank_keys_nymph["species"][sid] for sid in index_imgs_sids]

    if TEXT_BASE == "sci":
        texts_base = read_pickle(paths["metadata_o"] / f"base_texts/nymph_sci.pkl")
    elif TEXT_BASE == "tax":
        texts_base = read_pickle(paths["metadata_o"] / f"base_texts/nymph_tax.pkl")

    if TEXT_PREPENDING == "bioclip":
        texts_prepending = "a photo of "  # BioCLIP-style prepending
    elif TEXT_PREPENDING == "openai":
        texts_prepending = "a photo of a "  # OpenAI CLIP-style prepending
    elif TEXT_PREPENDING == "base":
        texts_prepending = ""  # no prepending

    dataset = ImageDataset(
        index_imgs_class_enc=index_imgs_class_enc,
        index_imgs_rfpaths=index_imgs_rfpaths,
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
        prefetch_factor=2,  # how many batches each worker will load in advance; higher prefetch_factor increases throughput, higher RAM cost; only takes effect when num_workers > 0
        collate_fn=collate_fn,
    )

    # reminder: make sure this sort doesn't happen every validation run
    sids             = sorted(set(index_imgs_sids))  # i.e. "classes"
    texts            = [texts_prepending + texts_base[sid] for sid in sids]
    classes_enc_txts = [rank_keys_nymph["species"][sid] for sid in sids]

    n_classes       = len(sids)
    n_samps         = len(dataset)
    counter_classes = Counter(index_imgs_class_enc)
    _, n_maj        = counter_classes.most_common(1)[0]
    print(
        f"Num. Classes ----------------------- {n_classes:,}",
        f"Expected Prec@1 Random Selection --- {1 / n_classes:.2%} ({n_samps / n_classes:.2f}/{n_samps:,})",
        f"Prec@1 Majority Selection ---------- {n_maj / n_samps:.2%} ({n_maj}/{n_samps:,})",
        sep="\n"
    )
    print()

    time_start = time.time()

    # for img2img and txt2img eval
    embs_imgs        = []
    classes_enc_imgs = []

    embs_txts = model.embed_texts(texts)  # ----------- Tensor(L, D)

    # eval loop
    n_correct = 0
    for idx_b, (imgs_b, targ_classes_enc_b) in enumerate(tqdm(loader, desc="Image-to-Text Eval (ID)", leave=False), start=1):

        embs_imgs_b = model.embed_images(imgs_b)  # --- Tensor(B, D)
        pred_classes_enc_txts_b, _ = model.img2txt_classify(embs_imgs_b, embs_txts, classes_enc_txts)

        embs_imgs.append(embs_imgs_b.cpu())
        classes_enc_imgs.append(torch.tensor(targ_classes_enc_b, dtype=torch.long))

        n_correct_b = sum(p == t for p, t in zip(pred_classes_enc_txts_b, targ_classes_enc_b))
        n_correct += n_correct_b

        B = len(imgs_b)
        prec1_b = n_correct_b / B
        tqdm.write(f" Batch {idx_b:3d} --- Prec@1: {prec1_b:.2%} ({n_correct_b}/{B:,})")

    # img2txt precision@1 computation
    prec1 = n_correct / n_samps
    print(f"\nImage-to-text Classification Precision@1 --- {prec1:.2%} ({n_correct}/{n_samps})")

    time_end = time.time()
    time_elapsed_img2txt = time_end - time_start

    # prepare image embedding and class encoding tensors for img2img and txt2img mAP computation
    embs_imgs        = torch.cat(embs_imgs, dim=0)  # ---------- Tensor(Q, D)
    classes_enc_imgs = torch.cat(classes_enc_imgs, dim=0)  # --- Tensor(Q)

    time_start = time.time()

    # img2img mAP computation
    map_img2img = compute_map_img2img(embs_imgs, classes_enc_imgs)
    print(f"Image-to-image Retrieval mAP --------------- {map_img2img:.4f}")

    time_end = time.time()
    time_elapsed_img2img = time_end - time_start
    time_start = time.time()

    # txt2img mAP computation
    map_txt2img = compute_map_txt2img(embs_txts.cpu(), torch.tensor(classes_enc_txts), embs_imgs, classes_enc_imgs)
    print(f"Text-to-image Retrieval mAP ---------------- {map_txt2img:.4f}")

    time_end = time.time()
    time_elapsed_txt2img = time_end - time_start

    print(
        f"Elapsed Time:",
        f"img2txt --- {time_elapsed_img2txt:.2f} (s)",
        f"img2img --- {time_elapsed_img2img:.2f} (s)",
        f"txt2img --- {time_elapsed_txt2img:.2f} (s)",
        sep="\n"
    )

if __name__ == "__main__":
    main()
