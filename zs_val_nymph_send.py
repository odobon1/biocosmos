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

from utils import paths

import pdb

torch.set_printoptions(profile="full")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


# config params
PRELOAD_IMAGES = True  # preload all images into memory -- be careful with RAM usage, if you exceed available memory you'll hit swapping and slow everything down
BATCH_SIZE     = 128
CLIP_TYPE      = "openai"
# CLIP_TYPE      = "bioclip"
N_SAMPS        = 2_000  # number of samples to run inference on
NUM_WORKERS    = 4  # adjust to CPU cores


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
                padding=True
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

    def zero_shot_classify(self, imgs, zs_labels):
        """
        Args:
        - imgs [Tensor(B, C, H, W)]
        - zs_labels [list] ------------ Raw labels to be used for zero-shot classification (let's call this length Z ~ number of zero-shot labels)
        """

        imgs = imgs.to(self.device)
        zs_label_tokens = self.txt_pp(zs_labels)

        with torch.no_grad():
            if self.type == "openai":
                img_feats = self.clip.get_image_features(pixel_values=imgs)
                txt_feats = self.clip.get_text_features(**zs_label_tokens)
            else:  # BioCLIP
                img_feats = self.clip.encode_image(imgs)
                txt_feats = self.clip.encode_text(zs_label_tokens)

        # cosine similarity + softmax
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        logits    = img_feats @ txt_feats.T
        probs     = logits.softmax(dim=-1)

        idxs_pred   = probs.argmax(dim=-1)
        scores      = probs[torch.arange(len(idxs_pred)), idxs_pred].tolist()
        label_preds = [zs_labels[i] for i in idxs_pred.tolist()]

        return label_preds, scores

class ImageTextDataset(Dataset):

    def __init__(self, df_data, dirpath_imgs, zs_labels, img_pp, preload=False):

        self.df           = df_data.reset_index(drop=True)  # reset_index() so row idx's are (0, 1, 2, ...) for __getitem__() indexing, drop=True prevents old indices being added to DF as "index" col
        self.dirpath_imgs = dirpath_imgs
        self.zs_labels    = zs_labels
        self.img_pp       = img_pp
        self.preload      = preload

        if preload:
            # load all images into memory (as preprocessed tensors)
            self.imgs_mem = []
            for filename in tqdm(self.df["fileNameAsDelivered"], desc="Preloading Images"):
                img   = Image.open(self.dirpath_imgs / filename).convert("RGB")
                img_t = self.img_pp(img)
                self.imgs_mem.append(img_t)

    def __len__(self):
        return len(self.df)
    
    # gets called in the background on indices of batch N+1 while GPU (and main process) are busy running zero_shot_classify() on batch N
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = row["scientificName"]

        if self.preload:
            img_t = self.imgs_mem[idx]
        else:
            # load + preprocess image

            # pdb.set_trace()

            img   = Image.open(self.dirpath_imgs / row["fileNameAsDelivered"]).convert("RGB")
            img_t = self.img_pp(img)
        
        return img_t, label

# collate_fn takes list of individual samples from Dataset and merges them into a single batch
# augmentation can be done here methinks
def collate_fn(batch):
    imgs, txts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # --- Tensor(B, C, H, W)
    return imgs, list(txts)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CLIPWrapper(CLIP_TYPE, device)

    base_dir        = paths["vlm4bio"]
    # group           = "Fish"
    group           = "Bird"
    # group           = "Butterfly"
    dirpath_imgs_g  = base_dir / group / "images"
    filepath_meta_g = base_dir / group / "metadata" / "metadata_10k.csv"

    # get labels
    df_meta_g = pd.read_csv(filepath_meta_g)
    zs_labels = sorted(df_meta_g["scientificName"].dropna().unique())

    df_meta_g_test = df_meta_g[:N_SAMPS]

    dataset = ImageTextDataset(
        df_data=df_meta_g_test,
        dirpath_imgs=dirpath_imgs_g,
        zs_labels=zs_labels,
        img_pp=model.img_pp,
        preload=PRELOAD_IMAGES,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # speeds up host --> GPU copies
        collate_fn=collate_fn,
    )

    # validation loop
    n_correct = 0
    for imgs, labels_targ in tqdm(loader, desc="Zero-Shot Eval"):
        labels_pred, _ = model.zero_shot_classify(imgs, zs_labels)
        n_correct += sum(p == t for p, t in zip(labels_pred, labels_targ))

    # performance computation
    accuracy = n_correct / N_SAMPS
    print(f"Correct: {n_correct}/{N_SAMPS}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
