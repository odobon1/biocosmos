import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tqdm

import pdb


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

def spawn_dataloader(
        index_imgs_class_enc,
        index_imgs_rfpaths,
        dpath_imgs,
        img_pp,
        cached_imgs,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
        prefetch_factor=2,
    ):
    """

    Args:
    - index_imgs_class_enc --- [list(int)] ----------------------------------- Class encodings corresponding to all images in the set
    - index_imgs_rfpaths ----- [list(int)] ----------------------------------- Relative filepaths to images
    - dpath_imgs ------------- [Path] ---------------------------------------- Directory path to images root
    - img_pp ----------------- [torchvision.transforms.transforms.Compose] --- The image preprocessor
    - cached_imgs ------------ [bool] ---------------------------------------- Whether to cache images in memory
    - batch_size ------------- [int] ----------------------------------------- Batch size
    - shuffle ---------------- [bool] ---------------------------------------- Whether to shuffle samples between cycles
    - num_workers ------------ [int] ----------------------------------------- Parallelism
    - pin_memory ------------- [bool] ---------------------------------------- (True) speeds up host --> GPU copies, higher RAM cost
    - prefetch_factor -------- [int] ----------------------------------------- How many batches each worker will load in advance;
                                                                               Higher prefetch_factor increases throughput, higher RAM cost;
                                                                               Only takes effect when num_workers > 0
    """

    dataset = ImageDataset(
        index_imgs_class_enc=index_imgs_class_enc,
        index_imgs_rfpaths=index_imgs_rfpaths,
        dpath_imgs=dpath_imgs,
        img_pp=img_pp,
        cached_imgs=cached_imgs,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )

    return loader
