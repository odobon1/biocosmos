import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from typing import Dict, List, Optional, Tuple


def spawn_dataloader(dpath_valid, img_pp, batch_size, n_workers):
    dataset    = ImageFolder(dpath_valid, transform=img_pp)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    return dataloader

@torch.no_grad()
def batch_prec1(logits: torch.tensor, targs: torch.Tensor) -> torch.Tensor:
    preds      = logits.argmax(dim=1)
    prec1_mean = (preds == targs).float().mean()
    return prec1_mean