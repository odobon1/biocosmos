import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from PIL import Image
import numpy as np
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    RandomResizedCrop,
    Normalize,
    InterpolationMode,
    RandomHorizontalFlip,
    ColorJitter,
    RandomApply,
    GaussianBlur,
    Lambda,
)
import torchvision.transforms.functional as F
import random
import fcntl
import getpass
import io
import mmap
import os
import shutil
import tempfile
import time
from pathlib import Path


from utils.text import get_text_generator
from utils.utils import paths, load_pickle, load_json, load_split, shuffle_list
from utils.config import EvalConfig, _default_train_aug_cfg

import pdb


def _merge_aug_cfg(aug_cfg: Mapping[str, Any] | None) -> dict:
    default_aug_cfg = _default_train_aug_cfg()
    if aug_cfg is None:
        return default_aug_cfg

    aug_cfg = dict(aug_cfg)
    merged = {**default_aug_cfg, **aug_cfg}
    merged["rrcrop"] = {**default_aug_cfg["rrcrop"], **dict(aug_cfg.get("rrcrop", {}))}
    merged["cjit"] = {**default_aug_cfg["cjit"], **dict(aug_cfg.get("cjit", {}))}
    merged["gblur"] = {**default_aug_cfg["gblur"], **dict(aug_cfg.get("gblur", {}))}
    return merged

def build_train_augmentation_transforms(
    img_res: int,
    aug_cfg: Mapping[str, Any] | None = None,
) -> Compose:
    aug_cfg = _merge_aug_cfg(aug_cfg)
    transforms = [
        RandomResizedCrop(
            size=img_res,
            scale=(aug_cfg["rrcrop"]["scale_min"], 1.0),
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        )
    ]
    if aug_cfg.get("hflip", False):
        transforms.append(RandomHorizontalFlip())

    if aug_cfg["cjit"]["prob"] != 0.0:
        transforms.append(RandomApply([
            ColorJitter(
                brightness=aug_cfg["cjit"]["brightness"],
                contrast=aug_cfg["cjit"]["contrast"],
                saturation=aug_cfg["cjit"]["saturation"],
                hue=aug_cfg["cjit"]["hue"],
            )
        ], p=aug_cfg["cjit"]["prob"]))

    if aug_cfg["sharpness"]["prob"] != 0.0:
        sharpness_factor = aug_cfg["sharpness"]["factor"]
        sharpness_min = 1.0 / sharpness_factor
        sharpness_max = sharpness_factor
        transforms.append(RandomApply([
            Lambda(
                lambda img: F.adjust_sharpness(
                    img,
                    sharpness_factor=random.uniform(sharpness_min, sharpness_max),
                )
            )
        ], p=aug_cfg["sharpness"]["prob"]))

    if aug_cfg["gblur"]["prob"] != 0.0:
        kernel_size_gb = int(aug_cfg["gblur"]["kernel_size"])
        sigma_min = aug_cfg["gblur"]["sigma"]["min"]
        sigma_max = aug_cfg["gblur"]["sigma"]["max"]
        sigma_min = max(float(sigma_min), 1.0e-6)
        sigma_max = max(float(sigma_max), sigma_min)
        transforms.append(RandomApply([
            GaussianBlur(kernel_size_gb, sigma=(sigma_min, sigma_max))
        ], p=aug_cfg["gblur"]["prob"]))

    return Compose(transforms)

def make_image_preprocessor_inference(
    img_res: int, 
    norm_mean: Tuple[float], 
    norm_std: Tuple[float],
):
    """
    Create a preprocessing pipeline for inference.

    Args:
        img_res (int): Image resolution
        norm_mean (Tuple[float]): Mean for normalization (3-tuple for RGB channels)
        norm_std (Tuple[float]): Standard deviation for normalization (3-tuple for RGB channels)

    Returns:
        Compose: A torchvision Compose object with the preprocessing steps
    """
    pp_inf = Compose([
        Resize(
            size=img_res,
            interpolation=InterpolationMode.BICUBIC,
        ),
        CenterCrop(size=(img_res, img_res)),
        MaybeConvertMode(),
        MaybeToTensor(),
        Normalize(mean=norm_mean, std=norm_std),
    ])
    return pp_inf

def make_image_preprocessor_train(
    img_res: int, 
    norm_mean: Tuple[float],
    norm_std: Tuple[float],
    aug_cfg: Mapping[str, Any] | None = None,
):
    """
    Create a preprocessing pipeline for training.

    Args:
        img_res (int): Image resolution
        norm_mean (Tuple[float]): Mean for normalization (3-tuple for RGB channels)
        norm_std (Tuple[float]): Standard deviation for normalization (3-tuple for RGB channels)

    Returns:
        Compose: A torchvision Compose object with the preprocessing steps
    """
    pp_train = Compose(
        list(build_train_augmentation_transforms(img_res, aug_cfg=aug_cfg).transforms)
        + [
            MaybeConvertMode(),
            MaybeToTensor(),
            Normalize(
                mean=norm_mean,
                std=norm_std,
            ),
        ]
    )
    return pp_train

# helper for make_image_preprocessor_inference() and make_image_preprocessor_train()
class MaybeConvertMode:
    def __call__(self, image):
        # convert PIL image to RGB if needed
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        return image

# helper for make_image_preprocessor_inference() and make_image_preprocessor_train()
class MaybeToTensor:
    def __call__(self, image):
        import torch
        from torchvision.transforms.functional import to_tensor
        # avoid double-conversion
        if isinstance(image, torch.Tensor):
            return image
        return to_tensor(image)

def assemble_data_index(data_indexes: Dict[str, Any], partition: str) -> List[Dict[str, Any]]:
    """Assemble the flat data-index list for `partition` from the nested data_indexes dict.

    Raw partitions return their underlying list; composite partitions concatenate:
      - "trainval" = train + val(id) + val(ood)
      - "whole"    = train + val(id) + val(ood) + test(id) + test(ood)
    """
    raw = {
        "train":    data_indexes["train"],
        "val_id":   data_indexes["val"]["id"],
        "val_ood":  data_indexes["val"]["ood"],
        "test_id":  data_indexes["test"]["id"],
        "test_ood": data_indexes["test"]["ood"],
    }
    if partition in raw:
        return raw[partition]
    if partition == "trainval":
        return raw["train"] + raw["val_id"] + raw["val_ood"]
    if partition == "whole":
        return raw["train"] + raw["val_id"] + raw["val_ood"] + raw["test_id"] + raw["test_ood"]
    raise KeyError(
        f"Unknown partition '{partition}'; valid: "
        "train, val_id, val_ood, test_id, test_ood, trainval, whole"
    )


class Split:
    def __init__(
        self,
        data_indexes: Dict[str, Any],
        enc2cid: Dict[int, str],
        nshot: Dict[str, Any],
        class_counts: Dict[str, np.ndarray],
        norm_mean: Dict[str, Tuple[float]],
        norm_std: Dict[str, Tuple[float]],
    ) -> None:
        self._data_indexes = data_indexes
        self.enc2cid = enc2cid
        self.nshot = nshot
        self.class_counts = class_counts
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def get_data(self, partition: str) -> List[Dict[str, Any]]:
        return assemble_data_index(self._data_indexes, partition)

class EpochEncodingDistributedSampler(DistributedSampler):
    """DistributedSampler that encodes the epoch into each yielded index.

    Yields ``epoch * len(dataset) + raw_idx`` so that ImageTextDataset can derive
    a per-(epoch, item) augmentation seed without needing to know the epoch itself.
    """
    def __iter__(self):
        epoch_offset = self.epoch * len(self.dataset)
        return (idx + epoch_offset for idx in super().__iter__())

class ExactDistributedSampler(Sampler[int]):
    """
    Distributed sampler that assigns each dataset index to exactly one rank.

    Unlike torch's DistributedSampler(drop_last=False), this sampler does not pad
    to make every rank see the same number of samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = shuffle_list(indices, self.seed + self.epoch)
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self) -> int:
        n = len(self.dataset)
        remaining = max(n - self.rank, 0)
        return (remaining + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

class DorsalVentralBatchSampler:
    """
    Homogeneous-Position Batch Sampler
    Batch sampler that yields batches composed entirely of either dorsal or ventral samples, drawn without replacement.

    Only for training
    
    Always shuffles
    Always drops partial batch
    """

    def __init__(
        self,
        index_pos:  List[str],
        batch_size: int,
        seed:       int,
    ) -> None:

        self.n_replicas = dist.get_world_size()
        self.subbatch_size = int(batch_size / self.n_replicas)
        self.rank = dist.get_rank()
        self.seed = seed
        self.epoch = 0
        self.n_samples = len(index_pos)

        self.idxs_d = [i for i, p in enumerate(index_pos) if p == "dorsal"]
        self.idxs_v = [i for i, p in enumerate(index_pos) if p == "ventral"]

        self.n_batches_d = len(self.idxs_d) // batch_size
        self.n_batches_v = len(self.idxs_v) // batch_size
        self.n_batches   = self.n_batches_d + self.n_batches_v

    def __len__(self) -> int:
        return self.n_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _get_idxs_pos_local(
        self,
        idxs_pos:      List[int],
        n_batches_pos: int,
        offset_seed:   int,
    ) -> List[int]:
        
        # shuffle ~ same seeds used across GPUs so shuffling is identical
        idxs_pos_shuf = shuffle_list(idxs_pos, self.seed + self.epoch + offset_seed)

        n_samps_local = self.subbatch_size * n_batches_pos

        idx_start = self.rank * n_samps_local
        idx_end   = idx_start + n_samps_local
        return idxs_pos_shuf[idx_start:idx_end]
        
    def __iter__(self):
        idxs_d_local = self._get_idxs_pos_local(self.idxs_d, self.n_batches_d, offset_seed=0)
        idxs_v_local = self._get_idxs_pos_local(self.idxs_v, self.n_batches_v, offset_seed=1)

        pool_tags: list[str] = (["dorsal"] * self.n_batches_d) + (["ventral"] * self.n_batches_v)
        pool_tags = shuffle_list(pool_tags, self.seed + self.epoch + 2)  # shuffle pool tags

        epoch_offset = self.epoch * self.n_samples

        idx_d = 0
        idx_v = 0
        for tag in pool_tags:
            if tag == "dorsal":
                subbatch = [i + epoch_offset for i in idxs_d_local[idx_d:idx_d+self.subbatch_size]]
                idx_d += self.subbatch_size
            else:
                subbatch = [i + epoch_offset for i in idxs_v_local[idx_v:idx_v+self.subbatch_size]]
                idx_v += self.subbatch_size

            yield subbatch

def dpath_img_cache_staged(dataset):
    """Node-local staging dir for a dataset's image pack. Root is picked cluster-agnostically: SLURM_TMPDIR,
    else TMPDIR (schedulers that provide a per-job node-local scratch set one of these), else /tmp. Keyed by
    user since /tmp is shared across users on a node."""
    root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    return Path(root) / f"img_cache-{getpass.getuser()}" / dataset

def _img_cache_staged_valid(dpath_src, dpath_staged):
    """A staged pack is complete+current iff its meta.json (copied LAST during staging) byte-matches the
    source's (a rebuilt source pack changes created_utc => stale) and pack.bin has the recorded size."""
    fpath_meta = dpath_staged / "meta.json"
    if not fpath_meta.exists():
        return False
    if fpath_meta.read_bytes() != (dpath_src / "meta.json").read_bytes():
        return False
    # tolerate partial eviction: a /tmp reaper can age out pack.bin/index.pkl (mmap reads don't refresh
    # atime) while the frequently-read meta.json survives -- classify as invalid, don't crash
    fpath_pack = dpath_staged / "pack.bin"
    if not fpath_pack.exists() or not (dpath_staged / "index.pkl").exists():
        return False
    return fpath_pack.stat().st_size == load_json(fpath_meta)["total_bytes"]

def stage_img_cache(dataset):
    """Ensure dataset's image pack (pack.bin + index.pkl + meta.json) is staged on node-local disk; return
    seconds spent. Concurrency-safe via flock on the node-local FS -- no torch.distributed dependency, so it
    works for 1 or N ranks and even concurrent jobs on the same node: the first process copies while the rest
    block on the lock, then take the already-staged fast path."""
    t0 = time.perf_counter()
    dpath_src = paths["img_cache"] / dataset
    if not (dpath_src / "meta.json").exists():
        raise FileNotFoundError(
            f"use_img_cache is enabled but there is no image pack for '{dataset}' at {dpath_src} -- "
            f"build it first: python -m tools.build_img_cache"
        )
    dpath_staged = dpath_img_cache_staged(dataset)
    if _img_cache_staged_valid(dpath_src, dpath_staged):  # lock-free fast path: files are immutable once marked
        return time.perf_counter() - t0
    dpath_staged.mkdir(parents=True, exist_ok=True)
    with open(dpath_staged.parent / f".{dataset}.lock", "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if _img_cache_staged_valid(dpath_src, dpath_staged):  # another proc staged it while we waited
            return time.perf_counter() - t0
        # drop stale/partial files (marker first) so the space check sees the reclaimable space and a
        # rebuilt-pack restage's transient footprint stays ~1x pack size instead of 2x
        (dpath_staged / "meta.json").unlink(missing_ok=True)
        for fpath_stale in dpath_staged.iterdir():
            fpath_stale.unlink()
        total_bytes = load_json(dpath_src / "meta.json")["total_bytes"]
        free = shutil.disk_usage(dpath_staged).free
        if free < total_bytes * 1.02:
            raise RuntimeError(
                f"staging '{dataset}' image pack needs {total_bytes / 1e9:.1f} GB but node-local "
                f"{dpath_staged} has only {free / 1e9:.1f} GB free"
            )
        print(f"[img cache] staging '{dataset}' ({total_bytes / 1e9:.1f} GB) -> {dpath_staged} ...", flush=True)
        for fname in ("pack.bin", "index.pkl", "meta.json"):  # meta.json last: it is the completion marker
            fpath_tmp = dpath_staged / f"{fname}.tmp"
            shutil.copyfile(dpath_src / fname, fpath_tmp)
            os.replace(fpath_tmp, dpath_staged / fname)
        print(f"[img cache] staged '{dataset}' in {time.perf_counter() - t0:.1f} s", flush=True)
    return time.perf_counter() - t0

_cache_indexes = {}  # dataset -> index dict, shared across the ImageTextDataset instances of one process

def _load_cache_index(dataset, dpath_staged):
    if dataset not in _cache_indexes:
        _cache_indexes[dataset] = load_pickle(dpath_staged / "index.pkl")
    return _cache_indexes[dataset]

class ImageTextDataset(Dataset):

    def __init__(
        self,
        index_data,
        enc2cid,
        text_template,
        img_pp,
        config,
    ):
        self.index_data = index_data
        self.enc2cid = enc2cid
        self.text_template = text_template
        self.img_pp = img_pp
        self.dataset = config.dataset
        self.text_generator = get_text_generator(self.dataset)
        self._aug_seed = getattr(config, "seed", None)

        self.use_img_cache = config.use_img_cache
        if self.use_img_cache:
            stage_img_cache(self.dataset)
            self._fpath_cache_pack = dpath_img_cache_staged(self.dataset) / "pack.bin"
            self._cache_index = _load_cache_index(self.dataset, dpath_img_cache_staged(self.dataset))
            # mmap opened lazily per process: DataLoader workers fork after __init__, and an mmap handle
            # must belong to the process that reads through it
            self._cache_file = None
            self._cache_mm = None

        self.n_samples = len(self.index_data)

        self.class_data = load_pickle(paths["metadata"][self.dataset] / "class_data.pkl")
        self.rank_encs = load_pickle(paths["metadata"][self.dataset] / "rank_encs.pkl")

        self.missing_class_data_cids = {
            cid
            for datum in self.index_data
            if (cid := self.enc2cid[datum["class_enc"]]) not in self.class_data
        }
        if self.missing_class_data_cids:
            print(
                "WARNING: Missing class_data for "
                f"{len(self.missing_class_data_cids)} class ids in dataset rows; "
                "using cid-based fallback metadata."
            )

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, encoded_idx):
        """
        Get processed image tensor, class encoding, and generated text.
        encoded_idx --> sample (preprocessed image, class encoding, text)

        encoded_idx encodes both the epoch and the item: encoded_idx = epoch * n_samples + raw_idx.
        This allows augmentation to be seeded per (epoch, item) without passing epoch explicitly
        to workers, which is needed for cross-session reproducibility with persistent workers.

        This method gets called in a background thread/process to prepare batch N+1 while GPU and main process are
        processing batch N.

        Args:
        - encoded_idx --- [int] --- Epoch-encoded sample index

        Returns:
        - [torch.Tensor]
        - [int]
        - [str]
        """
        idx = encoded_idx % self.n_samples

        class_enc = self.index_data[idx]["class_enc"]
        cid = self.enc2cid[class_enc]
        meta = self.index_data[idx]["meta"]

        rank_encs = []
        class_data_cid = self.class_data.get(cid)

        for rank_name, rank_map in self.rank_encs.items():
            rank_value = class_data_cid.get(rank_name)
            rank_encs.append(rank_map.get(rank_value, -1))

        text = self.text_generator.generate(class_data_cid, self.text_template, meta)

        if self._aug_seed is not None:
            random.seed(self._aug_seed + encoded_idx)
            torch.manual_seed(self._aug_seed + encoded_idx)

        # load + preprocess image
        if self.use_img_cache:
            if self._cache_mm is None:
                self._cache_file = open(self._fpath_cache_pack, "rb")
                self._cache_mm = mmap.mmap(self._cache_file.fileno(), 0, access=mmap.ACCESS_READ)
            offset, length = self._cache_index[self.index_data[idx]["rfpath"]]
            img = Image.open(io.BytesIO(self._cache_mm[offset:offset + length])).convert("RGB")
        else:
            img = Image.open(paths["imgs"][self.dataset] / self.index_data[idx]["rfpath"]).convert("RGB")
        img_t = self.img_pp(img)

        targ_data = {
            "class_enc": class_enc,
            "rank_encs": rank_encs,
            "cid": cid,
            "meta": meta,
            "dataset": self.dataset,
        }

        return img_t, text, class_enc, targ_data

def collate_fn(subbatch):
    imgs_sb, texts_sb, class_encs_sb, targ_data_sb = zip(*subbatch)
    imgs_sb = torch.stack(imgs_sb, dim=0)  # pt[SB, C, H, W]
    class_encs_sb = torch.tensor(class_encs_sb)  # pt[SB]
    return imgs_sb, texts_sb, class_encs_sb, targ_data_sb

def build_cid2enc(index_data, enc2cid):
    cid2enc = {}
    for datum in index_data:
        class_enc = datum["class_enc"]
        cid2enc[enc2cid[class_enc]] = class_enc
    return cid2enc

def spawn_partition_data(config: EvalConfig, partition: str):
    """

    Args:
    - split_type --- [str] --- "train" / "id" / "ood"
    - split --- [str] --- Name of the split directory e.g. "A" / "B" / etc.
    """
    split = load_split(config.dataset, config.split)
    if partition in ("train", "trainval"):
        index_data = split.get_data(partition)
    else:
        index_data = split.get_data(f"{config.eval_type}_{partition}")
    cid2enc = build_cid2enc(index_data, split.enc2cid)
    return index_data, cid2enc, split.enc2cid

def spawn_partition_indexes_txts(cid2enc, text_template, dataset):
    """
    
    Args:
    - cid2enc --- [dict(cid --> class enc)] --- generated by spawn_partition_data()
    """

    index_text_cids = list(cid2enc.keys())
    index_text_class_encs = list(cid2enc.values())
    class_data = load_pickle(paths["metadata"][dataset] / "class_data.pkl")
    text_generator = get_text_generator(dataset)

    index_text = []
    missing_cids = []
    for cid in index_text_cids:
        class_data_cid = class_data.get(cid)
        if class_data_cid is None:
            missing_cids.append(cid)
            class_data_cid = {
                "species": cid,
                "genus": cid,
                "common_name": None,
            }
        index_text.append(text_generator.generate(class_data_cid, text_template))

    if missing_cids:
        print(
            "WARNING: Missing class_data for "
            f"{len(missing_cids)}/{len(index_text_cids)} class ids while generating text indexes; "
            "using cid fallback text for missing entries."
        )

    return index_text, torch.tensor(index_text_class_encs)

def spawn_dataloader(
    index_data: List[Dict[str, Union[int, str]]],
    enc2cid: Dict[int, str],
    text_template: List[List[str]],
    config,
    shuffle: bool,
    drop_last: bool,
    img_pp: Callable,
    use_dv_sampler: bool,
    exact_distributed: bool = False,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, float | None]:
    """

    Args:
    - index_data ------ Data indexes: Class encodings, relative filepaths to images, species ID's, position, sex
    - text_template --- List of text prepending selections that are randomly picked from to assemble train texts
    - config ---------- contains 1. batch_size (int), 2. n_workers (int) ~ Parallelism, 
                        3. prefetch_factor (int) ~ How many batches each worker 
                        will load in advance; Higher prefetch_factor increases throughput, higher RAM cost; Only 
                        takes effect when n_workers > 0
    - shuffle --------- Whether to shuffle samples between cycles
    - drop_last ------- Whether to drop partial batch at the end of epoch (only need this arg for train)
    - img_pp ---------- The image preprocessor          
    """

    dataset = ImageTextDataset(index_data, enc2cid, text_template, img_pp, config)

    # conditionally use D/V sampling only when a config flag is set (so eval still uses the normal behaviour)
    if use_dv_sampler:
        index_pos = []
        n_invalid_pos = 0
        for datum in dataset.index_data:
            meta = datum.get("meta")
            pos = meta.get("pos") if isinstance(meta, Mapping) else None
            index_pos.append(pos)
            if pos not in ("dorsal", "ventral"):
                n_invalid_pos += 1

        # D/V batching requires every sample to carry a valid "dorsal"/"ventral" tag.
        # If unavailable (e.g., bryo/cub), fallback to the standard distributed sampler.
        use_dv_sampler = n_invalid_pos == 0
        if not use_dv_sampler:
            print(
                "WARNING: dv_batching=True but "
                f"{n_invalid_pos}/{len(index_pos)} samples have missing/invalid meta.pos; "
                "falling back to standard distributed sampling."
            )

    if use_dv_sampler:
        batch_sampler = DorsalVentralBatchSampler(
            index_pos=index_pos,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.n_workers,
            pin_memory=config.hw.pin_memory,  # (True) speeds up host --> GPU copies, higher RAM cost
            prefetch_factor=config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
    else:
        if exact_distributed:
            sampler = ExactDistributedSampler(
                dataset,
                shuffle=shuffle,
                seed=getattr(config, "seed", 0),
            )
        else:
            sampler = EpochEncodingDistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        SB = config.batch_size // dist.get_world_size()
        dataloader = DataLoader(
            dataset,
            batch_size=SB,
            shuffle=False,
            num_workers=config.n_workers,
            pin_memory=config.hw.pin_memory,  # (True) speeds up host --> GPU copies, higher RAM cost
            prefetch_factor=config.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            sampler=sampler,
        )

    return dataloader

def gen_text(
    class_data_cid: Mapping[str, Any],
    combo_temp,
    dataset: str,
    meta: Mapping[str, Any] | None = None,
):
    return get_text_generator(dataset).generate(class_data_cid, combo_temp, meta)

_PENULT_KEY: dict[str, str] = {
    "bryo": "family",
    "cub": "genus",
    "lepid": "genus",
    "nymph": "genus",
}

def load_cid_2_penult(dataset: str) -> dict:
    class_data = load_pickle(paths["metadata"][dataset] / "class_data.pkl")
    key = _PENULT_KEY[dataset]
    return {cid: entry[key] for cid, entry in class_data.items()}

def load_cid_2_nshot(dataset: str, split: str, eval_type: str):
    """(cid -> n-shot bucket name, ordered bucket names) for the eval partition's n-shot buckets
    ('train/val' for val, 'trainval/test' for test) -- the same buckets the learning curves use."""
    sp = load_split(dataset, split)
    bucket_key = "train/val" if eval_type == "val" else "trainval/test"
    cid_2_nshot = {cid: name for name in sp.nshot["names"] for cid in sp.nshot["buckets"][bucket_key][name]}
    return cid_2_nshot, list(sp.nshot["names"])

def truncate_subspecies(s: str) -> str:
    parts = s.split("_", 2)
    if len(parts) < 3:
        return s
    return parts[0] + "_" + parts[1]