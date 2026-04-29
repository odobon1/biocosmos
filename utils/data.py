import torch  # type: ignore[import]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import]
from torch.utils.data.distributed import DistributedSampler  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
from PIL import Image  # type: ignore[import]
import tqdm  # type: ignore[import]
import numpy as np  # type: ignore[import]
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union

from utils.text import get_text_generator
from utils.utils import paths, load_pickle, load_split, shuffle_list
from utils.config import EvalConfig

import pdb


@dataclass
class Split:
    data_indexes: list
    id_eval_nshot: dict
    class_counts_train: np.ndarray

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

        self.n_replicas    = dist.get_world_size()
        self.subbatch_size = int(batch_size / self.n_replicas)
        self.rank          = dist.get_rank()
        self.seed          = seed
        self.epoch         = 0

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

        idx_d = 0
        idx_v = 0
        for tag in pool_tags:
            if tag == "dorsal":
                subbatch = idxs_d_local[idx_d:idx_d+self.subbatch_size]
                idx_d += self.subbatch_size
            else:
                subbatch = idxs_v_local[idx_v:idx_v+self.subbatch_size]
                idx_v += self.subbatch_size

            yield subbatch

class ImageTextDataset(Dataset):

    def __init__(
        self, 
        index_data,
        text_template,
        img_pp, 
        config,
    ):
        self.index_data = index_data
        self.text_template = text_template
        self.img_pp = img_pp
        self.cached_imgs = config.hw.cached_imgs
        self.dataset = config.dataset
        self.text_generator = get_text_generator(self.dataset)

        self.n_samples = len(self.index_data)

        self.time_cache = None  # need to pass this var to metadata save
        if self.cached_imgs in ("pl", "pp"):
            time_start = time.time()

            def load_pp_img(rfpath):
                img = Image.open(paths["imgs"][self.dataset] / rfpath).convert("RGB")
                return img if self.cached_imgs == "pl" else img_pp(img)

            # load all images into memory (pl: as PIL images; pp: as preprocessed tensors)
            self.imgs_mem = []
            with ThreadPoolExecutor(max_workers=config.n_workers) as exe:
                for img in tqdm.tqdm(
                    exe.map(load_pp_img, [datum["rfpath"] for datum in self.index_data]), 
                    total=self.n_samples, 
                    desc="Caching Images"
                ):
                    self.imgs_mem.append(img)

            time_end        = time.time()
            self.time_cache = time_end - time_start
            print(f"Time Elapsed (image caching): {self.time_cache:.1f} s")

        self.class_data = load_pickle(paths["metadata"][self.dataset] / "class_data.pkl")
        self.rank_encs = load_pickle(paths["metadata"][self.dataset] / "rank_encs.pkl")

        self.missing_class_data_cids = {
            datum["cid"] for datum in self.index_data if datum["cid"] not in self.class_data
        }
        if self.missing_class_data_cids:
            print(
                "WARNING: Missing class_data for "
                f"{len(self.missing_class_data_cids)} class ids in dataset rows; "
                "using cid-based fallback metadata."
            )

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get processed image tensor, class encoding, and generated text.
        idx --> sample (preprocessed image, class encoding, text)

        This method gets called in a background thread/process to prepare batch N+1 while GPU and main process are 
        processing batch N.

        Args:
        - idx --- [int] --- Sample index

        Returns:
        - [torch.Tensor]
        - [int]
        - [str]
        """
        class_enc = self.index_data[idx]["class_enc"]
        cid = self.index_data[idx]["cid"]
        meta = self.index_data[idx]["meta"]

        rank_encs = []
        class_data_cid = self.class_data.get(cid)
        if class_data_cid is None:
            class_data_cid = {
                "species": cid,
                "genus": cid,
                "common_name": None,
            }

        for rank_name, rank_map in self.rank_encs.items():
            rank_value = class_data_cid.get(rank_name)
            rank_encs.append(rank_map.get(rank_value, -1))

        text = self.text_generator.generate(class_data_cid, self.text_template, meta)

        if self.cached_imgs == "pp":
            img_t = self.imgs_mem[idx]
        elif self.cached_imgs == "pl":
            img = self.imgs_mem[idx]
            img_t = self.img_pp(img)
        else:
            # load + preprocess image
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
    """
    collate_fn takes list of individual samples from Dataset and merges them into a single batch
    augmentation can be done here methinks
    """
    imgs_sb, texts_sb, class_encs_sb, targ_data_sb = zip(*subbatch)

    imgs_sb       = torch.stack(imgs_sb, dim=0)  # pt[B, C, H, W]
    class_encs_sb = torch.tensor(class_encs_sb)  # pt[B]

    return imgs_sb, texts_sb, class_encs_sb, targ_data_sb

def build_cid2enc(index_data):
    cid2enc = {}
    for datum in index_data:
        cid = datum["cid"]
        class_enc = datum["class_enc"]
        if cid in cid2enc and cid2enc[cid] != class_enc:
            raise ValueError(f"Inconsistent class_enc for cid '{cid}' in partition rows")
        cid2enc[cid] = class_enc
    return cid2enc

def spawn_partition_data(config: EvalConfig, partition_name: str):
    """

    Args:
    - split_type --- [str] --- "train" / "id" / "ood"
    - split_name --- [str] --- Name of the split directory e.g. "A" / "B" / etc.
    """
    split = load_split(config.split_name, dataset=config.dataset)
    if partition_name == "train":
        index_data = split.data_indexes["train"]
    else:
        index_data = split.data_indexes[config.eval_type][partition_name]
    cid2enc = build_cid2enc(index_data)
    return index_data, cid2enc

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
    text_template: List[List[str]],
    config,
    shuffle: bool,
    drop_last: bool,
    img_pp: Callable,
    use_dv_sampler: bool,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, float | None]:
    """

    Args:
    - index_data ------ Data indexes: Class encodings, relative filepaths to images, species ID's, position, sex
    - text_template --- List of text prepending selections that are randomly picked from to assemble train texts
    - config ---------- contains 1. batch_size (int), 2. cached_imgs (bool) ~ whether to cache images in memory, 
                        3. n_workers (int) ~ Parallelism, 4. prefetch_factor (int) ~ How many batches each worker 
                        will load in advance; Higher prefetch_factor increases throughput, higher RAM cost; Only 
                        takes effect when n_workers > 0
    - shuffle --------- Whether to shuffle samples between cycles
    - drop_last ------- Whether to drop partial batch at the end of epoch (only need this arg for train)
    - img_pp ---------- The image preprocessor          
    """

    dataset = ImageTextDataset(index_data, text_template, img_pp, config)

    bs_local = config.batch_size // dist.get_world_size()

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
            pin_memory=True,  # (True) speeds up host --> GPU copies, higher RAM cost
            prefetch_factor=config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            shuffle  =shuffle,
            drop_last=drop_last,
        )
        dataloader = DataLoader(
            dataset,
            batch_size        =bs_local,
            shuffle           =False,
            num_workers       =config.n_workers,
            pin_memory        =True,  # (True) speeds up host --> GPU copies, higher RAM cost
            prefetch_factor   =config.prefetch_factor,
            collate_fn        =collate_fn,
            drop_last         =drop_last,
            persistent_workers=persistent_workers,
            sampler           =sampler,
        )

    return dataloader, dataset.time_cache

def gen_text(
    class_data_cid: Mapping[str, Any],
    combo_temp,
    dataset: str,
    meta: Mapping[str, Any] | None = None,
):
    return get_text_generator(dataset).generate(class_data_cid, combo_temp, meta)

def species_to_genus(species: str) -> str:
    return species.split("_")[0]

def truncate_subspecies(s: str) -> str:
    parts = s.split("_", 2)
    if len(parts) < 3:
        return s
    return parts[0] + "_" + parts[1]