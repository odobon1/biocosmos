import torch  # type: ignore[import]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import]
from torch.utils.data.distributed import DistributedSampler  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
from PIL import Image  # type: ignore[import]
import tqdm  # type: ignore[import]
import numpy as np  # type: ignore[import]
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import requests  # type: ignore[import]
from typing import Dict, List, Callable, Union, Tuple

from utils import paths, load_pickle, load_split, shuffle_list

import pdb


@dataclass
class Split:
    data_indexes: dict
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
        text_preps,
        img_pp, 
        config,
    ):
        
        self.index_class_encs = index_data["class_encs"]
        self.index_rfpaths    = index_data["rfpaths"]
        self.index_sids       = index_data["sids"]
        self.index_pos        = index_data["pos"]
        self.index_sex        = index_data["sex"]
        self.text_preps       = text_preps
        self.img_pp           = img_pp
        self.cached_imgs      = config.hw.cached_imgs

        self.n_samples = len(self.index_class_encs)

        self.time_cache = None  # need to pass this var to metadata save
        if self.cached_imgs in ("pl", "pp"):
            time_start = time.time()

            def load_pp_img(rfpath):
                img   = Image.open(paths["nymph_imgs"] / rfpath).convert("RGB")
                return img if self.cached_imgs == "pl" else img_pp(img)

            # load all images into memory (pl: as PIL images; pp: as preprocessed tensors)
            self.imgs_mem = []
            with ThreadPoolExecutor(max_workers=config.n_workers) as exe:
                for img in tqdm.tqdm(exe.map(load_pp_img, self.index_rfpaths), total=len(self.index_rfpaths), desc="Caching Images"):
                    self.imgs_mem.append(img)

            time_end        = time.time()
            self.time_cache = time_end - time_start
            print(f"Time Elapsed (image caching): {self.time_cache:.1f} s")

        self.sids2commons   = load_pickle("metadata/species_ids/sids2commons.pkl")
        self.rank_keys_dict = load_pickle("metadata/rank_keys/nymph.pkl")

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
        class_enc = self.index_class_encs[idx]
        sid       = self.index_sids[idx]
        pos       = self.index_pos[idx]
        sex       = self.index_sex[idx]

        genus = sid.split("_")[0]

        rank_key_species = self.rank_keys_dict["species"][sid]
        rank_key_genus   = self.rank_keys_dict["genus"][genus]
        rank_keys        = [rank_key_genus, rank_key_species]

        text = gen_text(sid, self.text_preps, pos, sex, sids2commons=self.sids2commons)

        if self.cached_imgs == "pp":
            img_t = self.imgs_mem[idx]
        elif self.cached_imgs == "pl":
            img   = self.imgs_mem[idx]
            img_t = self.img_pp(img)
        else:
            # load + preprocess image
            img   = Image.open(paths["nymph_imgs"] / self.index_rfpaths[idx]).convert("RGB")
            img_t = self.img_pp(img)

        # could also add pos and sex here for aux heads
        targ_data = {
            "class_enc": class_enc,
            "rank_keys": rank_keys,
            "sid":       sid,
            "pos":       pos,
            "sex":       sex,
        }

        return img_t, text, class_enc, targ_data

def collate_fn(batch):
    """
    collate_fn takes list of individual samples from Dataset and merges them into a single batch
    augmentation can be done here methinks
    """
    imgs_b, texts_b, class_encs_b, targ_data_b = zip(*batch)

    imgs_b       = torch.stack(imgs_b, dim=0)  # --- Tensor(B, C, H, W)
    class_encs_b = torch.tensor(class_encs_b)  # --- Tensor(B)

    return imgs_b, texts_b, class_encs_b, targ_data_b

def assemble_indexes(data_index):
    """
    overall, this functionality needs to be moved to split gen
    """
    index_rfpaths = data_index["rfpaths"]
    index_sids    = data_index["sids"]
    index_pos     = data_index["pos"]
    index_sex     = data_index["sex"]

    sid_2_class_enc  = {}
    class_enc_new    = 0
    index_class_encs = []
    for sid in index_sids:
        if sid not in sid_2_class_enc.keys():
            sid_2_class_enc[sid] = class_enc_new
            class_enc_new += 1
        index_class_encs.append(sid_2_class_enc[sid])

    index_data = {
        "class_encs": index_class_encs,
        "rfpaths":    index_rfpaths,
        "sids":       index_sids,
        "pos":        index_pos,
        "sex":        index_sex,
    }

    return index_data, sid_2_class_enc

def spawn_indexes(split_name, splitset_name):
    """

    Args:
    - split_type --- [str] --- "train" / "id_val" / "id_test" / "ood_val" / "ood_test"
    - split_name --- [str] --- Name of the split directory e.g. "A" / "B" / etc.
    """
    split      = load_split(split_name)
    data_index = split.data_indexes[splitset_name]

    return assemble_indexes(data_index)

def spawn_indexes_txts(sid_2_class_enc, text_preps):
    """
    Think this is still needed but only for eval

    Note: This was split apart from the spawn_indexes() logic for ease of setting up the mixed text-types experiment
    
    Args:
    - sid_2_class_enc --- [dict(sid --> class enc)] --- generated by spawn_indexes()
    """

    index_text_sids       = list(sid_2_class_enc.keys())
    index_text_class_encs = list(sid_2_class_enc.values())

    index_text = [gen_text(sid, text_preps) for sid in index_text_sids]

    return index_text, torch.tensor(index_text_class_encs)

def spawn_dataloader(
    index_data:     Dict[str, List[Union[int, str]]],
    text_preps:     List[List[str]],
    config,
    shuffle:        bool,
    drop_last:      bool,
    img_pp:         Callable,
    use_dv_sampler: bool,
) -> Tuple[DataLoader, float | None]:
    """

    Args:
    - index_data -------- Data indexes: Class encodings, relative filepaths to images, species ID's, position data, sex
                          data (note: cleanup on aisle 5 needed with the assembly of class_encs at runtime, should be 
                          in split gen)
    - text_preps -------- List of text prepending selections that are randomly picked from to assemble train texts
    - config ------------ contains 1. batch_size (int), 2. cached_imgs (bool) ~ whether to cache images in memory, 
                          3. n_workers (int) ~ Parallelism, 4. prefetch_factor (int) ~ How many batches each worker 
                          will load in advance; Higher prefetch_factor increases throughput, higher RAM cost; Only 
                          takes effect when n_workers > 0
    - shuffle ----------- Whether to shuffle samples between cycles
    - drop_last --------- Whether to drop partial batch at the end of epoch (only need this arg for train)
    - img_pp ------------ The image preprocessor          
    """

    dataset = ImageTextDataset(index_data, text_preps, img_pp, config)

    bs_local = config.batch_size // dist.get_world_size()

    # conditionally use D/V sampling only when a config flag is set (so eval still uses the normal behaviour)
    if use_dv_sampler:
        batch_sampler = DorsalVentralBatchSampler(
            index_pos =dataset.index_pos,
            batch_size=config.batch_size,
            seed      =config.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler     =batch_sampler,
            num_workers       =config.n_workers,
            pin_memory        =True,  # (True) speeds up host --> GPU copies, higher RAM cost
            prefetch_factor   =config.prefetch_factor,
            collate_fn        =collate_fn,
            persistent_workers=True,
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
            persistent_workers=True,
            sampler           =sampler,
        )

    return dataloader, dataset.time_cache

def gen_text(sid, combo_temp, pos=None, sex=None, sids2commons=None):

    genus, species_epithet = sid.split("_", 1)
    text_sci = f"{genus} {species_epithet}"
    text_tax = "animalia arthropoda insecta lepidoptera nymphalidae " + text_sci
    if sids2commons is not None:
        text_com = sids2commons[sid]
    else:
        text_com = None

    prompt = ""

    for combo_seg in reversed(combo_temp):  # iterate through combo segments in reversed order

        seg = random.choice(combo_seg)

        if "$COM$" in seg:
            if text_com is not None:
                seg = seg.replace("$COM$", text_com)
            else:
                if random.choice((True, False)):
                    seg = seg.replace("$COM$", "$SCI$")
                else:
                    seg = seg.replace("$COM$", "$TAX$")
        if "$SCI$" in seg:
            seg = seg.replace("$SCI$", text_sci)
        if "$TAX$" in seg:
            seg = seg.replace("$TAX$", text_tax)

        if "$SEX$" in seg:
            if sex is None:
                seg = seg.replace("$SEX$", "")
            else:
                seg = seg.replace("$SEX$", f"{sex} ")

        if "$POS$" in seg:
            seg = seg.replace("$POS$", f", {pos} view")

        if "$AAN$" in seg:
            if prompt[0] in ["a", "e", "i", "o", "u"]:
                seg = seg.replace("$AAN$", "an")
            else:
                seg = seg.replace("$AAN$", "a")

        prompt = seg + prompt  # select random prepending from prepending-category, prepend to text

    return prompt

def gbif_common_name(scientific_name: str, lang: str = "eng") -> str | None:
    m = requests.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": scientific_name},
        timeout=10
    ).json()
    key = m.get("usageKey") or m.get("speciesKey") or m.get("acceptedUsageKey")
    if not key:
        return None

    data = requests.get(
        f"https://api.gbif.org/v1/species/{key}/vernacularNames",
        timeout=10
    ).json()
    items = [x for x in data.get("results", []) if (x.get("language") or "").lower() == lang.lower()]
    return (items[0].get("vernacularName").strip() or None) if items else None

def sid_to_genus(sid: str) -> str:
    return sid.split("_")[0]