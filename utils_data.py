import torch  # type: ignore[import]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import]
from PIL import Image  # type: ignore[import]
import tqdm  # type: ignore[import]
import numpy as np  # type: ignore[import]
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import requests  # type: ignore[import]
from typing import Dict, List, Callable, Union, Tuple

from utils import paths, load_pickle, load_split

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
    """
    def __init__(
        self, 
        index_pos:  List[str], 
        batch_size: int, 
    ) -> None:
        
        self.batch_size = batch_size

        self.idxs_dorsal  = [i for i, p in enumerate(index_pos) if p == "dorsal"]
        self.idxs_ventral = [i for i, p in enumerate(index_pos) if p == "ventral"]

        self.n_batches = len(self.idxs_dorsal)  // self.batch_size + len(self.idxs_ventral) // self.batch_size

    def __iter__(self):
        # freshly shuffled copies each epoch
        idxs_dorsal_copy  = self.idxs_dorsal.copy()
        idxs_ventral_copy = self.idxs_ventral.copy()
        random.shuffle(idxs_dorsal_copy)
        random.shuffle(idxs_ventral_copy)

        n_batches_dorsal  = len(idxs_dorsal_copy) // self.batch_size
        n_batches_ventral = len(idxs_ventral_copy) // self.batch_size

        pool_tags = (["dorsal"] * n_batches_dorsal) + (["ventral"] * n_batches_ventral)
        random.shuffle(pool_tags)

        # batch-start indexes for slicing into pools
        ib_dorsal  = 0
        ib_ventral = 0

        for tag in pool_tags:
            if tag == "dorsal":
                batch = idxs_dorsal_copy[ib_dorsal:ib_dorsal+self.batch_size]
                ib_dorsal += self.batch_size
            elif tag == "ventral":
                batch = idxs_ventral_copy[ib_ventral:ib_ventral+self.batch_size]
                ib_ventral += self.batch_size

            yield batch

    def __len__(self) -> int:
        return self.n_batches

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

    index_txts_sids      = list(sid_2_class_enc.keys())
    index_txts_class_enc = list(sid_2_class_enc.values())

    index_txts = [gen_text(sid, text_preps) for sid in index_txts_sids]

    return index_txts, index_txts_class_enc

def spawn_dataloader(
    index_data:      Dict[str, List[Union[int, str]]],
    text_preps:      List[List[str]],
    config,
    shuffle:         bool,
    drop_last:       bool,
    img_pp:          Callable,
    use_dv_sampler:  bool,
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

    dataset = ImageTextDataset(
        index_data,
        text_preps,
        img_pp,
        config,
    )

    # conditionally use D/V sampling only when a config flag is set (so eval still uses the normal behaviour)
    if use_dv_sampler:
        batch_sampler = DorsalVentralBatchSampler(dataset.index_pos, config.batch_size)
        dataloader = DataLoader(
            dataset,
            batch_sampler     =batch_sampler,
            num_workers       =config.n_workers,
            pin_memory        =False,  # (True) speeds up host --> GPU copies, higher RAM cost
            prefetch_factor   =config.prefetch_factor,
            collate_fn        =collate_fn,
            persistent_workers=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size        =config.batch_size,
            shuffle           =shuffle,
            num_workers       =config.n_workers,
            pin_memory        =False,  # (True) speeds up host --> GPU copies, higher RAM cost
            prefetch_factor   =config.prefetch_factor,
            collate_fn        =collate_fn,
            drop_last         =drop_last,
            persistent_workers=True,
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
