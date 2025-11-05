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
from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree, Clade  # type: ignore[import]
from itertools import combinations
from typing import Dict, List

from utils import paths, load_pickle, load_split

import pdb


@dataclass
class Split:
    data_indexes: dict
    id_eval_nshot: dict
    class_counts_train: np.ndarray


class ImageTextDataset(Dataset):
    """
    PyTorch requirements for custom Dataset:
    - Inheritance from torch.utils.data.Dataset
    - Implementations of:
        - __len__(self) --> int
        - __getitem__(self, idx) --> sample
    - Everything else is up to you!
    """

    def __init__(
            self, 
            index_class_encs, 
            index_rfpaths, 
            index_sids,
            index_pos,
            index_sex,
            text_preps,
            img_pp, 
            cached_imgs,
            n_workers,
        ):
        
        self.index_class_encs = index_class_encs
        self.index_rfpaths    = index_rfpaths
        self.index_sids       = index_sids
        self.index_pos        = index_pos
        self.index_sex        = index_sex
        self.text_preps       = text_preps
        self.img_pp           = img_pp
        self.cached_imgs      = cached_imgs

        self.n_samples = len(self.index_class_encs)

        self.time_cache = None  # need to pass this var to metadata save
        if self.cached_imgs in ("pl", "pp"):
            time_start = time.time()

            def load_pp_img(rfpath):
                img   = Image.open(paths["nymph_imgs"] / rfpath).convert("RGB")
                return img if cached_imgs == "pl" else img_pp(img)

            # load all images into memory (pl: as PIL images; pp: as preprocessed tensors)
            self.imgs_mem = []
            with ThreadPoolExecutor(max_workers=n_workers) as exe:
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

        This method gets called in a background thread/process to prepare batch N+1 while GPU and main process are processing batch N.

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

        return img_t, class_enc, text, rank_keys

def collate_fn(batch):
    """
    collate_fn takes list of individual samples from Dataset and merges them into a single batch
    augmentation can be done here methinks
    """
    imgs_b, class_encs_b, texts_b, rank_keys_b = zip(*batch)

    imgs_b = torch.stack(imgs_b, dim=0)  # --- Tensor(B, C, H, W)

    return imgs_b, class_encs_b, list(texts_b), torch.Tensor(rank_keys_b)

def assemble_indexes(data_index):
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

    return index_class_encs, index_rfpaths, index_sids, sid_2_class_enc, index_pos, index_sex

def spawn_indexes(split_name, split_type):
    """

    Args:
    - split_type --- [str] --- "train" / "id_val" / "id_test" / "ood_val" / "ood_test"
    - split_name --- [str] --- Name of the split directory e.g. "A" / "B" / etc.
    """
    split      = load_split(split_name)
    data_index = split.data_indexes[split_type]

    return assemble_indexes(data_index)

def spawn_indexes_txts(sid_2_class_enc, text_preps):
    """
    Think this is still needed but only for eval

    Note: This was split apart from the spawn_indexes() logic for ease of setting up the mixed text-types experiment
    
    Args:
    - sid_2_class_enc --- [dict(sid --> class enc)] --- generated by spawn_indexes()
    """

    index_txts_sids      = list(sid_2_class_enc.keys())
    index_txts_class_enc = np.array(list(sid_2_class_enc.values()))

    index_txts = [gen_text(sid, text_preps) for sid in index_txts_sids]

    return index_txts, index_txts_class_enc

def spawn_dataloader(
        index_class_encs,
        index_rfpaths,
        index_sids,
        index_pos,
        index_sex,
        text_preps,
        batch_size,
        shuffle,
        drop_last,
        img_pp,
        cached_imgs,
        n_workers,
        prefetch_factor,
    ):
    """

    Args:
    - index_class_encs ------ [list(int)] --------- Class encodings (data index)
    - index_rfpaths --------- [list(int)] --------- Relative filepaths to images (data index)
    - index_sids ------------ [list(str)] --------- Species Identifiers (data index)
    - index_pos ------------- [list(str)] --------- 
    - index_sex ------------- [list(str)] --------- 
    - text_preps ------------ [list(list(str))] --- List of text prepending selections that are randomly picked from to assemble train texts
    - batch_size ------------ [int] --------------- Batch size
    - shuffle --------------- [bool] -------------- Whether to shuffle samples between cycles
    - drop_last ------------- [bool] -------------- Whether to drop partial batch at the end of epoch (only need this arg for train)
    - img_pp ---------------- [callable] ---------- The image preprocessor
    - cached_imgs ----------- [bool] -------------- Whether to cache images in memory
    - n_workers ------------- [int] --------------- Parallelism
    - prefetch_factor ------- [int] --------------- How many batches each worker will load in advance;
                                                     Higher prefetch_factor increases throughput, higher RAM cost;
                                                     Only takes effect when n_workers > 0
    """

    dataset = ImageTextDataset(
        index_class_encs,
        index_rfpaths,
        index_sids,
        index_pos,
        index_sex,
        text_preps,
        img_pp,
        cached_imgs,
        n_workers,
    )

    dataloader = DataLoader(
        dataset,
        batch_size        =batch_size,
        shuffle           =shuffle,
        num_workers       =n_workers,
        pin_memory        =False,  # (True) speeds up host --> GPU copies, higher RAM cost
        prefetch_factor   =prefetch_factor,
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

class PhyloVCV:

    def __init__(self):

        self.tree: Tree = Phylo.read(paths["nymph_phylo_tree"], "newick")

        root:               Clade              = self.tree.root
        self._depth:        Dict[Clade, float] = {root: 0.0}
        self._sid_to_clade: Dict[str, Clade]   = {}

        # populate _depth and _sid_to_clade
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.clades:
                self._depth[child] = self._depth[node] + child.branch_length
                stack.append(child)
            if node.is_terminal():
                self._sid_to_clade[node.name] = node

        self._sids:       List[str]      = sorted(list(self._sid_to_clade.keys()))
        self._sid_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(self._sids)}

        vcv       = self.build_vcv_matrix()
        self.corr = vcv / vcv[0, 0]  # assuming all tips are at the same depth, all elements along the diagonal (variances) are of the same value and dividing by that value equates to the so called correlation matrix

    def build_vcv_matrix(self) -> np.ndarray:

        n_sids = len(self._sids)
        vcv    = np.zeros((n_sids, n_sids), dtype=np.float64)

        clade_to_tip_idxs: Dict[Clade, np.ndarray] = {}
        stack = [(self.tree.root, False)]

        while stack:
            node, seen = stack.pop()
            if not seen:
                stack.append((node, True))
                stack.extend((child, False) for child in node.clades)
                continue

            if node.is_terminal():
                idx = self._sid_to_idx[node.name]
                vcv[idx, idx] = self._depth[node]  # variance along diagonal
                clade_to_tip_idxs[node] = np.array([idx], dtype=np.intp)
                continue

            idxs_child_tips = [clade_to_tip_idxs[child] for child in node.clades if child in clade_to_tip_idxs]

            depth = self._depth[node]
            for tips_u, tips_v in combinations(idxs_child_tips, 2):
                vcv[tips_u[:, None], tips_v] += depth
                vcv[tips_v[:, None], tips_u] += depth

            clade_to_tip_idxs[node] = np.concatenate(idxs_child_tips, dtype=np.intp) 

        return vcv
    
    def get_targ(self, sid_a: str, sid_b: str) -> float:
        i = self._sid_to_idx[sid_a]
        j = self._sid_to_idx[sid_b]
        return float(np.clip(self.corr[i, j], 0.0, 1.0))

    def get_sids(self) -> list[str]:
        return self._sids
