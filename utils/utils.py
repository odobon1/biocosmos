import pickle
from pathlib import Path
import random
import os
import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
import json
from typing import List, Any, Dict, Optional

import pdb


# CLUSTER = "pace"
CLUSTER = "hpg"


if CLUSTER == "pace":

    dpath_root = Path(os.getcwd())

    paths = {
        "root":    dpath_root,
        "vlm4bio": dpath_root / "VLM4Bio/datasets"
    }
elif CLUSTER == "hpg":

    dpath_group   = Path("/lustre/blue2/arthur.porto-biocosmos")
    dpath_root    = Path(os.getcwd())
    dpath_data    = dpath_group / "data"
    dpath_nymph   = dpath_data / "datasets/nymphalidae_whole_specimen-v250613"
    dpath_vlm4bio = dpath_data / "datasets/VLM4Bio"

    paths = {
        "data":             dpath_data,
        "hf_cache":         dpath_data / "cache/huggingface/hub",
        "group":            dpath_group,
        "root":             dpath_root,
        "config":           dpath_root / "config",
        "metadata":         dpath_root / "metadata",
        "artifacts":        dpath_root / "artifacts",
        "nymph_imgs":       dpath_nymph / "images",
        "nymph_metadata":   dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv",
        "nymph_phylo_tree": dpath_nymph / "metadata/tree_nymphalidae_chazot2021_all.tree",
        "vlm4bio":          dpath_vlm4bio,
    }

def seed_libs(seed):
    random.seed(seed)
    os.putenv("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True  # (True) trades speed for reproducibility (default is False)
    torch.backends.cudnn.benchmark     = False  # (False) trades speed for reproducibility (default is False)

def save_json(data, fpath):
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
    return data

def save_pickle(obj, picklepath):
    with open(picklepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(picklepath):
    with open(picklepath, "rb") as f:
        obj = pickle.load(f)
    return obj

def load_split(split_name):
    split = load_pickle(paths["metadata"] / f"splits/{split_name}/split.pkl")
    return split

def get_text_preps(text_preps_type):

    TEXT_PREPS_MIXED = [
        [
            "",
            "a photo of ",  # BioCLIP-style prepending
            "a photo of a ",  # OpenAI CLIP-style prepending
        ],
    ]

    TEXT_PREPS_BIOCLIP_SCI = [["a photo of $SCI$"]]  # scientific name, BioCLIP-style prepending

    COMBO_TEMPS_TRAIN = [
        [
            "",
            "a photo of ",
        ],
        [
            "",
            "$AAN$ ",
        ],
        [
            "",
            "$SEX$",
        ],
        [
            "$SCI$",
            "$TAX$",
            "$COM$",
        ],
        [
            "",
            " butterfly",
        ],
        [
            "",
            "$POS$",
        ],
    ]

    if text_preps_type == "combo_temps":
        return COMBO_TEMPS_TRAIN
    if text_preps_type == "mixed":
        return TEXT_PREPS_MIXED
    elif text_preps_type == "bioclip_sci":
        return TEXT_PREPS_BIOCLIP_SCI
    
class RunningMean:
    """
    Track running mean via Welford's algorithm
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n

    def value(self):
        return self.mean
    
def shuffle_list(input: List[Any], seed: int) -> List[int]:
    rng        = random.Random(seed)
    input_shuf = input.copy()
    rng.shuffle(input_shuf)
    return input_shuf

class PrintLog:

    logging = False
    log_batch = None
    log_epoch = None
    log_init  = None

    @staticmethod
    def create_logs(dpath_logs):
        PrintLog.logging = True
        PrintLog.log_batch = open(dpath_logs / "batch.log", "a", buffering=1)
        PrintLog.log_epoch = open(dpath_logs / "epoch.log", "a", buffering=1)
        PrintLog.log_init  = open(dpath_logs / "init.log",  "a", buffering=1)

    @staticmethod
    def epoch_header(idx_epoch):
        header_epoch = f" Epoch {idx_epoch} "
        header_epoch = (
            f"{header_epoch:#^{75}}"
            f"\n"
        )
        print(header_epoch)
        if PrintLog.logging:
            PrintLog.log_batch.write(header_epoch)
            PrintLog.log_epoch.write(header_epoch)

    @staticmethod
    def train_header(lr):
        header_train = (
            f"{' Train ':=^{75}}\n"
            f"LR --- {lr:.2e}\n"
            f"\n"
        )
        print(header_train)
        if PrintLog.logging:
            PrintLog.log_epoch.write(header_train)

    @staticmethod
    def epoch(time_train, time_train_avg, time_val, time_val_avg, loss_train_avg, scores_val, loss_val_best):

        epoch_info = (
            f"Train Loss ------ {loss_train_avg:.4f}\n"
            f"Val Loss -------- {scores_val['comp_loss']:.4f} (Best: {loss_val_best:.4f})\n"
            f"\n"
            f"{' Elapsed Time ':=^{75}}\n"
            f"Train -------- {time_train:.2f} s (avg: {time_train_avg:.2f} s)\n"
            f"Validation --- {time_val:.2f} s (avg: {time_val_avg:.2f} s)\n"
            f"\n"
        )
        print(epoch_info)
        if PrintLog.logging:
            PrintLog.log_epoch.write(epoch_info)

    @staticmethod
    def batch(idx_batch, lr, loss_batch):
        batch_str = f"batch {idx_batch}:"
        if PrintLog.logging:
            PrintLog.log_batch.write(
                f"{batch_str:<10} "
                f"lr={lr:.3e} "
                f"loss={loss_batch:.3e} "
                f"\n"
            )

    @staticmethod
    def eval(
        scores_eval: Dict[str, float],
        best_comp_map: float = None,
        best_img2img_map: float = None,
        header: Optional[str] = None,
    ) -> None:
        
        if header is not None:
            header = f" {header} "
            header = (
                f"{header:#^{75}}"
                f"\n"
            )
            print(header)
            if PrintLog.logging:
                PrintLog.log_epoch.write(header)

        if best_comp_map is not None:
            best_comp_map_str = f" (best: {best_comp_map:.4f})"
            best_img2img_map_str = f" (best: {best_img2img_map:.4f})"
        else:
            best_comp_map_str = ""
            best_img2img_map_str = ""

        header = " Eval "

        eval_printout = (
            f"{header:=^{75}}\n"
            f"ID img2txt mAP ---- {scores_eval['id_img2txt_map']:.4f}\n"
            f"ID img2img mAP ---- {scores_eval['id_img2img_map']:.4f}\n"
            f"ID txt2img mAP ---- {scores_eval['id_txt2img_map']:.4f}\n"
            f"OOD img2txt mAP --- {scores_eval['ood_img2txt_map']:.4f}\n"
            f"OOD img2img mAP --- {scores_eval['ood_img2img_map']:.4f}\n"
            f"OOD txt2img mAP --- {scores_eval['ood_txt2img_map']:.4f}\n"
            f"{'':-^{75}}\n"
            f"Composite mAP --- {scores_eval['comp_map']:.4f}{best_comp_map_str}\n"
            f"img2img mAP ----- {scores_eval['img2img_map']:.4f}{best_img2img_map_str}\n"
            f"{'':-^{75}}\n"
            f"ID Loss ----- {scores_eval['id_loss']:.4f}\n"
            f"OOD Loss ---- {scores_eval['ood_loss']:.4f}\n"
            f"Comp Loss --- {scores_eval['comp_loss']:.4f}\n"
            f"\n"
        )
        print(eval_printout)

        if PrintLog.logging:
            PrintLog.log_epoch.write(eval_printout)

    @staticmethod
    def init_train(cfg_train):
        lines: list[str] = [
            "",
            f"Study ------------- {cfg_train.study_name}",
            f"Experiment -------- {cfg_train.experiment_name}",
            f"Seed -------------- {cfg_train.seed}",
            f"Trial File Path --- {cfg_train.rdpath_trial}",
            f"Split ------------- {cfg_train.split_name}",
            "",
            f"Batch Size ---- {cfg_train.batch_size}",
            f"DV Batching --- {cfg_train.dv_batching}",
            "",
            f"=== Architecture ===",
            f"Model Type --- {cfg_train.arch['model_type']}",
            f"Non-Causal --- {cfg_train.arch['non_causal']}",
            "",
        ]

        # primary loss block
        lines.extend(PrintLog._format_loss_block(cfg_train.loss))

        # secondary loss block (if enabled)
        if cfg_train.loss2["mix"] != 0.0:
            lines.extend(PrintLog._format_loss_block(cfg_train.loss2, secondary=True))

        # optimization block
        lines.extend([
            f"=== Optimization ===",
            f"LR Init ---- {cfg_train.opt['lr']['init']}",
            f"LR Sched --- {cfg_train.opt['lr']['sched']}",
            f"L2 Reg ----- {cfg_train.opt['l2reg']}",
            f"β1 --------- {cfg_train.opt['beta1']}",
            f"β2 --------- {cfg_train.opt['beta2']}",
            f"ε ---------- {cfg_train.opt['eps']}",
            "",
        ])

        # hardware block
        lines.extend(PrintLog._format_hw_block(cfg_train))

        print(*lines, sep="\n")
        if PrintLog.logging:
            PrintLog.log_init.write("\n".join(lines) + "\n")

    @staticmethod
    def init_eval(cfg_eval):
        lines: list[str] = [
            "",
            f"Trial (Loaded) --- {cfg_eval.rdpath_trial}{'' if cfg_eval.rdpath_trial is None else ' (' + cfg_eval.save_crit + ')'}",
            "",
            f"Model Type ------- {cfg_eval.arch['model_type']}",
            f"Loss Type -------- {cfg_eval.loss['type']}",
            f"Split ------------ {cfg_eval.split_name}",
            "",
            f"Batch Size ------- {cfg_eval.batch_size}",
            "",
        ]

        lines.extend(PrintLog._format_loss_block(cfg_eval.loss))

        mix = cfg_eval.loss2.get("mix", 0.0)
        if mix != 0.0:
            lines.extend(PrintLog._format_loss_block(cfg_eval.loss2, secondary=True))

        lines.extend(PrintLog._format_hw_block(cfg_eval))

        print(*lines, sep="\n")

    @staticmethod
    def _format_hw_block(cfg) -> list[str]:
        return [
            f"=== Hardware ===",
            f"Num. GPUs --------- {cfg.n_gpus}",
            f"Num. CPUs --------- {cfg.n_cpus}",
            f"RAM --------------- {cfg.ram} GB",
            f"Num. Workers ------ {cfg.n_workers}",
            f"Prefetch Factor --- {cfg.prefetch_factor}",
            f"Device ------------ {cfg.device}",
            "",
        ]

    @staticmethod
    def _format_loss_block(
        loss:      dict, 
        secondary: bool = False
    ) -> list[str]:

        lines: list[str] = []

        if not secondary:
            lines.append("=== Loss (Primary) ===")
        else:
            lines.append("=== Loss (Secondary) ===")
            lines.append(f"Mix --------------- {loss['mix']}")

        lines.append(f"Type -------------- {loss['type']}")
        lines.append(f"Similarity Type --- {loss['sim']}")
        lines.append(f"Target Type ------- {loss['targ']}")

        wting = loss["wting"]
        if wting and "class_weighting" in loss['cfg']:
            cw = loss["cfg"]["class_weighting"]
            lines.append(f"Class Weighting --- {cw['type']}")
            if "if_gamma" in cw:
                lines.append(f"  if_gamma ---------- {cw['if_gamma']}")
            if "cb_beta" in cw:
                lines.append(f"  cb_beta ----------- {cw['cb_beta']}")
            if "cp_type" in cw:
                lines.append(f"  cp_type ----------- {cw['cp_type']}")
        else:
            lines.append("Class Weighting ---- disabled")

        focal = loss["focal"]
        if focal and "focal" in loss['cfg']:
            cfg_focal = loss['cfg']["focal"]
            lines.append("Focal ------------- enabled")
            if "gamma" in cfg_focal:
                lines.append(f"  gamma ------------- {cfg_focal['gamma']}")
            if "comp_type" in cfg_focal:
                lines.append(f"  comp_type --------- {cfg_focal['comp_type']}")
        else:
            lines.append("Focal ------------- disabled")

        if "dyn_smr" in loss['cfg']:
            lines.append(f"Dyn SMR ----------- {loss['cfg']['dyn_smr']}")

        cfg_logits = loss['cfg']['logits']
        lines_logits: list[str] = [
            "Logits",
            f"  Scale Init -------- {cfg_logits['scale_init']}",
            f"  Bias Init --------- {cfg_logits['bias_init']}",
            f"  Freeze Scale ------ {cfg_logits['freeze_scale']}",
            f"  Freeze Bias ------- {cfg_logits['freeze_bias']}",
            "",
        ]
        lines.extend(lines_logits)

        return lines

    @staticmethod
    def close_logs():
        PrintLog.log_batch.close()
        PrintLog.log_epoch.close()
        PrintLog.log_init.close()