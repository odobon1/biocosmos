import pickle
from pathlib import Path
import random
import os
import numpy as np
import torch
import json
from typing import List, Any, Dict, Optional
import math
import time
from contextlib import contextmanager

from utils.text import get_text_template as get_dataset_text_template
from utils.ddp import rank0

import pdb


# CLUSTER = "pace"  # PACE
CLUSTER = "hpg"  # HiPerGator


if CLUSTER == "pace":

    dpath_root = Path(os.getcwd())
    dpath_haag = Path("/storage/ice-shared/cs8903onl")
    dpath_lepid = dpath_haag / "butterflies"
    dpath_hf_cache = dpath_haag / "huggingface_cache"

    paths = {
        "root": dpath_root,
        "hf_cache": dpath_hf_cache,
        "config": dpath_root / "config",
        "artifacts": dpath_root / "artifacts",
        "data": {
            "cub": dpath_root / "data/cub",
        },
        "imgs":{
            "bryo": dpath_haag / "bryo",
            "cub": dpath_root / "data/cub/CUB_200_2011/images",
            "lepid": dpath_lepid / "images",
        },
        "preproc": {
            "bryo": dpath_root / "preprocessing/bryo",
            "cub": dpath_root / "preprocessing/cub",
            "lepid": dpath_root / "preprocessing/lepid",
        },
        "metadata": {
            "bryo": dpath_root / "metadata/bryo",
            "cub": dpath_root / "metadata/cub",
            "lepid": dpath_root / "metadata/lepid",
        },
        "raw_tree": {
            "bryo": dpath_root / "data/bryo/SI_Fig1(BIG).newick",
            "cub": dpath_root / "data/cub/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy",
            "lepid": dpath_root / "data/lepid/tree_renamed_full.tre",
            "nymph": dpath_root / "data/nymph/tree_nymphalidae_chazot2021_all.tree",
        },
        "csv": {
            "lepid": {
                "imgs": dpath_lepid / "metadata/data_meta-clean_rot_512-butterflies_whole_specimen-v2025_05_07.csv",
                "tax": dpath_lepid / "metadata/data_tree_meta.csv",
            },
            "nymph": {
                "imgs": None,  # rip
                "tax": None,  # rip
            },
        },
    }
elif CLUSTER == "hpg":

    dpath_root = Path(os.getcwd())
    dpath_group = Path("/lustre/blue2/arthur.porto")
    dpath_data = dpath_group / "data"
    dpath_nymph = dpath_data / "datasets/nymphalidae_whole_specimen-v250613"
    dpath_lepid = dpath_data / "datasets/butterflies_whole_specimen-clean_rot_512-v2025_05_07"
    dpath_hf_cache = dpath_data / "cache/huggingface/hub"

    paths = {
        "root": dpath_root,
        "hf_cache": dpath_hf_cache,
        "config": dpath_root / "config",
        "artifacts": dpath_root / "artifacts",
        "data": {
            "cub": dpath_root / "data/cub",
        },
        "imgs":{
            "bryo": dpath_group / "odobon3.gatech/bryo",
            "cub": dpath_root / "data/cub/CUB_200_2011/images",
            "lepid": dpath_lepid / "images",
            "nymph": dpath_nymph / "images",
        },
        "preproc": {
            "bryo": dpath_root / "preprocessing/bryo",
            "cub": dpath_root / "preprocessing/cub",
            "lepid": dpath_root / "preprocessing/lepid",
            "nymph": dpath_root / "preprocessing/nymph",
        },
        "metadata": {
            "bryo": dpath_root / "metadata/bryo",
            "cub": dpath_root / "metadata/cub",
            "lepid": dpath_root / "metadata/lepid",
            "nymph": dpath_root / "metadata/nymph",
        },
        "raw_tree": {
            "bryo": dpath_root / "data/bryo/SI_Fig1(BIG).newick",
            "cub": dpath_root / "data/cub/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy",
            "lepid": dpath_root / "data/lepid/tree_renamed_full.tre",
            "nymph": dpath_root / "data/nymph/tree_nymphalidae_chazot2021_all.tree",
        },
        "csv": {
            "lepid": {
                "imgs": dpath_lepid / "metadata/data_meta-clean_rot_512-butterflies_whole_specimen-v2025_05_07.csv",
                "tax": dpath_lepid / "metadata/data_tree_meta.csv",
            },
            "nymph": {
                "imgs": dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv",
                "tax": dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv",
            },
        },
    }

def seed_libs(seed, seed_torch=True):
    random.seed(seed)
    os.putenv("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # (True) trades speed for reproducibility (default is False)
        torch.backends.cudnn.benchmark = False  # (False) trades speed for reproducibility (default is False)

def save_json(data, fpath):
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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

DATASET_ALIAS2NAME = {
    "bryo": "Bryozoa",
    "cub": "CUB",
    "lepid": "Lepidoptera",
    "nymph": "Nymphalidae",
}

def load_split(dataset, split):
    fpath_split = paths["metadata"][dataset] / f"splits/{split}/split.pkl"
    split = load_pickle(fpath_split)
    return split

def get_text_template(text_template_type, dataset=None):
    return get_dataset_text_template(text_template_type, dataset=dataset)
    

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


class TimeTracker:
    """
    Per-trial timing registry of running MEANS over repeated, homogeneous events (`train` per epoch;
    `eval`/`viz_compute` per eval). A bucket's total time is mean*n; the unattributed remainder of the
    trial wall-clock (checkpoint I/O, sync, etc.) is reported separately as "other" (trial - attributed()).
    """

    MEANS = ("train", "eval", "viz_compute")

    def __init__(self):
        self._means = {name: RunningMean() for name in self.MEANS}

    def add(self, name, value):
        self._means[name].update(value)

    @contextmanager
    def measure(self, name):
        """Time the with-block into bucket `name` (running mean via add()). Records only on a clean exit --
        if the block raises, the elapsed time is NOT folded in (so a failed eval doesn't pollute the mean)
        and the exception propagates."""
        timer = Timer()
        timer.start()
        yield
        timer.stop()
        self.add(name, timer.get_elapsed_time())

    def mean(self, name):
        rm = self._means[name]
        return rm.value() if rm.n > 0 else None

    def n(self, name):
        return self._means[name].n

    def total(self, name):
        rm = self._means[name]
        return rm.mean * rm.n

    def attributed(self):
        return sum(self.total(name) for name in self.MEANS)

    def state_dict(self):
        return {"means": {name: (rm.n, rm.mean) for name, rm in self._means.items()}}

    def load_state_dict(self, state):
        for name, (n, mean) in state["means"].items():
            self._means[name].n = n
            self._means[name].mean = mean


def model_grad_l2_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)
    
def shuffle_list(input: List[Any], seed: int) -> List[int]:
    rng = random.Random(seed)
    input_shuf = input.copy()
    rng.shuffle(input_shuf)
    return input_shuf


class PrintLog:

    logging = False
    log_batch_general = None
    log_batch_grad_norm = None
    log_batch_temp_bias = None
    log_epoch = None
    log_eval = None
    log_init = None
    log_text_train = None
    log_text_eval = None

    @staticmethod
    @rank0
    def create_logs(dpath_logs):
        PrintLog.logging = True
        dpath_batch_logs = dpath_logs / "batch"
        dpath_batch_logs.mkdir(parents=True, exist_ok=True)
        PrintLog.log_batch_general = open(dpath_batch_logs / "general.log", "a", buffering=1)
        PrintLog.log_batch_grad_norm = open(dpath_batch_logs / "grad_norm.log", "a", buffering=1)
        PrintLog.log_batch_temp_bias = open(dpath_batch_logs / "temp_bias.log", "a", buffering=1)
        PrintLog.log_epoch = open(dpath_logs / "epoch.log", "a", buffering=1)
        PrintLog.log_eval = open(dpath_logs / "eval.log", "a", buffering=1)
        PrintLog.log_init = open(dpath_logs / "init.log", "a", buffering=1)
        PrintLog.log_text_train = open(dpath_logs / "text_train.log", "a", buffering=1)
        PrintLog.log_text_eval = open(dpath_logs / "text_eval.log", "a", buffering=1)

    @staticmethod
    @rank0
    def texts(texts_sb):
        if PrintLog.logging:
            text_printout = "\n".join(texts_sb)
            PrintLog.log_text_train.write(text_printout)

    @staticmethod
    @rank0
    def texts_eval(eval_pipe):
        if PrintLog.logging:
            texts_by_partition = eval_pipe.get_eval_texts()
            lines = []
            for partition, texts in texts_by_partition.items():
                lines.append(f"[{partition}]")
                lines.extend(texts)
                lines.append("")
            PrintLog.log_text_eval.write("\n".join(lines))
            PrintLog.wrote_text_eval = True

    @staticmethod
    def _make_epoch_header(idx_epoch, n_epochs, width=75):
        return f"{f' Epoch {idx_epoch}/{n_epochs} ':#^{width}}"

    @staticmethod
    @rank0
    def batch_logs_epoch_header(idx_epoch, n_epochs):
        if PrintLog.logging:
            header_epoch = PrintLog._make_epoch_header(idx_epoch, n_epochs) + "\n"
            PrintLog.log_batch_general.write(header_epoch)
            PrintLog.log_batch_grad_norm.write(header_epoch)
            PrintLog.log_batch_temp_bias.write(header_epoch)

    @staticmethod
    @rank0
    def epoch(time_train, time_train_avg, loss_train_avg, loss_train_raw_avg, n_samps_seen, idx_epoch, n_epochs):

        SECTION_WIDTH = 66

        lines_epoch = [
            PrintLog._make_epoch_header(idx_epoch, n_epochs, width=SECTION_WIDTH),
            PrintLog._block_metric_lines((
                ("Loss", f"{loss_train_avg:.3e}"),
                ("Raw Loss", f"{loss_train_raw_avg:.3e}"),
                ("Samples Seen", f"{n_samps_seen:,}"),
                ("Time", f"{time_train:.2f} s (avg: {time_train_avg:.2f} s)"),
            )),
            f"{'':#^{SECTION_WIDTH}}",
            "",
        ]
        print("\n".join(lines_epoch))
        if PrintLog.logging:
            PrintLog.log_epoch.write("\n".join(lines_epoch))

    @staticmethod
    @rank0
    def batch(idx_batch, lr, loss_batch, embs_img_b, embs_txt_b, logits, model):

        def tensor_grad_l2_norm(x: torch.Tensor | None) -> float:
            if x is None:
                return float("nan")
            if x.grad is None:
                return float("nan")
            return x.grad.detach().pow(2).sum().sqrt().item()

        def tensor_scalar_item(x) -> float:
            if x is None:
                return float("nan")
            if isinstance(x, torch.Tensor):
                return x.detach().item()
            return float(x)

        logits1 = logits[0]
        logits2 = logits[1]
        if logits2 is None:
            line_logits = f"log={tensor_grad_l2_norm(logits1):.2e}, "
        else:
            line_logits = f"log1={tensor_grad_l2_norm(logits1):.2e}, "
            line_logits += f"log2={tensor_grad_l2_norm(logits2):.2e}, "

        line_grad_norm = (
            f"img={tensor_grad_l2_norm(embs_img_b):.2e}, "
            f"txt={tensor_grad_l2_norm(embs_txt_b):.2e}, "
            f"{line_logits}"
            f"model={model_grad_l2_norm(model):.2e}"
        )

        line_logits_param = ""
        if hasattr(model.module, "logit_scale") and model.module.logit_scale is not None:
            line_logits_param += f"s1={model.module.logit_scale.detach().exp().item():.2e}, "
        if hasattr(model.module, "logit_scale2") and model.module.logit_scale2 is not None:
            line_logits_param += f"s2={model.module.logit_scale2.detach().exp().item():.2e}, "
        if hasattr(model.module, "logit_bias") and model.module.logit_bias is not None:
            line_logits_param += f"b1={tensor_scalar_item(model.module.logit_bias):.2e}, "
        if hasattr(model.module, "logit_bias2") and model.module.logit_bias2 is not None:
            line_logits_param += f"b2={tensor_scalar_item(model.module.logit_bias2):.2e}, "
        line_logits_param = line_logits_param.rstrip(", ")

        batch_str = f"batch {idx_batch}:"
        if PrintLog.logging:
            PrintLog.log_batch_general.write(
                f"{batch_str:<10} "
                f"lr={lr:.2e} "
                f"loss={loss_batch:.2e} "
                f"\n"
            )
            PrintLog.log_batch_grad_norm.write(
                f"{batch_str:<10} "
                f"{line_grad_norm}"
                f"\n"
            )
            PrintLog.log_batch_temp_bias.write(
                f"{batch_str:<10} "
                f"{line_logits_param}"
                f"\n"
            )

    @staticmethod
    @rank0
    def eval(
        eval_metrics: Dict[str, Any],
        eval_pipe,
        header: Optional[str] = None,
        n_samps_seen: Optional[int] = None,
        time_eval: Optional[float] = None,
        time_eval_avg: Optional[float] = None,
    ) -> None:
        
        SECTION_WIDTH = 77

        def _format_composite_block(title: str, map_scores: Dict[str, Any]) -> str:
            pairs = [
                ("All", f"{map_scores['all']:.4f}"),
                ("I2I", f"{map_scores['i2i']:.4f}"),
                *((p.upper(), f"{map_scores[p]:.4f}") for p in partitions),
            ]
            return f"{title:-^{SECTION_WIDTH}}\n" + PrintLog._block_metric_lines(pairs) + "\n"
        
        partitions = eval_pipe.partitions

        lines_comp = f"{' ID/OOD Eval ':=^{SECTION_WIDTH}}\n"

        lines_comp += _format_composite_block(
            " Composite mAP ",
            eval_metrics["scores"]["closed_set"]["standard"]["comp"]["map"],
        )
        if "comp" in eval_metrics["scores"]["full_set"]["standard"]:
            lines_comp += _format_composite_block(
                " Composite Full-Set mAP ",
                eval_metrics["scores"]["full_set"]["standard"]["comp"]["map"],
            )

        lines_comp_macro = _format_composite_block(
            " Composite macro mAP ",
            eval_metrics["scores"]["closed_set"]["per_class"]["comp"]["map"],
        )
        if "comp" in eval_metrics["scores"]["full_set"]["per_class"]:
            lines_comp_macro += _format_composite_block(
                " Composite macro Full-Set mAP ",
                eval_metrics["scores"]["full_set"]["per_class"]["comp"]["map"],
            )
        
        loss_pairs = [
            (partition.upper(), f"{eval_metrics['loss_raw'][partition]:.3e}")
            for partition in partitions
            if eval_metrics["loss_raw"][partition] is not None
        ]
        lines_loss = (f"{' Loss ':-^{SECTION_WIDTH}}\n" + PrintLog._block_metric_lines(loss_pairs) + "\n") if loss_pairs else ""

        lines_info = ""
        info = []
        if n_samps_seen is not None:
            info.append(("Samples Seen", f"{n_samps_seen:,}"))
        if time_eval is not None:
            time_str = f"{time_eval:.2f} s"
            if time_eval_avg is not None:
                time_str += f" (avg: {time_eval_avg:.2f} s)"
            info.append(("Time", time_str))
        if len(info) > 0:
            lines_info += f"{' Info ':=^{SECTION_WIDTH}}\n"
            lines_info += PrintLog._block_metric_lines(info) + "\n"

        eval_header = f" Eval ({header}) " if header is not None else " Eval "
        eval_printout = (
            f"{eval_header:#^{SECTION_WIDTH}}\n"
            f"{lines_comp}"
            f"{lines_comp_macro}"
            f"{lines_loss}"
            f"{lines_info}"
            f"{'':#^{SECTION_WIDTH}}\n"
        )
        print(eval_printout)
        if PrintLog.logging:
            PrintLog.log_eval.write(eval_printout)

    @staticmethod
    @rank0
    def init_train(cfg_train):

        lines = [
            "",
            PrintLog._block_metric_lines((
                ("Campaign", cfg_train.campaign),
                ("Setting", cfg_train.setting),
                ("Dataset", cfg_train.dataset),
                ("Split", cfg_train.split),
                ("Seed", cfg_train.seed),
            )),
            "",
            PrintLog._block_metric_lines((
                ("Sample Volume", f"{cfg_train.sample_volume:,}"),
                ("Checkpoint Every", f"{cfg_train.chkpt_every:,} samples"),
                ("Batch Size", f"{cfg_train.batch_size}"),
                ("DV Batching", f"{cfg_train.dv_batching}"),
            )),
            "",
            "=== Architecture ===",
            PrintLog._block_metric_lines((
                ("Model Type", cfg_train.arch['model_type']),
                ("Non-Causal", cfg_train.arch['non_causal']),
            )),
            "",
        ]

        lines.extend([  # freeze block
            "=== Freeze ===",
            PrintLog._block_metric_lines((
                ("Image", cfg_train.freeze["image"]),
                ("Text", cfg_train.freeze["text"]),
            )),
            "",
        ])

        lines.extend(PrintLog._format_loss_block(cfg_train.loss))  # primary loss block

        if cfg_train.loss2["mix"] != 0.0:
            lines.extend(PrintLog._format_loss_block(cfg_train.loss2, secondary=True))  # secondary loss block (if enabled)

        lines.extend(PrintLog._format_aug_block(cfg_train.aug))  # image augmentation block

        lines.extend([  # text templates block
            "=== Text Templates ===",
            PrintLog._block_metric_lines((
                ("Train", cfg_train.text_template["train"]),
                ("Eval", cfg_train.text_template["eval"]),
            )),
            "",
        ])

        lines.extend([  # optimization block
            "=== Optimization ===",
            "LR",
            PrintLog._block_metric_lines([
                ("- Init",         cfg_train.opt["lr"]["init"]),
                ("- Decay Factor", cfg_train.opt["lr"]["decay_factor"]),
                ("- Warmup",       f"{cfg_train.opt['lr']['warmup']:,}"),
            ]),
            PrintLog._block_metric_lines([
                ("L2 Reg", cfg_train.opt['l2reg']),
                ("β1",     cfg_train.opt['beta1']),
                ("β2",     cfg_train.opt['beta2']),
                ("ε",      cfg_train.opt['eps']),
            ]),
            "",
        ])

        lines.extend(PrintLog._format_hw_block(cfg_train))  # hardware block

        print(*lines, sep="\n")
        if PrintLog.logging:
            PrintLog.log_init.write("\n".join(lines) + "\n")

    @staticmethod
    def _format_hw_block(cfg_train):
        chunk_size = cfg_train.hw.chunk_size
        lines_hw = [
            "=== Hardware ===",
            PrintLog._block_metric_lines((
                ("Num. GPUs", cfg_train.n_gpus),
                ("Num. CPUs", cfg_train.n_cpus),
                ("RAM", f"{cfg_train.ram} GB"),
                ("Num. Workers", cfg_train.n_workers),
                ("Prefetch Factor", cfg_train.prefetch_factor),
                ("Device", cfg_train.device),
            )),
            "Chunk Size",
            PrintLog._block_metric_lines((
                ("- img-to-img mAP", f"{chunk_size['map_img2img']:,}"),
                ("- cross-modal mAP", f"{chunk_size['map_cross_modal']:,}"),
            )),
            "",
        ]
        return lines_hw

    @staticmethod
    def init_eval(cfg_eval):
        lines: list[str] = [
            "",
            f"Checkpoint: {cfg_eval.rdpath_model}/",
            "",
            PrintLog._block_metric_lines((
                ("Dataset", cfg_eval.dataset),
                ("Split", cfg_eval.split),
                ("Eval Type", cfg_eval.eval_type),
            )),
            "",
            f"Batch Size --- {cfg_eval.batch_size}",
            "",
            "=== Architecture ===",
            PrintLog._block_metric_lines((
                ("Model Type", cfg_eval.arch['model_type']),
                ("Non-Causal", cfg_eval.arch['non_causal']),
            )),
            "",
            f"Image Norm --- {cfg_eval.img_norm}",
            "",
            f"Text Template --- {cfg_eval.text_template}",
            "",
        ]

        lines.extend(PrintLog._format_hw_block(cfg_eval))

        print(*lines, sep="\n")

    @staticmethod
    def _format_loss_block(
        cfg_loss: dict, 
        secondary: bool = False
    ) -> list[str]:

        lines = []
        info = []
        if not secondary:
            lines.append("=== Loss (Primary) ===")
        else:
            lines.append("=== Loss (Secondary) ===")
            info.append(("Mix", str(cfg_loss["mix"])))
        
        info.append(("Type", cfg_loss["type"]))
        info.append(("Sim", cfg_loss["sim"]))
        info.append(("Targs", cfg_loss["targ"]))
        lines.append(PrintLog._block_metric_lines(info))

        wting = cfg_loss.get("wting", {}).get("type") is not None
        if wting:
            cw_type = cfg_loss["wting"]["type"]
            lines_cw = [
                "Class Weighting",
                PrintLog._block_metric_lines((
                    ("- Type", cw_type),
                    *((("- gamma", cfg_loss["wting"]["inv_freq"]["gamma"]),) if cw_type == "inv_freq" else ()),
                    *((("- beta", cfg_loss["wting"]["class_bal"]["beta"]),) if cw_type == "class_bal" else ()),
                    ("- cp_type", cfg_loss["wting"]["cp_type"]),
                )),
            ]
            lines.extend(lines_cw)

        focal = cfg_loss.get("focal", {}).get("gamma", 0.0) != 0.0
        if focal:
            lines_focal = [
                "Focal",
                PrintLog._block_metric_lines((
                    ("- gamma", cfg_loss["focal"]["gamma"]),
                    ("- comp_type", cfg_loss["focal"]["comp_type"]),
                )),
            ]
            lines.extend(lines_focal)

        if cfg_loss.get("dsmr", False):
            lines.append("DSMR Enabled")

        cfg_logits = cfg_loss["logits"]
        lines_logits = [
            "Logits",
            PrintLog._block_metric_lines((
                ("- Scale Init", cfg_logits["scale_init"]),
                ("- Bias Init", cfg_logits["bias_init"]),
                ("- Freeze Scale", cfg_logits["freeze_scale"]),
                ("- Freeze Bias", cfg_logits["freeze_bias"]),
            )),
        ]
        lines.extend(lines_logits)

        lines.append("")

        return lines

    @staticmethod
    def _format_aug_block(aug: dict) -> list[str]:
        lines = ["=== Image Augmentation ==="]

        lines.append("RRCrop")
        lines.append(PrintLog._block_metric_lines([
            ("- Scale", aug["rrcrop"]["scale"]),
            ("- Ratio", aug["rrcrop"]["ratio"]),
        ]))

        if aug.get("hflip", False):
            lines.append("Horizontal Flip Enabled")

        if "cjit_prob" in aug:
            lines.append(f"Color Jitter (p={aug['cjit_prob']})")
            lines.append(PrintLog._block_metric_lines([
                ("- Brightness", aug["cjit"]["brightness"]),
                ("- Contrast",   aug["cjit"]["contrast"]),
                ("- Saturation", aug["cjit"]["saturation"]),
                ("- Hue",        aug["cjit"]["hue"]),
            ]))

        if "sharpness_prob" in aug:
            lines.append(PrintLog._block_metric_lines([
                (f"Sharpness (p={aug['sharpness_prob']})", aug["sharpness"]),
            ]))

        if "gblur_prob" in aug:
            lines.append(f"Gaussian Blur (p={aug['gblur_prob']})")
            lines.append(PrintLog._block_metric_lines([
                ("- Kernel Size", aug["gblur"]["kernel_size"]),
                ("- Sigma",       aug["gblur"]["sigma"]),
            ]))

        lines.append("")
        return lines

    @staticmethod
    def _block_metric_lines(metric_pairs):
        labels, values = zip(*metric_pairs)
        max_len = max(len(label) for label in labels)
        n_dashes = [3 + (max_len - len(label)) for label in labels]
        lines = [f"{label} {'-' * nd} {value}" for label, nd, value in zip(labels, n_dashes, values)]
        return "\n".join(lines)

    @staticmethod
    @rank0
    def close_logs():
        for handle in (
            PrintLog.log_batch_general,
            PrintLog.log_batch_grad_norm,
            PrintLog.log_batch_temp_bias,
            PrintLog.log_epoch,
            PrintLog.log_eval,
            PrintLog.log_init,
            PrintLog.log_text_train,
            PrintLog.log_text_eval,
        ):
            if handle is not None and not handle.closed:
                handle.close()

def get_subdirectory_names(dir_path):
    return [p.name for p in Path(dir_path).iterdir() if p.is_dir()]


class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0.0
        self._active = False

    def start(self):
        if self._active:
            raise RuntimeError("Timer is already active.")
        self._start_time = time.time()
        self._active = True

    def stop(self):
        if not self._active:
            raise RuntimeError("Timer is not active.")
        self._elapsed_time += time.time() - self._start_time
        self._start_time = None
        self._active = False

    def get_elapsed_time(self):
        if self._active:
            elapsed_time = time.time() - self._start_time
        else:
            elapsed_time = self._elapsed_time
        return elapsed_time

    def set_elapsed_time(self, elapsed_time):
        if self._active:
            raise RuntimeError("Timer is active. Stop it before setting elapsed time.")
        self._elapsed_time = elapsed_time

    def reset(self):
        self._start_time = None
        self._elapsed_time = 0.0
        self._active = False