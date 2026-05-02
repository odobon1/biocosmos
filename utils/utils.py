import pickle
from pathlib import Path
import random
import os
import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
import json
from typing import List, Any, Dict, Optional
import math

from utils.text import get_text_template as get_dataset_text_template

import pdb


CLUSTER = "pace"  # PACE
# CLUSTER = "hpg"  # HiPerGator


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
            "cub": dpath_root / "data/cub/CUB_200_2011/images",
            "lepid": dpath_lepid / "images",
        },
        "preproc": {
            "cub": dpath_root / "preprocessing/cub",
            "lepid": dpath_root / "preprocessing/lepid",
        },
        "metadata": {
            "cub": dpath_root / "metadata/cub",
            "lepid": dpath_root / "metadata/lepid",
        },
        "cub_tree_raw": dpath_root / "data/cub/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy",
        "lepid_metadata_imgs": dpath_lepid / "metadata/data_meta-clean_rot_512-butterflies_whole_specimen-v2025_05_07.csv",
        "lepid_metadata_tax": dpath_lepid / "metadata/data_tree_meta.csv",
        "lepid_tree_raw": dpath_root / "data/lepid/tree_renamed_full.tre",
        "nymph_tree_raw": dpath_root / "data/nymph/tree_nymphalidae_chazot2021_all.tree",
    }
elif CLUSTER == "hpg":

    dpath_root = Path(os.getcwd())
    dpath_group = Path("/lustre/blue2/arthur.porto-biocosmos")
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
        "bryo_tree_raw": dpath_root / "data/bryo/SI_Fig1(BIG).newick",
        "cub_tree_raw": dpath_root / "data/cub/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy",
        "lepid_metadata_imgs": dpath_lepid / "metadata/data_meta-clean_rot_512-butterflies_whole_specimen-v2025_05_07.csv",
        "lepid_metadata_tax": dpath_lepid / "metadata/data_tree_meta.csv",
        "lepid_tree_raw": dpath_root / "data/lepid/tree_renamed_full.tre",
        "nymph_metadata": dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv",
        "nymph_tree_raw": dpath_root / "data/nymph/tree_nymphalidae_chazot2021_all.tree",
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

def load_split(split_name, dataset="nymph"):
    split = load_pickle(paths["metadata"][dataset] / f"splits/{split_name}/split.pkl")
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
    
def shuffle_list(input: List[Any], seed: int) -> List[int]:
    rng        = random.Random(seed)
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
    log_init  = None
    log_text  = None
    log_text_eval = None
    wrote_text_eval = False

    @staticmethod
    def create_logs(dpath_logs):
        PrintLog.logging = True
        PrintLog.wrote_text_eval = False
        dpath_batch_logs = dpath_logs / "batch"
        dpath_batch_logs.mkdir(parents=True, exist_ok=True)
        PrintLog.log_batch_general = open(dpath_batch_logs / "general.log", "a", buffering=1)
        PrintLog.log_batch_grad_norm = open(dpath_batch_logs / "grad_norm.log", "a", buffering=1)
        PrintLog.log_batch_temp_bias = open(dpath_batch_logs / "temp_bias.log", "a", buffering=1)
        PrintLog.log_epoch = open(dpath_logs / "epoch.log", "a", buffering=1)
        PrintLog.log_eval = open(dpath_logs / "eval.log", "a", buffering=1)
        PrintLog.log_init  = open(dpath_logs / "init.log",  "a", buffering=1)
        PrintLog.log_text  = open(dpath_logs / "text.log",  "a", buffering=1)
        PrintLog.log_text_eval = open(dpath_logs / "text_eval.log",  "a", buffering=1)

    @staticmethod
    def texts(texts_sb):
        text_printout = "\n".join(texts_sb)
        if PrintLog.logging:
            PrintLog.log_text.write(text_printout)

    @staticmethod
    def texts_eval(texts_by_partition):
        if not PrintLog.logging or PrintLog.wrote_text_eval:
            return

        lines = []
        for partition_name, texts in texts_by_partition.items():
            lines.append(f"[{partition_name}]")
            lines.extend(texts)
            lines.append("")

        PrintLog.log_text_eval.write("\n".join(lines))
        PrintLog.wrote_text_eval = True

    @staticmethod
    def epoch_header(idx_epoch, n_epochs):
        header_epoch = f" Epoch {idx_epoch}/{n_epochs} "
        header_epoch = (
            f"{header_epoch:#^{75}}"
            f"\n"
        )
        print(header_epoch)
        if PrintLog.logging:
            PrintLog.log_batch_general.write(header_epoch)
            PrintLog.log_batch_grad_norm.write(header_epoch)
            PrintLog.log_batch_temp_bias.write(header_epoch)
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
    def epoch(time_train, time_train_avg, loss_train_avg, loss_train_raw_avg, samps_seen):

        epoch_info = (
            f"Samples Seen -------- {samps_seen:,}\n"
            f"Train Loss --------- {loss_train_avg:.3e}\n"
            f"Train Loss (raw) --- {loss_train_raw_avg:.3e}\n"
            f"\n"
            f"{' Elapsed Time ':=^{75}}\n"
            f"Train -------- {time_train:.2f} s (avg: {time_train_avg:.2f} s)\n"
            f"\n"
        )
        print(epoch_info)
        if PrintLog.logging:
            PrintLog.log_epoch.write(epoch_info)

    @staticmethod
    def batch(idx_batch, lr, loss_batch, embs_img_b, embs_txt_b, logits, model):

        def tensor_grad_l2_norm(x: torch.Tensor | None) -> float:
            if x is None:
                return float("nan")
            if x.grad is None:
                return float("nan")
            return x.grad.detach().pow(2).sum().sqrt().item()

        def model_grad_l2_norm(model: torch.nn.Module) -> float:
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total += p.grad.detach().pow(2).sum().item()
            return math.sqrt(total)

        def tensor_scalar_item(x) -> float:
            if x is None:
                return float("nan")
            if isinstance(x, torch.Tensor):
                return x.detach().item()
            return float(x)

        logits1 = logits[0]
        logits2 = logits[1]
        if logits2 is None:
            logits_line = f"log={tensor_grad_l2_norm(logits1):.2e}, "
        else:
            logits_line = f"log1={tensor_grad_l2_norm(logits1):.2e}, "
            logits_line += f"log2={tensor_grad_l2_norm(logits2):.2e}, "

        grad_norm_line = (
            f"img={tensor_grad_l2_norm(embs_img_b):.2e}, "
            f"txt={tensor_grad_l2_norm(embs_txt_b):.2e}, "
            f"{logits_line}"
            f"model={model_grad_l2_norm(model):.2e}"
        )

        logits_param_line = ""
        if hasattr(model.module, "logit_scale") and model.module.logit_scale is not None:
            logits_param_line += f"s1={model.module.logit_scale.detach().exp().item():.2e}, "
        if hasattr(model.module, "logit_scale2") and model.module.logit_scale2 is not None:
            logits_param_line += f"s2={model.module.logit_scale2.detach().exp().item():.2e}, "
        if hasattr(model.module, "logit_bias") and model.module.logit_bias is not None:
            logits_param_line += f"b1={tensor_scalar_item(model.module.logit_bias):.2e}, "
        if hasattr(model.module, "logit_bias2") and model.module.logit_bias2 is not None:
            logits_param_line += f"b2={tensor_scalar_item(model.module.logit_bias2):.2e}, "
        logits_param_line = logits_param_line.rstrip(", ")

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
                f"{grad_norm_line}"
                f"\n"
            )
            PrintLog.log_batch_temp_bias.write(
                f"{batch_str:<10} "
                f"{logits_param_line}"
                f"\n"
            )

    @staticmethod
    def eval(
        scores_eval: Dict[str, float],
        eval_pipe,
        header: Optional[str] = None,
        samps_seen: Optional[int] = None,
        idx_epoch: Optional[int] = None,
        time_val: Optional[float] = None,
        log_to: str = "epoch",
    ) -> None:
        
        partition_names = eval_pipe.partition_names
        nshot_bucket_names = eval_pipe.nshot_bucket_names
        bucket_partition_name = eval_pipe.bucket_partition_name

        if log_to == "eval":
            target_log = PrintLog.log_eval
        elif log_to == "epoch":
            target_log = PrintLog.log_epoch
        else:
            target_log = None

        header_str = ""
        if header is not None:
            header_wrapped = f" {header} "
            header_str = (
                f"{header_wrapped:#^{75}}"
                f"\n"
            )
            print(header_str)
            if PrintLog.logging and target_log is not None:
                target_log.write(header_str)

        if eval_pipe.best_comp_map is not None:
            best_comp_map_str = f" (best: {eval_pipe.best_comp_map:.4f})"
            best_i2i_map_str = f" (best: {eval_pipe.best_i2i_map:.4f})"
        else:
            best_comp_map_str = ""
            best_i2i_map_str = ""

        bucket_comp_keys = []
        nshot_comp_lines = ""

        bucket_comp_keys = [
            f"{bucket_partition_name}_{bucket_name}_comp"
            for bucket_name in nshot_bucket_names
            if f"{bucket_partition_name}_{bucket_name}_comp" in scores_eval
        ]
        if bucket_comp_keys:
            nshot_comp_lines = f"{' N-Shot Composite mAP ':-^{75}}\n"
            labels = [
                f"{k.removeprefix(bucket_partition_name + '_').removesuffix('_comp')}"
                for k in bucket_comp_keys
            ]
            len_max = max(len(label) for label in labels)
            for k, label in zip(bucket_comp_keys, labels):
                n_dashes = len_max - len(label) + 3
                nshot_comp_lines += f"{label} {'-' * n_dashes} {scores_eval[k]:.4f}\n"

        partition_lines = ""
        for partition_name in partition_names:
            partition_lines += (
                f"{f' {partition_name} mAP ':-^{75}}\n"
                f"I2T --- {scores_eval[f'{partition_name}_i2t_map']:.4f}\n"
                f"I2I --- {scores_eval[f'{partition_name}_i2i_map']:.4f}\n"
                f"T2I --- {scores_eval[f'{partition_name}_t2i_map']:.4f}\n"
            )

        def _metric_line(label: str, value_str: str) -> str:
            # Keep a minimum of 3 dashes while aligning labels to a fixed width.
            n_dashes = max(3, 14 - len(label))
            return f"{label} {'-' * n_dashes} {value_str}\n"

        composite_lines = f"{' Composite mAP ':-^{75}}\n"
        composite_lines += _metric_line("All", f"{scores_eval['comp_map']:.4f}{best_comp_map_str}")
        composite_lines += _metric_line("I2I", f"{scores_eval['i2i_map']:.4f}{best_i2i_map_str}")
        for partition_name in partition_names:
            composite_lines += _metric_line(partition_name, f"{scores_eval[f'{partition_name}_map']:.4f}")

        loss_lines = f"{' Loss ':-^{75}}\n"
        for partition_name in partition_names:
            loss_lines += _metric_line(partition_name, f"{scores_eval[f'{partition_name}_loss']:.3e}")

        context_lines = ""
        if idx_epoch is not None:
            context_lines += f"Epoch -------- {idx_epoch}\n"
        if samps_seen is not None:
            context_lines += f"Samples Seen - {samps_seen:,}\n"
        if time_val is not None:
            context_lines += f"Validation --- {time_val:.2f} s\n"
        if context_lines:
            context_lines += "\n"

        header = " Eval "
        eval_printout = (
            f"{header:=^{75}}\n"
            f"{context_lines}"
            f"{partition_lines}"
            f"{nshot_comp_lines}"
            f"{composite_lines}"
            f"{loss_lines}"
            f"\n"
        )
        print(eval_printout)

        if PrintLog.logging and target_log is not None:
            target_log.write(eval_printout)

    @staticmethod
    def init_train(cfg_train):
        lines: list[str] = [
            "",
            f"Study ------------- {cfg_train.study_name}",
            f"Experiment -------- {cfg_train.experiment_name}",
            f"Seed -------------- {cfg_train.seed}",
            f"Trial File Path --- {cfg_train.rdpath_trial}",
            f"Dataset ----------- {cfg_train.dataset}",
            f"Split ------------- {cfg_train.split_name}",
            "",
            f"Batch Size ---- {cfg_train.batch_size}",
            f"DV Batching --- {cfg_train.dv_batching}",
            f"Eval Every ---- {cfg_train.eval_every:,} samples",
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
            f"Dataset ---------- {cfg_eval.dataset}",
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

        wting = loss.get("wting", False)
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

        focal = loss.get("focal", False)
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
        for handle in (
            PrintLog.log_batch_general,
            PrintLog.log_batch_grad_norm,
            PrintLog.log_batch_temp_bias,
            PrintLog.log_epoch,
            PrintLog.log_eval,
            PrintLog.log_init,
            PrintLog.log_text,
            PrintLog.log_text_eval,
        ):
            if handle is not None and not handle.closed:
                handle.close()

def get_subdirectory_names(dir_path):
    return [p.name for p in Path(dir_path).iterdir() if p.is_dir()]