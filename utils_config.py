import torch  # type: ignore[import]
from dataclasses import dataclass, field
import yaml  # type: ignore[import]

from utils import load_json, paths
from utils_hardware import compute_dataloader_workers_prefetch

import pdb


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

    if "dyn_posneg" in loss['cfg']:
        lines.append(f"Dyn Pos/Neg ------- {loss['cfg']['dyn_posneg']}")

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

@dataclass
class TrainConfig:

    study_name: str
    experiment_name: str
    seed: int | None
    split_name: str

    n_epochs: int
    chkpt_every: int
    batch_size: int
    dv_batching: bool

    dev: dict
    arch: dict
    loss: dict
    loss2: dict
    opt: dict
    freeze: dict
    text_preps: dict

    hw: dict = field(init=False, default_factory=dict)

    def __post_init__(self):

        if self.freeze["image"] and self.freeze["text"]:
            raise ValueError("Image and text encoders are both set to frozen!")
        
        if self.loss['type'] not in ("infonce1", "infonce2", "bce", "mse", "huber"):
            raise ValueError(f"Unknown Loss 1 Type: '{self.loss['type']}', must be one of {{infonce1, infonce2, bce, mse, huber}}")
        if self.loss2['type'] not in ("infonce1", "infonce2", "bce", "mse", "huber"):
            raise ValueError(f"Unknown Loss 2 Type: '{self.loss2['type']}', must be one of {{infonce1, infonce2, bce, mse, huber}}")
        
        if self.loss["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown Loss 1 sim_type: '{self.loss['sim']}', must be one of {{cos, geo1, geo2}}")
        if self.loss2["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown Loss 2 sim_type: '{self.loss2['sim']}', must be one of {{cos, geo1, geo2}}")
        
        if self.loss['targ'] not in ("aligned", "multipos", "hierarchical", "phylogenetic"):
            raise ValueError(f"Unknown Loss 1 targ_type: '{self.loss['targ']}', must be one of {{aligned, multipos, hierarchical, phylogenetic}}")
        if self.loss2['targ'] not in ("aligned", "multipos", "hierarchical", "phylogenetic"):
            raise ValueError(f"Unknown Loss 2 targ_type: '{self.loss2['targ']}', must be one of {{aligned, multipos, hierarchical, phylogenetic}}")
        
        if self.opt['lr']['sched'] not in ("exp", "plat", "cos", "coswr", "cosXexp", "coswrXexp"):
            raise ValueError(f"Unknown LR scheduler type: '{self.opt['lr']['sched']}', must be one of {{exp, plat, cos, coswr, cosXexp, coswrXexp}}")
        
        if not 0.0 <= self.loss2["mix"] <= 1.0:
            raise ValueError(f"Secondary loss mix out of bounds: {self.loss2['mix']}, must be between 0.0 and 1.0")

        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch()
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        self.rdpath_trial = f"artifacts/{self.study_name}/{self.experiment_name}/{self.seed}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

    def print_init_info(self):
        lines: list[str] = [
            "",
            f"Study --------------- {self.study_name}",
            f"Experiment ---------- {self.experiment_name}",
            f"Seed ---------------- {self.seed}",
            f"Trial Path (Save) --- {self.rdpath_trial}",
            f"Split --------------- {self.split_name}",
            "",
            f"Batch Size ---- {self.batch_size}",
            f"DV Batching --- {self.dv_batching}",
            "",
            f"=== Architecture ===",
            f"Model Type --- {self.arch['model_type']}",
            f"Non-Causal --- {self.arch['non_causal']}",
            "",
        ]

        # primary loss block
        lines.extend(_format_loss_block(self.loss))

        # secondary loss block (if enabled)
        if self.loss2["mix"] != 0.0:
            lines.extend(_format_loss_block(self.loss2, secondary=True))

        # optimization block
        lines.extend([
            f"=== Optimization ===",
            f"LR Init ---- {self.opt['lr']['init']}",
            f"LR Sched --- {self.opt['lr']['sched']}",
            f"L2 Reg ----- {self.opt['l2reg']}",
            f"β1 --------- {self.opt['beta1']}",
            f"β2 --------- {self.opt['beta2']}",
            f"ε ---------- {self.opt['eps']}",
            "",
        ])

        # hardware block
        lines.extend(_format_hw_block(self))

        print(*lines, sep="\n")

def get_config_train(verbose=True):
    with open(paths["config"] / "train/train.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainConfig(**cfg_dict)
    cfg.lr_sched_params = get_config_lr_sched(cfg.opt['lr']['sched'])
    cfg.loss["cfg"]  = get_config_loss(cfg)
    if cfg.loss2["mix"] != 0.0:
        cfg.loss2["cfg"] = get_config_loss(cfg, secondary=True)
    cfg.hw = get_config_hardware()
    if verbose:
        cfg.print_init_info()
    return cfg

def get_config_lr_sched(lr_sched_type):
    with open(paths["config"] / "train/lr_sched.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict[lr_sched_type]

# helper for get_config_train()
def get_config_loss(cfg_train, secondary=False):
    if not secondary:
        fpath = "loss.yaml"
        cfg_train_loss = cfg_train.loss
    else:
        fpath = "loss2.yaml"
        cfg_train_loss = cfg_train.loss2

    with open(paths["config"] / fpath) as f:
        cfg_loss = yaml.safe_load(f)  # ok we gotta stop referring to both loss.yaml and the loss portions of train.yaml as cfg_loss...

    if cfg_train_loss["type"] in ("infonce1", "infonce2", "bce"):
        if not (cfg_loss["logits"]["scale_init"] is None and cfg_loss["logits"]["bias_init"] is None):
            print(f"\nWARNING: loss_type = '{cfg_train_loss['type']}' and logit scale/bias overridden!\n")
    elif cfg_train_loss["type"] in ("mse", "huber"):
        if not (cfg_loss["logits"]["scale_init"] == 0.0 and cfg_loss["logits"]["bias_init"]  == 0.0):
            print(f"\nWARNING: loss_type = '{cfg_train_loss['type']}' and logit scale/bias not initialized to 0.0!\n")

    if not cfg_train_loss["wting"]:
        del cfg_loss["class_weighting"]
    if not cfg_train_loss["focal"]:
        del cfg_loss["focal"]
    if cfg_train_loss["type"] not in ("bce", "mse", "huber"):
        del cfg_loss["dyn_posneg"]
    if cfg_train_loss["type"] not in ("mse", "huber"):
        del cfg_loss["regression"]

    return cfg_loss

@dataclass
class HardwareConfig:

    cached_imgs: str | None
    mixed_prec: bool
    act_chkpt: bool

def get_config_hardware():
    with open(paths["config"] / "hardware.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = HardwareConfig(**cfg_dict)
    return cfg

@dataclass
class EvalConfig:
    rdpath_trial: str | None
    save_crit: str  # model save criterion (only applicable if DPATH_TRIAL != None)

    split_name: str  # overridden if rdpath_trial is specified

    batch_size: int

    dev: dict
    arch: dict
    loss: dict
    loss2: dict

    text_preps: str

    hw: dict = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch()
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        if self.rdpath_trial is not None:
            metadata_experiment = load_json(paths["root"] / self.rdpath_trial / "../metadata_experiment.json")
            self.arch['model_type'] = metadata_experiment["arch"]["model_type"]  # override model_type
            self.arch['non_causal'] = metadata_experiment["arch"]["non_causal"]  # override non_causal
            self.loss['type']       = metadata_experiment["loss"]["type"]  # override loss_type

            if "loss2" in metadata_experiment:
                self.loss2.update(metadata_experiment["loss2"])
            else:
                self.loss2["mix"] = 0.0

            metadata_study  = load_json(paths["root"] / self.rdpath_trial / "../../metadata_study.json")
            self.split_name = metadata_study["split_name"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

    def print_init_info(self):
        lines: list[str] = [
            "",
            f"Trial (Loaded) ---- {self.rdpath_trial}{'' if self.rdpath_trial is None else ' (' + self.save_crit + ')'}",
            "",
            f"Model Type -------- {self.arch['model_type']}",
            f"Loss Type --------- {self.loss['type']}",
            f"Split ------------- {self.split_name}",
            "",
            f"Batch Size -------- {self.batch_size}",
            "",
        ]

        lines.extend(_format_loss_block(self.loss))

        mix = self.loss2.get("mix", 0.0)
        if mix != 0.0:
            lines.extend(_format_loss_block(self.loss2, secondary=True))

        lines.extend(_format_hw_block(self))

        print(*lines, sep="\n")

def get_config_eval(verbose=True):
    with open(paths["config"] / "eval.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EvalConfig(**cfg_dict)
    cfg.loss["cfg"]  = get_config_loss(cfg)
    if cfg.loss2["mix"] != 0.0:
        cfg.loss2["cfg"] = get_config_loss(cfg, secondary=True)
    cfg.hw = get_config_hardware()
    if verbose:
        cfg.print_init_info()
    return cfg

@dataclass
class GenSplitConfig:

    seed: int
    split_name: str

    allow_overwrite: bool

    pct_eval: float
    pct_ood_tol: float

    nst_names: list
    nst_seps: list

def get_config_gen_split():
    with open(paths["config"] / "gen_split.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = GenSplitConfig(**cfg_dict)
    return cfg