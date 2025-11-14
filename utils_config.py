import torch  # type: ignore[import]
from dataclasses import dataclass, field
import yaml  # type: ignore[import]

from utils import load_json, paths
from utils_hardware import compute_dataloader_workers_prefetch

import pdb


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
    opt: dict
    freeze: dict
    text_preps: dict

    hw: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if self.freeze["text"] and self.freeze_image_encoder:
            raise ValueError("Text and image encoders are both set to frozen!")
        if self.loss['type'] not in ("infonce1", "infonce2", "bce", "mse", "huber"):
            raise ValueError(f"Unknown loss_type: '{self.loss['type']}', must be one of {{infonce1, infonce2, bce, mse, huber}}")
        if self.loss["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown sim_type: '{self.loss['sim']}', must be one of {{cos, geo1, geo2}}")
        if self.loss['targ'] not in ("aligned", "multipos", "hierarchical", "phylogenetic"):
            raise ValueError(f"Unknown targ_type: '{self.loss['targ']}', must be one of {{aligned, multipos, hierarchical, phylogenetic}}")
        if self.opt['lr']['sched'] not in ("exp", "plat", "cos", "coswr", "cosXexp", "coswrXexp"):
            raise ValueError(f"Unknown LR scheduler type: '{self.opt['lr']['sched']}', must be one of {{exp, plat, cos, coswr, cosXexp, coswrXexp}}")

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
        print(
            f"",
            f"Study -------- {self.study_name}",
            f"Experiment --- {self.experiment_name}",
            f"Seed --------- {self.seed}",
            f"Trial Path --- {self.rdpath_trial}",
            f"Split -------- {self.split_name}",
            f"",
            f"Batch Size --- {self.batch_size}",
            f"",
            f"=== Architecture ===",
            f"Model Type --- {self.arch['model_type']}",
            f"Non-Causal --- {self.arch['non_causal']}",
            f"",
            f"=== Loss ===",
            f"Type -------------- {self.loss['type']}",
            f"Similarity Type --- {self.loss['sim']}",
            f"Target Type ------- {self.loss['targ']}",
            f"Params: _",
            f"",
            f"=== Optimization ===",
            f"LR Init ---- {self.opt['lr']['init']}",
            f"LR Sched --- {self.opt['lr']['sched']}",
            f"  Params: _",
            f"L2 Reg ----- {self.opt['l2reg']}",
            f"β1 --------- {self.opt['beta1']}",
            f"β2 --------- {self.opt['beta2']}",
            f"ε ---------- {self.opt['eps']}",
            f"",
            f"Num. GPUs --------------- {self.n_gpus}",
            f"Num. CPUs --------------- {self.n_cpus}",
            f"RAM --------------------- {self.ram} GB",
            f"",
            f"Num. Workers ------------ {self.n_workers}",
            f"Prefetch Factor --------- {self.prefetch_factor}",
            f"",
            f"Device ------------------ {self.device}",
            sep="\n"
        )

def get_config_train(verbose=True):
    with open(paths["config"] / "train/train.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainConfig(**cfg_dict)
    if verbose:
        cfg.print_init_info()
    cfg.lr_sched_params = get_config_lr_sched(cfg.opt['lr']['sched'])
    cfg.loss["cfg"] = get_config_loss(cfg)
    cfg.hw          = get_config_hardware()
    return cfg

def get_config_lr_sched(lr_sched_type):
    with open(paths["config"] / "train/lr_sched.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict[lr_sched_type]

# helper for get_config_train()
def get_config_loss(cfg_train):
    with open(paths["config"] / "loss.yaml") as f:
        cfg_loss = yaml.safe_load(f)

    if cfg_train.loss['type'] in ("infonce1", "infonce2", "bce"):
        if not (cfg_loss["logits"]["scale_init"] is None and cfg_loss["logits"]["bias_init"] is None):
            print(f"\nWARNING: loss_type = '{cfg_train.loss['type']}' and logit scale/bias overridden!\n")
    elif cfg_train.loss['type'] in ("mse", "huber"):
        if not (cfg_loss["logits"]["scale_init"] == 0.0 and cfg_loss["logits"]["bias_init"]  == 0.0):
            print(f"\nWARNING: loss_type = '{cfg_train.loss['type']}' and logit scale/bias not initialized to 0.0!\n")

    if not cfg_train.loss['wting']:
        del cfg_loss["class_weighting"]
    if not cfg_train.loss['focal']:
        del cfg_loss["focal"]
    if cfg_train.loss['type'] not in ("bce", "mse", "huber"):
        del cfg_loss["dyn_posneg"]
    if cfg_train.loss['type'] not in ("mse", "huber"):
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

    text_preps: str

    hw: dict = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch()
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        if self.rdpath_trial is not None:
            metadata_experiment = load_json(paths["root"] / self.rdpath_trial / "../metadata_experiment.json")
            self.model_type     = metadata_experiment["model_type"]  # override model_type
            self.non_causal     = metadata_experiment["non_causal"]  # override non_causal
            self.loss['type']   = metadata_experiment["loss_type"]  # override loss_type

            metadata_study  = load_json(paths["root"] / self.rdpath_trial / "../../metadata_study.json")
            self.split_name = metadata_study["split_name"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.print_init_info()

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

    def print_init_info(self):
        print(
            f"",
            f"Trial ------------- {self.rdpath_trial}{'' if self.rdpath_trial is None else ' (' + self.save_crit + ')'}",
            f"",
            f"Model Type -------- {self.arch['model_type']}",
            f"Loss Type --------- {self.loss['type']}",
            f"Split ------------- {self.split_name}",
            f"",
            f"Batch Size -------- {self.batch_size}",
            f"",
            f"Num. GPUs --------- {self.n_gpus}",
            f"Num. CPUs --------- {self.n_cpus}",
            f"RAM --------------- {self.ram} GB",
            f"",
            f"Num. Workers ------ {self.n_workers}",
            f"Prefetch Factor --- {self.prefetch_factor}",
            f"",
            f"Device ------------ {self.device}",
            f"",
            sep="\n"
        )

def get_config_eval():
    with open(paths["config"] / "eval.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EvalConfig(**cfg_dict)
    cfg.loss["cfg"] = get_config_loss(cfg)
    cfg.hw          = get_config_hardware()
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