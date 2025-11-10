import torch  # type: ignore[import]
from dataclasses import dataclass, field
from pathlib import Path
import yaml  # type: ignore[import]

from utils import compute_dataloader_workers_prefetch, load_json, paths

import pdb


@dataclass
class TrainConfig:

    study_name: str
    experiment_name: str
    seed: int | None
    split_name: str

    allow_overwrite_trial: bool
    allow_diff_study: bool
    allow_diff_experiment: bool

    model_type: str
    non_causal: bool

    loss_type: str
    sim_type: str
    targ_type: str
    class_wting: bool
    focal: bool
    cfg_loss: dict = field(init=False, default_factory=dict)

    n_epochs: int
    chkpt_every: int
    batch_size: int
    
    lr_init: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float

    lr_sched_type: str

    freeze_text_encoder: bool
    freeze_image_encoder: bool

    cached_imgs: str | None
    mixed_prec: bool
    act_chkpt: bool
    drop_partial_batch_train: bool
    verbose_batch_loss: bool

    text_preps_type_train: str
    text_preps_type_val: str

    def __post_init__(self):
        if self.freeze_text_encoder and self.freeze_image_encoder:
            raise ValueError("Text and image encoders are both set to frozen!")
        # if self.lr_sched_type == "plat" and self.lr_sched["args"]["valid_type"] not in ("loss", "perf"):
        #     raise ValueError("For plateau LR scheduler, valid_type must be one of: {loss, perf}")
        if self.loss_type not in ("infonce1", "infonce2", "bce", "mse", "huber"):
            raise ValueError(f"Unknown loss_type: '{self.loss_type}', must be one of {{infonce1, infonce2, bce, mse, huber}}")
        if self.sim_type not in ("cos", "geo"):
            raise ValueError(f"Unknown sim_type: '{self.sim_type}', must be one of {{cos, geo}}")
        if self.targ_type not in ("aligned", "multipos", "hierarchical", "phylogenetic"):
            raise ValueError(f"Unknown targ_type: '{self.targ_type}', must be one of {{aligned, multipos, hierarchical, phylogenetic}}")
        # if self.focal["comp_type"] not in (1, 2):
        #     raise ValueError(f"Unknown focal computation type focal['comp_type']: '{self.focal['comp_type']}', must be one of {{1, 2}}")
        if self.lr_sched_type not in ("exp", "plat", "cos", "coswr", "cosXexp", "coswrXexp"):
            raise ValueError(f"Unknown LR scheduler type: '{self.lr_sched_type}', must be one of {{exp, plat, cos, coswr, cosXexp, coswrXexp}}")
        # if self.class_weighting["type"] not in ("inv_freq", "class_balanced"):
        #     raise ValueError(f"Unknown class weighting type: '{self.class_weighting['type']}', must be one of {{inv_freq, class_balanced}}")
        # if self.class_weighting["cp_type"] not in (1, 2):
        #     raise ValueError(f"Unknown class weighting type: '{self.class_weighting['type']}', must be one of {{1, 2}}")

        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch()
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        self.rdpath_trial = f"artifacts/{self.study_name}/{self.experiment_name}/{self.seed}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.print_init_info()

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

    def print_init_info(self):
        print(
            f"",
            f"Study ------------------- {self.study_name}",
            f"Experiment -------------- {self.experiment_name}",
            f"Seed -------------------- {self.seed}",
            f"Trial Path -------------- {self.rdpath_trial}",
            f"",
            f"Model Type -------------- {self.model_type}",
            f"Loss Type --------------- {self.loss_type}",
            f"Similarity Type --------- {self.sim_type}",
            f"Targ Type --------------- {self.targ_type}",
            f"Split ------------------- {self.split_name}",
            f"",
            f"Batch Size -------------- {self.batch_size}",
            f"",
            f"LR Init ----------------- {self.lr_init}",
            f"Weight Decay ------------ {self.weight_decay}",
            f"(β1, β2) ---------------- ({self.beta1}, {self.beta2})",
            f"ε (Optimizer) ----------- {self.eps}",
            f"",
            f"LR Scheduler ------------ {self.lr_sched_type}",
            sep="\n"
        )

        # if self.lr_sched["type"] == "exp":
        #     print(f"~ Gamma (Decay) --------- {self.lr_sched['args_sched']['gamma']}")
        # elif self.lr_sched["type"] == "plat":
        #     print(f"~ Factor (Decay) -------- {self.lr_sched['args_sched']['factor']}")
        #     print(f"~ Patience -------------- {self.lr_sched['args_sched']['patience']}")
        #     print(f"~ Cooldown -------------- {self.lr_sched['args_sched']['cooldown']}")
        #     print(f"~ LR Min ---------------- {self.lr_sched['args_sched']['min_lr']}")
        # elif self.lr_sched["type"] == "cos":
        #     print(f"~ Half-Period (T_max) --- {self.lr_sched['args_sched']['T_max']}")
        #     print(f"~ LR Min (eta_min) ------ {self.lr_sched['args_sched']['eta_min']}")
        # elif self.lr_sched["type"] == "coswr":
        #     print(f"~ Period (T_0) ---------- {self.lr_sched['args_sched']['T_0']}")
        #     print(f"~ LR Min (eta_min) ------ {self.lr_sched['args_sched']['eta_min']}")
        # elif self.lr_sched["type"] == "cosXexp":
        #     print("foo")
        # elif self.lr_sched["type"] == "coswrXexp":
        #     print("bar")

        print(
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

def get_config_train():
    with open(paths["config"] / "train/train.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainConfig(**cfg_dict)
    cfg.lr_sched_params = get_config_lr_sched(cfg.lr_sched_type)
    cfg.cfg_loss        = get_config_loss(cfg)
    return cfg

def get_config_lr_sched(lr_sched_type):
    with open(paths["config"] / "train/lr_sched.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict[lr_sched_type]

# helper for get_config_train()
def get_config_loss(cfg_train):
    with open(paths["config"] / "loss.yaml") as f:
        cfg_loss = yaml.safe_load(f)

    if cfg_train.loss_type in ("infonce1", "infonce2", "bce"):
        if not (cfg_loss["logits"]["scale_init"] is None and cfg_loss["logits"]["bias_init"] is None):
            print(f"\nWARNING: loss_type = '{cfg_train.loss_type}' and logit scale/bias overridden!\n")
    elif cfg_train.loss_type in ("mse", "huber"):
        if not (cfg_loss["logits"]["scale_init"] == 0.0 and cfg_loss["logits"]["bias_init"]  == 0.0):
            print(f"\nWARNING: loss_type = '{cfg_train.loss_type}' and logit scale/bias not initialized to 0.0!\n")

    if not cfg_train.class_wting:
        del cfg_loss["class_weighting"]
    if not cfg_train.focal:
        del cfg_loss["focal"]
    if cfg_train.loss_type not in ("bce", "mse", "huber"):
        del cfg_loss["dyn_posneg"]
    if cfg_train.loss_type not in ("mse", "huber"):
        del cfg_loss["regression"]

    return cfg_loss

@dataclass
class EvalConfig:
    rdpath_trial: str | None
    save_crit: str  # model save criterion (only applicable if DPATH_TRIAL != None)

    split_name: str  # overridden if rdpath_trial is specified

    verbose_batch_loss: bool

    model_type: str
    
    loss_type: str
    sim_type: str
    targ_type: str
    class_wting: bool
    focal: bool

    batch_size: int
    text_preps_type: str

    cached_imgs: bool
    act_chkpt:   bool

    non_causal: bool = False
    
    def __post_init__(self):
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch()
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        if self.rdpath_trial is not None:
            metadata_experiment = load_json(paths["root"] / self.rdpath_trial / "../metadata_experiment.json")
            self.model_type     = metadata_experiment["model_type"]  # override model_type
            self.loss_type      = metadata_experiment["loss_type"]  # override loss_type
            self.non_causal     = metadata_experiment["non_causal"]  # override non_causal

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
            f"Model Type -------- {self.model_type}",
            f"Loss Type --------- {self.loss_type}",
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
    cfg.cfg_loss = get_config_loss(cfg)
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
