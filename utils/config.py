import torch  # type: ignore[import]
from dataclasses import dataclass, field
import yaml  # type: ignore[import]

from utils.utils import PrintLog, load_json, paths
from utils.hardware import compute_dataloader_workers_prefetch

import pdb


@dataclass
class TrainConfig:

    study_name: str
    experiment_name: str
    seed: int | None
    dataset: str
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
    text_template: dict

    logging: bool
    eval_type: str = "validation"

    hw: dict = field(init=False, default_factory=dict)

    def __post_init__(self):

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

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
        
        if self.loss['targ'] not in ("aligned", "multipos", "tax", "phylo"):
            raise ValueError(f"Unknown Loss 1 targ_type: '{self.loss['targ']}', must be one of {{aligned, multipos, tax, phylo}}")
        if self.loss2['targ'] not in ("aligned", "multipos", "tax", "phylo"):
            raise ValueError(f"Unknown Loss 2 targ_type: '{self.loss2['targ']}', must be one of {{aligned, multipos, tax, phylo}}")
        
        if self.opt['lr']['sched'] not in ("exp", "plat", "cos", "coswr", "cosXexp", "coswrXexp"):
            raise ValueError(f"Unknown LR scheduler type: '{self.opt['lr']['sched']}', must be one of {{exp, plat, cos, coswr, cosXexp, coswrXexp}}")

        if self.eval_type not in ("validation", "test"):
            raise ValueError(f"Unknown eval_type: '{self.eval_type}', must be one of {{validation, test}}")

        if not 0.0 <= self.loss2["mix"] <= 1.0:
            raise ValueError(f"Secondary loss mix out of bounds: {self.loss2['mix']}, must be between 0.0 and 1.0")

        cfg_hw = get_config_hardware()
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            max_n_workers_gpu=cfg_hw.max_n_workers_gpu,
            prefetch_factor=cfg_hw.prefetch_factor,
        )
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        self.rdpath_trial = f"artifacts/{self.study_name}/{self.experiment_name}/{self.seed}"

        self.device = torch.device("cuda")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

def get_config_train():
    with open(paths["config"] / "train/train.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainConfig(**cfg_dict)
    cfg.lr_sched_params = get_config_lr_sched(cfg.opt['lr']['sched'])
    cfg.loss["cfg"] = get_config_loss(cfg)
    if cfg.loss2["mix"] != 0.0:
        cfg.loss2["cfg"] = get_config_loss(cfg, secondary=True)
    cfg.hw = get_config_hardware()
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
        cfg_loss = yaml.safe_load(f)

    if not (cfg_loss["logits"]["scale_init"] is None and cfg_loss["logits"]["bias_init"] is None):
        print(f"\nWARNING: loss_type = '{cfg_train_loss['type']}' and logit scale/bias overridden!\n")

    wting = cfg_train_loss.get("wting", False)
    focal = cfg_train_loss.get("focal", False)
    dyn_smr = cfg_train_loss.get("dyn_smr", False)

    if not wting and "class_weighting" in cfg_loss:
        del cfg_loss["class_weighting"]
    if not focal and "focal" in cfg_loss:
        del cfg_loss["focal"]

    if cfg_train_loss["type"] == "bce":
        cfg_loss["dyn_smr"] = dyn_smr
    elif "dyn_smr" in cfg_loss:
        del cfg_loss["dyn_smr"]

    return cfg_loss

@dataclass
class HardwareConfig:

    cached_imgs: str | None
    mixed_prec: bool
    act_chkpt: bool
    prefetch_factor: int
    max_n_workers_gpu: int | None
    persistent_workers_train: bool
    persistent_workers_eval: bool

def get_config_hardware():
    with open(paths["config"] / "hardware.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = HardwareConfig(**cfg_dict)
    return cfg

@dataclass
class EvalConfig:

    rdpath_trial: str | None
    save_crit: str  # model save criterion (only applicable if rdpath_trial != None)
    dataset: str
    split_name: str  # overridden if rdpath_trial is specified
    eval_type: str  # validation or test

    batch_size: int

    arch: dict
    loss: dict
    loss2: dict

    text_template: str

    hw: dict = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

        cfg_hw = get_config_hardware()
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            max_n_workers_gpu=cfg_hw.max_n_workers_gpu,
            prefetch_factor=cfg_hw.prefetch_factor,
        )
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

        self.device = torch.device("cuda")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

def get_config_eval(verbose=True):
    with open(paths["config"] / "eval.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EvalConfig(**cfg_dict)
    cfg.loss["cfg"]  = get_config_loss(cfg)
    if cfg.loss2["mix"] != 0.0:
        cfg.loss2["cfg"] = get_config_loss(cfg, secondary=True)
    cfg.hw = get_config_hardware()
    if verbose:
        PrintLog.init_eval(cfg)
    return cfg

@dataclass
class GenSplitConfig:

    seed: int
    split_name: str

    pct_partition: float
    pct_eval: float = field(init=False)
    pct_ood_tol: float
    size_dev: int

    nst_names: list
    nst_seps: list

    pos_filter: str | None = None
    ood_family_name: str | None = None

    def __post_init__(self):

        self.pct_eval = 2 * self.pct_partition

        if self.pos_filter not in (None, "dorsal"):
            raise ValueError(
                f"Unknown pos_filter: '{self.pos_filter}', must be one of {{None, dorsal}}"
            )

        if self.size_dev <= 0:
            raise ValueError(f"size_dev must be greater than 0, got {self.size_dev}")

        if len(self.nst_names) != len(self.nst_seps) + 1:
            raise ValueError(
                f"len(nst_names) ({len(self.nst_names)}) != "
                f"len(nst_seps) + 1 ({len(self.nst_seps)})"
            )

def get_config_splits():
    with open(paths["config"] / "splits.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = GenSplitConfig(**cfg_dict)
    return cfg