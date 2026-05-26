import torch
from dataclasses import dataclass, field
from copy import deepcopy
import math
import yaml

from utils.utils import PrintLog, load_json, load_split, paths
from utils.hardware import compute_dataloader_workers_prefetch

import pdb


@dataclass
class TrainConfig:

    campaign_name: str
    setting_name: str
    seed: int | None
    dataset: str
    split_name: str

    sample_volume: int
    eval_every: int
    chkpt_every: int
    batch_size: int
    dv_batching: bool

    dev: dict
    arch: dict
    img_norm: str
    loss: dict
    loss2: dict
    opt: dict
    freeze: dict
    text_template: dict

    logging: bool
    metrics_plot_every_batches: int = 100
    
    eval_type: str = "validation"

    hw: dict = field(init=False, default_factory=dict)

    def __post_init__(self):

        split = load_split(self.split_name, dataset_name=self.dataset)
        size_train = len(split.data_indexes["train"])
        samps_per_epoch = size_train - size_train % self.batch_size
        self.n_epochs = math.ceil(self.sample_volume / samps_per_epoch)

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")
        
        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be greater than 0, got {self.eval_every}")

        if self.img_norm not in ("default", "dataset"):
            raise ValueError(f"Unknown img_norm option: '{self.img_norm}', must be one of {{default, dataset}}")

        if self.loss["type"] not in ("infonce1", "infonce2", "bce"):
            raise ValueError(f"Unknown Loss 1 Type: '{self.loss['type']}', must be one of {{infonce1, infonce2, bce}}")
        if self.loss2["type"] not in ("infonce1", "infonce2", "bce"):
            raise ValueError(f"Unknown Loss 2 Type: '{self.loss2['type']}', must be one of {{infonce1, infonce2, bce}}")
        
        if self.loss["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown Loss 1 sim_type: '{self.loss['sim']}', must be one of {{cos, geo1, geo2}}")
        if self.loss2["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown Loss 2 sim_type: '{self.loss2['sim']}', must be one of {{cos, geo1, geo2}}")
        
        if self.loss["targ"] not in ("aligned", "multipos", "tax", "phylo"):
            raise ValueError(f"Unknown Loss 1 targ_type: '{self.loss['targ']}', must be one of {{aligned, multipos, tax, phylo}}")
        if self.loss2["targ"] not in ("aligned", "multipos", "tax", "phylo"):
            raise ValueError(f"Unknown Loss 2 targ_type: '{self.loss2['targ']}', must be one of {{aligned, multipos, tax, phylo}}")
        
        if not 0.0 <= self.loss2["mix"] <= 1.0:
            raise ValueError(f"Secondary loss mix out of bounds: {self.loss2['mix']}, must be between 0.0 and 1.0")

        if self.opt["lr"]["sched"] not in ("cos",):
            raise ValueError(f"Unknown LR scheduler type: '{self.opt['lr']['sched']}', must be one of {{cos}}")

        if self.freeze["image"] and self.freeze["text"]:
            raise ValueError("Image and text encoders are both set to frozen!")

        if self.metrics_plot_every_batches <= 0:
            raise ValueError(f"metrics_plot_every_batches must be greater than 0, got {self.metrics_plot_every_batches}")

        if self.eval_type not in ("validation", "test"):
            raise ValueError(f"Unknown eval_type: '{self.eval_type}', must be one of {{validation, test}}")

        cfg_hw = get_config_hardware()
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            max_n_workers_gpu=cfg_hw.max_n_workers_gpu,
            prefetch_factor=cfg_hw.prefetch_factor,
        )
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram = slurm_alloc["ram"]

        self.rdpath_trial = f"artifacts/{self.campaign_name}/{self.setting_name}/{self.dataset}/{self.seed}"

        self.device = torch.device("cuda")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__


def apply_train_debug_overrides(cfg_dict: dict) -> dict:
    cfg_dict = dict(cfg_dict)
    dev_cfg = cfg_dict.get("dev", {}) or {}
    if dev_cfg.get("debug_mode", False):
        cfg_dict["split_name"] = "dev"
        cfg_dict["sample_volume"] = 20_000
        cfg_dict["eval_every"] = 5_000
    return cfg_dict

def _deep_merge_dict(base: dict, updates: dict) -> dict:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out

def _set_by_slash_path(cfg_dict: dict, key_path: str, value) -> None:
    keys = [key for key in key_path.split("/") if key]
    if not keys:
        raise ValueError(f"Invalid slash-style override key: '{key_path}'")

    cursor = cfg_dict
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = deepcopy(value)

def _has_nested_path(cfg_dict: dict, key_path: str) -> bool:
    keys = [key for key in key_path.split("/") if key]
    if not keys:
        return False

    cursor = cfg_dict
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    return True

def apply_overrides(cfg_dict: dict, overrides: dict | None) -> dict:
    """
    Apply overrides with deterministic precedence:
    1) Nested dict keys are applied first and take precedence.
    2) Slash-style keys are fallback only.
    """
    if not overrides:
        return deepcopy(cfg_dict)

    nested_updates = {}
    slash_updates = {}
    for key, value in overrides.items():
        if isinstance(key, str) and "/" in key:
            slash_updates[key] = value
        else:
            nested_updates[key] = value

    merged = _deep_merge_dict(cfg_dict, nested_updates)
    for key_path, value in slash_updates.items():
        if _has_nested_path(nested_updates, key_path):
            continue
        _set_by_slash_path(merged, key_path, value)

    return merged

def load_train_config_dict() -> dict:
    with open(paths["config"] / "train/train.yaml") as f:
        return yaml.safe_load(f)

def load_model_specific_config_dict() -> dict:
    with open(paths["config"] / "train/model_specific.yaml") as f:
        return yaml.safe_load(f)

def _resolve_model_family(model_type: str) -> str:
    model_type_lower = model_type.lower()
    if "siglip" in model_type_lower:
        return "siglip"
    if "clip" in model_type_lower:
        return "clip"
    raise ValueError(
        f"Could not resolve model family for model_type '{model_type}'. "
        "Expected a CLIP or SigLIP model type."
    )

def apply_model_specific_opt_defaults(cfg_dict: dict) -> dict:
    cfg_out = deepcopy(cfg_dict)
    opt = cfg_out.get("opt", {})

    if not isinstance(opt, dict):
        raise ValueError("Config field 'opt' must be a dict.")

    needs_l2reg = opt.get("l2reg") is None
    needs_beta2 = opt.get("beta2") is None
    if not (needs_l2reg or needs_beta2):
        return cfg_out

    arch = cfg_out.get("arch", {})
    if not isinstance(arch, dict) or "model_type" not in arch:
        raise ValueError("Config field 'arch/model_type' is required to resolve model-specific defaults.")

    model_type = arch["model_type"]
    family = _resolve_model_family(model_type)
    model_specific_config = load_model_specific_config_dict()
    family_defaults = model_specific_config.get(family)

    if not isinstance(family_defaults, dict):
        raise ValueError(f"Missing model hyperparameter defaults for family '{family}'.")

    if needs_l2reg:
        if "l2reg" not in family_defaults:
            raise ValueError(f"Missing '{family}/l2reg' in model hyperparameter defaults.")
        opt["l2reg"] = deepcopy(family_defaults["l2reg"])

    if needs_beta2:
        if "beta2" not in family_defaults:
            raise ValueError(f"Missing '{family}/beta2' in model hyperparameter defaults.")
        opt["beta2"] = deepcopy(family_defaults["beta2"])

    cfg_out["opt"] = opt
    return cfg_out

def build_train_config(cfg_dict: dict) -> TrainConfig:
    cfg_dict = apply_train_debug_overrides(cfg_dict)
    cfg_dict = apply_model_specific_opt_defaults(cfg_dict)
    cfg = TrainConfig(**cfg_dict)
    cfg.lr_sched_params = get_config_lr_sched(cfg.opt['lr']['sched'])
    cfg.loss["cfg"] = get_config_loss(cfg)
    if cfg.loss2["mix"] != 0.0:
        cfg.loss2["cfg"] = get_config_loss(cfg, secondary=True)
    cfg.hw = get_config_hardware()
    return cfg

def get_config_train(cfg_dict: dict | None = None):
    if cfg_dict is None:
        cfg_dict = load_train_config_dict()
    return build_train_config(cfg_dict)

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
    img_norm: str
    loss: dict
    loss2: dict

    text_template: str

    hw: dict = field(init=False, default_factory=dict)
    
    def __post_init__(self):

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

        if self.img_norm not in ("default", "dataset"):
            raise ValueError(f"Unknown img_norm option: '{self.img_norm}', must be one of {{default, dataset}}")

        cfg_hw = get_config_hardware()
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            max_n_workers_gpu=cfg_hw.max_n_workers_gpu,
            prefetch_factor=cfg_hw.prefetch_factor,
        )
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        if self.rdpath_trial is not None:
            metadata_setting = load_json(paths["root"] / self.rdpath_trial / "../../metadata_setting.json")
            self.arch['model_type'] = metadata_setting["arch"]["model_type"]  # override model_type
            self.arch['non_causal'] = metadata_setting["arch"]["non_causal"]  # override non_causal
            self.loss['type']       = metadata_setting["loss"]["type"]  # override loss_type

            if "loss2" in metadata_setting:
                self.loss2.update(metadata_setting["loss2"])
            else:
                self.loss2["mix"] = 0.0

            metadata_campaign  = load_json(paths["root"] / self.rdpath_trial / "../../../metadata_campaign.json")
            self.split_name = metadata_campaign["split_name"]

        self.device = torch.device("cuda")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

def get_config_eval(verbose=True):
    with open(paths["config"] / "eval.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EvalConfig(**cfg_dict)
    cfg.loss["cfg"] = get_config_loss(cfg)
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