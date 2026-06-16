import torch
from dataclasses import dataclass, field
from copy import deepcopy
import math
import yaml

from utils.utils import PrintLog, load_json, load_split, paths
from utils.hardware import compute_dataloader_workers_prefetch

import pdb


# equivalent to OpenCLIP default train preprocessor
def _default_train_aug_cfg() -> dict:
    return {
        "rrcrop": {
            "scale": [0.9, 1.0],
            "ratio": [0.75, 1.3333],
        },
        "hflip": False,
        "cjit": {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
        },
        "cjit_prob": 0.0,
        "sharpness": [1.0, 1.0],
        "sharpness_prob": 0.0,
        "gblur": {
            "kernel_size": 3,
            "sigma": [0.0, 0.0],
        },
        "gblur_prob": 0.0,
    }

@dataclass
class TrainConfig:

    campaign: str
    setting: str
    seed: int | None
    dataset: str
    split: str
    train_pt: str

    sample_volume: int
    chkpt_every: int
    batch_size: int
    dv_batching: bool

    arch: dict
    freeze: dict
    loss: dict
    loss2: dict
    text_template: dict
    img_norm: str
    opt: dict

    dev: dict

    standalone: bool = True
    aug: dict = field(default_factory=_default_train_aug_cfg)

    eval_type: str = field(init=False)  # derived from train_pt: "train" -> "val", "trainval" -> None (eval skipped)

    hw: dict = field(init=False, default_factory=dict)

    def __post_init__(self):

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

        if self.train_pt not in ("train", "trainval"):
            raise ValueError(f"Unknown train partition: '{self.train_pt}', must be one of {{train, trainval}}")

        self.eval_type = "val" if self.train_pt == "train" else None

        split = load_split(self.dataset, self.split)
        size_train = len(split.data_indexes[self.train_pt])
        if self.batch_size > size_train:
            raise ValueError(f"batch_size {self.batch_size} exceeds training set size {size_train}")
        samps_per_epoch = size_train - size_train % self.batch_size
        self.n_epochs = math.ceil(self.sample_volume / samps_per_epoch)
        
        if self.chkpt_every <= 0:
            raise ValueError(f"chkpt_every must be greater than 0, got {self.chkpt_every}")

        if self.freeze["image"] and self.freeze["text"]:
            raise ValueError("Image and text encoders are both set to frozen!")

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

        if self.img_norm not in ("default", "dataset"):
            raise ValueError(f"Unknown img_norm option: '{self.img_norm}', must be one of {{default, dataset}}")

        for loss in [self.loss, self.loss2]:
            logits = loss["logits"]
            if logits["scale_init"] is not None or logits["bias_init"] is not None:
                print(f"\nWARNING: logit scale/bias overridden!\n")

        if self.aug.get("cjit_prob", 0.0) == 0.0:
            self.aug.pop("cjit", None)
            self.aug.pop("cjit_prob", None)
        if self.aug.get("sharpness_prob", 0.0) == 0.0:
            self.aug.pop("sharpness", None)
            self.aug.pop("sharpness_prob", None)
        if self.aug.get("gblur_prob", 0.0) == 0.0:
            self.aug.pop("gblur", None)
            self.aug.pop("gblur_prob", None)

        cfg_hw = get_config_hardware()
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            max_n_workers_gpu=cfg_hw.max_n_workers_gpu,
            prefetch_factor=cfg_hw.prefetch_factor,
        )
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram = slurm_alloc["ram"]

        self.device = torch.device("cuda")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__


def apply_train_debug_overrides(cfg_dict: dict) -> dict:
    cfg_dict = dict(cfg_dict)
    dev_cfg = cfg_dict.get("dev", {}) or {}
    if dev_cfg.get("debug_mode", False):
        cfg_dict["split"] = "dev"
        cfg_dict["sample_volume"] = dev_cfg["debug"]["sample_volume"]
        cfg_dict["chkpt_every"] = dev_cfg["debug"]["chkpt_every"]
        cfg_dict["batch_size"] = dev_cfg["debug"]["batch_size"]
    return cfg_dict

def _set_by_dot_path(cfg_dict: dict, key_path: str, value) -> None:
    keys = [key for key in key_path.split(".") if key]
    if not keys:
        raise ValueError(f"Invalid dot-style override key: '{key_path}'")

    cursor = cfg_dict
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = deepcopy(value)

def apply_overrides(cfg_dict: dict, overrides: dict | None) -> dict:
    if not overrides:
        return deepcopy(cfg_dict)

    merged = deepcopy(cfg_dict)
    for key_path, value in overrides.items():
        _set_by_dot_path(merged, key_path, value)

    return merged

def load_train_config_dict() -> dict:
    with open(paths["config"] / "train.yaml") as f:
        return yaml.safe_load(f)

def load_model_specific_config_dict() -> dict:
    with open(paths["config"] / "model_specific.yaml") as f:
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
    setting_overrides = cfg_dict.pop("_setting_overrides", None)
    cfg_dict = apply_train_debug_overrides(cfg_dict)
    cfg_dict = apply_model_specific_opt_defaults(cfg_dict)
    if setting_overrides is not None:
        cfg_dict = apply_overrides(cfg_dict, setting_overrides)
    cfg = TrainConfig(**cfg_dict)
    cfg.hw = get_config_hardware()
    return cfg

def get_config_train(cfg_dict: dict | None = None):
    if cfg_dict is None:
        cfg_dict = load_train_config_dict()
    return build_train_config(cfg_dict)


@dataclass
class HardwareConfig:

    mixed_prec: bool
    act_chkpt: bool
    prefetch_factor: int
    max_n_workers_gpu: int | None
    persistent_workers_train: bool
    persistent_workers_eval: bool
    chunk_size: dict


def get_config_hardware():
    with open(paths["config"] / "hardware.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = HardwareConfig(**cfg_dict)
    return cfg


@dataclass
class EvalConfig:

    rfpath_model: str | None
    dataset: str
    split: str
    eval_type: str

    batch_size: int

    arch: dict
    img_norm: str

    text_template: str

    hw: dict = field(init=False, default_factory=dict)
    
    def __post_init__(self):

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

        if self.eval_type not in ("val", "test"):
            raise ValueError(f"Unknown eval partition: '{self.eval_type}', must be one of {{val, test}}")

        if self.img_norm not in ("default", "dataset"):
            raise ValueError(f"Unknown img_norm option: '{self.img_norm}', must be one of {{default, dataset}}")

        if self.rfpath_model is None and self.img_norm == "dataset":
            raise ValueError("img_norm='dataset' requires a model checkpoint (rfpath_model) to infer which partition's norm stats were used during training")

        cfg_hw = get_config_hardware()
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            max_n_workers_gpu=cfg_hw.max_n_workers_gpu,
            prefetch_factor=cfg_hw.prefetch_factor,
        )
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram = slurm_alloc["ram"]

        if self.rfpath_model is not None:
            fpath_model = paths["root"] / self.rfpath_model
            if not fpath_model.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {fpath_model}")
            
            fpath_metadata_trial = fpath_model.parent / "../../trial_metadata.json"
            fpath_metadata_setting = fpath_model.parent / "../../../../setting_metadata.json"
            metadata_setting = load_json(fpath_metadata_setting)
            metadata_trial = load_json(fpath_metadata_trial)

            self.arch["model_type"] = metadata_setting["arch"]["model_type"]  # override model_type
            self.arch["non_causal"] = metadata_setting["arch"]["non_causal"]  # override non_causal
            self.img_norm = metadata_setting["img_norm"]  # override img_norm
            self.dataset = metadata_trial["dataset"]  # override dataset
            self.split = metadata_trial["split"]  # override split

        self.device = torch.device("cuda")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__


def get_config_eval(verbose=True):
    with open(paths["config"] / "eval.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EvalConfig(**cfg_dict)
    cfg.hw = get_config_hardware()
    if verbose:
        PrintLog.init_eval(cfg)
    return cfg


@dataclass
class GenSplitConfig:

    seed: int
    split: str

    pct_partition: float
    pct_ood_tol: float
    size_dev: int

    nst_names: list
    nst_seps: list

    pos_filter: str | None

    dev: dict

    def __post_init__(self):

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


def apply_splits_debug_overrides(cfg_dict: dict) -> dict:
    cfg_dict = dict(cfg_dict)
    dev_cfg = cfg_dict.get("dev", {}) or {}
    if dev_cfg.get("debug_mode", False):
        cfg_dict["pct_ood_tol"] = dev_cfg["debug"]["pct_ood_tol"]
    return cfg_dict

def get_config_splits():
    with open(paths["config"] / "split_gen.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict = apply_splits_debug_overrides(cfg_dict)
    cfg = GenSplitConfig(**cfg_dict)
    return cfg