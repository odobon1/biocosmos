import torch
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import math
import yaml

from utils.utils import PrintLog, load_json, load_split, paths
from utils.hardware import compute_dataloader_workers_prefetch

import pdb


# Aliases used when building setting names from `baseline_overrides` keys/values (derived names
# for unnamed items, and combo-list name components): keys map through CFG_PARAM_ALIASES and
# values through CFG_PARAM_VALUE_ALIASES (per original key); anything without an alias passes
# through verbatim.
CFG_PARAM_ALIASES = {
    "batch_size": "BS",
    "loss.targ": "L1T",
    "loss2.targ": "L2T",
    "opt.lr.init": "LR",
    "loss2.mix": "Mix",
}

CFG_PARAM_VALUE_ALIASES = {
    "batch_size": {
        1_024: "1k",
        2_048: "2k",
        4_096: "4k",
        8_192: "8k",
        16_384: "16k",
        32_768: "32k",
    },
    "loss.targ": {
        "sp": "SP",
        "mp": "MP",
        "phylo": "hp",
    },
    "loss2.targ": {
        "phylo": "hp",
    },
}


# equivalent to OpenCLIP default train preprocessor
def _default_train_aug_cfg() -> dict:
    return {
        "rrcrop": {
            "scale_min": 0.9,
        },
        "hflip": False,
        "cjit": {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
            "prob": 0.0,
        },
        "sharpness": {"factor": 1.0, "prob": 0.0},
        "gblur": {
            "kernel_size": 3,
            "sigma": {"min": 0.0, "max": 0.0},
            "prob": 0.0,
        },
    }

@dataclass
class TrainConfig:

    campaign: str
    setting: str
    seed: int | None
    dataset: str
    split: str
    train_pt: str

    n_epochs: int | float
    chain_floor: int | None
    n_chkpts: int
    batch_size: int
    dv_batching: bool

    arch: dict
    dropout: dict
    freeze: dict
    htarg_shuf: bool
    loss: dict
    loss2: dict
    text_template: dict
    opt: dict

    dev: dict

    aug: dict = field(default_factory=_default_train_aug_cfg)
    manifold_viz: dict | None = None  # manifold_viz.yaml contents; resolved from the yaml when not supplied
    idx_seed: int = 0  # index of this trial's seed within the campaign seed sweep
    idx_trial: int | None = None  # 1-based position of this trial in the campaign launch order
    n_trials_total: int | None = None  # total planned trials in the campaign matrix

    eval_type: str = field(init=False)  # derived from train_pt: "train" -> "val", "trainval" -> None (eval skipped)

    hw: dict = field(default_factory=dict)  # hardware.yaml contents; campaign trials freeze it into the baseline, otherwise loaded live (converted to HardwareConfig in __post_init__)

    def __post_init__(self):

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

        if self.train_pt not in ("train", "trainval"):
            raise ValueError(f"Unknown train partition: '{self.train_pt}', must be one of {{train, trainval}}")

        self.eval_type = "val" if self.train_pt == "train" else None

        split = load_split(self.dataset, self.split)
        size_train = len(split.get_data(self.train_pt))

        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be greater than 0, got {self.n_epochs}")

        if self.chain_floor is not None and self.chain_floor <= 0:
            raise ValueError(f"chain_floor must be greater than 0 or null, got {self.chain_floor}")
        # chain-shuffle: when the train set is smaller than chain_floor, each dataloader pass chains
        # this many full shuffled permutations of it (ChainShuffleDistributedSampler)
        if self.chain_floor is not None and size_train < self.chain_floor:
            self.chain_perms = math.ceil(self.chain_floor / size_train)  # E_chain_nom
        else:
            self.chain_perms = None
        samps_pass_nom = size_train * (self.chain_perms or 1)  # X_chain_nom
        if self.batch_size > samps_pass_nom:
            raise ValueError(
                f"batch_size {self.batch_size} exceeds epoch size {samps_pass_nom} "
                f"(train set size {size_train} x {self.chain_perms or 1} chained permutations)"
            )
        # samples a pass actually consumes, batch-aligned (X_chain): without chain-shuffle drop_last
        # discards the remainder; with chain-shuffle a pass is a window of one continuous permutation
        # stream, so the remainder isn't dropped -- it leads the next pass (ChainShuffleDistributedSampler)
        self.samps_per_pass = samps_pass_nom - samps_pass_nom % self.batch_size
        # epochs' worth of samples credited per pass: E_chain
        self.epochs_per_pass = math.ceil(self.samps_per_pass / size_train)
        # samples per credited epoch: without chain-shuffle an epoch is one batch-truncated pass
        # (dropped samples don't count toward duration, so n_epochs never bleeds into an extra pass);
        # with chain-shuffle epochs stay nominal train-set permutations
        self.samps_per_epoch = self.samps_per_pass if self.chain_perms is None else size_train
        # epochs specify duration; everything downstream still drives on samples
        self.sample_volume = round(self.n_epochs * self.samps_per_epoch)
        self.n_passes = math.ceil(self.sample_volume / self.samps_per_pass)

        if self.n_chkpts <= 0:
            raise ValueError(f"n_chkpts must be greater than 0, got {self.n_chkpts}")
        # sample interval between checkpoint/eval thresholds; sample_volume is data-derived so it need
        # not divide evenly -- the trainer skips the last mid-train threshold and the final eval covers
        # it at sample_volume
        self.chkpt_interval = self.sample_volume // self.n_chkpts
        if self.chkpt_interval == 0:
            raise ValueError(f"n_chkpts ({self.n_chkpts}) exceeds sample_volume ({self.sample_volume})")

        for key, val in (
            ("opt.lr.init", self.opt["lr"]["init"]),
            ("opt.wd", self.opt["wd"]),
            ("loss.logits.scalar_lr_factor", self.loss["logits"]["scalar_lr_factor"]),
            ("loss2.logits.scalar_lr_factor", self.loss2["logits"]["scalar_lr_factor"]),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"{key} must be numeric, got {val!r} -- note YAML parses scientific notation "
                    f"without a decimal point (e.g. 1e-6) as a string; write 1.0e-6"
                )

        lr_warmup = self.opt["lr"]["warmup"]
        if not 0.0 <= lr_warmup < 1.0:
            raise ValueError(
                f"opt.lr.warmup must be a fraction of sample_volume in [0.0, 1.0), got {lr_warmup}"
            )

        n_trials_viz = self.dev["manifold_viz"]["n_trials"]
        if n_trials_viz < 0:
            raise ValueError(f"dev.manifold_viz.n_trials must be >= 0, got {n_trials_viz}")

        pooled_budget = self.dev["manifold_viz"]["pooled"]["budget"]
        if pooled_budget <= 0:
            raise ValueError(f"dev.manifold_viz.pooled.budget must be > 0, got {pooled_budget}")

        pca_bounds = self.dev["manifold_viz"]["pooled"]["pca_bounds"]
        if pca_bounds not in (None, "final"):
            raise ValueError(f"dev.manifold_viz.pooled.pca_bounds must be null or 'final', got {pca_bounds!r}")

        if self.dev["plot_every"] not in ("trial", "chkpt"):
            raise ValueError(f"dev.plot_every must be 'trial' or 'chkpt', got {self.dev['plot_every']!r}")

        if self.freeze["image"] and self.freeze["text"]:
            raise ValueError("Image and text encoders are both set to frozen!")

        if self.arch["siglip"]["vis_proj_head"] is None and self.dropout["siglip"]["proj_head"] > 0.0:
            raise ValueError(
                "dropout.siglip.proj_head > 0 requires arch.siglip.vis_proj_head to be 'linear' or 'mlp' "
                "(projection-head dropout needs a projection head)"
            )

        if self.htarg_shuf:
            phylo_active = self.loss["targ"] == "phylo" or (self.loss2["targ"] == "phylo" and self.loss2["mix"] != 0.0)
            if not phylo_active:
                raise ValueError(
                    "htarg_shuf=True requires an active phylo target: "
                    "loss.targ must be 'phylo', or loss2.targ must be 'phylo' with loss2.mix != 0.0"
                )
            if self.seed is None:
                raise ValueError("htarg_shuf=True requires a non-null seed (the shuffle permutation is derived from it and must match across DDP ranks)")

        if self.loss["crit"] not in ("infonce", "bce", "bif_bce"):
            raise ValueError(f"Unknown Loss 1 crit: '{self.loss['crit']}', must be one of {{infonce, bce, bif_bce}}")
        if self.loss2["crit"] not in ("infonce", "bce", "bif_bce"):
            raise ValueError(f"Unknown Loss 2 crit: '{self.loss2['crit']}', must be one of {{infonce, bce, bif_bce}}")
        
        if self.loss["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown Loss 1 sim_type: '{self.loss['sim']}', must be one of {{cos, geo1, geo2}}")
        if self.loss2["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown Loss 2 sim_type: '{self.loss2['sim']}', must be one of {{cos, geo1, geo2}}")
        
        if self.loss["targ"] not in ("sp", "mp", "tax", "phylo"):
            raise ValueError(f"Unknown Loss 1 targ_type: '{self.loss['targ']}', must be one of {{sp, mp, tax, phylo}}")
        if self.loss2["targ"] not in ("sp", "mp", "tax", "phylo"):
            raise ValueError(f"Unknown Loss 2 targ_type: '{self.loss2['targ']}', must be one of {{sp, mp, tax, phylo}}")

        if self.loss["logits"]["bce"]["center"] not in (None, "sim", "grad_proj", "grad_proj2"):
            raise ValueError(f"Unknown Loss 1 logits.bce.center: '{self.loss['logits']['bce']['center']}', must be one of {{null, sim, grad_proj, grad_proj2}}")
        if self.loss2["logits"]["bce"]["center"] not in (None, "sim", "grad_proj", "grad_proj2"):
            raise ValueError(f"Unknown Loss 2 logits.bce.center: '{self.loss2['logits']['bce']['center']}', must be one of {{null, sim, grad_proj, grad_proj2}}")

        if not 0.0 <= self.loss2["mix"] <= 1.0:
            raise ValueError(f"Secondary loss mix out of bounds: {self.loss2['mix']}, must be between 0.0 and 1.0")

        if self.aug.get("cjit", {}).get("prob", 0.0) == 0.0:
            self.aug.pop("cjit", None)
        if self.aug.get("sharpness", {}).get("prob", 0.0) == 0.0:
            self.aug.pop("sharpness", None)
        if self.aug.get("gblur", {}).get("prob", 0.0) == 0.0:
            self.aug.pop("gblur", None)

        # focal toggle: gamma 0.0 disables -> block dropped from the working config; downstream keys off presence
        for cfg_loss in (self.loss, self.loss2):
            if cfg_loss["wting"]["focal"]["gamma"] == 0.0:
                del cfg_loss["wting"]["focal"]

        self.hw = HardwareConfig(**self.hw)
        self.use_img_cache = self.hw.use_img_cache
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            batch_size=self.batch_size,
            model_type=self.arch["model_type"],
            max_n_workers_gpu=self.hw.max_n_workers_gpu,
            prefetch_factor=self.hw.prefetch_factor,
        )
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram = slurm_alloc["ram"]

        if self.hw.loss_chunk_size is not None:
            from utils.loss import chunking_supported  # local: avoid importing Bio.Phylo at config load
            if not chunking_supported(self.loss, self.loss2):  # tiled loss supports the full BCE-family config (bce/bif_bce); inert with infonce
                self.hw.loss_chunk_size = None
            else:
                # center: sim needs the full-batch sim mean IN-GRAPH per tile; the tiled path recovers it
                # exactly only through the cos-sim mean factorization mean(sim) = mean(img) . mean(txt)
                # (see utils/loss.py) -- geo sims have no such closed form
                for name, cfg_l in (("loss", self.loss), ("loss2", self.loss2)):
                    if name == "loss2" and self.loss2["mix"] == 0.0:
                        continue
                    if cfg_l["logits"]["bce"]["center"] == "sim" and cfg_l["sim"] != "cos":
                        raise ValueError(
                            f"{name}.logits.bce.center: sim requires {name}.sim: cos under hardware.loss_chunk_size "
                            f"(got {name}.sim: {cfg_l['sim']}): the tiled loss reproduces full-batch sim-centering "
                            f"exactly only via the cos mean factorization; use center: grad_proj/grad_proj2 or "
                            f"disable chunking"
                        )
                world_size = max(1, self.n_gpus)  # one rank per GPU (torchrun --nproc-per-node=auto)
                if self.batch_size % (world_size * self.hw.loss_chunk_size) != 0:
                    raise ValueError(
                        f"batch_size ({self.batch_size}) must be an exact multiple of world_size ({world_size}) "
                        f"x hardware.loss_chunk_size ({self.hw.loss_chunk_size}): the chunked loss shards the BxB rows "
                        f"into equal per-rank bands of whole C-row blocks"
                    )

        self.device = torch.device("cuda")

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__


def apply_train_debug_overrides(cfg_dict: dict) -> dict:
    cfg_dict = dict(cfg_dict)
    dev_cfg = cfg_dict.get("dev", {}) or {}
    if dev_cfg.get("debug_mode", False):
        cfg_dict = apply_overrides(cfg_dict, dev_cfg["debug"])  # keys are dot-paths into this config (e.g. opt.lr.warmup)
        cfg_dict["split"] = "dev"  # forced after the overrides: debug always runs on the dev split
    return cfg_dict

def _set_by_dot_path(cfg_dict: dict, key_path: str, value) -> None:
    """Overwrite an existing config field addressed by dot-path.

    Every segment must already be declared: overrides replace declared fields, never create them.
    A typo'd or stale key (a renamed param still swept in a campaign's `baseline_overrides`) would
    otherwise land silently in a field nothing reads, and the campaign runs the baseline value
    under a setting name advertising the override.
    """
    keys = [key for key in key_path.split(".") if key]
    if not keys:
        raise ValueError(f"Invalid dot-style override key: '{key_path}'")

    cursor = cfg_dict
    for depth, key in enumerate(keys):
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError(
                f"Unknown override key '{key_path}': '{'.'.join(keys[:depth + 1])}' is not a config field."
            )
        if depth == len(keys) - 1:
            cursor[key] = deepcopy(value)
        else:
            cursor = cursor[key]

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

def apply_model_specific_opt_defaults(cfg_dict: dict, model_specific_config: dict | None = None) -> dict:
    cfg_out = deepcopy(cfg_dict)
    opt = cfg_out.get("opt", {})

    if not isinstance(opt, dict):
        raise ValueError("Config field 'opt' must be a dict.")

    needs_wd = opt.get("wd") is None
    needs_beta2 = opt.get("beta2") is None
    if not (needs_wd or needs_beta2):
        return cfg_out

    arch = cfg_out.get("arch", {})
    if not isinstance(arch, dict) or "model_type" not in arch:
        raise ValueError("Config field 'arch/model_type' is required to resolve model-specific defaults.")

    model_type = arch["model_type"]
    family = _resolve_model_family(model_type)
    if model_specific_config is None:  # load live when no snapshot supplied; campaign trials pass the frozen snapshot
        model_specific_config = load_model_specific_config_dict()
    family_defaults = model_specific_config.get(family)

    if not isinstance(family_defaults, dict):
        raise ValueError(f"Missing model hyperparameter defaults for family '{family}'.")

    if needs_wd:
        if "wd" not in family_defaults:
            raise ValueError(f"Missing '{family}/wd' in model hyperparameter defaults.")
        opt["wd"] = deepcopy(family_defaults["wd"])

    if needs_beta2:
        if "beta2" not in family_defaults:
            raise ValueError(f"Missing '{family}/beta2' in model hyperparameter defaults.")
        opt["beta2"] = deepcopy(family_defaults["beta2"])

    cfg_out["opt"] = opt
    return cfg_out

def get_config_train(cfg_dict: dict) -> TrainConfig:
    setting_overrides = cfg_dict.pop("_setting_overrides", None)
    model_specific = cfg_dict.pop("model_specific", None)  # campaign trials inject the frozen snapshot; otherwise read live
    cfg_dict = apply_train_debug_overrides(cfg_dict)
    cfg_dict = apply_model_specific_opt_defaults(cfg_dict, model_specific)
    if setting_overrides is not None:
        cfg_dict = apply_overrides(cfg_dict, setting_overrides)
    cfg_dict.setdefault("hw", load_hardware_config_dict())  # campaign trials freeze hw into the baseline; otherwise load live
    cfg = TrainConfig(**cfg_dict)
    if cfg.manifold_viz is None:  # load live when campaign trials haven't injected the snapshot
        cfg.manifold_viz = asdict(get_config_manifold_viz())
    return cfg


@dataclass
class HardwareConfig:

    mixed_prec: bool  # bf16 autocast mixed precision for training and validation (bf16 needs no GradScaler)
    act_chkpt: bool
    loss_chunk_size: int | None  # row-block height for the global-batch (BxB) loss, row-band-sharded across ranks; None -> full BxB (no tiling/sharding). See config/hardware.yaml.
    cudnn_benchmark: bool  # torch.backends.cudnn.benchmark
    prefetch_factor: int
    max_n_workers_gpu: int | None
    pin_memory: bool  # dataloader pinned host RAM for faster host -> GPU copies
    persistent_workers: dict  # {train: bool, eval: bool}; keep dataloader workers alive across epochs
    use_img_cache: bool  # read images from the prebuilt pack (tools/build_img_cache.py), staged to node-local scratch, instead of per-sample files on the shared FS
    eval: dict  # {map_chunk_size: {img2img, cross_modal}, tsne_chunk_log2: int} -- mAP sim-matrix chunking + t-SNE GPU-buffer tiling (buffer = 2^X fp32)
    ram_poll_interval: float  # seconds between cgroup RAM polls by the peak-RAM tracker (utils/hardware.py)
    pg_timeout: int  # NCCL PG watchdog timeout in seconds; passed to setup_ddp
    max_retries: int  # campaign runner: consecutive no-progress trial retries before giving up

    def __post_init__(self):
        c = self.loss_chunk_size
        if c is not None and not (isinstance(c, int) and c > 0 and (c & (c - 1)) == 0):
            raise ValueError(f"hardware.loss_chunk_size must be null or a positive power of 2; got {c!r}")


def load_hardware_config_dict() -> dict:
    with open(paths["config"] / "hardware.yaml") as f:
        return yaml.safe_load(f)

def get_config_hardware():
    return HardwareConfig(**load_hardware_config_dict())


@dataclass
class EvalConfig:

    rdpath_model: str | None
    dataset: str
    split: str
    eval_type: str

    batch_size: int

    arch: dict

    text_template: str

    hw: dict = field(init=False, default_factory=dict)
    
    def __post_init__(self):

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

        if self.eval_type not in ("val", "test"):
            raise ValueError(f"Unknown eval partition: '{self.eval_type}', must be one of {{val, test}}")

        # standalone base-model eval (rdpath_model: null) defaults to the released arch -- eval.yaml
        # exposes no non_causal/vis_proj_head knobs; checkpoint eval overrides from the setting's config.json below
        self.arch["clip"] = {"non_causal": False}
        self.arch["siglip"] = {"vis_proj_head": None}

        if self.rdpath_model is not None:
            dpath_model = paths["root"] / self.rdpath_model
            fpath_model = dpath_model / "model.pt"
            if not fpath_model.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {fpath_model}")

            fpath_metadata_trial = dpath_model / "../../trial_metadata.json"
            fpath_config_setting = dpath_model / "../../../../config.json"
            config_setting = load_json(fpath_config_setting)
            metadata_trial = load_json(fpath_metadata_trial)

            self.arch["model_type"] = config_setting["arch"]["model_type"]  # override model_type
            # config.json carries only the checkpoint family's arch section (the other family's is
            # pruned as inert); the absent one keeps its released default set above
            if "clip" in config_setting["arch"]:
                self.arch["clip"]["non_causal"] = config_setting["arch"]["clip"]["non_causal"]  # override non_causal
            if "siglip" in config_setting["arch"]:
                self.arch["siglip"]["vis_proj_head"] = config_setting["arch"]["siglip"]["vis_proj_head"]  # override vis_proj_head (projection head must match checkpoint)
            self.dataset = metadata_trial["dataset"]  # override dataset
            self.split = metadata_trial["split"]  # override split

        # after the arch override: the worker RAM bound keys off the checkpoint's actual model_type
        cfg_hw = get_config_hardware()
        self.use_img_cache = cfg_hw.use_img_cache
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch(
            batch_size=self.batch_size,
            model_type=self.arch["model_type"],
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


def get_config_eval(verbose=True):
    with open(paths["config"] / "eval.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EvalConfig(**cfg_dict)
    cfg.hw = get_config_hardware()
    if verbose:
        PrintLog.init_eval(cfg)
    return cfg


# per-dataset manifold-viz scatter marker size (nymph/lepid have many points -> smaller markers)
DATASET2MARKER_SIZE = {
    "bryo": 7,
    "cub": 7,
    "lepid": 1,
    "nymph": 2,
}


@dataclass
class ManifoldVizConfig:

    n_stoch_layers: int
    eval_duration: int
    bg_color: str | None
    
    tsne: dict = field(default_factory=dict)
    color: dict = field(default_factory=dict)

    def __post_init__(self):
        # n_stoch_layers sample draw-order shuffles per eval frame: =1 -> static PNG, >1 -> strobe GIF
        if self.n_stoch_layers < 1:
            raise ValueError(f"n_stoch_layers must be >= 1, got {self.n_stoch_layers}")


def load_manifold_viz_config_dict() -> dict:
    with open(paths["config"] / "manifold_viz.yaml") as f:
        return yaml.safe_load(f)

def get_config_manifold_viz():
    return ManifoldVizConfig(**load_manifold_viz_config_dict())


@dataclass
class StatsConfig:
    """stats.yaml contents -- campaign stats-artifact rendering settings (stats tables + metrics workbooks).
    Render-time only: read live at each render (trial completion / tools.regen_stats), never frozen
    into campaign baselines, so edits apply to the next re-render of any campaign."""

    spread_type: str  # {std, ste}
    bold_high: bool
    ordered: bool
    heatmap: bool  # (True) shade score cells by value over a fixed 0 -> 100 range
    supp_scores: dict  # {primitive: bool, n_shot: bool}; supplemental score columns appended right of the composite columns
    baseline_overrides: bool  # (True) append a "Baseline Overrides" config table to each xlsx sheet

    def __post_init__(self):

        if self.spread_type not in ("std", "ste"):
            raise ValueError(f"Unknown stats spread_type: '{self.spread_type}', must be one of {{std, ste}}")

        if set(self.supp_scores) != {"primitive", "n_shot"}:
            raise ValueError(f"stats supp_scores must have exactly the keys {{primitive, n_shot}}, got {sorted(self.supp_scores)}")


def load_stats_config_dict() -> dict:
    with open(paths["config"] / "stats.yaml") as f:
        return yaml.safe_load(f)

def get_config_stats():
    return StatsConfig(**load_stats_config_dict())


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
