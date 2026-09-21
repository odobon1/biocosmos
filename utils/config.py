import torch
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import math
import yaml

from utils.utils import PrintLog, load_split, paths
from utils.hardware import compute_dataloader_workers_prefetch

import pdb


def load_aliases_config_dict() -> dict:
    with open(paths["config"] / "trial" / "aliases.yaml") as f:
        return yaml.safe_load(f)

# Aliases used when building arm / coord names from `ablation_arms` / `hpo_coords` keys/values (derived
# names for unnamed items, and combo-list name components): keys map through CFG_PARAM_ALIASES and
# values through CFG_PARAM_VALUE_ALIASES (per original key), falling back to
# CFG_UNIVERSAL_VALUE_ALIASES (key-independent) when no per-key alias exists; anything without
# an alias passes through verbatim. Read once at import from config/trial/aliases.yaml; a campaign names
# its arm / coord dirs from its own FROZEN copy instead (campaign_runner._alias_tables), so they can
# never shift mid-campaign.
_CFG_ALIASES = load_aliases_config_dict()["config"]
CFG_PARAM_ALIASES = _CFG_ALIASES["param"]
CFG_PARAM_VALUE_ALIASES = _CFG_ALIASES["param_value"]
CFG_UNIVERSAL_VALUE_ALIASES = _CFG_ALIASES["universal_value"]
# config file stem -> the cfg key a campaign's `ablation_arms` / `hpo_coords` overrides reach its
# contents through (config/trial/aliases.yaml's config.file); only the overridable configs are listed
CFG_FILE_ALIASES = _CFG_ALIASES["file"]
# eval group key -> the name it is reported under (config/trial/aliases.yaml's eval_groups); a group
# without an alias passes through verbatim. Read live like CFG_FILE_ALIASES rather than from a campaign's
# frozen copy: these are display labels only, so relabelling one re-renders an existing campaign under it.
CFG_EVAL_GROUP_ALIASES = load_aliases_config_dict()["eval_groups"]

# Every eval group a trial can score, in reporting order (utils.eval maps each to the gallery its scores
# are computed against x the rollup over that gallery's classes). `native` is always scored;
# config/trial/reporting.yaml's `eval` block toggles the rest.
EVAL_GROUPS = ("native", "native_macro", "joint", "joint_macro")

def eval_groups(cfg_reporting: dict) -> dict:
    """The eval groups in play, {group key: reported name}, in EVAL_GROUPS order: `native`, always in,
    plus every group reporting.yaml's `eval` switches on. The KEY names the group's scores subtree and
    every per-group artifact file/dir; the name is CFG_EVAL_GROUP_ALIASES' display label."""
    return {
        key: CFG_EVAL_GROUP_ALIASES.get(key, key)
        for key in EVAL_GROUPS
        if key == "native" or cfg_reporting["eval"][key]
    }

def inject_snapshots(cfg_dict: dict, cfg_snapshot: dict) -> None:
    """Lay every sibling snapshot of a frozen campaign config onto a trial's config dict (`train` is
    the base config itself, not a sibling)."""
    for stem, key in CFG_FILE_ALIASES.items():  # the overridable ones, under their campaign-facing key
        cfg_dict[key] = cfg_snapshot[stem]
    # campaign-global: no arm / coord sweeps them. dev.yaml lands under `dev_overrides`, not `dev` --
    # that key is train.yaml's on/off switch, and a (truthy) dict there would force dev overrides on
    cfg_dict["manifold_viz"] = cfg_snapshot["manifold_viz"]
    cfg_dict["model_specific"] = cfg_snapshot["model_specific"]
    cfg_dict["dataset_specific"] = cfg_snapshot["dataset_specific"]
    cfg_dict["augmentation"] = cfg_snapshot["augmentation"]
    cfg_dict["reporting"] = cfg_snapshot["reporting"]
    cfg_dict["dev_overrides"] = cfg_snapshot["dev"]


# OpenCLIP's default train preprocessor (open_clip.transform.AugmentationCfg: scale (0.9, 1.0), every
# photometric aug off); the config `aug: openclip` selects it (config/trial/train/train.yaml). OpenAI published no
# training-aug config of its own -- only the inference transform, the CLIP paper giving a random square
# crop as the sole augmentation -- so these concrete values are OpenCLIP's
def _aug_cfg_openclip() -> dict:
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
            "kernel_size": 1,
            "sigma": {"min": 0.0, "max": 1.0},
            "prob": 0.0,
        },
    }

@dataclass
class TrainConfig:

    campaign: str
    phase: str  # the campaign phase whose tree the trial writes to: '_screen' | 'qual' | 'trainval' (campaign_runner)
    arm: str
    coord: str
    seed: int | None
    dataset: str
    split: str
    train_pt: str

    n_epochs: int
    chain_floor: int | None
    n_chkpts: int
    batch_size: int
    dv_batching: bool

    arch: dict
    dropout: dict
    freeze: dict
    htarg: dict
    loss: dict
    text_template: dict
    opt: dict
    lr: dict

    dev: bool  # {true, false}; (true) lay config/trial/train/dev.yaml's overrides over the config and force the dev split
    reporting: dict  # reporting.yaml contents (get_config_train reads it live when not supplied)
    kill_thresh: float | None  # {null, (0.0, 1.0)}; fraction of the run at which a trial that has not beaten its base eval is killed
    del_base_eval_cache: str | None  # {null, campaign, trial}; when the campaign runner deletes base_eval_cache/

    aug: str = "openclip"  # {openclip, custom}; which augmentation config the trial trains under -> self.aug_cfg
    augmentation: dict | None = None  # augmentation.yaml contents, read by aug: custom; resolved from the yaml when not supplied
    manifold_viz: dict | None = None  # manifold_viz.yaml contents; resolved from the yaml when not supplied
    idx_seed: int = 0  # index of this trial's seed within the campaign seed sweep
    idx_trial: int | None = None  # 1-based position of this trial in the campaign launch order
    n_trials_total: int | None = None  # total planned trials in the campaign matrix
    chkpt_stop: int | None = None  # trainval phase: stop training once this checkpoint index (1..n_chkpts, the pick's
    # qual-selected one) is reached instead of running to sample_volume; the LR schedule keeps its full horizon

    hw: dict = field(default_factory=dict)  # hardware.yaml contents; campaign trials freeze it into the baseline, otherwise loaded live (converted to HardwareConfig in __post_init__)

    def __post_init__(self):

        if self.dataset not in ("bryo", "cub", "lepid", "nymph"):
            raise ValueError(f"Unknown dataset: '{self.dataset}', must be one of {{bryo, cub, lepid, nymph}}")

        if self.train_pt not in ("train", "trainval"):
            raise ValueError(f"Unknown train partition: '{self.train_pt}', must be one of {{train, trainval}}")

        split = load_split(self.dataset, self.split)
        size_train = len(split.get_data(self.train_pt))

        # bool guard: True/False are ints; a null n_epochs / n_chkpts reaches here only when it skipped
        # dataset-specific resolution (get_config_train / apply_dataset_specific_defaults)
        if isinstance(self.n_epochs, bool) or not isinstance(self.n_epochs, int):
            raise ValueError(f"n_epochs must be an int, got {self.n_epochs!r}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be greater than 0, got {self.n_epochs}")
        if isinstance(self.n_chkpts, bool) or not isinstance(self.n_chkpts, int):
            raise ValueError(f"n_chkpts must be an int, got {self.n_chkpts!r}")

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
        self.sample_volume = self.n_epochs * self.samps_per_epoch
        self.n_passes = math.ceil(self.sample_volume / self.samps_per_pass)

        if self.n_chkpts <= 0:
            raise ValueError(f"n_chkpts must be greater than 0, got {self.n_chkpts}")
        if self.chkpt_stop is not None and not 1 <= self.chkpt_stop <= self.n_chkpts:
            raise ValueError(f"chkpt_stop must be a checkpoint index in 1..n_chkpts ({self.n_chkpts}), got {self.chkpt_stop}")
        # sample interval between checkpoint/eval thresholds; sample_volume is data-derived so it need
        # not divide evenly -- the trainer skips the last mid-train threshold and the final eval covers
        # it at sample_volume
        self.chkpt_interval = self.sample_volume // self.n_chkpts
        if self.chkpt_interval == 0:
            raise ValueError(f"n_chkpts ({self.n_chkpts}) exceeds sample_volume ({self.sample_volume})")

        for key, val in (
            ("lr.init", self.lr["init"]),
            ("opt.wd", self.opt["wd"]),
            ("loss.logits.scalar_lr_factor", self.loss["logits"]["scalar_lr_factor"]),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"{key} must be numeric, got {val!r} -- note YAML parses scientific notation "
                    f"without a decimal point (e.g. 1e-6) as a string; write 1.0e-6"
                )

        lr_warmup = self.lr["warmup"]
        if not 0.0 <= lr_warmup < 1.0:
            raise ValueError(
                f"lr.warmup must be a fraction of sample_volume in [0.0, 1.0), got {lr_warmup}"
            )

        if self.reporting["plot_every"] not in ("trial", "chkpt"):
            raise ValueError(f"reporting.plot_every must be 'trial' or 'chkpt', got {self.reporting['plot_every']!r}")

        # native is always in play, so it carries no toggle; a typo here would otherwise drop a whole
        # eval group from the campaign's artifacts without a word
        if set(self.reporting["eval"]) != set(EVAL_GROUPS) - {"native"}:
            raise ValueError(
                f"reporting.eval must toggle exactly {sorted(set(EVAL_GROUPS) - {'native'})} "
                f"(native is always in play), got {sorted(self.reporting['eval'])}"
            )

        if self.del_base_eval_cache not in (None, "campaign", "trial"):
            raise ValueError(f"del_base_eval_cache must be null, 'campaign' or 'trial', got {self.del_base_eval_cache!r}")

        if self.kill_thresh is not None and not 0.0 < self.kill_thresh < 1.0:
            raise ValueError(f"kill_thresh must be null or a fraction in (0.0, 1.0), got {self.kill_thresh!r}")

        if self.arch["siglip"]["vis_proj_head"] is None and self.dropout["siglip"]["proj_head"] > 0.0:
            raise ValueError(
                "dropout.siglip.proj_head > 0 requires arch.siglip.vis_proj_head to be 'linear' or 'mlp' "
                "(projection-head dropout needs a projection head)"
            )

        if self.htarg["kernel"] not in ("bm", "laplace", "ou"):
            raise ValueError(f"Unknown htarg.kernel: '{self.htarg['kernel']}', must be one of {{bm, laplace, ou}}")

        htarg_beta = self.htarg["exp"]["beta"]
        if isinstance(htarg_beta, bool) or not isinstance(htarg_beta, (int, float)) or htarg_beta <= 0:
            raise ValueError(f"htarg.exp.beta must be a positive number, got {htarg_beta!r}")

        lambda_ = self.loss["blend"]["lambda"]
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"loss.blend.lambda out of bounds: {lambda_}, must be between 0.0 and 1.0")
        # the live target specs: loss1 carries weight 1 - lambda, loss2 weight lambda (utils.loss.targ_specs)
        live_targs = [cfg_targ["targ"] for w, cfg_targ in ((1.0 - lambda_, self.loss["loss1"]), (lambda_, self.loss["loss2"])) if w != 0.0]

        if self.htarg["shuffle"]:
            if "phylo" not in live_targs:
                raise ValueError(
                    "htarg.shuffle=True requires a live phylo target: "
                    "loss.loss1.targ 'phylo' under loss.blend.lambda != 1.0, or loss.loss2.targ 'phylo' under loss.blend.lambda != 0.0"
                )
            if self.seed is None:
                raise ValueError("htarg.shuffle=True requires a non-null seed (the shuffle permutation is derived from it and must match across DDP ranks)")

        if self.loss["crit"] not in ("infonce", "bce", "bif_bce"):
            raise ValueError(f"Unknown loss.crit: '{self.loss['crit']}', must be one of {{infonce, bce, bif_bce}}")

        if self.loss["sim"] not in ("cos", "geo1", "geo2"):
            raise ValueError(f"Unknown loss.sim: '{self.loss['sim']}', must be one of {{cos, geo1, geo2}}")

        for name, cfg_targ in (("loss.loss1", self.loss["loss1"]), ("loss.loss2", self.loss["loss2"])):
            if cfg_targ["targ"] not in ("sp", "mp", "tax", "phylo"):
                raise ValueError(f"Unknown {name}.targ: '{cfg_targ['targ']}', must be one of {{sp, mp, tax, phylo}}")

        # a blend of two identical target distributions is that distribution: lambda would do nothing. Under InfoNCE
        # the same targ type still blends two distributions when the specs' tsm differ
        if 0.0 < lambda_ < 1.0 and self.loss["loss1"]["targ"] == self.loss["loss2"]["targ"] and (
            self.loss["crit"] != "infonce" or self.loss["loss1"]["infonce"]["tsm"] == self.loss["loss2"]["infonce"]["tsm"]
        ):
            raise ValueError(
                f"loss.loss1 and loss.loss2 specify the same target distribution (targ '{self.loss['loss1']['targ']}') under loss.blend.lambda "
                f"{lambda_}: the blend is that target itself, so loss.blend.lambda is inert"
            )

        if self.loss["blend"]["type"] not in ("targ", "loss"):
            raise ValueError(f"Unknown loss.blend.type: '{self.loss['blend']['type']}', must be one of {{targ, loss}}")
        if not isinstance(self.loss["unitless"], bool):
            raise ValueError(f"loss.unitless must be a bool, got {self.loss['unitless']!r}")
        if not isinstance(self.loss["logits"]["shared"], bool):
            raise ValueError(f"loss.logits.shared must be a bool, got {self.loss['logits']['shared']!r}")

        if self.loss["logits"]["bce"]["center"] not in (None, "sim", "grad_proj", "grad_proj2"):
            raise ValueError(f"Unknown loss.logits.bce.center: '{self.loss['logits']['bce']['center']}', must be one of {{null, sim, grad_proj, grad_proj2}}")

        bias_init = self.loss["logits"]["bce"]["bias"]["init"]
        if bias_init == "pos_prevalence":
            if self.loss["crit"] not in ("bce", "bif_bce"):
                raise ValueError(f"loss.logits.bce.bias.init: pos_prevalence requires a BCE-family crit (bce, bif_bce), got '{self.loss['crit']}'")
        elif bias_init is not None and (isinstance(bias_init, bool) or not isinstance(bias_init, (int, float))):
            raise ValueError(f"Unknown loss.logits.bce.bias.init: {bias_init!r}, must be one of {{null, pos_prevalence, [float]}}")

        block_resid = self.loss["infonce"]["block_residuals"]
        if not isinstance(block_resid, bool):
            raise ValueError(f"loss.infonce.block_residuals must be a bool, got {block_resid!r}")
        if block_resid:
            # utils.loss.infonce_block_resid removes the closed-form residual of a hard binary target from
            # the plain cross-entropy's scale gradient: every loss term must train against such a target,
            # and carry no weight the residual's derivation doesn't account for
            if self.loss["crit"] != "infonce":
                raise ValueError(f"loss.infonce.block_residuals requires loss.crit: infonce, got '{self.loss['crit']}'")
            live_specs = [(name, cfg_targ) for w, name, cfg_targ in
                          ((1.0 - lambda_, "loss.loss1", self.loss["loss1"]), (lambda_, "loss.loss2", self.loss["loss2"])) if w != 0.0]
            for name, cfg_targ in live_specs:
                if cfg_targ["targ"] not in ("sp", "mp") or cfg_targ["infonce"]["tsm"]["type"] != "linear":
                    raise ValueError(
                        f"loss.infonce.block_residuals applies to hard binary targets only: {name}.targ must be one of "
                        f"{{sp, mp}} under {name}.infonce.tsm.type: linear, got targ '{cfg_targ['targ']}' under tsm.type "
                        f"'{cfg_targ['infonce']['tsm']['type']}'"
                    )
            if len(live_specs) > 1 and self.loss["blend"]["type"] == "targ":
                raise ValueError(
                    "loss.infonce.block_residuals under two live targets requires loss.blend.type: loss -- a target "
                    "blend's (1 - lambda) * Y1 + lambda * Y2 is not a hard binary distribution"
                )
            if self.loss["wting"]["cls_imb"]["type"] is not None or self.loss["wting"]["focal"]["gamma"] != 0.0:
                raise ValueError(
                    "loss.infonce.block_residuals is not implemented alongside class-imbalance weighting "
                    f"(loss.wting.cls_imb.type: {self.loss['wting']['cls_imb']['type']!r}) or focal loss "
                    f"(loss.wting.focal.gamma: {self.loss['wting']['focal']['gamma']}): the residual it removes is the "
                    "unweighted cross-entropy's"
                )

        # the augmentation config the `aug` switch selects: OpenCLIP's defaults, or config/trial/train/augmentation.yaml
        # (campaign trials inject the frozen snapshot; otherwise it is read live, and only when custom asks for it)
        if self.aug == "openclip":
            self.aug_cfg = _aug_cfg_openclip()
        elif self.aug == "custom":
            self.aug_cfg = deepcopy(self.augmentation) if self.augmentation is not None else load_augmentation_config_dict()
        else:
            raise ValueError(f"Unknown aug: '{self.aug}', must be one of {{openclip, custom}}")

        # an aug block whose prob is 0.0 is inert -> dropped from the working config; downstream keys off presence
        if self.aug_cfg["cjit"]["prob"] == 0.0:
            del self.aug_cfg["cjit"]
        if self.aug_cfg["sharpness"]["prob"] == 0.0:
            del self.aug_cfg["sharpness"]
        if self.aug_cfg["gblur"]["prob"] == 0.0:
            del self.aug_cfg["gblur"]

        # focal toggle: gamma 0.0 disables -> block dropped from the working config; downstream keys off presence
        if self.loss["wting"]["focal"]["gamma"] == 0.0:
            del self.loss["wting"]["focal"]

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
            if not chunking_supported(self.loss):  # tiled loss supports the full BCE-family config (bce/bif_bce); inert with infonce
                self.hw.loss_chunk_size = None
            else:
                # center: sim needs the full-batch sim mean IN-GRAPH per tile; the tiled path recovers it
                # exactly only through the cos-sim mean factorization mean(sim) = mean(img) . mean(txt)
                # (see utils/loss.py) -- geo sims have no such closed form
                if self.loss["logits"]["bce"]["center"] == "sim" and self.loss["sim"] != "cos":
                    raise ValueError(
                        f"loss.logits.bce.center: sim requires loss.sim: cos under hardware.loss_chunk_size "
                        f"(got loss.sim: {self.loss['sim']}): the tiled loss reproduces full-batch sim-centering "
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


def targ_dependent_loss(loss: dict) -> bool:
    """
    Whether anything sets a loss blend apart from the target blend (utils.loss.Criterion), given a `loss`
    config block: a loss factor that reads the term's own target -- without one the loss is affine in the
    target, so the two blends coincide in value and gradients (focal gamma 0.0 drops its block from the
    working config) -- or separate logit scalars, each term then scored on its own logits. So: whether
    loss.blend.type is read at all, under two live targets.

    Shared by inert_params and the config.json metadata cleaning (utils.train.ArtifactManager
    .save_metadata_coord), which have to agree: the first decides whether a campaign may override
    loss.blend.type, the second whether the trial's recorded config keeps it. Each factor is gated on the
    criterion that reads it -- DSMR is BCE-only, targ_mass_neut bifurcated-only, block_residuals
    InfoNCE-only -- since a criterion's own toggle sets nothing apart where it is never read.
    """
    crit = loss["crit"]
    return (
        loss["unitless"] or "focal" in loss["wting"] or not loss["logits"]["shared"]
        or (crit != "infonce" and loss["wting"]["bce"]["dsmr"])
        or (crit == "bif_bce" and loss["bce"]["targ_mass_neut"])
        # the residual blocked per term is its own target's, in closed form off that target's memberships
        # (utils.loss.infonce_block_resid) -- a factor no target blend has
        or (crit == "infonce" and loss["infonce"]["block_residuals"])
    )

def inert_params(cfg: TrainConfig) -> dict[str, str]:
    """The params the effective config never reads -- config/trial/train/train.yaml's 'Inert iff' annotations -- as
    {dot-path prefix: the setting that makes it so}; a prefix covers its whole subtree. A prefix two rules
    render inert keeps the first-listed reason."""
    is_siglip = "siglip" in cfg.arch["model_type"].lower()
    lambda_ = cfg.loss["blend"]["lambda"]
    crit = cfg.loss["crit"]
    cls_imb_type = cfg.loss["wting"]["cls_imb"]["type"]
    # the live target specs: loss1 carries weight 1 - lambda, loss2 weight lambda (utils.loss.targ_specs)
    live_targs = [cfg_targ["targ"] for w, cfg_targ in ((1.0 - lambda_, cfg.loss["loss1"]), (lambda_, cfg.loss["loss2"])) if w != 0.0]
    targ_dep = targ_dependent_loss(cfg.loss)
    rules = [
        ("arch.clip", is_siglip, "arch.model_type is a SigLIP model"),
        ("arch.siglip", not is_siglip, "arch.model_type is a CLIP model"),
        ("dropout.siglip", not is_siglip, "arch.model_type is a CLIP model"),
        ("dropout.siglip.proj_head", cfg.arch["siglip"]["vis_proj_head"] is None, "arch.siglip.vis_proj_head is null"),
        ("htarg", "phylo" not in live_targs, "no live target is phylo"),
        ("htarg.exp", cfg.htarg["kernel"] == "bm", "htarg.kernel is bm"),
        ("loss.loss1", lambda_ == 1.0, "loss.blend.lambda is 1.0"),
        ("loss.loss2", lambda_ == 0.0, "loss.blend.lambda is 0.0"),
        ("loss.blend.type", lambda_ in (0.0, 1.0), f"loss.blend.lambda is {lambda_} (a lone target)"),
        ("loss.blend.type", not targ_dep,
         "no target-dependent loss factor is live (loss.unitless, focal, DSMR, targ_mass_neut, block_residuals) on "
         "shared logit scalars: the blend types coincide"),
        ("loss.logits.shared", lambda_ in (0.0, 1.0), f"loss.blend.lambda is {lambda_} (a lone target)"),
        ("loss.logits.shared", cfg.loss["blend"]["type"] == "targ", "loss.blend.type is targ (one loss on one set of logits)"),
        ("loss.infonce", crit != "infonce", f"loss.crit is {crit}"),
        ("loss.bce", crit != "bif_bce", f"loss.crit is {crit}"),
        ("loss.bce", all(targ == "sp" for targ in live_targs), "every live target is sp (row mass already 1)"),
        ("loss.wting.bce", crit == "infonce", "loss.crit is infonce"),
        ("loss.wting.cls_imb.inv_freq", cls_imb_type != "inv_freq", f"loss.wting.cls_imb.type is {cls_imb_type}"),
        ("loss.wting.cls_imb.class_bal", cls_imb_type != "class_bal", f"loss.wting.cls_imb.type is {cls_imb_type}"),
        ("loss.wting.cls_imb.norm", cls_imb_type is None, f"loss.wting.cls_imb.type is {cls_imb_type}"),
        ("loss.wting.cls_imb.norm", cfg.loss["unitless"], "loss.unitless is true (its rescale cancels the per-batch normalizer)"),
        ("loss.logits.bce", crit == "infonce", "loss.crit is infonce (sigmoid/BCE-path logit params)"),
        # CLIP + bias.init null: logit_bias is a fixed 0.0 buffer (models.py), never learnable
        ("loss.logits.bce.bias.freeze", not is_siglip and cfg.loss["logits"]["bce"]["bias"]["init"] is None,
         "the CLIP logit bias is a fixed 0.0 buffer under loss.logits.bce.bias.init: null"),
    ]
    for key in ("loss1", "loss2"):
        cfg_targ = cfg.loss[key]
        rules += [
            (f"loss.{key}.infonce", crit != "infonce", f"loss.crit is {crit}"),
            (f"loss.{key}.infonce.tsm.sm_scale", cfg_targ["infonce"]["tsm"]["type"] == "linear",
             f"loss.{key}.infonce.tsm.type is linear"),
        ]

    inert = {}
    for prefix, cond, reason in rules:
        if cond:
            inert.setdefault(prefix, reason)
    return inert

def _check_overrides_live(cfg: TrainConfig, overrides: dict) -> None:
    """Refuse campaign overrides (dot-path keys) of params the effective config renders inert: the arm / coord
    would advertise a setting the trial never reads, whatever its value -- one equal to the baseline's included.
    Each hit is reported with its outermost cause (the shortest covering inert_params prefix)."""
    inert = inert_params(cfg)
    hits = {}
    for key in overrides:
        covering = [prefix for prefix in inert if key == prefix or key.startswith(prefix + ".")]
        if covering:
            hits[key] = inert[min(covering, key=len)]
    if hits:
        raise ValueError(
            "inert override(s), never read under this config: "
            + "; ".join(f"{key} ({reason})" for key, reason in hits.items())
        )

def apply_dev_overrides(cfg_dict: dict, dev_config: dict | None = None) -> dict:
    """With train.yaml's `dev` on, lays config/trial/train/dev.yaml over the config -- its keys are dot-paths into
    this config (e.g. lr.warmup) -- and forces the dev split. Inert with `dev` off."""
    if not cfg_dict["dev"]:
        return dict(cfg_dict)
    if dev_config is None:  # load live when no snapshot supplied; campaign trials pass the frozen snapshot
        dev_config = load_dev_config_dict()
    cfg_dict = apply_overrides(cfg_dict, dev_config)
    cfg_dict["split"] = "dev"  # forced after the overrides: a dev run always uses the dev split
    return cfg_dict

def _set_by_dot_path(cfg_dict: dict, key_path: str, value) -> None:
    """Overwrite an existing config field addressed by dot-path.

    Every segment must already be declared: overrides replace declared fields, never create them.
    A typo'd or stale key (a renamed param still swept in a campaign's `ablation_arms` / `hpo_coords`)
    would otherwise land silently in a field nothing reads, and the campaign runs the baseline value
    under an arm / coord name advertising the override.
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
    """The base training config, assembled from config/trial/ plus config/trial/operational.yaml. Its shape is
    what every dot-path override addresses (opt.wd, loss.blend.lambda, loss.loss1.targ, ...), so the pieces
    merge into the one flat dict the rest of the pipeline has always seen: optimizer.yaml and
    lr_schedule.yaml land whole under `opt` and `lr`, and loss.yaml whole under `loss` -- its per-target
    `loss1` / `loss2` blocks included, so they are addressed as loss.loss1.* / loss.loss2.*."""
    def _load(fpath):
        with open(fpath) as f:
            return yaml.safe_load(f)

    dpath_trial = paths["config"] / "trial"
    dpath = dpath_trial / "train"
    cfg = _load(dpath / "train.yaml")
    cfg.update(_load(dpath_trial / "split.yaml"))
    cfg.update(_load(dpath / "model.yaml"))
    cfg.update(_load(dpath_trial / "operational.yaml"))
    cfg["opt"] = _load(dpath / "optimizer.yaml")
    cfg["lr"] = _load(dpath / "lr_schedule.yaml")
    cfg["loss"] = _load(dpath / "loss.yaml")
    return cfg

def load_augmentation_config_dict() -> dict:
    with open(paths["config"] / "trial" / "train" / "augmentation.yaml") as f:
        return yaml.safe_load(f)

def load_reporting_config_dict() -> dict:
    with open(paths["config"] / "trial" / "reporting.yaml") as f:
        return yaml.safe_load(f)

def load_dev_config_dict() -> dict:
    with open(paths["config"] / "trial" / "train" / "dev.yaml") as f:
        return yaml.safe_load(f)

def load_htargs_config_dict() -> dict:
    with open(paths["config"] / "trial" / "train" / "htargs.yaml") as f:
        return yaml.safe_load(f)

def load_model_specific_config_dict() -> dict:
    with open(paths["config"] / "trial" / "train" / "model_specific.yaml") as f:
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

def _get_by_dot_path(cfg_dict: dict, key_path: str):
    """Read an existing config field addressed by dot-path. Every segment must already be declared --
    the same contract as _set_by_dot_path, so a stale key in a defaults file fails loudly."""
    cursor = cfg_dict
    for depth, key in enumerate(key_path.split(".")):
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError(
                f"Unknown config key '{key_path}': '{'.'.join(key_path.split('.')[:depth + 1])}' is not a config field."
            )
        cursor = cursor[key]
    return cursor

def apply_model_specific_defaults(cfg_dict: dict, model_specific_config: dict | None = None) -> dict:
    """Fills the params config/trial/train/model_specific.yaml declares for the trial's model family
    (clip / siglip), each ONLY if null. Its keys are dot-paths into this config (opt.wd, loss.crit, ...),
    so a value set in the config proper wins and a family default only ever fills a hole."""
    cfg_out = deepcopy(cfg_dict)
    arch = cfg_out.get("arch", {})
    if not isinstance(arch, dict) or "model_type" not in arch:
        raise ValueError("Config field 'arch/model_type' is required to resolve model-specific defaults.")

    family = _resolve_model_family(arch["model_type"])
    if model_specific_config is None:  # load live when no snapshot supplied; campaign trials pass the frozen snapshot
        model_specific_config = load_model_specific_config_dict()
    family_defaults = model_specific_config.get(family)
    if not isinstance(family_defaults, dict):
        raise ValueError(f"Missing model hyperparameter defaults for family '{family}'.")

    for key_path, value in family_defaults.items():
        if _get_by_dot_path(cfg_out, key_path) is None:
            _set_by_dot_path(cfg_out, key_path, value)
    return cfg_out

def load_dataset_specific_config_dict() -> dict:
    with open(paths["config"] / "trial" / "train" / "dataset_specific.yaml") as f:
        return yaml.safe_load(f)

def apply_dataset_specific_defaults(cfg_dict: dict, dataset_specific_config: dict | None = None) -> dict:
    """Fills n_epochs / n_chkpts, each only if null, from the trial's dataset entry in
    config/trial/train/dataset_specific.yaml."""
    cfg_out = deepcopy(cfg_dict)
    keys = [key for key in ("n_epochs", "n_chkpts") if cfg_out[key] is None]
    if not keys:
        return cfg_out
    if dataset_specific_config is None:  # load live when no snapshot supplied; campaign trials pass the frozen snapshot
        dataset_specific_config = load_dataset_specific_config_dict()
    for key in keys:
        cfg_out[key] = dataset_specific_config[cfg_out["dataset"]][key]
    return cfg_out

def get_config_train(cfg_dict: dict) -> TrainConfig:
    overrides = cfg_dict.pop("_overrides", None)  # campaign trials: the merged arm + coord overrides
    model_specific = cfg_dict.pop("model_specific", None)  # campaign trials inject the frozen snapshot; otherwise read live
    dataset_specific = cfg_dict.pop("dataset_specific", None)  # ditto
    dev_overrides = cfg_dict.pop("dev_overrides", None)  # ditto (config/trial/train/dev.yaml, read only when `dev` is on)
    cfg_dict = apply_dev_overrides(cfg_dict, dev_overrides)
    cfg_dict = apply_model_specific_defaults(cfg_dict, model_specific)
    cfg_dict = apply_dataset_specific_defaults(cfg_dict, dataset_specific)
    # htarg is a swept dimension (htarg.* dot-paths in ablation_arms / hpo_coords), so it has to be in place
    # BEFORE the overrides land on it; campaign trials inject the frozen snapshot, otherwise load live
    cfg_dict.setdefault("htarg", load_htargs_config_dict())
    if overrides is not None:
        cfg_dict = apply_overrides(cfg_dict, overrides)
    # campaign trials freeze these into the baseline and inject them per trial; otherwise load live
    cfg_dict.setdefault("hw", load_hardware_config_dict())
    cfg_dict.setdefault("reporting", load_reporting_config_dict())
    cfg = TrainConfig(**cfg_dict)
    if overrides is not None:
        _check_overrides_live(cfg, overrides)
    # campaign trials inject the frozen snapshot; otherwise load live. Either way it goes through
    # ManifoldVizConfig so the injected dict is validated too (not just the live yaml).
    cfg_manifold_viz = cfg.manifold_viz if cfg.manifold_viz is not None else load_manifold_viz_trial_config_dict()
    cfg.manifold_viz = asdict(ManifoldVizConfig(**{**cfg_manifold_viz, **load_manifold_viz_render_config_dict()}))
    return cfg


@dataclass
class HardwareConfig:

    mixed_prec: bool  # bf16 autocast mixed precision for training and validation (bf16 needs no GradScaler)
    act_chkpt: bool
    loss_chunk_size: int | None  # row-block height for the global-batch (BxB) loss, row-band-sharded across ranks; None -> full BxB (no tiling/sharding). See config/trial/hardware.yaml.
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
    with open(paths["config"] / "trial" / "hardware.yaml") as f:
        return yaml.safe_load(f)

def get_config_hardware():
    return HardwareConfig(**load_hardware_config_dict())


# per-dataset manifold-viz scatter marker size (nymph/lepid have many points -> smaller markers)
DATASET2MARKER_SIZE = {
    "bryo": 7,
    "cub": 7,
    "lepid": 1,
    "nymph": 2,
}


@dataclass
class ManifoldVizConfig:
    """manifold_viz.yaml contents -- the whole manifold-viz subsystem: which trials run it (the
    n_seeds/n_seeds_offset seed window), which panel groups are emitted, the pooled shared-frame fit, the
    per-method params, and colors."""

    n_seeds: int
    n_seeds_offset: int
    eval_duration: int
    bg_color: str | None
    plot_2panel: bool
    plot_7panel: bool
    plot_8panel: bool
    pooled: dict
    
    tsne: dict = field(default_factory=dict)
    umap: dict = field(default_factory=dict)
    orient: dict = field(default_factory=dict)
    color: dict = field(default_factory=dict)

    def __post_init__(self):
        # window of seeds (per arm/coord/dataset group) that get manifold viz: idx_seed in
        # [n_seeds_offset, n_seeds_offset + n_seeds). n_seeds 0 disables the subsystem; a window past the
        # end of the campaign's seed sweep is legal and simply selects nothing (see train.py's gate).
        if self.n_seeds < 0:
            raise ValueError(f"n_seeds must be >= 0, got {self.n_seeds}")

        if self.n_seeds_offset < 0:
            raise ValueError(f"n_seeds_offset must be >= 0, got {self.n_seeds_offset}")

        if self.pooled["budget"] <= 0:
            raise ValueError(f"pooled.budget must be > 0, got {self.pooled['budget']}")

        if self.pooled["pca_bounds"] not in (None, "final"):
            raise ValueError(f"pooled.pca_bounds must be null or 'final', got {self.pooled['pca_bounds']!r}")

        if self.umap["n_neighbors"] < 2:
            raise ValueError(f"umap.n_neighbors must be >= 2, got {self.umap['n_neighbors']}")

        if not 0.0 <= self.umap["min_dist"] < 1.0:
            raise ValueError(f"umap.min_dist must be in [0.0, 1.0), got {self.umap['min_dist']}")

        if not 0.0 < self.orient["ema_tau"] <= 1.0:
            raise ValueError(f"orient.ema_tau must be in (0.0, 1.0], got {self.orient['ema_tau']}")


def load_manifold_viz_trial_config_dict() -> dict:
    """The half that decides what is computed and cached (projections.npz): frozen with the campaign."""
    with open(paths["config"] / "trial" / "manifold_viz.yaml") as f:
        return yaml.safe_load(f)

def load_manifold_viz_render_config_dict() -> dict:
    """The half that only decides how the cached projections are drawn: read live at every render."""
    with open(paths["config"] / "render" / "manifold_viz.yaml") as f:
        return yaml.safe_load(f)

def load_manifold_viz_config_dict() -> dict:
    """Both halves merged -- the whole ManifoldVizConfig, as the compute and render paths expect it. The
    render half is always the live file, so a campaign's frozen trial half never pins its styling."""
    return {**load_manifold_viz_trial_config_dict(), **load_manifold_viz_render_config_dict()}

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
    overrides: bool  # (True) append the "Arm Overrides" / "Coord Overrides" config tables to each xlsx sheet

    def __post_init__(self):

        if self.spread_type not in ("std", "ste"):
            raise ValueError(f"Unknown stats spread_type: '{self.spread_type}', must be one of {{std, ste}}")

        if set(self.supp_scores) != {"primitive", "n_shot"}:
            raise ValueError(f"stats supp_scores must have exactly the keys {{primitive, n_shot}}, got {sorted(self.supp_scores)}")


def load_stats_config_dict() -> dict:
    with open(paths["config"] / "render" / "stats.yaml") as f:
        return yaml.safe_load(f)

def get_config_stats():
    return StatsConfig(**load_stats_config_dict())


def load_zip_config_dict() -> dict:
    with open(paths["config"] / "render" / "zip.yaml") as f:
        return yaml.safe_load(f)


@dataclass
class CampaignConfig:
    """config/campaigns/<name>.yaml contents -- one campaign's trial matrix (see campaign_runner): its arms
    (ablation_arms) x coords (hpo_coords) x datasets, run for n_trials_screen seeds each in the screening phase,
    then each arm's best coord per dataset topped up to n_trials_qual seeds in the qual phase (null: no qual
    phase), then -- with trainval -- each of those picks retrained on the trainval partition up to its
    qual-selected checkpoint, one run per qual seed (the trainval phase). baseline_overrides is a flat set of
    overrides laid on every trial of the campaign, nameless -- arm / coord names read as if it were empty.
    suffix is appended to the campaign name (null: none)."""

    n_trials_screen: int
    n_trials_qual: int | None
    trainval: bool
    datasets: list
    baseline_overrides: dict
    ablation_arms: list
    hpo_coords: list
    suffix: str | None

    def __post_init__(self):

        # baseline_overrides is not a matrix dimension: it is one flat set of overrides laid on every trial
        # of the campaign, so it takes scalars only -- a dict would be a combo group and a list a combo list
        if not isinstance(self.baseline_overrides, dict):
            raise ValueError(f"baseline_overrides must be a mapping of override key -> value, got {type(self.baseline_overrides).__name__}")
        nested = {k: v for k, v in self.baseline_overrides.items() if isinstance(v, (dict, list))}
        if nested:
            raise ValueError(
                f"baseline_overrides takes scalar values -- no combo groups or combo lists: {sorted(nested)}. "
                f"It applies to every trial, so it has nothing to vary over; sweep in ablation_arms / hpo_coords instead."
            )

        if self.n_trials_qual is not None and self.n_trials_screen > self.n_trials_qual:
            raise ValueError(
                f"n_trials_screen ({self.n_trials_screen}) exceeds n_trials_qual ({self.n_trials_qual}): the qual phase "
                f"tops each pick up FROM its n_trials_screen screening trials TO n_trials_qual, so n_trials_qual must be "
                f">= n_trials_screen (or null to skip the qual phase)"
            )

        if self.trainval and self.n_trials_qual is None:
            raise ValueError(
                "trainval: true requires a qual phase (n_trials_qual set): the trainval phase trains each qual pick "
                "up to its qual-selected checkpoint"
            )


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
    with open(paths["config"] / "preprocessing" / "split_gen.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict = apply_splits_debug_overrides(cfg_dict)
    cfg = GenSplitConfig(**cfg_dict)
    return cfg
