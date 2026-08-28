from dataclasses import asdict
from datetime import datetime, timezone
import math
import os
import time
import random
import numpy as np
import torch
import shutil

from utils.utils import (
    paths,
    save_pickle,
    load_pickle,
    save_json,
    load_json,
    TimeTracker,
    Timer,
)
from utils.ddp import rank0

import pdb


# checkpoint-selection criteria: criterion (also the evals/_best/ + stats subdir name) ->
# the scores.comp.<key>.<metric> it maximizes
BEST_CRITERIA = {"map": ("map", "all"), "acc": ("acc", "i2t")}


def format_scores(scores):
    if scores is None:
        return None
    if isinstance(scores, dict):
        return {k: format_scores(v) for k, v in scores.items()}
    return f"{float(scores):.4f}"

def parse_scores(scores):
    if scores is None:
        return None
    if isinstance(scores, dict):
        return {k: parse_scores(v) for k, v in scores.items()}
    return float(scores)

def format_mem(mem):
    # {"ram": (used_bytes, total_bytes), "vram": ...} -> {"ram": "4.2/128.0 GB", "vram": ...}
    return {k: f"{used / 2**30:.1f}/{total / 2**30:.1f} GB" for k, (used, total) in mem.items()}

def merge_mem(mem_prev, mem_new):
    """Per key keep whichever formatted 'used/total GB' reading has the higher used -- a running max
    across snapshots (kept as a pair so used/total stay from the same reading if totals ever differ
    across relaunches). None (no reading yet) is always superseded."""
    def used(s):
        return float(s.split("/")[0])
    return {
        k: mem_new[k] if mem_prev[k] is None or used(mem_new[k]) > used(mem_prev[k]) else mem_prev[k]
        for k in mem_new
    }


class TrialData:

    def __init__(self, dpath_trial):

        self.fpath_data = dpath_trial / "data_trial.pkl"

        self.data_epoch = {
            "n_samps_seen": [],
            "lr": [],
            "loss_train": [],
            "loss_raw_train": [],
            "grad_norm_model": [],
            "delta_norm_model": [],  # ||delta theta||: the L2 norm of each step's parameter update
            "grad_sum_sim1": [],
            "grad_sum_sim2": [],
            # learnable logit scalars, temp as tau = exp(-logit_scale); a series stays empty when its
            # scalar is untracked (TrainPipeline._tracked_logit_scalars) and then gets no curve panel
            "temp1": [],
            "bias1": [],
            "temp2": [],
            "bias2": [],
            "sim1_min": [],
            "sim1_max": [],
            "sim1_median": [],
            "sim1_mean": [],
            "sim2_min": [],
            "sim2_max": [],
            "sim2_median": [],
            "sim2_mean": [],
            # targ*_hist / p*_hist: per batch, the targets and predicted pair probabilities binned
            # over [0, 1] (a list of bin fractions, not a scalar) -- the curve strips render them as
            # heatmap columns. targ* is recorded only for branches with graded targets
            # (TrainPipeline._tracked_targ_stats) and p* only for BCE-family ones
            # (sim_targ_batch_stats), so an excluded branch's series stays empty.
            "targ1_hist": [],
            "targ2_hist": [],
            "p1_hist": [],
            "p2_hist": [],
        }
        self.data_eval = {
            "n_samps_seen": [],
        }
        self.data = {
            "epoch": self.data_epoch,
            "eval": self.data_eval,
        }

        self.n_evals = 0

        self.eval_metrics = None  # most recent eval metrics
        self.time_eval = None  # most recent eval time

        self.timer_trial = Timer()
        self.timer_trial.start()

    def update_train_batch(self, n_samps_seen, lr=None, loss_train=None, loss_raw_train=None, grad_norm_model=None,
                           delta_norm_model=None, batch_stats=None,
                           grad_sum_sim1=None, grad_sum_sim2=None, logit_scalars=None):

        self.data_epoch["n_samps_seen"].append(n_samps_seen)

        if lr is not None:
            self.data_epoch["lr"].append(lr)
        if loss_train is not None:
            self.data_epoch["loss_train"].append(loss_train)
        if loss_raw_train is not None:
            self.data_epoch["loss_raw_train"].append(loss_raw_train)
        if grad_norm_model is not None:
            self.data_epoch["grad_norm_model"].append(grad_norm_model)
        if delta_norm_model is not None:
            self.data_epoch["delta_norm_model"].append(delta_norm_model)
        if grad_sum_sim1 is not None:
            self.data_epoch["grad_sum_sim1"].append(grad_sum_sim1)
        if grad_sum_sim2 is not None:
            self.data_epoch["grad_sum_sim2"].append(grad_sum_sim2)
        if batch_stats is not None:
            for stat_key, stat_value in batch_stats.items():
                self.data_epoch[stat_key].append(stat_value)
        if logit_scalars is not None:
            for key, value in logit_scalars.items():
                self.data_epoch[key].append(value)

    def update_eval(self, n_samps_seen):

        def append_nested(dst, src):
            for score_name, score_value in src.items():
                if isinstance(score_value, dict):
                    if score_name not in dst:
                        dst[score_name] = {}
                    append_nested(dst[score_name], score_value)
                else:
                    if score_name not in dst:
                        dst[score_name] = []
                    dst[score_name].append(score_value)

        self.n_evals += 1

        self.data_eval["n_samps_seen"].append(n_samps_seen)

        for k, v in self.eval_metrics.items():
            if k not in self.data_eval:
                self.data_eval[k] = {}
            append_nested(self.data_eval[k], v)

    @classmethod
    def resume(cls, dpath_trial, trial_state):
        obj = cls(dpath_trial)
        obj.data = load_pickle(obj.fpath_data)
        obj.data_epoch = obj.data["epoch"]
        obj.data_eval = obj.data["eval"]
        obj.n_evals = trial_state["n_evals"]
        obj.timer_trial = Timer()
        obj.timer_trial.set_elapsed_time(trial_state["timer_trial_elapsed"])
        obj.timer_trial.start()
        return obj

    @rank0
    def save(self):
        save_pickle(self.data, self.fpath_data)


class ArtifactManager:

    dpath_campaign = None
    dpath_coord = None
    dpath_trial = None
    fpath_metadata_trial = None
    dpath_eval_final = None
    dpath_model_checkpoint = None
    resuming = False
    dataset = None
    split = None

    @staticmethod
    def set_paths(cfg_train):

        ArtifactManager.dpath_campaign = paths["artifacts"] / cfg_train.campaign
        ArtifactManager.dpath_coord = (ArtifactManager.dpath_campaign / "datasets" / cfg_train.dataset / "arms" / cfg_train.arm
                                       / "coords" / cfg_train.coord)
        ArtifactManager.dataset = cfg_train.dataset
        ArtifactManager.split = cfg_train.split

        trial_name = cfg_train.seed
        ArtifactManager.dpath_trial = ArtifactManager.dpath_coord / str(trial_name)
        ArtifactManager.fpath_metadata_trial = ArtifactManager.dpath_trial / "trial_metadata.json"

        ArtifactManager.dpath_eval_final = ArtifactManager.dpath_trial / "evals" / f"eval{cfg_train.n_chkpts}"
        ArtifactManager.dpath_model_checkpoint = ArtifactManager.dpath_trial / "chkpts/in_progress"

        if ArtifactManager.dpath_trial.exists():
            if (ArtifactManager.dpath_model_checkpoint / "train_state.pt").exists():
                ArtifactManager.resuming = True
            else:
                shutil.rmtree(ArtifactManager.dpath_trial, ignore_errors=True)
                ArtifactManager.resuming = False
        else:
            ArtifactManager.resuming = False

    @staticmethod
    @rank0
    def create_trial_dirs():
        if ArtifactManager.resuming:
            return
        for subdir in ("logs", "chkpts/in_progress", "learning_curves"):
            (ArtifactManager.dpath_trial / subdir).mkdir(parents=True)

    @staticmethod
    @rank0
    def update_campaign_time():

        def format_duration(seconds: float) -> str:
            seconds = int(seconds)
            days, seconds = divmod(seconds, 86400)
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
        
        fpath_pkl = ArtifactManager.dpath_campaign / "time.pkl"
        fpath_json = ArtifactManager.dpath_campaign / "campaign_metadata.json"

        time_data = load_pickle(fpath_pkl)

        time_last_updated = time_data["last_updated"]
        time_curr = time.time()

        time_elapsed = time_data["elapsed"]
        time_elapsed = time_elapsed + (time_curr - time_last_updated)

        time_data["last_updated"] = time_curr
        time_data["elapsed"] = time_elapsed
        save_pickle(time_data, fpath_pkl)
        
        metadata_camp = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")
        metadata_camp["duration"] = format_duration(time_elapsed)
        save_json(metadata_camp, fpath_json)

    @staticmethod
    @rank0
    def update_campaign_memory(mem):
        # campaign-level memory = running max across every trial's snapshots (== max across
        # trial-level values, maintained incrementally so it survives an OOM-killed trial)
        fpath_meta = ArtifactManager.dpath_campaign / "campaign_metadata.json"
        metadata_camp = load_json(fpath_meta)
        metadata_camp["memory"] = merge_mem(metadata_camp["memory"], format_mem(mem))
        save_json(metadata_camp, fpath_meta)

    @staticmethod
    @rank0
    def save_metadata_coord(cfg_train):

        def clean_metadata(metadata):
            """Params rendered inert by other config params are pruned, so absence in config.json
            is the inert signal (the stats overrides tables render absent params as '-')."""

            del metadata["campaign"]
            del metadata["arm"]
            del metadata["coord"]
            del metadata["seed"]
            del metadata["idx_seed"]
            del metadata["idx_trial"]
            del metadata["n_trials_total"]
            del metadata["dataset"]
            del metadata["split"]
            # dataset-resolved duration and checkpoint count (config/dataset_specific.yaml fills a null
            # n_epochs / n_chkpts per dataset), not arm/coord params; the duration is recorded in
            # coord_metadata.json's horizon (and each trial's trial_metadata.json progress), the
            # checkpoint count in every eval file's chkpt field
            del metadata["n_epochs"]
            del metadata["n_chkpts"]

            del metadata["dev"]

            # family-specific sections: models.py reads arch.siglip / dropout.siglip only when
            # is_siglip, and arch.clip.non_causal drives disable_causal_mask_text (CLIPWrapper-only)
            is_siglip = "siglip" in metadata["arch"]["model_type"].lower()
            if is_siglip:
                del metadata["arch"]["clip"]
                if metadata["arch"]["siglip"]["vis_proj_head"] is None:
                    del metadata["dropout"]["siglip"]["proj_head"]  # head dropout needs a projection head
            else:
                del metadata["arch"]["siglip"]
                del metadata["dropout"]["siglip"]

            if metadata["loss2"]["mix"] == 0.0:
                del metadata["loss2"]

            # CLIP + bias.init null: logit_bias becomes a fixed 0.0 buffer (models.py), so the
            # whole bias block is a no-op (logits = sim * scale.exp() + 0). loss2's logit params
            # are always fresh learnable Parameters, so loss2.logits is never pruned.
            if not is_siglip and metadata["loss"]["logits"]["bce"]["bias"]["init"] is None:
                del metadata["loss"]["logits"]["bce"]["bias"]

            # per-loss weighting: drop params the loss type never reads (the bce sub-block is
            # BCE-only -- absent from InfoNCE's 1D weighting),
            # params their own toggle disables (cls_imb.type null; focal.gamma 0.0 is already
            # pruned from the working config at load), and the scalar
            # cancellation noted in train.yaml: the unit-scale blend (loss / loss.detach()) cancels
            # any per-batch scalar factor on a loss, making cls_imb.norm's rescale inert under
            # unit-scaling. loss2 is already gone when mix = 0.0, under which mix_unit_scale never
            # applies.
            unit_scaled = "loss2" in metadata and metadata["loss2"]["mix_unit_scale"]
            for key in ("loss", "loss2"):
                if key not in metadata:
                    continue
                wting = metadata[key]["wting"]
                crit = metadata[key]["crit"]
                is_bce_family = crit in ("bce", "bif_bce")  # sigmoid-BCE losses; wting.bce applies

                # infonce sub-block: the BCE losses never read it, and under sp the linear tsm
                # mapping is an identical no-op (row sums already 1)
                if is_bce_family or metadata[key]["targ"] == "sp":
                    del metadata[key]["infonce"]
                # bce sub-block (targ_mass_neut): read only by the bifurcated variant
                if crit != "bif_bce":
                    del metadata[key]["bce"]

                cls_imb_on = wting["cls_imb"]["type"] is not None
                focal_on = "focal" in wting
                dsmr_on = is_bce_family and wting["bce"]["dsmr"]
                if not (cls_imb_on or focal_on or dsmr_on):
                    del metadata[key]["wting"]  # no active weight factor -> W == ones -> whole block inert
                    continue

                cls_imb = wting["cls_imb"]
                if not cls_imb_on:
                    del wting["cls_imb"]
                else:
                    if cls_imb["type"] == "inv_freq":
                        del cls_imb["class_bal"]
                    elif cls_imb["type"] == "class_bal":
                        del cls_imb["inv_freq"]
                    if unit_scaled:
                        del cls_imb["norm"]

                if not is_bce_family:
                    del wting["bce"]

        metadata = asdict(cfg_train)
        clean_metadata(metadata)

        # coord-level config params (the arm's + coord's effective config) live in config.json (write-once;
        # asserted unchanged on later trials of the coord). coord_metadata.json holds only n_crashes --
        # mutable per-cause counters the campaign runner bumps on crashes -- so the two concerns don't
        # share a file.
        fpath_config = ArtifactManager.dpath_coord / "config.json"
        if fpath_config.exists():
            assert metadata == load_json(fpath_config), "Coord params changed!"
        else:
            save_json(metadata, fpath_config)

        fpath_meta = ArtifactManager.dpath_coord / "coord_metadata.json"
        if not fpath_meta.exists():
            # best_chkpt: per criterion x eval group, the checkpoint every trial of this
            # coord is scored at -- filled in at each trial end by report.update_chkpt_selection
            save_json(
                {
                    "n_crashes": {"ram": 0, "vram": 0, "other": 0},
                    "horizon": {},
                    "best_chkpt": {},
                },
                fpath_meta,
            )
        # horizon: the trial duration in samples and optimizer steps, with the LR warmup's share OF
        # each total (not additional to it). Sample volume is data-derived (train-set size); identical
        # across trials of a coord/dataset, so overwriting is idempotent. Every batch is a full
        # batch_size (drop_last) and training breaks the moment n_samps_seen >= sample_volume, hence
        # ceil; the warmup converts the way the trainer does (warmup fraction -> samples -> steps,
        # train.py's scheduler warmup-step count).
        metadata_coord = load_json(fpath_meta)
        warmup_samps = round(cfg_train.opt["lr"]["warmup"] * cfg_train.sample_volume)
        metadata_coord["horizon"] = {
            "n_samps": {"total": cfg_train.sample_volume, "warmup": warmup_samps},
            "n_steps": {
                "total": math.ceil(cfg_train.sample_volume / cfg_train.batch_size),
                "warmup": math.ceil(warmup_samps / cfg_train.batch_size),
            },
        }
        save_json(metadata_coord, fpath_meta)

    @staticmethod
    def _get_trial_runtime_data(data: TrialData, idx_epoch: int, time_tracker: TimeTracker):

        def fmt(seconds):
            return f"{seconds:.2f}" if seconds is not None else None

        def mean_bucket(name):
            return {"mean": fmt(time_tracker.mean(name)), "n": time_tracker.n(name)}

        # train mean keyed on idx_epoch (epochs *started*) to match its per-epoch cadence; shows
        # "0.00" between the first epoch's start and finish, as before.
        if idx_epoch > 0:
            mean_time_train = fmt(time_tracker.mean("train") or 0.0)
        else:
            mean_time_train = None

        time_trial = data.timer_trial.get_elapsed_time()
        # remainder of the trial wall-clock not attributed to a tracked bucket -- checkpoint I/O,
        # sync/barriers, and any in-progress epoch's train time not yet folded into the train mean
        time_other = time_trial - time_tracker.attributed()

        runtime_data = {
            "train": {"mean": mean_time_train, "n": idx_epoch},
            "eval": mean_bucket("eval"),
            "viz_compute": mean_bucket("viz_compute"),
            "other": fmt(time_other),
            "trial": fmt(time_trial),
        }

        return runtime_data

    @staticmethod
    @rank0
    def save_metadata_trial(data: TrialData, idx_epoch: int, time_tracker: TimeTracker, epoch: int, n_epochs, n_samps_seen: int, mem, init_flag=False):
        runtime_data = ArtifactManager._get_trial_runtime_data(data, idx_epoch, time_tracker)
        # epoch/n_epochs feed the manifest's progress display; n_samps_seen stays for crash-log keying
        progress_data = {"epoch": epoch, "n_epochs": n_epochs, "n_samps_seen": n_samps_seen}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if init_flag:
            metadata_trial = {
                "dataset": ArtifactManager.dataset,
                "split": ArtifactManager.split,
                "runtime": runtime_data,
                "progress": progress_data,
                "memory": format_mem(mem),
                "datetime_start": now,
                "datetime_last_seen": now,
                "complete": False,
                "n_crashes": {"ram": 0, "vram": 0, "other": 0},  # crashes this trial has recovered from, bucketed by cause; bumped by campaign_runner._bump_crash_counts
            }
        else:
            metadata_trial = load_json(ArtifactManager.fpath_metadata_trial)
            metadata_trial["runtime"] = runtime_data
            metadata_trial["progress"] = progress_data
            metadata_trial["memory"] = merge_mem(metadata_trial["memory"], format_mem(mem))
            metadata_trial["datetime_last_seen"] = now
        save_json(metadata_trial, ArtifactManager.fpath_metadata_trial)

    @staticmethod
    @rank0
    def save_eval_data(dpath_model, eval_metrics, idx_eval, n_chkpts, n_samps_seen, sample_volume):
        # one metrics file per eval group; the shared loss_raw/sim/targ/chkpt fields repeat in each
        dpath_model.mkdir(parents=True, exist_ok=True)
        formatted = format_scores(eval_metrics)
        chkpt = f"{idx_eval}/{n_chkpts} ({n_samps_seen / 1e6:.1f}M/{sample_volume / 1e6:.1f}M samples)"
        for group_key, scores_grp in formatted["scores"].items():
            metadata = {
                "scores": scores_grp,
                "loss_raw": formatted["loss_raw"],
                "sim": formatted["sim"],
                "targ": formatted["targ"],
                "chkpt": chkpt,
            }
            save_json(metadata, dpath_model / f"{group_key}.json")

    @staticmethod
    def base_eval_cache_fpath(cfg_train):
        # one pickle per combo in a flat dir, named by the serialized combo key -- a save touches
        # only its own combo's file (no shared-file rewrite) and a load reads only its own
        fname = "__".join(str(c) for c in ArtifactManager.base_eval_key(cfg_train)) + ".pkl"
        return paths["root"] / "base_eval_cache" / fname

    @staticmethod
    def base_eval_key(cfg_train):
        """Combo key for one base-eval reading: the config settings that determine the base model's
        eval output. Numerics-level knobs (hw mixed_prec, t-SNE perplexity) are
        deliberately not keyed. Family-inert components are normalized to None so equivalent configs
        share one entry: non_causal is CLIP-only, vis_proj_head is SigLIP-only, and seed only enters
        through the random init of a linear/mlp vis_proj_head."""
        from models import CLIP_MODELS, SIGLIP_MODELS  # local: models pulls open_clip/transformers, too heavy for module import
        model_type = cfg_train.arch["model_type"]
        non_causal = cfg_train.arch["clip"]["non_causal"] if model_type in CLIP_MODELS else None
        vis_proj_head = cfg_train.arch["siglip"]["vis_proj_head"] if model_type in SIGLIP_MODELS else None
        seed = cfg_train.seed if vis_proj_head is not None else None
        return (
            model_type,
            cfg_train.dataset,
            cfg_train.split,
            non_causal,
            cfg_train.text_template["eval"],
            vis_proj_head,
            seed,
        )

    @staticmethod
    @rank0
    def load_base_eval_cache(cfg_train, require_projections, require_embs):
        # @rank0: a concurrent same-combo campaign can create/replace this combo's file at any
        # moment, so independent per-rank reads could disagree on hit/miss; rank 0 alone reads and
        # the caller broadcasts the decision. Entries carry only what the caching trial computed
        # (metrics always; projections + embs for viz trials): a trial must read
        # an entry missing a piece it needs as a miss (recompute, overwriting the entry with the
        # richer version) rather than trip _write_base_eval on the missing piece downstream.
        # Leaner entries stay valid hits for trials that don't need the missing pieces.
        fpath = ArtifactManager.base_eval_cache_fpath(cfg_train)
        if not fpath.exists():
            return None
        entry = load_pickle(fpath)
        if require_projections and entry["projections"] is None:
            return None
        if require_embs and entry["embs"] is None:
            return None
        return entry

    @staticmethod
    @rank0
    def save_base_eval_cache(cfg_train, eval_metrics):
        """Write this combo's entry to its own cache file and return the entry. The npz arrays are
        ingested from this trial's evals/base/, where compute_projections just wrote them
        (both absent for non-viz trials; UMAP is fit post-trial, so entries never carry it). Written via temp file +
        atomic replace: concurrent same-combo campaigns overwrite each other with equivalent entries,
        and readers never see a torn file; other combos' files are untouched."""
        entry = {
            "metrics": {k: format_scores(v) for k, v in eval_metrics.items() if k not in ("loss_raw", "sim", "targ")},
            "projections": None,
            "embs": None,
        }
        dpath_base = ArtifactManager.dpath_trial / "evals" / "base"
        for name in ("projections", "embs"):
            fpath_npz = dpath_base / f"{name}.npz"
            if fpath_npz.exists():
                entry[name] = dict(np.load(fpath_npz))
        fpath = ArtifactManager.base_eval_cache_fpath(cfg_train)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath_tmp = fpath.with_name(f"{fpath.name}.{os.uname().nodename}.{os.getpid()}.tmp")  # node+pid: campaigns on different nodes can share a pid
        save_pickle(entry, fpath_tmp)
        fpath_tmp.replace(fpath)
        return entry

    @staticmethod
    @rank0
    def save_train_state(train_pipe, idx_batch):
        state = {
            "model": train_pipe.modelw._unwrapped_model.state_dict(),
            "optimizer": train_pipe.opt.state_dict(),
            "lr_sched": train_pipe.lr_sched.state_dict(),
            "n_samps_seen": train_pipe.n_samps_seen,
            "n_batches_seen": train_pipe.n_batches_seen,
            "idx_epoch": train_pipe.idx_epoch,
            "idx_batch": idx_batch,
            "chkpt_thresh": train_pipe.chkpt_thresh,
            "times": train_pipe.time_tracker.state_dict(),
        }
        torch.save(state, ArtifactManager.dpath_model_checkpoint / "train_state.pt")

    @staticmethod
    def save_rng_states(rank):
        rng_state = {
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "rng_numpy": np.random.get_state(),
            "rng_random": random.getstate(),
        }
        torch.save(rng_state, ArtifactManager.dpath_model_checkpoint / f"rng_state_rank{rank}.pt")

    @staticmethod
    @rank0
    def save_trial_state(data):
        state = {
            "n_evals": data.n_evals,
            "timer_trial_elapsed": data.timer_trial.get_elapsed_time(),
        }
        save_pickle(state, ArtifactManager.dpath_model_checkpoint / "trial_state.pkl")

    @staticmethod
    def load_train_state():
        return torch.load(
            ArtifactManager.dpath_model_checkpoint / "train_state.pt",
            map_location="cpu",
            weights_only=False,
        )

    @staticmethod
    def load_rng_state(rank):
        return torch.load(
            ArtifactManager.dpath_model_checkpoint / f"rng_state_rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )

    @staticmethod
    def load_trial_state():
        return load_pickle(ArtifactManager.dpath_model_checkpoint / "trial_state.pkl")
