import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from dataclasses import asdict
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
    DATASET_ALIAS2NAME,
)
from utils.ddp import rank0

import pdb


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


class TrialData:

    def __init__(self, dpath_trial):

        self.fpath_data = dpath_trial / "data_trial.pkl"  #!

        self.data_epoch = {
            "n_samps_seen": [],
            "lr": [],
            "loss_train": [],
            "loss_raw_train": [],
            "grad_norm_model": [],
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

    def update_train_batch(self, n_samps_seen, lr=None, loss_train=None, loss_raw_train=None, grad_norm_model=None):

        self.data_epoch["n_samps_seen"].append(n_samps_seen)

        if lr is not None:
            self.data_epoch["lr"].append(lr)
        if loss_train is not None:
            self.data_epoch["loss_train"].append(loss_train)
        if loss_raw_train is not None:
            self.data_epoch["loss_raw_train"].append(loss_raw_train)
        if grad_norm_model is not None:
            self.data_epoch["grad_norm_model"].append(grad_norm_model)

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
    dpath_setting = None
    dpath_trial = None
    fpath_metadata_trial = None
    dpath_model_final = None
    dpath_eval_final = None
    dpath_model_checkpoint = None
    resuming = False
    dataset = None
    split = None
    model_type = None
    img_norm = None

    @staticmethod
    def set_paths(cfg_train):

        ArtifactManager.dpath_campaign = paths["artifacts"] / cfg_train.campaign
        ArtifactManager.dpath_setting = ArtifactManager.dpath_campaign / cfg_train.setting
        ArtifactManager.dataset = cfg_train.dataset
        ArtifactManager.split = cfg_train.split
        ArtifactManager.model_type = cfg_train.arch["model_type"]
        ArtifactManager.img_norm = cfg_train.img_norm

        trial_name = cfg_train.seed
        ArtifactManager.dpath_trial = ArtifactManager.dpath_setting / cfg_train.dataset / str(trial_name)
        ArtifactManager.fpath_metadata_trial = ArtifactManager.dpath_trial / "trial_metadata.json"

        ArtifactManager.dpath_model_final = ArtifactManager.dpath_trial / "chkpts/final"
        ArtifactManager.dpath_eval_final = ArtifactManager.dpath_trial / "evals/final"
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
        for subdir in ("logs", "chkpts", "chkpts/in_progress", "learning_curves"):
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
    def save_metadata_setting(cfg_train):
        
        def clean_metadata(metadata):

            del metadata["campaign"]
            del metadata["setting"]
            del metadata["seed"]
            del metadata["idx_seed"]
            del metadata["dataset"]
            del metadata["split"]
            del metadata["standalone"]

            del metadata["dev"]
            
            metadata["loss"].pop("wting", None)
            metadata["loss"].pop("focal", None)
            metadata["loss"].pop("dsmr", None)

            if "loss2" in metadata:
                metadata["loss2"].pop("wting", None)
                metadata["loss2"].pop("focal", None)
                metadata["loss2"].pop("dsmr", None)

            if metadata["loss2"]["mix"] == 0.0:
                del metadata["loss2"]
        
        fpath_meta = ArtifactManager.dpath_setting / "setting_metadata.json"
        metadata = asdict(cfg_train)
        clean_metadata(metadata)
        if fpath_meta.exists():
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Setting params changed!"
        else:
            save_json(metadata, fpath_meta)

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
    def save_metadata_trial(data: TrialData, idx_epoch: int, time_tracker: TimeTracker, init_flag=False):
        runtime_data = ArtifactManager._get_trial_runtime_data(data, idx_epoch, time_tracker)
        if init_flag:
            metadata_trial = {
                "dataset": ArtifactManager.dataset,
                "split": ArtifactManager.split,
                "runtime": runtime_data,
                "complete": False,
            }
        else:
            metadata_trial = load_json(ArtifactManager.fpath_metadata_trial)
            metadata_trial["runtime"] = runtime_data
        save_json(metadata_trial, ArtifactManager.fpath_metadata_trial)

    @staticmethod
    @rank0
    def save_eval_data(dpath_model, eval_metrics, n_samps_seen_chkpt, n_samps_seen):
        dpath_model.mkdir(parents=True, exist_ok=True)
        fpath_meta = dpath_model / "metrics.json"
        metadata = {
            **format_scores(eval_metrics),
            "n_samps_seen": f"{n_samps_seen_chkpt:,}/{n_samps_seen:,}",
        }
        save_json(metadata, fpath_meta)

    @staticmethod
    def _aggregate_metric_stats(values, percent=True):
        first = values[0]
        if first is None:  # leaf is None for every trial (e.g. loss_raw["ood"] is never computed)
            return None
        if isinstance(first, dict):
            return {
                k: ArtifactManager._aggregate_metric_stats(
                    [v[k] for v in values], percent=percent and k != "loss_raw"
                )
                for k in first
            }
        if len(values) == 1:
            return values[0]
        nums = np.array([float(v) for v in values])
        mean = nums.mean()
        std = nums.std(ddof=1)
        if percent:
            return f"{mean * 100:.2f}% ± {std * 100:.2f}%"
        return f"{mean:.4f} ± {std:.4f}"

    @staticmethod
    @rank0
    def update_metric_stats():
        dpath_dataset = ArtifactManager.dpath_setting / ArtifactManager.dataset
        metric_dicts = []
        for dpath_trial in sorted(dpath_dataset.iterdir()):
            # a written final-eval metrics file is the signal a trial finished; the `complete` flag is
            # marked later (campaign_runner, after a clean exit) so it can't gate this aggregation
            fpath_metrics = dpath_trial / "evals/final/metrics.json"
            if not fpath_metrics.exists():
                continue
            metrics = load_json(fpath_metrics)
            metrics.pop("n_samps_seen", None)
            metric_dicts.append(metrics)

        if not metric_dicts:
            return

        stats = {
            "n_trials": len(metric_dicts),
            **ArtifactManager._aggregate_metric_stats(metric_dicts),
        }
        save_json(stats, dpath_dataset / "metric_stats.json")

    @staticmethod
    def base_eval_cache_dpath():
        return (
            paths["root"]
            / "base_eval_cache"
            / ArtifactManager.model_type
            / ArtifactManager.img_norm
            / ArtifactManager.dataset
            / ArtifactManager.split
        )

    @staticmethod
    def base_eval_cache_path():
        return ArtifactManager.base_eval_cache_dpath() / "metrics.json"

    @staticmethod
    def load_base_eval_cache():
        fpath = ArtifactManager.base_eval_cache_path()
        if not fpath.exists():
            return None
        return load_json(fpath)

    @staticmethod
    @rank0
    def save_base_eval_cache(eval_metrics):
        fpath = ArtifactManager.base_eval_cache_path()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        metadata = {k: format_scores(v) for k, v in eval_metrics.items() if k != "loss_raw"}
        save_json(metadata, fpath)

    @staticmethod
    @rank0
    def save_train_state(train_pipe, idx_batch):
        state = {
            "model": train_pipe.modelw._unwrapped_model.state_dict(),
            "norm_mean": train_pipe.modelw.norm_mean,
            "norm_std": train_pipe.modelw.norm_std,
            "optimizer": train_pipe.opt.state_dict(),
            "lr_sched": train_pipe.lr_sched.state_dict(),
            "n_samps_seen": train_pipe.n_samps_seen,
            "n_batches_seen": train_pipe.n_batches_seen,
            "idx_epoch": train_pipe.idx_epoch,
            "idx_batch": idx_batch,
            "chkpt_thresh": train_pipe.chkpt_thresh,
            "times": train_pipe.time_tracker.state_dict(),
        }
        if train_pipe.cfg.hw.mixed_prec:
            state["scaler"] = train_pipe.scaler.state_dict()
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


def _samples_seen_tick_formatter(value, _pos):
    return f"{value / 1_000_000:g}"

@rank0
def plot_metrics(
        data_tracker, 
        dpath_trial,
        fontsize_axes=12, 
        fontsize_ticks=8, 
        fontsize_legend=8,
        subplot_border_width=1,
        figsize=(10, 16),
        height_ratios=[2, 2, 2, 2, 2, 1, 1],
    ):
    data = data_tracker.data
    data_epoch = data["epoch"]
    data_eval = data["eval"]
    title_suffix = f" -- {ArtifactManager.dpath_setting.name}, {DATASET_ALIAS2NAME[ArtifactManager.dataset]}"

    # eval panels (retrieval / n-shot / accuracy) are populated only when eval ran;
    # train panels (loss / grad norm / lr) plot whenever train data is present (e.g. train_pt=trainval).
    partitions = [k for k in data_eval.get("scores", {}).get("closed_set", {}).get("standard", {}).keys() if k != "comp"]

    x_eval = data_eval["n_samps_seen"]
    x_train = data_epoch["n_samps_seen"]

    bucket_partition = "id" if "id" in partitions else None
    bucket_comp_keys_standard = [
        key for key in data_eval.get("scores", {}).get("closed_set", {}).get("standard", {}).get(bucket_partition, {}).get("map", {}).get("n-shot", {}).keys()
    ]
    bucket_comp_keys_full_set = [
        key
        for key in data_eval.get("scores", {}).get("full_set", {}).get("standard", {}).get(bucket_partition, {}).get("map", {}).get("n-shot", {}).keys()
    ]

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_standard,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="standard",
        full_set=False,
        retrieval_ylabel="mAP Scores",
        accuracy_ylabel="I2T Accuracy",
        nshot_accuracy_ylabel="n-shot Accuracy (ID)",
        plot_title=f"Train Metrics{title_suffix}",
        output_filename="closed_standard.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_standard,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="per_class",
        full_set=False,
        retrieval_ylabel="Macro mAP Scores",
        accuracy_ylabel="I2T Per-Class Accuracy",
        nshot_accuracy_ylabel="n-shot Per-Class\nAccuracy (ID)",
        plot_title=f"Train Metrics (Macro){title_suffix}",
        output_filename="closed_macro.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_full_set,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="standard",
        full_set=True,
        retrieval_ylabel="Full-Set mAP Scores",
        accuracy_ylabel="Full-Set I2T Accuracy",
        nshot_accuracy_ylabel="Full-Set n-shot Accuracy (ID)",
        plot_title=f"Train Metrics (Full-Set){title_suffix}",
        output_filename="full_standard.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partitions,
        bucket_partition,
        bucket_comp_keys_full_set,
        fontsize_axes,
        fontsize_ticks,
        fontsize_legend,
        subplot_border_width,
        figsize,
        height_ratios,
        partition_metric_group="per_class",
        full_set=True,
        retrieval_ylabel="Full-Set Macro mAP Scores",
        accuracy_ylabel="Full-Set I2T Per-Class Accuracy",
        nshot_accuracy_ylabel="Full-Set n-shot Per-Class\nAccuracy (ID)",
        plot_title=f"Train Metrics (Macro Full-Set){title_suffix}",
        output_filename="full_macro.png",
    )

def plot_composite_metrics(
    data_epoch,
    data_eval,
    x_train,
    x_eval,
    dpath_trial,
    partitions,
    bucket_partition,
    bucket_comp_keys,
    fontsize_axes,
    fontsize_ticks,
    fontsize_legend,
    subplot_border_width,
    figsize,
    height_ratios,
    partition_metric_group,
    full_set,
    retrieval_ylabel,
    accuracy_ylabel,
    nshot_accuracy_ylabel,
    plot_title,
    output_filename,
):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    ax0 = fig.add_subplot(gs[0, 0])

    id_partition = "id" if "id" in partitions else None
    ood_partition = "ood" if "ood" in partitions else None
    retrieval_specs = (
        ("i2t", "I2T", "blue"),
        ("i2i", "I2I", "red"),
        ("t2i", "T2I", "green"),
    )
    set_key = "full_set" if full_set else "closed_set"
    style_specs = (
        (id_partition, "ID", "-"),
        (ood_partition, "OOD", "--"),
    )
    for partition, partition_label, linestyle in style_specs:
        if partition is None:
            continue
        partition_group_scores = data_eval.get("scores", {}).get(set_key, {}).get(partition_metric_group, {}).get(partition, {})
        partition_group_scores = partition_group_scores.get("map", {})
        for metric_name, metric_label, color in retrieval_specs:
            if metric_name in partition_group_scores:
                ax0.plot(
                    x_eval,
                    partition_group_scores[metric_name],
                    label=f"{partition_label} {metric_label}",
                    color=color,
                    linestyle=linestyle,
                )

    ax0.set_ylabel(retrieval_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax0.set_ylim(0, 1)
    if partitions:
        ax0.legend(loc="lower right", fontsize=fontsize_legend)
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    is_macro_plot = partition_metric_group == "per_class"
    id_mode_scores = data_eval.get("scores", {}).get(set_key, {}).get(partition_metric_group, {}).get(bucket_partition, {})
    nshot_key = "n-shot"
    if is_macro_plot:
        nshot_ylabel = "Full-Set n-shot Macro mAP (ID)" if full_set else "n-shot Macro mAP (ID)"
    else:
        nshot_ylabel = "Full-Set n-shot mAP (ID)" if full_set else "n-shot mAP (ID)"
    comp_nshot = id_mode_scores.get("map", {}).get(nshot_key, {})
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            label = key
            maybe_plot(ax1, x_eval, comp_nshot, key, label, linestyle=":")
        if comp_nshot:
            ax1.legend(loc="lower right", fontsize=fontsize_legend)
    ax1.set_ylabel(nshot_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    for partition in partitions:
        partition_group_scores = data_eval.get("scores", {}).get(set_key, {}).get(partition_metric_group, {}).get(partition, {})
        partition_group_scores = partition_group_scores.get("acc", {})
        if "i2t" in partition_group_scores:
            ax2.plot(
                x_eval,
                partition_group_scores["i2t"],
                label="-".join([s.upper() if i == 0 else s.title() for i, s in enumerate(partition.split("_"))]),
            )
    ax2.set_ylabel(accuracy_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax2.set_ylim(0, 1)
    if partitions:
        ax2.legend(loc="lower right", fontsize=fontsize_legend)
    ax2.grid(True)
    ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    comp_nshot_acc = id_mode_scores.get("acc", {}).get("n-shot", {})
    if bucket_comp_keys:
        for key in reversed(bucket_comp_keys):
            maybe_plot(ax3, x_eval, comp_nshot_acc, key, key, linestyle=":")
        if comp_nshot_acc:
            ax3.legend(loc="lower right", fontsize=fontsize_legend)
    ax3.set_ylabel(nshot_accuracy_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax3.set_ylim(0, 1)
    ax3.grid(True)
    ax3.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
    if len(data_epoch.get("loss_train", [])) == len(x_train):
        ax4.plot(x_train, data_epoch["loss_train"], label="Train Loss")
    if len(data_epoch.get("loss_raw_train", [])) == len(x_train):
        ax4.plot(x_train, data_epoch["loss_raw_train"], label="Train Loss (Raw)")
    for partition in partitions:
        if len(data_eval.get("loss_raw", {}).get(partition, [])) == len(x_eval):
            ax4.plot(
                x_eval,
                data_eval["loss_raw"][partition],
                label=f'{"-".join([s.upper() if i == 0 else s.title() for i, s in enumerate(partition.split("_"))])} Val Loss',
            )
    ax4.set_ylabel("Loss", fontsize=fontsize_axes, fontweight="bold")
    ax4.set_yscale("log")
    ax4.minorticks_on()
    ax4.grid(which="minor", axis="y")
    ax4.legend(loc="upper right", fontsize=fontsize_legend)
    ax4.grid(True)
    ax4.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)
    if len(data_epoch.get("grad_norm_model", [])) == len(x_train):
        ax5.plot(x_train, data_epoch["grad_norm_model"], color="green")
    ax5.set_ylabel("Grad Norm", fontsize=fontsize_axes, fontweight="bold")
    ax5.set_yscale("log")
    ax5.minorticks_on()
    ax5.grid(which="minor", axis="y")
    ax5.grid(True)
    ax5.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax6 = fig.add_subplot(gs[6, 0], sharex=ax0)
    if len(data_epoch.get("lr", [])) == len(x_train):
        ax6.plot(x_train, data_epoch["lr"])
    ax6.set_ylabel("Learning Rate", fontsize=fontsize_axes, fontweight="bold")
    ax6.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax6.yaxis.set_offset_position("right")
    ax6.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax6.yaxis.get_offset_text().set_visible(False)
    ax6.set_xlabel("Samples Seen (M)", fontsize=fontsize_axes, fontweight="bold")
    ax6.xaxis.set_major_formatter(FuncFormatter(_samples_seen_tick_formatter))
    ax6.grid(True)
    ax6.tick_params(labelsize=fontsize_ticks)

    for ax in (ax0, ax1, ax2, ax3, ax4, ax5, ax6):
        ax.label_outer()

    for idx_ax, ax in enumerate((ax0, ax1, ax2, ax3, ax4, ax5, ax6)):
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_border_width)
            spine.set_edgecolor("black")
        if idx_ax % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    fig.suptitle(plot_title, fontweight="bold", y=0.98, fontsize=20)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plots_dir = dpath_trial / "learning_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / output_filename)
    plt.close(fig)

def maybe_plot(ax, x, data, key, label, **kwargs):
    """
    Helper for plot_metrics() (N-Shot Composites)
    """
    if key in data and len(data[key]) > 0:
        ax.plot(x, data[key], label=label, **kwargs)