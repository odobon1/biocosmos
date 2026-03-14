import matplotlib  # type: ignore[import]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.gridspec as gridspec  # type: ignore[import]
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator  # type: ignore[import]
import numpy as np  # type: ignore[import]
import shutil
from dataclasses import asdict

from utils.utils import (
    paths, 
    save_json, 
    load_json, 
    get_text_preps, 
)


class ArtifactManager:

    dpath_study = None
    dpath_experiment = None
    dpath_trial = None
    dpath_model_best_comp = None
    dpath_model_best_img2img = None
    dpath_model_checkpoint = None

    @staticmethod
    def set_paths(cfg_train):

        ArtifactManager.dpath_study = paths["artifacts"] / cfg_train.study_name
        ArtifactManager.dpath_experiment = ArtifactManager.dpath_study / cfg_train.experiment_name

        trial_name = cfg_train.seed
        ArtifactManager.dpath_trial = ArtifactManager.dpath_experiment / str(trial_name)

        if ArtifactManager.dpath_trial.exists():
            if cfg_train.dev['allow_overwrite_trial']:
                shutil.rmtree(ArtifactManager.dpath_trial)
            else:
                raise ValueError(f"Trial directory '{cfg_train.study_name}/{cfg_train.experiment_name}/{cfg_train.seed}' already exists!")

        ArtifactManager.dpath_model_best_comp    = ArtifactManager.dpath_trial / "models/best_comp"
        ArtifactManager.dpath_model_best_img2img = ArtifactManager.dpath_trial / "models/best_img2img"
        ArtifactManager.dpath_model_checkpoint   = ArtifactManager.dpath_trial / "models/checkpoint"

    @staticmethod
    def create_trial_dirs():
        for subdir in ("logs", "models", "models/checkpoint", "models/best_comp", "models/best_img2img", "plots"):
            (ArtifactManager.dpath_trial / subdir).mkdir(parents=True)

    @staticmethod
    def save_metadata_study(cfg_train):
        fpath_meta = ArtifactManager.dpath_study / "metadata_study.json"
        metadata   = {
            "split_name":      cfg_train.split_name,
            "n_gpus":          cfg_train.n_gpus,
            "n_cpus":          cfg_train.n_cpus,
            "ram":             f"{cfg_train.ram} GB",
            "n_workers":       cfg_train.n_workers,
            "prefetch_factor": cfg_train.prefetch_factor,
        }
        if fpath_meta.exists() and not cfg_train.dev['allow_diff_study']:
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Study params changed!"
        else:
            save_json(metadata, fpath_meta)

    @staticmethod
    def save_metadata_experiment(cfg_train):
        
        def clean_metadata(metadata):

            del metadata["study_name"]
            del metadata["experiment_name"]
            del metadata["seed"]
            del metadata["split_name"]

            del metadata["dev"]

            del metadata["chkpt_every"]
            
            metadata["loss"].pop("wting", None)
            metadata["loss"].pop("focal", None)
            metadata["loss"].pop("dyn_smr", None)

            if "loss2" in metadata:
                metadata["loss2"].pop("wting", None)
                metadata["loss2"].pop("focal", None)
                metadata["loss2"].pop("dyn_smr", None)

            if metadata["loss2"]["mix"] == 0.0:
                del metadata["loss2"]
        
        fpath_meta = ArtifactManager.dpath_experiment / "metadata_experiment.json"
        metadata   = asdict(cfg_train)

        # save full text combo-templates themselves and not just the names
        text_preps_full = {}
        for split_name, text_preps in metadata["text_preps"].items():
            text_preps_full[split_name] = get_text_preps(text_preps)
        metadata["text_preps"] = text_preps_full

        clean_metadata(metadata)
        if fpath_meta.exists() and not cfg_train.dev['allow_diff_experiment']:
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Experiment params changed!"
        else:
            save_json(metadata, fpath_meta)

    @staticmethod
    def save_metadata_trial(time_train_mean=None, time_val_mean=None):
        fpath_meta = ArtifactManager.dpath_trial / "metadata_trial.json"
        if time_train_mean is not None:
            time_train_mean = f"{time_train_mean:.2f}"
            time_val_mean   = f"{time_val_mean:.2f}"
        metadata = {
            "runtime_perf": {
                "train_mean": time_train_mean, 
                "val_mean": time_val_mean,
            },
        }
        save_json(metadata, fpath_meta)

    @staticmethod
    def save_metadata_model(dpath_model, scores_val, idx_epoch_chkpt, idx_epoch):
        fpath_meta = dpath_model / "metadata_model.json"
        scores_val = {k: f"{v:.4f}" for k, v in scores_val.items()}
        metadata = {
            "scores_val": scores_val,
            "idx_epoch":  f"{idx_epoch_chkpt}/{idx_epoch}",
        }
        save_json(metadata, fpath_meta)

def plot_metrics(
        data_tracker, 
        dpath_trial,
        fontsize_axes       =12, 
        fontsize_ticks      =8, 
        fontsize_legend     =8,
        subplot_border_width=1,
        figsize             =(10, 12),
        height_ratios       =[2, 2, 2, 2, 1],
    ):
    data = data_tracker.data

    x_len = len(data["id_img2txt_prec1"])
    x     = list(range(0, x_len))

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    ##########################################################################################################
    ##########################################################################################################
    # Plot 1: mAP Scores
    ax0 = fig.add_subplot(gs[0, 0])

    ax0.plot(x, data["id_img2txt_map"],  label="ID img2txt mAP",  color="blue")
    ax0.plot(x, data["id_img2img_map"],  label="ID img2img mAP",  color="red")
    ax0.plot(x, data["id_txt2img_map"],  label="ID txt2img mAP",  color="green")
    ax0.plot(x, data["ood_img2txt_map"], label="OOD img2txt mAP", color="blue",  linestyle="--")
    ax0.plot(x, data["ood_img2img_map"], label="OOD img2img mAP", color="red",   linestyle="--")
    ax0.plot(x, data["ood_txt2img_map"], label="OOD txt2img mAP", color="green", linestyle="--")

    ax0.set_ylabel("mAP Scores", fontsize=fontsize_axes, fontweight="bold")
    ax0.set_ylim(0, 1)

    ax0.legend(loc="lower right", fontsize=fontsize_legend)
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 2: mAP Composites
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    ax1.plot(x, data["comp_map"], label="Composite", color="black")
    ax1.plot(x, data["img2img_map"], label="img2img Composite", color="#B22222")

    ax1.set_ylabel("mAP Composites", fontsize=fontsize_axes, fontweight="bold")
    ax1.set_ylim(0, 1)

    ax1.legend(loc="lower right", fontsize=fontsize_legend)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 3: Precision@1
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    ax2.plot(x, data["id_img2txt_prec1"], label="ID img2txt Prec@1", color="blue")
    ax2.plot(x, data["ood_img2txt_prec1"], label="OOD img2txt Prec@1", color="blue", linestyle="--")

    ax2.set_ylabel("Precision@1", fontsize=fontsize_axes, fontweight="bold")
    ax2.set_ylim(0, 1)

    ax2.legend(loc="lower right", fontsize=fontsize_legend)
    ax2.grid(True)
    ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 4: Loss
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    ax3.plot(x, [np.nan] + data["loss_train"], label="Train Loss")
    ax3.plot(x, [np.nan] + data["loss_raw_train"], label="Train Loss (Raw)")
    ax3.plot(x, data["id_loss"], label="ID Val Loss")
    ax3.plot(x, data["ood_loss"], label="OOD Val Loss")
    ax3.plot(x, data["comp_loss"], label="Comp Val Loss")

    ax3.set_ylabel("Loss", fontsize=fontsize_axes, fontweight="bold")
    ax3.set_yscale("log")
    # for plotting minor gridlines on the y axis
    ax3.minorticks_on()
    ax3.grid(which="minor", axis="y")

    ax3.legend(loc="upper right", fontsize=fontsize_legend)
    ax3.grid(True)
    ax3.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 5: Learning Rate
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)

    ax4.plot(x, [np.nan] + data["lr"])

    ax4.set_ylabel("Learning Rate", fontsize=fontsize_axes, fontweight="bold")
    ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax4.yaxis.set_offset_position("right")
    ax4.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax4.yaxis.get_offset_text().set_visible(False)

    ax4.set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")  # last subgraph gets the x label
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.xaxis.set_minor_locator(NullLocator())

    ax4.grid(True)
    ax4.tick_params(labelsize=fontsize_ticks)
    ##########################################################################################################

    for ax in (ax0, ax1, ax2, ax3):
        ax.label_outer()

    # thick black borders on all subplots
    for idx_ax, ax in enumerate([ax0, ax1, ax2, ax3, ax4]):
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_border_width)
            spine.set_edgecolor("black")
        if idx_ax % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    fig.suptitle("Train Metrics", fontweight="bold", y=0.98, fontsize=20)

    plt.subplots_adjust(hspace=0)  # remove extra whitespace to ensure plots are vertically flush
    plt.tight_layout()

    savepath = dpath_trial / "plots" / f"train_metrics.png"
    fig.savefig(savepath)
    plt.close(fig)
