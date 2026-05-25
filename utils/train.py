import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import shutil
from dataclasses import asdict
import torch
from PIL import Image

from utils.utils import (
    paths, 
    save_json, 
    load_json, 
    get_text_template, 
)


class ArtifactManager:

    dpath_campaign = None
    dpath_setting = None
    dpath_trial = None
    dpath_model_best_comp = None
    dpath_model_best_i2i = None
    dpath_model_checkpoint = None
    dpath_train_imgs = None
    fpath_train_imgs_manifest = None

    @staticmethod
    def set_paths(cfg_train):

        ArtifactManager.dpath_campaign = paths["artifacts"] / cfg_train.campaign_name
        ArtifactManager.dpath_setting = ArtifactManager.dpath_campaign / cfg_train.setting_name

        trial_name = cfg_train.seed
        ArtifactManager.dpath_trial = ArtifactManager.dpath_setting / cfg_train.dataset / str(trial_name)

        if ArtifactManager.dpath_trial.exists():
            if cfg_train.dev['allow_overwrite_trial']:
                shutil.rmtree(ArtifactManager.dpath_trial)
            else:
                raise ValueError(f"Trial directory '{cfg_train.campaign_name}/{cfg_train.setting_name}/{cfg_train.dataset}/{cfg_train.seed}' already exists!")

        ArtifactManager.dpath_model_best_comp    = ArtifactManager.dpath_trial / "models/best_comp"
        ArtifactManager.dpath_model_best_i2i = ArtifactManager.dpath_trial / "models/best_img2img"
        ArtifactManager.dpath_model_checkpoint   = ArtifactManager.dpath_trial / "models/checkpoint"

        view_imgs = int(cfg_train.dev.get("view_imgs", 0) or 0)
        if view_imgs > 0:
            ArtifactManager.dpath_train_imgs = ArtifactManager.dpath_trial / "train_imgs"
            ArtifactManager.fpath_train_imgs_manifest = ArtifactManager.dpath_train_imgs / "manifest.json"
        else:
            ArtifactManager.dpath_train_imgs = None
            ArtifactManager.fpath_train_imgs_manifest = None

    @staticmethod
    def create_trial_dirs():
        for subdir in ("logs", "models", "models/checkpoint", "models/best_comp", "models/best_img2img", "plots"):
            (ArtifactManager.dpath_trial / subdir).mkdir(parents=True)
        if ArtifactManager.dpath_train_imgs is not None:
            ArtifactManager.dpath_train_imgs.mkdir(parents=True)

    @staticmethod
    def save_metadata_campaign(cfg_train):
        fpath_meta = ArtifactManager.dpath_campaign / "metadata_campaign.json"
        metadata   = {
            "dataset":         cfg_train.dataset,
            "split_name":      cfg_train.split_name,
            "n_gpus":          cfg_train.n_gpus,
            "n_cpus":          cfg_train.n_cpus,
            "ram":             f"{cfg_train.ram} GB",
            "n_workers":       cfg_train.n_workers,
            "prefetch_factor": cfg_train.prefetch_factor,
        }
        if fpath_meta.exists() and not cfg_train.dev['allow_diff_campaign']:
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Campaign params changed!"
        else:
            save_json(metadata, fpath_meta)

    @staticmethod
    def save_metadata_setting(cfg_train):
        
        def clean_metadata(metadata):

            del metadata["campaign_name"]
            del metadata["setting_name"]
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
        
        fpath_meta = ArtifactManager.dpath_setting / "metadata_setting.json"
        metadata   = asdict(cfg_train)

        # save full text combo-templates themselves and not just the names
        text_template_full = {}
        for split_name, text_template in metadata["text_template"].items():
            text_template_full[split_name] = get_text_template(text_template, dataset=metadata["dataset"])
        metadata["text_template"] = text_template_full

        clean_metadata(metadata)
        if fpath_meta.exists() and not cfg_train.dev['allow_diff_setting']:
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Setting params changed!"
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
    def save_metadata_model(dpath_model, scores_val, samps_seen_chkpt, samps_seen):
        def format_scores(scores):
            if isinstance(scores, dict):
                return {k: format_scores(v) for k, v in scores.items()}
            return f"{float(scores):.4f}"

        fpath_meta = dpath_model / "metadata_model.json"
        scores_val = format_scores(scores_val)
        metadata = {
            "scores_val": scores_val,
            "samps_seen": f"{samps_seen_chkpt:,}/{samps_seen:,}",
        }
        save_json(metadata, fpath_meta)


def _samples_seen_tick_formatter(value, _pos):
    return f"{value / 1_000_000:g}"


class TrainImageDumper:

    def __init__(self, cfg, gpu_rank: int, gpu_world_size: int, img_pp_train):
        self.target = int(cfg.dev.get("view_imgs", 0) or 0)
        self.saved = 0
        self.gpu_rank = gpu_rank
        self.gpu_world_size = gpu_world_size
        self.dpath_train_imgs = ArtifactManager.dpath_train_imgs
        self.fpath_manifest = ArtifactManager.fpath_train_imgs_manifest

        self.norm_mean, self.norm_std = self._infer_norm_stats(img_pp_train)
        self.enabled = (
            self.target > 0
            and self.gpu_rank == 0
            and self.dpath_train_imgs is not None
            and self.fpath_manifest is not None
        )

        if self.enabled:
            self._save_manifest()

    def _infer_norm_stats(self, img_pp_train):
        transforms = getattr(img_pp_train, "transforms", None)
        if transforms is None:
            return None, None

        for transform in transforms:
            mean = getattr(transform, "mean", None)
            std = getattr(transform, "std", None)
            if mean is None or std is None:
                continue

            mean_vals = [float(v) for v in mean]
            std_vals = [float(v) for v in std]
            if len(mean_vals) == len(std_vals) and len(mean_vals) > 0:
                return mean_vals, std_vals

        return None, None

    def _sanitize(self, value) -> str:
        text = str(value)
        return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)

    def _save_manifest(self):
        normalization = None
        if self.norm_mean is not None and self.norm_std is not None:
            normalization = {
                "mean": self.norm_mean,
                "std": self.norm_std,
                "source": "img_pp_train",
            }

        save_json(
            {
                "view_imgs_target": self.target,
                "saved": self.saved,
                "gpu_world_size": self.gpu_world_size,
                "writer_gpu_rank": self.gpu_rank,
                "normalization": normalization,
            },
            self.fpath_manifest,
        )

    def _tensor_to_pil(self, img_t: torch.Tensor) -> Image.Image:
        img = img_t.detach().cpu().float()

        if img.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims [C,H,W], got shape {tuple(img.shape)}")

        if self.norm_mean is not None and self.norm_std is not None and img.shape[0] == len(self.norm_mean):
            mean_t = torch.tensor(self.norm_mean, dtype=img.dtype).view(-1, 1, 1)
            std_t = torch.tensor(self.norm_std, dtype=img.dtype).view(-1, 1, 1)
            img = img * std_t + mean_t

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if img.shape[0] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[0]}")

        img = img.clamp(0.0, 1.0)
        img_u8 = img.mul(255.0).add(0.5).to(torch.uint8)
        img_u8 = img_u8.permute(1, 2, 0).contiguous().numpy()
        return Image.fromarray(img_u8)

    def dump(self, imgs_sb: torch.Tensor, targ_data_sb):
        if not self.enabled or self.saved >= self.target:
            return

        n_save = min(self.target - self.saved, imgs_sb.size(0))

        for idx in range(n_save):
            cid = "na"
            class_enc = "na"
            if targ_data_sb and idx < len(targ_data_sb):
                targ = targ_data_sb[idx]
                if isinstance(targ, dict):
                    cid = self._sanitize(targ.get("cid", "na"))
                    class_enc = self._sanitize(targ.get("class_enc", "na"))

            fpath_img = self.dpath_train_imgs / f"{self.saved:06d}_cid-{cid}_class-{class_enc}.png"
            self._tensor_to_pil(imgs_sb[idx]).save(fpath_img)
            self.saved += 1

        self._save_manifest()


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

    partition_names = [
        partition_name
        for partition_name, partition_scores in data_eval.items()
        if isinstance(partition_scores, dict)
        and "standard" in partition_scores
        and "per_class" in partition_scores
    ]
    if not partition_names or "comp" not in data_eval:
        return

    x_eval = data_eval["samps_seen"]
    x_train = data_epoch["samps_seen"]

    bucket_partition_name = next((name for name in partition_names if name.startswith("id")), None)
    bucket_comp_keys_standard = [
        key for key in data_eval.get(bucket_partition_name, {}).get("standard", {}).get("map", {}).get("n-shot", {}).keys()
    ]
    bucket_comp_keys_full_set = [
        key
        for key in data_eval.get(bucket_partition_name, {}).get("standard", {}).get("full_set", {}).get("map", {}).get("n-shot", {}).keys()
    ]

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partition_names,
        bucket_partition_name,
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
        plot_title="Train Metrics",
        output_filename="train_metrics.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partition_names,
        bucket_partition_name,
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
        plot_title="Train Metrics (Macro)",
        output_filename="train_metrics_macro.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partition_names,
        bucket_partition_name,
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
        plot_title="Train Metrics (Full-Set)",
        output_filename="train_metrics_fullset.png",
    )

    plot_composite_metrics(
        data_epoch,
        data_eval,
        x_train,
        x_eval,
        dpath_trial,
        partition_names,
        bucket_partition_name,
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
        plot_title="Train Metrics (Macro Full-Set)",
        output_filename="train_metrics_macro_fullset.png",
    )

def plot_composite_metrics(
    data_epoch,
    data_eval,
    x_train,
    x_eval,
    dpath_trial,
    partition_names,
    bucket_partition_name,
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

    id_partition_name = next((name for name in partition_names if name.startswith("id")), None)
    ood_partition_name = next((name for name in partition_names if name.startswith("ood")), None)
    retrieval_specs = (
        ("i2t", "I2T", "blue"),
        ("i2i", "I2I", "red"),
        ("t2i", "T2I", "green"),
    )
    style_specs = (
        (id_partition_name, "ID", "-"),
        (ood_partition_name, "OOD", "--"),
    )
    for partition_name, partition_label, linestyle in style_specs:
        if partition_name is None:
            continue
        partition_group_scores = data_eval.get(partition_name, {}).get(partition_metric_group, {})
        if full_set:
            partition_group_scores = partition_group_scores.get("full_set", {})
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
    ax0.legend(loc="lower right", fontsize=fontsize_legend)
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    is_macro_plot = partition_metric_group == "per_class"
    id_mode_scores = data_eval.get(bucket_partition_name, {}).get(partition_metric_group, {})
    if full_set:
        id_mode_scores = id_mode_scores.get("full_set", {})
    nshot_key = "n-shot"
    if is_macro_plot:
        nshot_ylabel = "Full-Set n-shot Macro mAP (ID)" if full_set else "n-shot Macro mAP (ID)"
    else:
        nshot_ylabel = "Full-Set n-shot mAP (ID)" if full_set else "n-shot mAP (ID)"
    comp_nshot = id_mode_scores.get("map", {}).get(nshot_key, {})
    if bucket_comp_keys:
        for key in bucket_comp_keys:
            label = key
            maybe_plot(ax1, x_eval, comp_nshot, key, label, linestyle=":")
        if comp_nshot:
            ax1.legend(loc="lower right", fontsize=fontsize_legend)
    ax1.set_ylabel(nshot_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    for partition_name in partition_names:
        partition_group_scores = data_eval.get(partition_name, {}).get(partition_metric_group, {})
        if full_set:
            partition_group_scores = partition_group_scores.get("full_set", {})
        partition_group_scores = partition_group_scores.get("acc", {})
        if "i2t" in partition_group_scores:
            ax2.plot(
                x_eval,
                partition_group_scores["i2t"],
                label="-".join([s.upper() if i == 0 else s.title() for i, s in enumerate(partition_name.split("_"))]),
            )
    ax2.set_ylabel(accuracy_ylabel, fontsize=fontsize_axes, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right", fontsize=fontsize_legend)
    ax2.grid(True)
    ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    comp_nshot_acc = id_mode_scores.get("acc", {}).get("n-shot", {})
    if bucket_comp_keys:
        for key in bucket_comp_keys:
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
    for partition_name in partition_names:
        if len(data_eval.get(partition_name, {}).get("loss", [])) == len(x_eval):
            ax4.plot(
                x_eval,
                data_eval[partition_name]["loss"],
                label=f'{"-".join([s.upper() if i == 0 else s.title() for i, s in enumerate(partition_name.split("_"))])} Val Loss',
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
        ax5.plot(x_train, data_epoch["grad_norm_model"], label="Model Grad Norm", color="green")
    ax5.set_ylabel("Model Grad Norm", fontsize=fontsize_axes, fontweight="bold")
    ax5.set_yscale("log")
    ax5.minorticks_on()
    ax5.grid(which="minor", axis="y")
    ax5.legend(loc="upper right", fontsize=fontsize_legend)
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
    fig.savefig(dpath_trial / f"plots/{output_filename}")
    plt.close(fig)

def maybe_plot(ax, x, data, key, label, **kwargs):
    """
    Helper for plot_metrics() (N-Shot Composites)
    """
    if key in data and len(data[key]) > 0:
        ax.plot(x, data[key], label=label, **kwargs)