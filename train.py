import torch  # type: ignore[import]
from torch.amp import autocast, GradScaler  # type: ignore[import]
from torch.optim.lr_scheduler import (  # type: ignore[import]
    ExponentialLR, 
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
)
from tqdm import tqdm  # type: ignore[import]
import time
from datetime import datetime, timezone
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.gridspec as gridspec  # type: ignore[import]
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator  # type: ignore[import]
import numpy as np


from utils import (
    paths, 
    save_json, 
    load_json, 
    save_pickle, 
    seed_libs,
    get_text_preps, 
    compute_dataloader_workers_prefetch,
)
from models import VLMWrapper
from utils_data import spawn_dataloader, spawn_indexes_imgs
from utils_eval import ValidationPipeline

import pdb

torch.set_printoptions(
    precision=4,
    sci_mode =False,
    threshold=1000,  # total elements before summarizing
    edgeitems=3,  # num items to show at the start/end of each dim
    linewidth=120
)


@dataclass
class TrainConfig:
    study_name: str
    experiment_name: str
    seed: int | None

    allow_overwrite_trial: bool
    allow_diff_study: bool
    allow_diff_experiment: bool

    model_type: str
    loss_type: str
    split_name: str

    n_epochs: int
    checkpoint_every: int
    batch_size_train: int
    batch_size_val: int
    
    lr_init: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float

    lr_sched: dict

    freeze_text_encoder: bool
    freeze_image_encoder: bool

    cached_imgs: str | None
    mixed_prec: bool
    drop_partial_batch_train: bool
    verbose_batch_loss: bool

    text_preps_type_train: str
    text_preps_type_val: str

    def __post_init__(self):
        if self.freeze_text_encoder and self.freeze_image_encoder:
            raise ValueError("Text and image encoders are both set to frozen!")
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
            f"Split ------------------- {self.split_name}",
            f"",
            f"Batch Size (Train) ------ {self.batch_size_train}",
            f"",
            f"LR Init ----------------- {self.lr_init}",
            f"Weight Decay ------------ {self.weight_decay}",
            f"(β1, β2) ---------------- ({self.beta1}, {self.beta2})",
            f"ε (Optimizer) ----------- {self.eps}",
            f"",
            f"LR Scheduler ------------ {self.lr_sched['type']}",
            sep="\n"
        )

        if self.lr_sched['type'] == "exp":
            print(f"~ Gamma (Decay) --------- {self.lr_sched['args']['gamma']}")
        elif self.lr_sched['type'] == "plat":
            print(f"~ Factor (Decay) -------- {self.lr_sched['args']['factor']}")
            print(f"~ Patience -------------- {self.lr_sched['args']['patience']}")
            print(f"~ Cooldown -------------- {self.lr_sched['args']['cooldown']}")
            print(f"~ LR Min ---------------- {self.lr_sched['args']['min_lr']}")
        elif self.lr_sched['type'] == "cos":
            print(f"~ Half-Period (T_max) --- {self.lr_sched['args']['T_max']}")
            print(f"~ LR Min (eta_min) ------ {self.lr_sched['args']['eta_min']}")
        elif self.lr_sched['type'] == "coswr":
            print(f"~ Period (T_0) ---------- {self.lr_sched['args']['T_0']}")
            print(f"~ LR Min (eta_min) ------ {self.lr_sched['args']['eta_min']}")
        
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
    with open(Path(__file__).parent / "config_train.yaml") as f:
        config_train_dict = yaml.safe_load(f)
    config_train = TrainConfig(**config_train_dict)
    return config_train

class TrialDataTracker:

    def __init__(self, dpath_trial):

        self.fpath_data = dpath_trial / "data_trial.pkl"

        self.data = {
            "id_img2txt_prec1":  [],
            "id_img2txt_map":    [],
            "id_img2img_map":    [],
            "id_txt2img_map":    [],
            "id_loss":           [],
            "ood_img2txt_prec1": [],
            "ood_img2txt_map":   [],
            "ood_img2img_map":   [],
            "ood_txt2img_map":   [],
            "ood_loss":          [],
            "comp":              [],
            "comp_img2img":      [],
            "comp_loss":         [],
            "lr":                [],
            "loss_train":        [],
        }

    def update(self, scores_val, lr=None, loss_train=None):

        for score in scores_val.keys():
            self.data[score].append(float(scores_val[score]))

        if lr is not None:
            self.data["lr"].append(lr)
        if loss_train is not None:
            self.data["loss_train"].append(loss_train)

    def save(self):
        save_pickle(self.data, self.fpath_data)

class TrainPipeline:

    def __init__(self, modelw, config_train):
        self.datetime_init = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC"

        self.modelw = modelw
        self.cfg    = config_train
        self.lr     = self.cfg.lr_init

        self.modelw.freeze(self.cfg.freeze_text_encoder, self.cfg.freeze_image_encoder)

        self.set_paths()
        self.create_trial_dirs()
        self.save_metadata_study()
        self.save_metadata_experiment()

        index_imgs_class_enc, index_imgs_rfpaths, index_imgs_sids, _ = spawn_indexes_imgs(
            split_type="train",
            split_name=self.cfg.split_name,
        )

        text_preps_train                  = get_text_preps(self.cfg.text_preps_type_train)
        self.dataloader, time_cache_train = spawn_dataloader(
            index_imgs_class_enc=index_imgs_class_enc,
            index_imgs_rfpaths  =index_imgs_rfpaths,
            index_imgs_sids     =index_imgs_sids,
            text_preps          =text_preps_train,
            batch_size          =self.cfg.batch_size_train,
            shuffle             =True,
            drop_last           =self.cfg.drop_partial_batch_train,
            img_pp              =self.modelw.img_pp_train,
            cached_imgs         =self.cfg.cached_imgs,
            n_workers           =self.cfg.n_workers,
            prefetch_factor     =self.cfg.prefetch_factor,
        )

        text_preps_val = get_text_preps(self.cfg.text_preps_type_val)
        self.val_pipe  = ValidationPipeline(
            split_name     =self.cfg.split_name,
            text_preps     =text_preps_val,
            batch_size     =self.cfg.batch_size_val,
            img_pp         =self.modelw.img_pp_val,
            cached_imgs    =self.cfg.cached_imgs,
            n_workers      =self.cfg.n_workers,
            prefetch_factor=self.cfg.prefetch_factor,
        )

        self.set_time_cache(time_cache_train)
        self.save_metadata_trial()
        self.init_opt_and_lr_sched()

    def set_time_cache(self, time_cache_train):
        if time_cache_train is not None:
            time_cache      = time_cache_train + self.val_pipe.time_cache
            self.time_cache = f"{time_cache:.2f}"
        else:
            self.time_cache = None

    def set_paths(self):

        self.dpath_study      = paths["artifacts"] / self.cfg.study_name
        self.dpath_experiment = self.dpath_study / self.cfg.experiment_name

        if self.cfg.seed is not None:
            trial_name       = self.cfg.seed
            self.dpath_trial = self.dpath_experiment / str(trial_name)
            if self.dpath_trial.exists():
                if self.cfg.allow_overwrite_trial:
                    shutil.rmtree(self.dpath_trial)
                else:
                    raise ValueError(f"Trial directory '{self.cfg.study_name}/{self.cfg.experiment_name}/{self.cfg.seed}' already exists!")
        else:
            counter = 0
            while True:
                trial_name       = f"seedless{counter}"
                self.dpath_trial = self.dpath_experiment / trial_name
                if self.dpath_trial.exists():
                    counter += 1
                else:
                    break

        self.dpath_model_best_comp    = self.dpath_trial / "models/best_comp"
        self.dpath_model_best_img2img = self.dpath_trial / "models/best_img2img"
        self.dpath_model_checkpoint   = self.dpath_trial / "models/checkpoint"

    def create_trial_dirs(self):
        for subdir in ("logs", "models", "models/checkpoint", "models/best_comp", "models/best_img2img", "plots"):
            (self.dpath_trial / subdir).mkdir(parents=True)

    def save_metadata_study(self):
        fpath_meta = self.dpath_study / "metadata_study.json"
        metadata   = {
            "split_name":      self.cfg.split_name,
            "n_gpus":          self.cfg.n_gpus,
            "n_cpus":          self.cfg.n_cpus,
            "ram":             f"{self.cfg.ram} GB",
            "n_workers":       self.cfg.n_workers,
            "prefetch_factor": self.cfg.prefetch_factor,
        }
        if fpath_meta.exists() and not self.cfg.allow_diff_study:
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Study params changed!"
        else:
            save_json(metadata, fpath_meta)

    def save_metadata_experiment(self):
        
        def clean_metadata(metadata):
            del metadata["study_name"]
            del metadata["experiment_name"]
            del metadata["seed"]
            del metadata["split_name"]
            del metadata["allow_overwrite_trial"]
            del metadata["allow_diff_study"]
            del metadata["allow_diff_experiment"]
            del metadata["checkpoint_every"]
            del metadata["verbose_batch_loss"]
        
        fpath_meta = self.dpath_experiment / "metadata_experiment.json"
        metadata   = asdict(self.cfg)
        clean_metadata(metadata)
        if fpath_meta.exists() and not self.cfg.allow_diff_experiment:
            metadata_loaded = load_json(fpath_meta)
            assert metadata == metadata_loaded, "Experiment params changed!"
        else:
            save_json(metadata, fpath_meta)

    def save_metadata_trial(self, time_train_avg=None, time_val_avg=None):
        fpath_meta = self.dpath_trial / "metadata_trial.json"
        if time_train_avg is not None:
            time_train_avg = f"{time_train_avg:.2f}"
            time_val_avg   = f"{time_val_avg:.2f}"
        metadata = {
            "runtime_perf":  {"cache": self.time_cache, "train_avg": time_train_avg, "val_avg": time_val_avg},      
            "datetime_init": self.datetime_init,
        }
        save_json(metadata, fpath_meta)

    def save_metadata_model(self, dpath_model, scores_val, idx_epoch_chkpt, idx_epoch):
        fpath_meta = dpath_model / "metadata_model.json"
        scores_val = {k: f"{v:.4f}" for k, v in scores_val.items()}
        metadata   = {
            "scores_val": scores_val,
            "idx_epoch":  f"{idx_epoch_chkpt}/{idx_epoch}",
        }
        save_json(metadata, fpath_meta)

    def init_opt_and_lr_sched(self):

        params_decay, params_no_decay = [], []
        for name, param in self.modelw.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or param.ndim == 1 or "norm" in name.lower():
                params_no_decay.append(param)
            else:
                params_decay.append(param)

        param_groups = [
            {"params": params_decay,    "weight_decay": self.cfg.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups, 
            lr   =self.cfg.lr_init,
            betas=(self.cfg.beta1, self.cfg.beta2),
            eps  =self.cfg.eps,
        )

        lr_sched_type = self.cfg.lr_sched["type"]
        lr_sched_args = self.cfg.lr_sched.get("args")

        if lr_sched_type == "exp":
            self.lr_sched = ExponentialLR(self.optimizer, **lr_sched_args)
        elif lr_sched_type == "plat":
            self.lr_sched = ReduceLROnPlateau(self.optimizer, mode="min", **lr_sched_args)
        elif lr_sched_type == "cos":
            self.lr_sched = CosineAnnealingLR(self.optimizer, **lr_sched_args)
        elif lr_sched_type == "coswr":
            self.lr_sched = CosineAnnealingWarmRestarts(self.optimizer, **lr_sched_args)
        else:
            raise ValueError(f"Unknown lr_sched type: '{lr_sched_type}'")

        if self.cfg.mixed_prec:
            self.scaler = GradScaler()

    def plot_metrics(
            self, 
            data_tracker, 
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

        ax1.plot(x, data["comp"], label="Composite", color="black")
        ax1.plot(x, data["comp_img2img"], label="img2img Composite", color="#B22222")

        ax1.set_ylabel("mAP Composites", fontsize=fontsize_axes, fontweight="bold")
        ax1.set_ylim(0, 1)

        ax1.legend(loc="lower right", fontsize=fontsize_legend)
        ax1.grid(True)
        ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        ##########################################################################################################
        # Plot 3: Precision@1
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

        ax2.plot(x, data["id_img2txt_prec1"], label="ID img2txt Prec@1")
        ax2.plot(x, data["ood_img2txt_prec1"], label="OOD img2txt Prec@1")

        ax2.set_ylabel("Precision@1", fontsize=fontsize_axes, fontweight="bold")
        ax2.set_ylim(0, 1)

        ax2.legend(loc="lower right", fontsize=fontsize_legend)
        ax2.grid(True)
        ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)
        ##########################################################################################################
        # Plot 4: Loss
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

        ax3.plot(x, [np.nan] + data["loss_train"], label="Train Loss")
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

        savepath = self.dpath_trial / "plots" / f"train_metrics.png"
        fig.savefig(savepath)
        plt.close(fig)

    def train(self):
        
        print(
            f"{' Fine-Tuning Init ':#^{75}}",
            f"",
            sep="\n"
        )
        
        data_tracker        = TrialDataTracker(self.dpath_trial)
        scores_val, _, _, _ = self.val_pipe.run_validation(
            self.modelw, 
            verbose           =True, 
            verbose_batch_loss=self.cfg.verbose_batch_loss
        )
        data_tracker.update(scores_val)

        time_train_avg          = 0.0
        time_val_avg            = 0.0
        idx_epoch_best_comp     = 0
        idx_epoch_best_img2img  = 0
        scores_val_best_comp    = {}
        scores_val_best_img2img = {}
        loss_val_best           = np.inf
        for idx_epoch in range(1, self.cfg.n_epochs + 1):
            self.print_epoch_header(idx_epoch)

            time_train_start = time.time()
            self.modelw.model.train()
            loss_train_total = 0.0
            for imgs_b, class_encs_b, texts_b in tqdm(self.dataloader, desc="Train", leave=False):
                imgs_b = imgs_b.to(self.cfg.device, non_blocking=True)

                self.optimizer.zero_grad()

                if self.cfg.mixed_prec:
                    with autocast(device_type=self.cfg.device.type):
                        loss_train_b = self.batch_forward_loss(imgs_b, texts_b, class_encs_b)
                    self.scaler.scale(loss_train_b).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_train_b = self.batch_forward_loss(imgs_b, texts_b, class_encs_b)
                    loss_train_b.backward()
                    self.optimizer.step()

                with torch.no_grad():
                    loss_train_b = loss_train_b.detach().item() * imgs_b.size(0)
                    loss_train_total += loss_train_b
                    if self.cfg.verbose_batch_loss:
                        print(f"Batch Loss: {loss_train_b:.4f}")
            
            # compute avg. train loss per sample
            if self.cfg.drop_partial_batch_train:
                n_full_batches = len(self.dataloader.dataset) // self.cfg.batch_size_train
                loss_train_avg = loss_train_total / (n_full_batches * self.cfg.batch_size_train)
            else:
                loss_train_avg = loss_train_total / len(self.dataloader.dataset)

            time_train_end = time.time()
            time_train     = time_train_end - time_train_start

            # validation
            scores_val, is_best_comp, is_best_img2img, time_val = self.val_pipe.run_validation(
                self.modelw, 
                verbose           =True, 
                verbose_batch_loss=self.cfg.verbose_batch_loss,
            )
            data_tracker.update(scores_val, lr=self.lr, loss_train=loss_train_avg)

            if scores_val["comp_loss"] < loss_val_best:
                loss_val_best = scores_val["comp_loss"]

            self.step_lr_scheduler(scores_val)

            print(
                f"Train Loss ------ {loss_train_avg:.4f}",
                f"Val Loss -------- {scores_val['comp_loss']:.4f} (Best: {loss_val_best:.4f})",
                f"",
                sep="\n"
            )

            # track running means via Welford's algorithm
            time_train_avg += (time_train - time_train_avg) / idx_epoch
            time_val_avg += (time_val - time_val_avg) / idx_epoch

            if is_best_comp:
                self.modelw.save(self.dpath_model_best_comp)
                idx_epoch_best_comp  = idx_epoch
                scores_val_best_comp = scores_val
                print(f"~ Best comp model saved to file ~\n")
            if is_best_img2img:
                self.modelw.save(self.dpath_model_best_img2img)
                idx_epoch_best_img2img  = idx_epoch
                scores_val_best_img2img = scores_val
                print(f"~ Best img2img model saved to file ~\n")
            if idx_epoch % self.cfg.checkpoint_every == 0 or is_best_comp or is_best_img2img:
                self.modelw.save(self.dpath_model_checkpoint)
                self.save_metadata_model(self.dpath_model_checkpoint, scores_val, idx_epoch, idx_epoch)
                self.save_metadata_model(self.dpath_model_best_comp, scores_val_best_comp, idx_epoch_best_comp, idx_epoch)
                self.save_metadata_model(self.dpath_model_best_img2img, scores_val_best_img2img, idx_epoch_best_img2img, idx_epoch)
                self.save_metadata_trial(time_train_avg=time_train_avg, time_val_avg=time_val_avg)
                data_tracker.save()
                self.plot_metrics(data_tracker)

            print(
                f"{' Elapsed Time ':=^{75}}",
                f"Train -------- {time_train:.2f} s (avg: {time_train_avg:.2f} s)",
                f"Validation --- {time_val:.2f} s (avg: {time_val_avg:.2f} s)",
                f"",
                sep="\n"
            )

    def print_epoch_header(self, idx_epoch):
        header_epoch = f" Epoch {idx_epoch} "
        print(
            f"{header_epoch:#^{75}}{'' if self.cfg.experiment_name is None else ' (' + self.cfg.rdpath_trial + ')'}",
            f"",
            f"{' Train ':=^{75}}",
            f"LR ----- {self.lr:.2e}",
            sep="\n"
        )

    def step_lr_scheduler(self, scores_val):

        lr_prev = self.optimizer.param_groups[0]["lr"]

        lr_sched_type = self.cfg.lr_sched["type"]
        if lr_sched_type in ("exp", "cos", "coswr"):
            self.lr_sched.step()
        elif lr_sched_type == "plat":
            self.lr_sched.step(scores_val["comp_loss"])
        else:
            raise ValueError(f"Unknown lr_sched_type: '{lr_sched_type}'")

        self.lr = self.optimizer.param_groups[0]["lr"]

        if lr_sched_type == "plat" and self.lr < lr_prev:
            self.lr_sched.best = scores_val["comp_loss"]

    def batch_forward_loss(self, imgs_b, texts_b, class_encs_b):

        embs_imgs = self.modelw.embed_images(imgs_b)  # ------- Tensor(B, D)
        embs_txts = self.modelw.embed_texts(texts_b)  # ------- Tensor(B, D)

        sim          = embs_imgs @ embs_txts.T  # ------------- Tensor(B, B)
        logits       = self.modelw.compute_logits(sim)
        loss_train_b = self.modelw.compute_loss(logits, class_encs_b)

        return loss_train_b

def main():

    config_train = get_config_train()
    seed_libs(config_train.seed)

    modelw = VLMWrapper.build(config_train)

    train_pipe = TrainPipeline(modelw, config_train)
    train_pipe.train()

if __name__ == "__main__":
    main()
