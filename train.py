import torch  # type: ignore[import]
from torch.amp import autocast, GradScaler  # type: ignore[import]
from torch.optim.lr_scheduler import (  # type: ignore[import]
    ExponentialLR, 
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    LambdaLR,
)
import torch.distributed as dist  # type: ignore[import]
from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore[import]
from torch.utils.data.distributed import DistributedSampler  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]
import time
from datetime import datetime, timezone
from dataclasses import asdict
import shutil
import numpy as np  # type: ignore[import]
import math

from utils import (
    paths, 
    save_json, 
    load_json, 
    save_pickle, 
    seed_libs,
    get_text_preps, 
    RunningMean,
)
from models import VLMWrapper
from utils_data import spawn_dataloader, spawn_indexes
from utils_eval import ValidationPipeline
from utils_config import get_config_train
from utils_train import plot_metrics
from utils_ddp import setup_ddp

import pdb

torch.set_printoptions(
    precision=4,
    sci_mode =False,
    threshold=1000,  # total elements before summarizing
    edgeitems=3,  # num items to show at the start/end of each dim
    linewidth=120
)


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
            "comp_map":          [],
            "img2img_map":       [],
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

class LRSchedulerWrapper:

    def __init__(self, optimizer, config):

        self.opt  = optimizer
        self.type = config.opt['lr']['sched']

        args       = config.lr_sched_params.get("args")
        args_sched = config.lr_sched_params.get("args_sched")

        if self.type == "exp":
            self.lr_min = args["lr_min"]
            self.sched  = ExponentialLR(self.opt, **args_sched)
        elif self.type == "plat":
            self.valid_type       = args["valid_type"]
            self.reset_best_valid = args["reset_best_valid"]
            if self.valid_type == "loss":
                self.sched = ReduceLROnPlateau(self.opt, mode="min", **args_sched)
            elif self.valid_type == "perf":
                self.sched = ReduceLROnPlateau(self.opt, mode="max", **args_sched)
        elif self.type == "cos":
            self.sched = CosineAnnealingLR(self.opt, **args_sched)
        elif self.type == "coswr":
            self.sched = CosineAnnealingWarmRestarts(self.opt, **args_sched)
        elif self.type == "cosXexp":
            self.sched = CosineExponentialLR(self.opt, **args)
        elif self.type == "coswrXexp":
            self.sched = CosineWRExponentialLR(self.opt, **args)

    def step(self, scores_val):

        if self.type == "plat":
            if self.valid_type == "loss":
                valid_signal = scores_val["comp_loss"]
            elif self.valid_type == "perf":
                valid_signal = scores_val["comp_map"]
            lr_prev = self.get_lr()
            self.sched.step(valid_signal)
            if self.get_lr() < lr_prev:  # if current LR < previous LR
                if self.reset_best_valid:
                    self.sched.best = valid_signal
        else:
            self.sched.step()
            if self.type == "exp" and self.get_lr() < self.lr_min:
                for pg in self.opt.param_groups:
                    pg["lr"] = self.lr_min

    def get_lr(self):
        return self.opt.param_groups[0]["lr"]

class CosineExponentialLR(LambdaLR):

    def __init__(self, optimizer, gamma, period, peak_ratio, lr_nom_min):

        self.lr_init     = optimizer.param_groups[0]["lr"]
        self.gamma       = gamma
        self.peak_ratio  = peak_ratio
        self.period      = period
        self.lr_nom_min  = lr_nom_min

        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, idx_epoch):

        peak = self.lr_init * (self.gamma ** idx_epoch)
        if peak < self.lr_nom_min:
            peak = self.lr_nom_min
        valley     = peak / self.peak_ratio
        cos_factor = 0.5 * (1 + math.cos(2 * math.pi * (idx_epoch / self.period)))

        lr = valley + (peak - valley) * cos_factor

        return lr / self.lr_init

class CosineWRExponentialLR(LambdaLR):

    def __init__(self, optimizer, gamma, period, peak_ratio, lr_nom_min):

        self.lr_init    = optimizer.param_groups[0]["lr"]
        self.gamma      = gamma
        self.peak_ratio = peak_ratio
        self.period     = period
        self.lr_nom_min = lr_nom_min

        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, idx_epoch):

        peak = self.lr_init * (self.gamma ** idx_epoch)
        if peak < self.lr_nom_min:
            peak = self.lr_nom_min
        valley     = peak / self.peak_ratio
        cos_factor = 0.5 * (1 + math.cos(math.pi * ((idx_epoch % self.period) / self.period)))

        lr = valley + (peak - valley) * cos_factor

        return lr / self.lr_init

class TrainPipeline:

    def __init__(
        self, 
        modelw, 
        config_train,
        rank:       int = 0,
        world_size: int = 1,
    ):
        self.datetime_init = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC"

        self.modelw = modelw
        self.cfg    = config_train

        self.rank       = rank  # GPU rank
        self.world_size = world_size

        self.modelw.freeze(self.cfg.freeze["text"], self.cfg.freeze["image"])

        self.set_paths()
        if self.rank == 0:
            self.create_trial_dirs()
            self.save_metadata_study()
            self.save_metadata_experiment()

        index_data, _ = spawn_indexes(
            split_name   =self.cfg.split_name,
            splitset_name="train",
        )

        text_preps_train                  = get_text_preps(self.cfg.text_preps["train"])
        self.dataloader, time_cache_train = spawn_dataloader(
            index_data    =index_data,
            text_preps    =text_preps_train,
            config        =self.cfg,
            shuffle       =True,
            drop_last     =True,
            img_pp        =self.modelw.img_pp_train,
            use_dv_sampler=self.cfg.dv_batching,
        )

        text_preps_val = get_text_preps(self.cfg.text_preps["valid"])
        self.val_pipe  = ValidationPipeline(self.cfg, text_preps_val, self.modelw.img_pp_val)

        self.set_time_cache(time_cache_train)
        if self.rank == 0: self.save_metadata_trial()
        self.init_opt_and_lr_sched()

        self.lr_warmup   = self.cfg.opt["lr"]["warmup"]
        self.samps_seen  = 0
        self.lr_init_nom = self.cfg.opt["lr"]["init"]

    def set_time_cache(self, time_cache_train):
        if time_cache_train is not None:
            time_cache      = time_cache_train + self.val_pipe.time_cache
            self.time_cache = f"{time_cache:.2f}"
        else:
            self.time_cache = None

    def set_paths(self):

        self.dpath_study      = paths["artifacts"] / self.cfg.study_name
        self.dpath_experiment = self.dpath_study / self.cfg.experiment_name

        trial_name       = self.cfg.seed
        self.dpath_trial = self.dpath_experiment / str(trial_name)

        if self.dpath_trial.exists() and self.rank == 0:
            if self.cfg.dev['allow_overwrite_trial']:
                shutil.rmtree(self.dpath_trial)
            else:
                raise ValueError(f"Trial directory '{self.cfg.study_name}/{self.cfg.experiment_name}/{self.cfg.seed}' already exists!")

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
        if fpath_meta.exists() and not self.cfg.dev['allow_diff_study']:
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

            del metadata["dev"]

            del metadata["chkpt_every"]
            
            del metadata["loss"]["wting"]
            del metadata["loss"]["focal"]

            if metadata["loss2"]["mix"] == 0.0:
                del metadata["loss2"]
        
        fpath_meta = self.dpath_experiment / "metadata_experiment.json"
        metadata   = asdict(self.cfg)

        # save full text combo-templates themselves and not just the names
        text_preps_full = {}
        for split_name, text_preps in metadata["text_preps"].items():
            text_preps_full[split_name] = get_text_preps(text_preps)
        metadata["text_preps"] = text_preps_full

        clean_metadata(metadata)
        if fpath_meta.exists() and not self.cfg.dev['allow_diff_experiment']:
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
            # decoupled weight decay ~ decoupling biases, affine params, & logit scale/shift (temp/bias)
            if name.endswith(".bias") or param.ndim == 1 or "norm" in name.lower():
                params_no_decay.append(param)
            else:
                params_decay.append(param)

        lr_init_nom = self.cfg.opt["lr"]["init"]
        param_groups = [
            {"params": params_decay,    "weight_decay": self.cfg.opt["l2reg"], "lr": lr_init_nom},
            {"params": params_no_decay, "weight_decay": 0.0,                   "lr": lr_init_nom},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups, 
            lr   =lr_init_nom,
            betas=(self.cfg.opt["beta1"], self.cfg.opt["beta2"]),
            eps  =self.cfg.opt["eps"],
        )

        self.lr_schedw = LRSchedulerWrapper(
            self.optimizer, 
            self.cfg,
        )

        if self.cfg.hw.mixed_prec:
            self.scaler = GradScaler()

    def _update_lr_warmup(self) -> float:
        if self.lr_warmup == 0 or self.samps_seen >= self.lr_warmup:
            lr = self.optimizer.param_groups[0]["lr"]
        else:
            frac = self.samps_seen / self.lr_warmup
            lr = self.lr_init_nom * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        return lr

    def _broadcast_obj(self, obj):
        """
        DDP ~ broadcast python object from rank 0 to all ranks
        """
        if self.world_size == 1:
            return obj
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    def train(self):
        
        if self.rank == 0:  # if GPU:0
            print(
                f"{' Fine-Tuning Init ':#^{75}}",
                f"",
                sep="\n"
            )
            data_tracker = TrialDataTracker(self.dpath_trial)

        scores_val, _, _, _ = self.val_pipe.run_validation(
            self.modelw, 
            verbose           =(self.rank == 0), 
            verbose_batch_loss=self.cfg.dev['verbose_batch_loss'],
        )

        if self.rank == 0: data_tracker.update(scores_val)

        time_train_avg          = 0.0
        time_val_avg            = 0.0
        idx_epoch_best_comp     = 0
        idx_epoch_best_img2img  = 0
        scores_val_best_comp    = {}
        scores_val_best_img2img = {}
        loss_val_best           = np.inf
        for idx_epoch in range(1, self.cfg.n_epochs + 1):
            if self.rank == 0: self.print_epoch_header(idx_epoch)

            # Let samplers know current epoch (crucial for shuffling)
            sampler = getattr(self.dataloader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(idx_epoch)
            batch_sampler = getattr(self.dataloader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(idx_epoch)

            time_train_start = time.time()
            self.modelw.model.train()

            lr_mean   = RunningMean()
            loss_mean = RunningMean()

            for imgs_b, texts_b, class_encs_b, targ_data_b in tqdm(self.dataloader, desc="Train", leave=False, disable=(self.rank != 0)):

                imgs_b       = imgs_b.to(self.cfg.device, non_blocking=True)
                class_encs_b = class_encs_b.to(self.cfg.device)
                B            = imgs_b.size(0)
                self.samps_seen += B

                if self.lr_warmup > 0:
                    lr = self._update_lr_warmup()
                else:
                    lr = self.optimizer.param_groups[0]["lr"]

                lr_mean.update(lr)

                self.optimizer.zero_grad()

                if self.cfg.hw.mixed_prec:
                    with autocast(device_type=self.cfg.device.type):
                        loss_batch, _ = self.modelw.batch_step(imgs_b, texts_b, class_encs_b, targ_data_b)
                    self.scaler.scale(loss_batch).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_batch, _ = self.modelw.batch_step(imgs_b, texts_b, class_encs_b, targ_data_b)
                    loss_batch.backward()
                    self.optimizer.step()

                with torch.no_grad():
                    loss_batch = loss_batch.detach().item()
                    loss_mean.update(loss_batch)
                    if self.cfg.dev['verbose_batch_loss']:
                        print(f"Batch Loss: {loss_batch:.4f}")

            loss_train_avg = loss_mean.value()  # should be average across all devices, not just rank 0

            time_train_end = time.time()
            time_train     = time_train_end - time_train_start

            scores_val, is_best_comp, is_best_img2img, time_val = self.val_pipe.run_validation(
                self.modelw, 
                verbose           =(self.rank == 0), 
                verbose_batch_loss=self.cfg.dev['verbose_batch_loss'],
            )

            # broadcast results from rank 0 to all ranks
            scores_val      = self._broadcast_obj(scores_val)
            is_best_comp    = self._broadcast_obj(is_best_comp)
            is_best_img2img = self._broadcast_obj(is_best_img2img)
            time_val        = self._broadcast_obj(time_val)

            if self.rank == 0: data_tracker.update(scores_val, lr=lr_mean.value(), loss_train=loss_train_avg)

            if scores_val["comp_loss"] < loss_val_best:
                loss_val_best = scores_val["comp_loss"]

            self.lr_schedw.step(scores_val)

            if self.rank == 0:
                print(
                    f"Train Loss ------ {loss_train_avg:.4f}",
                    f"Val Loss -------- {scores_val['comp_loss']:.4f} (Best: {loss_val_best:.4f})",
                    f"",
                    sep="\n"
                )

            # track running means via Welford's algorithm
            time_train_avg += (time_train - time_train_avg) / idx_epoch
            time_val_avg += (time_val - time_val_avg) / idx_epoch

            if self.rank == 0:
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
                if idx_epoch % self.cfg.chkpt_every == 0 or is_best_comp or is_best_img2img:
                    self.modelw.save(self.dpath_model_checkpoint)
                    self.save_metadata_model(self.dpath_model_checkpoint, scores_val, idx_epoch, idx_epoch)
                    self.save_metadata_model(self.dpath_model_best_comp, scores_val_best_comp, idx_epoch_best_comp, idx_epoch)
                    self.save_metadata_model(self.dpath_model_best_img2img, scores_val_best_img2img, idx_epoch_best_img2img, idx_epoch)
                    self.save_metadata_trial(time_train_avg=time_train_avg, time_val_avg=time_val_avg)
                    data_tracker.save()
                    plot_metrics(data_tracker, self.dpath_trial)

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
            f"LR ----- {self.lr_schedw.get_lr():.2e}",
            sep="\n"
        )

def main():
    rank, world_size, local_rank, device = setup_ddp()

    config_train        = get_config_train(verbose=False)
    config_train.device = device  # set local device
    seed_libs(config_train.seed)

    if rank == 0: config_train.print_init_info()

    modelw = VLMWrapper.build(config_train, verbose=(rank == 0))
    modelw.set_class_wts(config_train)
    if config_train.loss2["mix"] != 0.0:
        modelw.set_class_wts(config_train, secondary=True)

    modelw.model = DDP(modelw.model, device_ids=[local_rank], output_device=local_rank)
    train_pipe   = TrainPipeline(modelw, config_train, rank=rank, world_size=world_size)
    train_pipe.train()

    dist.destroy_process_group()  # DDP cleanup

if __name__ == "__main__":
    main()