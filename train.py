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
import math

from utils.utils import (
    save_pickle, 
    seed_libs,
    get_text_template, 
    RunningMean,
    PrintLog,
)
from models import VLMWrapper
from utils.data import spawn_dataloader, spawn_partition_data
from utils.eval import ValidationPipeline
from utils.config import get_config_train
from utils.train import ArtifactManager, plot_metrics
from utils.ddp import setup_ddp, cleanup_ddp

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

        self.data_epoch = {
            "samps_seen": [],
            "lr": [],
            "loss_train": [],
            "loss_raw_train": [],
        }
        self.data_eval = {
            "samps_seen": [],
            "idx_epoch": [],
            "comp": {
                "standard": {
                    "map": {},
                },
                "per_class": {
                    "map": {},
                },
            },
        }
        self.data = {
            "epoch": self.data_epoch,
            "eval": self.data_eval,
        }

    def update_epoch(self, samps_seen, lr=None, loss_train=None, loss_raw_train=None):

        self.data_epoch["samps_seen"].append(samps_seen)

        if lr is not None:
            self.data_epoch["lr"].append(lr)
        if loss_train is not None:
            self.data_epoch["loss_train"].append(loss_train)
        if loss_raw_train is not None:
            self.data_epoch["loss_raw_train"].append(loss_raw_train)

    def update_eval(self, scores_val, samps_seen, idx_epoch):

        def append_nested(dst, src):
            for score_name, score_value in src.items():
                if isinstance(score_value, dict):
                    if score_name not in dst:
                        dst[score_name] = {}
                    append_nested(dst[score_name], score_value)
                else:
                    if score_name not in dst:
                        dst[score_name] = []
                    dst[score_name].append(float(score_value))

        self.data_eval["samps_seen"].append(samps_seen)
        self.data_eval["idx_epoch"].append(idx_epoch)

        for partition_name, partition_scores in scores_val.items():
            if partition_name not in self.data_eval:
                self.data_eval[partition_name] = {}
            append_nested(self.data_eval[partition_name], partition_scores)

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
            self.reset_best_valid = args["reset_best_valid"]
            self.sched = ReduceLROnPlateau(self.opt, mode="max", **args_sched)
        elif self.type == "cos":
            n_epochs = getattr(config, "n_epochs")
            eta_min_factor = args.get("eta_min_factor")
            lr_init = self.opt.param_groups[0]["lr"]
            eta_min = lr_init * float(eta_min_factor)
            self.sched = CosineAnnealingLR(self.opt, T_max=n_epochs, eta_min=eta_min)
        elif self.type == "coswr":
            self.sched = CosineAnnealingWarmRestarts(self.opt, **args_sched)
        elif self.type == "cosXexp":
            self.sched = CosineExponentialLR(self.opt, **args)
        elif self.type == "coswrXexp":
            self.sched = CosineWRExponentialLR(self.opt, **args)

    def step(self, scores_val):

        if self.type == "plat":
            valid_signal = scores_val["comp"]["standard"]["map"]["all"]
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

        self.lr_init = optimizer.param_groups[0]["lr"]
        self.gamma = gamma
        self.peak_ratio = peak_ratio
        self.period = period
        self.lr_nom_min = lr_nom_min

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
        config,
        gpu_rank: int = 0,
        gpu_world_size: int = 1,
    ):

        self.modelw = modelw
        self.cfg = config

        self.gpu_rank = gpu_rank
        self.gpu_world_size = gpu_world_size

        self.modelw.freeze(self.cfg.freeze["text"], self.cfg.freeze["image"])

        index_data, _ = spawn_partition_data(config=self.cfg, partition_name="train")
        text_template_train = get_text_template(self.cfg.text_template["train"], dataset=self.cfg.dataset)
        self.dataloader, _ = spawn_dataloader(
            index_data=index_data,
            text_template=text_template_train,
            config=self.cfg,
            shuffle=True,
            drop_last=True,
            img_pp=self.modelw.img_pp_train,
            use_dv_sampler=self.cfg.dv_batching,
            persistent_workers=self.cfg.hw.persistent_workers_train,
        )

        text_template_val = get_text_template(self.cfg.text_template["valid"], dataset=self.cfg.dataset)
        self.val_pipe = ValidationPipeline(self.cfg, text_template_val, self.modelw.img_pp_val)

        if self.gpu_rank == 0:
            ArtifactManager.save_metadata_trial()
        self.init_opt_and_lr_sched()

        self.lr_warmup = self.cfg.opt["lr"]["warmup"]
        self.n_samps_seen = 0
        self.next_eval_threshold = self.cfg.eval_every
        self.lr_init_nom = self.cfg.opt["lr"]["init"]
        self.logged_train_text = False

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
        if self.lr_warmup == 0 or self.n_samps_seen >= self.lr_warmup:
            lr = self.optimizer.param_groups[0]["lr"]
        else:
            frac = self.n_samps_seen / self.lr_warmup
            lr = self.lr_init_nom * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        return lr

    def _broadcast_obj(self, obj):
        """
        DDP ~ broadcast python object from GPU rank 0 to all ranks
        """
        if self.gpu_world_size == 1:
            return obj
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    def train(self):
        try:
            scores_val, _, _, time_val_base = self.val_pipe.run_validation(self.modelw)
            scores_val = self._broadcast_obj(scores_val)
            time_val_base = self._broadcast_obj(time_val_base)

            if self.gpu_rank == 0:
                data_tracker = TrialDataTracker(ArtifactManager.dpath_trial)
                data_tracker.update_eval(scores_val, samps_seen=0, idx_epoch=0)
                PrintLog.eval(
                    scores_val,
                    self.val_pipe,
                    header="Base",
                    samps_seen=0,
                    idx_epoch=0,
                    time_val=time_val_base,
                    log_to="eval",
                )
                PrintLog.texts_eval(self.val_pipe.get_eval_texts())

                time_train_mean = RunningMean()
                time_val_mean = RunningMean()
                time_val_mean.update(time_val_base)

            samps_seen_best_comp = 0
            samps_seen_best_i2i = 0
            scores_val_best_comp = scores_val
            scores_val_best_i2i = scores_val
            scores_val_latest = scores_val

            for idx_epoch in range(1, self.cfg.n_epochs + 1):

                if self.gpu_rank == 0:
                    PrintLog.epoch_header(idx_epoch, self.cfg.n_epochs)
                    lr = self.lr_schedw.get_lr()
                    PrintLog.train_header(lr)

                # Let samplers know current epoch (crucial for shuffling)
                sampler = getattr(self.dataloader, "sampler", None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(idx_epoch)
                batch_sampler = getattr(self.dataloader, "batch_sampler", None)
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(idx_epoch)

                time_train_start = time.time()
                self.modelw.model.train()

                lr_mean = RunningMean()
                loss_mean = RunningMean()
                loss_raw_mean = RunningMean()

                for idx_batch, data_sb in enumerate(tqdm(self.dataloader, desc="Train", leave=False, disable=(self.gpu_rank != 0))):
                    imgs_sb, texts_sb, class_encs_sb, targ_data_sb = data_sb

                    if self.gpu_rank == 0 and idx_epoch == 1 and idx_batch == 0 and not self.logged_train_text:
                        PrintLog.texts(texts_sb)
                        self.logged_train_text = True

                    imgs_sb = imgs_sb.to(self.cfg.device, non_blocking=True)
                    class_encs_sb = class_encs_sb.to(self.cfg.device)
                    B = imgs_sb.size(0) * self.gpu_world_size
                    self.n_samps_seen += B

                    if self.lr_warmup > 0:
                        lr = self._update_lr_warmup()
                    else:
                        lr = self.optimizer.param_groups[0]["lr"]
                    lr_mean.update(lr)

                    self.optimizer.zero_grad(set_to_none=True)

                    if self.cfg.hw.mixed_prec:
                        with autocast(device_type=self.cfg.device.type):
                            loss, loss_raw, embs_img_b, embs_txt_b, logits, _ = self.modelw.batch_step(
                                imgs_sb, texts_sb, class_encs_sb, targ_data_sb
                            )
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        if self.gpu_rank == 0:
                            PrintLog.batch(idx_batch, lr, loss, embs_img_b, embs_txt_b, logits, self.modelw.model)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss, loss_raw, embs_img_b, embs_txt_b, logits, _ = self.modelw.batch_step(
                            imgs_sb, texts_sb, class_encs_sb, targ_data_sb
                        )
                        loss.backward()
                        if self.gpu_rank == 0:
                            PrintLog.batch(idx_batch, lr, loss, embs_img_b, embs_txt_b, logits, self.modelw.model)
                        self.optimizer.step()

                    with torch.no_grad():
                        loss = loss.detach().item()
                        loss_raw = loss_raw.detach().item()
                        loss_mean.update(loss)
                        loss_raw_mean.update(loss_raw)

                    while self.n_samps_seen >= self.next_eval_threshold:
                        threshold_hit = self.next_eval_threshold
                        self.next_eval_threshold += self.cfg.eval_every

                        scores_val, is_best_comp, is_best_i2i, time_val = self.val_pipe.run_validation(self.modelw)

                        # broadcast results from GPU rank 0 to all ranks
                        scores_val = self._broadcast_obj(scores_val)
                        is_best_comp = self._broadcast_obj(is_best_comp)
                        is_best_i2i = self._broadcast_obj(is_best_i2i)
                        time_val = self._broadcast_obj(time_val)
                        scores_val_latest = scores_val

                        if self.gpu_rank == 0:
                            time_val_mean.update(time_val)
                            data_tracker.update_eval(scores_val, samps_seen=self.n_samps_seen, idx_epoch=idx_epoch)
                            PrintLog.eval(
                                scores_val,
                                self.val_pipe,
                                header=f"{threshold_hit:,}",
                                samps_seen=self.n_samps_seen,
                                idx_epoch=idx_epoch,
                                time_val=time_val,
                                log_to="eval",
                            )

                            if is_best_comp:
                                self.modelw.save(ArtifactManager.dpath_model_best_comp)
                                samps_seen_best_comp = self.n_samps_seen
                                scores_val_best_comp = scores_val
                            if is_best_i2i:
                                self.modelw.save(ArtifactManager.dpath_model_best_i2i)
                                samps_seen_best_i2i = self.n_samps_seen
                                scores_val_best_i2i = scores_val
                            if is_best_comp or is_best_i2i:
                                self.modelw.save(ArtifactManager.dpath_model_checkpoint)
                                ArtifactManager.save_metadata_model(
                                    ArtifactManager.dpath_model_checkpoint,
                                    scores_val,
                                    self.n_samps_seen,
                                    self.n_samps_seen,
                                )
                                ArtifactManager.save_metadata_model(
                                    ArtifactManager.dpath_model_best_comp,
                                    scores_val_best_comp,
                                    samps_seen_best_comp,
                                    self.n_samps_seen,
                                )
                                ArtifactManager.save_metadata_model(
                                    ArtifactManager.dpath_model_best_i2i,
                                    scores_val_best_i2i,
                                    samps_seen_best_i2i,
                                    self.n_samps_seen,
                                )
                                ArtifactManager.save_metadata_trial(
                                    time_train_mean=time_train_mean.value(),
                                    time_val_mean=time_val_mean.value(),
                                )
                                data_tracker.save()
                                plot_metrics(data_tracker, ArtifactManager.dpath_trial)

                loss_train_avg = loss_mean.value()

                time_train_end = time.time()
                time_train = time_train_end - time_train_start

                self.lr_schedw.step(scores_val_latest)

                if self.gpu_rank == 0:

                    time_train_mean.update(time_train)
                    data_tracker.update_epoch(
                        samps_seen=self.n_samps_seen,
                        lr=lr_mean.value(),
                        loss_train=loss_mean.value(),
                        loss_raw_train=loss_raw_mean.value(),
                    )

                    if idx_epoch % self.cfg.chkpt_every == 0:
                        self.modelw.save(ArtifactManager.dpath_model_checkpoint)
                        ArtifactManager.save_metadata_model(
                            ArtifactManager.dpath_model_checkpoint,
                            scores_val_latest,
                            self.n_samps_seen,
                            self.n_samps_seen,
                        )
                        ArtifactManager.save_metadata_model(
                            ArtifactManager.dpath_model_best_comp, 
                            scores_val_best_comp, 
                            samps_seen_best_comp,
                            self.n_samps_seen,
                        )
                        ArtifactManager.save_metadata_model(
                            ArtifactManager.dpath_model_best_i2i, 
                            scores_val_best_i2i, 
                            samps_seen_best_i2i,
                            self.n_samps_seen,
                        )
                        ArtifactManager.save_metadata_trial(time_train_mean=time_train_mean.value(), time_val_mean=time_val_mean.value())
                        data_tracker.save()
                        plot_metrics(data_tracker, ArtifactManager.dpath_trial)

                    PrintLog.epoch(
                        time_train,
                        time_train_mean.value(),
                        loss_train_avg,
                        loss_raw_mean.value(),
                        self.n_samps_seen,
                    )

            scores_val, is_best_comp, is_best_i2i, time_val = self.val_pipe.run_validation(self.modelw)

            # broadcast results from GPU rank 0 to all ranks
            scores_val = self._broadcast_obj(scores_val)
            is_best_comp = self._broadcast_obj(is_best_comp)
            is_best_i2i = self._broadcast_obj(is_best_i2i)
            time_val = self._broadcast_obj(time_val)

            if self.gpu_rank == 0:
                time_val_mean.update(time_val)
                data_tracker.update_eval(scores_val, samps_seen=self.n_samps_seen, idx_epoch=self.cfg.n_epochs)
                PrintLog.eval(
                    scores_val,
                    self.val_pipe,
                    header="Final",
                    samps_seen=self.n_samps_seen,
                    idx_epoch=self.cfg.n_epochs,
                    time_val=time_val,
                    log_to="eval",
                )

                if is_best_comp:
                    self.modelw.save(ArtifactManager.dpath_model_best_comp)
                    samps_seen_best_comp = self.n_samps_seen
                    scores_val_best_comp = scores_val
                if is_best_i2i:
                    self.modelw.save(ArtifactManager.dpath_model_best_i2i)
                    samps_seen_best_i2i = self.n_samps_seen
                    scores_val_best_i2i = scores_val

                self.modelw.save(ArtifactManager.dpath_model_checkpoint)
                ArtifactManager.save_metadata_model(
                    ArtifactManager.dpath_model_checkpoint,
                    scores_val,
                    self.n_samps_seen,
                    self.n_samps_seen,
                )
                ArtifactManager.save_metadata_model(
                    ArtifactManager.dpath_model_best_comp,
                    scores_val_best_comp,
                    samps_seen_best_comp,
                    self.n_samps_seen,
                )
                ArtifactManager.save_metadata_model(
                    ArtifactManager.dpath_model_best_i2i,
                    scores_val_best_i2i,
                    samps_seen_best_i2i,
                    self.n_samps_seen,
                )
                ArtifactManager.save_metadata_trial(
                    time_train_mean=time_train_mean.value(),
                    time_val_mean=time_val_mean.value(),
                )
                data_tracker.save()
                plot_metrics(data_tracker, ArtifactManager.dpath_trial)

        finally:
            if self.gpu_rank == 0:
                PrintLog.close_logs()

def main():
    gpu_rank, gpu_world_size, local_gpu_rank, device = setup_ddp()

    cfg = get_config_train()
    cfg.device = device  # set local device
    seed_libs(cfg.seed)

    if gpu_rank == 0:
        ArtifactManager.set_paths(cfg)
        ArtifactManager.create_trial_dirs()
        ArtifactManager.save_metadata_study(cfg)
        ArtifactManager.save_metadata_experiment(cfg)
        if cfg.logging:
            PrintLog.create_logs(ArtifactManager.dpath_trial / "logs")
        PrintLog.init_train(cfg)

    modelw = VLMWrapper.build(cfg, verbose=(gpu_rank == 0))
    modelw.set_class_wts(cfg)
    if cfg.loss2["mix"] != 0.0:
        modelw.set_class_wts(cfg, secondary=True)
    modelw.model = DDP(modelw.model, device_ids=[local_gpu_rank], output_device=local_gpu_rank)

    train_pipe = TrainPipeline(modelw, cfg, gpu_rank=gpu_rank, gpu_world_size=gpu_world_size)
    train_pipe.train()

    cleanup_ddp()


if __name__ == "__main__":
    main()