import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time
import math

from utils.utils import (
    save_pickle, 
    seed_libs,
    get_text_template, 
    RunningMean,
    PrintLog,
    model_grad_l2_norm,
)
from models import VLMWrapper
from utils.data import spawn_dataloader, spawn_partition_data
from utils.eval import ValidationPipeline
from utils.config import get_config_train
from utils.train import ArtifactManager, plot_metrics
from utils.train import TrainImageDumper
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
            "grad_norm_model": [],
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

    def update_train_batch(self, samps_seen, lr=None, loss_train=None, loss_raw_train=None, grad_norm_model=None):

        self.data_epoch["samps_seen"].append(samps_seen)

        if lr is not None:
            self.data_epoch["lr"].append(lr)
        if loss_train is not None:
            self.data_epoch["loss_train"].append(loss_train)
        if loss_raw_train is not None:
            self.data_epoch["loss_raw_train"].append(loss_raw_train)
        if grad_norm_model is not None:
            self.data_epoch["grad_norm_model"].append(grad_norm_model)

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

    def __init__(self, optimizer, config, total_steps):

        self.opt  = optimizer
        self.type = config.opt['lr']['sched']

        args       = config.lr_sched_params.get("args")
        if self.type != "cos":
            raise ValueError(f"Unsupported LR scheduler type: '{self.type}', expected 'cos'")

        eta_min_factor = args.get("eta_min_factor")
        lr_init = self.opt.param_groups[0]["lr"]
        eta_min = lr_init * float(eta_min_factor)
        self.sched = CosineAnnealingLR(self.opt, T_max=max(1, int(total_steps)), eta_min=eta_min)

    def step(self):
        self.sched.step()

    def get_lr(self):
        return self.opt.param_groups[0]["lr"]

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
        self.dataloader = spawn_dataloader(
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
        self.val_pipe = ValidationPipeline(
            self.cfg,
            text_template_val,
            self.modelw.img_pp_inf,
            compute_loss=True,
        )
        self.val_pipe_base = ValidationPipeline(
            self.cfg,
            text_template_val,
            self.modelw.img_pp_inf,
            compute_loss=False,
        )

        if self.gpu_rank == 0:
            ArtifactManager.save_metadata_trial()

        self.lr_warmup = self.cfg.opt["lr"]["warmup"]
        self.init_opt_and_lr_sched()
        self.n_samps_seen = 0
        self.n_batches_seen = 0
        self.next_eval_threshold = self.cfg.eval_every
        self.lr_init_nom = self.cfg.opt["lr"]["init"]
        self.logged_train_text = False
        self.train_img_dumper = TrainImageDumper(
            cfg=self.cfg,
            gpu_rank=self.gpu_rank,
            gpu_world_size=self.gpu_world_size,
            img_pp_train=self.modelw.img_pp_train,
        )

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
            total_steps=max(1, math.ceil(self.cfg.sample_volume / (self.cfg.batch_size)) - math.ceil(self.lr_warmup / (self.cfg.batch_size))),
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
            scores_val, _, _, time_val_base = self.val_pipe_base.run_validation(self.modelw)
            scores_val = self._broadcast_obj(scores_val)
            time_val_base = self._broadcast_obj(time_val_base)

            if self.gpu_rank == 0:
                data_tracker = TrialDataTracker(ArtifactManager.dpath_trial)
                data_tracker.update_eval(scores_val, samps_seen=0, idx_epoch=0)
                PrintLog.eval(
                    scores_val,
                    self.val_pipe_base,
                    header="Base",
                    samps_seen=0,
                    idx_epoch=0,
                    time_val=time_val_base,
                    log_to="eval",
                )
                PrintLog.texts_eval(self.val_pipe_base.get_eval_texts())

                time_train_mean = RunningMean()
                time_val_mean = RunningMean()
                time_val_mean.update(time_val_base)

            samps_seen_best_comp = 0
            samps_seen_best_i2i = 0
            scores_val_best_comp = scores_val
            scores_val_best_i2i = scores_val
            scores_val_latest = scores_val
            stop_trial = False

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

                for idx_batch, data_sb in enumerate(tqdm(self.dataloader, desc="Train", leave=False, disable=(dist.get_rank() != 0))):
                    imgs_sb, texts_sb, class_encs_sb, targ_data_sb = data_sb

                    self.train_img_dumper.dump(imgs_sb, targ_data_sb)

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
                        with torch.no_grad():
                            grad_norm_model = model_grad_l2_norm(self.modelw.model)
                        if self.gpu_rank == 0:
                            PrintLog.batch(idx_batch, lr, loss, embs_img_b, embs_txt_b, logits, self.modelw.model)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.n_samps_seen >= self.lr_warmup:
                            self.lr_schedw.step()
                    else:
                        loss, loss_raw, embs_img_b, embs_txt_b, logits, _ = self.modelw.batch_step(
                            imgs_sb, texts_sb, class_encs_sb, targ_data_sb
                        )
                        loss.backward()
                        with torch.no_grad():
                            grad_norm_model = model_grad_l2_norm(self.modelw.model)
                        if self.gpu_rank == 0:
                            PrintLog.batch(idx_batch, lr, loss, embs_img_b, embs_txt_b, logits, self.modelw.model)
                        self.optimizer.step()
                        if self.n_samps_seen >= self.lr_warmup:
                            self.lr_schedw.step()

                    with torch.no_grad():
                        loss = loss.detach().item()
                        loss_raw = loss_raw.detach().item()
                        loss_mean.update(loss)
                        loss_raw_mean.update(loss_raw)
                        self.n_batches_seen += 1

                    if self.gpu_rank == 0:
                        data_tracker.update_train_batch(
                            samps_seen=self.n_samps_seen,
                            lr=lr,
                            loss_train=loss,
                            loss_raw_train=loss_raw,
                            grad_norm_model=grad_norm_model,
                        )
                        if self.n_batches_seen % self.cfg.metrics_plot_every_batches == 0:
                            data_tracker.save()
                            plot_metrics(data_tracker, ArtifactManager.dpath_trial)

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

                    if self.n_samps_seen >= self.cfg.sample_volume:
                        stop_trial = True
                        break

                loss_train_avg = loss_mean.value()

                time_train_end = time.time()
                time_train = time_train_end - time_train_start

                if self.gpu_rank == 0:

                    time_train_mean.update(time_train)

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

                    # Refresh train metric plots at every epoch boundary (in addition to batch cadence).
                    data_tracker.save()
                    plot_metrics(data_tracker, ArtifactManager.dpath_trial)

                    PrintLog.epoch(
                        time_train,
                        time_train_mean.value(),
                        loss_train_avg,
                        loss_raw_mean.value(),
                        self.n_samps_seen,
                    )

                if stop_trial:
                    break

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

def run_training(cfg=None):
    gpu_rank, gpu_world_size, local_gpu_rank, device = setup_ddp()

    if cfg is None:
        cfg = get_config_train()
    cfg.device = device  # set local device
    seed_libs(cfg.seed)

    if gpu_rank == 0:
        ArtifactManager.set_paths(cfg)
        ArtifactManager.create_trial_dirs()
        ArtifactManager.save_metadata_campaign(cfg)
        ArtifactManager.save_metadata_setting(cfg)
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

def main():
    run_training()


if __name__ == "__main__":
    main()