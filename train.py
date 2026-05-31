"""
torchrun --standalone --nproc-per-node=auto -m train
"""

import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import math

from utils.utils import (
    seed_libs,
    get_text_template,
    RunningMean,
    Timer,
    PrintLog,
    model_grad_l2_norm,
)
from models import VLMWrapper
from utils.data import spawn_dataloader, spawn_partition_data
from utils.eval import EvaluationPipeline
from utils.config import get_config_train
from utils.train import TrialData, ArtifactManager, plot_metrics
from utils.ddp import setup_ddp, cleanup_ddp, rank0

import pdb


torch.set_printoptions(
    precision=4,
    sci_mode =False,
    threshold=1000,  # total elements before summarizing
    edgeitems=3,  # num items to show at the start/end of each dim
    linewidth=120
)


class LRSchedulerWrapper:

    def __init__(self, optimizer, config, total_steps):

        self.opt = optimizer
        self.type = config.opt['lr']['sched']

        args = config.lr_sched_params.get("args")
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
    ):

        self.modelw = modelw
        self.cfg = config

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

        text_template_eval = get_text_template(self.cfg.text_template["eval"], dataset=self.cfg.dataset)
        self.eval_pipe = EvaluationPipeline(self.cfg, text_template_eval, self.modelw.img_pp_inf)

        self.lr_warmup = self.cfg.opt["lr"]["warmup"]
        self.init_opt_and_lr_sched()
        self.n_batches_seen = 0
        self.eval_threshold = self.cfg.eval_every
        self.lr_init_nom = self.cfg.opt["lr"]["init"]

        self.n_samps_seen = 0
        self.idx_epoch = 0
        self.timer_train = Timer()
        self.rmean_time_train = RunningMean()
        self.data = TrialData(ArtifactManager.dpath_trial) if dist.get_rank() == 0 else None

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
            lr=lr_init_nom,
            betas=(self.cfg.opt["beta1"], self.cfg.opt["beta2"]),
            eps=self.cfg.opt["eps"],
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

    @rank0
    def _save_eval_data(self):
        ArtifactManager.save_eval_data(
            ArtifactManager.dpath_model_checkpoint,
            self.data.eval_metrics,
            self.n_samps_seen,
            self.n_samps_seen,
        )
        ArtifactManager.save_eval_data(
            ArtifactManager.dpath_model_best_comp,
            self.data.eval_metrics_best_comp,
            self.data.n_samps_seen_best_comp,
            self.n_samps_seen,
        )
        ArtifactManager.save_eval_data(
            ArtifactManager.dpath_model_best_i2i,
            self.data.eval_metrics_best_i2i,
            self.data.n_samps_seen_best_i2i,
            self.n_samps_seen,
        )

    @rank0
    def _print_log_eval(self, header):
        PrintLog.eval(
            self.data.eval_metrics,
            self.eval_pipe,
            header=header,
            n_samps_seen=self.n_samps_seen,
            idx_epoch=self.idx_epoch,
            time_eval=self.data.time_eval,
            log_to="eval",
        )

    @rank0
    def _checkpoint(self, header, checkpoint_best_comp, checkpoint_best_i2i):
        self.data.update_eval(self.n_samps_seen)
        self._print_log_eval(header)
        if checkpoint_best_comp:
            self.modelw.save(ArtifactManager.dpath_model_best_comp)
            self.data.n_samps_seen_best_comp = self.n_samps_seen
            self.data.eval_metrics_best_comp = self.data.eval_metrics
        if checkpoint_best_i2i:
            self.modelw.save(ArtifactManager.dpath_model_best_i2i)
            self.data.n_samps_seen_best_i2i = self.n_samps_seen
            self.data.eval_metrics_best_i2i = self.data.eval_metrics

        self.modelw.save(ArtifactManager.dpath_model_checkpoint)
        self._save_eval_data()
        ArtifactManager.save_runtime_data(self.data, self.idx_epoch, self.rmean_time_train)
        if not self.cfg.standalone:
            ArtifactManager.update_campaign_time()

        self.data.save()
        plot_metrics(self.data, ArtifactManager.dpath_trial)

    def train(self):
        try:

            PrintLog.texts_eval(self.eval_pipe.get_eval_texts())

            # BASE EVAL

            eval_metrics, _, _, time_eval = self.eval_pipe.evaluate(self.modelw, loss_flag=False)
            if self.data is not None:
                self.data.eval_metrics = eval_metrics
                self.data.time_eval = time_eval
            self._checkpoint(header="Base", checkpoint_best_comp=True, checkpoint_best_i2i=True)

            for _ in range(self.cfg.n_epochs):
                self.timer_train.start()
                self.idx_epoch += 1

                PrintLog.epoch_header(self.idx_epoch, self.cfg.n_epochs)
                lr = self.lr_schedw.get_lr()
                PrintLog.train_header(lr)

                # Let samplers know current epoch (crucial for shuffling)
                sampler = getattr(self.dataloader, "sampler", None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(self.idx_epoch)
                batch_sampler = getattr(self.dataloader, "batch_sampler", None)
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(self.idx_epoch)

                self.modelw.model.train()

                lr_mean = RunningMean()
                loss_mean = RunningMean()
                loss_raw_mean = RunningMean()

                for idx_batch, data_sb in enumerate(tqdm(self.dataloader, desc="Train", leave=False, disable=(dist.get_rank() != 0))):
                    imgs_sb, texts_sb, class_encs_sb, targ_data_sb = data_sb

                    if self.idx_epoch == 1 and idx_batch == 0:
                        PrintLog.texts(texts_sb)

                    imgs_sb = imgs_sb.to(self.cfg.device, non_blocking=True)
                    class_encs_sb = class_encs_sb.to(self.cfg.device)
                    B = imgs_sb.size(0) * dist.get_world_size()
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

                    if self.data is not None:
                        self.data.update_train_batch(
                            self.n_samps_seen,
                            lr=lr,
                            loss_train=loss,
                            loss_raw_train=loss_raw,
                            grad_norm_model=grad_norm_model,
                        )

                    if self.n_samps_seen >= self.eval_threshold:
                        self.timer_train.stop()
                        while self.n_samps_seen >= self.eval_threshold:
                            threshold_hit = self.eval_threshold
                            self.eval_threshold += self.cfg.eval_every

                        # TRAIN-TIME EVAL

                        eval_metrics, is_best_comp, is_best_i2i, time_eval = self.eval_pipe.evaluate(self.modelw, loss_flag=True)
                        if self.data is not None:
                            self.data.eval_metrics = eval_metrics
                            self.data.time_eval = time_eval
                        self._checkpoint(header=f"{threshold_hit:,}", checkpoint_best_comp=is_best_comp, checkpoint_best_i2i=is_best_i2i)

                        self.timer_train.start()

                    if self.n_samps_seen >= self.cfg.sample_volume:
                        break

                # EPOCH DONE

                loss_train_avg = loss_mean.value()

                self.timer_train.stop()
                time_train = self.timer_train.get_elapsed_time()
                self.timer_train.reset()

                self.rmean_time_train.update(time_train)

                PrintLog.epoch(
                    time_train,
                    self.rmean_time_train.value(),
                    loss_train_avg,
                    loss_raw_mean.value(),
                    self.n_samps_seen,
                )

            # FINAL EVAL

            eval_metrics, is_best_comp, is_best_i2i, time_eval = self.eval_pipe.evaluate(self.modelw, loss_flag=True)
            if self.data is not None:
                self.data.eval_metrics = eval_metrics
                self.data.time_eval = time_eval
            self._checkpoint(header="Final", checkpoint_best_comp=is_best_comp, checkpoint_best_i2i=is_best_i2i)

        finally:
            PrintLog.close_logs()

def run_training(cfg=None):
    local_gpu_rank, device = setup_ddp()

    if cfg is None:
        cfg = get_config_train()
    cfg.device = device  # set local device
    seed_libs(cfg.seed)

    ArtifactManager.set_paths(cfg)
    ArtifactManager.create_trial_dirs()
    ArtifactManager.save_metadata_setting(cfg)
    if cfg.logging:
        PrintLog.create_logs(ArtifactManager.dpath_trial / "logs")
    PrintLog.init_train(cfg)

    modelw = VLMWrapper.build(cfg, verbose=(dist.get_rank() == 0))
    modelw.set_class_wts(cfg)
    if cfg.loss2["mix"] != 0.0:
        modelw.set_class_wts(cfg, secondary=True)
    modelw.model = DDP(modelw.model, device_ids=[local_gpu_rank], output_device=local_gpu_rank)

    train_pipe = TrainPipeline(modelw, cfg)
    train_pipe.train()

    cleanup_ddp()

def main():
    run_training()


if __name__ == "__main__":
    main()