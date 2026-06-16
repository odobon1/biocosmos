"""
torchrun --standalone --nproc-per-node=auto -m train
"""

import sys
import random
import numpy as np
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
    load_json,
    save_json,
)
from models import VLMWrapper
from utils.data import spawn_dataloader, spawn_partition_data
from utils.eval import EvaluationPipeline
from utils.config import get_config_train
from utils.train import TrialData, ArtifactManager, plot_metrics, parse_scores
from utils.ddp import setup_ddp, cleanup_ddp, rank0

import pdb


torch.set_printoptions(
    precision=4,
    sci_mode =False,
    threshold=1000,  # total elements before summarizing
    edgeitems=3,  # num items to show at the start/end of each dim
    linewidth=120
)


class TrainPipeline:

    def __init__(
        self,
        modelw,
        config,
        resume_state=None,
        trial_state=None,
        local_rank=0,
    ):

        self.modelw = modelw
        self.cfg = config
        self._resume_state = resume_state
        self._local_rank = local_rank

        self.modelw.freeze(self.cfg.freeze["text"], self.cfg.freeze["image"])

        index_data, _, enc2cid = spawn_partition_data(config=self.cfg, partition=self.cfg.train_pt)
        text_template_train = get_text_template(self.cfg.text_template["train"], dataset=self.cfg.dataset)
        self.dataloader = spawn_dataloader(
            index_data=index_data,
            enc2cid=enc2cid,
            text_template=text_template_train,
            config=self.cfg,
            shuffle=True,
            drop_last=True,
            img_pp=self.modelw.img_pp_train,
            use_dv_sampler=self.cfg.dv_batching,
            persistent_workers=self.cfg.hw.persistent_workers_train,
        )

        self.eval_enabled = self.cfg.train_pt != "trainval"
        if self.eval_enabled:
            text_template_eval = get_text_template(self.cfg.text_template["eval"], dataset=self.cfg.dataset)
            self.eval_pipe = EvaluationPipeline(self.cfg, text_template_eval, self.modelw.img_pp_inf)
        else:
            self.eval_pipe = None

        self.lr_warmup = self.cfg.opt["lr"]["warmup"]
        self.init_opt_and_lr_sched()
        self.n_batches_seen = 0
        self.chkpt_thresh = self.cfg.chkpt_every
        self.lr_init_nom = self.cfg.opt["lr"]["init"]

        self.n_samps_seen = 0
        self.idx_epoch = 0
        self.timer_train = Timer()
        self.rmean_time_train = RunningMean()
        self.rmean_time_eval = RunningMean()

        if dist.get_rank() == 0:
            if resume_state is not None and trial_state is not None:
                self.data = TrialData.resume(ArtifactManager.dpath_trial, trial_state)
            else:
                self.data = TrialData(ArtifactManager.dpath_trial)
        else:
            self.data = None

        if resume_state is not None:
            self.n_samps_seen = resume_state["n_samps_seen"]
            self.n_batches_seen = resume_state["n_batches_seen"]
            self.idx_epoch = max(0, resume_state["idx_epoch"] - 1)
            self.chkpt_thresh = resume_state["chkpt_thresh"]
            self.rmean_time_train.n = resume_state["rmean_time_train_n"]
            self.rmean_time_train.mean = resume_state["rmean_time_train_mean"]
            self.rmean_time_eval.n = resume_state["rmean_time_eval_n"]
            self.rmean_time_eval.mean = resume_state["rmean_time_eval_mean"]
            self.opt.load_state_dict(resume_state["optimizer"])
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.cfg.device)
            self.lr_sched.load_state_dict(resume_state["lr_sched"])
            if self.cfg.hw.mixed_prec:
                self.scaler.load_state_dict(resume_state["scaler"])

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

        self.opt = torch.optim.AdamW(
            param_groups, 
            lr=lr_init_nom,
            betas=(self.cfg.opt["beta1"], self.cfg.opt["beta2"]),
            eps=self.cfg.opt["eps"],
        )

        lr_init = self.opt.param_groups[0]["lr"]
        eta_min = lr_init * float(self.cfg.opt['lr']['decay_factor'])
        total_steps = max(1, math.ceil(self.cfg.sample_volume / self.cfg.batch_size) - math.ceil(self.lr_warmup / self.cfg.batch_size))
        self.lr_sched = CosineAnnealingLR(self.opt, T_max=total_steps, eta_min=eta_min)

        if self.cfg.hw.mixed_prec:
            self.scaler = GradScaler()

    def _update_lr_warmup(self) -> float:
        if self.lr_warmup == 0 or self.n_samps_seen >= self.lr_warmup:
            lr = self.opt.param_groups[0]["lr"]
        else:
            frac = self.n_samps_seen / self.lr_warmup
            lr = self.lr_init_nom * frac
            for pg in self.opt.param_groups:
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

    @rank0
    def _print_log_eval(self, header):
        sys.stdout.write('\r')
        sys.stdout.flush()
        PrintLog.eval(
            self.data.eval_metrics,
            self.eval_pipe,
            header=header,
            n_samps_seen=self.n_samps_seen,
            time_eval=self.data.time_eval,
            time_eval_avg=self.rmean_time_eval.value() if self.rmean_time_eval.n > 0 else None,
        )

    @rank0
    def _checkpoint(self, header, idx_batch):
        if self.eval_enabled:
            self.data.update_eval(self.n_samps_seen)
            self._print_log_eval(header)
            self._save_eval_data()
        ArtifactManager.save_metadata_trial(self.data, self.idx_epoch, self.rmean_time_train)
        if not self.cfg.standalone:
            ArtifactManager.update_campaign_time()

        self.data.save()
        ArtifactManager.save_train_state(self, idx_batch)
        ArtifactManager.save_trial_state(self.data)
        plot_metrics(self.data, ArtifactManager.dpath_trial)

    @rank0
    def mark_complete(self):
        metadata_trial = load_json(ArtifactManager.fpath_metadata_trial)
        metadata_trial["complete"] = True
        save_json(metadata_trial, ArtifactManager.fpath_metadata_trial)

    def _step_train(self, imgs_sb, texts_sb, class_encs_sb, targ_data_sb):
        if self.cfg.hw.mixed_prec:
            with autocast(device_type=self.cfg.device.type):
                loss, loss_raw, embs_img_b, embs_txt_b, logits, _ = self.modelw.batch_step(
                    imgs_sb, texts_sb, class_encs_sb, targ_data_sb
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss, loss_raw, embs_img_b, embs_txt_b, logits, _ = self.modelw.batch_step(
                imgs_sb, texts_sb, class_encs_sb, targ_data_sb
            )
            loss.backward()
        return loss, loss_raw, embs_img_b, embs_txt_b, logits

    def _step_optimizer(self):
        if self.cfg.hw.mixed_prec:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

    def train(self):
        try:

            if self._resume_state is None:
                ArtifactManager.save_metadata_trial(self.data, self.idx_epoch, self.rmean_time_train, init_flag=True)
                if self.eval_enabled:
                    PrintLog.texts_eval(self.eval_pipe)

            # BASE EVAL

            if self._resume_state is None and self.eval_enabled:
                cached = ArtifactManager.load_base_eval_cache()
                if cached is not None:
                    scores = parse_scores(cached["scores"])
                    eval_metrics = {
                        "scores": scores,
                        "loss_raw": {p: None for p in self.eval_pipe.partitions},
                    }
                    time_eval = None  # cached base eval was not run; don't pollute eval-time mean
                else:
                    eval_metrics, time_eval = self.eval_pipe.evaluate(self.modelw, loss_flag=False)
                    ArtifactManager.save_base_eval_cache(eval_metrics)
                if time_eval is not None:
                    self.rmean_time_eval.update(time_eval)
                if self.data is not None:
                    self.data.eval_metrics = eval_metrics
                    self.data.time_eval = time_eval
                header = "Base - Cached" if cached is not None else "Base"
                self._checkpoint(header=header, idx_batch=-1)
                dist.barrier()  # wait for rank0 to finish _checkpoint (creates checkpoint dir) before all ranks write rng state
                ArtifactManager.save_rng_states(self._local_rank)
                dist.barrier()
            elif self._resume_state is not None:
                # Resuming from base-eval checkpoint (no training done yet): restore RNG before epoch loop
                if self._resume_state["idx_epoch"] == 0 and self._resume_state["idx_batch"] == -1:
                    if "rng_states" in self._resume_state and self._local_rank in self._resume_state["rng_states"]:
                        rng = ArtifactManager.load_rng_state(self._local_rank)
                        torch.set_rng_state(rng["rng_cpu"])
                        torch.cuda.set_rng_state_all(rng["rng_cuda"])
                        np.random.set_state(rng["rng_numpy"])
                        random.setstate(rng["rng_random"])
                    self._resume_state = None

            for _ in range(self.cfg.n_epochs - self.idx_epoch):
                self.timer_train.start()
                self.idx_epoch += 1

                PrintLog.batch_logs_epoch_header(self.idx_epoch, self.cfg.n_epochs)

                # Let samplers know current epoch (crucial for shuffling)
                sampler = getattr(self.dataloader, "sampler", None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(self.idx_epoch)
                batch_sampler = getattr(self.dataloader, "batch_sampler", None)
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(self.idx_epoch)

                self.modelw.model.train()

                loss_mean = RunningMean()
                loss_raw_mean = RunningMean()

                for idx_batch, data_sb in enumerate(pbar := tqdm(
                    self.dataloader,
                    desc=f"Train ({self.idx_epoch}/{self.cfg.n_epochs})",
                    leave=False,
                    disable=(dist.get_rank() != 0),
                    file=sys.stdout,
                )):
                    # Skip already-processed batches when resuming the interrupted epoch;
                    # on the last skipped batch restore RNG to match original run state.
                    if (
                        self._resume_state is not None
                        and self.idx_epoch == self._resume_state["idx_epoch"]
                        and idx_batch <= self._resume_state["idx_batch"]
                    ):
                        if idx_batch == self._resume_state["idx_batch"]:
                            rng = ArtifactManager.load_rng_state(self._local_rank)
                            torch.set_rng_state(rng["rng_cpu"])
                            torch.cuda.set_rng_state_all(rng["rng_cuda"])
                            np.random.set_state(rng["rng_numpy"])
                            random.setstate(rng["rng_random"])
                            self._resume_state = None
                        continue

                    imgs_sb, texts_sb, class_encs_sb, targ_data_sb = data_sb

                    if self.idx_epoch == 1 and idx_batch == 0:
                        PrintLog.texts(texts_sb)

                    imgs_sb = imgs_sb.to(self.cfg.device, non_blocking=True)
                    class_encs_sb = class_encs_sb.to(self.cfg.device)
                    B = imgs_sb.size(0) * dist.get_world_size()
                    self.n_samps_seen += B

                    lr = self._update_lr_warmup() if self.lr_warmup > 0 else self.opt.param_groups[0]["lr"]

                    self.opt.zero_grad(set_to_none=True)
                    loss, loss_raw, embs_img_b, embs_txt_b, logits = self._step_train(
                        imgs_sb, 
                        texts_sb, 
                        class_encs_sb, 
                        targ_data_sb,
                    )
                    with torch.no_grad():
                        grad_norm_model = model_grad_l2_norm(self.modelw.model)
                    PrintLog.batch(idx_batch, lr, loss, embs_img_b, embs_txt_b, logits, self.modelw.model)
                    self._step_optimizer()

                    if self.n_samps_seen >= self.lr_warmup:
                        self.lr_sched.step()

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

                    if self.n_samps_seen >= self.chkpt_thresh:
                        pbar.clear()
                        self.timer_train.stop()

                        while self.n_samps_seen >= self.chkpt_thresh:
                            threshold_hit = self.chkpt_thresh
                            self.chkpt_thresh += self.cfg.chkpt_every

                        # TRAIN-TIME EVAL

                        if self.eval_enabled:
                            eval_metrics, time_eval = self.eval_pipe.evaluate(
                                self.modelw,
                                loss_flag=True,
                            )
                            self.rmean_time_eval.update(time_eval)
                            if self.data is not None:
                                self.data.eval_metrics = eval_metrics
                                self.data.time_eval = time_eval
                        self._checkpoint(
                            header=f"{threshold_hit:,}",
                            idx_batch=idx_batch,
                        )
                        ArtifactManager.save_rng_states(self._local_rank)
                        dist.barrier()

                        self.timer_train.start()
                        pbar.refresh()

                    if self.n_samps_seen >= self.cfg.sample_volume:
                        break

                # EPOCH DONE

                self.timer_train.stop()
                time_train = self.timer_train.get_elapsed_time()
                self.timer_train.reset()
                self.rmean_time_train.update(time_train)

                PrintLog.epoch(
                    time_train,
                    self.rmean_time_train.value(),
                    loss_mean.value(),
                    loss_raw_mean.value(),
                    self.n_samps_seen,
                    self.idx_epoch,
                    self.cfg.n_epochs,
                )

            # FINAL EVAL

            if self.eval_enabled:
                eval_metrics, time_eval = self.eval_pipe.evaluate(
                    self.modelw,
                    loss_flag=True,
                )
                self.rmean_time_eval.update(time_eval)
                if self.data is not None:
                    self.data.eval_metrics = eval_metrics
                    self.data.time_eval = time_eval
            self._checkpoint(
                header="Final",
                idx_batch=-1,
            )
            if self.data is not None:
                self.modelw.save(ArtifactManager.dpath_model_final)
                if self.eval_enabled:
                    ArtifactManager.save_eval_data(
                        ArtifactManager.dpath_model_final,
                        eval_metrics,
                        self.n_samps_seen,
                        self.n_samps_seen,
                    )
            ArtifactManager.save_rng_states(self._local_rank)
            dist.barrier()

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
    dist.barrier()  # ensure rank0 finishes creating dirs before other ranks proceed
    ArtifactManager.save_metadata_setting(cfg)
    if cfg.dev["logging"]:
        PrintLog.create_logs(ArtifactManager.dpath_trial / "logs")
    PrintLog.init_train(cfg)

    modelw = VLMWrapper.build(cfg, verbose=(dist.get_rank() == 0))
    modelw.set_class_wts(cfg)
    if cfg.loss2["mix"] != 0.0:
        modelw.set_class_wts(cfg, secondary=True)

    resume_state = None
    trial_state = None
    if ArtifactManager.resuming:
        resume_state = ArtifactManager.load_train_state()
        modelw._unwrapped_model.load_state_dict(resume_state["model"])
        modelw.norm_mean = resume_state["norm_mean"]
        modelw.norm_std = resume_state["norm_std"]
        modelw.set_image_preprocessors()
        trial_state = ArtifactManager.load_trial_state()

    modelw.model = DDP(modelw.model, device_ids=[local_gpu_rank], output_device=local_gpu_rank)

    train_pipe = TrainPipeline(
        modelw, 
        cfg, 
        resume_state=resume_state, 
        trial_state=trial_state, 
        local_rank=local_gpu_rank,
    )
    train_pipe.train()
    train_pipe.mark_complete()

    cleanup_ddp()

def main():
    run_training()


if __name__ == "__main__":
    main()