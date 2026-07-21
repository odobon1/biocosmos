import sys
import random
import time
import numpy as np
import torch
from torch.amp import autocast
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
    save_json,
    RunningMean,
    Timer,
    TimeTracker,
    PrintLog,
    model_grad_l2_norm,
)
from models import VLMWrapper
from utils.data import spawn_dataloader, spawn_partition_data
from utils.loss import configure_htarg_shuf
from utils.eval import EvaluationPipeline
from utils.manifold_viz import compute_projections, compute_pooled_projections
from utils.train import TrialData, ArtifactManager, plot_metrics, parse_scores
from utils.hardware import apply_backend_flags, read_cgroup_ram
from utils.ddp import setup_ddp, cleanup_ddp, rank0

import pdb


torch.set_printoptions(
    precision=4,
    sci_mode =False,
    threshold=1000,  # total elements before summarizing
    edgeitems=3,  # num items to show at the start/end of each dim
    linewidth=120
)


def _timed_next(loader, wait_acc):
    """Yield batches from loader, accumulating into wait_acc[0] the seconds spent blocked in next() -- the
    time the train loop sat waiting on the DataLoader (prefetch queue empty => GPU starved by data)."""
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            return
        wait_acc[0] += time.perf_counter() - t0
        yield batch


def _fmt_thresh(n_samps):
    """Compact dir name for a sample-count eval threshold, e.g. 100_000 -> '100k'."""
    if n_samps % 1000 == 0:
        return f"{n_samps // 1000}k"
    return str(n_samps)


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
            persistent_workers=self.cfg.hw.persistent_workers["train"],
        )

        self.eval_enabled = self.cfg.train_pt != "trainval"
        # manifold viz runs for the first dev.manifold_viz.n_trials seeds of each setting/dataset group
        self._viz_manifold = self.cfg.idx_seed < self.cfg.dev["manifold_viz"]["n_trials"]
        # pooled shared-frame viz (dev.manifold_viz.pooled.enabled): cache each eval's embeddings and fit
        # one pooled projection over all thresholds at end-of-trial (compute_pooled_projections)
        self._pooled_manifold = self._viz_manifold and self.cfg.dev["manifold_viz"]["pooled"]["enabled"]
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
        self.time_tracker = TimeTracker()

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
            self.time_tracker.load_state_dict(resume_state["times"])
            self.opt.load_state_dict(resume_state["optimizer"])
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.cfg.device)
            self.lr_sched.load_state_dict(resume_state["lr_sched"])

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
    def _write_base_eval(self, entry):
        # materialize the base-eval cache entry into evals/_base/ so base is a uniform member of the
        # eval sequence the render pass sweeps. metrics.json always (every trial records its base eval);
        # projections only for viz trials -- a non-viz trial computes none of its own projections, so it
        # mustn't inherit the cache's base projections either; embs only for pooled trials, whose cache
        # hit is gated on the entry carrying them (require_embs). On a fresh base eval the npz files were
        # computed straight into _base/ (already on disk, skipped here); only a cache hit writes them.
        dst = ArtifactManager.dpath_trial / "evals" / "_base"
        dst.mkdir(parents=True, exist_ok=True)
        save_json(entry["metrics"], dst / "metrics.json")
        if self._viz_manifold and not (dst / "projections.npz").exists():
            np.savez(dst / "projections.npz", **entry["projections"])
        if self._pooled_manifold and not (dst / "embs.npz").exists():
            np.savez(dst / "embs.npz", **entry["embs"])

    def _compute_projections_timed(self, eval_bundles, dpath_cache):
        # COLLECTIVE (sharded t-SNE) -- every rank enters; elapsed folds into the viz_compute mean.
        # All ranks spend ~the same wall time here; only rank-0's tracker is persisted.
        # Return the training step's reserved-but-unallocated allocator pool to CUDA before the t-SNE's
        # transient kNN/repulsion chunk buffers allocate. empty_cache is a local allocator op (no
        # collective) so it can't desync.
        torch.cuda.empty_cache()
        with self.time_tracker.measure("viz_compute"):
            compute_projections(eval_bundles["id"], eval_bundles["ood"], dpath_cache, self.cfg.manifold_viz,
                                1 << self.cfg.hw.eval["tsne_chunk_log2"], cache_embs=self._pooled_manifold)

    def _viz_eval(self, eval_bundles, eval_name):
        """Compute + cache this eval's manifold projections (COLLECTIVE -- every rank must enter) under
        evals/<eval_name>/. Rendering from the cache is done off-process post-trial by the campaign render
        worker (tools/manifold_viz.py), so no rank blocks in a collective while rank 0 renders."""
        dpath_eval = ArtifactManager.dpath_trial / "evals" / eval_name
        self._compute_projections_timed(eval_bundles, dpath_eval)

    def _pooled_eval(self):
        """Fit the pooled shared-frame projection over every threshold's cached embeddings (COLLECTIVE --
        sharded t-SNE, every rank must enter) and cache the per-threshold masked blocks under evals/*/,
        rendered post-trial off-process. Runs once at end-of-trial, after every per-eval cache is written."""
        torch.cuda.empty_cache()  # release the training step's reserved pool before the pooled t-SNE buffers
        budget = self.cfg.dev["manifold_viz"]["pooled"]["budget"]
        with self.time_tracker.measure("viz_compute"):
            compute_pooled_projections(ArtifactManager.dpath_trial / "evals", self.cfg.manifold_viz, budget,
                                       1 << self.cfg.hw.eval["tsne_chunk_log2"])

    def _save_mid_eval(self, threshold_hit, eval_metrics, eval_bundles):
        # NOT @rank0: _viz_eval -> compute_projections runs the sharded t-SNE collectively, so every rank
        # must enter it. Only the metrics write is rank-0 gated.
        eval_name = _fmt_thresh(threshold_hit)
        if dist.get_rank() == 0:
            ArtifactManager.save_eval_data(ArtifactManager.dpath_trial / "evals" / eval_name, eval_metrics, self.n_samps_seen, self.n_samps_seen)
        if self._viz_manifold:
            self._viz_eval(eval_bundles, eval_name)

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
            time_eval_avg=self.time_tracker.mean("eval"),
        )

    def _snapshot_memory(self):
        # COLLECTIVE -- every rank contributes its GPU's reading and the trial VRAM value is the max
        # across GPUs. Peak reserved catches transient highs between snapshots (eval sim matrices,
        # t-SNE buffers); total - free (device-wide, right now) additionally counts the CUDA context.
        free, total = torch.cuda.mem_get_info(self.cfg.device)
        used = max(torch.cuda.max_memory_reserved(self.cfg.device), total - free)
        vram = torch.tensor([used, total], device=self.cfg.device, dtype=torch.int64)
        dist.all_reduce(vram, op=dist.ReduceOp.MAX)
        return {"ram": read_cgroup_ram(), "vram": tuple(vram.tolist())}

    def _checkpoint(self, header, idx_batch, record_eval=True):
        # NOT @rank0: the memory snapshot all-reduces across ranks, so every rank must enter.
        # Only rank 0 proceeds to the writes, as before.
        mem = self._snapshot_memory()
        if dist.get_rank() != 0:
            return
        if self.eval_enabled and record_eval:
            self.data.update_eval(self.n_samps_seen)
            self._print_log_eval(header)
            self._save_eval_data()
        ArtifactManager.save_metadata_trial(self.data, self.idx_epoch, self.time_tracker, self.n_samps_seen, self.cfg.sample_volume, mem)
        ArtifactManager.update_campaign_time()
        ArtifactManager.update_campaign_memory(mem)

        self.data.save()
        ArtifactManager.save_train_state(self, idx_batch)
        ArtifactManager.save_trial_state(self.data)
        plot_metrics(self.data, ArtifactManager.dpath_trial)

    def _step_train(self, imgs_sb, texts_sb, class_encs_sb, targ_data_sb):
        if self.cfg.hw.mixed_prec:
            with autocast(device_type=self.cfg.device.type, dtype=torch.bfloat16):
                loss, loss_raw, embs_img_b, embs_txt_b, logits, _, batch_stats = self.modelw.batch_step(
                    imgs_sb, texts_sb, class_encs_sb, targ_data_sb
                )
        else:
            loss, loss_raw, embs_img_b, embs_txt_b, logits, _, batch_stats = self.modelw.batch_step(
                imgs_sb, texts_sb, class_encs_sb, targ_data_sb
            )
        loss.backward()
        return loss, loss_raw, embs_img_b, embs_txt_b, logits, batch_stats

    def _step_optimizer(self):
        self.opt.step()

    def train(self):
        try:

            if self._resume_state is None:
                mem = self._snapshot_memory()  # COLLECTIVE -- every rank must enter
                ArtifactManager.save_metadata_trial(self.data, self.idx_epoch, self.time_tracker, self.n_samps_seen, self.cfg.sample_volume, mem, init_flag=True)
                if self.eval_enabled:
                    PrintLog.texts_eval(self.eval_pipe)

            # BASE EVAL

            if self._resume_state is None and self.eval_enabled:
                cached = ArtifactManager.load_base_eval_cache(self.cfg, require_projections=self._viz_manifold, require_embs=self._pooled_manifold)  # @rank0; None elsewhere
                # single-source the hit/miss decision: a concurrent campaign's cache write landing between
                # independent per-rank reads would split the branch, and the miss branch enters collective
                # ops (evaluate / sharded t-SNE) that every rank must join. Only the small metrics dict is
                # broadcast; the bulky projections/embs stay on rank 0, the only rank that writes _base.
                sync = [cached["metrics"] if cached is not None else None]
                dist.broadcast_object_list(sync, src=0)
                metrics_cached = sync[0]
                if metrics_cached is not None:
                    scores = parse_scores(metrics_cached["scores"])
                    eval_metrics = {
                        "scores": scores,
                        "loss_raw": {p: None for p in self.eval_pipe.partitions},
                        "sim": {stat: None for stat in ("min", "max", "median", "mean")},
                        "targ": {stat: None for stat in ("min", "max", "median", "mean")},
                    }
                    time_eval = None  # cached base eval was not run; don't pollute eval-time mean
                    entry = cached
                else:
                    eval_metrics, time_eval, eval_bundles = self.eval_pipe.evaluate(
                        self.modelw,
                        loss_flag=False,
                        collect_eval_bundles=self._viz_manifold,
                    )
                    if self._viz_manifold:
                        self._viz_eval(eval_bundles, "_base")  # projections (+ embs if pooled) straight into evals/_base
                    entry = ArtifactManager.save_base_eval_cache(self.cfg, eval_metrics)  # rank0 gets the entry back; None elsewhere
                self._write_base_eval(entry)  # base -> evals/_base (uniform member of the eval sequence)
                if time_eval is not None:
                    self.time_tracker.add("eval", time_eval)
                if self.data is not None:
                    self.data.eval_metrics = eval_metrics
                    self.data.time_eval = time_eval
                header = "Base - Cached" if metrics_cached is not None else "Base"
                self._checkpoint(header=header, idx_batch=-1)
                dist.barrier()  # wait for rank0 to finish _checkpoint (creates checkpoint dir) before all ranks write rng state
                ArtifactManager.save_rng_states(self._local_rank)
                dist.barrier()
            elif self._resume_state is not None:
                # Resuming from base-eval checkpoint (no training done yet): restore RNG before epoch loop
                if self._resume_state["idx_epoch"] == 0 and self._resume_state["idx_batch"] == -1:
                    if (ArtifactManager.dpath_model_checkpoint / f"rng_state_rank{self._local_rank}.pt").exists():
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
                data_wait = [0.0]

                for idx_batch, data_sb in enumerate(pbar := tqdm(
                    _timed_next(self.dataloader, data_wait),
                    total=len(self.dataloader),
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
                    loss, loss_raw, embs_img_b, embs_txt_b, logits, batch_stats = self._step_train(
                        imgs_sb,
                        texts_sb,
                        class_encs_sb,
                        targ_data_sb,
                    )
                    with torch.no_grad():
                        grad_norm_model = model_grad_l2_norm(self.modelw.model)
                    PrintLog.batch(idx_batch, lr, loss, embs_img_b, embs_txt_b, logits, self.modelw.model, batch_stats)
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
                            batch_stats=batch_stats,
                        )

                    if self.n_samps_seen >= self.chkpt_thresh:
                        pbar.clear()
                        self.timer_train.stop()

                        while self.n_samps_seen >= self.chkpt_thresh:
                            threshold_hit = self.chkpt_thresh
                            self.chkpt_thresh += self.cfg.chkpt_every

                        # skip the train-time eval+checkpoint when the threshold lands on
                        # sample_volume -- the final eval+checkpoint right below cover it
                        if threshold_hit != self.cfg.sample_volume:
                            # TRAIN-TIME EVAL (skipped entirely when traintime_evals is off -- only base/final eval)
                            traintime_eval = self.eval_enabled and self.cfg.dev["traintime_evals"]
                            if traintime_eval:
                                eval_metrics, time_eval, eval_bundles = self.eval_pipe.evaluate(
                                    self.modelw,
                                    loss_flag=True,
                                    collect_eval_bundles=self._viz_manifold,
                                )
                                self.time_tracker.add("eval", time_eval)
                                if self.data is not None:
                                    self.data.eval_metrics = eval_metrics
                                    self.data.time_eval = time_eval
                                self._save_mid_eval(threshold_hit, eval_metrics, eval_bundles)
                            self._checkpoint(
                                header=f"{threshold_hit:,}",
                                idx_batch=idx_batch,
                                record_eval=traintime_eval,
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
                self.time_tracker.add("train", time_train)

                # gather per-rank dataloader wait so rank0 can log all ranks (collective: every rank must reach this)
                wait_t = torch.tensor([data_wait[0]], device=self.cfg.device)
                waits = [torch.zeros_like(wait_t) for _ in range(dist.get_world_size())]
                dist.all_gather(waits, wait_t)
                time_data_wait_ranks = [w.item() for w in waits]

                PrintLog.epoch(
                    time_train,
                    self.time_tracker.mean("train"),
                    time_data_wait_ranks,
                    loss_mean.value(),
                    loss_raw_mean.value(),
                    self.n_samps_seen,
                    self.idx_epoch,
                    self.cfg.n_epochs,
                )

            # FINAL EVAL

            if self.eval_enabled:
                eval_metrics, time_eval, eval_bundles = self.eval_pipe.evaluate(
                    self.modelw,
                    loss_flag=True,
                    collect_eval_bundles=self._viz_manifold,
                )
                self.time_tracker.add("eval", time_eval)
                if self.data is not None:
                    self.data.eval_metrics = eval_metrics
                    self.data.time_eval = time_eval
                    ArtifactManager.save_eval_data(
                        ArtifactManager.dpath_eval_final,
                        eval_metrics,
                        self.n_samps_seen,
                        self.n_samps_seen,
                    )
                if self._viz_manifold:
                    self._viz_eval(eval_bundles, "final")  # COLLECTIVE compute+cache; rendered post-trial off-process
            self._checkpoint(
                header="Final",
                idx_batch=-1,
            )
            if self.data is not None:
                self.modelw.save(ArtifactManager.dpath_model_final)
            ArtifactManager.save_rng_states(self._local_rank)
            dist.barrier()  # all per-eval caches (incl. final) now on disk -> safe to pool them

            if self._pooled_manifold:
                self._pooled_eval()  # COLLECTIVE: pooled shared-frame projection over all cached embeddings

        finally:
            PrintLog.close_logs()

def run_training(cfg):
    local_gpu_rank, device = setup_ddp(cfg.hw.pg_timeout)
    cfg.device = device  # set local device
    seed_libs(cfg.seed)
    apply_backend_flags(cfg.hw)
    configure_htarg_shuf(cfg.htarg_shuf, cfg.seed)

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
    ArtifactManager.update_metric_stats(cfg.stats["spread_type"])
    ArtifactManager.update_stats_tables(cfg.stats["table_eval_group"], cfg.stats["spread_type"])
    ArtifactManager.update_metrics_xlsx(cfg.stats["table_eval_group"], cfg.stats["spread_type"], cfg.stats["xlsx"]["bold_high"], cfg.stats["xlsx"]["ordered"], cfg.stats["xlsx"]["heatmap"])

    cleanup_ddp()
