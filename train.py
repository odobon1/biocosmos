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
    RunningMean,
    Timer,
    TimeTracker,
    PrintLog,
    model_grad_l2_norm,
)
from models import VLMWrapper
from utils.config import get_config_stats
from utils.data import spawn_dataloader, spawn_partition_data
from utils.loss import configure_phylo_targs, Criterion
from utils.eval import EvaluationPipeline
from utils.manif_viz import compute_projections, compute_pooled_projections
from utils.train import TrialData, ArtifactManager, parse_scores
from utils.report import (
    plot_metrics,
    update_metric_stats,
    update_chkpt_selection,
    arm_sweep_complete,
    dataset_sweep_complete,
    seed_sweep_complete,
    update_arm_stats,
    update_dataset_stats,
    update_campaign_stats,
)
from utils.hardware import apply_backend_flags, read_cgroup_ram, start_ram_peak_tracker
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

def pass_epoch_span(cfg, idx_pass):
    """(first epoch, last epoch), 1-based inclusive, that dataloader pass `idx_pass` (1-based) draws
    from -- the epochs its window of the sample stream touches, where a pass consumes samps_per_pass
    samples and the last one stops at sample_volume. Under chain-shuffle a pass's partial-batch
    permutation tail carries into the next pass (ChainShuffleDistributedSampler), so passes do NOT
    tile epochs: consecutive passes share the epoch straddling their boundary. Without chaining an
    epoch IS a pass, so both ends land on idx_pass."""
    samps_start = (idx_pass - 1) * cfg.samps_per_pass
    samps_end = min(idx_pass * cfg.samps_per_pass, cfg.sample_volume)
    return samps_start // cfg.samps_per_epoch + 1, math.ceil(samps_end / cfg.samps_per_epoch)

def samps_stop(cfg):
    """The sample count training runs to: sample_volume, or -- under cfg.chkpt_stop (the trainval phase) -- that
    checkpoint index's threshold, chkpt_stop * chkpt_interval. The last index runs to sample_volume itself, which
    covers the skipped last mid-train threshold the way the final eval does. The LR schedule is untouched (built
    over sample_volume), so a stopped run sees the same LR at every checkpoint as a full one."""
    if cfg.chkpt_stop is None or cfg.chkpt_stop == cfg.n_chkpts:
        return cfg.sample_volume
    return cfg.chkpt_stop * cfg.chkpt_interval


class TrainPipeline:
    """DDP rank discipline: every undecorated method is entered by ALL ranks and may run
    collectives (all_reduce / all_gather / sharded t-SNE), so no caller may rank-gate one.
    Every @rank0 method holds rank-0-only side effects (artifact writes, logging, TrialData
    access -- self.data is None off rank 0) and must never run a collective. @rank0 is the
    single gate: no inline dist.get_rank() gating in this class (display-only flags like
    tqdm's disable are fine -- every rank still executes the same work)."""

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
        # manifold viz runs for a window of each arm/coord/dataset group's seed sweep: the manif_viz.n_seeds
        # seeds starting at manif_viz.n_seeds_offset. A window the sweep hasn't reached selects nothing (no
        # error, no warning) -- raising the campaign's seed count later pulls those trials into it.
        seed_off = self.cfg.manif_viz["n_seeds_offset"]
        self._manif_viz = seed_off <= self.cfg.idx_seed < seed_off + self.cfg.manif_viz["n_seeds"]
        # pooled shared-frame viz (manif_viz.pooled.enabled): fit one pooled projection over all
        # thresholds at end-of-trial (compute_pooled_projections). Every viz trial caches its per-eval
        # embeddings regardless -- the post-trial UMAP fits read them too.
        self._pooled_manif_viz = self._manif_viz and self.cfg.manif_viz["pooled"]["enabled"]
        if self.eval_enabled:
            text_template_eval = get_text_template(self.cfg.text_template["eval"], dataset=self.cfg.dataset)
            self.eval_pipe = EvaluationPipeline(self.cfg, text_template_eval, self.modelw.img_pp_inf)
        else:
            self.eval_pipe = None

        self.lr_warmup = round(self.cfg.opt["lr"]["warmup"] * self.cfg.sample_volume)  # warmup fraction -> samples
        self.init_opt_and_lr_sched()
        self.n_batches_seen = 0
        self.chkpt_thresh = self.cfg.chkpt_interval
        self.samps_stop = samps_stop(self.cfg)
        self.lr_init_nom = self.cfg.opt["lr"]["init"]

        self.n_samps_seen = 0
        self.idx_epoch = 0
        self.timer_train = Timer()
        self.time_tracker = TimeTracker()

        self.data = self._init_trial_data(trial_state)  # TrialData on rank 0; None elsewhere
        self._logit_scalars_tracked = self._tracked_logit_scalars()
        self._targ_stats_tracked = self._tracked_targ_stats()
        self._batch_diag = self.cfg.dev["batch_diagnostics"]
        self._params_prev = None  # pre-step parameter snapshot, allocated on the first step

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

        # logit scalars (temp/bias) train at lr * their loss's logits.scalar_lr_factor, decay-decoupled
        scalar_factors = {
            "logit_scale":  self.cfg.loss["logits"]["scalar_lr_factor"],
            "logit_bias":   self.cfg.loss["logits"]["scalar_lr_factor"],
            "logit_scale2": self.cfg.loss2["logits"]["scalar_lr_factor"],
            "logit_bias2":  self.cfg.loss2["logits"]["scalar_lr_factor"],
        }

        params_decay, params_no_decay = [], []
        params_scalar = {}  # scalar_lr_factor -> params
        for name, param in self.modelw.model.named_parameters():
            if not param.requires_grad:
                continue
            leaf = name.split(".")[-1]  # DDP prefixes names with "module."
            if leaf in scalar_factors:
                params_scalar.setdefault(scalar_factors[leaf], []).append(param)
            # decoupled weight decay ~ decoupling biases & affine params
            elif name.endswith(".bias") or param.ndim == 1 or "norm" in name.lower():
                params_no_decay.append(param)
            else:
                params_decay.append(param)

        lr_init_nom = self.cfg.opt["lr"]["init"]
        param_groups = [
            {"params": params_decay,    "weight_decay": self.cfg.opt["wd"], "lr": lr_init_nom},
            {"params": params_no_decay, "weight_decay": 0.0,                "lr": lr_init_nom},
        ]
        param_groups += [
            {"params": params, "weight_decay": 0.0, "lr": lr_init_nom * factor}
            for factor, params in params_scalar.items()
        ]

        self.opt = torch.optim.AdamW(
            param_groups, 
            lr=lr_init_nom,
            betas=(self.cfg.opt["beta1"], self.cfg.opt["beta2"]),
            eps=self.cfg.opt["eps"],
        )

        lr_init = self.opt.param_groups[0]["lr"]
        decay_factor = self.cfg.opt["lr"]["decay_factor"]
        eta_min = 0.0 if decay_factor is None else lr_init * float(decay_factor)
        total_steps = max(1, math.ceil(self.cfg.sample_volume / self.cfg.batch_size) - math.ceil(self.lr_warmup / self.cfg.batch_size))
        self.lr_sched = CosineAnnealingLR(self.opt, T_max=total_steps, eta_min=eta_min)

    def _update_lr_warmup(self) -> float:
        if self.lr_warmup == 0 or self.n_samps_seen >= self.lr_warmup:
            lr = self.opt.param_groups[0]["lr"]
        else:
            frac = self.n_samps_seen / self.lr_warmup
            lr = self.lr_init_nom * frac
            # per-group bases so the logit scalars keep their scalar_lr_factor through warmup
            for pg, lr_base in zip(self.opt.param_groups, self.lr_sched.base_lrs):
                pg["lr"] = lr_base * frac
        return lr

    @rank0
    def _init_trial_data(self, trial_state):
        if self._resume_state is not None and trial_state is not None:
            return TrialData.resume(ArtifactManager.dpath_trial, trial_state)
        return TrialData(ArtifactManager.dpath_trial)

    @rank0
    def _record_eval(self, eval_metrics, time_eval):
        self.data.eval_metrics = eval_metrics
        self.data.time_eval = time_eval

    @rank0
    def _tracked_logit_scalars(self):
        """{TrialData series -> model attribute} for the logit scalars that get a learning-curve panel:
        a loss's temp whenever it's learnable, its bias only when the loss is BCE-family (inert under
        InfoNCE) and learnable; loss2's only when loss2 is active. Frozen scalars (and non-parameter
        buffers) are left out -- a flat line says nothing."""
        model = self.modelw._unwrapped_model
        tracked = {}
        for tag, cfg_loss, attr_scale, attr_bias in (
            ("1", self.cfg.loss, "logit_scale", "logit_bias"),
            ("2", self.cfg.loss2, "logit_scale2", "logit_bias2"),
        ):
            if tag == "2" and self.cfg.loss2["mix"] == 0.0:
                continue
            if getattr(model, attr_scale).requires_grad:
                tracked[f"temp{tag}"] = attr_scale
            if cfg_loss["crit"] in ("bce", "bif_bce") and getattr(model, attr_bias).requires_grad:
                tracked[f"bias{tag}"] = attr_bias
        return tracked

    def _logit_scalar_values(self):
        # temp series carry tau = exp(-logit_scale) (the quantity temp.init specifies), bias series the raw bias
        model = self.modelw._unwrapped_model
        return {
            key: (-getattr(model, attr)).exp().item() if key.startswith("temp") else getattr(model, attr).item()
            for key, attr in self._logit_scalars_tracked.items()
        }

    def _tracked_targ_stats(self):
        """Batch-stat key prefixes ('targ1'/'targ2') for the target distributions worth curving: phylo
        and tax targets are graded, so their min/max/mean/median carry signal, while sp/mp targets are
        0/1 indicators whose spread says nothing. loss2's only when loss2 is mixed in. Untracked
        branches are dropped before TrialData records them, so they get no learning-curve panel."""
        tracked = set()
        for tag, cfg_loss in (("1", self.cfg.loss), ("2", self.cfg.loss2)):
            if tag == "2" and self.cfg.loss2["mix"] == 0.0:
                continue
            if cfg_loss["targ"] in ("phylo", "tax"):
                tracked.add(f"targ{tag}")
        return tracked

    @rank0
    def _record_train_batch(self, lr, loss, loss_raw, grad_norm_model, delta_norm_model, batch_stats,
                            grad_sum_sim1, grad_sum_sim2):
        # the batch logs still get every stat (incl. the targ point stats sim_targ.log prints); the
        # curve series keep only the histogram, and only for branches whose targets are worth curving
        if batch_stats is not None:
            batch_stats = {
                key: val for key, val in batch_stats.items()
                if not key.startswith("targ") or (key.endswith("_hist") and key[:-len("_hist")] in self._targ_stats_tracked)
            }
        self.data.update_train_batch(
            self.n_samps_seen,
            lr=lr,
            loss_train=loss,
            loss_raw_train=loss_raw,
            grad_norm_model=grad_norm_model,
            delta_norm_model=delta_norm_model,
            batch_stats=batch_stats,
            grad_sum_sim1=grad_sum_sim1,
            grad_sum_sim2=grad_sum_sim2,
            logit_scalars=self._logit_scalar_values(),
        )

    @rank0
    def _save_eval_data(self, dpath, idx_eval):
        ArtifactManager.save_eval_data(dpath, self.data.eval_metrics, idx_eval, self.cfg.n_chkpts, self.n_samps_seen, self.cfg.sample_volume)

    @rank0
    def _write_base_eval(self, entry, eval_metrics):
        # materialize the base-eval cache entry into evals/base/ so base is a uniform member of the
        # eval sequence the render pass sweeps. The per-group metrics files always (every trial records
        # its base eval, written from eval_metrics as eval 0 so base carries the shared fields too);
        # projections + embs only for viz trials -- a non-viz trial computes none of its own projections,
        # so it mustn't inherit the cache's base projections either, and its cache hit is gated on the
        # entry carrying them (require_projections/require_embs). On a fresh base eval the npz files were
        # computed straight into base/ (already on disk, skipped here); only a cache hit writes them.
        dst = ArtifactManager.dpath_trial / "evals" / "base"
        dst.mkdir(parents=True, exist_ok=True)
        ArtifactManager.save_eval_data(dst, eval_metrics, 0, self.cfg.n_chkpts, self.n_samps_seen, self.cfg.sample_volume)
        if self._manif_viz and not (dst / "projections.npz").exists():
            np.savez(dst / "projections.npz", **entry["projections"])
        if self._manif_viz and not (dst / "embs.npz").exists():
            np.savez(dst / "embs.npz", **entry["embs"])

    def _compute_projections_timed(self, eval_bundles, dpath_cache):
        # COLLECTIVE (sharded t-SNE) -- every rank enters; elapsed folds into the viz_compute mean.
        # All ranks spend ~the same wall time here; only rank-0's tracker is persisted.
        # Return the training step's reserved-but-unallocated allocator pool to CUDA before the t-SNE's
        # transient kNN/repulsion chunk buffers allocate. empty_cache is a local allocator op (no
        # collective) so it can't desync.
        torch.cuda.empty_cache()
        with self.time_tracker.measure("viz_compute"):
            compute_projections(eval_bundles["id"], eval_bundles["ood"], dpath_cache, self.cfg.manif_viz,
                                1 << self.cfg.hw.eval["tsne_chunk_log2"])

    def _viz_eval(self, eval_bundles, eval_name):
        """Compute + cache this eval's manifold projections (COLLECTIVE -- every rank must enter) under
        evals/<eval_name>/. Rendering from the cache is done off-process post-trial by the campaign render
        worker (tools/regen_manif_viz.py), so no rank blocks in a collective while rank 0 renders."""
        dpath_eval = ArtifactManager.dpath_trial / "evals" / eval_name
        self._compute_projections_timed(eval_bundles, dpath_eval)

    def _pooled_eval(self):
        """Fit the pooled shared-frame projection over every threshold's cached embeddings (COLLECTIVE --
        sharded t-SNE, every rank must enter) and cache the per-threshold masked blocks under evals/*/,
        rendered post-trial off-process. Runs once at end-of-trial, after every per-eval cache is written."""
        torch.cuda.empty_cache()  # release the training step's reserved pool before the pooled t-SNE buffers
        budget = self.cfg.manif_viz["pooled"]["budget"]
        with self.time_tracker.measure("viz_compute"):
            compute_pooled_projections(ArtifactManager.dpath_trial / "evals", self.cfg.manif_viz, budget,
                                       1 << self.cfg.hw.eval["tsne_chunk_log2"])

    def _save_mid_eval(self, threshold_hit, eval_bundles):
        # _viz_eval -> compute_projections runs the sharded t-SNE collectively, so every rank must
        # enter here; the metrics write (_save_eval_data) is @rank0.
        idx_eval = threshold_hit // self.cfg.chkpt_interval
        eval_name = f"eval{idx_eval}"
        self._save_eval_data(ArtifactManager.dpath_trial / "evals" / eval_name, idx_eval)
        if self._manif_viz:
            self._viz_eval(eval_bundles, eval_name)

    @rank0
    def _print_log_eval(self, header):
        sys.stdout.write('\r')
        sys.stdout.flush()
        PrintLog.eval(
            self.data.eval_metrics,
            self.eval_pipe,
            header=header,
            banner_suffix=f"[{self.cfg.idx_trial}/{self.cfg.n_trials_total}] ({self.cfg.campaign}/{self.cfg.phase}/{self.cfg.dataset}/{self.cfg.arm}/{self.cfg.coord}/{self.cfg.seed})",
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

    def _checkpoint(self, header, idx_batch, final=False):
        # the memory snapshot all-reduces across ranks, so every rank must enter; the writes are @rank0
        mem = self._snapshot_memory()
        self._checkpoint_writes(header, idx_batch, mem, final)

    @rank0
    def _checkpoint_writes(self, header, idx_batch, mem, final):
        if self.eval_enabled:
            self.data.update_eval(self.n_samps_seen)
            self._print_log_eval(header)
            self._save_eval_data(ArtifactManager.dpath_model_checkpoint, self.chkpt_thresh // self.cfg.chkpt_interval - 1)
        ArtifactManager.save_metadata_trial(self.data, self.idx_epoch, self.time_tracker, self.n_samps_seen // self.cfg.samps_per_epoch, self.cfg.n_epochs, self.n_samps_seen, mem)
        ArtifactManager.update_campaign_time()
        ArtifactManager.update_campaign_memory(mem)

        self.data.save()
        ArtifactManager.save_train_state(self, idx_batch)
        ArtifactManager.save_trial_state(self.data)
        if final or self.cfg.dev["plot_every"] == "chkpt":
            plot_metrics(self.data, ArtifactManager.dpath_trial, self.eval_pipe.nshot_bucket_names if self.eval_enabled else [], self.cfg.samps_per_epoch)

    def _step_train(self, imgs_sb, texts_sb, class_encs_sb, targ_data_sb):
        if self.cfg.hw.loss_chunk_size is not None:
            # tiled path: encoder forward, loss, AND backward happen inside (representation gradients),
            # so no loss.backward() here. Autocast + DDP grad sync are handled internally. The sims slot
            # already carries the (grad_sum_sim1, grad_sum_sim2) floats, accumulated tile-by-tile.
            loss, loss_raw, embs_img_b, embs_txt_b, logits, _, batch_stats, grad_sum_sims = self.modelw.batch_step_chunked(
                imgs_sb, texts_sb, class_encs_sb, targ_data_sb
            )
            return loss, loss_raw, embs_img_b, embs_txt_b, logits, batch_stats, grad_sum_sims
        if self.cfg.hw.mixed_prec:
            with autocast(device_type=self.cfg.device.type, dtype=torch.bfloat16):
                loss, loss_raw, embs_img_b, embs_txt_b, logits, _, batch_stats, sims = self.modelw.batch_step(
                    imgs_sb, texts_sb, class_encs_sb, targ_data_sb
                )
        else:
            loss, loss_raw, embs_img_b, embs_txt_b, logits, _, batch_stats, sims = self.modelw.batch_step(
                imgs_sb, texts_sb, class_encs_sb, targ_data_sb
            )
        loss.backward()
        if not self._batch_diag["sim_grad_sums"]:  # no grads were retained on the sims
            return loss, loss_raw, embs_img_b, embs_txt_b, logits, batch_stats, (None, None)
        with torch.no_grad():
            # .float(): the retained grads are bf16 under mixed_prec, and casting the SUM result back
            # to bf16 quantizes it (~3 significant digits)
            # aggregate over a loss's branch tuple: one branch non-bifurcated; for bif_bce the two
            # un-halved branches each carry the full incoming grad, so the aggregate reads 2x the
            # non-bifurcated sum(dL/dsim) -- consistent with the 2x loss reading. A branch with no
            # retained grad (e.g. the t2i branch under a frozen text tower) contributes nothing.
            def sim_grad_sum(branches):
                return sum(s.grad.float().sum().item() for s in branches if s.grad is not None)
            grad_sum_sims = (
                sim_grad_sum(sims[0]),
                sim_grad_sum(sims[1]) if sims[1] is not None else None,
            )
        return loss, loss_raw, embs_img_b, embs_txt_b, logits, batch_stats, grad_sum_sims

    def _step_optimizer(self):
        """Take the optimizer step and return ||delta theta||, the L2 norm of the resulting parameter
        update. Distinct from ||grad theta||: Adam rescales per parameter, so the step length is set
        by the LR and the moment ratio rather than by the raw gradient magnitude -- the two can move
        in opposite directions. Measured against a pre-step snapshot, which is optimizer-agnostic
        (no reliance on AdamW's internals) at the cost of one extra copy of the trainable params;
        the buffers are allocated once and reused, so there's no per-step allocation churn.
        With dev.batch_diagnostics.delta_norm_model off the snapshot/delta is skipped entirely (returns None)."""
        if not self._batch_diag["delta_norm_model"]:
            self.opt.step()
            return None
        params = [p for p in self.modelw.model.parameters() if p.requires_grad]
        if self._params_prev is None:
            self._params_prev = [torch.empty_like(p) for p in params]
        with torch.no_grad():
            for buf, p in zip(self._params_prev, params):
                buf.copy_(p.detach())
        self.opt.step()
        with torch.no_grad():
            total = torch.zeros((), device=params[0].device)
            for buf, p in zip(self._params_prev, params):
                total += (p.detach() - buf).pow(2).sum()  # accumulate on-device; single host sync at the end
        return total.sqrt().item()

    def train(self):
        try:

            if self._resume_state is None:
                mem = self._snapshot_memory()  # COLLECTIVE -- every rank must enter
                ArtifactManager.save_metadata_trial(self.data, self.idx_epoch, self.time_tracker, self.n_samps_seen // self.cfg.samps_per_epoch, self.cfg.n_epochs, self.n_samps_seen, mem, init_flag=True)
                if self.eval_enabled:
                    PrintLog.texts_eval(self.eval_pipe)

            # BASE EVAL

            if self._resume_state is None and self.eval_enabled:
                cached = ArtifactManager.load_base_eval_cache(self.cfg, require_projections=self._manif_viz, require_embs=self._manif_viz)  # @rank0; None elsewhere
                # single-source the hit/miss decision: a concurrent campaign's cache write landing between
                # independent per-rank reads would split the branch, and the miss branch enters collective
                # ops (evaluate / sharded t-SNE) that every rank must join. Only the small metrics dict is
                # broadcast; the bulky projections/embs stay on rank 0, the only rank that writes base/.
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
                        collect_eval_bundles=self._manif_viz,
                    )
                    if self._manif_viz:
                        self._viz_eval(eval_bundles, "base")  # projections (+ embs if pooled) straight into evals/base
                    entry = ArtifactManager.save_base_eval_cache(self.cfg, eval_metrics)  # rank0 gets the entry back; None elsewhere
                self._write_base_eval(entry, eval_metrics)  # base -> evals/base (uniform member of the eval sequence)
                if time_eval is not None:
                    self.time_tracker.add("eval", time_eval)
                self._record_eval(eval_metrics, time_eval)
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

            for _ in range(self.cfg.n_passes - self.idx_epoch):
                self.timer_train.start()
                self.idx_epoch += 1

                epoch_first, epoch_last = pass_epoch_span(self.cfg, self.idx_epoch)
                PrintLog.batch_logs_epoch_header(epoch_first, epoch_last, self.cfg.n_epochs)
                epoch_label = f"{epoch_last}" if epoch_first == epoch_last else f"{epoch_first}-{epoch_last}"

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
                    desc=f"Train ({epoch_label}/{self.cfg.n_epochs})",
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

                    imgs_sb = self.modelw.prep_imgs(imgs_sb)  # uint8 -> device, fp32, normalized
                    class_encs_sb = class_encs_sb.to(self.cfg.device, non_blocking=True)
                    B = imgs_sb.size(0) * dist.get_world_size()
                    self.n_samps_seen += B

                    lr = self._update_lr_warmup() if self.lr_warmup > 0 else self.opt.param_groups[0]["lr"]

                    self.opt.zero_grad(set_to_none=True)
                    loss, loss_raw, embs_img_b, embs_txt_b, logits, batch_stats, grad_sum_sims = self._step_train(
                        imgs_sb,
                        texts_sb,
                        class_encs_sb,
                        targ_data_sb,
                    )
                    grad_norm_model = None
                    if self._batch_diag["grad_norm_model"]:
                        with torch.no_grad():
                            grad_norm_model = model_grad_l2_norm(self.modelw.model)
                    # the step is taken before logging so the line can carry the update norm too;
                    # grads survive it (zero_grad only runs at the top of the next iteration)
                    delta_norm_model = self._step_optimizer()
                    PrintLog.batch(idx_batch, lr, loss, embs_img_b, embs_txt_b, logits, self.modelw.model,
                                   grad_norm_model, delta_norm_model, batch_stats, self._batch_diag)

                    if self.n_samps_seen >= self.lr_warmup:
                        self.lr_sched.step()

                    with torch.no_grad():
                        loss = loss.detach().item()
                        loss_raw = loss_raw.detach().item()
                        loss_mean.update(loss)
                        loss_raw_mean.update(loss_raw)
                        self.n_batches_seen += 1

                    self._record_train_batch(lr, loss, loss_raw, grad_norm_model, delta_norm_model, batch_stats,
                                             grad_sum_sims[0], grad_sum_sims[1])

                    if self.n_samps_seen >= self.chkpt_thresh:
                        pbar.clear()
                        self.timer_train.stop()

                        while self.n_samps_seen >= self.chkpt_thresh:
                            threshold_hit = self.chkpt_thresh
                            self.chkpt_thresh += self.cfg.chkpt_interval

                        # skip the last train-time eval+checkpoint (idx >= n_chkpts) -- the final
                        # eval+checkpoint right below cover eval<n_chkpts> at sample_volume
                        if threshold_hit // self.cfg.chkpt_interval < self.cfg.n_chkpts:
                            # TRAIN-TIME EVAL
                            if self.eval_enabled:
                                eval_metrics, time_eval, eval_bundles = self.eval_pipe.evaluate(
                                    self.modelw,
                                    loss_flag=True,
                                    collect_eval_bundles=self._manif_viz,
                                )
                                self.time_tracker.add("eval", time_eval)
                                self._record_eval(eval_metrics, time_eval)
                                self._save_mid_eval(threshold_hit, eval_bundles)
                            self._checkpoint(
                                header=f"{threshold_hit:,}",
                                idx_batch=idx_batch,
                            )
                            ArtifactManager.save_rng_states(self._local_rank)
                            dist.barrier()

                        self.timer_train.start()
                        pbar.refresh()

                    if self.n_samps_seen >= self.samps_stop:
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
                    epoch_first,
                    epoch_last,
                    self.cfg.n_epochs,
                )

                if self.n_samps_seen >= self.samps_stop:
                    break  # chkpt_stop reached mid-pass (trainval phase): no further passes

            # FINAL EVAL

            if self.eval_enabled:
                eval_metrics, time_eval, eval_bundles = self.eval_pipe.evaluate(
                    self.modelw,
                    loss_flag=True,
                    collect_eval_bundles=self._manif_viz,
                )
                self.time_tracker.add("eval", time_eval)
                self._record_eval(eval_metrics, time_eval)
                self._save_eval_data(ArtifactManager.dpath_eval_final, self.cfg.n_chkpts)
                if self._manif_viz:
                    self._viz_eval(eval_bundles, f"eval{self.cfg.n_chkpts}")  # COLLECTIVE compute+cache; rendered post-trial off-process
            self._checkpoint(
                header="Final",
                idx_batch=-1,
                final=True,
            )
            ArtifactManager.save_rng_states(self._local_rank)
            dist.barrier()  # all per-eval caches (incl. final) now on disk -> safe to pool them

            if self._pooled_manif_viz:
                self._pooled_eval()  # COLLECTIVE: pooled shared-frame projection over all cached embeddings

            PrintLog.trial_time(self.data)

        finally:
            PrintLog.close_logs()

def run_training(cfg):
    local_gpu_rank, device = setup_ddp(cfg.hw.pg_timeout)
    start_ram_peak_tracker(cfg.hw.ram_poll_interval)
    cfg.device = device  # set local device
    seed_libs(cfg.seed)
    apply_backend_flags(cfg.hw)
    configure_phylo_targs(cfg.split, cfg.train_pt, cfg.batch_size, cfg.htarg_shuf, cfg.seed)

    ArtifactManager.set_paths(cfg)
    ArtifactManager.create_trial_dirs()
    dist.barrier()  # ensure rank0 finishes creating dirs before other ranks proceed
    ArtifactManager.save_metadata_coord(cfg)
    if cfg.dev["logging"]:
        PrintLog.create_logs(ArtifactManager.dpath_trial / "logs")
    PrintLog.init_train(cfg)

    modelw = VLMWrapper.build(cfg, verbose=(dist.get_rank() == 0))
    modelw.crit1 = Criterion.build(cfg.loss, cfg.dataset, cfg.split, cfg.train_pt, device, cfg.batch_size)
    modelw.crit2 = Criterion.build(cfg.loss2, cfg.dataset, cfg.split, cfg.train_pt, device, cfg.batch_size) if cfg.loss2["mix"] != 0.0 else None

    resume_state = None
    trial_state = None
    if ArtifactManager.resuming:
        resume_state = ArtifactManager.load_train_state()
        modelw._unwrapped_model.load_state_dict(resume_state["model"])
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
    if cfg.phase == "trainval":
        # the trainval phase's product: the weights at the qual-selected checkpoint (chkpt_stop); it runs no evals,
        # so there is nothing to select or aggregate
        ArtifactManager.save_model(train_pipe.modelw)
    else:
        cfg_stats = get_config_stats()  # stats.yaml is render-time only: read live, not frozen into the campaign
        # reselects this coord/dataset's checkpoint over ALL its completed trials (this one included) and
        # rewrites their evals/_best/, so the aggregates below see the current selection
        update_chkpt_selection(cfg_stats.spread_type)
        update_metric_stats(cfg_stats.spread_type)
        # every cross-coord table/plot refreshes only at the end of its own seed cycle -- once this seed has a
        # completed trial in every coord of the arm (arm_stats), every arm x coord of the dataset
        # (dataset_stats), and the whole matrix (campaign_stats) -- so it is never rendered from a mix of
        # coords reselected against different trial counts
        if arm_sweep_complete(cfg.seed, cfg.dataset, cfg.arm):
            update_arm_stats(cfg.dataset, cfg.arm, cfg_stats.spread_type, cfg_stats.bold_high, cfg_stats.ordered, cfg_stats.heatmap,
                             cfg_stats.supp_scores)
        if dataset_sweep_complete(cfg.seed, cfg.dataset):
            update_dataset_stats(cfg.dataset, cfg_stats.spread_type, cfg_stats.bold_high, cfg_stats.ordered, cfg_stats.heatmap,
                                 cfg_stats.supp_scores)
        if seed_sweep_complete(cfg.seed):
            update_campaign_stats(cfg_stats.spread_type, cfg_stats.bold_high, cfg_stats.ordered, cfg_stats.heatmap, cfg_stats.supp_scores,
                                  cfg_stats.overrides)

    cleanup_ddp()
