import re
from pathlib import Path
from typing import List, Any, Dict, Optional

import torch

from utils.ddp import rank0
from utils.utils import atomic_write, load_json


class PrintLog:

    logging = False
    log_batch_general = None
    log_batch_grad_norm = None
    log_batch_scale_bias = None
    log_batch_similarity = None
    log_batch_kl = None
    log_epoch = None
    log_eval = None
    log_init = None
    log_text_train = None
    log_text_eval = None

    @staticmethod
    @rank0
    def create_logs(dpath_logs, eval_logs):
        PrintLog.logging = True
        dpath_batch_logs = dpath_logs / "batch"
        dpath_batch_logs.mkdir(parents=True, exist_ok=True)
        PrintLog.log_batch_general = open(dpath_batch_logs / "general.log", "a", buffering=1)
        PrintLog.log_batch_grad_norm = open(dpath_batch_logs / "grad_norm.log", "a", buffering=1)
        PrintLog.log_batch_scale_bias = open(dpath_batch_logs / "scale_bias.log", "a", buffering=1)
        PrintLog.log_batch_similarity = open(dpath_batch_logs / "sim_targ.log", "a", buffering=1)
        PrintLog.log_batch_kl = open(dpath_batch_logs / "kl.log", "a", buffering=1)
        PrintLog.log_epoch = open(dpath_logs / "epoch.log", "a", buffering=1)
        PrintLog.log_init = open(dpath_logs / "init.log", "a", buffering=1)
        PrintLog.log_text_train = open(dpath_logs / "text_train.log", "a", buffering=1)
        if eval_logs:  # the trainval phase runs no evals: nothing ever writes these
            PrintLog.log_eval = open(dpath_logs / "eval.log", "a", buffering=1)
            PrintLog.log_text_eval = open(dpath_logs / "text_eval.log", "a", buffering=1)

    @staticmethod
    def manifest(dpath_phase: Path, trials: List[tuple], seeds: List[int], in_progress: Optional[tuple]) -> None:
        """Write <dpath_phase>/manifest.log: a human-readable snapshot bucketing every planned trial
        into Failed / Completed / In Progress / Queued. Regenerated at campaign kickoff and at each trial's
        start and finish so it tracks progress. `trials` is the full planned set of (dataset, arm, coord, seed)
        tuples in launch order, `seeds` the phase's seed sweep (a trial's dir is _trial/<n>, n its seed's 1-based index
        there); `in_progress` is the trial currently running (None when nothing is). A trial
        is Completed if its metadata says so, In Progress if it's the running one, Failed if it left an
        error.log behind, else Queued. Completed and Failed entries carry the trial's recorded wall-clock
        as 'trial_id --- D-HH:MM:SS', dash-aligned per section; a Completed trial that kill_thresh cut
        short additionally carries ' --- ABANDON SHIP' (metadata's killed flag); Failed entries additionally carry sample
        progress as ' --- E/N' (epoch index / n_epochs) and the failure type as
        ' --- RAM|VRAM|Other|Mixed' (the aggregate cause over the fatal retry loop's crashes, parsed
        from error.log's 'failure=' marker written by campaign_runner._log_trial_error). A trial that
        failed before ever writing metadata shows 'n/a'. Trials run sequentially, so the filesystem can't tell a running trial
        from a crashed one (both leave chkpts/in_progress); the runner passes the live trial in explicitly."""

        def fmt_trial_time(metadata_trial):
            seconds = int(float(metadata_trial["runtime"]["trial"]))
            days, seconds = divmod(seconds, 86400)
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"

        def fmt_trial_progress(metadata_trial):
            progress = metadata_trial["progress"]
            return f"{progress['epoch']}/{progress['n_epochs']}"

        buckets: Dict[str, List[Any]] = {"Failed": [], "Completed": [], "In Progress": [], "Queued": []}
        for trial in trials:
            dataset, arm, coord, seed = trial
            dpath_trial = dpath_phase / "_dataset" / dataset / "_arm" / arm / "_coord" / coord / "_trial" / str(seeds.index(seed) + 1)
            trial_id = f"{dataset}/{arm}/{coord}/{seed}"
            fpath_metadata_trial = dpath_trial / "trial_metadata.json"
            metadata_trial = load_json(fpath_metadata_trial) if fpath_metadata_trial.exists() else None
            if metadata_trial is not None and metadata_trial["complete"]:
                killed = metadata_trial["killed"]
                buckets["Completed"].append((trial_id, fmt_trial_time(metadata_trial) + (" --- ABANDON SHIP" if killed else "")))
            elif trial == in_progress:
                buckets["In Progress"].append(trial_id)
            elif (dpath_trial / "error.log").exists():
                failure = re.search(r"failure=(\w+)", (dpath_trial / "error.log").read_text()).group(1)
                if metadata_trial is None:
                    buckets["Failed"].append((trial_id, f"n/a --- {failure}"))
                else:
                    buckets["Failed"].append((trial_id, f"{fmt_trial_time(metadata_trial)} --- {fmt_trial_progress(metadata_trial)} --- {failure}"))
            else:
                buckets["Queued"].append(trial_id)

        section_emojis = {"Failed": "❌", "Completed": "✅", "In Progress": "🏃", "Queued": "⏳"}
        lines: List[str] = []
        for title in ("Failed", "Completed", "In Progress", "Queued"):
            lines.append(f"{section_emojis[title]} {title}:")
            if title in ("Failed", "Completed"):
                if buckets[title]:
                    lines.append(PrintLog._dash_aligned_lines(buckets[title]))
            else:
                lines.extend(buckets[title])
            lines.append("")
        with atomic_write(dpath_phase / "manifest.log", "w", encoding="utf-8") as f:  # a reader never sees it half-rewritten
            f.write("\n".join(lines).rstrip() + "\n")

    @staticmethod
    @rank0
    def texts(texts_sb):
        if PrintLog.logging:
            text_printout = "\n".join(texts_sb)
            PrintLog.log_text_train.write(text_printout)

    @staticmethod
    @rank0
    def texts_eval(eval_pipe):
        if PrintLog.logging:
            texts_by_partition = eval_pipe.get_eval_texts()
            lines = []
            for partition, texts in texts_by_partition.items():
                lines.append(f"[{partition}]")
                lines.extend(texts)
                lines.append("")
            PrintLog.log_text_eval.write("\n".join(lines))
            PrintLog.wrote_text_eval = True

    @staticmethod
    def _make_epoch_header(epoch_first, epoch_last, n_epochs, width=75):
        label = f"Epoch {epoch_last}" if epoch_first == epoch_last else f"Epochs {epoch_first}-{epoch_last}"
        return f"{f' {label}/{n_epochs} ':#^{width}}"

    @staticmethod
    @rank0
    def batch_logs_epoch_header(epoch_first, epoch_last, n_epochs):
        if PrintLog.logging:
            header_epoch = PrintLog._make_epoch_header(epoch_first, epoch_last, n_epochs) + "\n"
            PrintLog.log_batch_general.write(header_epoch)
            PrintLog.log_batch_grad_norm.write(header_epoch)
            PrintLog.log_batch_scale_bias.write(header_epoch)
            PrintLog.log_batch_similarity.write(header_epoch)
            PrintLog.log_batch_kl.write(header_epoch)

    @staticmethod
    @rank0
    def epoch(time_train, time_train_avg, time_data_wait_ranks, loss_train_avg, loss_train_raw_avg, n_samps_seen, epoch_first, epoch_last, n_epochs):

        SECTION_WIDTH = 66

        lines_epoch = [
            PrintLog._make_epoch_header(epoch_first, epoch_last, n_epochs, width=SECTION_WIDTH),
            PrintLog._dash_aligned_lines((
                ("Loss", f"{loss_train_avg:.3e}"),
                ("Raw Loss", f"{loss_train_raw_avg:.3e}"),
                ("Samples Seen", f"{n_samps_seen:,}"),
                ("Time", f"{time_train:.2f} s (avg: {time_train_avg:.2f} s)"),
                ("Data Wait", " | ".join(f"r{i}: {w:.2f} s" for i, w in enumerate(time_data_wait_ranks))),
            )),
            f"{'':#^{SECTION_WIDTH}}",
            "",
        ]
        print("\n".join(lines_epoch))
        if PrintLog.logging:
            PrintLog.log_epoch.write("\n".join(lines_epoch))

    @staticmethod
    @rank0
    def trial_time(data):
        # data is the trial's TrialData (None off-rank0; the @rank0 body never runs there); the timer
        # accumulates across resumes, so this is the trial's total wall-clock
        seconds = int(data.timer_trial.get_elapsed_time())
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        print(f"Total Trial Time: {days}-{hours:02}:{minutes:02}:{seconds:02}")

    @staticmethod
    @rank0
    def batch_logit_scalars(model):
        """The scale_bias.log line's values, {s / b -> float} for the logit scalars the model carries
        (frozen ones included; the scale as the raw exp(logit_scale), which sits above 100 while
        logits.scale.clamp holds). Read by the train loop BEFORE the batch's optimizer step, so the line
        carries the values the batch's grad_norm / sim_targ lines were computed under, then handed to
        .batch, which writes after the step (its grad_norm line carries the update norm)."""
        values = {}
        for key, attr in (("s", "logit_scale"), ("b", "logit_bias"), ("s2", "logit_scale2"), ("b2", "logit_bias2")):
            scalar = getattr(model.module, attr, None)
            if scalar is None:
                continue
            if key.startswith("s"):
                values[key] = scalar.detach().exp().item()
            else:
                values[key] = scalar.detach().item() if isinstance(scalar, torch.Tensor) else float(scalar)
        return values

    @staticmethod
    @rank0
    def batch(idx_batch, lr, loss_batch, embs_img_b, embs_txt_b, logits, logit_scalars, grad_norm_model,
              delta_norm_model, batch_stats, diag):

        def tensor_grad_l2_norm(x: torch.Tensor | None) -> float:
            if x is None:
                return float("nan")
            if x.grad is None:
                return float("nan")
            return x.grad.detach().pow(2).sum().sqrt().item()

        def branch_grad_l2_norm(branches) -> float:
            # aggregate over the branch tuple ([img-row, txt-col] each): norm of the
            # elementwise-summed branch grads. One branch non-bifurcated (== its grad norm); for
            # bif_bce the two un-halved branches each carry the full incoming grad, so the
            # aggregate reads 2x the non-bifurcated norm -- consistent with the 2x loss reading.
            grads = [b.grad.float() for b in branches if b.grad is not None]
            if not grads:
                return float("nan")
            return sum(grads).pow(2).sum().sqrt().item()

        # each enabled reporting.batch_diagnostics component contributes its own grad_norm.log fields; with
        # every component off the line has none and is skipped outright (as is the sim_targ line when
        # sim_targ_stats is off -- batch_stats arrives None)
        fields_grad_norm = []
        if diag["emb_logit_grads"]:
            fields_grad_norm.append(f"img={tensor_grad_l2_norm(embs_img_b):.4e}")
            fields_grad_norm.append(f"txt={tensor_grad_l2_norm(embs_txt_b):.4e}")
            # logits: the branch tuple, or None on the chunked path (loss_chunk_size != null), where no
            # full logit matrix exists, so the diagnostic is structurally unavailable -- omit the field
            # rather than log nan
            if logits is not None:
                fields_grad_norm.append(f"logit={branch_grad_l2_norm(logits):.4e}")
        if diag["grad_norm_model"]:
            fields_grad_norm.append(f"model={grad_norm_model:.4e}")
        if diag["delta_norm_model"]:
            fields_grad_norm.append(f"step={delta_norm_model:.4e}")  # ||delta theta||: the update the optimizer actually applied
        line_grad_norm = " ".join(fields_grad_norm)

        # the pre-step read (batch_logit_scalars), so the line sits on the batch's own alpha / bias
        line_logits_param = " ".join(f"{key}={value:.4e}" for key, value in logit_scalars.items())

        batch_str = f"batch {idx_batch}:"
        if PrintLog.logging:
            PrintLog.log_batch_general.write(
                f"{batch_str:<10} "
                f"lr={lr:.2e} "
                f"loss={loss_batch:.4e} "
                # a unitless loss blend's effective lambda (batch_stats: needs reporting.batch_diagnostics.sim_targ_stats)
                + (f"lambda_eff={batch_stats['lambda_eff']:.4f} " if batch_stats is not None and "lambda_eff" in batch_stats else "")
                + f"\n"
            )
            if line_grad_norm:
                PrintLog.log_batch_grad_norm.write(
                    f"{batch_str:<10} "
                    f"{line_grad_norm}"
                    f"\n"
                )
            PrintLog.log_batch_scale_bias.write(
                f"{batch_str:<10} "
                f"{line_logits_param}"
                f"\n"
            )
            if batch_stats is not None:
                stat_groups = ["sim", "targ"]
                PrintLog.log_batch_similarity.write(
                    f"{batch_str:<10} "
                    + " | ".join(
                        f"{group}: min={batch_stats[f'{group}_min']: .4f} max={batch_stats[f'{group}_max']: .4f} "
                        f"med={batch_stats[f'{group}_median']: .4f} mean={batch_stats[f'{group}_mean']: .4f}"
                        for group in stat_groups
                    )
                    + "\n"
                )
            # the InfoNCE KL decomposition, the kl.log line's five values -- the raw D_KL(y || p), its
            # structural part, the two representational parts summed (E_R = E_UR + E_IR, derived here as
            # it is in the KL panels) and each of them. InfoNCE only (infonce_batch_stats), so a
            # BCE-family loss's line is skipped like its panels are
            if batch_stats is not None and "kl" in batch_stats:
                PrintLog.log_batch_kl.write(
                    f"{batch_str:<10} "
                    f"KL={batch_stats['kl']:.4e} "
                    f"E_U={batch_stats['kl_u']:.4e} "
                    f"E_R={batch_stats['kl_ur'] + batch_stats['kl_ir']:.4e} "
                    f"E_UR={batch_stats['kl_ur']:.4e} "
                    f"E_IR={batch_stats['kl_ir']:.4e}"
                    f"\n"
                )

    @staticmethod
    @rank0
    def eval(
        eval_metrics: Dict[str, Any],
        eval_pipe,
        eval_groups: Dict[str, str],
        header: Optional[str] = None,
        banner_suffix: Optional[str] = None,
        n_samps_seen: Optional[int] = None,
        time_eval: Optional[float] = None,
        time_eval_avg: Optional[float] = None,
    ) -> None:
        
        SECTION_WIDTH = 77

        def _format_composite_block(title: str, map_scores: Dict[str, Any]) -> str:
            pairs = [
                ("All", f"{map_scores['all']:.4f}"),
                ("I2I", f"{map_scores['i2i']:.4f}"),
                *((p.upper(), f"{map_scores[p]:.4f}") for p in partitions),
            ]
            return f"{title:-^{SECTION_WIDTH}}\n" + PrintLog._dash_aligned_lines(pairs) + "\n"
        
        partitions = eval_pipe.partitions

        lines_comp = f"{' ID/OOD Eval ':=^{SECTION_WIDTH}}\n"

        # one block per eval group in play (reporting.yaml's `eval`), under its reported name
        for group_key, group_name in eval_groups.items():
            lines_comp += _format_composite_block(
                f" {group_name} Composite mAP ",
                eval_metrics["scores"][group_key]["comp"]["map"],
            )

        loss_pairs = [
            (partition.upper(), f"{eval_metrics['loss_raw'][partition]:.3e}")
            for partition in partitions
            if eval_metrics["loss_raw"][partition] is not None
        ]
        lines_loss = (f"{' Loss ':-^{SECTION_WIDTH}}\n" + PrintLog._dash_aligned_lines(loss_pairs) + "\n") if loss_pairs else ""

        lines_sim_targ = ""
        if eval_metrics["sim_targ"]["comb"] is not None:  # the point stats over every partition's chunks together
            def _fmt_stats(prefix):
                stats = eval_metrics["sim_targ"]["comb"]
                return (
                    f"min={stats[f'{prefix}_min']: .4f} max={stats[f'{prefix}_max']: .4f} "
                    f"med={stats[f'{prefix}_median']: .4f} mean={stats[f'{prefix}_mean']: .4f}"
                )
            sim_targ_pairs = [
                ("Sim", _fmt_stats("sim")),
                ("Targ", _fmt_stats("targ")),
            ]
            lines_sim_targ = f"{' Sim/Targ ':-^{SECTION_WIDTH}}\n" + PrintLog._dash_aligned_lines(sim_targ_pairs) + "\n"

        lines_info = ""
        info = []
        if n_samps_seen is not None:
            info.append(("Samples Seen", f"{n_samps_seen:,}"))
        if time_eval is not None:
            time_str = f"{time_eval:.2f} s"
            if time_eval_avg is not None:
                time_str += f" (avg: {time_eval_avg:.2f} s)"
            info.append(("Time", time_str))
        if len(info) > 0:
            lines_info += f"{' Info ':=^{SECTION_WIDTH}}\n"
            lines_info += PrintLog._dash_aligned_lines(info) + "\n"

        eval_header = f" Eval {header} " if header is not None else " Eval "
        banner = f"{eval_header:#^{SECTION_WIDTH}}"
        if banner_suffix is not None:
            banner += f" {banner_suffix}"
        eval_printout = (
            f"{banner}\n"
            f"{lines_comp}"
            f"{lines_loss}"
            f"{lines_sim_targ}"
            f"{lines_info}"
            f"{'':#^{SECTION_WIDTH}}\n"
        )
        print(eval_printout)
        if PrintLog.logging:
            PrintLog.log_eval.write(eval_printout)

    @staticmethod
    @rank0
    def init_train(cfg_train):

        lines = [
            "",
            PrintLog._dash_aligned_lines((
                ("Campaign", cfg_train.campaign),
                ("Phase",    cfg_train.phase),
                ("Arm",      cfg_train.arm),
                ("Coord",    cfg_train.coord),
                ("Dataset",  cfg_train.dataset),
                ("Split",    cfg_train.split["split"]),
                ("Seed",     cfg_train.seed),
            )),
            "",
            PrintLog._dash_aligned_lines((
                ("Epochs",      f"{cfg_train.n_epochs} ({cfg_train.sample_volume:,} samples)"),
                ("Checkpoints", f"{cfg_train.n_chkpts} (every {cfg_train.chkpt_interval:,} samples)"),
                ("Batch Size",  f"{cfg_train.batch_size}"),
                ("DV Batching", f"{cfg_train.dv_batching}"),
            )),
            "",
            "=== Architecture ===",
            PrintLog._dash_aligned_lines((
                ("Model Type", cfg_train.model["arch"]["model_type"]),
                ("Non-Causal", cfg_train.model["arch"]["clip"]["non_causal"]),
            )),
            "",
        ]

        lines.extend([  # freeze block
            "=== Freeze ===",
            PrintLog._dash_aligned_lines((
                ("Image", cfg_train.model["freeze"]["image"]),
                ("Text",  cfg_train.model["freeze"]["text"]),
            )),
            "",
        ])

        lines.extend(PrintLog._format_loss_block(cfg_train.loss, cfg_train.loss["loss1"], cfg_train.loss["loss2"]))  # loss block

        lines.extend(PrintLog._format_aug_block(cfg_train.aug_cfg))  # image augmentation block

        lines.extend([  # text templates block
            "=== Text Templates ===",
            PrintLog._dash_aligned_lines((
                ("Train", cfg_train.text_template["train"]),
                ("Eval",  cfg_train.text_template["eval"]),
            )),
            "",
        ])

        lines.extend([  # optimization block
            "=== Optimization ===",
            "LR",
            PrintLog._dash_aligned_lines([
                ("- Init",         cfg_train.lr["init"]),
                ("- Decay Factor", cfg_train.lr["decay_factor"]),
                ("- Warmup",       f"{cfg_train.lr['warmup']:,}"),
            ]),
            PrintLog._dash_aligned_lines([
                ("Weight Decay", cfg_train.opt['wd']),
                ("β1", cfg_train.opt['beta1']),
                ("β2", cfg_train.opt['beta2']),
                ("ε",  cfg_train.opt['eps']),
            ]),
            "",
        ])

        lines.extend(PrintLog._format_hw_block(cfg_train))  # hardware block

        print(*lines, sep="\n")
        if PrintLog.logging:
            PrintLog.log_init.write("\n".join(lines) + "\n")

    @staticmethod
    def _format_hw_block(cfg_train):
        map_chunk_size = cfg_train.hw.eval["map_chunk_size"]
        lines_hw = [
            "=== Hardware ===",
            PrintLog._dash_aligned_lines((
                ("Num. GPUs",       cfg_train.n_gpus),
                ("Num. CPUs",       cfg_train.n_cpus),
                ("RAM",             f"{cfg_train.ram} GB"),
                ("Num. Workers",    cfg_train.n_workers),
                ("Prefetch Factor", cfg_train.prefetch_factor),
                ("Device",          cfg_train.device),
            )),
            "Chunk Size",
            PrintLog._dash_aligned_lines((
                ("- img-to-img mAP",  f"{map_chunk_size['img2img']:,}"),
                ("- cross-modal mAP", f"{map_chunk_size['cross_modal']:,}"),
            )),
            "",
        ]
        return lines_hw

    @staticmethod
    def _format_loss_block(cfg_loss: dict, cfg_loss1: dict, cfg_loss2: dict) -> list[str]:

        lines = ["=== Loss ==="]
        # the live target specs with their blend weights (utils.loss.targ_specs)
        lambda_ = cfg_loss["blend"]["lambda"]
        targs = [(w, cfg_targ["targ"]) for w, cfg_targ in ((1.0 - lambda_, cfg_loss1), (lambda_, cfg_loss2)) if w != 0.0]
        info = [
            ("Crit", cfg_loss["crit"]),
            ("Sim", cfg_loss["sim"]),
            ("Targs", " + ".join(f"{w:g} {targ}" for w, targ in targs) if len(targs) > 1 else targs[0][1]),
            *((("Blend Type", cfg_loss["blend"]["type"]),) if len(targs) > 1 else ()),
            *((("Shared Logit Scalars", cfg_loss["logits"]["shared"]),) if len(targs) > 1 and cfg_loss["blend"]["type"] == "loss" else ()),
            ("Unitless", cfg_loss["unitless"]),
        ]
        lines.append(PrintLog._dash_aligned_lines(info))

        wting = cfg_loss.get("wting", {}).get("cls_imb", {}).get("type") is not None
        if wting:
            cw_type = cfg_loss["wting"]["cls_imb"]["type"]
            lines_cw = [
                "Class Weighting",
                PrintLog._dash_aligned_lines((
                    ("- Type", cw_type),
                    *((("- gamma", cfg_loss["wting"]["cls_imb"]["inv_freq"]["gamma"]),) if cw_type == "inv_freq" else ()),
                    *((("- beta", cfg_loss["wting"]["cls_imb"]["class_bal"]["beta"]),) if cw_type == "class_bal" else ()),
                )),
            ]
            lines.extend(lines_cw)

        focal = cfg_loss.get("wting", {}).get("focal", {}).get("gamma", 0.0) != 0.0
        if focal:
            lines_focal = [
                "Focal",
                PrintLog._dash_aligned_lines((
                    ("- gamma", cfg_loss["wting"]["focal"]["gamma"]),
                )),
            ]
            lines.extend(lines_focal)

        if cfg_loss.get("wting", {}).get("bce", {}).get("dsmr", False):
            lines.append("DSMR Enabled")

        cfg_logits = cfg_loss["logits"]
        lines_logits = [
            "Logits",
            PrintLog._dash_aligned_lines((
                ("- Scalar LR Factor", cfg_logits["scalar_lr_factor"]),
                ("- Scale Init",   cfg_logits["scale"]["init"]),
                ("- Scale Freeze", cfg_logits["scale"]["freeze"]),
                ("- Scale Clamp",  cfg_logits["scale"]["clamp"]),
                ("- Center",       cfg_logits["bce"]["center"]),
                ("- Bias Init",    cfg_logits["bce"]["bias"]["init"]),
                ("- Bias Freeze",  cfg_logits["bce"]["bias"]["freeze"]),
            )),
        ]
        lines.extend(lines_logits)

        lines.append("")

        return lines

    @staticmethod
    def _format_aug_block(aug: dict) -> list[str]:
        lines = ["=== Image Augmentation ==="]

        lines.append("RRCrop")
        lines.append(PrintLog._dash_aligned_lines([
            ("- Scale", (aug["rrcrop"]["scale_min"], 1.0)),
        ]))

        if aug.get("hflip", False):
            lines.append("Horizontal Flip Enabled")

        if "cjit" in aug:
            lines.append(f"Color Jitter (p={aug['cjit']['prob']})")
            lines.append(PrintLog._dash_aligned_lines([
                ("- Brightness", aug["cjit"]["brightness"]),
                ("- Contrast",   aug["cjit"]["contrast"]),
                ("- Saturation", aug["cjit"]["saturation"]),
                ("- Hue",        aug["cjit"]["hue"]),
            ]))

        if "sharpness" in aug:
            lines.append(PrintLog._dash_aligned_lines([
                (f"Sharpness (p={aug['sharpness']['prob']})", aug["sharpness"]["factor"]),
            ]))

        if "gblur" in aug:
            lines.append(f"Gaussian Blur (p={aug['gblur']['prob']})")
            lines.append(PrintLog._dash_aligned_lines([
                ("- Kernel Size", aug["gblur"]["kernel_size"]),
                ("- Sigma",       (aug["gblur"]["sigma"]["min"], aug["gblur"]["sigma"]["max"])),
            ]))

        lines.append("")
        return lines

    @staticmethod
    def _dash_aligned_lines(metric_pairs):
        labels, values = zip(*metric_pairs)
        max_len = max(len(label) for label in labels)
        n_dashes = [3 + (max_len - len(label)) for label in labels]
        lines = [f"{label} {'-' * nd} {value}" for label, nd, value in zip(labels, n_dashes, values)]
        return "\n".join(lines)

    @staticmethod
    @rank0
    def close_logs():
        for handle in (
            PrintLog.log_batch_general,
            PrintLog.log_batch_grad_norm,
            PrintLog.log_batch_scale_bias,
            PrintLog.log_batch_similarity,
            PrintLog.log_batch_kl,
            PrintLog.log_epoch,
            PrintLog.log_eval,
            PrintLog.log_init,
            PrintLog.log_text_train,
            PrintLog.log_text_eval,
        ):
            if handle is not None and not handle.closed:
                handle.close()
