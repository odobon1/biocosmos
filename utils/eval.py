import torch  # type: ignore[import]
from torch.amp import autocast  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import time
from tqdm import tqdm  # type: ignore[import]
from typing import Tuple, Any, List, Callable, Dict, Union, Optional
from collections import defaultdict
import math

from utils.data import spawn_dataloader, spawn_partition_indexes, spawn_partition_indexes_txts
from utils.head import compute_sim
from utils.utils import load_split

import pdb


class SplitPartitionEvalPipeline:

    def __init__(
            self, 
            partition_name: str, 
            config: Any, 
            text_template: List[List[str]],
            img_pp: Callable,
        ) -> None:

        assert all(len(text_template_cat) == 1 for text_template_cat in text_template), \
               "text_template: each inner list must contain exactly one element for eval"

        index_data, sid_2_class_enc = spawn_partition_indexes(
            config=config,
            partition_name=partition_name,
        )

        self.index_data = index_data
        self.sid_2_class_enc = sid_2_class_enc

        self.index_text, self.index_text_class_encs = spawn_partition_indexes_txts(
            sid_2_class_enc=sid_2_class_enc,
            text_template=text_template,
        )

        self.dataloader, self.time_cache = spawn_dataloader(
            index_data=index_data,
            text_template=text_template,
            config=config,
            shuffle=False,
            drop_last=False,
            img_pp=img_pp,
            use_dv_sampler=False,
        )

        self.cfg = config
        self.partition_name = partition_name
        self.batch_size = self.dataloader.batch_size
        self.mixed_prec = config.hw.mixed_prec

        if self.partition_name == "id":
            split = load_split(config.split_name)
            self.nshot_bucket_names = list(split.id_eval_nshot["names"])
            self.class_enc_to_bucket = build_class_enc_to_train_nshot_bucket(
                split_name=config.split_name,
                sid_2_class_enc=self.sid_2_class_enc,
            )
        else:
            self.nshot_bucket_names = []
            self.class_enc_to_bucket = {}

    @torch.no_grad()
    def evaluate_split(
        self, 
        modelw: Any,
    ) -> Tuple[Dict[str, float], float, float]:
        
        time_start = time.time()
        modelw.model.eval()

        # text embeddings
        if self.mixed_prec:
            with autocast(device_type=modelw.device.type):
                embs_text_all = modelw.embed_texts(self.index_text)  # pt[L, D]
        else:
            embs_text_all = modelw.embed_texts(self.index_text)
        
        # image embeddings
        embs_imgs      = []
        class_encs_img = []
        loss_total     = 0.0
        n_samps_loss   = 0

        for imgs_sb, texts_sb, class_encs_img_sb, targ_data_sb in tqdm(
            self.dataloader, 
            desc=f"Eval ({self.partition_name})", 
            leave=False
        ):
            imgs_sb = imgs_sb.to(modelw.device, non_blocking=True)
            class_encs_img_sb = class_encs_img_sb.to(modelw.device, non_blocking=True)

            B = imgs_sb.size(0)

            if self.mixed_prec:
                with autocast(device_type=modelw.device.type):
                    loss, _, embs_img_b, _, _, class_encs_img_b = modelw.batch_step(imgs_sb, texts_sb, class_encs_img_sb, targ_data_sb)
            else:
                loss, _, embs_img_b, _, _, class_encs_img_b = modelw.batch_step(imgs_sb, texts_sb, class_encs_img_sb, targ_data_sb)

            embs_imgs.append(embs_img_b.cpu())
            class_encs_img.append(class_encs_img_b.cpu())

            batch_loss = loss.detach().item() * B
            loss_total += batch_loss
            n_samps_loss += B

        embs_img_all = torch.cat(embs_imgs, dim=0).to(modelw.device)  # pt[Q, D]
        class_encs_img_all = torch.cat(class_encs_img, dim=0).to(modelw.device)  # pt[Q]

        eval_scores  = self.compute_map_scores(embs_img_all, class_encs_img_all, embs_text_all)

        # reduce scalars (loss & accuracy)
        stats = torch.tensor([
            loss_total,
            n_samps_loss,
        ], device=modelw.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        loss_total   = stats[0].item()
        n_samps_loss = stats[1].item()

        loss_avg = loss_total / n_samps_loss

        modelw.model.train()

        time_end = time.time()
        time_elapsed = time_end - time_start

        return eval_scores, loss_avg, time_elapsed

    def compute_map_scores(
        self, 
        embs_img_all: torch.Tensor, 
        class_encs_img_all: torch.Tensor, 
        embs_text_all: torch.Tensor
    ) -> Dict[str, float]:

        scores_i2t = self.compute_map_cross_modal(
            embs_q=embs_img_all.cpu(),
            class_encs_q=class_encs_img_all.cpu(),
            embs_g=embs_text_all.cpu(),
            class_encs_g=self.index_text_class_encs,
        )
        scores_i2i = self.compute_map_img2img(
            embs_q=embs_img_all,
            class_encs_q=class_encs_img_all,
            embs_g=embs_img_all,
            class_encs_g=class_encs_img_all,
        )
        scores_t2i = self.compute_map_cross_modal(
            embs_q=embs_text_all.cpu(),
            class_encs_q=self.index_text_class_encs,
            embs_g=embs_img_all.cpu(),
            class_encs_g=class_encs_img_all.cpu(),
        )

        eval_scores = {
            "i2t_prec1": scores_i2t["prec1"].mean().item(),
            "i2t_map":   scores_i2t["map"],
            "i2i_map":   scores_i2i["map"],
            "t2i_map":   scores_t2i["map"],
        }

        if self.partition_name == "id":
            bucket_i2t_prec1 = reduce_bucketed_query_metric_by_class_enc(
                class_encs_q=class_encs_img_all.cpu(),
                values=scores_i2t["prec1"],
                class_enc_to_bucket=self.class_enc_to_bucket,
                bucket_names=self.nshot_bucket_names,
            )

            bucket_i2t_map = reduce_bucketed_query_metric_by_class_enc(
                class_encs_q=class_encs_img_all.cpu(),
                values=scores_i2t["ap"],
                class_enc_to_bucket=self.class_enc_to_bucket,
                bucket_names=self.nshot_bucket_names,
            )

            bucket_t2i_map = reduce_bucketed_query_metric_by_class_enc(
                class_encs_q=self.index_text_class_encs.cpu(),
                values=scores_t2i["ap"],
                class_enc_to_bucket=self.class_enc_to_bucket,
                bucket_names=self.nshot_bucket_names,
            )

            bucket_i2i_map = reduce_bucketed_query_metric_by_class_enc(
                class_encs_q=class_encs_img_all.cpu(),
                values=scores_i2i["ap"],
                class_enc_to_bucket=self.class_enc_to_bucket,
                bucket_names=self.nshot_bucket_names,
                valid_mask=scores_i2i["has_pos"],
                distributed=True,
                device=self.cfg.device,
            )

            for bucket_name in self.nshot_bucket_names:
                bucket_comp = (
                    bucket_i2t_map[bucket_name]
                    + bucket_i2i_map[bucket_name]
                    + bucket_t2i_map[bucket_name]
                ) / 3.0
                eval_scores[f"{bucket_name}_comp"] = bucket_comp

        return eval_scores

    def compute_map_img2img(
        self,
        embs_q: torch.Tensor,
        class_encs_q: torch.Tensor,
        embs_g: torch.Tensor,
        class_encs_g: torch.Tensor,
        chunk_size: int = 16384,
    ) -> Union[float, Dict[str, Any]]:
        """
        GPU-accelerated chunked vectorized mAP for evaluating image-to-image retrieval.
        Computes N x N similarity matrix in chunks on GPU to avoid OOM on GPU and slowness on CPU.
        Each image is queried against all others.

        Args:
        - embs_q --------- Query image embeddings
        - class_encs_q --- Query class encodings (corresponding to query image embeddings; 1D tensor of integers)
        - embs_g --------- Gallery image embeddings
        - class_encs_g --- Gallery class encodings (corresponding to gallery image embeddings; 1D tensor of integers)
        - chunk_size ----- Chunk size for computing massive similarity matrix (to avoid OOM on GPU)

        Returns:
        - [float] ------------ Image-to-image Mean Average Precision (mAP) over all non-singleton queries
        - [Dict[str, Any]] --- Optional dictionary containing per-query statistics if return_query_stats is True
        """

        device = embs_q.device
        rank   = dist.get_rank()

        Q = embs_q.size(0)  # num. query images
        N = embs_g.size(0)  # num. gallery images

        rank_offset = rank * Q
        
        ap_local = None
        has_pos_local = None
        ap_local = torch.full((Q,), float("nan"), device=device)
        has_pos_local = torch.zeros(Q, dtype=torch.bool, device=device)

        ap_sum_local = 0.0
        n_nsq_local  = 0  # num. non-singleton queries
        for i in range(0, Q, chunk_size):
            i_end = min(i + chunk_size, Q)
            
            embs_q_chunk       = embs_q[i:i_end]
            class_encs_q_chunk = class_encs_q[i:i_end]
            
            # chunked similarity matrix (chunk x all)
            sim_chunk = compute_sim(embs_q_chunk, embs_g, "cos")  # pt[U, N]
            # mask self-similarity
            sim_chunk[torch.arange(i_end - i), torch.arange(rank_offset + i, rank_offset + i_end)] = float('-inf')

            # sorted indices of top-N neighbors per query (all of them except query image itself)
            _, idxs_chunk = sim_chunk.sort(dim=1, descending=True)
            # positives mask / boolean relevance mask (True wherever the query-image class matches the gallery-image class)
            pos_mask_chunk = class_encs_q_chunk.unsqueeze(1) == class_encs_g[idxs_chunk]  # pt[U, N]
            # Remove self-match column
            pos_mask_chunk = pos_mask_chunk[:, :N-1]  # pt[U, N]  
            
            # cumulative precision at each rank
            ranks = torch.arange(1, N, device=device).float()  # pt[N-1]
            cum_prec = pos_mask_chunk.cumsum(dim=1).float() / ranks  # pt[U, N-1]

            # compute AP per query: sum(precision@hit) / num. positives
            pos_counts_chunk = pos_mask_chunk.sum(dim=1).float()  # pt[U]
            
            ap_chunk = (cum_prec * pos_mask_chunk.float()).sum(dim=1)  # pt[U]
            
            # avoid div-by-zero for queries with no positives
            has_pos = pos_counts_chunk > 0
            ap_chunk_valid = torch.full_like(pos_counts_chunk, float("nan"))
            # only divide where pos_counts_chunk > 0
            ap_chunk_valid[has_pos] = ap_chunk[has_pos] / pos_counts_chunk[has_pos]
            
            ap_sum_local += ap_chunk_valid[has_pos].sum().item()
            n_nsq_local += has_pos.sum().item()
            
            ap_local[i:i_end] = ap_chunk_valid
            has_pos_local[i:i_end] = has_pos

            # free up VRAM
            del embs_q_chunk, class_encs_q_chunk, sim_chunk, idxs_chunk, pos_mask_chunk, cum_prec
        
        stats = torch.tensor([ap_sum_local, n_nsq_local], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        ap_sum_global = stats[0].item()
        n_nsq_global  = stats[1].item()

        map_i2i = ap_sum_global / n_nsq_global

        scores_i2i = {
            "map": map_i2i,
            "ap": ap_local.detach().cpu(),
            "has_pos": has_pos_local.detach().cpu(),
        }

        return scores_i2i

    def compute_map_cross_modal(
        self, 
        embs_q: torch.Tensor, 
        class_encs_q: torch.Tensor, 
        embs_g: torch.Tensor, 
        class_encs_g: torch.Tensor,
    ) -> Union[float, Dict[str, Any]]:
        """
        Vectorized mAP for evaluating cross-modal retrieval (image-to-text & text-to-image)
        Note: Tensors stay on CPU for this ~ it's safe, it's fast (enough)
        
        Args:
        - embs_q --------- Query embeddings
        - class_encs_q --- Query class encodings (corresponding to query embeddings)
        - embs_g --------- Gallery embeddings
        - class_encs_g --- Gallery class encodings (corresponding to gallery embeddings)

        Returns:
        - [float] ------------ Mean Average Precision (mAP) over all queries
        - [Dict[str, Any]] --- Optional dictionary containing per-query statistics if return_query_stats is True
        """

        device       = embs_q.device
        class_encs_g = class_encs_g.to(device)
        class_encs_q = class_encs_q.to(device)

        N = embs_g.size(0)  # num. gallery embeddings

        # full similarity matrix
        sim = compute_sim(embs_q, embs_g, "cos")  # pt[Q, N]

        # get top-N neighbors per query (all of them)
        _, idxs = sim.topk(N, dim=1)  # pt[Q, N]

        # positives mask / boolean relevance mask (True wherever the query class matches the candidate class)
        pos_mask = class_encs_q.unsqueeze(1) == class_encs_g[idxs]  # pt[Q, N]

        # cumulative precision at each rank
        ranks    = torch.arange(1, N+1, device=sim.device)  # pt[N]
        cum_prec = pos_mask.cumsum(dim=1).float() / ranks  # pt[Q, N]

        # compute AP per query: sum(precision@hit) / num. positives
        pos_counts = pos_mask.sum(dim=1).float()  # pt[Q]
        ap         = (cum_prec * pos_mask.float()).sum(dim=1) / pos_counts  # pt[Q]

        map_score = ap.mean().item()

        prec1 = pos_mask[:, 0].float()

        scores_cross_modal = {
            "map": map_score,
            "ap": ap.detach().cpu(),
            "prec1": prec1.detach().cpu(),
        }

        return scores_cross_modal

class ValidationPipeline:

    def __init__(
        self,
        config: Any,
        text_template: List[List[str]],
        img_pp: Callable,
        header_tag: Optional[str] = None,
    ):

        self.header_tag = header_tag

        self.best_comp_map = None
        self.best_i2i_map = None

        self.val_pipe_id = SplitPartitionEvalPipeline(
            partition_name="id",
            config=config,
            text_template=text_template,
            img_pp=img_pp,
        )

        self.val_pipe_ood = SplitPartitionEvalPipeline(
            partition_name="ood",
            config=config,
            text_template=text_template,
            img_pp=img_pp,
        )

        self.set_time_cache()

    def set_time_cache(self) -> None:
        if self.val_pipe_id.time_cache is not None:
            self.time_cache = self.val_pipe_id.time_cache + self.val_pipe_ood.time_cache
        else:
            self.time_cache = None

    def run_validation(
        self, 
        modelw: Any, 
    ) -> Tuple[Dict[str, float], bool, bool, float]:

        scores_id, loss_avg_id, time_elapsed_id    = self.val_pipe_id.evaluate_split(modelw)
        scores_ood, loss_avg_ood, time_elapsed_ood = self.val_pipe_ood.evaluate_split(modelw)

        id_map = (scores_id["i2t_map"] + scores_id["i2i_map"] + scores_id["t2i_map"]) / 3
        ood_map = (scores_ood["i2t_map"] + scores_ood["i2i_map"] + scores_ood["t2i_map"]) / 3

        comp_map = (id_map + ood_map) / 2
        i2i_map = (scores_id["i2i_map"] + scores_ood["i2i_map"]) / 2

        scores_val = {
            "id_i2t_prec1": scores_id["i2t_prec1"],
            "id_i2t_map": scores_id["i2t_map"],
            "id_i2i_map": scores_id["i2i_map"],
            "id_t2i_map": scores_id["t2i_map"],
            "id_map": id_map,
            "id_loss": loss_avg_id,
            "ood_i2t_prec1": scores_ood["i2t_prec1"],
            "ood_i2t_map": scores_ood["i2t_map"],
            "ood_i2i_map": scores_ood["i2i_map"],
            "ood_t2i_map": scores_ood["t2i_map"],
            "ood_map": ood_map,
            "ood_loss": loss_avg_ood,
            "comp_map": comp_map,
            "i2i_map": i2i_map,
            "comp_loss": (loss_avg_id + loss_avg_ood) / 2
        }

        for key, val in scores_id.items():
            if key in {"i2t_prec1", "i2t_map", "i2i_map", "t2i_map"}:
                continue
            scores_val[f"id_{key}"] = val

        is_best_comp, is_best_i2i = self.check_bests(comp_map, i2i_map)

        time_elapsed_val = time_elapsed_id + time_elapsed_ood

        return scores_val, is_best_comp, is_best_i2i, time_elapsed_val

    def check_bests(self, comp_map: float, i2i_map: float) -> Tuple[bool, bool]:
        is_best_comp, is_best_i2i = False, False
        if self.best_comp_map is None:
            self.best_comp_map = comp_map
            self.best_i2i_map = i2i_map
        else:
            if comp_map > self.best_comp_map:
                self.best_comp_map = comp_map
                is_best_comp = True
            if i2i_map > self.best_i2i_map:
                self.best_i2i_map = i2i_map
                is_best_i2i = True
        return is_best_comp, is_best_i2i

def build_class_enc_to_train_nshot_bucket(
    split_name: str,
    sid_2_class_enc: Dict[str, int],
) -> Dict[int, str]:
    """
    Build class_enc -> bucket_name using ID-val bucket memberships.
    """

    split = load_split(split_name)
    class_enc_to_bucket = {}

    for bucket_name in split.id_eval_nshot["names"]:
        skeys_id_val = split.id_eval_nshot["buckets"][bucket_name]["id_val"]

        for sid, _ in skeys_id_val:
            class_enc = sid_2_class_enc[sid]

            if class_enc in class_enc_to_bucket and class_enc_to_bucket[class_enc] != bucket_name:
                raise ValueError(
                    f"class_enc '{class_enc}' appears in multiple n-shot buckets: "
                    f"{class_enc_to_bucket[class_enc]} and {bucket_name}"
                )

            class_enc_to_bucket[class_enc] = bucket_name

    return class_enc_to_bucket

def reduce_bucketed_query_metric_by_class_enc(
    class_encs_q,
    values,
    class_enc_to_bucket: Dict[int, str],
    bucket_names: List[str],
    valid_mask=None,
    distributed: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Reduce per-query values into bucket means using query class encodings.

    Args:
    - class_encs_q -------- Query class encodings
    - values -------------- Per-query metric values
    - class_enc_to_bucket - Maps class encoding -> bucket name
    - bucket_names -------- Ordered list of bucket names
    - valid_mask ---------- Optional boolean mask for valid queries only
    - distributed --------- If True, all-reduce bucket sums/counts across ranks
    - device -------------- Required when distributed=True
    """

    if hasattr(class_encs_q, "tolist"):
        class_encs_q = class_encs_q.tolist()

    if hasattr(values, "tolist"):
        values = values.tolist()

    if valid_mask is not None and hasattr(valid_mask, "tolist"):
        valid_mask = valid_mask.tolist()

    if distributed:
        if device is None:
            raise ValueError("device must be provided when distributed=True")

        bucket_to_idx = {name: i for i, name in enumerate(bucket_names)}

        sums = torch.zeros(len(bucket_names), device=device, dtype=torch.float32)
        counts = torch.zeros(len(bucket_names), device=device, dtype=torch.float32)

        for i, class_enc in enumerate(class_encs_q):
            if valid_mask is not None and not valid_mask[i]:
                continue

            val = values[i]
            if isinstance(val, float) and math.isnan(val):
                continue

            bucket_name = class_enc_to_bucket[int(class_enc)]
            bucket_idx = bucket_to_idx[bucket_name]

            sums[bucket_idx] += float(val)
            counts[bucket_idx] += 1.0

        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        out = {}
        for bucket_name, bucket_idx in bucket_to_idx.items():
            count = counts[bucket_idx].item()
            if count > 0:
                out[bucket_name] = (sums[bucket_idx] / counts[bucket_idx]).item()

        return out

    sums = defaultdict(float)
    counts = defaultdict(int)

    for i, class_enc in enumerate(class_encs_q):
        if valid_mask is not None and not valid_mask[i]:
            continue

        val = values[i]
        if isinstance(val, float) and math.isnan(val):
            continue

        bucket_name = class_enc_to_bucket[int(class_enc)]
        sums[bucket_name] += float(val)
        counts[bucket_name] += 1

    out = {}
    for bucket_name in bucket_names:
        if counts[bucket_name] > 0:
            out[bucket_name] = sums[bucket_name] / counts[bucket_name]

    return out