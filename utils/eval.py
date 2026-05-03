import torch  # type: ignore[import]
from torch.amp import autocast  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import time
from tqdm import tqdm  # type: ignore[import]
from typing import Tuple, Any, List, Callable, Dict, Union, Optional
from collections import defaultdict
import math

from utils.data import spawn_dataloader, spawn_partition_data, spawn_partition_indexes_txts
from utils.head import compute_sim
from utils.utils import load_split

import pdb


RETRIEVAL_MODALITIES = ("i2t_map", "i2i_map", "t2i_map")


def compute_partition_map(scores_partition: Dict[str, float]) -> float:
    return sum(scores_partition[metric] for metric in RETRIEVAL_MODALITIES) / len(RETRIEVAL_MODALITIES)

def compute_partition_macro_map(scores_partition: Dict[str, float]) -> float:
    macro_keys = tuple(f"{m.replace('_map', '_macro_map')}" for m in RETRIEVAL_MODALITIES)
    return sum(scores_partition[k] for k in macro_keys) / len(macro_keys)

def list_eval_partition_names(split: Any, eval_type: str) -> List[str]:
    partition_names = []
    seen_partition_ids = set()

    for partition_name, data_index in split.data_indexes[eval_type].items():
        data_index_id = id(data_index)
        if data_index_id in seen_partition_ids:
            continue
        seen_partition_ids.add(data_index_id)
        partition_names.append(partition_name)

    return partition_names

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

        index_data, cid2enc = spawn_partition_data(
            config=config,
            partition_name=partition_name,
        )

        self.index_data = index_data
        self.cid2enc = cid2enc

        self.index_text, self.index_text_class_encs = spawn_partition_indexes_txts(
            cid2enc=cid2enc,
            text_template=text_template,
            dataset=config.dataset,
        )

        self.dataloader, self.time_cache = spawn_dataloader(
            index_data=index_data,
            text_template=text_template,
            config=config,
            shuffle=False,
            drop_last=False,
            img_pp=img_pp,
            use_dv_sampler=False,
            persistent_workers=config.hw.persistent_workers_eval,
        )

        self.cfg = config
        self.partition_name = partition_name
        self.batch_size = self.dataloader.batch_size
        self.mixed_prec = config.hw.mixed_prec

        if self.partition_name == "id":
            split = load_split(config.split_name, dataset=config.dataset)
            self.nshot_bucket_names = list(split.id_eval_nshot["names"])
            self.class_enc_to_bucket = build_class_enc_to_train_nshot_bucket(
                split_name=config.split_name,
                dataset=config.dataset,
                cid2enc=self.cid2enc,
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

        scores_i2t = compute_map_cross_modal(
            embs_q=embs_img_all.cpu(),
            class_encs_q=class_encs_img_all.cpu(),
            embs_g=embs_text_all.cpu(),
            class_encs_g=self.index_text_class_encs,
        )
        scores_i2i = compute_map_img2img(
            embs_q=embs_img_all,
            class_encs_q=class_encs_img_all,
            embs_g=embs_img_all,
            class_encs_g=class_encs_img_all,
        )
        scores_t2i = compute_map_cross_modal(
            embs_q=embs_text_all.cpu(),
            class_encs_q=self.index_text_class_encs,
            embs_g=embs_img_all.cpu(),
            class_encs_g=class_encs_img_all.cpu(),
        )

        eval_scores = {
            "i2t_prec1":     scores_i2t["prec1"].mean().item(),
            "i2t_map":       scores_i2t["map"],
            "i2t_macro_map": scores_i2t["macro_map"],
            "i2i_map":       scores_i2i["map"],
            "i2i_macro_map": scores_i2i["macro_map"],
            "t2i_map":       scores_t2i["map"],
            "t2i_macro_map": scores_t2i["macro_map"],
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
                bucket_vals = []
                for bucket_scores in (bucket_i2t_map, bucket_i2i_map, bucket_t2i_map):
                    val = bucket_scores.get(bucket_name)
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        continue
                    bucket_vals.append(val)

                # Some splits may not contain classes for every n-shot bucket.
                # Skip empty buckets instead of raising due to missing keys.
                if not bucket_vals:
                    continue

                bucket_comp = sum(bucket_vals) / len(bucket_vals)
                eval_scores[f"{bucket_name}_comp"] = bucket_comp

        return eval_scores


def compute_map_img2img(
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
    n_nsq_global = stats[1].item()

    map = ap_sum_global / n_nsq_global

    # Macro mAP: class-balanced mean AP, excluding singleton queries (NaN ap)
    if Q > 0:
        global_max_class = class_encs_q.max().reshape(1).clone()
    else:
        global_max_class = torch.zeros(1, dtype=torch.long, device=device)
    dist.all_reduce(global_max_class, op=dist.ReduceOp.MAX)
    num_classes = int(global_max_class.item()) + 1

    class_ap_sums = torch.zeros(num_classes, device=device)
    class_ap_counts = torch.zeros(num_classes, device=device)
    if has_pos_local.any():
        valid_encs = class_encs_q[has_pos_local]
        valid_aps = ap_local[has_pos_local]
        class_ap_sums.scatter_add_(0, valid_encs, valid_aps)
        class_ap_counts.scatter_add_(0, valid_encs, torch.ones_like(valid_aps))
    dist.all_reduce(class_ap_sums,   op=dist.ReduceOp.SUM)
    dist.all_reduce(class_ap_counts, op=dist.ReduceOp.SUM)
    active_classes = class_ap_counts > 0
    per_class_means = class_ap_sums[active_classes] / class_ap_counts[active_classes]
    macro_map = per_class_means.mean().item() if active_classes.any() else float("nan")

    scores_i2i = {
        "map": map,
        "macro_map": macro_map,
        "ap": ap_local.detach().cpu(),
        "has_pos": has_pos_local.detach().cpu(),
    }

    return scores_i2i

def compute_map_cross_modal(
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

    map = ap.mean().item()

    # Macro mAP: class-balanced mean AP
    num_classes_q = int(class_encs_q.max().item()) + 1
    class_ap_sums = torch.zeros(num_classes_q, device=device)
    class_ap_counts = torch.zeros(num_classes_q, device=device)
    valid_mask = ~torch.isnan(ap)
    if valid_mask.any():
        valid_encs = class_encs_q[valid_mask]
        valid_aps = ap[valid_mask]
        class_ap_sums.scatter_add_(0, valid_encs, valid_aps)
        class_ap_counts.scatter_add_(0, valid_encs, torch.ones_like(valid_aps))
    active_classes = class_ap_counts > 0
    per_class_means = class_ap_sums[active_classes] / class_ap_counts[active_classes]
    macro_map = per_class_means.mean().item() if active_classes.any() else float("nan")

    prec1 = pos_mask[:, 0].float()

    scores_cross_modal = {
        "map": map,
        "macro_map": macro_map,
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

        self.split = load_split(config.split_name, dataset=config.dataset)
        self.partition_names = list_eval_partition_names(self.split, config.eval_type)
        self.partition_pipes = {
            partition_name: SplitPartitionEvalPipeline(
                partition_name=partition_name,
                config=config,
                text_template=text_template,
                img_pp=img_pp,
            )
            for partition_name in self.partition_names
        }
        self.bucket_partition_name = next(
            (
                partition_name
                for partition_name, pipe in self.partition_pipes.items()
                if pipe.nshot_bucket_names
            ),
            None,
        )
        self.nshot_bucket_names = [] if self.bucket_partition_name is None else list(
            self.partition_pipes[self.bucket_partition_name].nshot_bucket_names
        )

        self.set_time_cache()

    def set_time_cache(self) -> None:
        time_caches = [pipe.time_cache for pipe in self.partition_pipes.values() if pipe.time_cache is not None]
        self.time_cache = None if not time_caches else sum(time_caches)

    def get_eval_texts(self) -> Dict[str, List[str]]:
        return {
            partition_name: list(self.partition_pipes[partition_name].index_text)
            for partition_name in self.partition_names
        }

    def run_validation(
        self, 
        modelw: Any, 
    ) -> Tuple[Dict[str, Any], bool, bool, float]:

        scores_val: Dict[str, Any] = {}
        partition_maps = []
        partition_macro_maps = []
        partition_i2i_maps = []
        partition_i2i_macro_maps = []
        nshot_scores: Dict[str, float] = {}
        time_elapsed_val = 0.0

        for partition_name in self.partition_names:
            scores_partition, loss_avg_partition, time_elapsed_partition = self.partition_pipes[partition_name].evaluate_split(modelw)

            scores_partition_core = {
                key: val
                for key, val in scores_partition.items()
                if not key.endswith("_comp")
            }
            if partition_name == self.bucket_partition_name:
                nshot_scores = {
                    key.removesuffix("_comp"): val
                    for key, val in scores_partition.items()
                    if key.endswith("_comp")
                }

            partition_map = compute_partition_map(scores_partition_core)
            partition_macro_map = compute_partition_macro_map(scores_partition_core)
            partition_maps.append(partition_map)
            partition_macro_maps.append(partition_macro_map)
            partition_i2i_maps.append(scores_partition_core["i2i_map"])
            partition_i2i_macro_maps.append(scores_partition_core["i2i_macro_map"])
            time_elapsed_val += time_elapsed_partition

            scores_val[partition_name] = {
                **scores_partition_core,
                "loss": loss_avg_partition,
            }

        comp_map = sum(partition_maps) / len(partition_maps)
        comp_macro_map = sum(partition_macro_maps) / len(partition_macro_maps)
        i2i_map = sum(partition_i2i_maps) / len(partition_i2i_maps)
        i2i_macro_map = sum(partition_i2i_macro_maps) / len(partition_i2i_macro_maps)

        scores_val["comp"] = {
            "map": comp_map,
            "macro_map": comp_macro_map,
        }
        for p_name, p_map, p_macro_map in zip(self.partition_names, partition_maps, partition_macro_maps):
            scores_val["comp"][p_name] = {"map": p_map, "macro_map": p_macro_map}
        scores_val["comp"]["i2i_map"] = i2i_map
        scores_val["comp"]["i2i_macro_map"] = i2i_macro_map
        if nshot_scores:
            scores_val["comp"]["n-shot"] = nshot_scores

        is_best_comp, is_best_i2i = self.check_bests(comp_map, i2i_map)

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
    dataset: str,
    cid2enc: Dict[str, int],
) -> Dict[int, str]:
    """
    Build class_enc -> bucket_name using ID-val bucket memberships.
    """

    split = load_split(split_name, dataset=dataset)
    class_enc_to_bucket = {}

    for bucket_name in split.id_eval_nshot["names"]:
        skeys_id_val = split.id_eval_nshot["buckets"][bucket_name]["id_val"]

        for cid, _ in skeys_id_val:
            # Some split variants (e.g., dev subsets) intentionally omit many classes
            # from train; skip n-shot entries that are absent in current class encoding map.
            if cid not in cid2enc:
                continue

            class_enc = cid2enc[cid]

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

            class_enc_i = int(class_enc)
            if class_enc_i not in class_enc_to_bucket:
                continue

            bucket_name = class_enc_to_bucket[class_enc_i]
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

        class_enc_i = int(class_enc)
        if class_enc_i not in class_enc_to_bucket:
            continue

        bucket_name = class_enc_to_bucket[class_enc_i]
        sums[bucket_name] += float(val)
        counts[bucket_name] += 1

    out = {}
    for bucket_name in bucket_names:
        if counts[bucket_name] > 0:
            out[bucket_name] = sums[bucket_name] / counts[bucket_name]

    return out