import torch
from torch.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from typing import Tuple, Any, List, Callable, Dict, Union, Optional
from collections import defaultdict
import math

from utils.data import spawn_dataloader, spawn_partition_data, spawn_partition_indexes_txts
from utils.head import compute_sim
from utils.utils import load_split, Timer
from utils.config import TrainConfig, EvalConfig

import pdb


RETRIEVAL_MODALITIES = ("i2t", "i2i", "t2i")


def harmonic_mean(values):
    if any(v == 0 for v in values):
        return 0.0
    n = len(values)
    reciprocal_sum = sum(1 / v for v in values)
    return n / reciprocal_sum

def compute_class_means_from_query_metric(
    class_encs_q: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Reduce per-query metric values into per-class means indexed by class encoding.
    Inactive classes are returned as NaN.
    """

    device = class_encs_q.device
    num_classes_q = int(class_encs_q.max().item()) + 1
    class_value_sums = torch.zeros(num_classes_q, device=device)
    class_value_counts = torch.zeros(num_classes_q, device=device)

    valid_mask = ~torch.isnan(values)
    if valid_mask.any():
        valid_encs = class_encs_q[valid_mask]
        valid_values = values[valid_mask]
        class_value_sums.scatter_add_(0, valid_encs, valid_values)
        class_value_counts.scatter_add_(0, valid_encs, torch.ones_like(valid_values))

    active_classes = class_value_counts > 0
    class_values = torch.full((num_classes_q,), float("nan"), device=device)
    class_values[active_classes] = class_value_sums[active_classes] / class_value_counts[active_classes]
    return class_values

def list_eval_partitions(split: Any, eval_type: str) -> List[str]:
    partitions = []
    seen_partition_ids = set()

    for partition, data_index in split.data_indexes[eval_type].items():
        data_index_id = id(data_index)
        if data_index_id in seen_partition_ids:
            continue
        seen_partition_ids.add(data_index_id)
        partitions.append(partition)

    return partitions

def gather_variable_rows(tensor: torch.Tensor) -> torch.Tensor:
    if dist.get_world_size() == 1:
        return tensor

    device = tensor.device
    size_local = torch.tensor([tensor.size(0)], device=device, dtype=torch.long)
    size_parts = [torch.zeros_like(size_local) for _ in range(dist.get_world_size())]
    dist.all_gather(size_parts, size_local)
    sizes = [int(size.item()) for size in size_parts]
    max_size = max(sizes, default=0)

    if tensor.size(0) < max_size:
        pad_shape = (max_size - tensor.size(0), *tensor.shape[1:])
        pad = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)

    return torch.cat([part[:sizes[rank]] for rank, part in enumerate(gathered)], dim=0)

def gather_object_list(items: List[Any]) -> List[Any]:
    if dist.get_world_size() == 1:
        return items

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, list(items))

    merged = []
    for part in gathered:
        merged.extend(part)
    return merged

class PartitionEvaluationPipeline:

    def __init__(
            self, 
            partition: str, 
            config: Union[TrainConfig, EvalConfig], 
            text_template: List[List[str]],
            img_pp: Callable,
        ) -> None:

        assert all(len(text_template_cat) == 1 for text_template_cat in text_template), \
               "text_template: each inner list must contain exactly one element for eval"

        index_data, cid2enc, enc2cid = spawn_partition_data(
            config,
            partition,
        )

        self.index_data = index_data
        self.cid2enc = cid2enc
        self.index_text_cids = list(cid2enc.keys())

        self.index_text, self.index_text_class_encs = spawn_partition_indexes_txts(
            cid2enc,
            text_template,
            config.dataset,
        )

        self.dataloader = spawn_dataloader(
            index_data=index_data,
            enc2cid=enc2cid,
            text_template=text_template,
            config=config,
            shuffle=False,
            drop_last=False,
            img_pp=img_pp,
            use_dv_sampler=False,
            exact_distributed=True,
            persistent_workers=config.hw.persistent_workers_eval,
        )

        self.cfg = config
        self.partition = partition
        self.batch_size = self.dataloader.batch_size
        self.mixed_prec = config.hw.mixed_prec

        if self.partition == "id":
            split = load_split(config.dataset, config.split)
            self.nshot_bucket_names = list(split.nshot["names"])
            self.class_enc_to_bucket = build_class_enc_to_train_nshot_bucket(
                config.dataset,
                config.split,
                self.cid2enc,
            )
        else:
            self.nshot_bucket_names = []
            self.class_enc_to_bucket = {}

    @torch.no_grad()
    def collect_eval_artifacts(
        self, 
        modelw: Any,
        loss_flag: bool,
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        
        modelw.model.eval()

        # text embeddings
        if self.mixed_prec:
            with autocast(device_type=modelw.device.type):
                embs_text_all = modelw.embed_texts(self.index_text)  # pt[L, D]
        else:
            embs_text_all = modelw.embed_texts(self.index_text)
        
        # image embeddings (+ paired text embeddings & targets for the loss)
        embs_imgs = []
        embs_txts = []
        class_encs_img = []
        targ_data_loss = []
        cids_img = []
        for imgs_sb, texts_sb, class_encs_img_sb, targ_data_sb in tqdm(
            self.dataloader,
            desc=f"Eval ({self.partition})",
            leave=False,
            disable=(dist.get_rank() != 0),
        ):
            imgs_sb = imgs_sb.to(modelw.device, non_blocking=True)

            if self.mixed_prec:
                with autocast(device_type=modelw.device.type):
                    embs_img_b, embs_txt_b = modelw.batch_step_local(imgs_sb, texts_sb)
            else:
                embs_img_b, embs_txt_b = modelw.batch_step_local(imgs_sb, texts_sb)

            embs_imgs.append(embs_img_b.cpu())
            class_encs_img.append(class_encs_img_sb.cpu())
            cids_img.extend(targ_data["cid"] for targ_data in targ_data_sb)

            if loss_flag:
                embs_txts.append(embs_txt_b.cpu())
                targ_data_loss.extend(targ_data_sb)

        if embs_imgs:
            embs_img_local = torch.cat(embs_imgs, dim=0).to(modelw.device)
            class_encs_img_local = torch.cat(class_encs_img, dim=0).to(modelw.device)
        else:
            embs_img_local = torch.empty((0, modelw.embed_dim), device=modelw.device)
            class_encs_img_local = torch.empty((0,), device=modelw.device, dtype=torch.long)

        embs_img_all = gather_variable_rows(embs_img_local)
        class_encs_img_all = gather_variable_rows(class_encs_img_local)
        cids_img = gather_object_list(cids_img)

        loss_avg = None
        if loss_flag:
            if embs_txts:
                embs_txt_local = torch.cat(embs_txts, dim=0).to(modelw.device)
            else:
                embs_txt_local = torch.empty((0, modelw.embed_dim), device=modelw.device)

            embs_txt_all = gather_variable_rows(embs_txt_local)
            targ_data_all = gather_object_list(targ_data_loss)

            # global negative pool: per-rank eval batch x world size (mirrors the global training batch)
            chunk_size = self.batch_size * dist.get_world_size()
            if self.mixed_prec:
                with autocast(device_type=modelw.device.type):
                    loss_avg = modelw.eval_loss_chunked(
                        embs_img_all, embs_txt_all, class_encs_img_all, targ_data_all, chunk_size,
                    )
            else:
                loss_avg = modelw.eval_loss_chunked(
                    embs_img_all, embs_txt_all, class_encs_img_all, targ_data_all, chunk_size,
                )

        modelw.model.train()

        artifacts = {
            "embs_img": embs_img_all,
            "class_encs_img": class_encs_img_all,
            "cids_img": cids_img,
            "embs_text": embs_text_all,
            "class_encs_text": self.index_text_class_encs,
            "cids_text": list(self.index_text_cids),
        }

        return artifacts, loss_avg

    def _build_metric_views(
        self,
        scores_i2t: Dict[str, Any],
        scores_i2i: Dict[str, Any],
        scores_t2i: Dict[str, Any],
        class_encs_img_q: torch.Tensor,
        class_encs_text_q: torch.Tensor,
        class_enc_to_bucket: Optional[Dict[int, str]] = None,
        nshot_bucket_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        eval_scores = {
            "standard": {
                "acc": {
                    "i2t": scores_i2t["acc"],
                },
                "map": {
                    "i2t": scores_i2t["map"],
                    "i2i": scores_i2i["map"],
                    "t2i": scores_t2i["map"],
                },
            },
            "per_class": {
                "acc": {
                    "i2t": scores_i2t["macro_acc"],
                },
                "map": {
                    "i2t": scores_i2t["macro_map"],
                    "i2i": scores_i2i["macro_map"],
                    "t2i": scores_t2i["macro_map"],
                },
            },
        }

        if not class_enc_to_bucket or not nshot_bucket_names:
            return eval_scores

        bucket_i2t_prec1 = reduce_bucketed_query_metric_by_class_enc(
            class_encs_q=class_encs_img_q.cpu(),
            values=scores_i2t["accs"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
        )

        bucket_i2t_map = reduce_bucketed_query_metric_by_class_enc(
            class_encs_q=class_encs_img_q.cpu(),
            values=scores_i2t["ap"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
        )

        bucket_t2i_map = reduce_bucketed_query_metric_by_class_enc(
            class_encs_q=class_encs_text_q.cpu(),
            values=scores_t2i["ap"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
        )

        bucket_i2i_map = reduce_bucketed_query_metric_by_class_enc(
            class_encs_q=class_encs_img_q.cpu(),
            values=scores_i2i["ap"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
            valid_mask=scores_i2i["has_pos"],
            distributed=True,
            device=self.cfg.device,
        )

        bucket_i2t_macro_map = reduce_bucketed_macro_map_from_class_ap(
            class_ap=scores_i2t["class_ap"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
        )

        bucket_i2t_macro_acc = reduce_bucketed_macro_metric_from_class_values(
            class_values=scores_i2t["class_acc"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
        )

        bucket_t2i_macro_map = reduce_bucketed_macro_map_from_class_ap(
            class_ap=scores_t2i["class_ap"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
        )

        bucket_i2i_macro_map = reduce_bucketed_macro_map_from_class_ap(
            class_ap=scores_i2i["class_ap"],
            class_enc_to_bucket=class_enc_to_bucket,
            bucket_names=nshot_bucket_names,
        )

        for bucket_name in nshot_bucket_names:
            bucket_vals = []
            for bucket_scores in (bucket_i2t_map, bucket_i2i_map, bucket_t2i_map):
                val = bucket_scores.get(bucket_name)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                bucket_vals.append(val)

            if bucket_vals:
                bucket_comp = harmonic_mean(bucket_vals)
                eval_scores["standard"]["map"].setdefault("n-shot", {})[bucket_name] = bucket_comp

            bucket_acc = bucket_i2t_prec1.get(bucket_name)
            if bucket_acc is not None and not (isinstance(bucket_acc, float) and math.isnan(bucket_acc)):
                eval_scores["standard"]["acc"].setdefault("n-shot", {})[bucket_name] = bucket_acc

            bucket_macro_vals = []
            for bucket_scores in (bucket_i2t_macro_map, bucket_i2i_macro_map, bucket_t2i_macro_map):
                val = bucket_scores.get(bucket_name)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                bucket_macro_vals.append(val)

            if bucket_macro_vals:
                bucket_comp_macro = harmonic_mean(bucket_macro_vals)
                eval_scores["per_class"]["map"].setdefault("n-shot", {})[bucket_name] = bucket_comp_macro

            bucket_macro_acc = bucket_i2t_macro_acc.get(bucket_name)
            if bucket_macro_acc is not None and not (isinstance(bucket_macro_acc, float) and math.isnan(bucket_macro_acc)):
                eval_scores["per_class"]["acc"].setdefault("n-shot", {})[bucket_name] = bucket_macro_acc

        return eval_scores

    def compute_map_scores(
        self, 
        embs_img_q: torch.Tensor,
        class_encs_img_q: torch.Tensor,
        embs_text_q: torch.Tensor,
        class_encs_text_q: torch.Tensor,
        embs_img_g: torch.Tensor,
        class_encs_img_g: torch.Tensor,
        embs_text_g: torch.Tensor,
        class_encs_text_g: torch.Tensor,
        self_match_idxs_g: Optional[torch.Tensor] = None,
        class_enc_to_bucket: Optional[Dict[int, str]] = None,
        nshot_bucket_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        chunk_size = self.cfg.hw.chunk_size

        scores_i2t = compute_map_cross_modal(
            embs_q=embs_img_q.cpu(),
            class_encs_q=class_encs_img_q.cpu(),
            embs_g=embs_text_g.cpu(),
            class_encs_g=class_encs_text_g.cpu(),
            compute_accuracy=True,
            chunk_size=chunk_size["map_cross_modal"],
        )
        scores_i2i = compute_map_img2img(
            embs_q=embs_img_q,
            class_encs_q=class_encs_img_q,
            embs_g=embs_img_g,
            class_encs_g=class_encs_img_g,
            self_match_idxs_g=self_match_idxs_g,
            chunk_size=chunk_size["map_img2img"],
        )
        scores_t2i = compute_map_cross_modal(
            embs_q=embs_text_q.cpu(),
            class_encs_q=class_encs_text_q.cpu(),
            embs_g=embs_img_g.cpu(),
            class_encs_g=class_encs_img_g.cpu(),
            chunk_size=chunk_size["map_cross_modal"],
        )

        return self._build_metric_views(
            scores_i2t=scores_i2t,
            scores_i2i=scores_i2i,
            scores_t2i=scores_t2i,
            class_encs_img_q=class_encs_img_q,
            class_encs_text_q=class_encs_text_q,
            class_enc_to_bucket=class_enc_to_bucket,
            nshot_bucket_names=nshot_bucket_names,
        )


def compute_map_img2img(
    embs_q: torch.Tensor,
    class_encs_q: torch.Tensor,
    embs_g: torch.Tensor,
    class_encs_g: torch.Tensor,
    self_match_idxs_g: Optional[torch.Tensor] = None,
    chunk_size: int = 512,
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
    Q = embs_q.size(0)  # num. query images
    N = embs_g.size(0)  # num. gallery images

    if self_match_idxs_g is None:
        self_match_idxs_g = torch.arange(Q, device=device, dtype=torch.long)
    else:
        self_match_idxs_g = self_match_idxs_g.to(device=device, dtype=torch.long)
    
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
        # mask self-similarity for exact query/gallery item matches
        self_match_chunk = self_match_idxs_g[i:i_end]
        valid_self_mask = (self_match_chunk >= 0) & (self_match_chunk < N)
        if valid_self_mask.any():
            sim_chunk[torch.arange(i_end - i, device=device)[valid_self_mask], self_match_chunk[valid_self_mask]] = float('-inf')

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
    class_ap = torch.full((num_classes,), float("nan"), device=device)
    class_ap[active_classes] = class_ap_sums[active_classes] / class_ap_counts[active_classes]
    macro_map = class_ap[active_classes].mean().item() if active_classes.any() else float("nan")

    scores_i2i = {
        "map": map,
        "macro_map": macro_map,
        "ap": ap_local.detach().cpu(),
        "has_pos": has_pos_local.detach().cpu(),
        "class_ap": class_ap.detach().cpu(),
    }

    return scores_i2i

def compute_map_cross_modal(
    embs_q: torch.Tensor, 
    class_encs_q: torch.Tensor, 
    embs_g: torch.Tensor, 
    class_encs_g: torch.Tensor,
    compute_accuracy: bool = False,
    chunk_size: int = 512,
) -> Union[float, Dict[str, Any]]:
    """
    Vectorized mAP for evaluating cross-modal retrieval (image-to-text & text-to-image)
    Computes the Q x N similarity matrix in chunks over the query dim to bound host memory.

    Args:
    - embs_q --------- Query embeddings
    - class_encs_q --- Query class encodings (corresponding to query embeddings)
    - embs_g --------- Gallery embeddings
    - class_encs_g --- Gallery class encodings (corresponding to gallery embeddings)
    - chunk_size ----- Chunk size (num. queries per chunk) for the similarity matrix (to bound memory)

    Returns:
    - [float] ------------ Mean Average Precision (mAP) over all queries
    - [Dict[str, Any]] --- Optional dictionary containing per-query statistics if return_query_stats is True
    """

    device       = embs_q.device
    class_encs_g = class_encs_g.to(device)
    class_encs_q = class_encs_q.to(device)

    Q = embs_q.size(0)  # num. query embeddings
    N = embs_g.size(0)  # num. gallery embeddings

    ranks = torch.arange(1, N+1, device=device).float()  # pt[N]
    ap    = torch.empty(Q, device=device)  # pt[Q]
    acc   = torch.empty(Q, device=device) if compute_accuracy else None  # pt[Q]

    # chunked over the query dim to bound the Q x N similarity matrix in memory
    for i in range(0, Q, chunk_size):
        i_end = min(i + chunk_size, Q)

        embs_q_chunk       = embs_q[i:i_end]
        class_encs_q_chunk = class_encs_q[i:i_end]

        # chunked similarity matrix (chunk x all)
        sim_chunk = compute_sim(embs_q_chunk, embs_g, "cos")  # pt[U, N]

        # get top-N neighbors per query (all of them)
        _, idxs_chunk = sim_chunk.topk(N, dim=1)  # pt[U, N]

        # positives mask / boolean relevance mask (True wherever the query class matches the candidate class)
        pos_mask_chunk = class_encs_q_chunk.unsqueeze(1) == class_encs_g[idxs_chunk]  # pt[U, N]

        # cumulative precision at each rank
        cum_prec_chunk = pos_mask_chunk.cumsum(dim=1).float() / ranks  # pt[U, N]

        # compute AP per query: sum(precision@hit) / num. positives
        pos_counts_chunk = pos_mask_chunk.sum(dim=1).float()  # pt[U]
        ap[i:i_end] = (cum_prec_chunk * pos_mask_chunk.float()).sum(dim=1) / pos_counts_chunk  # pt[U]

        if compute_accuracy:
            acc[i:i_end] = pos_mask_chunk[:, 0].float()  # pt[U]

        # free up memory
        del embs_q_chunk, class_encs_q_chunk, sim_chunk, idxs_chunk, pos_mask_chunk, cum_prec_chunk

    map = ap.mean().item()

    # Macro mAP: class-balanced mean AP
    class_ap = compute_class_means_from_query_metric(class_encs_q=class_encs_q, values=ap)
    active_classes = ~torch.isnan(class_ap)
    macro_map = class_ap[active_classes].mean().item() if active_classes.any() else float("nan")

    scores_cross_modal = {
        "map": map,
        "macro_map": macro_map,
        "ap": ap.detach().cpu(),
        "class_ap": class_ap.detach().cpu(),
    }

    if compute_accuracy:
        # Macro accuracy: class-balanced top-1 retrieval accuracy.
        class_acc = compute_class_means_from_query_metric(class_encs_q=class_encs_q, values=acc)
        active_acc_classes = ~torch.isnan(class_acc)
        macro_acc = class_acc[active_acc_classes].mean().item() if active_acc_classes.any() else float("nan")

        scores_cross_modal["acc"] = acc.detach().cpu().mean().item()
        scores_cross_modal["accs"] = acc.detach().cpu()
        scores_cross_modal["class_acc"] = class_acc.detach().cpu()
        scores_cross_modal["macro_acc"] = macro_acc

    return scores_cross_modal


class EvaluationPipeline:

    def __init__(
        self,
        config: Union[TrainConfig, EvalConfig],
        text_template: List[List[str]],
        img_pp: Callable,
        header_tag: Optional[str] = None,
    ):

        self.header_tag = header_tag

        self.best_comp_map = None
        self.best_i2i_map = None

        self.split = load_split(config.dataset, config.split)
        self.partitions = list_eval_partitions(self.split, config.eval_type)
        self.partition_pipes = {
            partition: PartitionEvaluationPipeline(
                partition=partition,
                config=config,
                text_template=text_template,
                img_pp=img_pp,
            )
            for partition in self.partitions
        }
        self.bucket_partition = next(
            (
                partition
                for partition, pipe in self.partition_pipes.items()
                if pipe.nshot_bucket_names
            ),
            None,
        )
        self.nshot_bucket_names = [] if self.bucket_partition is None else list(
            self.partition_pipes[self.bucket_partition].nshot_bucket_names
        )

    def get_eval_texts(self) -> Dict[str, List[str]]:
        return {
            partition: list(self.partition_pipes[partition].index_text)
            for partition in self.partitions
        }

    def evaluate(
        self, 
        modelw: Any,
        loss_flag: bool = True,
    ) -> Tuple[Dict[str, Any], bool, bool, float]:

        timer = Timer()
        timer.start()

        eval_metrics: Dict[str, Any] = {
            "scores": {
                "closed_set": {"standard": {}, "per_class": {}},
                "full_set": {"standard": {}, "per_class": {}},
            },
            "loss_raw": {},
        }
        accum = {
            (set_key, grp): {"all": [], "i2t": [], "i2i": [], "t2i": [], "acc_i2t": []}
            for set_key in ("closed_set", "full_set")
            for grp in ("standard", "per_class")
        }
        partition_artifacts: Dict[str, Dict[str, Any]] = {}
        partition_losses: Dict[str, Optional[float]] = {}

        for partition in self.partitions:
            pipe = self.partition_pipes[partition]
            artifacts_partition, loss_avg_partition = pipe.collect_eval_artifacts(modelw, loss_flag)
            partition_artifacts[partition] = artifacts_partition
            partition_losses[partition] = loss_avg_partition

        full_set_embs_img = torch.cat(
            [partition_artifacts[partition]["embs_img"] for partition in self.partitions],
            dim=0,
        )
        full_set_class_encs_img = torch.cat(
            [partition_artifacts[partition]["class_encs_img"] for partition in self.partitions],
            dim=0,
        )
        full_set_embs_text = torch.cat(
            [partition_artifacts[partition]["embs_text"] for partition in self.partitions],
            dim=0,
        )
        full_set_class_encs_text = torch.cat(
            [partition_artifacts[partition]["class_encs_text"] for partition in self.partitions],
            dim=0,
        )

        img_offset = 0
        for partition in self.partitions:
            pipe = self.partition_pipes[partition]
            artifacts_partition = partition_artifacts[partition]
            loss_avg_partition = partition_losses[partition]

            closed_set_scores = pipe.compute_map_scores(
                embs_img_q=artifacts_partition["embs_img"],
                class_encs_img_q=artifacts_partition["class_encs_img"],
                embs_text_q=artifacts_partition["embs_text"],
                class_encs_text_q=artifacts_partition["class_encs_text"].to(artifacts_partition["embs_img"].device),
                embs_img_g=artifacts_partition["embs_img"],
                class_encs_img_g=artifacts_partition["class_encs_img"],
                embs_text_g=artifacts_partition["embs_text"],
                class_encs_text_g=artifacts_partition["class_encs_text"],
                class_enc_to_bucket=pipe.class_enc_to_bucket if partition == self.bucket_partition else None,
                nshot_bucket_names=pipe.nshot_bucket_names if partition == self.bucket_partition else None,
            )

            self_match_idxs_g = torch.arange(
                img_offset,
                img_offset + artifacts_partition["embs_img"].size(0),
                device=artifacts_partition["embs_img"].device,
                dtype=torch.long,
            )
            img_offset += artifacts_partition["embs_img"].size(0)

            full_set_scores = pipe.compute_map_scores(
                embs_img_q=artifacts_partition["embs_img"],
                class_encs_img_q=artifacts_partition["class_encs_img"],
                embs_text_q=artifacts_partition["embs_text"],
                class_encs_text_q=artifacts_partition["class_encs_text"].to(artifacts_partition["embs_img"].device),
                embs_img_g=full_set_embs_img,
                class_encs_img_g=full_set_class_encs_img,
                embs_text_g=full_set_embs_text,
                class_encs_text_g=full_set_class_encs_text,
                self_match_idxs_g=self_match_idxs_g,
                class_enc_to_bucket=pipe.class_enc_to_bucket if partition == self.bucket_partition else None,
                nshot_bucket_names=self.nshot_bucket_names if partition == self.bucket_partition else None,
            )

            for set_key, scores in (("closed_set", closed_set_scores), ("full_set", full_set_scores)):
                for grp in ("standard", "per_class"):
                    a = accum[(set_key, grp)]
                    a["all"].append(harmonic_mean([scores[grp]["map"][m] for m in RETRIEVAL_MODALITIES]))
                    a["i2t"].append(scores[grp]["map"]["i2t"])
                    a["i2i"].append(scores[grp]["map"]["i2i"])
                    a["t2i"].append(scores[grp]["map"]["t2i"])
                    a["acc_i2t"].append(scores[grp]["acc"]["i2t"])

            eval_metrics["scores"]["closed_set"]["standard"][partition] = closed_set_scores["standard"]
            eval_metrics["scores"]["closed_set"]["per_class"][partition] = closed_set_scores["per_class"]
            eval_metrics["scores"]["full_set"]["standard"][partition] = full_set_scores["standard"]
            eval_metrics["scores"]["full_set"]["per_class"][partition] = full_set_scores["per_class"]
            if loss_flag and loss_avg_partition is not None:
                eval_metrics["loss_raw"][partition] = loss_avg_partition
            else:
                eval_metrics["loss_raw"][partition] = None

        for (set_key, grp), a in accum.items():
            eval_metrics["scores"][set_key][grp]["comp"] = {
                "acc": {"i2t": harmonic_mean(a["acc_i2t"])},
                "map": {
                    "all": harmonic_mean(a["all"]),
                    **dict(zip(self.partitions, a["all"])),
                    "i2t": harmonic_mean(a["i2t"]),
                    "i2i": harmonic_mean(a["i2i"]),
                    "t2i": harmonic_mean(a["t2i"]),
                },
            }

        is_best_comp, is_best_i2i = self.check_bests(
            harmonic_mean(accum[("closed_set", "standard")]["all"]),
            harmonic_mean(accum[("closed_set", "standard")]["i2i"]),
        )
        
        timer.stop()

        return eval_metrics, is_best_comp, is_best_i2i, timer.get_elapsed_time()

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
    dataset: str,
    split: str,
    cid2enc: Dict[str, int],
) -> Dict[int, str]:
    """
    Build class_enc -> bucket_name using ID-val bucket memberships.
    """

    split = load_split(dataset, split)
    class_enc_to_bucket = {}

    for bucket_name in split.nshot["names"]:
        cids_val_id = split.nshot["buckets"]["train/val"][bucket_name]

        for cid in cids_val_id:
            # Some split variants (e.g., dev) intentionally omit many classes from train; skip n-shot entries that are absent in current class encoding map.
            if cid not in cid2enc:
                continue

            class_enc = cid2enc[cid]
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

def reduce_bucketed_macro_map_from_class_ap(
    class_ap,
    class_enc_to_bucket: Dict[int, str],
    bucket_names: List[str],
) -> Dict[str, float]:
    """
    Group precomputed per-class AP means into bucket macro mAP.

    Args:
    - class_ap ------------ Per-class AP (indexed by class_enc, NaN for inactive classes).
                            Produced by compute_map_img2img / compute_map_cross_modal.
    - class_enc_to_bucket - Maps class encoding -> bucket name
    - bucket_names -------- Ordered list of bucket names
    """

    return reduce_bucketed_macro_metric_from_class_values(
        class_values=class_ap,
        class_enc_to_bucket=class_enc_to_bucket,
        bucket_names=bucket_names,
    )

def reduce_bucketed_macro_metric_from_class_values(
    class_values,
    class_enc_to_bucket: Dict[int, str],
    bucket_names: List[str],
) -> Dict[str, float]:
    """
    Group precomputed per-class metric means into bucket macro means.

    Args:
    - class_values -------- Per-class metric values (indexed by class_enc, NaN for inactive classes)
    - class_enc_to_bucket - Maps class encoding -> bucket name
    - bucket_names -------- Ordered list of bucket names
    """

    if hasattr(class_values, "tolist"):
        class_values = class_values.tolist()

    bucket_sum = defaultdict(float)
    bucket_count = defaultdict(int)

    for class_enc, metric_val in enumerate(class_values):
        if math.isnan(metric_val):
            continue
        if class_enc not in class_enc_to_bucket:
            continue
        bucket_name = class_enc_to_bucket[class_enc]
        bucket_sum[bucket_name] += float(metric_val)
        bucket_count[bucket_name] += 1

    out = {}
    for bucket_name in bucket_names:
        if bucket_count[bucket_name] > 0:
            out[bucket_name] = bucket_sum[bucket_name] / bucket_count[bucket_name]

    return out