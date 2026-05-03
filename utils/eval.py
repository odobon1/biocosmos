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


RETRIEVAL_MODALITIES = ("i2t", "i2i", "t2i")


def compute_partition_map(scores_partition: Dict[str, Dict[str, float]]) -> float:
    return sum(scores_partition["standard"]["map"][metric] for metric in RETRIEVAL_MODALITIES) / len(RETRIEVAL_MODALITIES)

def compute_partition_macro_map(scores_partition: Dict[str, Dict[str, float]]) -> float:
    return sum(scores_partition["per_class"]["map"][metric] for metric in RETRIEVAL_MODALITIES) / len(RETRIEVAL_MODALITIES)

def compute_partition_full_set_map(scores_partition: Dict[str, Dict[str, float]]) -> float:
    return sum(scores_partition["standard"]["full_set"]["map"][metric] for metric in RETRIEVAL_MODALITIES) / len(RETRIEVAL_MODALITIES)

def compute_partition_full_set_macro_map(scores_partition: Dict[str, Dict[str, float]]) -> float:
    return sum(scores_partition["per_class"]["full_set"]["map"][metric] for metric in RETRIEVAL_MODALITIES) / len(RETRIEVAL_MODALITIES)

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
        self.index_text_cids = list(cid2enc.keys())

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
    def collect_eval_artifacts(
        self, 
        modelw: Any,
    ) -> Tuple[Dict[str, Any], float]:
        
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
        cids_img       = []
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
            cids_img.extend(targ_data["cid"] for targ_data in targ_data_sb)

            batch_loss = loss.detach().item() * B
            loss_total += batch_loss
            n_samps_loss += B

        embs_img_all = torch.cat(embs_imgs, dim=0).to(modelw.device)  # pt[Q, D]
        class_encs_img_all = torch.cat(class_encs_img, dim=0).to(modelw.device)  # pt[Q]

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

        artifacts = {
            "embs_img": embs_img_all,
            "class_encs_img": class_encs_img_all,
            "cids_img": cids_img,
            "embs_text": embs_text_all,
            "class_encs_text": self.index_text_class_encs,
            "cids_text": list(self.index_text_cids),
        }

        return artifacts, loss_avg

    @staticmethod
    def _encode_cids(cids: List[str], cid2enc: Dict[str, int], device: torch.device) -> torch.Tensor:
        return torch.tensor([cid2enc[cid] for cid in cids], device=device, dtype=torch.long)

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
                bucket_comp = sum(bucket_vals) / len(bucket_vals)
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
                bucket_comp_macro = sum(bucket_macro_vals) / len(bucket_macro_vals)
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

        scores_i2t = compute_map_cross_modal(
            embs_q=embs_img_q.cpu(),
            class_encs_q=class_encs_img_q.cpu(),
            embs_g=embs_text_g.cpu(),
            class_encs_g=class_encs_text_g.cpu(),
            compute_accuracy=True,
        )
        scores_i2i = compute_map_img2img(
            embs_q=embs_img_q,
            class_encs_q=class_encs_img_q,
            embs_g=embs_img_g,
            class_encs_g=class_encs_img_g,
            self_match_idxs_g=self_match_idxs_g,
        )
        scores_t2i = compute_map_cross_modal(
            embs_q=embs_text_q.cpu(),
            class_encs_q=class_encs_text_q.cpu(),
            embs_g=embs_img_g.cpu(),
            class_encs_g=class_encs_img_g.cpu(),
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
        acc = pos_mask[:, 0].float()

        # Macro accuracy: class-balanced top-1 retrieval accuracy.
        class_acc = compute_class_means_from_query_metric(class_encs_q=class_encs_q, values=acc)
        active_acc_classes = ~torch.isnan(class_acc)
        macro_acc = class_acc[active_acc_classes].mean().item() if active_acc_classes.any() else float("nan")

        scores_cross_modal["acc"] = acc.detach().cpu().mean().item()
        scores_cross_modal["accs"] = acc.detach().cpu()
        scores_cross_modal["class_acc"] = class_acc.detach().cpu()
        scores_cross_modal["macro_acc"] = macro_acc

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
        self.best_full_set_comp_map = None
        self.best_full_set_i2i_map = None

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

        time_start = time.time()
        scores_val: Dict[str, Any] = {}
        partition_maps = []
        partition_macro_maps = []
        partition_i2i_maps = []
        partition_i2i_macro_maps = []
        partition_full_set_maps = []
        partition_full_set_macro_maps = []
        partition_full_set_i2i_maps = []
        partition_full_set_i2i_macro_maps = []
        partition_artifacts: Dict[str, Dict[str, Any]] = {}
        partition_losses: Dict[str, float] = {}

        full_set_index_data = []
        for partition_name in self.partition_names:
            full_set_index_data.extend(self.partition_pipes[partition_name].index_data)
        full_set_cid2enc = {
            cid: class_enc
            for class_enc, cid in enumerate(dict.fromkeys(datum["cid"] for datum in full_set_index_data))
        }
        full_set_bucket_map = build_class_enc_to_train_nshot_bucket(
            split_name=self.partition_pipes[self.bucket_partition_name].cfg.split_name,
            dataset=self.partition_pipes[self.bucket_partition_name].cfg.dataset,
            cid2enc=full_set_cid2enc,
        ) if self.bucket_partition_name is not None else {}

        for partition_name in self.partition_names:
            pipe = self.partition_pipes[partition_name]
            artifacts_partition, loss_avg_partition = pipe.collect_eval_artifacts(modelw)
            partition_artifacts[partition_name] = artifacts_partition
            partition_losses[partition_name] = loss_avg_partition

        full_set_embs_img = torch.cat(
            [partition_artifacts[partition_name]["embs_img"] for partition_name in self.partition_names],
            dim=0,
        )
        full_set_class_encs_img = torch.cat(
            [
                SplitPartitionEvalPipeline._encode_cids(
                    partition_artifacts[partition_name]["cids_img"],
                    full_set_cid2enc,
                    full_set_embs_img.device,
                )
                for partition_name in self.partition_names
            ],
            dim=0,
        )
        full_set_embs_text = torch.cat(
            [partition_artifacts[partition_name]["embs_text"] for partition_name in self.partition_names],
            dim=0,
        )
        full_set_class_encs_text = torch.cat(
            [
                SplitPartitionEvalPipeline._encode_cids(
                    partition_artifacts[partition_name]["cids_text"],
                    full_set_cid2enc,
                    full_set_embs_img.device,
                ).cpu()
                for partition_name in self.partition_names
            ],
            dim=0,
        )

        img_offset = 0
        for partition_name in self.partition_names:
            pipe = self.partition_pipes[partition_name]
            artifacts_partition = partition_artifacts[partition_name]
            loss_avg_partition = partition_losses[partition_name]

            standard_scores = pipe.compute_map_scores(
                embs_img_q=artifacts_partition["embs_img"],
                class_encs_img_q=artifacts_partition["class_encs_img"],
                embs_text_q=artifacts_partition["embs_text"],
                class_encs_text_q=artifacts_partition["class_encs_text"].to(artifacts_partition["embs_img"].device),
                embs_img_g=artifacts_partition["embs_img"],
                class_encs_img_g=artifacts_partition["class_encs_img"],
                embs_text_g=artifacts_partition["embs_text"],
                class_encs_text_g=artifacts_partition["class_encs_text"],
                class_enc_to_bucket=pipe.class_enc_to_bucket if partition_name == self.bucket_partition_name else None,
                nshot_bucket_names=pipe.nshot_bucket_names if partition_name == self.bucket_partition_name else None,
            )

            full_set_class_encs_img_q = SplitPartitionEvalPipeline._encode_cids(
                artifacts_partition["cids_img"],
                full_set_cid2enc,
                artifacts_partition["embs_img"].device,
            )
            full_set_class_encs_text_q = SplitPartitionEvalPipeline._encode_cids(
                artifacts_partition["cids_text"],
                full_set_cid2enc,
                artifacts_partition["embs_img"].device,
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
                class_encs_img_q=full_set_class_encs_img_q,
                embs_text_q=artifacts_partition["embs_text"],
                class_encs_text_q=full_set_class_encs_text_q,
                embs_img_g=full_set_embs_img,
                class_encs_img_g=full_set_class_encs_img,
                embs_text_g=full_set_embs_text,
                class_encs_text_g=full_set_class_encs_text,
                self_match_idxs_g=self_match_idxs_g,
                class_enc_to_bucket=full_set_bucket_map if partition_name == self.bucket_partition_name else None,
                nshot_bucket_names=self.nshot_bucket_names if partition_name == self.bucket_partition_name else None,
            )

            scores_partition = {
                "standard": {
                    **standard_scores["standard"],
                    "full_set": full_set_scores["standard"],
                },
                "per_class": {
                    **standard_scores["per_class"],
                    "full_set": full_set_scores["per_class"],
                },
            }

            partition_map = compute_partition_map(scores_partition)
            partition_macro_map = compute_partition_macro_map(scores_partition)
            partition_maps.append(partition_map)
            partition_macro_maps.append(partition_macro_map)
            partition_i2i_maps.append(scores_partition["standard"]["map"]["i2i"])
            partition_i2i_macro_maps.append(scores_partition["per_class"]["map"]["i2i"])
            partition_full_set_map = compute_partition_full_set_map(scores_partition)
            partition_full_set_macro_map = compute_partition_full_set_macro_map(scores_partition)
            partition_full_set_maps.append(partition_full_set_map)
            partition_full_set_macro_maps.append(partition_full_set_macro_map)
            partition_full_set_i2i_maps.append(scores_partition["standard"]["full_set"]["map"]["i2i"])
            partition_full_set_i2i_macro_maps.append(scores_partition["per_class"]["full_set"]["map"]["i2i"])

            scores_val[partition_name] = {
                **scores_partition,
                "loss": loss_avg_partition,
            }

        comp_map = sum(partition_maps) / len(partition_maps)
        comp_macro_map = sum(partition_macro_maps) / len(partition_macro_maps)
        i2i_map = sum(partition_i2i_maps) / len(partition_i2i_maps)
        i2i_macro_map = sum(partition_i2i_macro_maps) / len(partition_i2i_macro_maps)
        full_set_comp_map = sum(partition_full_set_maps) / len(partition_full_set_maps)
        full_set_comp_macro_map = sum(partition_full_set_macro_maps) / len(partition_full_set_macro_maps)
        full_set_i2i_map = sum(partition_full_set_i2i_maps) / len(partition_full_set_i2i_maps)
        full_set_i2i_macro_map = sum(partition_full_set_i2i_macro_maps) / len(partition_full_set_i2i_macro_maps)

        scores_val["comp"] = {
            "standard": {
                "map": {
                    "all": comp_map,
                    "i2i": i2i_map,
                },
                "full_set": {
                    "map": {
                        "all": full_set_comp_map,
                        "i2i": full_set_i2i_map,
                    },
                },
            },
            "per_class": {
                "map": {
                    "all": comp_macro_map,
                    "i2i": i2i_macro_map,
                },
                "full_set": {
                    "map": {
                        "all": full_set_comp_macro_map,
                        "i2i": full_set_i2i_macro_map,
                    },
                },
            },
        }
        for p_name, p_map, p_macro_map in zip(self.partition_names, partition_maps, partition_macro_maps):
            scores_val["comp"]["standard"]["map"][p_name] = p_map
            scores_val["comp"]["per_class"]["map"][p_name] = p_macro_map
        for p_name, p_map, p_macro_map in zip(self.partition_names, partition_full_set_maps, partition_full_set_macro_maps):
            scores_val["comp"]["standard"]["full_set"]["map"][p_name] = p_map
            scores_val["comp"]["per_class"]["full_set"]["map"][p_name] = p_macro_map

        is_best_comp, is_best_i2i = self.check_bests(comp_map, i2i_map)
        time_elapsed_val = time.time() - time_start

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