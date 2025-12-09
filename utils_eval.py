import torch  # type: ignore[import]
from torch.amp import autocast  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]
import time
from tqdm import tqdm  # type: ignore[import]
from typing import Tuple, Any, List, Callable, Dict, Union, Optional

from utils_data import spawn_dataloader, spawn_indexes, spawn_indexes_txts
from utils_head import compute_sim

import pdb


class SplitSetEvalPipeline:

    def __init__(
            self, 
            splitset_name: str, 
            config:        Any, 
            text_preps:    List[List[str]],
            img_pp:        Callable,
        ) -> None:

        assert all(len(text_preps_cat) == 1 for text_preps_cat in text_preps), \
               "text_preps: each inner list must contain exactly one element for eval"

        index_data, sid_2_class_enc = spawn_indexes(
            split_name   =config.split_name,
            splitset_name=splitset_name,
        )
        self.index_text, self.index_text_class_encs = spawn_indexes_txts(
            sid_2_class_enc=sid_2_class_enc,
            text_preps     =text_preps,
        )

        self.dataloader, self.time_cache = spawn_dataloader(
            index_data    =index_data,
            text_preps    =text_preps,
            config        =config,
            shuffle       =False,
            drop_last     =False,
            img_pp        =img_pp,
            use_dv_sampler=False,
        )

        self.cfg           = config
        self.splitset_name = splitset_name
        self.batch_size    = self.dataloader.batch_size
        self.mixed_prec    = config.hw.mixed_prec

    @torch.no_grad()
    def evaluate_split(
        self, 
        modelw:             Any, 
        verbose_batch_loss: bool = False
    ) -> Tuple[Dict[str, float], float, float]:
        
        time_start = time.time()
        modelw.model.eval()

        eval_scores = {}

        # text embeddings
        if self.mixed_prec:
            with autocast(device_type=modelw.device.type):
                embs_text_all = modelw.embed_texts(self.index_text)  # --- Tensor(L, D)
        else:
            embs_text_all = modelw.embed_texts(self.index_text)
        
        # image embeddings
        embs_imgs      = []
        class_encs_img = []
        loss_total     = 0.0
        n_samps_loss   = 0  # only full batches accumulated for loss computation
        n_correct      = 0
        n_samps        = 0
        for imgs_b, texts_b, class_encs_img_b, targ_data_b in tqdm(self.dataloader, desc=f"Eval ({self.splitset_name})", leave=False):
            imgs_b       = imgs_b.to(modelw.device, non_blocking=True)
            class_encs_img_b = class_encs_img_b.to(modelw.device, non_blocking=True)

            B = imgs_b.size(0)

            is_full_local  = 1.0 if B == self.batch_size else 0.0  # 1.0 if full sub-batch, 0.0 if partial
            is_full_tensor = torch.tensor([is_full_local], device=modelw.device)

            dist.all_reduce(is_full_tensor, op=dist.ReduceOp.MIN)  # is_full_tensor is 0.0 if any rank has partial sub-batch
            is_full_batch = (is_full_tensor.item() == 1.0)

            if self.mixed_prec:
                with autocast(device_type=modelw.device.type):
                    loss, embs_img_b = modelw.batch_step(
                        imgs_b,
                        texts_b,
                        class_encs_img_b,
                        targ_data_b,
                        loss_flag=is_full_batch,
                    )
            else:
                loss, embs_img_b = modelw.batch_step(
                    imgs_b,
                    texts_b,
                    class_encs_img_b,
                    targ_data_b,
                    loss_flag=is_full_batch,
                )

            embs_imgs.append(embs_img_b.cpu())
            class_encs_img.append(class_encs_img_b.cpu())

            """
            maybe wait until the end to do the Prec@1 computation so you can divide them up by n-shot buckets
            """
            pred_classes_enc_txts_b = modelw.img2txt_classify(embs_img_b, embs_text_all, self.index_text_class_encs)

            n_correct_b = sum(p == t for p, t in zip(pred_classes_enc_txts_b, class_encs_img_b))
            n_correct += n_correct_b
            n_samps += B

            if B == self.batch_size:  # only compute loss for full batches
                batch_loss = loss.detach().item() * B
                loss_total += batch_loss
                n_samps_loss += B
                if verbose_batch_loss:
                    print(f"Batch Loss: {batch_loss:.4f}")

        # local concatenation
        embs_img_local  = torch.cat(embs_imgs, dim=0).to(modelw.device)  # ------------- Tensor(Q/G, D)
        class_encs_img_local = torch.cat(class_encs_img, dim=0).to(modelw.device)  # --- Tensor(Q/G)

        map_i2t, map_i2i, map_t2i  = self.compute_map_scores(embs_img_local, class_encs_img_local, embs_text_all)
        eval_scores["img2txt_map"] = map_i2t
        eval_scores["img2img_map"] = map_i2i
        eval_scores["txt2img_map"] = map_t2i

        # reduce scalars (loss & accuracy)
        stats = torch.tensor([
            loss_total,
            n_samps_loss,
            n_correct,
            n_samps,
        ], device=modelw.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        loss_total   = stats[0].item()
        n_samps_loss = stats[1].item()
        n_correct    = stats[2].item()
        n_samps      = stats[3].item()

        prec1_img2txt                = n_correct / n_samps
        eval_scores["img2txt_prec1"] = prec1_img2txt

        loss_avg = loss_total / n_samps_loss

        modelw.model.train()

        time_end = time.time()
        time_elapsed = time_end - time_start

        return eval_scores, loss_avg, time_elapsed

    def compute_map_scores(
        self, 
        embs_img_local:       torch.Tensor, 
        class_encs_img_local: torch.Tensor, 
        embs_text_all:        torch.Tensor
    ) -> Tuple[float, float, float]:

        # distributed gathering
        world_size          = dist.get_world_size()
        embs_img_gath       = [torch.zeros_like(embs_img_local) for _ in range(world_size)]
        class_encs_img_gath = [torch.zeros_like(class_encs_img_local) for _ in range(world_size)]
        dist.all_gather(embs_img_gath, embs_img_local)
        dist.all_gather(class_encs_img_gath, class_encs_img_local)

        embs_img_global       = torch.cat(embs_img_gath, dim=0)
        class_encs_img_global = torch.cat(class_encs_img_gath, dim=0)

        map_i2t = self.compute_map_cross_modal(
            embs_q      =embs_img_global.cpu(),
            class_encs_q=class_encs_img_global.cpu(),
            embs_g      =embs_text_all.cpu(),
            class_encs_g=self.index_text_class_encs,
        )
        map_i2i = self.compute_map_img2img(
            embs_q      =embs_img_local,
            class_encs_q=class_encs_img_local,
            embs_g      =embs_img_global,
            class_encs_g=class_encs_img_global,
        )
        map_t2i = self.compute_map_cross_modal(
            embs_q      =embs_text_all.cpu(),
            class_encs_q=self.index_text_class_encs,
            embs_g      =embs_img_global.cpu(),
            class_encs_g=class_encs_img_global.cpu(),
        )

        return map_i2t, map_i2i, map_t2i

    def compute_map_img2img(
        self,
        embs_q:       torch.Tensor,
        class_encs_q: torch.Tensor,
        embs_g:       torch.Tensor,
        class_encs_g: torch.Tensor,
        chunk_size:   int = 16384,
    ) -> float:
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
        - [float] -------- Image-to-image Mean Average Precision (mAP) over all non-singleton queries
        """
        rank_offset = dist.get_rank() * embs_q.size(0)

        Q = embs_q.size(0)  # num. query images
        N = embs_g.size(0)  # num. gallery images
        
        ap_sum_local = 0.0
        n_nsq_local  = 0  # num. non-singleton queries
        for i in range(0, Q, chunk_size):
            i_end = min(i + chunk_size, Q)
            
            embs_q_chunk       = embs_q[i:i_end]
            class_encs_q_chunk = class_encs_q[i:i_end]
            
            # chunked similarity matrix (chunk x all)
            sim_chunk = compute_sim(embs_q_chunk, embs_g, "cos")  # --------------------------- Tensor(U, N)

            # mask self-similarity
            sim_chunk[torch.arange(i_end - i), torch.arange(rank_offset + i, rank_offset + i_end)] = float('-inf')

            # sorted indices of top-N neighbors per query (all of them except query image itself)
            _, idxs_chunk = sim_chunk.sort(dim=1, descending=True)
            # positives mask / boolean relevance mask (True wherever the query-image class matches the gallery-image class)
            pos_mask_chunk = class_encs_q_chunk.unsqueeze(1) == class_encs_g[idxs_chunk]  # --- Tensor(U, N)
            # Remove self-match column
            pos_mask_chunk = pos_mask_chunk[:, :N-1]  # --------------------------------------- Tensor(U, N)  
            
            # cumulative precision at each rank
            ranks    = torch.arange(1, N, device=torch.device("cuda")).float()  # ------------- Tensor(N-1)
            cum_prec = pos_mask_chunk.cumsum(dim=1).float() / ranks  # ------------------------ Tensor(U, N-1)

            # compute AP per query: sum(precision@hit) / num. positives
            pos_counts_chunk = pos_mask_chunk.sum(dim=1).float()  # --------------------------- Tensor(U)
            
            ap_chunk = (cum_prec * pos_mask_chunk.float()).sum(dim=1)  # ---------------------- Tensor(U)
            
            # avoid div-by-zero for queries with no positives
            has_pos = pos_counts_chunk > 0
            # only divide where pos_counts_chunk > 0
            ap_chunk[has_pos] /= pos_counts_chunk[has_pos]
            
            ap_sum_local += ap_chunk[has_pos].sum().item()
            n_nsq_local += has_pos.sum().item()
            
            # free up VRAM
            del embs_q_chunk, class_encs_q_chunk, sim_chunk, idxs_chunk, pos_mask_chunk, cum_prec
        
        stats = torch.tensor([
            ap_sum_local,
            n_nsq_local,
        ], device=torch.device("cuda"))
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        ap_sum_global = stats[0].item()
        n_nsq_global  = stats[1].item()

        map_i2i = ap_sum_global / n_nsq_global

        return map_i2i

    def compute_map_cross_modal(
        self, 
        embs_q:       torch.Tensor, 
        class_encs_q: torch.Tensor, 
        embs_g:       torch.Tensor, 
        class_encs_g: torch.Tensor,
    ) -> float:
        """
        Vectorized mAP for evaluating cross-modal retrieval (image-to-text & text-to-image)
        Note: Tensors stay on CPU for this ~ it's safe, it's fast (enough)
        
        Args:
        - embs_q --------- Query embeddings
        - class_encs_q --- Query class encodings (corresponding to query embeddings)
        - embs_g --------- Gallery embeddings
        - class_encs_g --- Gallery class encodings (corresponding to gallery embeddings)

        Returns:
        - [float] -------- Mean Average Precision (mAP) over all queries
        """

        device       = embs_q.device
        class_encs_g = class_encs_g.to(device)
        class_encs_q = class_encs_q.to(device)

        N = embs_g.size(0)  # num. gallery embeddings

        # full similarity matrix
        sim = compute_sim(embs_q, embs_g, "cos")  # ----------------------------- Tensor(Q, N)

        # get top-N neighbors per query (all of them)
        _, idxs = sim.topk(N, dim=1)  # ----------------------------------------- Tensor(Q, N)

        # positives mask / boolean relevance mask (True wherever the query class matches the candidate class)
        pos_mask = class_encs_q.unsqueeze(1) == class_encs_g[idxs]  # ----------- Tensor(Q, N)

        # cumulative precision at each rank
        ranks    = torch.arange(1, N+1, device=sim.device)  # ------------------- Tensor(N)
        cum_prec = pos_mask.cumsum(dim=1).float() / ranks  # -------------------- Tensor(Q, N)

        # compute AP per query: sum(precision@hit) / num. positives
        pos_counts = pos_mask.sum(dim=1).float()  # ----------------------------- Tensor(Q)
        ap         = (cum_prec * pos_mask.float()).sum(dim=1) / pos_counts  # --- Tensor(Q)

        map_score = ap.mean().item()

        return map_score

class ValidationPipeline:

    def __init__(
        self,
        config:     Any,
        text_preps: List[List[str]],
        img_pp:     Callable,
        header_tag: Optional[str] = None,
    ):

        self.header_tag = header_tag

        self.best_comp_map    = None
        self.best_img2img_map = None

        self.val_pipe_id = SplitSetEvalPipeline(
            splitset_name="id_val",
            config       =config,
            text_preps   =text_preps,
            img_pp       =img_pp,
        )

        self.val_pipe_ood = SplitSetEvalPipeline(
            splitset_name="ood_val",
            config       =config,
            text_preps   =text_preps,
            img_pp       =img_pp,
        )

        self.set_time_cache()

        self.scores_tracker = {
            "id_img2txt_prec1":  [],
            "id_img2txt_map":    [],
            "id_img2img_map":    [],
            "id_txt2img_map":    [],
            "ood_img2txt_prec1": [],
            "ood_img2txt_map":   [],
            "ood_img2img_map":   [],
            "ood_txt2img_map":   [],
            "comp_map":          [],
            "img2img_map":       [],
        }

    def set_time_cache(self) -> None:
        if self.val_pipe_id.time_cache is not None:
            self.time_cache = self.val_pipe_id.time_cache + self.val_pipe_ood.time_cache
        else:
            self.time_cache = None

    def run_validation(
        self, 
        modelw:             Any, 
        verbose:            bool = True, 
        verbose_batch_loss: bool = False,
    ) -> Tuple[Dict[str, float], bool, bool, float]:
        """
        `is_best` param in the return will dictate when models are saved (early stopping)
        """

        scores_id, loss_avg_id, time_elapsed_id    = self.val_pipe_id.evaluate_split(modelw, verbose_batch_loss)
        scores_ood, loss_avg_ood, time_elapsed_ood = self.val_pipe_ood.evaluate_split(modelw, verbose_batch_loss)

        comp_map = (scores_id["img2txt_map"] + \
                    scores_id["img2img_map"] + \
                    scores_id["txt2img_map"] + \
                    scores_ood["img2txt_map"] + \
                    scores_ood["img2img_map"] + \
                    scores_ood["txt2img_map"]) / 6
        img2img_map = (scores_id["img2img_map"] + scores_ood["img2img_map"]) / 2

        is_best_comp, is_best_img2img = self.check_bests(comp_map, img2img_map)

        scores_val = {
            "id_img2txt_prec1":  scores_id["img2txt_prec1"],
            "id_img2txt_map":    scores_id["img2txt_map"],
            "id_img2img_map":    scores_id["img2img_map"],
            "id_txt2img_map":    scores_id["txt2img_map"],
            "id_loss":           loss_avg_id,
            "ood_img2txt_prec1": scores_ood["img2txt_prec1"],
            "ood_img2txt_map":   scores_ood["img2txt_map"],
            "ood_img2img_map":   scores_ood["img2img_map"],
            "ood_txt2img_map":   scores_ood["txt2img_map"],
            "ood_loss":          loss_avg_ood,
            "comp_map":          comp_map,
            "img2img_map":       img2img_map,
            "comp_loss":         (loss_avg_id + loss_avg_ood) / 2
        }
        if verbose:
            self.print_val(scores_val)

        time_elapsed_val = time_elapsed_id + time_elapsed_ood

        return scores_val, is_best_comp, is_best_img2img, time_elapsed_val

    def check_bests(self, comp_map: float, img2img_map: float) -> Tuple[bool, bool]:
        is_best_comp, is_best_img2img = False, False
        if self.best_comp_map is None:
            self.best_comp_map = comp_map
            self.best_img2img_map = img2img_map
        else:
            if comp_map > self.best_comp_map:
                self.best_comp_map = comp_map
                is_best_comp = True
            if img2img_map > self.best_img2img_map:
                self.best_img2img_map = img2img_map
                is_best_img2img = True
        return is_best_comp, is_best_img2img

    def print_val(self, scores: Dict[str, float]) -> None:

        header = " Validation "
        if self.header_tag is not None:
            header += f"({self.header_tag}) "

        print(
            f"{header:=^{75}}",
            f"ID img2txt mAP ---- {scores['id_img2txt_map']:.4f}",
            f"ID img2img mAP ---- {scores['id_img2img_map']:.4f}",
            f"ID txt2img mAP ---- {scores['id_txt2img_map']:.4f}",
            f"OOD img2txt mAP --- {scores['ood_img2txt_map']:.4f}",
            f"OOD img2img mAP --- {scores['ood_img2img_map']:.4f}",
            f"OOD txt2img mAP --- {scores['ood_txt2img_map']:.4f}",
            f"{'':-^{75}}",
            f"Composite mAP --- {scores['comp_map']:.4f} (best: {self.best_comp_map:.4f})",
            f"img2img mAP ----- {scores['img2img_map']:.4f} (best: {self.best_img2img_map:.4f})",
            f"{'':-^{75}}",
            f"ID Loss ----- {scores['id_loss']:.4f}",
            f"OOD Loss ---- {scores['ood_loss']:.4f}",
            f"Comp Loss --- {scores['comp_loss']:.4f}",
            f"",
            sep="\n"
        )