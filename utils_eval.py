import torch  # type: ignore[import]
from torch.amp import autocast  # type: ignore[import]
import time
from tqdm import tqdm  # type: ignore[import]

from utils_data import spawn_dataloader, spawn_indexes, spawn_indexes_txts

import pdb


def compute_map_img2img(embs_imgs, classes_enc_imgs):
    """
    Vectorized mAP for evaluating image-to-image retrieval.
    Each image is queried against all others.
    Note: Tensors stay on CPU for this ~ it's safe, it's fast (enough)

    Args:
    - embs_imgs ---------- [Tensor(Q, D)] --- Image embeddings
    - classes_enc_imgs --- [Tensor(Q)] ------ Image class encodings (corresponding to image embeddings; 1D tensor of integers)

    Returns:
    - [float] ------------------------------- Mean Average Precision (mAP) over all image queries
    """

    Q = embs_imgs.size(0)  # num. queries
    N = Q - 1  # num. candidate-samples
    
    # full similarity matrix
    sim = embs_imgs @ embs_imgs.t()  # ------------------------------------------------------ Tensor(Q, Q)
    # mask self-similarity
    diag_idx                = torch.arange(Q, device=embs_imgs.device)
    sim[diag_idx, diag_idx] = float('-inf')

    # get top-N neighbors per query (all of them except query image i.e. N = Q - 1)
    _, idxs = sim.topk(N, dim=1)  # --------------------------------------------------------- Tensor(Q, N)

    # positives mask / boolean relevance mask (True wherever the query-image class matches the candidate-image class)
    pos_mask = classes_enc_imgs.unsqueeze(1) == classes_enc_imgs[idxs]  # ------------------- Tensor(Q, N)
    
    ranks    = torch.arange(1, N+1, device=embs_imgs.device)  # ----------------------------- Tensor(N)
    cum_prec = pos_mask.cumsum(dim=1).float() / ranks  # cumulative precision @ each rank --- Tensor(Q, N)
    
    # compute AP per query: sum(precision@hit) / num. positives (avoid div-by-zero, will produce NaN where pos_counts == 0)
    pos_counts = pos_mask.sum(dim=1).float()  # --------------------------------------------- Tensor(Q)
    ap         = (cum_prec * pos_mask.float()).sum(dim=1) / pos_counts  # ------------------- Tensor(Q)

    # mean over queries with >= 1 positive
    valid     = ~torch.isnan(ap)  # --------------------------------------------------------- Tensor(Q)
    map_score = ap[valid].mean().item()

    return map_score

def compute_map_cross_modal(embs_queries, classes_enc_queries, embs_cands, classes_enc_cands):
    """
    Vectorized mAP for evaluating cross-modal retrieval (image-to-text & text-to-image)
    Note: Tensors stay on CPU for this ~ it's safe, it's fast (enough)
    
    Args:
    - embs_queries ---------- [Tensor(Q, D)] --- Query embeddings
    - classes_enc_queries --- [Tensor(Q)] ------ Query class encodings (corresponding to query embeddings)
    - embs_cands ------------ [Tensor(N, D)] --- Candidate embeddings
    - classes_enc_cands ----- [Tensor(N)] ------ Candidate class encodings (corresponding to candidate embeddings)

    Returns:
    - [float] ---------------------------------- Mean Average Precision (mAP) over all queries
    """
    
    N = embs_cands.size(0)  # num. candidate-samples

    # full similarity matrix
    sim = embs_queries @ embs_cands.t()  # -------------------------------------------------- Tensor(Q, N)

    # get top-N neighbors per query (all of them)
    _, idxs = sim.topk(N, dim=1)  # --------------------------------------------------------- Tensor(Q, N)

    # positives mask / boolean relevance mask (True wherever the query class matches the candidate class)
    pos_mask = classes_enc_queries.unsqueeze(1) == classes_enc_cands[idxs]  # --------------- Tensor(Q, N)

    ranks    = torch.arange(1, N+1, device=sim.device)  # ----------------------------------- Tensor(N)
    cum_prec = pos_mask.cumsum(dim=1).float() / ranks  # cumulative precision @ each rank --- Tensor(Q, N)

    # compute AP per query: sum(precision@hit) / num. positives
    pos_counts = pos_mask.sum(dim=1).float()  # --------------------------------------------- Tensor(Q)
    ap         = (cum_prec * pos_mask.float()).sum(dim=1) / pos_counts  # ------------------- Tensor(Q)

    map_score = ap.mean().item()

    return map_score

class EvaluationPipeline:

    def __init__(
            self, 
            split_type, 
            split_name, 
            text_preps,
            img_pp,
            cached_imgs,
            batch_size,
            n_workers,
            prefetch_factor,
        ):

        assert all(len(text_preps_cat) == 1 for text_preps_cat in text_preps), "text_preps: each inner list must contain exactly one element for eval"

        self.split_type = split_type
        self.batch_size = batch_size

        index_class_encs, index_rfpaths, index_sids, sid_2_class_enc, index_pos, index_sex = spawn_indexes(
            split_name=split_name,
            split_type=split_type,
        )
        self.index_txts, self.index_txts_class_enc = spawn_indexes_txts(
            sid_2_class_enc=sid_2_class_enc,
            text_preps     =text_preps,
        )

        self.dataloader, self.time_cache = spawn_dataloader(
            index_class_encs=index_class_encs,
            index_rfpaths   =index_rfpaths,
            index_sids      =index_sids,
            index_pos       =index_pos,
            index_sex       =index_sex,
            text_preps      =text_preps,
            batch_size      =batch_size,
            shuffle         =False,
            drop_last       =False,
            img_pp          =img_pp,
            cached_imgs     =cached_imgs,
            n_workers       =n_workers,
            prefetch_factor =prefetch_factor,
        )

    def evaluate_split(self, modelw, verbose_batch_loss=False):
        time_start = time.time()
        modelw.model.eval()

        eval_scores = {}  # return structure

        with torch.no_grad(), autocast(device_type=modelw.device.type):
            embs_txts_all = modelw.embed_texts(self.index_txts)  # --- Tensor(L, D)

        embs_imgs        = []
        classes_enc_imgs = []
        loss_total       = 0.0
        n_samples_loss   = 0  # only full batches accumulated for loss computation
        n_correct        = 0
        for imgs_b, targ_classes_enc_b, texts_b, rank_keys_b, sids_b in tqdm(self.dataloader, desc=f"Eval ({self.split_type})", leave=False):
            imgs_b = imgs_b.to(modelw.device, non_blocking=True)

            with torch.no_grad(), autocast(device_type=modelw.device.type):
                embs_imgs_b = modelw.embed_images(imgs_b)  # --------- Tensor(B, D)
                B           = imgs_b.size(0)

            embs_imgs.append(embs_imgs_b.cpu())
            classes_enc_imgs.append(torch.tensor(targ_classes_enc_b, dtype=torch.long))

            """
            maybe wait until the end to do the Prec@1 computation so you can divide them up by n-shot buckets
            """
            pred_classes_enc_txts_b, _ = modelw.img2txt_classify(embs_imgs_b, embs_txts_all, self.index_txts_class_enc)

            n_correct_b = sum(p == t for p, t in zip(pred_classes_enc_txts_b, targ_classes_enc_b))
            n_correct += n_correct_b

            if B == self.batch_size:  # only compute loss for full batches
                with torch.no_grad(), autocast(device_type=modelw.device.type):
                    embs_txts_b = modelw.embed_texts(texts_b)
                    sim_b       = embs_imgs_b @ embs_txts_b.T
                    logits_b    = modelw.compute_logits(sim_b)
                    loss_b      = modelw.compute_batch_loss(logits_b, targ_classes_enc_b, rank_keys_b, sids_b)

                batch_loss = loss_b.detach().item() * B
                loss_total += batch_loss
                n_samples_loss += B
                if verbose_batch_loss:
                    print(f"Batch Loss: {batch_loss:.4f}")

        # prepare image embedding and class encoding tensors for mAP computation
        embs_imgs        = torch.cat(embs_imgs, dim=0)  # ---------- Tensor(Q, D)
        classes_enc_imgs = torch.cat(classes_enc_imgs, dim=0)  # --- Tensor(Q)

        # img2txt precision@1 computation
        n_samps                      = len(self.dataloader.dataset)
        prec1_img2txt                = n_correct / n_samps
        eval_scores["img2txt_prec1"] = prec1_img2txt

        # img2txt mAP computation
        map_img2txt = compute_map_cross_modal(
            embs_imgs, 
            classes_enc_imgs, 
            embs_txts_all.cpu(), 
            torch.tensor(self.index_txts_class_enc),
        )
        eval_scores["img2txt_map"] = map_img2txt

        # img2img mAP computation
        map_img2img = compute_map_img2img(
            embs_imgs, 
            classes_enc_imgs,
        )
        eval_scores["img2img_map"] = map_img2img

        # txt2img mAP computation
        map_txt2img = compute_map_cross_modal(
            embs_txts_all.cpu(), 
            torch.tensor(self.index_txts_class_enc), 
            embs_imgs, 
            classes_enc_imgs,
        )
        eval_scores["txt2img_map"] = map_txt2img

        # loss aggregation
        loss_avg = loss_total / n_samples_loss

        modelw.model.train()

        time_end = time.time()
        time_elapsed = time_end - time_start

        return eval_scores, loss_avg, time_elapsed

class ValidationPipeline:

    def __init__(
            self,
            split_name,
            text_preps,
            batch_size,
            img_pp,
            cached_imgs,
            n_workers,
            prefetch_factor,
            header_tag=None,
        ):

        self.header_tag = header_tag

        self.best_comp_map    = None
        self.best_img2img_map = None

        self.val_pipe_id = EvaluationPipeline(
            split_type     ="id_val",
            split_name     =split_name,
            text_preps     =text_preps,
            img_pp         =img_pp,
            cached_imgs    =cached_imgs,
            batch_size     =batch_size,
            n_workers      =n_workers,
            prefetch_factor=prefetch_factor,
        )

        self.val_pipe_ood = EvaluationPipeline(
            split_type     ="ood_val",
            split_name     =split_name,
            text_preps     =text_preps,
            img_pp         =img_pp,
            cached_imgs    =cached_imgs,
            batch_size     =batch_size,
            n_workers      =n_workers,
            prefetch_factor=prefetch_factor,
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

    def set_time_cache(self):
        if self.val_pipe_id.time_cache is not None:
            self.time_cache = self.val_pipe_id.time_cache + self.val_pipe_ood.time_cache
        else:
            self.time_cache = None

    def run_validation(self, modelw, verbose=True, verbose_batch_loss=False):
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

    def check_bests(self, comp_map, img2img_map):
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

    def print_val(self, scores):

        header = " Validation "
        if self.header_tag is not None:
            header += f"({self.header_tag}) "

        print(
            f"{header:=^{75}}",
            f"ID img2txt mAP ------ {scores['id_img2txt_map']:.4f}",
            f"ID img2img mAP ------ {scores['id_img2img_map']:.4f}",
            f"ID txt2img mAP ------ {scores['id_txt2img_map']:.4f}",
            f"OOD img2txt mAP ----- {scores['ood_img2txt_map']:.4f}",
            f"OOD img2img mAP ----- {scores['ood_img2img_map']:.4f}",
            f"OOD txt2img mAP ----- {scores['ood_txt2img_map']:.4f}",
            f"{'':-^{75}}",
            f"Composite mAP ------- {scores['comp_map']:.4f} (best: {self.best_comp_map:.4f})",
            f"img2img mAP --------- {scores['img2img_map']:.4f} (best: {self.best_img2img_map:.4f})",
            f"{'':-^{75}}",
            f"ID Loss ----- {scores['id_loss']:.4f}",
            f"OOD Loss ---- {scores['ood_loss']:.4f}",
            f"Comp Loss --- {scores['comp_loss']:.4f}",
            f"",
            sep="\n"
        )
