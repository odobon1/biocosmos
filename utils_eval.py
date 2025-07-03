import torch
import torch.nn.functional as F
from torch.amp import autocast
import time
from tqdm import tqdm

from utils_data import spawn_dataloader, spawn_indexes_imgs, spawn_indexes_txts

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
    N = Q - 1  # num. corpus-samples
    
    # full similarity matrix
    sim = embs_imgs @ embs_imgs.t()  # ------------------------------------------------------ Tensor(Q, Q)
    # mask self-similarity
    diag_idx                = torch.arange(Q, device=embs_imgs.device)
    sim[diag_idx, diag_idx] = float('-inf')

    # get top-N neighbors per query (all of them except query image i.e. N = Q - 1)
    _, idxs = sim.topk(N, dim=1)  # --------------------------------------------------------- Tensor(Q, N)

    # positives mask / boolean relevance mask (True wherever the query-image class matches the corpus-image class)
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
    Originally: Vectorized mAP for evaluating text-to-image retrieval.
    Now: Generalized such that it can also be used for computing mAP (RR) for image-to-text retrieval.
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

    # positives mask / boolean relevance mask (True wherever the query class matches the corpus class)
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
            text_base_type, 
            text_prep_type,
            img_pp,
            cached_imgs,
            batch_size,
            num_workers,
            prefetch_factor,
            modes=["img2txt", "img2img", "txt2img"],
        ):

        self.split_type = split_type
        self.modes      = modes

        index_imgs_class_enc, index_imgs_rfpaths, sid_2_class_enc = spawn_indexes_imgs(
            split_type=split_type,
            split_name=split_name,
        )
        self.index_txts, self.index_txts_class_enc = spawn_indexes_txts(
            sid_2_class_enc=sid_2_class_enc,
            text_base_type =text_base_type,
            text_prep_type =text_prep_type,
        )

        self.loader = spawn_dataloader(
            index_imgs_class_enc=index_imgs_class_enc,
            index_imgs_rfpaths  =index_imgs_rfpaths,
            img_pp              =img_pp,
            cached_imgs         =cached_imgs,
            batch_size          =batch_size,
            shuffle             =False,
            num_workers         =num_workers,
            prefetch_factor     =prefetch_factor,
            index_txts=self.index_txts,
            index_txts_class_enc=self.index_txts_class_enc,
            drop_last=False,
        )

    def evaluate(self, modelw):
        time_start = time.time()
        modelw.model.eval()

        # return structure
        eval_scores = {}

        if "img2img" in self.modes or "txt2img" in self.modes:
            embs_imgs        = []
            classes_enc_imgs = []

        if "img2txt" in self.modes or "txt2img" in self.modes:
            with torch.no_grad(), autocast(device_type=modelw.device.type):
                embs_txts = modelw.embed_texts(self.index_txts)  # --- Tensor(L, D)

        n_correct = 0
        for imgs_b, targ_classes_enc_b, _ in tqdm(self.loader, desc=f"Eval ({self.split_type})", leave=False):
            imgs_b = imgs_b.to(modelw.device, non_blocking=True)

            with torch.no_grad(), autocast(device_type=modelw.device.type):
                embs_imgs_b = modelw.embed_images(imgs_b)  # --------- Tensor(B, D)

            if "img2img" in self.modes or "txt2img" in self.modes:
                embs_imgs.append(embs_imgs_b.cpu())
                classes_enc_imgs.append(torch.tensor(targ_classes_enc_b, dtype=torch.long))

            if "img2txt" in self.modes:
                """
                maybe wait until the end to do the Prec@1 computation so you can divide them up by n-shot buckets
                """
                pred_classes_enc_txts_b, _ = modelw.img2txt_classify(embs_imgs_b, embs_txts, self.index_txts_class_enc)

                n_correct_b = sum(p == t for p, t in zip(pred_classes_enc_txts_b, targ_classes_enc_b))
                n_correct += n_correct_b

        # prepare image embedding and class encoding tensors for mAP computation
        embs_imgs        = torch.cat(embs_imgs, dim=0)  # ---------- Tensor(Q, D)
        classes_enc_imgs = torch.cat(classes_enc_imgs, dim=0)  # --- Tensor(Q)

        if "img2txt" in self.modes:
            # img2txt precision@1 computation
            n_samps       = len(self.loader.dataset)
            prec1_img2txt = n_correct / n_samps

            eval_scores["img2txt_prec1"] = prec1_img2txt

            # img2txt mAP (RR) computation
            map_img2txt                = compute_map_cross_modal(embs_imgs, classes_enc_imgs, embs_txts.cpu(), torch.tensor(self.index_txts_class_enc))
            eval_scores["img2txt_map"] = map_img2txt

        if "img2img" in self.modes:
            map_img2img                = compute_map_img2img(embs_imgs, classes_enc_imgs)
            eval_scores["img2img_map"] = map_img2img

        if "txt2img" in self.modes:
            map_txt2img                = compute_map_cross_modal(embs_txts.cpu(), torch.tensor(self.index_txts_class_enc), embs_imgs, classes_enc_imgs)
            eval_scores["txt2img_map"] = map_txt2img

        modelw.model.train()

        time_end = time.time()
        time_elapsed = time_end - time_start

        return eval_scores, time_elapsed
