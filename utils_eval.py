import torch
import torch.nn.functional as F
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

    embs_imgs = F.normalize(embs_imgs, dim=1)
    
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

def compute_map_txt2img(embs_txts, classes_enc_txts, embs_imgs, classes_enc_imgs):
    """
    Vectorized mAP for evaluating text-to-image retrieval.
    Note: Tensors stay on CPU for this ~ it's safe, it's fast (enough)
    
    Args:
    - embs_txts ---------- [Tensor(Q, D)] --- Text embeddings
    - classes_enc_txts --- [Tensor(Q)] ------ Text class encodings (corresponding to text embeddings)
    - embs_imgs ---------- [Tensor(N, D)] --- Image embeddings
    - classes_enc_imgs --- [Tensor(N)] ------ Image class encodings (corresponding to image embeddings)

    Returns:
    - [float] ------------------------------- Mean Average Precision (mAP) over all text queries
    """
    
    N = embs_imgs.size(0)  # num. corpus-samples

    embs_txts = F.normalize(embs_txts, dim=1)
    embs_imgs = F.normalize(embs_imgs, dim=1)

    # full similarity matrix
    sim = embs_txts @ embs_imgs.t()  # ------------------------------------------------------ Tensor(Q, N)

    # get top-N neighbors per query (all of them)
    _, idxs = sim.topk(N, dim=1)  # --------------------------------------------------------- Tensor(Q, N)

    # positives mask / boolean relevance mask (True wherever the query-text class matches the corpus-image class)
    pos_mask = classes_enc_txts.unsqueeze(1) == classes_enc_imgs[idxs]  # ------------------- Tensor(Q, N)

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
            model,
            cached_imgs,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
            prefetch_factor,
            modes=["img2txt", "img2img", "txt2img"],
        ):

        self.modes = modes

        index_imgs_class_enc, index_imgs_rfpaths, index_txts_sids = spawn_indexes_imgs(
            split_type=split_type,
            split_name=split_name,
        )
        self.index_txts, self.index_txts_class_enc = spawn_indexes_txts(
            index_txts_sids=index_txts_sids,
            text_base_type=text_base_type,
            text_prep_type=text_prep_type,
        )

        self.loader = spawn_dataloader(
            index_imgs_class_enc=index_imgs_class_enc,
            index_imgs_rfpaths=index_imgs_rfpaths,
            img_pp=model.img_pp,
            cached_imgs=cached_imgs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    def eval(self, model):

        # return structures
        eval_scores = {}
        eval_times = {}

        time_start = time.time()

        if "img2img" in self.modes or "txt2img" in self.modes:
            embs_imgs        = []
            classes_enc_imgs = []

        embs_txts = model.embed_texts(self.index_txts)  # --- Tensor(L, D)

        n_correct = 0
        for imgs_b, targ_classes_enc_b in tqdm(self.loader, desc="Image-to-Text Eval (ID)", leave=False):

            embs_imgs_b = model.embed_images(imgs_b)  # --- Tensor(B, D)

            if "img2img" in self.modes or "txt2img" in self.modes:
                embs_imgs.append(embs_imgs_b.cpu())
                classes_enc_imgs.append(torch.tensor(targ_classes_enc_b, dtype=torch.long))

            if "img2txt" in self.modes:
                pred_classes_enc_txts_b, _ = model.img2txt_classify(embs_imgs_b, embs_txts, self.index_txts_class_enc)

                n_correct_b = sum(p == t for p, t in zip(pred_classes_enc_txts_b, targ_classes_enc_b))
                n_correct += n_correct_b

        if "img2txt" in self.modes:
            # img2txt precision@1 computation
            n_samps = len(self.loader.dataset)
            prec1_img2txt   = n_correct / n_samps
            
            time_end = time.time()
            time_elapsed_img2txt = time_end - time_start

            eval_scores["img2txt_prec1"] = prec1_img2txt
            eval_times["img2txt"] = time_elapsed_img2txt

        if "img2img" in self.modes or "txt2img" in self.modes:
            # prepare image embedding and class encoding tensors for img2img and txt2img mAP computation
            embs_imgs        = torch.cat(embs_imgs, dim=0)  # ---------- Tensor(Q, D)
            classes_enc_imgs = torch.cat(classes_enc_imgs, dim=0)  # --- Tensor(Q)

        if "img2img" in self.modes:
            time_start = time.time()

            map_img2img = compute_map_img2img(embs_imgs, classes_enc_imgs)

            time_end = time.time()
            time_elapsed_img2img = time_end - time_start

            eval_scores["img2img_map"] = map_img2img
            eval_times["img2img"] = time_elapsed_img2img

        if "txt2img" in self.modes:
            time_start = time.time()

            map_txt2img = compute_map_txt2img(embs_txts.cpu(), torch.tensor(self.index_txts_class_enc), embs_imgs, classes_enc_imgs)

            time_end = time.time()
            time_elapsed_txt2img = time_end - time_start

            eval_scores["txt2img_map"] = map_txt2img
            eval_times["txt2img"] = time_elapsed_txt2img

        return eval_scores, eval_times
