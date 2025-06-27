import torch
import torch.nn.functional as F

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
