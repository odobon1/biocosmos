import torch
import torch.nn.functional as F

import pdb


def compute_map_img2img(embs, labels_enc):
    """
    Vectorized mAP for evaluating image-to-image retrieval.
    Each image is queried against all others.

    Args:
    - embs --------- [Tensor(Q, D)] --- Image embeddings
    - labels_enc --- [Tensor(Q)] ------ Label/Class encodings corresponding to image embeddings (1D tensor of integers)

    Returns:
    - [float] ------------------------- Mean Average Precision (mAP) over all image queries
    """

    Q = embs.size(0)  # num. queries
    N = Q - 1  # num. corpus-samples

    embs = F.normalize(embs, dim=1)
    
    # full similarity matrix
    sim = embs @ embs.t()
    # mask self-similarity
    diag_idx = torch.arange(Q, device=embs.device)
    sim[diag_idx, diag_idx] = float('-inf')
    
    # get top-N neighbors per query (all of them except query image i.e. N = Q - 1)
    _, idxs = sim.topk(N, dim=1)

    # positives mask / boolean relevance mask (True wherever the image-query label/class matches the corpus-image label/class)
    pos_mask = labels_enc.unsqueeze(1) == labels_enc[idxs]
    
    ranks = torch.arange(1, N+1, device=embs.device)
    cum_prec = pos_mask.cumsum(dim=1).float() / ranks  # cumulative precision @ each rank
    
    # compute AP per query: sum(precision@hit) / num. positives
    pos_counts = pos_mask.sum(dim=1).float()
    ap = (cum_prec * pos_mask.float()).sum(dim=1) / pos_counts  # avoid div-by-zero, will produce NaN where pos_counts == 0

    # mean over queries with >= 1 positive
    valid = ~torch.isnan(ap)
    map_score = ap[valid].mean().item()

    return map_score

def compute_map_txt2img(embs_txts, labels_enc_txts, embs_imgs, labels_enc_imgs):
    """
    Vectorized mAP for evaluating text-to-image retrieval.

    Args:
    - embs_txts --------- [Tensor(Q, D)] --- Text embeddings
    - labels_enc_txts --- [Tensor(Q)] ------ Label/Class encodings corresponding to text embeddings
    - embs_imgs --------- [Tensor(N, D)] --- Image embeddings
    - labels_enc_imgs --- [Tensor(N)] ------ Label/Class encodings corresponding to image embeddings

    Returns:
    - [float] ------------------------------ Mean Average Precision (mAP) over all text queries
    """
    
    N = embs_imgs.size(0)  # num. corpus-samples

    embs_txts = F.normalize(embs_txts, dim=1)
    embs_imgs = F.normalize(embs_imgs, dim=1)

    # full similarity matrix
    sim = embs_txts @ embs_imgs.t()  # --- Tensor(Q, N)

    # get top-N neighbors per query (all of them)
    _, idxs = sim.topk(N, dim=1)

    # positives mask / boolean relevance mask (True wherever the text-query label/class matches the corpus-image label/class)
    pos_mask = labels_enc_txts.unsqueeze(1) == labels_enc_imgs[idxs]

    ranks = torch.arange(1, N+1, device=sim.device)
    cum_prec = pos_mask.cumsum(dim=1).float() / ranks  # cumulative precision @ each rank

    # compute AP per query: sum(precision@hit) / num. positives
    pos_counts = pos_mask.sum(dim=1).float()
    ap = (cum_prec * pos_mask.float()).sum(dim=1) / pos_counts

    map_score = ap.mean().item()

    return map_score
