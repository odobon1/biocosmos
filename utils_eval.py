import torch
import torch.nn.functional as F
import faiss
from torchmetrics.retrieval import RetrievalMAP

import pdb


def compute_map_img2img(embeddings, labels):
    """
    Computes mAP score for evaluating image-to-image retrieval using each image embedding as a query against all others.

    Args:
    - embeddings [Tensor(Q, D)] --- The image embeddings
    - labels [Tensor(Q)] ---------- Labels/Classes (encoded as integers) corresponding to embeddings

    Returns:
    - [float] --------------------- mAP score
    """

    embs = F.normalize(embeddings, dim=1)
    Q = embs.size(0)
    
    # full similarity matrix
    sim = embs @ embs.t()
    # mask self-similarity
    diag_idx = torch.arange(Q, device=embs.device)
    sim[diag_idx, diag_idx] = float('-inf')
    
    # get top-(Q-1) neighbors per query
    k = Q - 1
    scores, idxs = sim.topk(k, dim=1)

    tgt = labels.unsqueeze(1) == labels[idxs]
    
    ranks = torch.arange(1, k+1, device=embs.device).float()  # per-query average precision
    cum_prec = (tgt.cumsum(dim=1).float() / ranks[None, :])  # cumulative precision @ each rank
    
    # sum of precisions * relevant / num. positives
    pos_counts = tgt.sum(dim=1).float()
    ap = (cum_prec * tgt.float()).sum(dim=1) / pos_counts  # avoid div-by-zero, will produce NaN where pos_counts == 0
    
    # mean over queries with >= 1 positive
    valid = ~torch.isnan(ap)
    map_score = ap[valid].mean().item()

    return map_score
