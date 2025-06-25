import torch
import torch.nn.functional as F
import faiss
from torchmetrics.retrieval import RetrievalMAP

import pdb


def compute_map_img2img(embeddings, labels, topk=None):
    """
    Computes mAP score for evaluating image-to-image retrieval using all image embeddings as queries against all others.

    Args:
    - embeddings [Tensor(Q, D)] --- The embeddings
    - labels [Tensor(Q)] ---------- Labels (encoded as integers) corresponding to embeddings
    - topk [int or None] ---------- How many retrieved neighbors per query included in mAP computation (defaults to all of them with topk=None)

    Returns:
    - [float] --------------------- mAP score
    """

    embeddings = F.normalize(embeddings, dim=1)

    N, D = embeddings.shape
    topk = topk or N - 1

    cpu_index = faiss.IndexFlatIP(D)
    index = faiss.index_cpu_to_all_gpus(cpu_index)

    index.add(embeddings.cpu().numpy())  # add vectors to index

    # search
    distances, indices = index.search(embeddings.cpu().numpy(), topk + 1)
    scores = torch.from_numpy(distances)[:, 1:]  # ---------------------------------------- Tensor(Q, topk)
    idxs   = torch.from_numpy(indices)[:, 1:]    # ---------------------------------------- Tensor(Q, topk)

    # binary relevance: same-label = 1, else 0; mask out self hits
    target = (labels.unsqueeze(1) == labels[idxs]).long()
    self_mask = idxs == torch.arange(N).unsqueeze(1)
    target[self_mask] = 0

    # index‐map tensor where each row is the same query-id repeated
    query_idxs = torch.arange(N, device=scores.device).unsqueeze(1).expand(-1, topk)  # --- Tensor(N, topk)

    # compute mAP, skipping queries with zero positives
    metric = RetrievalMAP(empty_target_action='skip')
    map_score = metric(scores, target, query_idxs)

    return map_score.item()

embeddings = torch.tensor([
    [ 1.3, 0.01, 0.00],
    [ 1.1, 0.01, 0.00],
    [ 1.0, 0.01, 0.00],
    [-1.0, 0.01, 0.00],
    [-1.1, 0.01, 0.00],
    [-1.3, 0.01, 0.00],
], dtype=torch.float32)
labels = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long)

mAP = compute_map_img2img(embeddings, labels)
print(f"mAP: {mAP:.8f}")
