import torch


def compute_rank_dists(tax_vecs_b):
    """
    Computes rank distance between every pair of taxonomy-encoding vectors (e.g. same species = 0, same genus = 1, etc.)

    Args:
    - tax_vecs_b --- [Tensor(B, L)] --- Batch of taxonomy-encoding vectors (L = tree depth considered)

    Returns:
    - [Tensor(B, B)] ------------------ Tensor where entry (i, j) represents the rank at which vectors i and j differ
    """
    B, L = tax_vecs_b.shape
    
    # expand dimensions to compare every pair (i, j)
    x1 = tax_vecs_b.unsqueeze(1)  # ----------------- Tensor(B, 1, L)
    x2 = tax_vecs_b.unsqueeze(0)  # ----------------- Tensor(1, B, L)

    # compare each level of the tree between every vector
    eq_mask = (x1 == x2)  # ------------------------- Tensor(B, B, L)

    # invert to find divergence points
    neq_mask = ~eq_mask  # -------------------------- Tensor(B, B, L)
    neq_mask = neq_mask.int()  # convert to int so we can do argmax

    # argmax over levels to find the first divergence, returns the first True along L (first differing level)
    divergence_levels = neq_mask.argmax(dim=2)  # --- Tensor(B, B)

    # argmax is undefined for vectors that completely match, set to L
    all_eq_mask = neq_mask.sum(dim=2) == 0
    divergence_levels[all_eq_mask] = L

    rank_dists = L - divergence_levels

    return rank_dists

def compute_rank_dists_chunked(tax_vecs_b, chunk_size=1024):
    """
    Memory-efficient version to compute rank distance in chunks.
    Sacrifices a bit of speed for scalability (e.g. batch size = 32k)
    Can adjust `chunk_size` to trade off speed vs. memory usage ~ memory usage scales as O(chunk_size x B x L) instead of O(B^2 x L)

    This can easily be adapted for parallelization across multiple workers or devices
    Maybe could modify such that computations aren't all duplicated i.e. only need to compute the upper or lower triangle bc rank_dists matrix is symmetric
    """
    B, L = tax_vecs_b.shape
    rank_dists = torch.empty((B, B), dtype=torch.int16, device=tax_vecs_b.device)

    for i_start in range(0, B, chunk_size):
        i_end = min(i_start + chunk_size, B)
        chunk_i = tax_vecs_b[i_start:i_end]

        for j_start in range(0, B, chunk_size):
            j_end = min(j_start + chunk_size, B)
            chunk_j = tax_vecs_b[j_start:j_end]

            x1 = chunk_i.unsqueeze(1)
            x2 = chunk_j.unsqueeze(0)

            eq_mask = (x1 == x2)
            neq_mask = ~eq_mask
            neq_mask = neq_mask.int()

            divergence_levels = neq_mask.argmax(dim=2)
            all_eq_mask = neq_mask.sum(dim=2) == 0
            divergence_levels[all_eq_mask] = L

            rank_dists[i_start:i_end, j_start:j_end] = L - divergence_levels

    return rank_dists
