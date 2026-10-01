import torch


def compute_sim(embs_img, embs_txt, sim_type):
    """
    For batch of image and text embeddings, both of shape (B, D), produces (B, B) similarity matrix

    Whether cosine similarity or geodesic distance is used, outputs are cast to range [-1, 1]
    for application of logit scale and bias.
    """
    cos_sim = embs_img @ embs_txt.T

    if sim_type == "cos":
        sim = cos_sim
    elif sim_type == "geo":
        # fp32: in bf16 the eps bounds round to +/-1.0 (guard no-op), and acos
        # backward at |cos| = 1 is inf
        cos_sim = cos_sim.float()
        eps      = 1e-6
        geo_dist = torch.acos(torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps))  # geodesic distance on range [0, pi]
        sim = 1.0 - 2.0 * (geo_dist / torch.pi)  # reverse + map to [-1, 1]

    return sim
