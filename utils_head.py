import torch  # type: ignore[import]


def compute_sim(embs_img, embs_txt, sim_type):
    """
    For batch of image and text embeddings, both of shape (B, D), produces (B, B) similarity matrix

    Whether cosine similarity or geodesic distance is used, outputs are cast to range [-1, 1]
    for application of logit scale and bias.
    """
    cos_sim = embs_img @ embs_txt.T

    if sim_type == "cos":
        sim = cos_sim
    else:
        if sim_type == "geo1":
            geo_dist = torch.atan2(torch.sqrt(torch.clamp(1 - torch.square(cos_sim), min=0.0)), cos_sim)  # geodesic distance on range [0, pi]
        elif sim_type == "geo2":
            eps      = 1e-6
            geo_dist = torch.acos(torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps))  # geodesic distance on range [0, pi]
        sim = 1.0 - 2.0 * (geo_dist / torch.pi)  # reverse + map to [-1, 1]

    return sim