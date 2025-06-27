import torch
import torch.nn.functional as F

from utils_eval import compute_map_img2img

import pdb


"""
probably should turn these into unit tests
"""

# embeddings = torch.tensor([
#     [ 1.30, 0.01, 0.00],
#     [ 1.10, 0.01, 0.00],
#     [ 1.00, 0.01, 0.00],
#     [-1.00, 0.01, 0.00],
#     [-1.10, 0.01, 0.00],
#     [-1.30, 0.01, 0.00],
# ], dtype=torch.float32)
# classes_enc = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long)

embeddings = torch.tensor([
    [-1.00, 0.01, 1.00],
    [ 1.30, 0.01, 1.00],
    [ 1.10, 0.01, 1.00],
    [-1.10, 0.01, 1.00],
    [-1.30, 0.01, 1.00],
    [ 1.00, 0.01, 1.00],
    [ 0.01, 1.10,-1.00],
    [ 0.01, 1.00,-1.00],
    [ 0.01,-1.10,-1.00],
    [ 0.01,-1.00,-1.00],
], dtype=torch.float32)
classes_enc = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3, 4, 5], dtype=torch.long)

map_score = compute_map_img2img(embeddings, classes_enc)
print(f"mAP: {map_score:.8f}")

"""
CONFIRMED: SINGLETONS HANDLED APPROPRIATELY (excludes singletons from mAP computation)
"""
