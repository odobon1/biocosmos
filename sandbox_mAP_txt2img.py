import torch

from utils_eval import compute_map_txt2img

import pdb


"""
probably should turn these into unit tests
"""

"""
Test Example 1:
mAP should be: 0.7222
mAP = (1/2) * ((1/3)*((1/2)+(2/3)+(3/4)) + (1/3)*((1/1)+(2/3)+(3/4))) = 0.7222
"""
# embs_imgs = torch.tensor([
#     [ 1.30, 0.01, 0.00],  # img emb 0
#     [ 1.10, 0.01, 0.00],  # img emb 1
#     [ 1.00, 0.01, 0.00],  # img emb 2
#     [-1.00, 0.01, 0.00],  # img emb 3
#     [-1.10, 0.01, 0.00],  # img emb 4
#     [-1.30, 0.01, 0.00],  # img emb 5
# ], dtype=torch.float32)
# classes_enc_imgs = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long)

# embs_txts = torch.tensor([
#     [ 1.01, 0.01, 0.00],  # txt emb 0 (very close to img emb 2)
#     [-1.11, 0.01, 0.00],  # txt emb 1 (very close to img emb 4)
# ], dtype=torch.float32)
# classes_enc_txts = torch.tensor([0, 1])

"""
Test Example 2:
mAP should be: 0.8148
mAP = (1/3) * ((1/3)*((1/2)+(2/3)+(3/4)) + (1/3)*((1/1)+(2/3)+(3/4)) + (1/1)*(1/1)) = 0.8148
"""
# embs_imgs = torch.tensor([
#     [ 1.30, 0.01, 1.00],  # img emb 0
#     [-1.10, 0.01, 1.00],  # img emb 4
#     [-1.30, 0.01, 1.00],  # img emb 5
#     [ 1.10, 0.01, 1.00],  # img emb 1
#     [ 1.00, 0.01, 1.00],  # img emb 2
#     [-1.00, 0.01, 1.00],  # img emb 3
#     [ 0.01, 1.00,-1.00],  # img emb 6 (singleton)
# ], dtype=torch.float32)
# classes_enc_imgs = torch.tensor([0, 1, 1, 0, 1, 0, 2], dtype=torch.long)

# embs_txts = torch.tensor([
#     [ 1.01, 0.01, 1.00],  # txt emb 0 (very close to img emb 2)
#     [-1.11, 0.01, 1.00],  # txt emb 1 (very close to img emb 4)
#     [ 0.00, 1.00,-1.00],  # txt emb 2 (very close to img emb 6)
# ], dtype=torch.float32)
# classes_enc_txts = torch.tensor([0, 1, 2])

"""
Test Example 3:
mAP should be: 0.5291
mAP = (1/3) * ((1/3)*((1/2)+(2/3)+(3/4)) + (1/3)*((1/1)+(2/3)+(3/4)) + (1/1)*(1/7)) = 0.5291
"""
embs_imgs = torch.tensor([
    [ 1.30, 0.01, 1.00],  # img emb 0
    [-1.10, 0.01, 1.00],  # img emb 4
    [-1.30, 0.01, 1.00],  # img emb 5
    [ 1.10, 0.01, 1.00],  # img emb 1
    [ 1.00, 0.01, 1.00],  # img emb 2
    [-1.00, 0.01, 1.00],  # img emb 3
    [ 0.01, 1.00,-1.00],  # img emb 6 (singleton)
], dtype=torch.float32)
classes_enc_imgs = torch.tensor([0, 1, 1, 0, 1, 0, 2], dtype=torch.long)

embs_txts = torch.tensor([
    [ 1.01, 0.01, 1.00],  # txt emb 0 (very close to img emb 2)
    [-1.11, 0.01, 1.00],  # txt emb 1 (very close to img emb 4)
    [ 0.00,-1.00, 1.00],  # txt emb 2 (very far from img emb 6)
], dtype=torch.float32)
classes_enc_txts = torch.tensor([0, 1, 2])


map_score = compute_map_txt2img(embs_txts, classes_enc_txts, embs_imgs, classes_enc_imgs)
print(f"mAP: {map_score:.4f}")
