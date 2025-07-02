import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import CLIPWrapper
from utils_data import spawn_dataloader, spawn_indexes_imgs, spawn_indexes_txts
from utils_eval import EvaluationPipeline

import pdb


# config params
CLIP_TYPE        = "bioclip"  # "openai" / "bioclip"
CACHED_IMGS      = False  # preload, preprocess, cache all images into memory
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_VAL   = 256
NUM_WORKERS      = 4  # adjust to CPU cores
SPLIT_NAME       = "D"
TEXT_PREP_TYPE   = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
TEXT_BASE_TYPE   = "tax"  # "tax" / "sci"

N_EPOCHS = 100


def train_pipeline(modelw, loader_train, id_val_pipe, ood_val_pipe, device, n_epochs, lr):
    optimizer = torch.optim.AdamW(modelw.model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    val_scores = {
        "id_img2txt_prec1" : [],
        "id_img2img_map" : [],
        "id_txt2img_map" : [],
        "ood_img2txt_prec1" : [],
        "ood_img2img_map" : [],
        "ood_txt2img_map" : [],
    }
    
    id_val_scores, time_elapsed_id_val = id_val_pipe.evaluate(modelw)
    val_scores["id_img2txt_prec1"].append(id_val_scores["img2txt_prec1"])
    val_scores["id_img2img_map"].append(id_val_scores["img2img_map"])
    val_scores["id_txt2img_map"].append(id_val_scores["txt2img_map"])

    ood_val_scores, time_elapsed_ood_val = ood_val_pipe.evaluate(modelw)
    val_scores["ood_img2txt_prec1"].append(ood_val_scores["img2txt_prec1"])
    val_scores["ood_img2img_map"].append(ood_val_scores["img2img_map"])
    val_scores["ood_txt2img_map"].append(ood_val_scores["txt2img_map"])

    print(
        f"==========================================",
        f"Out-of-box Performance",
        f"------------------------------------------",
        f"~ ID ~",
        f"img2txt Prec@1 --- {id_val_scores['img2txt_prec1']:.2%}",
        f"img2img mAP ------ {id_val_scores['img2img_map']:.4f}",
        f"txt2img mAP ------ {id_val_scores['txt2img_map']:.4f}",
        f"Elapsed Time: {time_elapsed_id_val:.2f} (s)",
        f"------------------------------------------",
        f"~ OOD ~",
        f"img2txt Prec@1 --- {ood_val_scores['img2txt_prec1']:.2%}",
        f"img2img mAP ------ {ood_val_scores['img2img_map']:.4f}",
        f"txt2img mAP ------ {ood_val_scores['txt2img_map']:.4f}",
        f"Elapsed Time: {time_elapsed_ood_val:.2f} (s)",
        sep="\n"
    )

    for idx_epoch in range(1, n_epochs + 1):
        modelw.model.train()
        loss_total_train = 0.0
        for imgs_b, _, texts_b in tqdm(loader_train, desc="Train", leave=False):
            imgs_b = imgs_b.to(device, non_blocking=True)            
            
            targets = torch.arange(imgs_b.size(0), device=device, dtype=torch.long)

            optimizer.zero_grad()

            embs_imgs = modelw.embed_images(imgs_b)  # --- Tensor(B, D)
            embs_txts = modelw.embed_texts(texts_b)  # --- Tensor(B, D)

            sim = embs_imgs @ embs_txts.T * modelw.model.logit_scale.exp()  # ------------- Tensor(B, B) --- logits

            loss_i2t_b = criterion(sim, targets)
            loss_t2i_b = criterion(sim.T, targets)
            loss_b     = 0.5 * (loss_i2t_b + loss_t2i_b)

            loss_b.backward()
            optimizer.step()

            loss_total_train += loss_b.item() * imgs_b.size(0)

        loss_epoch = loss_total_train / len(loader_train.dataset)
        print(f"Epoch {idx_epoch} train loss: {loss_epoch:.4f}")

        # VALIDATION

        id_val_scores, time_elapsed_id_val = id_val_pipe.evaluate(modelw)
        val_scores["id_img2txt_prec1"].append(id_val_scores["img2txt_prec1"])
        val_scores["id_img2img_map"].append(id_val_scores["img2img_map"])
        val_scores["id_txt2img_map"].append(id_val_scores["txt2img_map"])

        ood_val_scores, time_elapsed_ood_val = ood_val_pipe.evaluate(modelw)
        val_scores["ood_img2txt_prec1"].append(ood_val_scores["img2txt_prec1"])
        val_scores["ood_img2img_map"].append(ood_val_scores["img2img_map"])
        val_scores["ood_txt2img_map"].append(ood_val_scores["txt2img_map"])

        print(
            f"==========================================",
            f"Epoch {idx_epoch}",
            f"------------------------------------------",
            f"~ ID ~",
            f"img2txt Prec@1 --- {id_val_scores['img2txt_prec1']:.2%}",
            f"img2img mAP ------ {id_val_scores['img2img_map']:.4f}",
            f"txt2img mAP ------ {id_val_scores['txt2img_map']:.4f}",
            f"Elapsed Time: {time_elapsed_id_val:.2f} (s)",
            f"------------------------------------------",
            f"~ OOD ~",
            f"img2txt Prec@1 --- {ood_val_scores['img2txt_prec1']:.2%}",
            f"img2img mAP ------ {ood_val_scores['img2img_map']:.4f}",
            f"txt2img mAP ------ {ood_val_scores['txt2img_map']:.4f}",
            f"Elapsed Time: {time_elapsed_ood_val:.2f} (s)",
            sep="\n"
        )

    # plot validation curve
    plt.figure()
    plt.plot(range(0, n_epochs + 1), val_scores["id_img2txt_prec1"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("CLIP Fine-Tuning Validation Curve")
    plt.grid(True)
    plt.show()

    return val_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelw = CLIPWrapper(CLIP_TYPE, device)

    index_imgs_class_enc_train, index_imgs_rfpaths_train, sid_2_class_enc_train = spawn_indexes_imgs(
        # split_type="id_test",  # using id_test as "train" rn for dev (just bc it's smaller)
        split_type="train",
        split_name=SPLIT_NAME,
    )
    index_txts_train, index_txts_class_enc_train = spawn_indexes_txts(
        sid_2_class_enc=sid_2_class_enc_train,
        text_base_type =TEXT_BASE_TYPE,
        text_prep_type =TEXT_PREP_TYPE,
    )
    loader_train = spawn_dataloader(
        index_imgs_class_enc=index_imgs_class_enc_train,
        index_imgs_rfpaths  =index_imgs_rfpaths_train,
        img_pp              =modelw.img_pp,
        cached_imgs         =CACHED_IMGS,
        batch_size          =BATCH_SIZE_TRAIN,
        shuffle             =True,
        num_workers         =NUM_WORKERS,
        prefetch_factor     =2,
        index_txts=index_txts_train,
        index_txts_class_enc=index_txts_class_enc_train,
    )

    id_val_pipe = EvaluationPipeline(
        split_type     ="id_val", 
        split_name     =SPLIT_NAME, 
        text_base_type =TEXT_BASE_TYPE, 
        text_prep_type =TEXT_PREP_TYPE,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
        modes          =["img2txt", "img2img", "txt2img"],
    )

    ood_val_pipe = EvaluationPipeline(
        split_type     ="ood_val", 
        split_name     =SPLIT_NAME, 
        text_base_type =TEXT_BASE_TYPE, 
        text_prep_type =TEXT_PREP_TYPE,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
        modes          =["img2txt", "img2img", "txt2img"],
    )

    train_pipeline(
        modelw, 
        loader_train, 
        id_val_pipe, 
        ood_val_pipe, 
        device, 
        n_epochs=N_EPOCHS, 
        lr=1e-6,
    )


if __name__ == "__main__":
    main()
