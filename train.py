import torch
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from models import CLIPWrapper
from utils_data import spawn_dataloader, spawn_indexes_imgs, spawn_indexes_txts
from utils_eval import EvaluationPipeline
from utils import seed_libs

import pdb


# config params
CLIP_TYPE        = "bioclip"  # "openai" / "bioclip"
CACHED_IMGS      = False  # preload, preprocess, cache all images into memory
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL   = 2048
NUM_WORKERS      = 4  # adjust to CPU cores
SPLIT_NAME       = "D"
TEXT_PREP_TYPE   = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
TEXT_BASE_TYPE   = "tax"  # "tax" / "sci"

N_EPOCHS                 = 100
SEED                     = 42
MP_TRAIN                 = True  # whether mixed precision is used for training
DROP_PARTIAL_BATCH_TRAIN = True


print(f"Batch Size (Train): {BATCH_SIZE_TRAIN:,}")


def print_val(header, score_composite, scores_id_val, scores_ood_val, time_elapsed_id_val, time_elapsed_ood_val, loss_train=None, time_elapsed_train=None):

    print(
        f"==========================================",
        # f"Out-of-box Performance",
        f"{header}",
        sep="\n"
    )

    if loss_train is not None:
        print(
            f"------------------------------------------",
            f"Train Loss ------------- {loss_train:.4f}",
            sep="\n"
        )

    print(
        f"------------------------------------------",
        f"composite -------------- {score_composite:.4f}",
        f"------------------------------------------",
        f"~ ID ~",
        f"img2txt Prec@1 --------- {scores_id_val['img2txt_prec1']:.2%}",
        f"img2txt mAP (RR) ------- {scores_id_val['img2txt_map']:.4f}",
        f"img2img mAP ------------ {scores_id_val['img2img_map']:.4f}",
        f"txt2img mAP ------------ {scores_id_val['txt2img_map']:.4f}",
        f"------------------------------------------",
        f"~ OOD ~",
        f"img2txt Prec@1 --------- {scores_ood_val['img2txt_prec1']:.2%}",
        f"img2txt mAP (RR) ------- {scores_ood_val['img2txt_map']:.4f}",
        f"img2img mAP ------------ {scores_ood_val['img2img_map']:.4f}",
        f"txt2img mAP ------------ {scores_ood_val['txt2img_map']:.4f}",
        f"------------------------------------------",
        f"Elapsed Time (Val) ----- {time_elapsed_id_val + time_elapsed_ood_val:.2f} (s)",
        sep="\n"
    )
    if time_elapsed_train is not None:
        print(f"Elapsed Time (Train) --- {time_elapsed_train:.2f} (s)")

def train_pipeline(modelw, loader_train, id_val_pipe, ood_val_pipe, device, n_epochs, lr):
    optimizer = torch.optim.AdamW(modelw.model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    if MP_TRAIN:
        scaler = GradScaler()

    scores_val_tracker = {
        "id_img2txt_prec1" : [],
        "id_img2txt_map" : [],
        "id_img2img_map" : [],
        "id_txt2img_map" : [],
        "ood_img2txt_prec1" : [],
        "ood_img2txt_map" : [],
        "ood_img2img_map" : [],
        "ood_txt2img_map" : [],
        "composite" : [],
    }
    
    scores_id_val, time_elapsed_id_val = id_val_pipe.evaluate(modelw)
    scores_val_tracker["id_img2txt_prec1"].append(scores_id_val["img2txt_prec1"])
    scores_val_tracker["id_img2txt_map"].append(scores_id_val["img2txt_map"])
    scores_val_tracker["id_img2img_map"].append(scores_id_val["img2img_map"])
    scores_val_tracker["id_txt2img_map"].append(scores_id_val["txt2img_map"])

    scores_ood_val, time_elapsed_ood_val = ood_val_pipe.evaluate(modelw)
    scores_val_tracker["ood_img2txt_prec1"].append(scores_ood_val["img2txt_prec1"])
    scores_val_tracker["ood_img2txt_map"].append(scores_ood_val["img2txt_map"])
    scores_val_tracker["ood_img2img_map"].append(scores_ood_val["img2img_map"])
    scores_val_tracker["ood_txt2img_map"].append(scores_ood_val["txt2img_map"])

    score_composite = (scores_id_val["img2txt_map"] + \
                 scores_id_val["img2img_map"] + \
                 scores_id_val["txt2img_map"] + \
                 scores_ood_val["img2txt_map"] + \
                 scores_ood_val["img2img_map"] + \
                 scores_ood_val["txt2img_map"]) / 6
    scores_val_tracker["composite"].append(score_composite)

    print_val(
        "Out-of-box Performance", 
        score_composite,
        scores_id_val, 
        scores_ood_val, 
        time_elapsed_id_val, 
        time_elapsed_ood_val,
    )

    for idx_epoch in range(1, n_epochs + 1):
        time_start_train = time.time()
        modelw.model.train()
        loss_total_train = 0.0
        for imgs_b, _, texts_b in tqdm(loader_train, desc="Train", leave=False):
            imgs_b = imgs_b.to(device, non_blocking=True)            
            
            targets = torch.arange(imgs_b.size(0), device=device, dtype=torch.long)

            optimizer.zero_grad()

            if MP_TRAIN:
                with autocast(device_type=device.type):
                    embs_imgs = modelw.embed_images(imgs_b)  # -------------------------- Tensor(B, D)
                    embs_txts = modelw.embed_texts(texts_b)  # -------------------------- Tensor(B, D)

                    sim = embs_imgs @ embs_txts.T * modelw.model.logit_scale.exp()  # --- Tensor(B, B) --- logits

                    loss_i2t_b = criterion(sim, targets)
                    loss_t2i_b = criterion(sim.T, targets)
                    loss_b     = 0.5 * (loss_i2t_b + loss_t2i_b)

                scaler.scale(loss_b).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                embs_imgs = modelw.embed_images(imgs_b)  # ------------------------------ Tensor(B, D)
                embs_txts = modelw.embed_texts(texts_b)  # ------------------------------ Tensor(B, D)

                sim = embs_imgs @ embs_txts.T * modelw.model.logit_scale.exp()  # ------- Tensor(B, B) --- logits

                loss_i2t_b = criterion(sim, targets)
                loss_t2i_b = criterion(sim.T, targets)
                loss_b     = 0.5 * (loss_i2t_b + loss_t2i_b)

                loss_b.backward()
                optimizer.step()

            with torch.no_grad():
                loss_total_train += loss_b.detach().item() * imgs_b.size(0)

        # compute avg. train loss per sample
        if DROP_PARTIAL_BATCH_TRAIN:
            num_full_batches = len(loader_train.dataset) // BATCH_SIZE_TRAIN
            loss_epoch_train = loss_total_train / (num_full_batches * BATCH_SIZE_TRAIN)
        else:
            loss_epoch_train = loss_total_train / len(loader_train.dataset)

        time_end_train = time.time()
        time_elapsed_train = time_end_train - time_start_train

        # VALIDATION

        scores_id_val, time_elapsed_id_val = id_val_pipe.evaluate(modelw)
        scores_val_tracker["id_img2txt_prec1"].append(scores_id_val["img2txt_prec1"])
        scores_val_tracker["id_img2img_map"].append(scores_id_val["img2img_map"])
        scores_val_tracker["id_txt2img_map"].append(scores_id_val["txt2img_map"])

        scores_ood_val, time_elapsed_ood_val = ood_val_pipe.evaluate(modelw)
        scores_val_tracker["ood_img2txt_prec1"].append(scores_ood_val["img2txt_prec1"])
        scores_val_tracker["ood_img2img_map"].append(scores_ood_val["img2img_map"])
        scores_val_tracker["ood_txt2img_map"].append(scores_ood_val["txt2img_map"])

        score_composite = (scores_id_val["img2txt_map"] + \
                    scores_id_val["img2img_map"] + \
                    scores_id_val["txt2img_map"] + \
                    scores_ood_val["img2txt_map"] + \
                    scores_ood_val["img2img_map"] + \
                    scores_ood_val["txt2img_map"]) / 6
        scores_val_tracker["composite"].append(score_composite)

        print_val(
            f"Epoch {idx_epoch}", 
            score_composite,
            scores_id_val, 
            scores_ood_val, 
            time_elapsed_id_val, 
            time_elapsed_ood_val, 
            loss_train=loss_epoch_train,
            time_elapsed_train=time_elapsed_train,
        )

    # plot validation curve
    plt.figure()
    plt.plot(range(0, n_epochs + 1), scores_val_tracker["id_img2txt_prec1"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("CLIP Fine-Tuning Validation Curve")
    plt.grid(True)
    plt.show()

    return scores_val_tracker

def main():
    seed_libs(SEED)
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
        drop_last=DROP_PARTIAL_BATCH_TRAIN,
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
