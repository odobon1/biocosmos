import torch
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from models import CLIPWrapper
from utils_data import spawn_dataloader, spawn_indexes_imgs, spawn_indexes_txts
from utils_eval import ValidationPipeline
from utils import seed_libs

import pdb


# config params
CLIP_TYPE        = "bioclip"  # "openai" / "bioclip"
CACHED_IMGS      = False  # preload, preprocess, cache all images into memory
BATCH_SIZE_TRAIN = 2048
BATCH_SIZE_VAL   = 2048
NUM_WORKERS      = 4  # adjust to CPU cores
SPLIT_NAME       = "D"
TEXT_PREP_TYPE   = "openai"  # "bioclip" (BioCLIP-style prepending) / "openai" (OpenAI CLIP-style prepending) / "base" (no prepending)
# TEXT_BASE_TYPE   = "tax"  # "tax" / "sci"

N_EPOCHS                 = 1_000
SEED                     = 42
MP_TRAIN                 = True  # whether mixed precision is used for training
DROP_PARTIAL_BATCH_TRAIN = True
EXPERIMENT_NAME          = "mixed42"

TEXT_TRAIN         = "mixed"
VERBOSE_BATCH_LOSS = False

print(f"Batch Size (Train): {BATCH_SIZE_TRAIN:,}")


def train_pipeline(modelw, loader_train, val_pipe_sci, val_pipe_tax, device, n_epochs, lr):
    optimizer = torch.optim.AdamW(modelw.model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    if MP_TRAIN:
        scaler = GradScaler()
    
    print(
        f"",
        f"{' Fine-Tuning Init ':#^{75}}",
        f"",
        sep="\n"
    )
    
    scores_val_sci, _, _, time_elapsed_val_sci = val_pipe_sci.evaluate(modelw, verbose=True)
    scores_val_tax, _, _, time_elapsed_val_tax = val_pipe_tax.evaluate(modelw, verbose=True)

    time_elapsed_train_mean = 0.0
    time_elapsed_val_mean = 0.0
    for idx_epoch in range(1, n_epochs + 1):

        header_epoch = f" Epoch {idx_epoch} "
        print(
            f"{header_epoch:#^{75}}{'' if EXPERIMENT_NAME is None else ' (' + EXPERIMENT_NAME + ')'}",
            f"",
            sep="\n"
        )

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
                loss_b = loss_b.detach().item() * imgs_b.size(0)
                loss_total_train += loss_b
                if VERBOSE_BATCH_LOSS:
                    print(f"Batch Loss: {loss_b:.4f}")

        # compute avg. train loss per sample
        if DROP_PARTIAL_BATCH_TRAIN:
            num_full_batches = len(loader_train.dataset) // BATCH_SIZE_TRAIN
            loss_epoch_train = loss_total_train / (num_full_batches * BATCH_SIZE_TRAIN)
        else:
            loss_epoch_train = loss_total_train / len(loader_train.dataset)

        time_end_train = time.time()
        time_elapsed_train = time_end_train - time_start_train

        print(
            f"{' Train ':=^{75}}",
            f"Loss ---------------- {loss_epoch_train:.4f}",
            f"",
            sep="\n"
        )

        # VALIDATION

        scores_val_sci, _, _, time_elapsed_val_sci = val_pipe_sci.evaluate(modelw, verbose=True)
        scores_val_tax, _, _, time_elapsed_val_tax = val_pipe_tax.evaluate(modelw, verbose=True)
        time_elapsed_val = time_elapsed_val_sci + time_elapsed_val_tax

        # track running means via Welford's algorithm
        time_elapsed_train_mean += (time_elapsed_train - time_elapsed_train_mean) / idx_epoch
        time_elapsed_val_mean += (time_elapsed_val - time_elapsed_val_mean) / idx_epoch

        print(
            f"{' Elapsed Time ':=^{75}}",
            f"Train --------------- {time_elapsed_train:.2f} s (avg: {time_elapsed_train_mean:.2f} s)",
            f"Validation ---------- {time_elapsed_val:.2f} s (avg: {time_elapsed_val_mean:.2f} s)",
            f"",
            sep="\n"
        )

def main():
    seed_libs(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelw = CLIPWrapper(CLIP_TYPE, device)

    index_imgs_class_enc_train, index_imgs_rfpaths_train, sid_2_class_enc_train = spawn_indexes_imgs(
        # split_type="id_test",  # using id_test as "train" rn for dev (just bc it's smaller)
        split_type="train",
        split_name=SPLIT_NAME,
    )
    index_txts_train_sci, index_txts_class_enc_train_sci = spawn_indexes_txts(
        sid_2_class_enc=sid_2_class_enc_train,
        text_base_type ="sci",
        text_prep_type =TEXT_PREP_TYPE,
    )
    index_txts_train_tax, index_txts_class_enc_train_tax = spawn_indexes_txts(
        sid_2_class_enc=sid_2_class_enc_train,
        text_base_type ="tax",
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
        index_txts_sci=index_txts_train_sci,
        index_txts_class_enc_sci=index_txts_class_enc_train_sci,
        index_txts_tax=index_txts_train_tax,
        index_txts_class_enc_tax=index_txts_class_enc_train_tax,
        drop_last=DROP_PARTIAL_BATCH_TRAIN,
        text_train=TEXT_TRAIN,
    )

    val_pipe_sci = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_base_type ="sci",
        text_prep_type =TEXT_PREP_TYPE,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
        header_tag     ="sci",
    )

    val_pipe_tax = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_base_type ="tax",
        text_prep_type =TEXT_PREP_TYPE,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
        header_tag     ="tax",
    )

    train_pipeline(
        modelw, 
        loader_train, 
        val_pipe_sci,
        val_pipe_tax,
        device, 
        n_epochs=N_EPOCHS, 
        lr=1e-6,
    )

if __name__ == "__main__":
    main()
