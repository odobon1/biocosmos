import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

from models import CLIPWrapper
from utils_data import spawn_dataloader, spawn_indexes_imgs
from utils_eval import ValidationPipeline
from utils import seed_libs, paths

import pdb


""" CONFIG PARAMS """

EXPERIMENT_NAME   = "test_qg"
ALLOW_OVERWRITE = True  # whether to allow overwrites in the artifacts/ dir

# ----------------------------------------- largest batch size (1xB200 train w/ MP)
# CLIP_TYPE = "openai_vitb32_hf"            # 4_096
# CLIP_TYPE = "bioclip"                     # 2_048
# CLIP_TYPE = "bioclip2"                    # 512
# CLIP_TYPE = "openai_vitb32"               # 4_096
# CLIP_TYPE = "openai_vitb16"               # 1_024
# CLIP_TYPE = "openai_vitl14"               # 512
# CLIP_TYPE = "openai_vitl14_336"           # 256
# CLIP_TYPE = "openai_rn50"                 # 2_048
# CLIP_TYPE = "openai_rn101"                # 1_024
# CLIP_TYPE = "openai_rn101_yfcc15m"        # 1_024
# CLIP_TYPE = "openai_rn50x4"               # 1_024
# CLIP_TYPE = "openai_rn50x16"              # 256
CLIP_TYPE = "openai_rn50x64"              # 128

# SPLIT_NAME       = "D"
SPLIT_NAME       = "dev16k"
SEED             = 42
N_EPOCHS         = 1
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL   = 2_048
LR_INIT          = 3e-5
LR_DECAY         = 0.99

CACHED_IMGS              = False  # (True) preload, preprocess, cache all images into memory
NUM_WORKERS              = 4  # adjust to CPU cores
MP_TRAIN                 = True  # (True) use mixed precision for training
DROP_PARTIAL_BATCH_TRAIN = True
VERBOSE_BATCH_LOSS       = False

TEXT_PREPS_TRAIN = [
    [
        "",
        "a photo of ",  # BioCLIP-style prepending
        "a photo of a ",  # OpenAI CLIP-style prepending
    ],
    [
        "",  # scientific name
        "animalia arthropoda insecta lepidoptera nymphalidae ",  # full taxonomic name
    ]
]

TEXT_PREPS_VAL = [["a photo of "]]  # scientific name, BioCLIP-style prepending

print(
    f"",
    f"Split ---------------- {SPLIT_NAME}",
    f"CLIP-type ------------ {CLIP_TYPE}",
    f"Batch Size (Train) --- {BATCH_SIZE_TRAIN}",
    f"",
    sep="\n"
)


def train_pipeline(modelw, loader_train, val_pipe, dpath_run, device, n_epochs, lr_init):
    optimizer = torch.optim.AdamW(modelw.model.parameters(), lr=lr_init)
    lr_sched = ExponentialLR(optimizer, gamma=LR_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    if MP_TRAIN:
        scaler = GradScaler()
    
    print(
        f"{' Fine-Tuning Init ':#^{75}}",
        f"",
        sep="\n"
    )
    
    val_pipe.evaluate(modelw, verbose=True)

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

                sim = embs_imgs @ embs_txts.T  # ---------------------------------------- Tensor(B, B) --- logits
                logits = sim * modelw.model.logit_scale.exp()

                loss_i2t_b = criterion(logits, targets)
                loss_t2i_b = criterion(logits.T, targets)
                loss_b     = 0.5 * (loss_i2t_b + loss_t2i_b)

                loss_b.backward()
                optimizer.step()

            with torch.no_grad():
                loss_b = loss_b.detach().item() * imgs_b.size(0)
                loss_total_train += loss_b
                if VERBOSE_BATCH_LOSS:
                    print(f"Batch Loss: {loss_b:.4f}")

        lr_epoch = lr_sched.get_last_lr()[0]
        lr_sched.step()
        
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
            f"Learning Rate ------- {lr_epoch:.2e}",
            f"",
            sep="\n"
        )

        # VALIDATION

        scores_val, is_best_comp, is_best_img2img, time_elapsed_val = val_pipe.evaluate(modelw, verbose=True)

        if is_best_comp:
            fpath_chkpt = dpath_run / "models/best_comp.pt"
            chkpt = {
                "model_state_dict" : modelw.model.state_dict(),
                "scores_val" :       scores_val,
                "epoch" :            idx_epoch,
            }
            torch.save(chkpt, fpath_chkpt)
            print("~ BEST COMP MODEL SAVED TO FILE ~")
            print()

        if is_best_img2img:
            fpath_chkpt = dpath_run / "models/best_comp.pt"
            chkpt = {
                "model_state_dict" : modelw.model.state_dict(),
                "scores_val" :       scores_val,
                "epoch" :            idx_epoch,
            }
            torch.save(chkpt, fpath_chkpt)
            print("~ BEST IMG2IMG MODEL SAVED TO FILE ~")
            print()

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

def create_train_run_dir(experiment_name, seed=None):
    already_exists = False

    def create_train_run_subdirs(dpath_run):
        for subdir in ("logs", "models", "plots"):
            os.makedirs(os.path.join(dpath_run, subdir), exist_ok=True)

    if seed is None:
        dpath_run = paths["artifacts"] / f"{experiment_name}_seedless0"
        counter = 1
        while os.path.exists(dpath_run):
            dpath_run = paths["artifacts"] / f"{experiment_name}_seedless{counter}"
            counter += 1
        create_train_run_subdirs(dpath_run)
    else:
        dpath_run = paths["artifacts"] / f"{experiment_name}_{seed}"
        if os.path.exists(dpath_run):
            already_exists = True
        else:
            create_train_run_subdirs(dpath_run)

    return already_exists, dpath_run

def main():

    already_exists, dpath_run = create_train_run_dir(experiment_name=EXPERIMENT_NAME, seed=SEED)
    if already_exists and not ALLOW_OVERWRITE:
        print(f"Train-Run Name --- {EXPERIMENT_NAME}_{SEED}")
        raise ValueError("Train-Run Directory Exists!")

    seed_libs(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelw = CLIPWrapper(CLIP_TYPE, device)

    index_imgs_class_enc_train, index_imgs_rfpaths_train, index_imgs_sids_train, _ = spawn_indexes_imgs(
        split_type="train",
        split_name=SPLIT_NAME,
    )

    loader_train = spawn_dataloader(
        index_imgs_class_enc=index_imgs_class_enc_train,
        index_imgs_rfpaths  =index_imgs_rfpaths_train,
        index_imgs_sids     =index_imgs_sids_train,
        text_preps          =TEXT_PREPS_TRAIN,
        batch_size          =BATCH_SIZE_TRAIN,
        shuffle             =True,
        drop_last           =DROP_PARTIAL_BATCH_TRAIN,
        img_pp              =modelw.img_pp,
        cached_imgs         =CACHED_IMGS,
        num_workers         =NUM_WORKERS,
        prefetch_factor     =2,
    )

    val_pipe = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_preps     =TEXT_PREPS_VAL,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=2,
    )

    train_pipeline(
        modelw, 
        loader_train, 
        val_pipe,
        dpath_run,
        device, 
        n_epochs=N_EPOCHS, 
        lr_init=LR_INIT,
    )

if __name__ == "__main__":
    main()
