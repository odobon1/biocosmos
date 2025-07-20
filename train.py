import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from tqdm import tqdm
import time
import os

from models import VLMWrapper
from utils_data import spawn_dataloader, spawn_indexes_imgs
from utils_eval import ValidationPipeline
from utils import seed_libs, paths, get_slurm_alloc

import pdb

torch.set_printoptions(
    precision=4,
    sci_mode =False,
    threshold=1000,  # total elements before summarizing
    edgeitems=3,  # num items to show at the start/end of each dim
    linewidth=120
)


""" CONFIG PARAMS """

EXPERIMENT_NAME = "test_trial"
ALLOW_OVERWRITE = True  # whether to allow overwrites in the artifacts/ dir

# ----------------------------------------- current max batch size (1xB200 train w/ MP)
# MODEL_TYPE = "bioclip"                    # 2_048
# MODEL_TYPE = "bioclip2"                   # 512
# MODEL_TYPE = "clip_vitb32"                # 2_048 (4_096 intermittently exhausts VRAM, causing OOM failure)
# MODEL_TYPE = "clip_vitb16"                # 1_024
# MODEL_TYPE = "clip_vitl14"                # 512
# MODEL_TYPE = "clip_vitl14_336"            # 256
# MODEL_TYPE = "clip_rn50"                  # 2_048
# MODEL_TYPE = "clip_rn101"                 # 1_024
# MODEL_TYPE = "clip_rn101_yfcc15m"         # 1_024
# MODEL_TYPE = "clip_rn50x4"                # 1_024
# MODEL_TYPE = "clip_rn50x16"               # 256
# MODEL_TYPE = "clip_rn50x64"               # 128
MODEL_TYPE = "siglip_vitb16"              # 1_024
# MODEL_TYPE = "siglip_vitb16_384"          # x
# MODEL_TYPE = "siglip_vitl16_384"          # x
# MODEL_TYPE = "siglip_vitso400m14"         # x
# MODEL_TYPE = "siglip2_vitb16"             # x
# MODEL_TYPE = "siglip2_vitb16_384"         # x
# MODEL_TYPE = "siglip2_vitl16_384"         # x
# MODEL_TYPE = "siglip2_vitso400m14"        # x
# MODEL_TYPE = "siglip2_vitgopt16_384"      # x
# MODEL_TYPE = "vitamin_s"                  # x
# MODEL_TYPE = "vitamin_s_ltt"              # x  ~ LTT = "Locked-Text Tuning"
# MODEL_TYPE = "vitamin_b"                  # x
# MODEL_TYPE = "vitamin_b_ltt"              # x  ~ LTT = "Locked-Text Tuning"
# MODEL_TYPE = "vitamin_l"                  # x
# MODEL_TYPE = "vitamin_l_256"              # x
# MODEL_TYPE = "vitamin_l_336"              # x
# MODEL_TYPE = "vitamin_l_384"              # x
# MODEL_TYPE = "vitamin_l2"                 # x
# MODEL_TYPE = "vitamin_l2_384"             # x
# MODEL_TYPE = "vitamin_xl_384"             # x

SPLIT_NAME       = "S29-0"
# SPLIT_NAME       = "dev"
SEED             = 42
N_EPOCHS         = 1_000
BATCH_SIZE_TRAIN = 1_024
BATCH_SIZE_VAL   = 2_048
LR_INIT          = 1e-5
LR_DECAY         = 0.98

FREEZE_TEXT_ENCODER  = False
FREEZE_IMAGE_ENCODER = False

# LOSS_TYPE = "infonce"  # --------------------- standard CLIP loss
LOSS_TYPE = "pairwise_sigmoid"  # ------------ standard SigLIP loss
# LOSS_TYPE = "pairwise_sigmoid_upwtdpos"  # --- standard SigLIP loss with upweighted positives
# LOSS_TYPE = "multipos_sigmoid"  # ------------ custom SigLIP loss (multi-positives)

CACHED_IMGS              = None
# CACHED_IMGS              = "pl"  # preload, cache all images into memory
# CACHED_IMGS              = "pp"  # preload, preprocess, cache all images into memory

MP_TRAIN                 = True  # (True) use mixed precision for training
DROP_PARTIAL_BATCH_TRAIN = True
VERBOSE_BATCH_LOSS       = True

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

alloc           = get_slurm_alloc()
NUM_WORKERS     = alloc["cpus"]
PREFETCH_FACTOR = min(NUM_WORKERS, 8)

assert not (FREEZE_TEXT_ENCODER and FREEZE_IMAGE_ENCODER), "Text and image encoders are both set to frozen!"

print(
    f"",
    f"Experiment Name --- {EXPERIMENT_NAME}",
    f"Train-Run Name ---- {EXPERIMENT_NAME}_{SEED}",
    f"Split ------------- {SPLIT_NAME}",
    f"",
    f"Model Type ----------- {MODEL_TYPE}",
    f"Loss Type ------------ {LOSS_TYPE}",
    f"Batch Size (Train) --- {BATCH_SIZE_TRAIN}",
    f"LR Init -------------- {LR_INIT}",
    f"LR Decay ------------- {LR_DECAY}",
    f"",
    f"Num. GPUs ----------- {alloc['gpus']}",
    f"Num. CPUs/Workers --- {alloc['cpus']}",
    f"RAM ----------------- {alloc['ram']} GB",
    sep="\n"
)


def train_pipeline(modelw, loader_train, val_pipe, dpath_run, device, n_epochs, lr_init):

    modelw.freeze(FREEZE_TEXT_ENCODER, FREEZE_IMAGE_ENCODER)

    params_trainable = [p for p in modelw.model.parameters() if p.requires_grad]

    # use this for the cross-loss experiments
    # if LOSS_TYPE == "siglip_aligned":
    #     bias = torch.zeros((), device=device, requires_grad=True)
    #     params_trainable.append(bias)

    optimizer      = torch.optim.AdamW(params_trainable, lr=lr_init)
    lr_sched       = ExponentialLR(optimizer, gamma=LR_DECAY)
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
            f"{header_epoch:#^{75}}{'' if EXPERIMENT_NAME is None else ' (' + EXPERIMENT_NAME + '_' + str(SEED) + ')'}",
            f"",
            sep="\n"
        )

        time_start_train = time.time()
        modelw.model.train()
        loss_total_train = 0.0
        for imgs_b, class_encs_b, texts_b in tqdm(loader_train, desc="Train", leave=False):
            imgs_b = imgs_b.to(device, non_blocking=True)

            optimizer.zero_grad()

            if MP_TRAIN:
                with autocast(device_type=device.type):
                    embs_imgs = modelw.embed_images(imgs_b)  # --- Tensor(B, D)
                    embs_txts = modelw.embed_texts(texts_b)  # --- Tensor(B, D)

                    sim    = embs_imgs @ embs_txts.T  # ---------- Tensor(B, B)
                    logits = modelw.compute_logits(sim)
                    loss_b = modelw.compute_loss(logits, class_encs_b)

                scaler.scale(loss_b).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                embs_imgs = modelw.embed_images(imgs_b)  # ------- Tensor(B, D)
                embs_txts = modelw.embed_texts(texts_b)  # ------- Tensor(B, D)

                sim    = embs_imgs @ embs_txts.T  # -------------- Tensor(B, B)
                logits = modelw.compute_logits(sim)
                loss_b = modelw.compute_loss(logits)

                loss_b.backward()
                optimizer.step()

            with torch.no_grad():
                loss_b = loss_b.detach().item() * imgs_b.size(0)
                loss_total_train += loss_b
                if VERBOSE_BATCH_LOSS:
                    print(f"Batch Loss: {loss_b:.4f}")

        lr_epoch = optimizer.param_groups[0]["lr"]
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
            f"Loss --- {loss_epoch_train:.4f}",
            f"LR ----- {lr_epoch:.2e}",
            f"",
            sep="\n"
        )

        # VALIDATION

        scores_val, is_best_comp, is_best_img2img, time_elapsed_val = val_pipe.evaluate(modelw, verbose=True)

        if is_best_comp:
            modelw.save_checkpoint(dpath_run, scores_val, idx_epoch, "comp")
        if is_best_img2img:
            modelw.save_checkpoint(dpath_run, scores_val, idx_epoch, "img2img")

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
        raise ValueError(f"Train-run directory '{EXPERIMENT_NAME}_{SEED}' already exists!")

    seed_libs(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelw = VLMWrapper.build(MODEL_TYPE, device, loss_type=LOSS_TYPE)

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
        prefetch_factor     =PREFETCH_FACTOR,
    )

    val_pipe = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_preps     =TEXT_PREPS_VAL,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        num_workers    =NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
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
