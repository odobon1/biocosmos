import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import time
from datetime import datetime, timezone

from models import VLMWrapper
from utils_data import spawn_dataloader, spawn_indexes_imgs
from utils_eval import ValidationPipeline
from utils import seed_libs, get_text_preps, compute_dataloader_workers_prefetch
from utils_train import (
    get_train_config,
    get_trial_dpath,
    create_trial_dirs, 
    save_metadata_study,
    save_metadata_experiment,
    save_metadata_trial, 
    save_metadata_model, 
    print_train_init_info,  
    TrialDataTracker,
)

import pdb

torch.set_printoptions(
    precision=4,
    sci_mode =False,
    threshold=1000,  # total elements before summarizing
    edgeitems=3,  # num items to show at the start/end of each dim
    linewidth=120
)

train_config = get_train_config()

STUDY_NAME       = train_config.study_name
EXPERIMENT_NAME  = train_config.experiment_name
SEED             = train_config.seed
CHECKPOINT_EVERY = train_config.checkpoint_every
SPLIT_NAME       = train_config.split_name

ALLOW_OVERWRITE_TRIAL = train_config.allow_overwrite_trial
ALLOW_DIFF_STUDY      = train_config.allow_diff_study
ALLOW_DIFF_EXPERIMENT = train_config.allow_diff_experiment

MODEL_TYPE = train_config.model_type
LOSS_TYPE  = train_config.loss_type

N_EPOCHS         = train_config.n_epochs
BATCH_SIZE_TRAIN = train_config.batch_size_train
BATCH_SIZE_VAL   = train_config.batch_size_val
LR_INIT          = train_config.lr_init
LR_DECAY         = train_config.lr_decay

FREEZE_TEXT_ENCODER  = train_config.freeze_text_encoder
FREEZE_IMAGE_ENCODER = train_config.freeze_image_encoder

CACHED_IMGS              = train_config.cached_imgs
MP_TRAIN                 = train_config.mp_train
DROP_PARTIAL_BATCH_TRAIN = train_config.drop_partial_batch_train
VERBOSE_BATCH_LOSS       = train_config.verbose_batch_loss

TEXT_PREPS_TRAIN = get_text_preps(train_config.text_preps_type_train)
TEXT_PREPS_VAL   = get_text_preps(train_config.text_preps_type_val)

N_WORKERS, PREFETCH_FACTOR, SLURM_ALLOC = compute_dataloader_workers_prefetch()


def train_pipeline(modelw, loader_train, val_pipe, dpath_trial, trial_fullname, device, n_epochs, lr_init):
    datetime_init = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    save_metadata_trial(dpath_trial, modelw.type, datetime_init)

    dpath_model_best_comp    = dpath_trial / "models/best_comp"
    dpath_model_best_img2img = dpath_trial / "models/best_img2img"
    dpath_model_checkpoint   = dpath_trial / "models/checkpoint"

    modelw.freeze(FREEZE_TEXT_ENCODER, FREEZE_IMAGE_ENCODER)

    params_trainable = [p for p in modelw.model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params_trainable, lr=lr_init)
    lr_sched  = ExponentialLR(optimizer, gamma=LR_DECAY)
    if MP_TRAIN:
        scaler = GradScaler()
    
    print(
        f"{' Fine-Tuning Init ':#^{75}}",
        f"",
        sep="\n"
    )
    
    data_tracker        = TrialDataTracker(dpath_trial)
    scores_val, _, _, _ = val_pipe.evaluate(modelw, verbose=True)
    data_tracker.update(scores_val)

    time_train_avg = 0.0
    time_val_avg   = 0.0
    for idx_epoch in range(1, n_epochs + 1):

        header_epoch = f" Epoch {idx_epoch} "
        print(
            f"{header_epoch:#^{75}}{'' if EXPERIMENT_NAME is None else ' (' + trial_fullname + ')'}",
            f"",
            sep="\n"
        )

        time_train_start = time.time()
        modelw.model.train()
        loss_train_total = 0.0
        for imgs_b, class_encs_b, texts_b in tqdm(loader_train, desc="Train", leave=False):
            imgs_b = imgs_b.to(device, non_blocking=True)

            optimizer.zero_grad()

            if MP_TRAIN:
                with autocast(device_type=device.type):
                    embs_imgs = modelw.embed_images(imgs_b)  # --- Tensor(B, D)
                    embs_txts = modelw.embed_texts(texts_b)  # --- Tensor(B, D)

                    sim          = embs_imgs @ embs_txts.T  # ---- Tensor(B, B)
                    logits       = modelw.compute_logits(sim)
                    loss_train_b = modelw.compute_loss(logits, class_encs_b)

                scaler.scale(loss_train_b).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                embs_imgs = modelw.embed_images(imgs_b)  # ------- Tensor(B, D)
                embs_txts = modelw.embed_texts(texts_b)  # ------- Tensor(B, D)

                sim          = embs_imgs @ embs_txts.T  # -------- Tensor(B, B)
                logits       = modelw.compute_logits(sim)
                loss_train_b = modelw.compute_loss(logits)

                loss_train_b.backward()
                optimizer.step()

            with torch.no_grad():
                loss_train_b = loss_train_b.detach().item() * imgs_b.size(0)
                loss_train_total += loss_train_b
                if VERBOSE_BATCH_LOSS:
                    print(f"Batch Loss: {loss_train_b:.4f}")

        lr = optimizer.param_groups[0]["lr"]
        lr_sched.step()
        
        # compute avg. train loss per sample
        if DROP_PARTIAL_BATCH_TRAIN:
            n_full_batches = len(loader_train.dataset) // BATCH_SIZE_TRAIN
            loss_train_avg = loss_train_total / (n_full_batches * BATCH_SIZE_TRAIN)
        else:
            loss_train_avg = loss_train_total / len(loader_train.dataset)

        time_train_end = time.time()
        time_train     = time_train_end - time_train_start

        print(
            f"{' Train ':=^{75}}",
            f"Loss --- {loss_train_avg:.4f}",
            f"LR ----- {lr:.2e}",
            f"",
            sep="\n"
        )

        # validation
        scores_val, is_best_comp, is_best_img2img, time_val = val_pipe.evaluate(modelw, verbose=True)
        data_tracker.update(scores_val, lr=lr, loss_train=loss_train_avg)

        # track running means via Welford's algorithm
        time_train_avg += (time_train - time_train_avg) / idx_epoch
        time_val_avg += (time_val - time_val_avg) / idx_epoch

        if is_best_comp:
            modelw.save(dpath_model_best_comp)
            save_metadata_model(dpath_model_best_comp, scores_val, idx_epoch)
            print(f"~ Best comp model saved to file ~\n")
        if is_best_img2img:
            modelw.save(dpath_model_best_img2img)
            save_metadata_model(dpath_model_best_img2img, scores_val, idx_epoch)
            print(f"~ Best img2img model saved to file ~\n")

        if idx_epoch % CHECKPOINT_EVERY == 0:
            modelw.save(dpath_model_checkpoint)
            save_metadata_model(dpath_model_checkpoint, scores_val, idx_epoch)
            save_metadata_trial(
                dpath_trial, 
                modelw.type, 
                datetime_init, 
                time_train_avg=time_train_avg, 
                time_val_avg  =time_val_avg,
            )
            data_tracker.save()

        print(
            f"{' Elapsed Time ':=^{75}}",
            f"Train -------- {time_train:.2f} s (avg: {time_train_avg:.2f} s)",
            f"Validation --- {time_val:.2f} s (avg: {time_val_avg:.2f} s)",
            f"",
            sep="\n"
        )

def main():

    dpath_study, dpath_experiment, dpath_trial, trial_fullname = get_trial_dpath(STUDY_NAME, EXPERIMENT_NAME, SEED, ALLOW_OVERWRITE_TRIAL)
    print_train_init_info(
        STUDY_NAME,
        EXPERIMENT_NAME, 
        SEED, 
        trial_fullname,
        SPLIT_NAME, 
        MODEL_TYPE, 
        LOSS_TYPE, 
        BATCH_SIZE_TRAIN, 
        LR_INIT, 
        LR_DECAY, 
        SLURM_ALLOC,
        N_WORKERS,
        PREFETCH_FACTOR,
    )
    # (any dir-init-related Value Errors and etc. should be caught above, before trial dirs are created)
    create_trial_dirs(dpath_trial)
    save_metadata_study(dpath_study, SPLIT_NAME, SLURM_ALLOC, N_WORKERS, PREFETCH_FACTOR, ALLOW_DIFF_STUDY)
    save_metadata_experiment(dpath_experiment, train_config, ALLOW_DIFF_EXPERIMENT)

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
        n_workers           =N_WORKERS,
        prefetch_factor     =PREFETCH_FACTOR,
    )

    val_pipe = ValidationPipeline(
        split_name     =SPLIT_NAME,
        text_preps     =TEXT_PREPS_VAL,
        img_pp         =modelw.img_pp,
        cached_imgs    =CACHED_IMGS,
        batch_size     =BATCH_SIZE_VAL,
        n_workers      =N_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    train_pipeline(
        modelw, 
        loader_train, 
        val_pipe,
        dpath_trial,
        trial_fullname,
        device, 
        n_epochs=N_EPOCHS, 
        lr_init=LR_INIT,
    )

if __name__ == "__main__":
    main()
