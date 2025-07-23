import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil

from utils import paths, save_json, load_json, save_pickle

import pdb


@dataclass
class TrainConfig:
    study_name: str
    experiment_name: str
    seed: int | None
    checkpoint_every: int
    split_name: str

    allow_overwrite_trial: bool
    allow_diff_study: bool
    allow_diff_experiment: bool

    model_type: str
    loss_type: str

    n_epochs: int
    batch_size_train: int
    batch_size_val: int
    lr_init: float
    lr_decay: float

    freeze_text_encoder: bool
    freeze_image_encoder: bool

    cached_imgs: str | None
    mp_train: bool
    drop_partial_batch_train: bool
    verbose_batch_loss: bool

    text_preps_type_train: str
    text_preps_type_val: str

def get_train_config():
    with open(Path(__file__).parent / "config_train.yaml") as f:
        train_config_dict = yaml.safe_load(f)
    train_config = TrainConfig(**train_config_dict)

    assert not (train_config.freeze_text_encoder and train_config.freeze_image_encoder), "Text and image encoders are both set to frozen!"

    return train_config

def get_trial_dpath(study_name, experiment_name, seed, allow_overwrite_trial):
    """
    Throws an error if allow_overwrite_trial=False and dpath already exists (if a seed is specified)
    Deletes trial dirs if allow_overwrite_trial=True and dpath already exists (if a seed is specified)
    """

    dpath_study      = paths["artifacts"] / study_name
    dpath_experiment = dpath_study / experiment_name

    if seed is not None:
        trial_name = seed
        dpath_trial = dpath_experiment / str(trial_name)
        if dpath_trial.exists():
            if allow_overwrite_trial:
                shutil.rmtree(dpath_trial)
            else:
                raise ValueError(f"Trial directory '{study_name}/{experiment_name}/{seed}' already exists!")
    else:
        counter = 0
        while True:
            trial_name  = f"seedless{counter}"
            dpath_trial = dpath_experiment / trial_name
            if dpath_trial.exists():
                counter += 1
            else:
                break

    trial_fullname = str(dpath_trial).split("artifacts/")[1]

    return dpath_study, dpath_experiment, dpath_trial, trial_fullname

def create_trial_dirs(dpath_trial):
    for subdir in ("logs", "models", "models/checkpoint", "models/best_comp", "models/best_img2img", "plots"):
        (dpath_trial / subdir).mkdir(parents=True)

def save_metadata_study(dpath_study, split_name, slurm_alloc, n_workers, prefetch_factor, allow_diff_study):
    fpath_meta = dpath_study / "metadata_study.json"
    metadata = {
        "split_name":      split_name,
        "n_gpus":          slurm_alloc["gpus"],
        "n_cpus":          slurm_alloc["cpus"],
        "ram":             f"{slurm_alloc['ram']} GB",
        "n_workers":       n_workers,
        "prefetch_factor": prefetch_factor,
    }
    if fpath_meta.exists() and not allow_diff_study:
        metadata_loaded = load_json(fpath_meta)
        assert metadata == metadata_loaded, "Study params changed!"
    else:
        save_json(metadata, fpath_meta)

def save_metadata_experiment(dpath_experiment, train_config, allow_diff_experiment):
    
    def clean_metadata(metadata):
        del metadata["study_name"]
        del metadata["experiment_name"]
        del metadata["seed"]
        del metadata["split_name"]
        del metadata["allow_overwrite_trial"]
        del metadata["allow_diff_study"]
        del metadata["allow_diff_experiment"]
        del metadata["checkpoint_every"]
        del metadata["verbose_batch_loss"]
    
    fpath_meta = dpath_experiment / "metadata_experiment.json"
    metadata   = asdict(train_config)
    clean_metadata(metadata)
    if fpath_meta.exists() and not allow_diff_experiment:
        metadata_loaded = load_json(fpath_meta)
        assert metadata == metadata_loaded, "Experiment params changed!"
    else:
        save_json(metadata, fpath_meta)
    
def save_metadata_trial(dpath_trial, model_type, datetime_init, time_train_avg=None, time_val_avg=None):
    fpath_meta = dpath_trial / "metadata_trial.json"
    if time_train_avg is not None:
        time_train_avg = f"{time_train_avg:.2f}"
        time_val_avg   = f"{time_val_avg:.2f}"
    metadata = {
        "model_type":    model_type,
        "runtime_avgs":  {
                             "train": f"{time_train_avg}",
                             "val":   f"{time_val_avg}",
                         },      
        "datetime_init": datetime_init,
    }
    save_json(metadata, fpath_meta)

def save_metadata_model(dpath_model, scores_val, idx_epoch):
    fpath_meta = dpath_model / "metadata_model.json"
    scores_val = {k: f"{v:.4f}" for k, v in scores_val.items()}
    metadata   = {
        "scores_val": scores_val,
        "idx_epoch":  idx_epoch,
    }
    save_json(metadata, fpath_meta)

def print_train_init_info(study_name, experiment_name, seed, trial_name, split_name, model_type, loss_type, batch_size_train, lr_init, lr_decay, slurm_alloc, n_workers, prefetch_factor):
    print(
        f"",
        f"Study -------- {study_name}",
        f"Experiment --- {experiment_name}",
        f"Seed --------- {seed}",
        f"Trial -------- {trial_name}",
        f"Split -------- {split_name}",
        f"",
        f"Model Type ----------- {model_type}",
        f"Loss Type ------------ {loss_type}",
        f"Batch Size (Train) --- {batch_size_train}",
        f"LR Init -------------- {lr_init}",
        f"LR Decay ------------- {lr_decay}",
        f"",
        f"Num. GPUs --- {slurm_alloc['gpus']}",
        f"Num. CPUs --- {slurm_alloc['cpus']}",
        f"RAM --------- {slurm_alloc['ram']} GB",
        f"",
        f"Num. Workers ------ {n_workers}",
        f"Prefetch Factor --- {prefetch_factor}",
        sep="\n"
    )

class TrialDataTracker:

    def __init__(self, dpath_trial):

        self.fpath_data = dpath_trial / "data_trial.pkl"

        self.data = {
            "id_img2txt_prec1":  [],
            "id_img2txt_rr":     [],
            "id_img2img_map":    [],
            "id_txt2img_map":    [],
            "ood_img2txt_prec1": [],
            "ood_img2txt_rr":    [],
            "ood_img2img_map":   [],
            "ood_txt2img_map":   [],
            "comp":              [],
            "comp_img2img":      [],
            "lr":                [],
            "loss_train":        [],
        }

    def update(self, scores_val, lr=None, loss_train=None):

        for score in scores_val.keys():
            self.data[score].append(float(scores_val[score]))

        if lr is not None:
            self.data["lr"].append(lr)
        if loss_train is not None:
            self.data["loss_train"].append(loss_train)

    def save(self):
        save_pickle(self.data, self.fpath_data)
