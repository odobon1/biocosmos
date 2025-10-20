import torch  # type: ignore[import]
from pathlib import Path
import yaml
from dataclasses import dataclass

from models import VLMWrapper
from utils_eval import ValidationPipeline
from utils import paths, compute_dataloader_workers_prefetch, get_text_preps, load_json
from utils_imb import compute_class_wts

import pdb


@dataclass
class EvalConfig:
    rdpath_trial: str | None
    save_crit: str  # model save criterion (only applicable if DPATH_TRIAL != None)

    split_name: str  # overridden if rdpath_trial is specified

    verbose_batch_loss: bool

    model_type: str
    loss_type: str
    targ_type: str
    class_weighting: dict
    focal: dict

    batch_size: int
    text_preps_type: str

    cached_imgs: bool
    act_chkpt:   bool
    
    def __post_init__(self):
        self.n_workers, self.prefetch_factor, slurm_alloc = compute_dataloader_workers_prefetch()
        self.n_gpus = slurm_alloc["n_gpus"]
        self.n_cpus = slurm_alloc["n_cpus"]
        self.ram    = slurm_alloc["ram"]

        if self.rdpath_trial is not None:
            metadata_experiment = load_json(paths["root"] / self.rdpath_trial / "../metadata_experiment.json")
            self.model_type     = metadata_experiment["model_type"]  # override model_type
            self.loss_type      = metadata_experiment["loss_type"]  # override loss_type

            metadata_study  = load_json(paths["root"] / self.rdpath_trial / "../../metadata_study.json")
            self.split_name = metadata_study["split_name"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.print_init_info()

    @classmethod
    def has_field(cls, name_field):
        return name_field in cls.__dataclass_fields__

    def print_init_info(self):
        print(
            f"",
            f"Trial ------------- {self.rdpath_trial}{'' if self.rdpath_trial is None else ' (' + self.save_crit + ')'}",
            f"",
            f"Model Type -------- {self.model_type}",
            f"Loss Type --------- {self.loss_type}",
            f"Split ------------- {self.split_name}",
            f"",
            f"Batch Size -------- {self.batch_size}",
            f"",
            f"Num. GPUs --------- {self.n_gpus}",
            f"Num. CPUs --------- {self.n_cpus}",
            f"RAM --------------- {self.ram} GB",
            f"",
            f"Num. Workers ------ {self.n_workers}",
            f"Prefetch Factor --- {self.prefetch_factor}",
            f"",
            f"Device ------------ {self.device}",
            f"",
            sep="\n"
        )

def get_config_eval():
    with open(Path(__file__).parent / "config/eval.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EvalConfig(**cfg_dict)
    return cfg

def main():

    config_eval = get_config_eval()

    modelw = VLMWrapper.build(config_eval)

    class_wts, class_pair_wts = compute_class_wts(config_eval)
    modelw.set_class_wts(class_wts, class_pair_wts)
    modelw.set_focal(config_eval.focal)
    modelw.set_targ_type(config_eval.targ_type)

    text_preps = get_text_preps(config_eval.text_preps_type)
    val_pipe   = ValidationPipeline(
        split_name     =config_eval.split_name,
        text_preps     =text_preps,
        batch_size     =config_eval.batch_size,
        img_pp         =modelw.img_pp_val,
        cached_imgs    =config_eval.cached_imgs,
        n_workers      =config_eval.n_workers,
        prefetch_factor=config_eval.prefetch_factor,
    )

    val_pipe.run_validation(modelw, verbose_batch_loss=config_eval.verbose_batch_loss)

if __name__ == "__main__":
    main()
