import os
from pathlib import Path


def get_cluster():
    """Returns 'hipergator' if on UF HPG, returns 'ice-slurm' if on GATech PACE-ICE"""
    return os.environ.get("SLURM_CLUSTER_NAME")


cluster = get_cluster()

dpath_root = Path(os.getcwd())

# cluster-agnostic entries (all derived relative to the repo)
paths = {
    "root": dpath_root,
    "config": dpath_root / "config",
    "artifacts": dpath_root / "artifacts",
    "data": {
        "cub": dpath_root / "data/cub",
    },
    "raw_tree": {
        "bryo": dpath_root / "data/bryo/SI_Fig1(BIG).newick",
        "cub": dpath_root / "data/cub/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy",
        "cub_jetz": dpath_root / "data/cub/AllBirdsHackett1.tre",
        "lepid": dpath_root / "data/lepid/tree_renamed_full.tre",
        "nymph": dpath_root / "data/nymph/tree_nymphalidae_chazot2021_all.tree",
    },
    # Prebuilt image packs (tools/build_img_cache.py): one indexable blob per dataset, two levels above the
    # project root so a single cache serves every checkout.
    "img_cache": dpath_root.parent.parent / "img_cache",
}

# cluster-specific entries: shared storage roots, and the datasets available on the cluster
if cluster == "ice-slurm":  # PACE-ICE

    dpath_haag = Path("/storage/ice-shared/cs8903onl")
    dpath_lepid = dpath_haag / "butterflies"

    datasets = ["bryo", "cub", "lepid"]
    paths["hf_cache"] = dpath_haag / "huggingface_cache"
    paths["imgs"] = {
        "bryo": dpath_haag / "bryo",
        "cub": dpath_root / "data/cub/CUB_200_2011/images",
        "lepid": dpath_lepid / "images",
    }
    paths["csv"] = {
        "lepid": {
            "imgs": dpath_lepid / "metadata/data_meta-clean_rot_512-butterflies_whole_specimen-v2025_05_07.csv",
            "tax": dpath_lepid / "metadata/data_tree_meta.csv",
        },
    }
elif cluster == "hipergator":  # HiPerGator

    dpath_group = Path("/lustre/blue2/arthur.porto")
    dpath_data = dpath_group / "data"
    dpath_nymph = dpath_data / "datasets/nymphalidae_whole_specimen-v250613"
    dpath_lepid = dpath_data / "datasets/butterflies_whole_specimen-clean_rot_512-v2025_05_07"

    datasets = ["bryo", "cub", "lepid", "nymph"]
    paths["hf_cache"] = dpath_data / "cache/huggingface/hub"
    paths["imgs"] = {
        "bryo": dpath_group / "odobon3.gatech/bryo",
        "cub": dpath_root / "data/cub/CUB_200_2011/images",
        "lepid": dpath_lepid / "images",
        "nymph": dpath_nymph / "images",
    }
    paths["csv"] = {
        "lepid": {
            "imgs": dpath_lepid / "metadata/data_meta-clean_rot_512-butterflies_whole_specimen-v2025_05_07.csv",
            "tax": dpath_lepid / "metadata/data_tree_meta.csv",
        },
        "nymph": {
            "imgs": dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv",
            "tax": dpath_nymph / "metadata/data_meta-nymphalidae_whole_specimen-v250613.csv",
        },
    }

paths["preproc"] = {dataset: dpath_root / "preprocessing" / dataset for dataset in datasets}
paths["metadata"] = {dataset: dpath_root / "metadata" / dataset for dataset in datasets}

DATASET_ALIAS2NAME = {
    "bryo": "Bryozoa",
    "cub": "CUB",
    "lepid": "Lepidoptera",
    "nymph": "Nymphalidae",
}
