"""
python -m tools.src_comp
"""

from pathlib import Path

from utils.utils import paths


FPATH_OUT = Path("tools/src.txt")

dpath_root = paths["root"]

dpaths_logs = [
    # "artifacts/dev/dev_clip_cb-foc-2D/42/logs/",
    # "artifacts/dev/dev_clip_cb-foc/42/logs/",
    "artifacts/dev/dev/42/logs/",
]

fpaths_logs = []
# for dpath_logs in dpaths_logs:
#     # fpaths_logs.append(dpath_logs + "batch.log")
#     fpaths_logs.append(dpath_logs + "epoch.log")
#     fpaths_logs.append(dpath_logs + "init.log")

RFPATHS = fpaths_logs + [
    "config/train/lr_sched.yaml",
    "config/train/train.yaml",
    "config/eval.yaml",
    "config/gen_split.yaml",
    "config/hardware.yaml",
    "config/loss.yaml",
    "config/loss2.yaml",
    "metadata/nymph/splits/.gitkeep",
    "preprocessing/nymph/intermediaries/.gitkeep",
    "preprocessing/nymph/gen_class_data.py",
    "preprocessing/nymph/gen_rank_keys.py",
    "preprocessing/nymph/gen_sids2commons.py",
    "preprocessing/nymph/gen_split.py",
    "preprocessing/nymph/species_ids.py",
    "preprocessing/README.md",
    "tools/probe_batch_size.py",
    "tools/protos.py",
    "tools/readable_tree.py",
    "tools/src_comp.py",
    "tools/vis_manifold.py",
    "utils/config.py",
    "utils/data.py",
    "utils/ddp.py",
    "utils/eval.py",
    "utils/hardware.py",
    "utils/imb.py",
    "utils/loss.py",
    "utils/phylo.py",
    "utils/pp.py",
    "utils/train.py",
    "utils/utils.py",
    "environment_b200.yaml",
    "eval.py",
    "models.py",
    "README.md",
    "setup.sh",
    "train.py",
]

fpaths = [dpath_root / rfpath for rfpath in RFPATHS]

def main():
    out_lines: list[str] = []

    for rfpath in RFPATHS:

        fpath        = dpath_root / rfpath
        file_content = fpath.read_text(encoding="utf-8")

        out_lines.append("=== " + rfpath + " ===")
        out_lines.append("")
        out_lines.append(file_content.rstrip())
        out_lines.append("")
        out_lines.append("")

    FPATH_OUT.parent.mkdir(parents=True, exist_ok=True)
    FPATH_OUT.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()