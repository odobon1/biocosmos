from pathlib import Path

from utils import paths


FPATH_OUT = Path("tools/full_src.txt")

dpath_root = paths["root"]

RFPATHS = [
    "config/train/lr_sched.yaml",
    "config/train/train.yaml",
    "config/eval.yaml",
    "config/gen_split.yaml",
    "metadata/gen_rank_keys.py",
    "metadata/gen_species_ids.py",
    "metadata/gen_split.py",
    "metadata/gen_tax_gbif.py",
    "metadata/gen_tax_ncbi.py",
    "metadata/gen_tax_nymph.py",
    "environment_b200.yaml",
    "eval.py",
    "models.py",
    "train.py",
    "utils_data.py",
    "utils_eval.py",
    "utils_imb.py",
    "utils_pp.py",
    "utils.py",
]

fpaths = [dpath_root / rfpath for rfpath in RFPATHS]

def main():
    out_lines: list[str] = []

    for rfpath in RFPATHS:

        fpath = dpath_root / rfpath

        if not fpath.exists():
            print(f"Warning: {fpath} not found, skipping")
            continue

        try:
            file_content = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"Warning: {fpath} is not UTF-8 decodable; skipping.")
            continue

        out_lines.append("=== " + rfpath + " ===")
        out_lines.append("")
        out_lines.append(file_content.rstrip())
        out_lines.append("")
        out_lines.append("")

    FPATH_OUT.parent.mkdir(parents=True, exist_ok=True)
    FPATH_OUT.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote combined file to: {FPATH_OUT.resolve()}")

if __name__ == "__main__":
    main()