"""
python -m preprocessing.cub.rank_encs
"""

from preprocessing.common.rank_encs import build_rank_encs


def generate_rank_encs():
    build_rank_encs(
        dataset="cub",
        ranks=["order", "family", "genus", "species"],
    )

def main() -> None:
    print("Building rank encodings...")
    generate_rank_encs()
    print("Rank encodings complete")


if __name__ == "__main__":
    main()