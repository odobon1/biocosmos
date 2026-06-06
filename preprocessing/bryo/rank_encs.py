"""
python -m preprocessing.bryo.rank_encs
"""

from preprocessing.common.rank_encs import build_rank_encs


def generate_rank_encs():
    build_rank_encs(
        dataset="bryo",
        ranks=["family", "genus"],
    )

def main() -> None:
    print("Building rank encodings...")
    generate_rank_encs()
    print("Rank encodings complete")


if __name__ == "__main__":
    main()