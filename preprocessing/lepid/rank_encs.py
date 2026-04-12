"""
Takes metadata/lepid/class_data.pkl structure and produces metadata/lepid/rank_encs.pkl structure
"""

from preprocessing.common.rank_encs import build_rank_encs


def generate_rank_encs():
    build_rank_encs(
        dataset="lepid",
        ranks=["family", "genus", "species"],
    )

def main() -> None:
    print("Building rank keys...")
    generate_rank_encs()
    print("Rank keys complete")


if __name__ == "__main__":
    main()