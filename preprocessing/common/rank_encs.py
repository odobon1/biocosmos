from typing import Sequence
from bidict import bidict  # type: ignore[import]

from utils.utils import paths, load_pickle, save_pickle


def build_rank_encs(
    dataset: str,
    ranks: Sequence[str],
) -> dict[str, bidict]:
    """
    Build rank key maps with deterministic ordering.

    Args:
    - dataset: dataset name, e.g. "nymph" or "lepid"
    - ranks: ordered list of ranks to build, e.g. ["genus", "species"]

    Returns:
    - dict rank_name -> bidict(rank_value -> integer_key)
    """
    class_data = load_pickle(paths["metadata"][dataset] / "class_data.pkl")
    sids_sorted = sorted(class_data.keys())

    rank_encs: dict[str, bidict] = {rank: bidict() for rank in ranks}

    if "species" in rank_encs:
        for rkey_species, sid in enumerate(sids_sorted):
            rank_encs["species"][sid] = rkey_species

    for rank in ranks:
        if rank == "species":
            continue

        rank_values = sorted({class_data[sid][rank] for sid in sids_sorted})
        for rkey, value in enumerate(rank_values):
            rank_encs[rank][value] = rkey

    save_pickle(rank_encs, paths["metadata"][dataset] / "rank_encs.pkl")
