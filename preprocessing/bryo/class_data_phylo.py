"""
python -m preprocessing.bryo.class_data_phylo
"""

import re
from Bio import Phylo  # type: ignore[import]
from Bio.Phylo.BaseTree import Tree  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]
from pathlib import Path
import requests  # type: ignore[import]

from utils.utils import paths, save_pickle, get_subdirectory_names


def build_tree_bryo() -> Tree:
    tree = Phylo.read(paths["root"] / "metadata/bryo/SI_Fig1(BIG).newick", "newick")
    tip_names_trunc = set()
    for tip in tree.get_terminals():
        tip_name_trunc = trim_after_all_caps_underscore(tip.name).lower()
        # scrupocellaria_reptans is the one species that was having taxonomic conflicts
        if tip_name_trunc in tip_names_trunc or tip.name.startswith("UNKNOWN") or tip_name_trunc == "scrupocellaria_reptans":
            tree.prune(target=tip.name)
        else:
            tip.name = tip_name_trunc
            tip_names_trunc.add(tip_name_trunc)
    return tree

# build_tree_bryo() helper
def trim_after_all_caps_underscore(s: str) -> str:
    """
    Remove the first underscore followed by an ALL-CAPS token and everything after it.

    Examples
    --------
    'Fenestrulina_n_sp1_BLEED237' -> 'Fenestrulina_n_sp1'
    'Fenestrulina_sp_SEQ_BLEED1261' -> 'Fenestrulina_sp'
    """
    return re.sub(r'_[A-Z][A-Z0-9_]*$', '', s)

def group_by_genus(species_names):
    """
    Convert a list like:
        ['adeona_japonica', 'adeona_sp1', 'adeonella_calveti']
    into:
        {
            'adeona': ['adeona_japonica', 'adeona_sp1'],
            'adeonella': ['adeonella_calveti']
        }

    Assumes the genus is the part before the first underscore.
    """
    grouped = {}

    for name in species_names:
        genus = name.split("_", 1)[0]
        if genus not in grouped:
            grouped[genus] = []
        grouped[genus].append(name)

    return grouped

def build_class_data(genera_imgs, g2s):
    class_data = {}
    for g in tqdm(genera_imgs):
        if g in g2s.keys():
            for s in g2s[g]:
                tax = sci2tax(s)
                if tax is not None:
                    tax["common_name"] = None
                    class_data[g] = tax
                    # first taxonomic match is correct (this was verified manually)
                    break
    return class_data

# build_class_data() helper
def sci2tax(
    sci_name: str,
    timeout: int = 30,
) -> dict:
    """
    Resolve a scientific name with GBIF and return taxonomic info.

    Parameters
    ----------
    sci_name : str
        Full scientific name, e.g. "Orthoporidra compacta"
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict
        Structured result with match metadata and taxonomic ranks.
    """
    sci_name = sci_name.replace("_", " ")

    url = "https://api.gbif.org/v1/species/match"
    params = {
        "name": sci_name,
        "verbose": "true",
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    match_type = data.get("matchType")
    rank = data.get("rank")
    if rank != "SPECIES" or match_type != "EXACT" or match_type == "NONE":
        return None

    tax = {}

    # only genus that was missing family in GBIF
    if sci_name.startswith("klugeflustra"):
        tax["family"] = "flustridae"
    else:
        tax["family"] = data.get("family")
        if tax["family"] is not None:
            tax["family"] = tax["family"].lower()
        else:
            return None
    tax["genus"] = data.get("genus")
    if tax["genus"] is not None:
        tax["genus"] = tax["genus"].lower()
    else:
        return None

    return tax

def prune_tree(tree, class_data):
    genera_seen = set()
    for tip in tree.get_terminals():
        genus = tip.name.split("_", 1)[0]
        if genus not in class_data.keys() or genus in genera_seen:
            tree.prune(target=tip.name)
        else:
            tip.name = genus
        genera_seen.add(genus)
    return tree

def main():
    print("Building class data and tree...")

    tree = build_tree_bryo()
    sids_tree = [tip.name for tip in tree.get_terminals()]
    g2s = group_by_genus(sids_tree)  # genus --> [sids] mapping (species names used to grab tax info from GBIF)

    genera_imgs = get_subdirectory_names(paths["bryo_imgs"])
    class_data = build_class_data(genera_imgs, g2s)

    tree = prune_tree(tree, class_data)

    save_pickle(class_data, paths["metadata"]["bryo"] / "class_data.pkl")
    save_pickle(tree, paths["metadata"]["bryo"] / "tree.pkl")

    print("Class data and tree complete")


if __name__ == "__main__":
    main()