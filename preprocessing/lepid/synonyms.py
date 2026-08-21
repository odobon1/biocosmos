"""
python -m preprocessing.lepid.synonyms

Creates:
preprocessing/lepid/intermediaries/synonyms.pkl

The image directories and the raw trees sometimes use different generic combinations
for the same species (e.g. class chilasa_clytia is on the lepid tree as
papilio_clytia). For each class absent from both raw trees, this step resolves the
class's GBIF accepted name and synonyms and looks for one of them among the raw tree
tips. Hits are recorded for the phylo step in two maps:

- "rename" {cid: tip}: the tip is not itself a class -- the phylo step renames it to
  the class id, so the class inherits its real tree placement instead of being
  polytomy-grafted.
- "sister" {cid: host_cid}: the tip is another class (the dataset holds duplicate
  classes for the same species under two names) -- the phylo step grafts the class
  as a zero-length sister of that host tip, sharing its placement.

class_data and the tax CSV are untouched -- ranks keep coming from the CSV only.

Guards: a hit counts only if the tip name's own GBIF resolution does not point at a
different species than the class resolves to (one-directional GBIF chains toward a
separately-accepted species are almost always backbone errors; an ambiguous reverse
match, e.g. a historical homonym, does not veto an explicit forward synonymy); tips
found only on the nymph tree are usable only for nymphalid classes (the merge
attaches the whole nymph tree at the Nymphalidae anchor, so its outgroup tips carry
no usable placement for other families); a mapping is dropped if the tax CSV gives
the tip's genus a different family than the class's; and when a tip's claimants
resolve to different species (possible only via an ambiguous reverse match), the
tip's own epithet arbitrates. Classes that then share one rename tip are
conspecifics of one another: the first takes the tip, the rest join it as
zero-length sisters.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from preprocessing.common.cid2commons import _get_thread_session
from preprocessing.common.gbif import gbif_name_candidates, gbif_species_key
from preprocessing.lepid.phylo import build_tree_lepid
from preprocessing.nymph.phylo import build_tree_nymph
from utils.utils import paths, load_pickle, save_pickle

import pdb


def get_raw_tree_tips() -> Tuple[Set[str], Set[str]]:
    tips_lepid = {tip.name for tip in build_tree_lepid().get_terminals()}
    tips_nymph = {tip.name for tip in build_tree_nymph().get_terminals()}
    return tips_lepid, tips_nymph

def genus_to_family_csv() -> Dict[str, str]:
    df = pd.read_csv(paths["csv"]["lepid"]["tax"])
    fam_sets = df.groupby("genus")["family"].agg(set)
    return {genus: next(iter(fams)) for genus, fams in fam_sets.items() if len(fams) == 1}

def resolve_synonyms(
    cids_absent: List[str],
    tips: Set[str],
    tips_nymph_only: Set[str],
    cids_nymphalid: Set[str],
    max_workers: int = 16,
) -> Tuple[Dict[str, str], Dict[str, Optional[int]]]:
    """
    cid -> raw tree tip carrying the same species under another name, via GBIF,
    plus each mapped cid's GBIF speciesKey. A hit is rejected if the tip name's
    own GBIF resolution points at a different species than the cid resolves to.
    Tips found only on the nymph tree are usable only for nymphalid classes: the
    merge attaches the whole nymph tree at the Nymphalidae anchor, so its
    outgroup tips carry no usable placement for other families.
    """
    thread_local = threading.local()

    def fetch_one(cid: str) -> Tuple[str, Optional[str], Optional[int]]:
        session = _get_thread_session(thread_local, max_workers)
        try:
            cid_key = gbif_species_key(cid, session=session)
            for candidate in gbif_name_candidates(cid, session=session):
                if candidate in tips_nymph_only and cid not in cids_nymphalid:
                    continue
                if candidate in tips:
                    tip_key = gbif_species_key(candidate, session=session)
                    if tip_key is None or tip_key == cid_key:
                        return cid, candidate, cid_key
        except Exception as exc:
            print(f"[WARN] GBIF failed for {cid}: {exc}")
        return cid, None, None

    mapping: Dict[str, str] = {}
    keys: Dict[str, Optional[int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, cid): cid for cid in sorted(cids_absent)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resolving synonyms"):
            cid, tip, cid_key = future.result()
            if tip is not None:
                mapping[cid] = tip
                keys[cid] = cid_key

    return mapping, keys

def main():
    print("Resolving class synonyms against the raw trees...")

    class_data = load_pickle(paths["metadata"]["lepid"] / "class_data.pkl")
    tips_lepid, tips_nymph = get_raw_tree_tips()
    tips = tips_lepid | tips_nymph
    cids = set(class_data.keys())
    cids_absent = sorted(cids - tips)
    cids_nymphalid = {cid for cid in cids_absent if class_data[cid]["family"] == "nymphalidae"}

    mapping, keys = resolve_synonyms(cids_absent, tips, tips_nymph - tips_lepid, cids_nymphalid)
    print(f"{len(mapping)}/{len(cids_absent)} absent classes found on the trees under a synonym")

    # reject mappings that contradict the tax CSV's family for the tip's genus
    genus2fam = genus_to_family_csv()
    for cid, tip in sorted(mapping.items()):
        tip_family = genus2fam.get(tip.split("_", 1)[0])
        if tip_family is not None and tip_family != class_data[cid]["family"]:
            print(f"  dropping {cid} -> {tip}: tip family {tip_family} != class family {class_data[cid]['family']}")
            del mapping[cid]

    # contested tips: claimants resolving to different species can't all be
    # conspecific with the tip -- the tip's own epithet arbitrates (claimants
    # sharing it win); if that still leaves mixed species, the tip is dropped
    by_tip: Dict[str, List[str]] = {}
    for cid, tip in mapping.items():
        by_tip.setdefault(tip, []).append(cid)
    for tip, contenders in sorted(by_tip.items()):
        if len({keys[cid] for cid in contenders}) <= 1:
            continue
        tip_epithet = tip.split("_", 1)[1]
        winners = [cid for cid in contenders if cid.split("_", 1)[1] == tip_epithet]
        if winners and len({keys[cid] for cid in winners}) == 1:
            losers = sorted(set(contenders) - set(winners))
            print(f"  {tip} contested: keeping {sorted(winners)}, dropping {losers}")
        else:
            losers = sorted(contenders)
            print(f"  {tip} contested with no epithet arbiter: dropping {losers}")
        for cid in losers:
            del mapping[cid]

    # tips that are themselves classes host their synonym classes as zero-length
    # sisters; other tips get renamed. When several classes claim one rename tip
    # they are conspecifics of one another: the first takes the tip, the rest
    # join it as sisters.
    sister = {cid: tip for cid, tip in mapping.items() if tip in cids}
    rename = {cid: tip for cid, tip in mapping.items() if tip not in cids}
    claimed: Dict[str, List[str]] = {}
    for cid, tip in rename.items():
        claimed.setdefault(tip, []).append(cid)
    for tip, contenders in sorted(claimed.items()):
        if len(contenders) > 1:
            first, *rest = sorted(contenders)
            print(f"  {tip}: renamed to {first}; {rest} grafted as conspecific sisters")
            for cid in rest:
                del rename[cid]
                sister[cid] = first

    synonyms = {"rename": rename, "sister": sister}
    save_pickle(synonyms, paths["preproc"]["lepid"] / "intermediaries/synonyms.pkl")
    print(f"Synonyms complete: {len(rename)} classes mapped to tree tips, "
          f"{len(sister)} conspecific with another class")


if __name__ == "__main__":
    main()
