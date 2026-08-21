from typing import List

import requests

import pdb


def gbif_name_candidates(cid: str, session=None, timeout: int = 30) -> List[str]:
    """
    Candidate binomials for a cid, in priority order: the cid itself, its GBIF
    accepted binomial, then species-rank synonyms of the accepted usage. All
    lowercase genus_species strings.
    """
    http = session if session is not None else requests
    candidates = [cid]

    r = http.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": cid.replace("_", " ")},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("matchType") != "EXACT" or data.get("speciesKey") is None:
        return candidates

    accepted_key = data["speciesKey"]
    accepted = data.get("species")
    if accepted is not None and len(accepted.split()) == 2:
        candidates.append(accepted.lower().replace(" ", "_"))

    synonyms = []
    offset = 0
    while True:
        r = http.get(
            f"https://api.gbif.org/v1/species/{accepted_key}/synonyms",
            params={"limit": 100, "offset": offset},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        for result in data.get("results", []):
            canonical = result.get("canonicalName")
            if result.get("rank") == "SPECIES" and canonical is not None and len(canonical.split()) == 2:
                synonyms.append(canonical.lower().replace(" ", "_"))
        if data.get("endOfRecords", True):
            break
        offset += 100
    candidates.extend(sorted(set(synonyms)))

    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]

def gbif_species_key(name: str, session=None, timeout: int = 30):
    """
    speciesKey of the EXACT GBIF match for a binomial (the accepted species' key,
    also for synonyms), or None when the name doesn't resolve to a single species
    (e.g. unresolvable homonyms).
    """
    http = session if session is not None else requests
    r = http.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": name.replace("_", " ")},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("matchType") != "EXACT":
        return None
    return data.get("speciesKey")
