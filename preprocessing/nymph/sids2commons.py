"""
python -m preprocessing.nymph.sids2commons
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm  # type: ignore[import]

from utils.utils import paths, save_pickle
from preprocessing.nymph.species_ids import get_sids_nymph


MAX_WORKERS = 16


_thread_local = threading.local()

def make_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "biocosmos-common-name-fetcher/1.0"
    })
    return session

def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session()
    return _thread_local.session

def gbif_common_name(scientific_name: str, lang: str = "eng", timeout: int = 10) -> str | None:
    session = get_session()

    r1 = session.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": scientific_name},
        timeout=timeout,
    )
    r1.raise_for_status()
    m = r1.json()

    key = m.get("usageKey") or m.get("speciesKey") or m.get("acceptedUsageKey")
    if not key:
        return None

    r2 = session.get(
        f"https://api.gbif.org/v1/species/{key}/vernacularNames",
        timeout=timeout,
    )
    r2.raise_for_status()
    data = r2.json()

    items = [
        x for x in data.get("results", [])
        if (x.get("language") or "").lower() == lang.lower()
    ]
    if not items:
        return None

    name = (items[0].get("vernacularName") or "").strip()
    return name or None

def fetch_one(sid: str) -> tuple[str, str | None]:
    scientific_name = sid.replace("_", " ")
    try:
        common = gbif_common_name(scientific_name)
        return sid, common
    except Exception as e:
        print(f"[WARN] Failed for {sid}: {e}")
        return sid, None

def main() -> None:

    sids = get_sids_nymph()
    fpath_sids2commons = paths["preproc"]["nymph"] / "intermediaries/sids2commons.pkl"

    sids2commons = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, sid): sid for sid in sids}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Retrieving Common Names"):
            sid, common = fut.result()
            sids2commons[sid] = common

    save_pickle(sids2commons, fpath_sids2commons)


if __name__ == "__main__":
    main()