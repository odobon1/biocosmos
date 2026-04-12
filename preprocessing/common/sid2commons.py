from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, Iterable

import requests  # type: ignore[import]
from requests.adapters import HTTPAdapter  # type: ignore[import]
from urllib3.util.retry import Retry  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]


def _make_session(max_workers: int) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=max_workers, pool_maxsize=max_workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "biocosmos-common-name-fetcher/1.0"
    })
    return session

def _get_thread_session(thread_local: threading.local, max_workers: int) -> requests.Session:
    if not hasattr(thread_local, "session"):
        thread_local.session = _make_session(max_workers)
    return thread_local.session

def _gbif_common_name(
    scientific_name: str,
    thread_local: threading.local,
    max_workers: int,
    lang: str = "eng",
    timeout: int = 10,
) -> str | None:
    session = _get_thread_session(thread_local, max_workers)

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

def build_sids2commons(
    sids: Iterable[str],
    *,
    max_workers: int = 16,
    lang: str = "eng",
    timeout: int = 10,
    progress_desc: str = "Retrieving Common Names",
) -> Dict[str, str | None]:
    sids = sorted(sids)
    sids2commons: Dict[str, str | None] = {}
    thread_local = threading.local()

    def fetch_one(sid: str) -> tuple[str, str | None]:
        scientific_name = sid.replace("_", " ")
        try:
            common = _gbif_common_name(
                scientific_name,
                thread_local=thread_local,
                max_workers=max_workers,
                lang=lang,
                timeout=timeout,
            )
            return sid, common
        except Exception as exc:
            print(f"[WARN] Failed for {sid}: {exc}")
            return sid, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sid): sid for sid in sids}
        for future in tqdm(as_completed(futures), total=len(futures), desc=progress_desc):
            sid, common = future.result()
            sids2commons[sid] = common

    return sids2commons
