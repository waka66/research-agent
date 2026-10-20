# graph/cache.py
import hashlib
import json
import os
import time
from pathlib import Path

CACHE_DIR = Path(os.getenv("SEARCH_CACHE_DIR", ".cache/search"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OFFLINE = os.getenv("SEARCH_OFFLINE", "0") == "1"      # 只读缓存，不联网


def _key(query: str, namespace: str) -> str:
    norm = "".join(query.lower().split())
    return hashlib.sha1(f"{namespace}|{norm}".encode("utf-8")).hexdigest()[:16]


def get(query: str, namespace: str, max_age_days: float | None = None) -> dict | None:
    path = CACHE_DIR / f"{_key(query, namespace)}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if max_age_days is not None:
        age_days = (time.time() - data.get("fetched_at", 0)) / 86400
        if age_days > max_age_days:
            return None
    return data.get("payload")


def put(query: str, payload: dict, namespace: str) -> None:
    path = CACHE_DIR / f"{_key(query, namespace)}.json"
    path.write_text(
        json.dumps({"fetched_at": time.time(), "payload": payload}, ensure_ascii=False),
        encoding="utf-8",
    )