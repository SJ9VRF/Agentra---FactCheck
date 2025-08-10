import re, io, hashlib, time
from urllib.parse import urlparse
from typing import List, Optional

def clean_text(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())

def domain_ok(url: str, whitelist: Optional[List[str]]) -> bool:
    if not whitelist:
        return True
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host.endswith(w.strip().lower()) for w in whitelist if w.strip())

def dedupe_urls(items):
    seen = set()
    out = []
    for it in items:
        url = it.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(it)
    return out

class TTLCache:
    def __init__(self, ttl_sec=900, max_items=256):
        self.ttl = ttl_sec
        self.max = max_items
        self.store = {}

    def _prune(self):
        now = time.time()
        keys = list(self.store.keys())
        for k in keys:
            v, exp = self.store[k]
            if now > exp:
                self.store.pop(k, None)
        # size prune
        if len(self.store) > self.max:
            for k in list(self.store.keys())[:len(self.store)-self.max]:
                self.store.pop(k, None)

    def get(self, key):
        self._prune()
        item = self.store.get(key)
        if not item: return None
        v, exp = item
        if time.time() > exp:
            self.store.pop(key, None)
            return None
        return v

    def set(self, key, value):
        self._prune()
        self.store[key] = (value, time.time() + self.ttl)

def sha1(data: str) -> str:
    return hashlib.sha1(data.encode("utf-8")).hexdigest()
