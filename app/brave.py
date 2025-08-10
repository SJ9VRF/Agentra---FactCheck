import os, time, random, threading, requests
from typing import List, Dict, Any, Optional
from .utils import domain_ok, dedupe_urls, TTLCache, sha1
from .config import BRAVE_API_KEY, WHITELIST

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# ---- Tunables via env ----
_BRAVE_RPS = float(os.getenv("BRAVE_RPS", "1.0"))          # requests per second overall
_BRAVE_BURST = int(os.getenv("BRAVE_BURST", "2"))          # short burst capacity
_BRAVE_MAX_RETRIES = int(os.getenv("BRAVE_MAX_RETRIES", "4"))
_BACKOFF_BASE_MS = int(os.getenv("BRAVE_BACKOFF_BASE_MS", "250"))
_CACHE_TTL_SEC = int(os.getenv("BRAVE_CACHE_TTL_SEC", "1800"))  # 30 min

_cache = TTLCache(ttl_sec=_CACHE_TTL_SEC, max_items=1024)


class _TokenBucket:
    """
    Simple leaky bucket: allows BRAVE_BURST immediate tokens, then refills at BRAVE_RPS.
    Thread-safe and process-local.
    """
    def __init__(self, rps: float, burst: int):
        self.capacity = max(1, burst)
        self.tokens = self.capacity
        self.rate = max(0.1, rps)
        self.updated = time.time()
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            now = time.time()
            # refill
            delta = now - self.updated
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

_BUCKET = _TokenBucket(_BRAVE_RPS, _BRAVE_BURST)


def _sleep_ms(ms: int):
    time.sleep(ms / 1000.0)


class BraveClient:
    def __init__(self, api_key: Optional[str] = None, whitelist: Optional[str] = None):
        self.api_key = api_key or BRAVE_API_KEY
        self.whitelist = [d.strip() for d in (whitelist or ",".join(WHITELIST)).split(",") if d.strip()]

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing BRAVE_API_KEY")

        # Rate-limit gate
        while not _BUCKET.take():
            _sleep_ms(50)

        headers = {"X-Subscription-Token": self.api_key}
        # Retry with exponential backoff + jitter
        attempt = 0
        last_exc = None
        while attempt <= _BRAVE_MAX_RETRIES:
            try:
                r = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=15)
                # 429 or 5xx => retry
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    attempt += 1
                    # backoff with jitter
                    delay = (_BACKOFF_BASE_MS * (2 ** (attempt - 1))) + random.randint(0, 150)
                    _sleep_ms(min(4000, delay))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                # Non-retryable 4xx
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise
                last_exc = e
            except Exception as e:
                last_exc = e
                attempt += 1
                delay = (_BACKOFF_BASE_MS * (2 ** (attempt - 1))) + random.randint(0, 150)
                _sleep_ms(min(4000, delay))
        # exhausted
        if last_exc:
            raise last_exc
        raise RuntimeError("Brave request failed after retries")

    def search(self, query: str, count: int = 6) -> List[Dict[str, Any]]:
        key = "brv:" + sha1(f"{query}|{count}|{','.join(self.whitelist)}")
        cached = _cache.get(key)
        if cached is not None:
            return cached

        params = {
            "q": query,
            "count": max(1, min(count, 20)),
            "freshness": "month",
            "safesearch": "moderate",
        }
        try:
            data = self._request(params)
        except Exception as e:
            # On failure, attempt stale cache fallback by ignoring count in key
            stale = _cache.get("brv:" + sha1(f"{query}|*|{','.join(self.whitelist)}"))
            if stale is not None:
                return stale
            # bubble up if nothing cached
            raise

        results = []
        for cat in ("web", "news"):
            block = data.get(cat, {}).get("results", []) or []
            for item in block:
                results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("description") or item.get("snippet"),
                    "source": item.get("source"),
                    "published": item.get("published") or item.get("date"),
                })
        # apply whitelist + dedupe
        results = [r for r in results if domain_ok(r.get("url", ""), self.whitelist)]
        results = dedupe_urls(results)[:count]
        # write two cache keys (exact + wildcard fallback)
        _cache.set(key, results)
        _cache.set("brv:" + sha1(f"{query}|*|{','.join(self.whitelist)}"), results)
        return results
