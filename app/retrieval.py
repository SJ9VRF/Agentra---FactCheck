import math
import time
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

from .brave import BraveClient
from .utils import dedupe_urls, clean_text


# Simple credibility priors (tune as you like)
DOMAIN_CREDIBILITY = {
    # science / gov
    "nasa.gov": 1.00, "who.int": 0.98, "cdc.gov": 0.98, "esa.int": 0.96,
    "nih.gov": 0.96, "noaa.gov": 0.95,
    # major news
    "reuters.com": 0.93, "apnews.com": 0.92, "bbc.com": 0.92, "nytimes.com": 0.91,
    "washingtonpost.com": 0.90, "theguardian.com": 0.90,
    # knowledge bases
    "wikipedia.org": 0.85, "britannica.com": 0.88, "nature.com": 0.95, "sciencedirect.com": 0.93,
}

def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _credibility(url: str) -> float:
    h = _host(url)
    for dom, w in DOMAIN_CREDIBILITY.items():
        if h.endswith(dom):
            return w
    return 0.70  # default prior

def _freshness(published: str) -> float:
    """
    Very rough freshness score [0..1].
    If a date-like string exists and is within ~2 years -> higher.
    """
    if not published:
        return 0.5
    try:
        import dateutil.parser as dp
        dt = dp.parse(published, fuzzy=True)
        age_days = max(0.0, (time.time() - dt.timestamp()) / 86400.0)
        if age_days < 30:
            return 1.0
        if age_days < 180:
            return 0.9
        if age_days < 365:
            return 0.8
        if age_days < 365 * 2:
            return 0.7
        return 0.5
    except Exception:
        return 0.6

def _keyword_overlap(query: str, title: str, snippet: str) -> float:
    """
    Token overlap between query and (title+snippet).
    """
    q = set(clean_text(query).lower().split())
    hay = set((clean_text(title) + " " + clean_text(snippet)).lower().split())
    common = q.intersection(hay)
    if not q:
        return 0.3
    return min(1.0, 0.3 + 0.7 * (len(common) / max(1, len(q))))

def score_item(item: Dict[str, Any], query: str) -> Dict[str, Any]:
    url = item.get("url", "")
    cred = _credibility(url)
    fresh = _freshness(item.get("published"))
    over = _keyword_overlap(query, item.get("title", ""), item.get("snippet", ""))
    score = 0.55 * cred + 0.25 * fresh + 0.20 * over
    enriched = dict(item)
    enriched["score"] = round(float(score), 4)
    enriched["credibility"] = round(float(cred), 3)
    enriched["freshness"] = round(float(fresh), 3)
    enriched["overlap"] = round(float(over), 3)
    enriched["host"] = _host(url)
    enriched["query_matched"] = query
    return enriched


class EvidenceRetriever:
    """
    Wraps Brave retrieval with light ranking and returns a trace for transparency.
    """

    def __init__(self):
        self.client = BraveClient()

    def retrieve(self, queries: List[str], per_query: int = 6, top_k: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        trace = {
            "queries": [],
            "raw": [],
            "ranked": [],
            "explanations": "score = 0.55*credibility + 0.25*freshness + 0.20*keyword_overlap",
        }
        all_scored: List[Dict[str, Any]] = []
        for q in queries[:5]:
            res = self.client.search(q, count=per_query)
            trace["queries"].append(q)
            for r in res:
                scored = score_item(r, q)
                all_scored.append(scored)
                trace["raw"].append(scored)

        # Deduplicate by URL, keep max score
        by_url: Dict[str, Dict[str, Any]] = {}
        for it in all_scored:
            u = it.get("url")
            if not u:
                continue
            prev = by_url.get(u)
            if not prev or it["score"] > prev["score"]:
                by_url[u] = it

        ranked = sorted(by_url.values(), key=lambda x: x["score"], reverse=True)
        ranked = ranked[:top_k]
        trace["ranked"] = [
            {k: v for k, v in it.items() if k in ("title", "url", "host", "score", "credibility", "freshness", "overlap", "query_matched", "published", "snippet")}
            for it in ranked
        ]
        return ranked, trace
