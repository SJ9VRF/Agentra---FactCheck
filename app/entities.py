import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from dateutil import parser as dp

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
MONTHS = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
DATE_WORD_RE = re.compile(rf"\b({MONTHS})\s+\d{{1,2}},\s*(\d{{4}})\b", re.IGNORECASE)

def extract_years(text: str) -> List[int]:
    if not text:
        return []
    out = []
    for m in YEAR_RE.finditer(text):
        try:
            y = int(m.group(0))
            if 1900 <= y <= 2100:
                out.append(y)
        except Exception:
            pass
    return out

def extract_dates(text: str) -> List[str]:
    if not text:
        return []
    dates = []
    for m in DATE_WORD_RE.finditer(text):
        try:
            # normalize to ISO date
            dt = dp.parse(m.group(0), fuzzy=True)
            dates.append(dt.date().isoformat())
        except Exception:
            pass
    return dates

def evidence_years(evidence: List[Dict[str, Any]]) -> List[int]:
    years: List[int] = []
    for ev in evidence:
        for key in ("published", "title", "snippet"):
            val = ev.get(key)
            if not val:
                continue
            # First try strict ISO parse for 'published'
            if key == "published":
                try:
                    dt = dp.parse(val)
                    if 1900 <= dt.year <= 2100:
                        years.append(dt.year)
                        continue
                except Exception:
                    pass
            # Fallback regex on any field
            years.extend(extract_years(str(val)))
    return years

def consensus_year(years: List[int]) -> Optional[Tuple[int, int]]:
    if not years:
        return None
    c = Counter(years).most_common(1)[0]  # (year, count)
    return c

def temporal_checks(claim_text: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compare the claim's explicit year/date to the evidence majority year.
    Returns a list of checks, each containing mismatch info and suggested correction.
    """
    claim_years = extract_years(claim_text)
    ev_year_list = evidence_years(evidence)
    checks: List[Dict[str, Any]] = []
    if not claim_years or not ev_year_list:
        return checks

    # If the claim has exactly one explicit year, we can compare directly.
    if len(claim_years) == 1:
        claim_year = claim_years[0]
        cons = consensus_year(ev_year_list)
        if cons:
            top_year, freq = cons
            # confidence heuristic: support from at least 2 independent results
            conf = freq / max(3, len(ev_year_list))
            if top_year != claim_year and freq >= 2:
                # choose up to 3 supporting sources that mention the consensus year
                sup = []
                for ev in evidence:
                    if len(sup) >= 3:
                        break
                    hay = f"{ev.get('published','')} {ev.get('title','')} {ev.get('snippet','')}"
                    if str(top_year) in hay:
                        sup.append(ev.get("url"))
                checks.append({
                    "field": "year",
                    "status": "mismatch",
                    "claim": claim_year,
                    "evidence_consensus": top_year,
                    "supporting_sources": [u for u in sup if u],
                    "confidence": round(float(conf), 3),
                    "suggested_claim": YEAR_RE.sub(str(top_year), claim_text, count=1)
                })
    return checks
