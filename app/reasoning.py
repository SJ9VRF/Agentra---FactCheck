import os
import json
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from .config import OPENAI_API_KEY, MAX_PARALLEL


MODEL = "gpt-5"


def _client() -> OpenAI:
    key = OPENAI_API_KEY
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=key)


def _get_text(resp) -> str:
    """Extract text from Responses API across SDK variants."""
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if t:
                    parts.append(t)
        return "".join(parts)
    except Exception:
        return ""


def _responses_create_sync(prompt: str) -> str:
    """Single place to call the Responses API without extra/unstable params."""
    client = _client()
    resp = client.responses.create(model=MODEL, input=prompt)
    return _get_text(resp)


# Limit concurrent model calls
_SEM = asyncio.Semaphore(max(1, int(MAX_PARALLEL or 4)))


async def extract_claims_and_queries(text: str) -> Dict[str, Any]:
    """Planner: subclaims + queries (JSON enforced by instruction)."""
    prompt = f"""You are a multi-modal fact-checking planner.

INPUT CONTENT:
{text}

TASK:
1) Extract core claim(s) as short, atomic subclaims.
2) Detect entities and time/place (when present).
3) Produce 3–5 diverse web search queries (for live search).
RESPONSE:
Return ONLY valid JSON (no extra text):
{{
  "subclaims": [{{"id":"C1","text":"...","time":"...","place":"..."}}],  // time/place optional
  "queries": ["...", "..."]
}}
"""
    async with _SEM:
        raw = await asyncio.to_thread(_responses_create_sync, prompt)

    try:
        js = json.loads(raw)
        if isinstance(js, dict) and "subclaims" in js and "queries" in js:
            return js
    except Exception:
        pass

    return {
        "subclaims": [{"id": "C1", "text": text[:300]}],
        "queries": [text[:120]],
    }


async def judge_entailment(subclaim: str, evidence: List[Dict[str, str]]) -> Tuple[str, float, str]:
    """Single verdict on a subclaim using the whole evidence pool."""
    sources = "\n\n".join(
        [f"- {e.get('title')}: {e.get('snippet')} (URL: {e.get('url')})" for e in evidence]
    )
    prompt = f"""You are a rigorous fact-checking judge.

SUBCLAIM:
"{subclaim}"

EVIDENCE (independent sources, may agree or conflict):
{sources}

TASK:
- Decide TRUE, FAKE, or UNVERIFIED based ONLY on the above evidence snippets.
- Provide a short rationale (2–3 sentences).
- Provide a confidence 0..1 based on agreement, quality, and recency.

RESPONSE:
Return ONLY valid JSON:
{{
  "label": "TRUE|FAKE|UNVERIFIED",
  "confidence": 0.0,
  "rationale": "..."
}}
"""
    async with _SEM:
        raw = await asyncio.to_thread(_responses_create_sync, prompt)

    try:
        data = json.loads(raw)
    except Exception:
        # try loose JSON grab
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start:end+1])
            except Exception:
                data = None
        else:
            data = None

    if not isinstance(data, dict):
        return ("UNVERIFIED", 0.5, "Insufficient or ambiguous evidence.")

    label = str(data.get("label", "UNVERIFIED")).upper()
    conf = float(data.get("confidence", 0.5))
    why = data.get("rationale", "")
    if label not in ("TRUE", "FAKE", "UNVERIFIED"):
        label = "UNVERIFIED"
    return (label, conf, why)


async def adversarial_debate(subclaims: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Three-agent debate: Analyst, Skeptic, Judge → final JSON verdict."""
    ev_text = "\n".join([f"* {e['title']} — {e['snippet']} ({e['url']})" for e in evidence[:12]])
    claims_text = "\n".join([f"- [{c.get('id','C?')}] {c.get('text','')}" for c in subclaims])

    analyst_prompt = f"""ROLE: Analyst
Build the best confirming case for the subclaims using the evidence.

SUBCLAIMS:
{claims_text}

EVIDENCE:
{ev_text}

OUTPUT:
- Bullet points only.
"""
    skeptic_prompt = f"""ROLE: Skeptic
Build the strongest refutation and highlight weaknesses.

SUBCLAIMS:
{claims_text}

EVIDENCE:
{ev_text}

OUTPUT:
- Bullet points only.
"""

    async with _SEM:
        analyst_text = await asyncio.to_thread(_responses_create_sync, analyst_prompt)
    async with _SEM:
        skeptic_text = await asyncio.to_thread(_responses_create_sync, skeptic_prompt)

    judge_prompt = f"""ROLE: Judge
Read Analyst and Skeptic notes and issue a final verdict for the entire claim set.

RESPONSE:
Return ONLY valid JSON:
{{
  "label": "TRUE|FAKE|UNVERIFIED",
  "confidence": 0.0,
  "rationale": "..."
}}

Analyst:
{analyst_text}

Skeptic:
{skeptic_text}
"""
    async with _SEM:
        judge_raw = await asyncio.to_thread(_responses_create_sync, judge_prompt)

    try:
        judge_data = json.loads(judge_raw)
    except Exception:
        start, end = judge_raw.find("{"), judge_raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                judge_data = json.loads(judge_raw[start:end+1])
            except Exception:
                judge_data = None
        else:
            judge_data = None

    if not isinstance(judge_data, dict):
        judge_data = {"label": "UNVERIFIED", "confidence": 0.5, "rationale": "Debate inconclusive."}

    return {
        "analyst": analyst_text,
        "skeptic": skeptic_text,
        "judge": judge_data
    }




# --- Sequential debate helpers (rate-limit friendly) ---

async def analyst_notes(subclaims: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> str:
    ev_text = "\n".join([f"* {e.get('title','')} — {e.get('snippet','')} ({e.get('url','')})" for e in evidence[:12]])
    claims_text = "\n".join([f"- [{c.get('id','C?')}] {c.get('text','')}" for c in subclaims])
    prompt = f"""ROLE: Analyst
Build the best confirming case for the subclaims using the evidence.

SUBCLAIMS:
{claims_text}

EVIDENCE:
{ev_text}

OUTPUT:
- Bullet points only.
"""
    raw = await asyncio.to_thread(_responses_create_sync, prompt)
    return raw.strip()


async def skeptic_notes(subclaims: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> str:
    ev_text = "\n".join([f"* {e.get('title','')} — {e.get('snippet','')} ({e.get('url','')})" for e in evidence[:12]])
    claims_text = "\n".join([f"- [{c.get('id','C?')}] {c.get('text','')}" for c in subclaims])
    prompt = f"""ROLE: Skeptic
Build the strongest refutation and highlight weaknesses.

SUBCLAIMS:
{claims_text}

EVIDENCE:
{ev_text}

OUTPUT:
- Bullet points only.
"""
    raw = await asyncio.to_thread(_responses_create_sync, prompt)
    return raw.strip()


async def judge_from_notes(analyst_text: str, skeptic_text: str) -> Dict[str, Any]:
    prompt = f"""ROLE: Judge
Read Analyst and Skeptic notes and issue a final verdict for the entire claim set.

RESPONSE:
Return ONLY valid JSON:
{{
  "label": "TRUE|FAKE|UNVERIFIED",
  "confidence": 0.0,
  "rationale": "..."
}}

Analyst:
{analyst_text}

Skeptic:
{skeptic_text}
"""
    raw = await asyncio.to_thread(_responses_create_sync, prompt)
    # Parse JSON leniently
    try:
        return json.loads(raw)
    except Exception:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e+1])
            except Exception:
                pass
    return {"label": "UNVERIFIED", "confidence": 0.55, "rationale": "Debate JSON parse failed."}



# -------- New: Evidence Reasoning with Triangulation & Fusion --------

async def _per_source_entail(subclaim: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the model for entailment on a single source → {label, confidence, why}.
    """
    prompt = f"""Decide whether the SOURCE snippet entails the SUBCLAIM.

SUBCLAIM:
"{subclaim}"

SOURCE:
Title: {item.get('title')}
Snippet: {item.get('snippet')}
URL: {item.get('url')}
Meta: credibility={item.get('credibility')} freshness={item.get('freshness')}

Return ONLY valid JSON:
{{
  "label": "SUPPORTS|REFUTES|NEUTRAL",
  "confidence": 0.0,
  "rationale": "..."
}}
"""
    raw = await asyncio.to_thread(_responses_create_sync, prompt)
    try:
        data = json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end+1]) if (start != -1 and end != -1 and end > start) else {}
    label = str(data.get("label", "NEUTRAL")).upper()
    conf = float(data.get("confidence", 0.5))
    why = data.get("rationale", "")
    if label not in ("SUPPORTS", "REFUTES", "NEUTRAL"):
        label = "NEUTRAL"
    return {"label": label, "confidence": conf, "why": why}


async def evaluate_evidence(subclaim: str, evidence_ranked: List[Dict[str, Any]], visual_notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Per-subclaim evaluation across sources with triangulation + (optional) visual fusion.
    Returns:
      {
        "final": {"label": "TRUE|FAKE|UNVERIFIED", "confidence": float, "rationale": str},
        "votes": [... per-source entailments ...],
        "fusion_notes": "...",
        "rule": "explanation of how decision was made"
      }
    """
    # Per-source entailments (limit to top 8 to control latency)
    tasks = []
    for it in evidence_ranked[:8]:
        tasks.append(_per_source_entail(subclaim, it))

    votes = []
    for fut in asyncio.as_completed(tasks):
        try:
            res = await fut
            votes.append(res)
        except Exception:
            votes.append({"label": "NEUTRAL", "confidence": 0.5, "why": "error"})

    # Triangulation rule:
    # - If >=2 SUPPORTS with avg confidence >= 0.65 => TRUE
    # - Else if >=2 REFUTES with avg confidence >= 0.65 => FAKE
    # - Else UNVERIFIED
    sup = [v for v in votes if v["label"] == "SUPPORTS"]
    ref = [v for v in votes if v["label"] == "REFUTES"]

    def avg(xs): return sum(xs) / max(1, len(xs))

    fusion_extra = ""
    if visual_notes:
        fusion_extra = "Visual analysis notes considered: " + " | ".join(visual_notes[:3])

    label = "UNVERIFIED"
    conf = 0.55
    rule = "Needs ≥2 independent consistent signals."
    if len(sup) >= 2 and avg([x["confidence"] for x in sup]) >= 0.65:
        label, conf, rule = "TRUE", min(0.95, 0.6 + 0.2 * len(sup) + 0.2 * avg([x["confidence"] for x in sup])), "Triangulation: multiple sources SUPPORT."
    elif len(ref) >= 2 and avg([x["confidence"] for x in ref]) >= 0.65:
        label, conf, rule = "FAKE", min(0.95, 0.6 + 0.2 * len(ref) + 0.2 * avg([x["confidence"] for x in ref])), "Triangulation: multiple sources REFUTE."
    else:
        label, conf, rule = "UNVERIFIED", 0.55, "Signals insufficient or mixed."

    rationale = f"{rule} {fusion_extra}".strip()
    return {
        "final": {"label": label, "confidence": round(conf, 3), "rationale": rationale},
        "votes": votes,
        "fusion_notes": fusion_extra,
        "rule": rule
    }
