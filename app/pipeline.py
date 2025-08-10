import logging, asyncio, os, tempfile, time
from typing import Dict, Any, List, Optional



from .ocr import ocr_image, ela_heatmap
from .utils import clean_text
from .fetch import fetch_url_text
from .video import extract_keyframes
from .report import make_pdf_report, make_share_card
from .entities import temporal_checks


from .reasoning import (
    extract_claims_and_queries,
    evaluate_evidence,
    judge_entailment,
    analyst_notes,
    skeptic_notes,
    judge_from_notes,
)


from .retrieval import EvidenceRetriever

log = logging.getLogger("pipeline")

# -------- Low-RPM knobs (env) --------
LOW_RPM_MODE = os.getenv("OPENAI_LOW_RPM", "1") == "1"     # default ON for safety
OPENAI_RPM = max(1, int(os.getenv("OPENAI_RPM", "3")))     # default 3 rpm
DEBATE_ON = os.getenv("OPENAI_USE_DEBATE", "0") == "1"     # default OFF in low-rpm
MAX_SUBCLAIMS = max(1, int(os.getenv("OPENAI_MAX_SUBCLAIMS", "1")))  # verify only 1 subclaim in low-rpm
# spacing between requests (seconds): +1s buffer to be safe with clock skew
OPENAI_INTERVAL = int(60 / OPENAI_RPM) + 1                 # e.g., 21s for 3 rpm


class FactChecker:
    def __init__(self):
        self.retriever = EvidenceRetriever()
        self._last_llm_ts = 0.0

    # -------- simple rate limiter + 429 retry helper for GPT calls --------
    async def _await_slot(self):
        """Wait so we respect RPM when LOW_RPM_MODE is enabled."""
        if not LOW_RPM_MODE:
            return
        now = time.time()
        wait = (self._last_llm_ts + OPENAI_INTERVAL) - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_llm_ts = time.time()

    async def _with_retry(self, coro_fn, *args, **kwargs):
        """
        Call an async GPT wrapper (extract_claims_and_queries / judge_entailment / adversarial_debate)
        under rate limit and retry ONCE on 429 with a full interval sleep.
        """
        await self._await_slot()
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "Rate limit" in msg or "429" in msg:
                log.warning("Hit OpenAI rate limit. Sleeping %ss and retrying once...", OPENAI_INTERVAL)
                await asyncio.sleep(OPENAI_INTERVAL + 1)
                self._last_llm_ts = time.time()
                return await coro_fn(*args, **kwargs)
            raise

    async def run(self, text: Optional[str] = None, image_path: Optional[str] = None,
                  url: Optional[str] = None, audio_text: Optional[str] = None,
                  video_path: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.time()
        timings: Dict[str, int] = {}

        # 1) Ingest
        t_ing = time.time()
        raw_text = clean_text(text or "")
        source = "text"
        if not raw_text and url:
            source = "url"
            try:
                raw_text = fetch_url_text(url)
            except Exception as e:
                log.warning("URL fetch failed: %s", e)
        if not raw_text and audio_text:
            source = "audio"
            raw_text = clean_text(audio_text)
        if not raw_text and image_path:
            source = "image"
            ocr_txt = ocr_image(image_path)
            raw_text = clean_text(ocr_txt or "")
        if not raw_text:
            raise ValueError("No usable text found. Provide text/image/url/audio.")
        timings["ingest_ms"] = int((time.time() - t_ing) * 1000)
        log.info("[Input] %s", raw_text[:400])

        # Optional: keyframes if video provided
        t_vid = time.time()
        keyframes: List[str] = []
        visual_notes: List[str] = []
        if video_path:
            out_dir = tempfile.mkdtemp(prefix="keyframes_")
            keyframes = extract_keyframes(video_path, out_dir, max_frames=5)
            if keyframes:
                visual_notes.append(f"{len(keyframes)} keyframes extracted")
        timings["video_ms"] = int((time.time() - t_vid) * 1000)

        # Optional: ELA heatmap for image
        t_img = time.time()
        heatmap_path = None
        if image_path:
            heatmap_path = os.path.join(tempfile.gettempdir(), f"ela_{os.path.basename(image_path)}.png")
            ela_heatmap(image_path, heatmap_path)
            visual_notes.append("Image ELA heatmap generated")
        timings["image_ms"] = int((time.time() - t_img) * 1000)

        # 2) Plan subclaims and queries  (GPT-5)  [1 GPT call]
        t_plan = time.time()
        plan = await self._with_retry(extract_claims_and_queries, raw_text)
        subclaims = plan.get("subclaims", []) or [{"id": "C1", "text": raw_text[:280]}]
        queries = plan.get("queries", []) or [raw_text[:120]]
        timings["plan_ms"] = int((time.time() - t_plan) * 1000)
        log.info("[Plan] %d subclaims, %d queries", len(subclaims), len(queries))

        # 3) Retrieval with ranking (Brave) — unaffected by OpenAI RPM
        t_ret = time.time()
        # keep it lighter to avoid Brave 429s
        evidence_ranked, retrieval_trace = self.retriever.retrieve(queries, per_query=4, top_k=10)
        timings["retrieve_ms"] = int((time.time() - t_ret) * 1000)
        log.info("[Evidence] %d ranked items", len(evidence_ranked))

        # 3.5) Temporal/entity checks
        t_tmp = time.time()
        temp_checks = temporal_checks(raw_text, evidence_ranked)
        timings["temporal_ms"] = int((time.time() - t_tmp) * 1000)

        # 4) Evidence Reasoning
        t_ent = time.time()
        sub_results: List[Dict[str, Any]] = []
        reasoning_trace: List[Dict[str, Any]] = []

        if LOW_RPM_MODE:
            # Low-RPM path: use ONE GPT call per (limited) subclaim with judge_entailment
            limited_subclaims = subclaims[:MAX_SUBCLAIMS]
            for sc in limited_subclaims:
                sc_label, sc_conf, sc_why = await self._with_retry(judge_entailment, sc.get("text", ""), evidence_ranked)
                sub_results.append({
                    "id": sc.get("id"),
                    "text": sc.get("text"),
                    "label": sc_label,
                    "confidence": round(sc_conf, 3),
                    "why": sc_why
                })
                reasoning_trace.append({
                    "subclaim_id": sc.get("id"),
                    "votes": [{"label": sc_label, "confidence": sc_conf, "why": sc_why, "mode": "low-rpm-single"}],
                    "fusion_notes": "Low-RPM mode: skipped per-source entailment.",
                    "rule": "Single-call judge_entailment due to RPM limits.",
                    "final": {"label": sc_label, "confidence": sc_conf, "rationale": sc_why}
                })
            # If planner produced more subclaims than we verified, record that
            if len(subclaims) > len(limited_subclaims):
                reasoning_trace.append({
                    "note": f"Low-RPM mode verified only the first {len(limited_subclaims)} of {len(subclaims)} subclaims."
                })
        else:
            # Normal path: triangulation + fusion (calls inside evaluate_evidence)
            for sc in subclaims:
                res = await evaluate_evidence(sc.get("text", ""), evidence_ranked, visual_notes=visual_notes or None)
                sub_results.append({
                    "id": sc.get("id"),
                    "text": sc.get("text"),
                    "label": res["final"]["label"],
                    "confidence": res["final"]["confidence"],
                    "why": res["final"]["rationale"]
                })
                reasoning_trace.append({
                    "subclaim_id": sc.get("id"),
                    "votes": res["votes"],
                    "fusion_notes": res.get("fusion_notes"),
                    "rule": res.get("rule"),
                    "final": res["final"]
                })

        timings["entail_ms"] = int((time.time() - t_ent) * 1000)

        # 5) Adversarial Self-Check (Analyst/Skeptic/Judge)

        # 5) Adversarial Self-Check (sequential to respect RPM)
        t_deb = time.time()
        if True:  # debate always on, sequential + rate-limited
            # Analyst
            analyst_text = await self._with_retry(analyst_notes, subclaims, evidence_ranked)
            # space calls under LOW_RPM
            if LOW_RPM_MODE:
                await asyncio.sleep(OPENAI_INTERVAL)

            # Skeptic
            skeptic_text = await self._with_retry(skeptic_notes, subclaims, evidence_ranked)
            if LOW_RPM_MODE:
                await asyncio.sleep(OPENAI_INTERVAL)

            # Judge
            judge_json = await self._with_retry(judge_from_notes, analyst_text, skeptic_text)
            debate = {"analyst": analyst_text, "skeptic": skeptic_text, "judge": judge_json}
        else:
            # (kept for reference; not used now)
            debate = {"analyst": "", "skeptic": "", "judge": {"label": "UNVERIFIED", "confidence": 0.55, "rationale": "Disabled"}}
        timings["debate_ms"] = int((time.time() - t_deb) * 1000)



        # Final decision (combine triangulation result & judge)
        if sub_results and all(r["label"] == "TRUE" for r in sub_results):
            final_label = "TRUE"
            final_conf = min(0.95, sum(r["confidence"] for r in sub_results) / len(sub_results))
        elif sub_results and any(r["label"] == "FAKE" and r["confidence"] >= 0.7 for r in sub_results):
            final_label = "FAKE"
            final_conf = max(r["confidence"] for r in sub_results if r["label"] == "FAKE")
        else:
            final_label = debate["judge"].get("label", "UNVERIFIED")
            final_conf = float(debate["judge"].get("confidence", 0.55))


        # Final decision (combine subclaim signals & Judge) with partial-eval guard
        judge_label = debate["judge"].get("label", "UNVERIFIED")
        judge_conf  = float(debate["judge"].get("confidence", 0.55))

        # If we verified fewer subclaims than the planner produced (low-RPM mode),
        # DO NOT declare TRUE based on the partial set; defer to the Judge.
        partial_eval = len(sub_results) < len(subclaims)

        if not partial_eval and sub_results and all((r["label"] or "").upper() == "TRUE" for r in sub_results):
            final_label = "TRUE"
            final_conf = min(0.95, sum(r["confidence"] for r in sub_results) / len(sub_results))
        elif sub_results and any((r["label"] or "").upper() == "FAKE" and r["confidence"] >= 0.7 for r in sub_results):
            final_label = "FAKE"
            final_conf = max(r["confidence"] for r in sub_results if (r["label"] or "").upper() == "FAKE")
        else:
            # Default to the Judge's holistic view
            final_label, final_conf = judge_label, judge_conf




        # Adjust for strong temporal mismatch (e.g., wrong year)
        suggested_corrections: List[Dict[str, Any]] = []
        for c in temp_checks:
            if c.get("field") == "year" and c.get("status") == "mismatch":
                suggested_corrections.append({
                    "type": "year",
                    "from": c.get("claim"),
                    "to": c.get("evidence_consensus"),
                    "proposed_claim": c.get("suggested_claim"),
                    "confidence": c.get("confidence"),
                    "sources": c.get("supporting_sources", [])
                })
                if c.get("confidence", 0.0) >= 0.7 and final_label == "TRUE":
                    final_label, final_conf = "UNVERIFIED", 0.55

        # 6) Reporting assets
        t_rep = time.time()
        share_card = os.path.join(tempfile.gettempdir(), "share_card.png")
        make_share_card(final_label, final_conf, subclaims[0].get("text", "Claim"), share_card)
        pdf_path = os.path.join(tempfile.gettempdir(), "factcheck_report.pdf")
        make_pdf_report({
            "verdict": final_label,
            "confidence": final_conf,
            "subclaim_results": sub_results,
            "evidence": evidence_ranked
        }, pdf_path, heatmap_path=heatmap_path)
        timings["report_ms"] = int((time.time() - t_rep) * 1000)

        # Meta + traces
        evidence_domains: Dict[str, int] = {}
        for ev in evidence_ranked:
            host = ev.get("host")
            if host:
                evidence_domains[host] = evidence_domains.get(host, 0) + 1

        # Model calls (approx):
        if LOW_RPM_MODE:
            # 1 planner + MAX_SUBCLAIMS judge + (optional) 1 debate-bundle
            model_calls = 1 + min(MAX_SUBCLAIMS, len(subclaims)) + (1 if DEBATE_ON else 0)
        else:
            # 1 planner + per-source entails (inside evaluate_evidence) + debate (3)
            model_calls = 1 + len(subclaims) * (1 + min(8, len(evidence_ranked))) + 2

        timings["total_ms"] = int((time.time() - t0) * 1000)




        out: Dict[str, Any] = {
            "verdict": final_label,
            "confidence": round(final_conf, 3),
            "subclaim_results": sub_results,
            "evidence": evidence_ranked,
            "debate": debate,
            "keyframes": keyframes,
            "heatmap_path": heatmap_path,
            "temporal_checks": temp_checks,
            "suggested_corrections": suggested_corrections,

            # Traces (Steps 4–6):
            "retrieval_trace": retrieval_trace,
            "reasoning_trace": reasoning_trace,
            "adversarial_trace": debate,

            # Meta for process details (already used by your UI)
            "plan_raw": plan,
            "queries_used": retrieval_trace.get("queries", []),
            "meta": {
                "source": source,
                "model": "gpt-5",
                "model_calls": model_calls,
                "timings_ms": timings,
                "evidence_domains": evidence_domains,
                "subclaims_count": len(subclaims),
                "evidence_count": len(evidence_ranked),
                "low_rpm_mode": LOW_RPM_MODE,
                "rpm_interval_sec": OPENAI_INTERVAL,
                "debate_on": (LOW_RPM_MODE and DEBATE_ON) or (not LOW_RPM_MODE)
            },

            "share_card": share_card,
            "pdf_report": pdf_path
        }

        if LOW_RPM_MODE:
            out["reasoning_trace"].append({"note": "LOW_RPM_MODE enabled: sequential GPT calls with spacing to respect RPM."})

        if partial_eval:
            out.setdefault("meta", {}).update({
                "consistency_note": "Partial subclaim evaluation in low-RPM mode — overall verdict taken from Judge."
            })

        log.info("[Output] verdict=%s conf=%.2f", final_label, final_conf)
        return out
