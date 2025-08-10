import os
import logging
import tempfile
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from dotenv import load_dotenv

from .pipeline import FactChecker
from .transcribe import transcribe_audio

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="Agentra Multi-Modal Fact Checker (Full)")

# ----- OPTIONAL: enable CORS if you call this API from a frontend on another origin -----
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<meta charset="utf-8" />
<title>Agentra FactCheck</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  :root { --fg:#0f172a; --muted:#64748b; --bg:#fff; --card:#f8fafc; --accent:#111827; --pill:#eef2f7; }
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}
  .wrap{max-width:1000px;margin:32px auto;padding:0 16px}
  h1{margin:0 0 8px}
  .small{color:var(--muted);font-size:13px}
  .card{background:var(--card);border:1px solid #e5e7eb;border-radius:16px;padding:16px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  input,textarea,button,input[type=file]{font:inherit;padding:10px 12px;border-radius:10px;border:1px solid #e2e8f0;width:100%;box-sizing:border-box;background:#fff}
  textarea{min-height:120px;resize:vertical}
  .btn{background:var(--accent);color:#fff;border:none;cursor:pointer}
  .btn[disabled]{opacity:.6;cursor:not-allowed}
  .flex{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  .badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px}
  .bTRUE{background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0}
  .bFAKE{background:#fef2f2;color:#991b1b;border:1px solid #fecaca}
  .bUNVERIFIED{background:#fffbeb;color:#854d0e;border:1px solid #fde68a}
  .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:var(--pill);font-size:12px}
  .section{margin-top:18px}
  ul.list{list-style:none;padding-left:0;margin:0}
  ul.list li{margin:8px 0}
  details summary{cursor:pointer}
  pre.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap}
  .kv{display:grid;grid-template-columns:160px 1fr;gap:10px}
  .kv div{padding:6px 0;border-bottom:1px dashed #e5e7eb}
</style>

<div class="wrap">
  <h1>Agentra FactCheck</h1>
  <p class="small">Enter a claim or URL and optionally attach media. You’ll see a verdict, confidence, subclaims, evidence, and process details.</p>

  <div class="card">
    <form id="fc-form">
      <div class="row">
        <div>
          <label class="small">Claim text</label>
          <textarea name="text" placeholder="e.g., NASA landed the Perseverance rover on Mars on Feb 18, 2021."></textarea>
        </div>
        <div>
          <label class="small">URL (optional)</label>
          <input type="url" name="url" placeholder="https://example.com/article"/>
          <div class="row" style="margin-top:12px">
            <div>
              <label class="small">Image (optional)</label>
              <input type="file" name="image" accept="image/*"/>
            </div>
            <div>
              <label class="small">Audio (optional)</label>
              <input type="file" name="audio" accept="audio/*"/>
            </div>
          </div>
          <div style="margin-top:12px">
            <label class="small">Video (optional)</label>
            <input type="file" name="video" accept="video/*"/>
          </div>
        </div>
      </div>
      <div class="flex" style="margin-top:12px">
        <button class="btn" id="run" type="submit">Fact Check</button>
        <span id="stage" class="small"></span>
      </div>
    </form>
  </div>

  <div id="result" class="card" style="display:none; margin-top:16px;">
    <div class="flex">
      <h3 style="margin:0">Verdict</h3>
      <span id="v-badge" class="badge">—</span>
      <span id="v-conf" class="pill">conf: —</span>
    </div>
    <div id="v-why" class="small" style="margin-top:6px"></div>

    <div class="section">
      <h4 style="margin:0 0 8px">Entity / Temporal Checks</h4>
      <div id="temporal" class="small">—</div>
      <div id="proposed" class="small" style="margin-top:6px"></div>
    </div>

    <div class="section">
      <h4 style="margin:0 0 8px">Subclaims</h4>
      <div id="subs"></div>
    </div>

    <div class="section">
      <h4 style="margin:0 0 8px">Top Evidence</h4>
      <ul id="evidence" class="list"></ul>
    </div>

    <div id="assets" class="flex" style="margin-top:10px; display:none;">
      <a id="pdf" class="pill" href="#" target="_blank">Download PDF</a>
      <a id="share" class="pill" href="#" target="_blank">Download Share Card</a>
    </div>

    <div id="heat" class="section"></div>

    <details class="section">
      <summary><b>Process details</b> (stages, agents, retrieval & reasoning)</summary>
      <div class="kv" id="meta"></div>

      <h5>Stage Timings</h5>
      <div id="timings"></div>

      <h5>Queries Used</h5>
      <ul id="queries" class="list"></ul>

      <h5>Evidence Ranking (top 10)</h5>
      <div id="ranked"></div>

      <h5>Debate Trace</h5>
      <pre id="analyst" class="mono"></pre>
      <pre id="skeptic" class="mono"></pre>
      <pre id="judge" class="mono"></pre>
    </details>
  </div>

  <p class="small" style="margin-top:18px">
    Docs: <a href="/docs">/docs</a> • Live stages: <a href="/events" target="_blank">/events</a>
  </p>
</div>

<script>
const $ = (id) => document.getElementById(id);
const form = $("fc-form"), run = $("run"), stage = $("stage");
const result = $("result"), vbadge = $("v-badge"), vconf = $("v-conf"), vwhy = $("v-why");
const temporal = $("temporal"), proposed = $("proposed");
const subs = $("subs"), ev = $("evidence");
const assets = $("assets"), pdf = $("pdf"), share = $("share"), heat = $("heat");
const meta = $("meta"), timings = $("timings"), queries = $("queries"), ranked = $("ranked");
const analyst = $("analyst"), skeptic = $("skeptic"), judge = $("judge");

function setBusy(b){ run.disabled = b; run.textContent = b ? "Checking..." : "Fact Check"; }
function setStage(msg){ stage.textContent = msg || ""; }
function connectEvents(){
  try {
    const es = new EventSource("/events");
    es.onmessage = (e) => {
      try { const d = JSON.parse(e.data); setStage(d.stage + " — " + d.message); if (d.stage === "done") es.close(); } catch {}
    };
  } catch {}
}
function clsForLabel(L){
  const k = (L||"UNVERIFIED").toUpperCase();
  return k === "TRUE" ? "badge bTRUE" : k === "FAKE" ? "badge bFAKE" : "badge bUNVERIFIED";
}

form.addEventListener("submit", async (e)=>{
  e.preventDefault();
  setBusy(true); setStage("Starting…"); connectEvents();
  const fd = new FormData(form);
  try {
    const res = await fetch("/factcheck", { method:"POST", body: fd });
    const data = await res.json();
    if(!res.ok || data.error){ setStage("Error"); alert(data.error || "Failed"); setBusy(false); return; }
    render(data); setStage("Done");
  } catch(err){ setStage("Error"); alert(err); }
  finally{ setBusy(false); }
});

function render(data){
  result.style.display = "block";

  // Verdict
  const L = (data.verdict||"UNVERIFIED").toUpperCase();
  vbadge.textContent = L; vbadge.className = clsForLabel(L);
  vconf.textContent = "conf: " + (data.confidence != null ? data.confidence : "—");
  const judgeWhy = data.debate && data.debate.judge && (data.debate.judge.rationale || data.debate.judge.reason || "");
  const firstWhy = data.subclaim_results && data.subclaim_results[0] && data.subclaim_results[0].why;
  vwhy.textContent = judgeWhy || firstWhy || "";

  // Temporal checks
  temporal.innerHTML = "";
  (data.temporal_checks || []).forEach(ch=>{
    if(ch.field==="year" && ch.status==="mismatch"){
      const div = document.createElement("div");
      div.innerHTML = `Year mismatch: claim <b>${ch.claim}</b> vs evidence <b>${ch.evidence_consensus}</b> (confidence ${ch.confidence}).`;
      temporal.appendChild(div);
    }
  });
  if(!temporal.innerHTML) temporal.textContent = "No explicit temporal mismatches detected.";
  proposed.textContent = (data.suggested_corrections && data.suggested_corrections[0] && data.suggested_corrections[0].proposed_claim) ? 
    "Suggested fix: " + data.suggested_corrections[0].proposed_claim : "";

  // Subclaims
  subs.innerHTML = "";
  (data.subclaim_results || []).forEach(sc=>{
    const card = document.createElement("div");
    card.className = "card";
    card.style.padding = "12px";
    card.style.marginBottom = "8px";
    const k = (sc.label||"UNVERIFIED").toUpperCase();
    card.innerHTML = `
      <div class="flex">
        <div class="small" style="opacity:.7">${sc.id||""}</div>
        <span class="${clsForLabel(k)}">${k}</span>
        <span class="pill">conf: ${sc.confidence ?? "—"}</span>
      </div>
      <div style="margin-top:6px">${sc.text||""}</div>
      <div class="small" style="margin-top:6px">${sc.why||""}</div>
    `;
    subs.appendChild(card);
  });

  // Evidence
  ev.innerHTML = "";
  (data.evidence || []).forEach(it=>{
    const li = document.createElement("li");
    const t = it.title || it.url || "source";
    const s = it.snippet || "";
    li.innerHTML = `<a href="${it.url||"#"}" target="_blank">${t}</a><div class="small">${s}</div>`;
    ev.appendChild(li);
  });

  // Assets
  if (data.pdf_report || data.share_card){
    assets.style.display = "flex";
    if (data.pdf_report){ pdf.href = "/download/pdf?path=" + encodeURIComponent(data.pdf_report); }
    if (data.share_card){ share.href = "/download/share?path=" + encodeURIComponent(data.share_card); }
  } else { assets.style.display = "none"; }

  // Heatmap
  heat.innerHTML = "";
  if (data.heatmap_path){
    const im = document.createElement("img");
    im.src = "/download/share?path=" + encodeURIComponent(data.heatmap_path);
    im.style.maxWidth = "100%"; im.style.borderRadius = "10px"; im.alt = "ELA heatmap";
    heat.appendChild(im);
  }

  // Process details
  meta.innerHTML = "";
  const m = data.meta || {};
  const kv = (k,v)=>{ const row = document.createElement("div"); row.innerHTML = `<b>${k}</b>`; const val=document.createElement("div"); val.textContent = v; meta.appendChild(row); meta.appendChild(val); };
  kv("Input source", m.source||"—");
  kv("Model", m.model||"—");
  kv("Model calls", String(m.model_calls||0));
  if (m.low_rpm_mode !== undefined) kv("Low-RPM mode", String(m.low_rpm_mode));
  if (m.rpm_interval_sec !== undefined) kv("RPM interval (sec)", String(m.rpm_interval_sec));
  kv("Subclaims", String(m.subclaims_count||0));
  kv("Evidence items", String(m.evidence_count||0));

  timings.innerHTML = "";
  const tm = m.timings_ms || {};
  if (Object.keys(tm).length){
    const tbl = document.createElement("table");
    tbl.style.borderCollapse="collapse";
    tbl.innerHTML = "<tr><th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:6px 10px'>Stage</th><th style='text-align:left;border-bottom:1px solid #e5e7eb;padding:6px 10px'>Time (ms)</th></tr>";
    Object.entries(tm).forEach(([k,v])=>{
      const tr = document.createElement("tr");
      tr.innerHTML = `<td style='padding:6px 10px;border-bottom:1px solid #f1f5f9'>${k}</td><td style='padding:6px 10px;border-bottom:1px solid #f1f5f9'>${v}</td>`;
      tbl.appendChild(tr);
    });
    timings.appendChild(tbl);
  }

  queries.innerHTML = "";
  (data.queries_used || []).forEach(q=>{
    const li = document.createElement("li"); li.textContent = q; queries.appendChild(li);
  });

  ranked.innerHTML = "";
  (data.retrieval_trace && data.retrieval_trace.ranked ? data.retrieval_trace.ranked : []).forEach((r,i)=>{
    const row = document.createElement("div");
    row.className = "small";
    row.style.padding = "6px 0"; row.style.borderBottom = "1px dashed #e5e7eb";
    row.innerHTML = `<b>${i+1}.</b> <a href="${r.url}" target="_blank">${r.title||r.url}</a>
      <span class="pill">score: ${r.score}</span>
      <span class="pill">cred: ${r.credibility}</span>
      <span class="pill">fresh: ${r.freshness}</span>
      <div style="margin-top:4px;opacity:.85">${r.snippet||""}</div>`;
    ranked.appendChild(row);
  });

  analyst.textContent = (data.debate && data.debate.analyst) ? data.debate.analyst : "";
  skeptic.textContent = (data.debate && data.debate.skeptic) ? data.debate.skeptic : "";
  judge.textContent = (data.debate && data.debate.judge) ? JSON.stringify(data.debate.judge, null, 2) : "";
}
</script>
"""






@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/factcheck")
async def factcheck(
    text: str = Form(default=""),
    url: str = Form(default=""),
    image: UploadFile = File(default=None),
    audio: UploadFile = File(default=None),
    video: UploadFile = File(default=None),
):
    paths = []

    def save_upload(up: Optional[UploadFile]) -> Optional[str]:
        if up is None:
            return None
        p = os.path.join(tempfile.gettempdir(), f"_fc_{up.filename}")
        with open(p, "wb") as f:
            f.write(up.file.read())
        paths.append(p)
        return p

    img_path = save_upload(image)
    aud_path = save_upload(audio)
    vid_path = save_upload(video)

    audio_text: Optional[str] = None
    if aud_path:
        audio_text = transcribe_audio(aud_path)

    fc = FactChecker()
    try:
        result = await fc.run(
            text=text,
            image_path=img_path,
            url=url or None,
            audio_text=audio_text,
            video_path=vid_path,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        for p in paths:
            try: os.remove(p)
            except Exception: pass

@app.get("/download/pdf")
async def download_pdf(path: str):
    return FileResponse(path, filename="factcheck_report.pdf", media_type="application/pdf")

@app.get("/download/share")
async def download_share(path: str):
    return FileResponse(path, filename="share_card.png", media_type="image/png")

@app.get("/events")
async def events():
    async def gen():
        import json
        stages = [
            {"stage": "ingest", "message": "Reading inputs"},
            {"stage": "plan", "message": "Extracting subclaims & queries with GPT-5"},
            {"stage": "retrieve", "message": "Searching Brave for evidence"},
            {"stage": "reason", "message": "Scoring entailment/triangulation"},
            {"stage": "debate", "message": "Analyst vs. Skeptic, Judge verdict"},
            {"stage": "report", "message": "Generating share card & PDF"},
            {"stage": "done", "message": "Complete"},
        ]
        for s in stages:
            yield f"data: {json.dumps(s)}\n\n"
            await asyncio.sleep(0.4)
    return StreamingResponse(gen(), media_type="text/event-stream")
