# Agentra Multi‑Modal Fact Checker
Features implemented:
- Inputs: text, image, audio, video, or URL (auto-extract text via readability).
- Preprocess: OCR (Tesseract), audio transcription (OpenAI), video keyframes (OpenCV), initial summarization.
- Claim extraction & query planning (GPT‑5).
- Retrieval: Brave Search only (with whitelist + caching).
- Evidence reasoning: entailment per subclaim + triangulation.
- Adversarial debate (Analyst/Skeptic/Judge) for final verdict.
- Outputs: TRUE/FAKE/UNVERIFIED, confidences, rationale, sources.
- Imaging: ELA heatmap for images.
- Reporting: JSON, share-card PNG, and PDF report.
- Realtime: Server-Sent Events (/events) for live progress updates.
- Async pipeline and limited parallelism.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set OPENAI_API_KEY and BRAVE_API_KEY
uvicorn app.main:app --reload
```

### Minimal tests
Text:
```bash
curl -X POST http://127.0.0.1:8000/factcheck   -F 'text=NASA landed the Perseverance rover on Mars on Feb 18, 2021.'
```

Image (OCR + ELA heatmap will be produced, saved to /tmp):
```bash
curl -X POST http://127.0.0.1:8000/factcheck   -F "image=@/path/to/screenshot.png"
```

URL:
```bash
curl -X POST http://127.0.0.1:8000/factcheck   -F "url=https://en.wikipedia.org/wiki/Perseverance_(rover)"
```

Audio (transcription via OpenAI):
```bash
curl -X POST http://127.0.0.1:8000/factcheck   -F "audio=@/path/to/clip.m4a"
```
