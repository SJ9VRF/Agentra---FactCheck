# Agentra Team – CertiFy  
**Multi-Modal FactChecker & Information Verifier**  

---

## 📌 Overview  
In today’s **generative AI era**—where deepfakes, synthetic media, and AI-generated misinformation can spread globally in **minutes**—truth verification must match the **speed, scale, and complexity of the internet**.  

**CertiFy** is a **real-time, multi-modal fact-checking engine** powered by **GPT-5’s vision-language reasoning**. It ingests any combination of:  

- 📝 **Text**  
- 🖼 **Images**  
- 🎥 **Video frames**  
- 🎙 **Audio**  
- 🔗 **Links**  

…and automatically:  

1. **Extracts factual claims**  
2. **Retrieves high-credibility evidence** using **hybrid search** (web APIs, news APIs, domain trust scoring, deduplication)  
3. **Runs an adversarial, citation-driven debate** between AI agents  
4. **Issues a transparent verdict** — *True*, *Fake*, or *Unverified* — **with calibrated confidence**  

---

## 🧠 How It Works  

| **Agent**        | **Role** |
|------------------|----------|
| **Analyst Agent** | Builds the strongest *pro* case for each claim |
| **Skeptic Agent** | Finds contradictions, missing context, and counter-evidence |
| **Debating Agent**| Moderates a turn-based exchange with rebuttals, cross-examinations, and reliability scoring |
| **Judge Agent**   | Delivers the final verdict with confidence scores |

---

## 🔍   
- **Multi-modal parsing** → Understands text, images, and video **simultaneously**  
- **Advanced reasoning** → Weighs conflicting evidence in structured debates  
- **Long-context handling** → Processes transcripts, large evidence sets, and multi-claim documents  
- **Natural language explanation** → Generates verdicts that are **clear, trustworthy, and explainable**  

---

## 🎯 Output Example  
The system generates an **interactive Verdict Card** containing:  

- ✅ **Final verdict** (*True / Fake / Unverified*) with confidence  
- 🔗 **Clickable citations**  
- 💬 **Debate highlights**  
- 🗺 **Visual reasoning map**  

This provides **newsrooms, social platforms, NGOs, and educators** with **instant, explainable verification** — something no static database or single-pass model can match.  

---

## 🌍 Market & Use Cases  
Real-time, explainable fact-checking is **no longer optional** — it’s **foundational infrastructure** for any **generative AI ecosystem**.  

**Potential Integrations:**  
- Social media platforms  
- News verification tools  
- API trust layers for enterprise AI  
- Government & NGO misinformation monitoring  
- Educational fact-checking tools  

**Market Potential:**  
A fact-checker of this caliber **unlocks a multi-billion-dollar market**, as platforms, enterprises, and governments seek **trusted, real-time safeguards** against AI-driven misinformation.  

---

## Features implemented:
- Inputs: text, (image, audio, video, or URL (auto-extract text via readability): comming soon ...).
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
