from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont
import os, io
from .config import APP_BRAND

def make_share_card(verdict: str, confidence: float, title: str, out_path: str) -> str:
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), (255,255,255))
    d = ImageDraw.Draw(img)
    # simple layout
    d.rectangle([(0,0),(W, 120)], fill=(240,240,240))
    d.text((40, 40), APP_BRAND, fill=(0,0,0))
    d.text((40, 180), f"Verdict: {verdict}  (conf={confidence:.2f})", fill=(0,0,0))
    d.text((40, 260), title[:90], fill=(10,10,10))
    img.save(out_path, "PNG")
    return out_path

def make_pdf_report(payload: Dict[str, Any], out_path: str, heatmap_path: Optional[str] = None):
    c = canvas.Canvas(out_path, pagesize=A4)
    W, H = A4
    y = H - 2*cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, f"{APP_BRAND}: Fact Check Report")
    y -= 1.2*cm
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, y, f"Verdict: {payload.get('verdict')}")
    y -= 0.8*cm
    # subclaims
    for sc in payload.get("subclaim_results",[])[:6]:
        c.drawString(2*cm, y, f"[{sc.get('id')}] {sc.get('text')} → {sc.get('label')} ({sc.get('confidence')})")
        y -= 0.6*cm
        if y < 4*cm:
            c.showPage(); y = H - 2*cm
    # evidence
    y -= 0.4*cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2*cm, y, "Top Evidence:")
    y -= 0.8*cm
    c.setFont("Helvetica", 11)
    for ev in payload.get("evidence",[])[:10]:
        c.drawString(2*cm, y, f"- {ev.get('title','')}")
        y -= 0.5*cm
        if y < 4*cm:
            c.showPage(); y = H - 2*cm
    # heatmap
    if heatmap_path and os.path.exists(heatmap_path):
        c.showPage()
        c.drawString(2*cm, H-2*cm, "Image ELA Heatmap")
        try:
            c.drawImage(ImageReader(heatmap_path), 2*cm, 4*cm, width=W-4*cm, height=H-8*cm, preserveAspectRatio=True, anchor='c')
        except Exception:
            pass
    c.showPage()
    c.save()
    return out_path
