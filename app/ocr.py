from typing import Optional
from PIL import Image, ImageChops, ImageEnhance
import pytesseract
import io, os

def ocr_image(path: str) -> Optional[str]:
    try:
        img = Image.open(path)
        txt = pytesseract.image_to_string(img, lang="eng")
        txt = (txt or "").strip()
        return txt if txt else None
    except Exception:
        return None

def ela_heatmap(in_path: str, out_path: str, quality: int = 90) -> Optional[str]:
    """Simple Error Level Analysis heatmap (not a forensic guarantee)."""
    try:
        original = Image.open(in_path).convert("RGB")
        # recompress
        tmp_jpg = io.BytesIO()
        original.save(tmp_jpg, "JPEG", quality=quality)
        tmp_jpg.seek(0)
        recompressed = Image.open(tmp_jpg)
        diff = ImageChops.difference(original, recompressed)
        # boost differences
        enhancer = ImageEnhance.Brightness(diff)
        heat = enhancer.enhance(8.0)
        heat.save(out_path)
        return out_path
    except Exception:
        return None
