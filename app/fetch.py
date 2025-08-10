import requests
from readability import Document
from lxml import html
from .utils import clean_text

def fetch_url_text(url: str, timeout=15) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"AgentraFactCheck/1.0"})
    r.raise_for_status()
    doc = Document(r.text)
    summary_html = doc.summary()
    tree = html.fromstring(summary_html)
    txt = " ".join(tree.xpath("//text()"))
    return clean_text(txt)
