import sys
from pathlib import Path

from app.ingest.extract import extract_pages

path, language = Path(sys.argv[1]), sys.argv[2]
for p in extract_pages(path, language):
    flag = "OCR" if p.ocr else "  "
    print(f"p.{p.page:>3} ocr?: {p.ocr} {flag} {len(p.text):>5} chars | {p.text!r}")
