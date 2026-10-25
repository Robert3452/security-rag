"""Extracción del contenido principal de páginas HTML."""

import trafilatura

from app.ingest.extract import PageText, clean_text


def extract_html(html: str | bytes) -> list[PageText]:
    text = trafilatura.extract(html, include_tables=True, include_comments=False)
    if not text:
        return []
    # trafilatura separa los párrafos con un solo salto de línea; el chunker corta en "\n\n"
    return [PageText(page=1, text=clean_text(text.replace("\n", "\n\n")), ocr=False)]
