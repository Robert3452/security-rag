"""Extracción de texto por página, con OCR como fallback para páginas escaneadas"""

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract

MIN_QUALITY = 0.05
OCR_SCALE = 4
OCR_LANGS = {"es": "spa", "en": "eng"}
OCR_DPI = 300
MIN_CHARS = 50
_SPACES = re.compile(r"[ \t]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_BULLET = re.compile(r"[\x80-\x9f]\s*", re.MULTILINE)
_ICON = re.compile(r"^`\s*", re.MULTILINE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_TOKEN = re.compile(r"[^\W\d_]+")
_STOPWORDS = frozenset(
    {
        # español
        "que",
        "los",
        "las",
        "del",
        "por",
        "con",
        "para",
        "una",
        "como",
        "sus",
        "este",
        "esta",
        # inglés
        "the",
        "and",
        "for",
        "with",
        "that",
        "are",
        "from",
        "this",
        "not",
        "you",
    }
)


@dataclass(frozen=True, slots=True)
class PageText:
    page: int
    text: str
    ocr: bool


def text_quality(text: str) -> float:
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if not tokens:
        return 0.0
    return sum(t in _STOPWORDS for t in tokens) / len(tokens)


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.splitlines())
    text = _BULLET.sub("• ", text)
    text = _ICON.sub("", text)
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def needs_ocr(text: str, min_chars: int = MIN_CHARS, min_quality: float = MIN_QUALITY) -> bool:
    return len(text) < min_chars or text_quality(text) < min_quality


def _ocr_page(page: pdfium.PdfPage, lang: str) -> str:
    bitmap = page.render(scale=OCR_SCALE)
    return str(pytesseract.image_to_string(bitmap.to_pil(), lang=lang))


def extract_pages(path: Path | bytes, language: str) -> Iterator[PageText]:
    ocr_lang = OCR_LANGS.get(language, "eng")
    pdf = pdfium.PdfDocument(path)

    try:
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            text = clean_text(textpage.get_text_range())
            textpage.close()

            used_ocr = False
            if needs_ocr(text):
                text = clean_text(_ocr_page(page, ocr_lang))
                used_ocr = True
            page.close()

            if text:
                yield PageText(page=index + 1, text=text, ocr=used_ocr)
    finally:
        pdf.close()
