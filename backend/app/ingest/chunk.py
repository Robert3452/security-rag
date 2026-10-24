"""Fragmentación de páginas en chunks con solapamiento, respetando párrafos."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.ingest.extract import PageText

MAX_WORDS = 250
OVERLAP_WORDS = 40


@dataclass(frozen=True, slots=True)
class TextChunk:
    page: int
    index: int
    text: str


def _split_long(paragraph: str, size: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= size:
        return [paragraph]
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def split_text(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    if overlap >= max_words:
        raise ValueError("overlap debe ser menor a max_words")

    units = [
        unit
        for paragraph in text.split("\n\n")
        if paragraph.strip()
        for unit in _split_long(paragraph.strip(), max_words - overlap)
    ]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        n = len(unit.split())
        if current and current_words + n > max_words:
            chunks.append("\n\n".join(current))
            # Get the tail with the determined overlap
            tail = " ".join(chunks[-1].split()[-overlap:]) if overlap else ""
            current = [tail] if tail else []
            current_words = len(tail.split())  # overlap length
        current.append(unit)
        current_words += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_pages(
    pages: Iterable[PageText], max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS
) -> Iterator[TextChunk]:
    index = 0
    for page in pages:
        for text in split_text(page.text, max_words, overlap):
            yield TextChunk(page=page.page, index=index, text=text)
            index += 1
