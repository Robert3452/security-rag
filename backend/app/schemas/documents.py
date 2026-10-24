from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["es", "en"]


class DocumentMeta(BaseModel):
    source_id: str = Field(pattern="^[a-z0-9]+(-[a-z0-9]+)*$", max_length=100)
    title: str = Field(min_length=1, max_length=300)
    language: Language
    url: str | None = None


class IngestResult(BaseModel):
    source_id: str
    title: str
    pages: int
    chunks: int
    ocr_pages: int
