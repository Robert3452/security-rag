from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Language = Literal["es", "en"]
Kind = Literal["pdf", "html"]


class DocumentMeta(BaseModel):
    source_id: str = Field(pattern="^[a-z0-9]+(-[a-z0-9]+)*$", max_length=100)
    title: str = Field(min_length=1, max_length=300)
    language: Language
    url: str | None = None
    kind: Kind = "pdf"


class IngestResult(BaseModel):
    source_id: str
    title: str
    pages: int
    chunks: int
    ocr_pages: int


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=50)
    language: Language | None = None


class SearchHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source_id: str
    title: str
    url: str | None
    page: int
    content: str
    score: float


class AskRequest(SearchRequest):
    pass


class AskSource(BaseModel):
    ref: int
    source_id: str
    title: str
    page: int
    url: str | None
    score: float
    kind: Kind


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]
