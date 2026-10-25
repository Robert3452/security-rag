import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.chunks import RetrievedChunk
from app.services import ask as ask_module
from app.services.ask import NOT_FOUND, ask, build_user_prompt, extract_citations
from tests.fake import FakeEmbedder, FakeLLM


def make_chunk(page: int) -> RetrievedChunk:
    return RetrievedChunk(
        source_id="doc",
        title="Doc",
        url=None,
        page=page,
        content=f"texto {page}",
        score=0.8,
        kind="pdf",
    )


CHUNKS = [make_chunk(1), make_chunk(2), make_chunk(3)]


def test_prompt_numbers_fragments_from_one() -> None:
    prompt = build_user_prompt("¿pregunta?", CHUNKS)
    assert '<fragmento id="1"' in prompt
    assert '<fragmento id="3"' in prompt
    assert prompt.endswith("Pregunta: ¿pregunta?")


def test_citations_keep_order_dedupe_and_ignore_invalid() -> None:
    citations = extract_citations("A [3]. B [1][3]. C [9].", CHUNKS)
    assert [c.ref for c in citations] == [3, 1]
    assert citations[0].chunk.page == 3


async def fake_retrieve(*args: object, **kwargs: object) -> list[RetrievedChunk]:
    return CHUNKS


async def test_not_found_returns_no_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ask_module, "retrieve", fake_retrieve)
    llm = FakeLLM(NOT_FOUND)

    result = await ask(AsyncSession(), FakeEmbedder(), llm, "¿algo que no está?")

    assert result.answer == NOT_FOUND
    assert result.citations == []


async def test_answer_with_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ask_module, "retrieve", fake_retrieve)
    llm = FakeLLM("Se exigen guantes dieléctricos [2].")

    result = await ask(AsyncSession(), FakeEmbedder(), llm, "¿Qué EPP?")

    assert [c.ref for c in result.citations] == [2]
    assert llm.calls == 1
