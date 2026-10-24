import math

import pytest

from app.embeddings.e5 import E5Embedder


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def embedder() -> E5Embedder:
    return E5Embedder("intfloat/multilingual-e5-small")


@pytest.mark.slow
def test_dimension_and_normalization(embedder: E5Embedder) -> None:
    v = embedder.embed_query("¿Qué EPP se exige para trabajos con tensión?")
    assert len(v) == 384
    assert math.isclose(dot(v, v), 1.0, rel_tol=1e-3)


@pytest.mark.slow
def test_cross_lingual_retrieval(embedder: E5Embedder) -> None:
    query = embedder.embed_query("¿Qué es una frontera de arco eléctrico?")
    relevant, unrelated = embedder.embed_passages(
        [
            "The arc flash boundary is the distance at which a worker could receive "
            "a second-degree burn.",
            "Employers must keep records of work-related injuries and illnesses.",
        ]
    )
    assert dot(query, relevant) > dot(query, unrelated)
