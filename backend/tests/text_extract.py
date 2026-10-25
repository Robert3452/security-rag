from app.ingest.extract import clean_text, needs_ocr


def test_clean_text_joins_hyphenated_words() -> None:
    assert clean_text("seguri-\ndad eléctrica") == "seguridad eléctrica"


def test_clean_text_normalizes_ligatures() -> None:
    assert clean_text("eﬁciencia") == "eficiencia"


def test_clean_text_collapses_whitespace_keeps_paragraphs() -> None:
    assert clean_text("  Artículo   1  \n\n\n\nTexto ") == "Artículo 1\n\nTexto"


def test_needs_ocr() -> None:
    assert needs_ocr("12")
    assert not needs_ocr("a" * 100)


def test_clean_text_repairs_mojibake() -> None:
    assert clean_text("sentido de la lÌnea y la secciÛn") == "sentido de la línea y la sección"


def test_clean_text_converts_symbol_bullets() -> None:
    assert clean_text("\x92 Condiciones de trabajo") == "• Condiciones de trabajo"


def test_clean_text_removes_heading_icons() -> None:
    assert clean_text("` Derechos de los trabajadores") == "Derechos de los trabajadores"
