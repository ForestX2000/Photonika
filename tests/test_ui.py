"""Smoke-тест Streamlit-демо: страница строится, синтез запускается, ползунок z работает."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.skipif(not (ROOT / "data" / "testsets" / "mephi_letters").is_dir(),
                    reason="run `python -m app.cli make-testsets` first")
def test_ui_synthesis_flow():
    at = AppTest.from_file(str(ROOT / "app" / "ui.py"), default_timeout=180)
    at.run()
    assert not at.exception
    method =next(s for s in at.sidebar.selectbox if s.label == "Метод синтеза")
    method.set_value("gs")
    next(s for s in at.sidebar.slider if s.label == "Итераций GS").set_value(20)
    next(s for s in at.selectbox if s.label == "Набор").set_value("mephi_letters")
    at.run()
    next(b for b in at.button if b.label == "Синтезировать").click()
    at.run()
    assert not at.exception
    assert any(m.label == "PCC (среднее)" for m in at.metric)
    z = next(s for s in at.slider if s.label == "z, мм")
    z.set_value(26.5)
    at.run()
    assert not at.exception
