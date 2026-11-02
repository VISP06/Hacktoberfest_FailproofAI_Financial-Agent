import os
import sys

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from assistant import Assistant


def _app(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30).run()


def test_app_loads_without_exception(monkeypatch):
    at = _app(monkeypatch)
    assert not at.exception
    assert any("Buy or Wait?" in markdown.value for markdown in at.markdown)


def test_get_advice_works_without_model(monkeypatch):
    at = _app(monkeypatch)
    at.button[0].click().run()
    assert not at.exception
    assert any("Recommended:" in item.value for item in at.success)


def test_model_failure_keeps_deterministic_advice(monkeypatch):
    at = _app(monkeypatch)
    monkeypatch.setattr(Assistant, "_model_json", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("fake failure")))
    at.toggle[2].set_value(True)
    at.button[0].click().run()
    assert not at.exception
    assert any("AI wording is unavailable" in item.value for item in at.info)
    assert any("Recommended:" in item.value for item in list(at.success) + list(at.warning))
