import os
import sys

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from assistant import Assistant


def _app(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=45).run()


def _ask(at, prompt):
    return at.chat_input[0].set_value(prompt).run()


def test_page_loads_and_situation_tiles_have_values_before_chat(monkeypatch):
    at = _app(monkeypatch)
    assert not at.exception
    page = "\n".join(item.value for item in list(at.markdown) + list(at.caption))
    assert "Your situation" in page
    assert "Available" in page and "Minimum to keep" in page
    assert "<div class=\"stat-number\">" in page
    assert "text-overflow" not in page


def test_chat_amount_gets_advice_without_clarifying(monkeypatch):
    at = _app(monkeypatch)
    _ask(at, "Can I afford a 12,000 laptop?")
    assert not at.exception
    page = "\n".join(item.value for item in list(at.markdown) + list(at.caption))
    assert "Recommended:" in page
    assert "Compared options" in page
    assert "What amount are you considering?" not in page


def test_missing_amount_asks_only_one_amount_question(monkeypatch):
    at = _app(monkeypatch)
    _ask(at, "Can I afford a laptop?")
    assert not at.exception
    question_text = [item.value for item in at.markdown if "What amount are you considering?" in item.value]
    assert question_text == ["What amount are you considering?"]


def test_model_failure_uses_numeric_parser_and_deterministic_advice(monkeypatch):
    for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    class FakeFailingClient:
        def __call__(self, **_kwargs):
            raise ValueError("fake client failure")

    original_init = Assistant.__init__
    monkeypatch.setattr(Assistant, "__init__",
                        lambda self, *args, **kwargs: original_init(
                            self, *args, client=FakeFailingClient(), **kwargs))
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=45).run()
    _ask(at, "Can I afford a 12,000 laptop?")
    assert not at.exception
    page = "\n".join(item.value for item in list(at.markdown) + list(at.caption))
    assert "AI wording unavailable" in page
    assert "Recommended:" in page
    assert "What amount are you considering?" not in page
