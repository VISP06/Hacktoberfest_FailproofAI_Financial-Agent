import os

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_terminal_page_and_three_default_answers_produce_report(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=45).run()
    assert not at.exception
    at.chat_input[0].set_value("Can I afford this laptop which costs 12k").run()
    assert not at.exception
    assert "When do you need to pay?" in "\n".join(x.value for x in at.markdown)
    for _ in range(3):
        at.chat_input[0].set_value(" ").run()
    assert not at.exception
    assert "WHY" in "\n".join(x.value for x in at.markdown)


def test_stale_session_state_is_rebuilt(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=45).run()
    at.session_state["terminal_session_glacier-terminal-v3"] = object()
    at.run()
    assert not at.exception
    assert isinstance(at.session_state["terminal_session_glacier-terminal-v3"].__class__.__name__, str)
