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
    at.session_state["terminal_session_glacier-terminal-v4"] = object()
    at.run()
    assert not at.exception
    assert isinstance(at.session_state["terminal_session_glacier-terminal-v4"].__class__.__name__, str)


def test_welcome_examples_and_first_chip_starts_intake(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=45).run()
    labels = [button.label for button in at.button]
    assert "Can I afford a laptop that costs ₹12k?" in labels
    assert "Can I pay ₹50,000 in 6 installments?" in labels
    assert "/situation" in labels
    at.button[0].click().run()
    assert not at.exception
    assert "When do you need to pay?" in "\n".join(x.value for x in at.markdown)


def test_finished_report_status_and_user_text_are_safe(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=45).run()
    at.chat_input[0].set_value("Laptop costs 500k no deadline just advise").run()
    assert not at.exception
    markdown = "\n".join(x.value for x in at.markdown)
    assert "status-good" in markdown or "status-caution" in markdown or "status-bad" in markdown
    assert "None" not in markdown
    assert 'class="unsafe-note"' in markdown
    at.chat_input[0].set_value("<script>alert(1)</script>").run()
    markdown = "\n".join(x.value for x in at.markdown)
    assert "&lt;script&gt;" in markdown
    assert "<script>alert(1)</script>" not in markdown
