import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from terminal_flow import TerminalSession, parse_amount


@pytest.mark.parametrize(("text", "expected", "currency"), [
    ("12k", 12000, None), ("12K", 12000, None), ("1.5m", 1500000, None),
    ("12,000", 12000, None), ("12 000", 12000, None), ("1.2 lakh", 120000, None),
    ("2 crore", 20000000, None), ("₹500", 500, "INR"), ("$ 500", 500, "USD"),
    ("€500 EUR", 500, "EUR"), ("R 500", 500, "ZAR"), ("500 IDR", 500, "IDR"),
    ("500 USD", 500, "USD"), ("0 INR", None, "INR"), ("no amount", None, None),
])
def test_parse_amount(text, expected, currency):
    amount, actual_currency = parse_amount(text)
    assert amount == expected
    assert actual_currency == currency


def start(text="Can I afford a 12k laptop?"):
    session = TerminalSession("user_05", "2025-11-04")
    blocks = session.handle(text)
    return session, blocks


def test_intake_defaults_complete_report_and_partial_installment_paths():
    session, blocks = start()
    assert "When do you need" in blocks[0]["text"]
    for _ in range(3): blocks = session.handle("")
    assert blocks[0]["kind"] == "report"
    assert session.report
    session, _ = start("Can I pay half now for a 12k laptop?")
    assert "When do you need" in session.last_blocks[-1]["text"]
    session.handle("")
    blocks = session.handle("1")
    assert session.form["allow_partial"]
    assert blocks[0]["kind"] == "report"
    session, _ = start("Laptop 12k in 3 installments")
    assert session.form["allow_installments"] and session.form["installment_count"] == 3
    session, _ = start("Laptop 12k no deadline")
    session.handle("1")
    session.handle("2")
    blocks = session.handle("")
    assert blocks[0]["kind"] == "report"


def test_skipped_questions_and_invalid_answers():
    session, blocks = start("Laptop 12k, no deadline")
    assert "partial payment" in blocks[0]["text"]
    session, _ = start()
    session.handle("bogus")
    assert session.phase == "intake"
    session.handle("still bogus")
    assert session.question_index == 1
    session, _ = start()
    blocks = session.handle("/skip")
    assert blocks[0]["kind"] == "report"


def test_report_amounts_are_engine_backed():
    session, _ = start("Laptop 12k no deadline pay half now just advise")
    report = session.report
    assert report
    evidence_text = repr(report)
    for option in report["ranked_options"]:
        for amount in (option["total_cost"], option["lowest_projected_balance"]):
            assert str(amount) in evidence_text
