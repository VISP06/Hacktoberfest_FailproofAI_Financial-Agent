import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from terminal_flow import TerminalSession, parse_amount
from plan_dates import installment_completion_date


@pytest.mark.parametrize(("text", "expected", "currency"), [
    ("12k", 12000, None), ("12K", 12000, None), ("1.5m", 1500000, None),
    ("12,000", 12000, None), ("12 000", 12000, None), ("1.2 lakh", 120000, None),
    ("2 crore", 20000000, None), ("₹500", 500, "INR"), ("$ 500", 500, "USD"),
    ("€500 EUR", 500, "EUR"), ("R 500", 500, "ZAR"), ("500 IDR", 500, "IDR"),
    ("500 USD", 500, "USD"), ("0 INR", None, "INR"), ("no amount", None, None),
    ("Rs 500", 500, "INR"), ("Rs. 500", 500, "INR"), ("500 rupee", 500, "INR"),
    ("500 rupees", 500, "INR"),
    ("EUR 5,414.20", 5414.2, "EUR"), ("ZAR 15,488", 15488, "ZAR"),
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


def test_report_verdict_controls_steps_and_pending_items_are_reported_once():
    session = TerminalSession("user_07", "2025-11-04")
    session.handle("Laptop costs Rs. 12k no deadline just advise")
    result = session.report
    rendered = session.render_report(result)
    if result["recommended_option"]:
        assert "Confirm the selected plan" in rendered
        assert "Do not make this payment now" not in rendered

    result["recommended_option"] = None
    result["decision"]["affordability_status"] = "not_affordable"
    result["decision"]["amount_safe_to_pay"] = 2500
    result["decision"]["spending_changes_needed"] = "reduce dining out"
    report = session.render_report(result)
    next_steps = report.split("NEXT STEPS\n", 1)[1]
    assert "NOT RECOMMENDED" in report
    assert "Do not make this payment now" in next_steps
    assert "Amount safe to pay today: INR 2,500.00" in next_steps
    assert "Earliest safe date:" in next_steps
    assert "income resumes" in next_steps and "smaller amount" in next_steps
    assert "reduce dining out" in next_steps
    assert report.count("Pending items:") == 1
    assert "Pending items: considered " in report
    assert "; ignored " in report


def test_deadline_filters_installments_and_conflicting_deadline_reasks():
    # End-of-month deadline is earlier than the shortest available 3-payment plan.
    session, blocks = start("Laptop 12k end of this month")
    assert session.form["pay_by_date"] == "2025-11-30"
    assert any("shortest plan ends on 2026-01-03" in block["text"] for block in blocks)
    assert not any("Use installments?" in block["text"] for block in blocks)

    # If an installment count was stated first, a conflicting deadline is rejected
    # and the replacement prompt allows only no deadline or a date after completion.
    session, blocks = start("Laptop 12k in 3 installments")
    assert "When do you need" in blocks[0]["text"]
    blocks = session.handle("2")
    assert any("conflicts with the deadline" in block["text"] for block in blocks)
    prompt = blocks[-1]["text"]
    assert "none" in prompt and "on or after 2026-01-03" in prompt
    blocks = session.handle("1")
    assert session.form["pay_by_date"] is None


def test_installment_options_for_no_deadline_and_shared_completion_date():
    session, blocks = start("Laptop 12k no deadline")
    blocks = session.handle("1")  # no partial payment
    assert "3 months" in blocks[0]["text"]
    assert "6 months" in blocks[0]["text"]
    assert "12 months" in blocks[0]["text"]
    assert installment_completion_date("2025-11-04", 3) == __import__("datetime").date(2026, 1, 3)
    result = session.assistant.advise({"amount": 12000, "currency": "ZAR", "allow_installments": True,
                                       "installment_count": 3, "installment_fee_pct": 0})
    option = next(item for item in result["ranked_options"] if item["type"] == "installments")
    assert max(option["dates"]) == installment_completion_date("2025-11-04", 3).isoformat()


def test_deadline_answer_filters_later_installments_and_why_users_are_evidence_based(monkeypatch):
    session, _ = start("Laptop costs 12k")
    blocks = session.handle("3")  # end of this month
    assert any("No installment option completes by 2025-11-30" in block["text"] for block in blocks)
    assert not any("Use installments?" in block["text"] for block in blocks)

    demo = TerminalSession("user_07", "2025-11-04")
    assert demo.assistant.engine.get_user_home_currency("user_07") == "INR"
    assert demo.handle("Laptop costs Rs. 12k no deadline just advise")[0]["kind"] == "report"
    why = demo.handle("/why")[0]["text"]
    for label in ("Data loaded:", "Income streams/classes:", "Retrieved message IDs:",
                  "Options considered:", "Wording: plain text", "Failproof session: chat-"):
        assert label in why
    users = demo.handle("/users INR")[0]["text"]
    assert "user_07 · INR · INR" in users
    assert len(users.splitlines()) <= 9


def test_missing_direct_currency_rate_has_account_specific_message(monkeypatch):
    session = TerminalSession("user_07", "2025-11-04")
    session.form = {"amount": 10, "currency": "USD", "pay_by_date": None}
    monkeypatch.setattr(session.assistant, "_home_amount", lambda *_args: (None, "INR"))
    message = session._make_report()[0]["text"]
    assert message == ("Your account is in INR and I have no USD to INR rate. Type the amount in "
                       "INR, or switch accounts with /user.")


@pytest.mark.parametrize("followup", ["half now", "pay 40% now", "pay 5k now", "wait a month", "a month later",
                                       "wait 2 weeks", "6 installments", "by 2025-12-01", "actually 20k", "make it 20k"])
def test_report_followups_recompute_and_compare(followup):
    session = TerminalSession("user_07", "2025-11-04")
    session.handle("Laptop 12k no deadline just advise")
    blocks = session.handle(followup)
    assert any(block["text"].startswith("WHAT CHANGED") for block in blocks)
    assert any(block["kind"] == "report" for block in blocks)
    assert session.latest_result is session.report


@pytest.mark.parametrize("line", ["", "???", "banana"])
def test_unrecognized_or_empty_report_followup_never_reprints_report(line):
    session = TerminalSession("user_07", "2025-11-04")
    session.handle("Laptop 12k no deadline just advise")
    blocks = session.handle(line)
    assert not any(block["kind"] == "report" for block in blocks)
    if line:
        assert "I didn't understand that." in blocks[0]["text"]


@pytest.mark.parametrize(("command", "expected"), [("/user", "Usage: /user <id>"), ("/date", "Usage: /user <id>")])
def test_user_date_usage(command, expected):
    session = TerminalSession("user_07", "2025-11-04")
    assert session.handle(command)[0]["text"] == expected


def test_user_resets_to_latest_event_date_and_date_error_has_range():
    session = TerminalSession("user_07", "2025-11-04")
    response = session.handle("/user user_05")[0]["text"]
    assert "Data range:" in response and session.as_of_date in response
    error = session.handle("/date 1900-01-01")[0]["text"]
    assert "Allowed range" in error and "through" in error
