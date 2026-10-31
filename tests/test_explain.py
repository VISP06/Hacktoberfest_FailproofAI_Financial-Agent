import re
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "code"))
from explain import build_explanation
from explain_llm import polish_explanations


class ExplanationEvidenceTest(unittest.TestCase):
    def test_every_number_in_explanation_is_present_in_evidence(self):
        balances = {"2026-01-01": 100.0, "2026-01-02": 45.5}
        breakdown = {
            "initial_balance": 110.0,
            "flows_by_day": {
                "2026-01-02": [
                    {"source": "scheduled_event", "event_id": "event-42", "amount": -30.0,
                     "description": "Rent", "date": "2026-01-02"},
                    {"source": "income", "event_id": "income-7", "amount": 15.5,
                     "description": "Pay", "date": "2026-01-02"},
                ]
            },
            "income_decisions": [{"series_key": "salary:1", "decision": "counted",
                                  "reason": "fallback recurring-series rule"}],
            "pending_ignored": ["pending-9"],
        }
        explanation, evidence = build_explanation(
            balances=balances, breakdown=breakdown, minimum_balance=20.0,
            chosen_option={"method": "wait", "completion_date": "2026-01-02", "cost": 40.0,
                           "spending_changes": "none"}, desired_date="2026-01-03",
            retrieved_messages=[{"message_id": "message-18", "message_text": "private message text"}],
        )
        self.assertEqual(evidence["retrieved_message_ids"]["pending_item"], ["message-18"])
        self.assertNotIn("private message text", explanation)
        evidence_text = repr(evidence)
        for number in re.findall(r"\d+(?:\.\d+)?", explanation):
            self.assertIn(number, evidence_text)

    def test_income_occurrences_are_aggregated_and_unrecommended_is_explained(self):
        breakdown = {"initial_balance": 10, "flows_by_day": {
            "2026-01-01": [{"source": "income", "series_key": "salary:15", "event_id": "e1", "amount": 14740, "date": "2026-01-01"}],
            "2026-01-02": [{"source": "income", "series_key": "salary:15", "event_id": "e2", "amount": 15, "date": "2026-01-02"}],
        }, "pending_ignored": []}
        text, evidence = build_explanation(balances={"2026-01-01": 5}, breakdown=breakdown,
            minimum_balance=10, chosen_option={"method": "not_recommended", "completion_date": "none", "cost": 0,
            "spending_changes": "none"}, desired_date="2026-01-03")
        self.assertIn("salary:15:14755 per month, 2 payments in the window", text)
        self.assertIn("falls below the minimum", text)
        self.assertNotIn("none", text.lower())

    def test_polish_accepts_valid_and_rejects_invented_number(self):
        evidence = {"amount": 42, "event_ids": ["event-1"]}
        deterministic = "Original explanation. Keep this safe."
        class Client:
            def __init__(self, text): self.text = text
            def __call__(self, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{"explanations":[' + __import__("json").dumps(self.text) + ']}'))])
        def row(): return {"decision_explanation": deterministic, "explanation_evidence": evidence}
        valid = row()
        polish_explanations([valid], client=Client("The amount is 42. Event event-1 supports this plan."), cache_path="/tmp/opencode/explain-test-cache.json")
        self.assertNotEqual(valid["decision_explanation"], deterministic)
        invalid = row()
        polish_explanations([invalid], client=Client("The amount is 43. Event event-1 supports this plan."), cache_path="/tmp/opencode/explain-test-cache-invalid.json")
        self.assertEqual(invalid["decision_explanation"], deterministic)


if __name__ == "__main__":
    unittest.main()
