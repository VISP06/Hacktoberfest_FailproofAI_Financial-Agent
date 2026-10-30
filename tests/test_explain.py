import re
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "code"))
from explain import build_explanation


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


if __name__ == "__main__":
    unittest.main()
