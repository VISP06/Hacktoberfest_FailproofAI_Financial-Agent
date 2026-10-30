import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "code"))
import retrieval


def test_retrieval_is_user_scoped_and_strictly_before_cutoff(tmp_path, monkeypatch):
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame([
        {"message_id": "old", "user_id": "u1", "request_id": "secret_request_id",
         "sent_at": "2025-02-01T12:00:00Z", "source_type": "employer", "message_text": "salary payout confirmed"},
        {"message_id": "same_day", "user_id": "u1", "request_id": "r2",
         "sent_at": "2025-02-02T00:00:00Z", "source_type": "employer", "message_text": "salary payout"},
        {"message_id": "other_user", "user_id": "u2", "request_id": "r1",
         "sent_at": "2025-02-01T12:00:00Z", "source_type": "employer", "message_text": "salary payout confirmed"},
    ]).to_csv(messages_path, index=False)
    monkeypatch.setattr(retrieval, "MESSAGES_PATH", str(messages_path))
    monkeypatch.setattr(retrieval, "CACHE_PATH", str(tmp_path / ".cache" / "index.pkl"))
    monkeypatch.setattr(retrieval, "_INDEX", None)

    found = retrieval.retrieve_messages("u1", "2025-02-02", "salary payout", k=5)

    assert [message["message_id"] for message in found] == ["old"]


def test_generic_query_does_not_include_request_id():
    query = retrieval.generic_request_query({
        "request_id": "request_123", "request_type": "purchase",
        "request_text": "Could I pay for an annual pass?",
    })
    assert "request_123" not in query
    assert "income, payout" not in query  # Terms are added as plain retrieval tokens.
    assert all(term in query for term in ("income", "payout", "refund", "salary", "pending"))
