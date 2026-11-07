import os
import sys

import pytest
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from assistant import Assistant


@pytest.mark.parametrize("user_id", ["user_01", "user_05", "user_07", "user_09", "user_13", "user_20"])
@pytest.mark.parametrize("headroom_multiple", [0.5, 0.98, 1.02, 3.0])
def test_pay_in_full_safety_matches_engine_situation(user_id, headroom_multiple):
    events = pd.read_csv(os.path.join(ROOT, "dataset", "financial_events.csv"), dtype={"user_id": str})
    user_dates = events.loc[events.user_id.astype(str).eq(user_id), "event_date"].dropna().astype(str)
    assistant = Assistant(user_id, max(user_dates)[:10], dataset_dir=os.path.join(ROOT, "dataset"), no_llm=True)
    situation = assistant.situation()
    headroom = situation["lowest_projected_balance"] - situation["minimum_balance"]
    assert headroom > 0
    amount = headroom * headroom_multiple

    result = assistant.advise({
        "amount": amount,
        "currency": assistant.engine.get_user_home_currency(user_id),
        "item": "oracle regression payment",
        "pay_by_date": None,
        "allow_partial": False,
        "allow_installments": False,
    })
    full = next(option for option in result["ranked_options"] if option["type"] == "pay_in_full_now")
    expected_lowest = situation["lowest_projected_balance"] - amount
    assert full["lowest_projected_balance"] == pytest.approx(expected_lowest, abs=0.01)
    assert full["safe"] is (expected_lowest >= situation["minimum_balance"])
