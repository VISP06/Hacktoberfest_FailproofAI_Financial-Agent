import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from evaluation.compare_income_caches import compare_caches


def test_compare_caches_reports_matches_and_disagreements():
    cache_a = {
        ("u1", "salary:1"): "confirmed_recurring",
        ("u2", "description:Consulting"): "variable_unconfirmed",
    }
    cache_b = {
        ("u1", "salary:1"): "confirmed_recurring",
        ("u2", "description:Consulting"): "one_off",
        ("u3", "salary:2"): "new_recurring",
    }

    shared, matches, confusion, disagreements = compare_caches(cache_a, cache_b)

    assert len(shared) == 2
    assert matches == 1
    assert confusion[("confirmed_recurring", "confirmed_recurring")] == 1
    assert confusion[("variable_unconfirmed", "one_off")] == 1
    assert disagreements == [("u2", "description:Consulting", "variable_unconfirmed", "one_off")]
