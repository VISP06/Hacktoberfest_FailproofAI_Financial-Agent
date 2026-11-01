import csv

from evaluation.validate_output import EXPECTED_COLUMNS, validate


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_output_checks_shape_and_plan_invariants(tmp_path):
    requests = tmp_path / "requests.csv"
    output = tmp_path / "output.csv"
    _write_csv(requests, ["request_id", "requested_amount", "request_date"], [
        {"request_id": "r1", "requested_amount": "100", "request_date": "2026-01-01"},
        {"request_id": "r2", "requested_amount": "50", "request_date": "2026-01-02"},
    ])
    valid_row = {
        "request_id": "r1", "amount_safe_to_pay": "25",
        "affordability_status": "affordable_now",
        "recommended_payment_method": "full_payment",
        "payment_plan": "2026-01-01:100", "earliest_date_for_full_payment": "2026-01-01",
        "spending_changes_needed": "none", "decision_explanation": "safe",
    }
    invalid_row = {
        "request_id": "r2", "amount_safe_to_pay": "60",
        "affordability_status": "not_affordable", "recommended_payment_method": "installments",
        "payment_plan": "2026-02-02:20|2026-02-01:20",
        "earliest_date_for_full_payment": "2026-02-01",
        "spending_changes_needed": "none", "decision_explanation": "unsafe",
    }
    _write_csv(output, EXPECTED_COLUMNS, [valid_row, invalid_row, invalid_row])

    violations = validate(output, requests)

    assert "safe_amount_range" in violations
    assert "payment_plan_total" in violations
    assert "payment_plan_chronology" in violations
    assert "not_affordable_date" in violations
    assert "request_ids" in violations
