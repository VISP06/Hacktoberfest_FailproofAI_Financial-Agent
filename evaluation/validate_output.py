#!/usr/bin/env python3
"""Validate decision-engine output against its source requests."""

import csv
import datetime as dt
import math
import re
import sys
from collections import defaultdict


EXPECTED_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "installments", "partial_payment", "wait", "not_recommended"}
PLAN_METHODS = {"full_payment", "installments", "partial_payment"}
DATE_AMOUNT = re.compile(r"^(\d{4}-\d{2}-\d{2}):([0-9]+(?:\.[0-9]+)?)$")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def validate(output_path, requests_path):
    violations = defaultdict(list)
    columns, output = _read_csv(output_path)
    _, requests = _read_csv(requests_path)
    if columns != EXPECTED_COLUMNS:
        violations["exact_columns"].append(
            f"expected {EXPECTED_COLUMNS!r}, got {columns!r}"
        )

    request_rows = {}
    for index, row in enumerate(requests, start=2):
        request_id = row.get("request_id", "")
        if request_id in request_rows:
            violations["request_ids"].append(f"duplicate request id {request_id!r} in requests.csv")
        request_rows[request_id] = row

    output_by_id = {}
    for index, row in enumerate(output, start=2):
        request_id = row.get("request_id", "")
        if request_id in output_by_id:
            violations["request_ids"].append(f"duplicate output request id {request_id!r} at row {index}")
        else:
            output_by_id[request_id] = row
    for missing in sorted(set(request_rows) - set(output_by_id)):
        violations["request_ids"].append(f"missing output row for {missing!r}")
    for extra in sorted(set(output_by_id) - set(request_rows)):
        violations["request_ids"].append(f"unknown output request id {extra!r}")

    for request_id, row in output_by_id.items():
        if request_id not in request_rows:
            continue
        request = request_rows[request_id]
        try:
            amount = float(row["amount_safe_to_pay"])
            requested = float(request["requested_amount"])
            if not math.isfinite(amount) or amount < 0 or amount > requested:
                violations["safe_amount_range"].append(
                    f"{request_id}: amount_safe_to_pay={amount!r}, requested_amount={requested!r}"
                )
        except (KeyError, TypeError, ValueError):
            violations["safe_amount_range"].append(f"{request_id}: invalid amount value")

        status = row.get("affordability_status", "")
        method = row.get("recommended_payment_method", "")
        if status not in STATUSES:
            violations["status_enum"].append(f"{request_id}: invalid status {status!r}")
        if method not in METHODS:
            violations["method_enum"].append(f"{request_id}: invalid method {method!r}")

        earliest = row.get("earliest_date_for_full_payment", "")
        if status == "affordable_now" and earliest != request.get("request_date", ""):
            violations["affordable_now_date"].append(
                f"{request_id}: expected {request.get('request_date')!r}, got {earliest!r}"
            )
        if (status == "not_affordable") != (earliest == ""):
            violations["not_affordable_date"].append(
                f"{request_id}: status={status!r}, earliest_date_for_full_payment={earliest!r}"
            )

        plan = row.get("payment_plan", "")
        plan_amounts = []
        plan_dates = []
        if plan != "none":
            valid_plan = True
            for part in plan.split("|"):
                match = DATE_AMOUNT.fullmatch(part)
                if not match:
                    violations["payment_plan_format"].append(
                        f"{request_id}: invalid payment entry {part!r}"
                    )
                    valid_plan = False
                    continue
                date_text, amount_text = match.groups()
                try:
                    parsed_date = dt.date.fromisoformat(date_text)
                    if parsed_date.isoformat() != date_text:
                        raise ValueError
                    payment_amount = float(amount_text)
                except ValueError:
                    violations["payment_plan_format"].append(
                        f"{request_id}: invalid date or amount in {part!r}"
                    )
                    valid_plan = False
                    continue
                plan_dates.append(parsed_date)
                plan_amounts.append(payment_amount)
            if valid_plan and plan_dates != sorted(plan_dates):
                violations["payment_plan_chronology"].append(
                    f"{request_id}: dates are not chronological"
                )
            if method in PLAN_METHODS and valid_plan:
                try:
                    requested = float(request["requested_amount"])
                    if not math.isclose(sum(plan_amounts), requested, abs_tol=0.01, rel_tol=0.0):
                        violations["payment_plan_total"].append(
                            f"{request_id}: plan totals {sum(plan_amounts)!r}, requested {requested!r}"
                        )
                except (KeyError, ValueError):
                    violations["payment_plan_total"].append(f"{request_id}: invalid requested amount")

    return violations


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: python evaluation/validate_output.py <output.csv> <requests.csv>", file=sys.stderr)
        return 2
    violations = validate(*argv)
    if not violations:
        print("No violations.")
        return 0
    for rule in sorted(violations):
        print(f"{rule}:")
        for violation in violations[rule]:
            print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
