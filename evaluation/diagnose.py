"""Print detailed forecast diagnostics for the sample requests."""

import contextlib
import datetime as dt
import io
import os
import sys
from collections import defaultdict

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "code"))

from financial_engine import FinancialEngine
from main import run_pipeline


DIAGNOSE_FORECAST_OPTIONS = {
    "variable_category_set": "protected_categories_only",
    "variable_factor": 0.875,
    "trailing_window_days": 60,
}


def forecast_category_totals(engine, user_id, start_text, options):
    start = engine.forecast._date(start_text)
    end = start + dt.timedelta(days=90)
    events = engine.forecast.valid_events(user_id)
    series = engine.forecast.recurring_series(
        events, start, options.get("min_recurring_months", 3),
        options.get("day_stability_tolerance", 0),
        options.get("include_request_date", False), options.get("amount_mode", "latest"),
    )
    replacement_months = set()
    totals = {"income": defaultdict(float), "expense": defaultdict(float)}
    future_variable_dates = set()
    for _, row in events.iterrows():
        day = engine.forecast._date(row.settlement_date) if (
            str(row.status).lower() in {"pending", "scheduled"}
            and engine.forecast._date(row.settlement_date)
        ) else engine.forecast._date(row.event_date)
        if day is None or not (start <= day <= end) or pd.isna(row.amount):
            continue
        amount = engine.forecast.convert_to_home_currency(
            row.amount, row.currency, engine.profiles.loc[str(user_id)].home_currency, day.isoformat()
        )
        is_income = engine.forecast._is_credit(row)
        kind = "income" if is_income else "expense"
        category = str(row.category)
        totals[kind][category] += amount
        if str(row.status).lower() == "scheduled":
            replacement_months.add((day.year, day.month, str(row.direction).lower(), category))
        if engine.forecast._is_variable(row, {s["key"] for s in series}):
            future_variable_dates.add(day)

    home = engine.profiles.loc[str(user_id)].home_currency
    for item in series:
        month = pd.Timestamp(start.year, start.month, 1).date()
        for _ in range(5):
            day = engine.forecast._month_date(month.year, month.month, item["day"])
            if start <= day <= end and (day.year, day.month, item["key"][1].lower(), item["category"]) not in replacement_months:
                amount = engine.forecast.convert_to_home_currency(item["amount"], item["currency"], home, day.isoformat())
                kind = "income" if item["key"][1].lower() in {"credit", "in"} else "expense"
                totals[kind][item["category"]] += amount
            month = (month.replace(year=month.year + 1, month=1)
                     if month.month == 12 else month.replace(month=month.month + 1))

    categories = None
    if options.get("variable_category_set") in {"protected_categories_only", "protected"}:
        categories = {x.strip().lower() for x in str(engine.profiles.loc[str(user_id)].expense_categories_to_protect).split("|") if x.strip()}
    historical_start = start - dt.timedelta(days=int(options.get("trailing_window_days", 90)))
    historical = events.copy()
    historical["parsed_date"] = historical.event_date.map(engine.forecast._date)
    historical = historical[historical.parsed_date.notna() & (historical.status.astype(str).str.lower() == "settled") &
                            (historical.parsed_date >= historical_start) & (historical.parsed_date < start)]
    variable_total = sum(
        engine.forecast.convert_to_home_currency(row.amount, row.currency, home, row.parsed_date.isoformat())
        for _, row in historical.iterrows()
        if engine.forecast._is_variable(row, {s["key"] for s in series})
        and (categories is None or str(row.category).lower() in categories) and not pd.isna(row.amount)
    )
    daily = variable_total / float(options.get("trailing_window_days", 90)) * float(options.get("variable_factor", 1.0))
    totals["expense"]["variable_spending"] += daily * sum(day not in future_variable_dates for day in (start + dt.timedelta(days=i) for i in range(91)))
    return {kind: dict(sorted(values.items())) for kind, values in totals.items()}


def main():
    sample_path = os.path.join(ROOT_DIR, "dataset", "sample_requests.csv")
    expected = pd.read_csv(sample_path)
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        actual = run_pipeline(
            requests_path=sample_path, output_file=os.devnull,
            forecast_options=DIAGNOSE_FORECAST_OPTIONS,
        )
    options = DIAGNOSE_FORECAST_OPTIONS
    engine = FinancialEngine(os.path.join(ROOT_DIR, "dataset"))
    actual_by_id = actual.set_index("request_id")
    errors = []
    for _, request in expected.iterrows():
        request_id = str(request.request_id)
        user_id = str(request.user_id)
        balances, minimum = engine.simulate_90_days(user_id, request.request_date, **options)
        minimum_date = min(balances, key=balances.get)
        totals = forecast_category_totals(engine, user_id, request.request_date, options)
        predicted = float(actual_by_id.loc[request_id, "amount_safe_to_pay"])
        expected_amount = float(request.amount_safe_to_pay)
        absolute_error = abs(predicted - expected_amount)
        relative_error = absolute_error / max(abs(expected_amount), 1.0)
        errors.append((relative_error, absolute_error, request_id))
        print(f"{request_id}: balance={engine.profiles.loc[user_id, 'current_available_balance']}, minimum_balance={minimum}; expected_amount_safe_to_pay={expected_amount}, predicted_amount_safe_to_pay={predicted}; forecast_minimum_date={minimum_date}, forecast_minimum_value={balances[minimum_date]}; forecast_income_by_category={totals['income']}; forecast_expense_by_category={totals['expense']}")
    print("\n7 requests with largest relative error:")
    for relative_error, absolute_error, request_id in sorted(errors, reverse=True)[:7]:
        print(f"{request_id}: relative_error={relative_error}, absolute_error={absolute_error}")


if __name__ == "__main__":
    main()
