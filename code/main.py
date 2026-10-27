import os
import itertools
import pandas as pd
import datetime
from dotenv import load_dotenv
from financial_engine import FinancialEngine
import failproofai_sdk as fp

load_dotenv()
fp.configure(environment="dev")

DATASET_DIR = "dataset"
OUTPUT_FILE = "output.csv"
DEFAULT_FORECAST_OPTIONS = {
    "variable_category_set": "all_non_recurring_fixed_flexibility_spending",
    "variable_factor": 1.0,
    "trailing_window_days": 60,
}

def format_payment_plan(plan_dict):
    if not plan_dict:
        return "none"
    sorted_dates = sorted(plan_dict.keys())
    def format_amount(value):
        value = float(value)
        return str(int(round(value))) if value.is_integer() else f"{value:.2f}"
    return "|".join([f"{d}:{format_amount(plan_dict[d])}" for d in sorted_dates])


def _eligible_spending_changes(engine, user_id, request_date, profile):
    events = engine.forecast.valid_events(user_id)
    series = engine.forecast.recurring_series(events, datetime.date.fromisoformat(request_date[:10]))
    recurring_keys = {item["key"] for item in series}
    protected = {x.strip().lower() for x in str(profile.expense_categories_to_protect).split("|") if x.strip()}
    reducible = {x.strip().lower() for x in str(profile.expense_categories_user_is_willing_to_reduce).split("|") if x.strip()}
    stoppable = {x.strip().lower() for x in str(profile.expense_categories_user_is_willing_to_stop).split("|") if x.strip()}
    candidates = {}
    for _, row in events.iterrows():
        key = (str(row.user_id), str(row.direction), str(row.description))
        category = str(row.category).lower()
        flexibility = str(row.flexibility).lower()
        if str(row.direction).lower() not in {"debit", "out"}:
            continue
        if category in protected or category not in reducible.union(stoppable):
            continue
        if flexibility == "reducible" and category in reducible:
            action, amount = "reduce_to", row.minimum_allowed_amount
        elif flexibility == "reducible_or_stoppable" and category in reducible:
            action, amount = "reduce_to", row.minimum_allowed_amount
        elif flexibility == "stoppable" and category in stoppable:
            action, amount = "stop", 0.0
        else:
            continue
        if action == "reduce_to" and pd.isna(amount):
            continue
        # Use the latest row as the representative event for its series.
        parsed = engine.forecast._date(row.event_date)
        previous = candidates.get(key)
        if previous is None or parsed > previous["date"]:
            candidates[key] = {
                "event_id": str(row.event_id), "action": action,
                "amount": float(amount), "date": parsed,
                "benefit": max(0.0, float(row.amount) - float(amount)) if not pd.isna(row.amount) else 0.0,
                "text": (f"{action}:{row.event_id}:{float(amount):g}"
                          if action == "reduce_to" else f"stop:{row.event_id}"),
            }
    return list(candidates.values())


def _find_spending_changes(engine, user_id, request_date, requested_amount, forecast_options, profile):
    candidates = _eligible_spending_changes(engine, user_id, request_date, profile)
    for count in range(1, min(3, len(candidates)) + 1):
        for selected in itertools.combinations(candidates, count):
            changes = [{key: item[key] for key in ("event_id", "action", "amount")} for item in selected]
            balances, minimum = engine.simulate_90_days(
                user_id, request_date, spending_changes=changes, **forecast_options
            )
            if min(balances.values()) - float(requested_amount) >= minimum:
                return changes, "|".join(item["text"] for item in selected)
    return None, "none"

def run_pipeline(requests_path=None, output_file=OUTPUT_FILE, forecast_options=None):
    print("=== Running Buy or Wait? Final Decision Engine ===")

    requests_path = requests_path or os.path.join(DATASET_DIR, "requests.csv")
    requests_df = pd.read_csv(requests_path)
    engine = FinancialEngine(DATASET_DIR)
    forecast_options = {**DEFAULT_FORECAST_OPTIONS, **(forecast_options or {})}
    simulation_options = {
        key: value for key, value in forecast_options.items()
        if key != "enable_spending_changes"
    }

    results = []

    for _, row in requests_df.iterrows():
        req_id = str(row["request_id"])
        with fp.session(f"req-{req_id}"):
            with fp.agent("decision-engine", goal=str(row["request_text"])):
                user_id = str(row["user_id"])
                req_date = str(row["request_date"])
                req_amount = float(row["requested_amount"])
                desired_date = str(row["desired_completion_date"])
                allows_partial = str(row["allows_partial_payment"]).lower() == "true"

                user_profile = engine.profiles.loc[user_id]
                considered_methods = str(user_profile.get("payment_methods_user_will_consider", "")).lower()

                # 1. Get Base 90-Day Simulation
                with fp.tool_call(
                    "simulate_90_days",
                    input={"user_id": user_id, "start_date": req_date},
                ) as tool:
                    balances, min_bal = engine.simulate_90_days(
                        user_id, req_date, **simulation_options
                    )
                    tool.output = {"min_balance": min_bal}

                # Helper: Check if a payment dictionary {date: amount} is safe
                def is_plan_safe(payments):
                    for day_str, bal in balances.items():
                        cumulative_paid = sum(amt for p_date, amt in payments.items() if p_date <= day_str)
                        if bal - cumulative_paid < min_bal:
                            return False
                    return True

                # Calculate amount_safe_to_pay today
                buffer_today = min(balances.values()) - min_bal
                amount_safe_today = min(req_amount, max(0.0, buffer_today)) if buffer_today > 0 else 0.0

                spending_changes = None
                spending_changes_text = "none"
                if (forecast_options.get("enable_spending_changes", True)
                        and amount_safe_today < req_amount
                        and "full_payment" in considered_methods):
                    spending_changes, spending_changes_text = _find_spending_changes(
                        engine, user_id, req_date, req_amount, simulation_options, user_profile
                    )

                # The date is capacity-only: for each candidate d, check every day
                # from d through request_date+90 (not the days before d).
                with fp.tool_call(
                    "forecast_safety",
                    input={"request_id": req_id, "horizon_days": 90},
                ) as tool:
                    earliest_full_date = ""
                    for d in sorted(balances.keys()):
                        if d < req_date:
                            continue
                        if all(
                            balances[day] - req_amount >= min_bal
                            for day in balances
                            if day >= d
                        ):
                            earliest_full_date = d
                            break
                    tool.output = {"earliest_safe_date": earliest_full_date or None}

                # 2. Generate Eligible Candidate Plans
                candidates = []

                # Full Payment
                if "full_payment" in considered_methods and is_plan_safe({req_date: req_amount}):
                    candidates.append({
                        "method": "full_payment", "plan_dict": {req_date: req_amount},
                        "status": "affordable_now", "completion_date": req_date,
                        "cost": req_amount, "start_date": req_date, "num_payments": 1, "opt_id": "zzzzzz",
                        "spending_changes": "none", "change_count": 0,
                    })
                elif spending_changes:
                    candidates.append({
                        "method": "full_payment", "plan_dict": {req_date: req_amount},
                        "status": "affordable_with_plan", "completion_date": req_date,
                        "cost": req_amount, "start_date": req_date, "num_payments": 1, "opt_id": "zzzzzz",
                        "spending_changes": spending_changes_text, "change_count": len(spending_changes),
                    })

                # Partial Payment
                if allows_partial and "partial_payment" in considered_methods:
                    if 0 < amount_safe_today < req_amount and earliest_full_date:
                        if earliest_full_date <= desired_date:
                                partial_plan = {req_date: amount_safe_today, earliest_full_date: req_amount - amount_safe_today}
                            if is_plan_safe(partial_plan):
                                candidates.append({
                                    "method": "partial_payment", "plan_dict": partial_plan,
                                    "status": "affordable_with_plan", "completion_date": earliest_full_date,
                                    "cost": req_amount, "start_date": req_date, "num_payments": 2, "opt_id": "zzzzzz",
                                    "spending_changes": "none", "change_count": 0,
                                })

                # Installments
                if "installments" in considered_methods:
                    req_options = engine.payment_options[engine.payment_options["request_id"] == req_id]
                    for _, opt in req_options.iterrows():
                        opt_id = str(opt["payment_option_id"]) if not pd.isna(opt["payment_option_id"]) else "zzzzzz"

                        num_pmts = int(opt["number_of_payments"]) if not pd.isna(opt["number_of_payments"]) else 1
                        freq = int(opt["payment_frequency_days"]) if not pd.isna(opt["payment_frequency_days"]) else 0
                        pmt_amt = float(opt["payment_amount"]) if not pd.isna(opt["payment_amount"]) else 0.0
                        total_cost = float(opt["total_payable_amount"]) if not pd.isna(opt["total_payable_amount"]) else (pmt_amt * num_pmts)

                        first_date_str = str(opt["first_payment_date"]) if not pd.isna(opt["first_payment_date"]) else req_date

                        try:
                            first_date = datetime.date.fromisoformat(first_date_str)
                        except ValueError:
                            first_date = datetime.date.fromisoformat(req_date)

                        inst_plan = {}
                        for i in range(num_pmts):
                            p_date = (first_date + datetime.timedelta(days=i*freq)).isoformat()
                            inst_plan[p_date] = inst_plan.get(p_date, 0.0) + pmt_amt

                        last_date = sorted(inst_plan.keys())[-1]

                        if last_date in balances and is_plan_safe(inst_plan):
                            candidates.append({
                                "method": "installments", "plan_dict": inst_plan,
                                "status": "affordable_with_plan", "completion_date": last_date,
                                "cost": total_cost, "start_date": first_date_str, "num_payments": num_pmts, "opt_id": opt_id,
                                "spending_changes": "none", "change_count": 0,
                            })

                # Wait
                if "full_payment" in considered_methods and earliest_full_date and earliest_full_date > req_date:
                    candidates.append({
                        "method": "wait", "plan_dict": {earliest_full_date: req_amount},
                        "status": "affordable_later", "completion_date": earliest_full_date,
                        "cost": req_amount, "start_date": earliest_full_date, "num_payments": 1, "opt_id": "zzzzzz",
                        "spending_changes": "none", "change_count": 0,
                    })

                # 3. Apply Strict Type-Safe 6-Rule Tie-Breaker
                if candidates:
                    def sort_key(c):
                        meets_deadline = 0 if str(c["completion_date"]) <= desired_date else 1
                        spending_change_count = int(c.get("change_count", 0))
                        return (
                            meets_deadline,
                            spending_change_count,
                            float(c["cost"]),
                            str(c["start_date"]),
                            int(c["num_payments"]),
                            str(c["opt_id"])
                        )

                    with fp.tool_call(
                        "rank_candidates",
                        input={"request_id": req_id, "candidate_count": len(candidates)},
                    ) as tool:
                        candidates.sort(key=sort_key)
                        best_plan = candidates[0]
                        tool.output = {"chosen_method": best_plan["method"]}

                    results.append({
                        "request_id": req_id,
                        "amount_safe_to_pay": round(amount_safe_today, 2),
                        "affordability_status": best_plan["status"],
                        "recommended_payment_method": best_plan["method"],
                        "payment_plan": format_payment_plan(best_plan["plan_dict"]),
                        "earliest_date_for_full_payment": earliest_full_date,
                        "spending_changes_needed": best_plan["spending_changes"],
                        "decision_explanation": f"Recommended {best_plan['method']} with total cost {best_plan['cost']}."
                    })
                else:
                    results.append({
                        "request_id": req_id,
                        "amount_safe_to_pay": round(amount_safe_today, 2),
                        "affordability_status": "not_affordable",
                        "recommended_payment_method": "not_recommended",
                        "payment_plan": "none",
                        "earliest_date_for_full_payment": earliest_full_date,
                        "spending_changes_needed": "none",
                        "decision_explanation": "No safe eligible payment method found without dropping below minimum balance."
                    })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Successfully evaluated and ranked {len(output_df)} requests into {output_file}")
    return output_df

if __name__ == "__main__":
    run_pipeline()
