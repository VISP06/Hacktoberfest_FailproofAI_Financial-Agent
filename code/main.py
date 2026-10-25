import os
import pandas as pd
import datetime
from dotenv import load_dotenv
from financial_engine import FinancialEngine

load_dotenv()

DATASET_DIR = "dataset"
OUTPUT_FILE = "output.csv"

def format_payment_plan(plan_dict):
    if not plan_dict: 
        return "none"
    sorted_dates = sorted(plan_dict.keys())
    return "|".join([f"{d}:{round(plan_dict[d], 2)}" for d in sorted_dates])

def run_pipeline():
    print("=== Running Buy or Wait? Final Decision Engine ===")
    
    requests_path = os.path.join(DATASET_DIR, "requests.csv")
    requests_df = pd.read_csv(requests_path)
    engine = FinancialEngine(DATASET_DIR)
    
    results = []
    
    for _, row in requests_df.iterrows():
        req_id = str(row["request_id"])
        user_id = str(row["user_id"])
        req_date = str(row["request_date"])
        req_amount = float(row["requested_amount"])
        desired_date = str(row["desired_completion_date"])
        allows_partial = str(row["allows_partial_payment"]).lower() == "true"
        
        user_profile = engine.profiles.loc[user_id]
        considered_methods = str(user_profile.get("payment_methods_user_will_consider", "")).lower()
        
        # 1. Get Base 90-Day Simulation
        balances, min_bal = engine.simulate_90_days(user_id, req_date)
        
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
        
        # Calculate earliest_date_for_full_payment
        earliest_full_date = ""
        for d in sorted(balances.keys()):
            if d < req_date: 
                continue
            if is_plan_safe({d: req_amount}):
                earliest_full_date = d
                break

        # 2. Generate Eligible Candidate Plans
        candidates = []
        
        # Full Payment
        if "full_payment" in considered_methods and is_plan_safe({req_date: req_amount}):
            candidates.append({
                "method": "full_payment", "plan_dict": {req_date: req_amount},
                "status": "affordable_now", "completion_date": req_date,
                "cost": req_amount, "start_date": req_date, "num_payments": 1, "opt_id": "zzzzzz"
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
                            "cost": req_amount, "start_date": req_date, "num_payments": 2, "opt_id": "zzzzzz"
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
                        "cost": total_cost, "start_date": first_date_str, "num_payments": num_pmts, "opt_id": opt_id
                    })
                    
        # Wait
        if "full_payment" in considered_methods and earliest_full_date and earliest_full_date > req_date:
            candidates.append({
                "method": "wait", "plan_dict": {earliest_full_date: req_amount},
                "status": "affordable_later", "completion_date": earliest_full_date,
                "cost": req_amount, "start_date": earliest_full_date, "num_payments": 1, "opt_id": "zzzzzz"
            })

        # 3. Apply Strict Type-Safe 6-Rule Tie-Breaker
        if candidates:
            def sort_key(c):
                meets_deadline = 0 if str(c["completion_date"]) <= desired_date else 1
                spending_changes = 0
                return (
                    meets_deadline, 
                    spending_changes, 
                    float(c["cost"]), 
                    str(c["start_date"]), 
                    int(c["num_payments"]), 
                    str(c["opt_id"])
                )
                
            candidates.sort(key=sort_key)
            best_plan = candidates[0]
            
            results.append({
                "request_id": req_id,
                "amount_safe_to_pay": round(amount_safe_today, 2),
                "affordability_status": best_plan["status"],
                "recommended_payment_method": best_plan["method"],
                "payment_plan": format_payment_plan(best_plan["plan_dict"]),
                "earliest_date_for_full_payment": earliest_full_date,
                "spending_changes_needed": "none",
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
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully evaluated and ranked {len(output_df)} requests into {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()