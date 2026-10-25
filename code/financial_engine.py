import pandas as pd
import datetime
import os

class FinancialEngine:
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = dataset_dir
        
        self.profiles = pd.read_csv(f"{dataset_dir}/financial_profiles.csv", skipinitialspace=True)
        self.profiles.columns = self.profiles.columns.str.strip()
        self.profiles.set_index("user_id", inplace=True)
        
        patched_events_path = os.path.join(dataset_dir, "patched_financial_events.csv")
        if os.path.exists(patched_events_path):
            self.events = pd.read_csv(patched_events_path, skipinitialspace=True)
        else:
            self.events = pd.read_csv(f"{dataset_dir}/financial_events.csv", skipinitialspace=True)
        self.events.columns = self.events.columns.str.strip()
        
        self.exchange_rates = pd.read_csv(f"{dataset_dir}/exchange_rates.csv", skipinitialspace=True)
        self.exchange_rates.columns = self.exchange_rates.columns.str.strip()
        
        self.payment_options = pd.read_csv(f"{dataset_dir}/request_payment_options.csv", skipinitialspace=True)
        self.payment_options.columns = self.payment_options.columns.str.strip()

        # --- LOAD PARSED TEXT MESSAGES ---
        parsed_messages_path = os.path.join(dataset_dir, "parsed_messages.csv")
        if os.path.exists(parsed_messages_path):
            self.parsed_messages = pd.read_csv(parsed_messages_path, skipinitialspace=True)
            self.parsed_messages.columns = self.parsed_messages.columns.str.strip()
        else:
            self.parsed_messages = pd.DataFrame(columns=["message_id", "related_event_id", "action", "details"])

    def get_user_home_currency(self, user_id):
        return self.profiles.loc[user_id, "home_currency"]

    def convert_to_home_currency(self, amount, from_curr, to_curr, date_str):
        if from_curr == to_curr or pd.isna(amount):
            return amount
            
        rate_row = self.exchange_rates[
            (self.exchange_rates["rate_date"] == date_str) &
            (self.exchange_rates["from_currency"] == from_curr) &
            (self.exchange_rates["to_currency"] == to_curr)
        ]
        
        if not rate_row.empty:
            return float(amount) * float(rate_row.iloc[0]["rate"])
        
        return float(amount)

    def simulate_90_days(self, user_id, start_date_str, additional_payments=None):
        user_profile = self.profiles.loc[user_id]
        home_curr = user_profile["home_currency"]
        
        current_balance = float(user_profile["current_available_balance"])
        min_balance = float(user_profile["minimum_balance_to_keep"])
        
        start_date = datetime.date.fromisoformat(start_date_str)
        user_events = self.events[self.events["user_id"] == user_id].copy()
        
        # --- EXCLUDE CANCELLED EVENTS FROM MESSAGES ---
        cancelled_event_ids = []
        if not self.parsed_messages.empty:
            cancels = self.parsed_messages[self.parsed_messages["action"] == "cancel"]
            cancelled_event_ids = cancels["related_event_id"].dropna().tolist()
            
        valid_events = user_events[
            ~user_events["status"].astype(str).str.lower().isin(["failed", "cancelled", "unrealized"]) &
            ~user_events["event_id"].isin(cancelled_event_ids)
        ]

        daily_balances = {}
        running_balance = current_balance

        for day_offset in range(91):
            eval_date = start_date + datetime.timedelta(days=day_offset)
            eval_date_str = eval_date.isoformat()
            
            day_events = valid_events[valid_events["event_date"] == eval_date_str]
            
            day_net_flow = 0.0
            for _, event in day_events.iterrows():
                amt = event["amount"]
                if pd.isna(amt):
                    continue
                
                converted_amt = self.convert_to_home_currency(
                    amt, event["currency"], home_curr, eval_date_str
                )
                
                direction = str(event.get("direction", "")).lower()
                event_type = str(event.get("event_type", "")).lower()
                
                if direction in ["credit", "in"] or event_type in ["income", "confirmed_salary", "refund"]:
                    day_net_flow += converted_amt
                else:
                    day_net_flow -= converted_amt

            if additional_payments and eval_date_str in additional_payments:
                day_net_flow -= additional_payments[eval_date_str]

            running_balance += day_net_flow
            daily_balances[eval_date_str] = running_balance

        return daily_balances, min_balance

    def calculate_amount_safe_to_pay(self, user_id, request_date_str, requested_amount):
        balances, min_bal = self.simulate_90_days(user_id, request_date_str)
        lowest_projected_balance = min(balances.values())
        
        available_buffer = lowest_projected_balance - min_bal
        
        if available_buffer <= 0:
            return 0.0
        
        return min(float(requested_amount), max(0.0, available_buffer))