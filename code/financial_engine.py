import os

import failproofai_sdk as fp
import pandas as pd

from forecast import Forecast


class FinancialEngine:
    """Public decision-engine facade over the deterministic forecast."""

    CATEGORY_SETS = {
        "protected_categories_only": "protected_categories_only",
        "protected": "protected_categories_only",
        "all_non_recurring_fixed_flexibility_spending": "all_non_recurring_fixed_flexibility_spending",
        # Kept for callers of the original public API.
        "groceries_only": "protected_categories_only",
        "groceries+transport": "protected_categories_only",
    }

    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = dataset_dir
        self.forecast = Forecast(dataset_dir)
        self.profiles = self.forecast.profiles
        self.events = self.forecast.events
        self.exchange_rates = self.forecast.exchange_rates
        options_path = os.path.join(dataset_dir, "request_payment_options.csv")
        self.payment_options = pd.read_csv(options_path, skipinitialspace=True)
        self.payment_options.columns = self.payment_options.columns.str.strip()

    @staticmethod
    def _emit_error(error_type, message, **fields):
        current = fp.current()
        fp.event.error(
            session_id=(current.session_id if current else None) or "financial-engine",
            agent_id="decision-engine", error_type=error_type, message=message, **fields,
        )

    def get_user_home_currency(self, user_id):
        return self.profiles.loc[str(user_id), "home_currency"]

    def convert_to_home_currency(self, amount, from_curr, to_curr, date_str):
        return self.forecast.convert_to_home_currency(amount, from_curr, to_curr, date_str)

    def _selected_variable_categories(self, category_set):
        if category_set not in self.CATEGORY_SETS:
            raise ValueError(f"Unknown variable spending category set: {category_set}")
        return self.CATEGORY_SETS[category_set]

    def _is_variable(self, event, categories=None):
        # Compatibility helper for callers that passed a category set.
        recurring_keys = set()
        result = self.forecast._is_variable(event, recurring_keys)
        if categories is not None and categories != "all_non_recurring_fixed_flexibility_spending":
            result = result and str(event.get("category", "")).lower() in set(categories)
        return result

    def simulate_90_days(self, user_id, start_date_str, additional_payments=None,
                         variable_category_set="all_non_recurring_fixed_flexibility_spending",
                         variable_factor=1.0, trailing_window_days=90, **forecast_options):
        category_set = self._selected_variable_categories(variable_category_set)
        return self.forecast.simulate(
            user_id, start_date_str, additional_payments=additional_payments,
            variable_category_set=category_set, variable_factor=variable_factor,
            trailing_window_days=trailing_window_days, **forecast_options,
        )

    def calculate_amount_safe_to_pay(self, user_id, request_date_str, requested_amount, **forecast_options):
        balances, minimum = self.simulate_90_days(user_id, request_date_str, **forecast_options)
        capacity = min(balances.values()) - minimum
        return min(float(requested_amount), max(0.0, capacity))
