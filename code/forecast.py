"""Deterministic cash-flow forecasting primitives.

The forecast is deliberately independent of the decision/ranking layer.  It
uses only supplied financial records and reports exchange-rate fallbacks via
the same FailproofAI SDK used by the rest of the agent.
"""

import datetime as dt
import os

import failproofai_sdk as fp
import pandas as pd


class Forecast:
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = dataset_dir
        self.profiles = pd.read_csv(os.path.join(dataset_dir, "financial_profiles.csv"), skipinitialspace=True)
        self.profiles.columns = self.profiles.columns.str.strip()
        self.profiles["user_id"] = self.profiles["user_id"].astype(str)
        self.profiles.set_index("user_id", inplace=True)

        events_path = os.path.join(dataset_dir, "patched_financial_events.csv")
        self.events = pd.read_csv(
            events_path if os.path.exists(events_path) else os.path.join(dataset_dir, "financial_events.csv"),
            skipinitialspace=True,
        )
        self.events.columns = self.events.columns.str.strip()
        self.events["user_id"] = self.events["user_id"].astype(str)
        self.exchange_rates = pd.read_csv(os.path.join(dataset_dir, "exchange_rates.csv"), skipinitialspace=True)
        self.exchange_rates.columns = self.exchange_rates.columns.str.strip()
        self.exchange_rates["rate_date_parsed"] = pd.to_datetime(
            self.exchange_rates["rate_date"], errors="coerce"
        ).dt.date

    @staticmethod
    def _error(error_type, message, **fields):
        current = fp.current()
        fp.event.error(
            session_id=(current.session_id if current else None) or "financial-engine",
            agent_id="decision-engine", error_type=error_type, message=message, **fields,
        )

    def convert_to_home_currency(self, amount, from_curr, to_curr, date_str):
        if pd.isna(amount):
            return amount
        if str(from_curr) == str(to_curr):
            return float(amount)
        target = dt.date.fromisoformat(str(date_str))
        source, destination = str(from_curr), str(to_curr)
        direct = self.exchange_rates[
            (self.exchange_rates.from_currency == source)
            & (self.exchange_rates.to_currency == destination)
            & (self.exchange_rates.rate_date_parsed <= target)
        ].sort_values("rate_date_parsed")
        if not direct.empty:
            row = direct.iloc[-1]
            if row.rate_date_parsed != target:
                self._error(
                    "ExchangeRateFallback", "Using the nearest earlier exchange rate",
                    from_currency=source, to_currency=destination, date=str(date_str),
                    rate_date=str(row.rate_date_parsed),
                )
            return float(amount) * float(row.rate)

        inverse = self.exchange_rates[
            (self.exchange_rates.from_currency == destination)
            & (self.exchange_rates.to_currency == source)
            & (self.exchange_rates.rate_date_parsed <= target)
        ].sort_values("rate_date_parsed")
        if not inverse.empty:
            row = inverse.iloc[-1]
            self._error(
                "ExchangeRateFallback", "Using an inverse exchange-rate pair",
                from_currency=source, to_currency=destination, date=str(date_str),
                rate_date=str(row.rate_date_parsed),
            )
            return float(amount) / float(row.rate)

        self._error(
            "MissingExchangeRate", "No exchange rate exists on or before the requested date",
            from_currency=source, to_currency=destination, date=str(date_str),
        )
        raise ValueError(f"No exchange rate for {source}/{destination} on or before {date_str}")

    @staticmethod
    def _status_rank(status):
        return {"settled": 4, "scheduled": 3, "pending": 2, "failed": 1, "cancelled": 0}.get(
            str(status).lower(), 0
        )

    def valid_events(self, user_id):
        """Return cash events, removing retries/lifecycle duplicates safely."""
        events = self.events[self.events.user_id == str(user_id)].copy()
        events = events.drop_duplicates().reset_index(drop=True)
        status = events.status.astype(str).str.lower()
        events = events[~status.isin({"failed", "cancelled", "unrealized"})].copy()
        # Pending credits are not cash available to the user yet.
        events = events[~((status.loc[events.index] == "pending") &
                          events.direction.astype(str).str.lower().isin({"credit", "in"}))].copy()
        event_type = events.event_type.astype(str).str.lower()
        description = events.description.astype(str).str.lower()
        events = events[~event_type.eq("investment_valuation") &
                        ~description.str.contains("portfolio valuation|unrealized", na=False)].copy()

        # A linked record with the same cash-flow shape is usually a retry or
        # replacement.  Refunds and investment sales have the opposite cash
        # direction and remain separate legitimate flows.
        by_id = events.set_index("event_id")
        remove = set()
        for _, row in events[events.linked_event_id.notna()].iterrows():
            linked_id = str(row.linked_event_id)
            if linked_id not in by_id.index:
                continue
            other = by_id.loc[linked_id]
            same_shape = (
                str(row.direction).lower() == str(other.direction).lower()
                and str(row.event_type).lower() == str(other.event_type).lower()
                and str(row.category) == str(other.category)
                and (pd.isna(row.amount) or pd.isna(other.amount) or
                     float(row.amount) == float(other.amount))
            )
            if same_shape:
                winner, loser = (row, other) if self._status_rank(row.status) >= self._status_rank(other.status) else (other, row)
                if self._status_rank(row.status) == self._status_rank(other.status):
                    winner, loser = (row, other) if str(row.event_date) >= str(other.event_date) else (other, row)
                remove.add(str(loser.event_id))
        return events[~events.event_id.astype(str).isin(remove)].copy()

    @staticmethod
    def _date(value):
        try:
            return dt.date.fromisoformat(str(value)[:10])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_credit(row):
        return str(row.direction).lower() in {"credit", "in"} or str(row.event_type).lower() in {"income", "refund", "investment_sale"}

    def recurring_series(self, events, request_date, min_months=3, day_tolerance=0, include_request_date=False, amount_mode="latest"):
        settled = events[events.status.astype(str).str.lower().eq("settled")].copy()
        settled["parsed_date"] = settled.event_date.map(self._date)
        settled = settled[settled.parsed_date.notna() & (settled.parsed_date <= request_date)]
        if not include_request_date:
            settled = settled[settled.parsed_date < request_date]
        series = []
        settled["_salary_group"] = (
            settled.direction.astype(str).str.lower().isin({"credit", "in"})
            & settled.category.astype(str).str.lower().eq("salary")
        )
        # Salary descriptions and categories can vary.  The day of month is
        # the income-series identity; debit series retain description identity.
        settled["_series_group"] = settled.apply(
            lambda row: f"salary:{row.parsed_date.day}"
            if row._salary_group else f"description:{row.description}", axis=1
        )
        for (user_id, direction, series_group), group in settled.groupby(
            ["user_id", "direction", "_series_group"], dropna=False
        ):
            is_salary = str(series_group).startswith("salary:")
            description = group.description.iloc[-1]
            months = {(d.year, d.month) for d in group.parsed_date}
            days = [d.day for d in group.parsed_date]
            required_months = 2 if is_salary else int(min_months)
            if len(months) < required_months or max(days) - min(days) > int(day_tolerance):
                continue
            latest = group.sort_values("parsed_date").iloc[-1]
            latest_date = latest.parsed_date
            if (request_date - latest_date).days > 35:
                continue
            amount = float(group.amount.dropna().iloc[-1]) if amount_mode == "latest" else float(group.amount.mean())
            series.append({
                "key": (
                    (str(user_id), str(direction), str(series_group))
                    if is_salary
                    else (str(user_id), str(direction), str(description))
                ),
                "category": str(latest.category), "event_type": str(latest.event_type),
                "amount": amount, "currency": str(latest.currency), "day": latest_date.day,
            })
        return series

    @staticmethod
    def _month_date(year, month, day):
        if month == 12:
            next_month = dt.date(year + 1, 1, 1)
        else:
            next_month = dt.date(year, month + 1, 1)
        last_day = (next_month - dt.timedelta(days=1)).day
        return dt.date(year, month, min(day, last_day))

    @staticmethod
    def _is_variable(row, recurring_keys):
        return (
            str(row.direction).lower() in {"debit", "out"}
            and str(row.event_type).lower() == "expense"
            and str(row.category).lower() not in {"transfer", "investment", "investment_purchase"}
            and (str(row.user_id), str(row.direction), str(row.description)) not in recurring_keys
        )

    def simulate(self, user_id, start_date_str, variable_category_set="all_non_recurring_fixed_flexibility_spending",
                 variable_factor=1.0, trailing_window_days=90, day_stability_tolerance=0,
                 min_recurring_months=3, include_request_date=False, amount_mode="latest", additional_payments=None,
                 spending_changes=None):
        start = dt.date.fromisoformat(str(start_date_str)[:10])
        end = start + dt.timedelta(days=90)
        profile = self.profiles.loc[str(user_id)]
        home = profile.home_currency
        events = self.valid_events(user_id)
        series = self.recurring_series(events, start, min_recurring_months, day_stability_tolerance, include_request_date, amount_mode)
        recurring_keys = {s["key"] for s in series}
        changes_by_key = {}
        for change in spending_changes or []:
            event = events[events.event_id.astype(str).eq(str(change["event_id"]))]
            if event.empty:
                continue
            row = event.iloc[0]
            key = (str(row.user_id), str(row.direction), str(row.description))
            changes_by_key[key] = change
        by_day, replacement_months = {}, set()

        future = events.copy()
        future["forecast_date"] = future.apply(
            lambda r: self._date(r.settlement_date) if str(r.status).lower() in {"pending", "scheduled"} and self._date(r.settlement_date) else self._date(r.event_date), axis=1
        )
        for _, row in future.iterrows():
            day = row.forecast_date
            if day is None or not (start <= day <= end) or pd.isna(row.amount):
                continue
            if str(row.status).lower() == "pending" and not self._is_credit(row):
                pass
            amount = self.convert_to_home_currency(row.amount, row.currency, home, day.isoformat())
            change = changes_by_key.get((str(row.user_id), str(row.direction), str(row.description)))
            if change and not self._is_credit(row):
                amount = self.convert_to_home_currency(change["amount"], row.currency, home, day.isoformat())
            by_day[day.isoformat()] = by_day.get(day.isoformat(), 0.0) + (amount if self._is_credit(row) else -amount)
            if str(row.status).lower() == "scheduled":
                replacement_months.add((day.year, day.month, str(row.direction).lower(), str(row.category)))

        for s in series:
            month = dt.date(start.year, start.month, 1)
            for _ in range(5):
                day = self._month_date(month.year, month.month, s["day"])
                if start <= day <= end and (day.year, day.month, s["key"][1].lower(), s["category"]) not in replacement_months:
                    change = changes_by_key.get(s["key"])
                    series_amount = change["amount"] if change else s["amount"]
                    amount = self.convert_to_home_currency(series_amount, s["currency"], home, day.isoformat())
                    by_day[day.isoformat()] = by_day.get(day.isoformat(), 0.0) + (amount if s["key"][1].lower() in {"credit", "in"} else -amount)
                month = dt.date(month.year + (month.month == 12), 1 if month.month == 12 else month.month + 1, 1)

        categories = None
        if variable_category_set in {"protected_categories_only", "protected"}:
            categories = {x.strip().lower() for x in str(profile.expense_categories_to_protect).split("|") if x.strip()}
        historical_start = start - dt.timedelta(days=int(trailing_window_days))
        historical = events.copy()
        historical["parsed_date"] = historical.event_date.map(self._date)
        historical = historical[historical.parsed_date.notna() & (historical.status.astype(str).str.lower() == "settled") &
                                (historical.parsed_date >= historical_start) & (historical.parsed_date < start)]
        variable_total = 0.0
        for _, row in historical.iterrows():
            if self._is_variable(row, recurring_keys) and (categories is None or str(row.category).lower() in categories) and not pd.isna(row.amount):
                change = changes_by_key.get((str(row.user_id), str(row.direction), str(row.description)))
                historical_amount = change["amount"] if change else row.amount
                variable_total += self.convert_to_home_currency(
                    historical_amount, row.currency, home, row.parsed_date.isoformat()
                )
        daily_variable = variable_total / float(trailing_window_days) * float(variable_factor)
        balances, running = {}, float(profile.current_available_balance)
        for offset in range(91):
            day = start + dt.timedelta(days=offset)
            key = day.isoformat()
            flow = by_day.get(key, 0.0)
            if daily_variable and not any(key == str(r.forecast_date) and self._is_variable(r, recurring_keys) for _, r in future.iterrows()):
                flow -= daily_variable
            if additional_payments and key in additional_payments:
                flow -= float(additional_payments[key])
            running += flow
            balances[key] = running
        return balances, float(profile.minimum_balance_to_keep)
