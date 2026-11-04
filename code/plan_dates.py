"""Shared payment-plan date arithmetic."""

import datetime as dt


def installment_completion_date(as_of, count, interval_days=30):
    """Return the date of the last installment, with the first paid on as_of."""
    if isinstance(as_of, dt.datetime):
        start = as_of.date()
    elif isinstance(as_of, dt.date):
        start = as_of
    else:
        start = dt.date.fromisoformat(str(as_of)[:10])
    count = int(count)
    interval_days = int(interval_days)
    if count < 1 or interval_days < 1:
        raise ValueError("count and interval_days must be positive")
    return start + dt.timedelta(days=(count - 1) * interval_days)
