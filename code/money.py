"""Consistent currency formatting for assistant and report text."""


def money(value, currency):
    return f"{currency} {float(value):,.2f}"
