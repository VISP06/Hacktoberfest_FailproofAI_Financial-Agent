"""Deterministic, evidence-backed decision explanations."""

def build_explanation(*, balances, breakdown, minimum_balance, chosen_option, desired_date,
                      retrieved_messages=()):
    """Return a short explanation and the exact engine facts used to write it."""
    lowest_date = min(balances, key=balances.get)
    flows = breakdown.get("flows_by_day", {})
    incomes, income_ids = [], []
    for day_flows in flows.values():
        for flow in day_flows:
            amount = float(flow["amount"])
            if amount > 0 and flow["source"] == "income":
                incomes.append({"amount": amount, "event_id": flow.get("event_id"),
                                "series_key": flow.get("series_key"), "date": flow.get("date")})
                if flow.get("event_id"):
                    income_ids.append(flow["event_id"])
    largest = breakdown.get("largest_outflow_drivers", [])
    ignored = list(breakdown.get("pending_ignored", []))
    retrieved_ids = sorted({str(message["message_id"]) for message in retrieved_messages
                            if message.get("message_id")})
    excluded_or_fallback = breakdown.get("income_decisions", [])
    refund_event_ids = sorted({str(flow["event_id"])
                               for day_flows in flows.values() for flow in day_flows
                               if flow.get("event_id") and any(
                                   "refund" in str(flow.get(field, "")).lower()
                                   for field in ("category", "description", "series_key"))})
    evidence = {
        "current_balance": breakdown["initial_balance"],
        "minimum_balance_to_keep": minimum_balance,
        "lowest_projected_balance": balances[lowest_date],
        "lowest_projected_date": lowest_date,
        "largest_outflow_drivers": largest,
        "income_counted": incomes,
        "income_excluded_or_fallback": breakdown.get("income_decisions", []),
        "pending_items_ignored": ignored,
        "refund_event_ids": refund_event_ids,
        "retrieved_message_ids": {
            "income_exclusion": retrieved_ids if any(
                item.get("decision") == "excluded" or "fallback" in item.get("reason", "").lower()
                for item in excluded_or_fallback
            ) else [],
            "pending_item": retrieved_ids if ignored else [],
            "refund": retrieved_ids if refund_event_ids else [],
        },
        "chosen_option": {
            "method": chosen_option["method"], "completion_date": chosen_option["completion_date"],
            "desired_date": desired_date, "spending_changes": chosen_option.get("spending_changes", "none"),
            "cost": chosen_option["cost"], "rank": 1,
        },
    }
    drivers_text = ", ".join(
        f"{item['label']} ({item['amount']:g}; events {', '.join(item['event_ids']) or 'recurring'})"
        for item in evidence["largest_outflow_drivers"]
    ) or "none"
    income_text = ", ".join(
        f"{item['series_key'] or item['event_id']} ({item['amount']:g})" for item in incomes
    ) or "none"
    excluded_text = "; ".join(
        f"{item.get('series_key', item.get('event_id', 'income'))}: {item.get('reason', item.get('decision', 'excluded'))}"
        for item in evidence["income_excluded_or_fallback"]
    ) or "no excluded or fallback income"
    pending_text = ", ".join(ignored) or "none"
    refund_text = f" Refund events counted: {', '.join(refund_event_ids)}." if refund_event_ids else ""
    option = evidence["chosen_option"]
    change_text = option["spending_changes"]
    explanation = (
        f"Current balance is {evidence['current_balance']:g}; keep at least {minimum_balance:g}. "
        f"The lowest projected balance is {evidence['lowest_projected_balance']:g} on {lowest_date}. "
        f"Largest outflows: {drivers_text}; counted income: {income_text}; {excluded_text}; pending items ignored: {pending_text}.{refund_text} "
        f"{option['method']} ranked first (rank {option['rank']}) because it completes by {option['completion_date']} "
        f"against the {desired_date} deadline, uses spending changes {change_text}, and costs {option['cost']:g}."
    )
    return explanation, evidence
