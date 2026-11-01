"""Deterministic, evidence-backed decision explanations."""

def build_explanation(*, balances, breakdown, minimum_balance, chosen_option, desired_date,
                      retrieved_messages=(), earliest_date_for_full_payment=""):
    """Return a short explanation and the exact engine facts used to write it."""
    lowest_date = min(balances, key=balances.get)
    flows = breakdown.get("flows_by_day", {})
    income_groups = {}
    for day_flows in flows.values():
        for flow in day_flows:
            amount = float(flow["amount"])
            if amount > 0 and flow["source"] == "income":
                key = flow.get("series_key") or flow.get("event_id") or "income"
                group = income_groups.setdefault(key, {"amount": 0.0, "event_ids": [], "dates": []})
                group["amount"] += amount
                if flow.get("event_id"):
                    group["event_ids"].append(str(flow["event_id"]))
                if flow.get("date"):
                    group["dates"].append(str(flow["date"]))
    incomes = [{"series_key": key, "amount": item["amount"],
                "payment_count": len(item["dates"]), "event_ids": item["event_ids"],
                "dates": item["dates"]} for key, item in sorted(income_groups.items())]
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
    safe_plan = chosen_option.get("method") != "not_recommended"
    evidence = {
        "current_balance": breakdown["initial_balance"],
        "minimum_balance_to_keep": minimum_balance,
        "lowest_projected_balance": balances[lowest_date],
        "lowest_projected_date": lowest_date,
        "safe_plan_found": safe_plan,
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
            "earliest_date_for_full_payment": earliest_date_for_full_payment,
        },
    }
    def human_label(label):
        return {"variable_spending": "everyday spending"}.get(label, str(label).replace("_", " "))

    def describe_driver(item):
        ids = list(map(str, item.get("event_ids", [])))
        more = f", +{len(ids) - 3} more" if len(ids) > 3 else ""
        return f"{human_label(item['label'])} ({item['amount']:g}; events {', '.join(ids[:3]) or 'recurring'}{more})"

    drivers_text = ", ".join(describe_driver(item) for item in evidence["largest_outflow_drivers"]) or "no major outflow drivers identified"
    income_text = ", ".join(
        f"{item['series_key']}:{item['amount']:g} per month, {item['payment_count']} payments in the window"
        for item in incomes
    ) or "no income counted"
    excluded_text = "; ".join(
        f"{item.get('series_key', item.get('event_id', 'income'))}: {item.get('reason', item.get('decision', 'excluded'))}"
        for item in evidence["income_excluded_or_fallback"]
    ) or "no excluded or fallback income"
    excluded_variable = [
        item for item in evidence["income_excluded_or_fallback"]
        if item.get("decision") == "excluded"
        and any(word in str(item.get("reason", "")).lower() for word in ("variable", "unconfirmed"))
    ]
    if excluded_variable:
        excluded_text += "; " + "; ".join(
            f"forecast counts no income from {item.get('series_key', item.get('event_id', 'it'))}"
            for item in excluded_variable
        )
    pending_text = ", ".join(ignored) or "no pending items ignored"
    refund_text = f" Refund events counted: {', '.join(refund_event_ids)}." if refund_event_ids else ""
    option = evidence["chosen_option"]
    change_text = option["spending_changes"]
    spending_change_explanation = ""
    if change_text != "none" and not earliest_date_for_full_payment:
        expense_names = {}
        for day_flows in flows.values():
            for flow in day_flows:
                identifiers = set(map(str, flow.get("event_ids", [])))
                if flow.get("event_id"):
                    identifiers.add(str(flow["event_id"]))
                name = flow.get("description") or flow.get("category") or flow.get("series_key")
                if name:
                    for identifier in identifiers:
                        expense_names[identifier] = human_label(name)
        for driver in largest:
            for identifier in map(str, driver.get("event_ids", [])):
                expense_names.setdefault(identifier, human_label(driver["label"]))

        change_actions = []
        for change in change_text.split("|"):
            pieces = change.split(":")
            action = pieces[0].lower()
            event_id = pieces[1] if len(pieces) > 1 else ""
            expense = expense_names.get(event_id, "the named expense")
            verb = "stopping" if action == "stop" else "reducing"
            change_actions.append(f"{verb} {expense}")
        named_changes = " and ".join(change_actions)
        spending_change_explanation = (
            " Without these changes, no safe date for the full amount appears in the 90-day forecast. "
            f"{named_changes.capitalize()} is what makes paying in full on "
            f"{option['completion_date']} safe."
        )
    message_citations = sorted({str(mid) for ids in evidence["retrieved_message_ids"].values() for mid in ids})
    message_text = f" A message from this user was considered (message IDs: {', '.join(message_citations)})." if message_citations else ""
    if not safe_plan:
        reason = (f"No option is recommended because the lowest projected balance falls below the minimum to keep."
                  if balances[lowest_date] < minimum_balance else
                  "No safe plan meets the requested deadline.")
        explanation = (f"{reason} The lowest projected balance is {evidence['lowest_projected_balance']:g} "
                       f"on {lowest_date}, against a minimum of {minimum_balance:g}. "
                       f"Largest outflows: {drivers_text}; counted income: {income_text}; {excluded_text}; "
                       f"pending items ignored: {pending_text}.{refund_text}{message_text}")
        return explanation, evidence
    explanation = (
        f"Current balance is {evidence['current_balance']:g}; keep at least {minimum_balance:g}. "
        f"The lowest projected balance is {evidence['lowest_projected_balance']:g} on {lowest_date}. "
        f"Largest outflows: {drivers_text}; counted income: {income_text}; {excluded_text}; pending items ignored: {pending_text}.{refund_text} "
        f"{option['method']} is recommended because it completes by {option['completion_date']} "
        f"against the {desired_date} deadline, uses spending changes {change_text}, and costs {option['cost']:g}.{message_text}"
        f"{spending_change_explanation}"
    )
    return explanation, evidence
