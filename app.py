"""Streamlit chat UI for the deterministic buy-or-wait assistant."""

import datetime as dt
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from assistant import Assistant


DATASET = os.path.join(ROOT, "dataset")
profiles = pd.read_csv(os.path.join(DATASET, "financial_profiles.csv"), dtype={"user_id": str})
requests = pd.read_csv(os.path.join(DATASET, "requests.csv"), dtype={"user_id": str})

st.set_page_config(page_title="Buy or Wait?", layout="wide")
st.title("Buy or Wait?")

with st.sidebar:
    st.header("Your profile")
    user_ids = profiles["user_id"].astype(str).tolist()
    user_id = st.selectbox(
        "User", user_ids,
        format_func=lambda value: (
            f"{value} — {profiles.loc[profiles.user_id.astype(str).eq(value), 'home_currency'].iloc[0]} "
            f"{profiles.loc[profiles.user_id.astype(str).eq(value), 'current_available_balance'].iloc[0]:,.2f} available"
        ),
    )
    profile = profiles[profiles.user_id.astype(str).eq(user_id)].iloc[0]
    user_requests = requests[requests.user_id.astype(str).eq(user_id)]
    latest = (pd.to_datetime(user_requests.request_date, errors="coerce").max()
              if not user_requests.empty else pd.Timestamp.today())
    default_date = latest.date() if not pd.isna(latest) else dt.date.today()
    as_of_date = st.date_input("As-of date", value=default_date)
    st.caption(
        f"Minimum balance to keep: {profile.home_currency} "
        f"{float(profile.minimum_balance_to_keep):,.2f}"
    )
    st.info("This is an information tool, not financial advice.")
    deterministic_mode = st.toggle("Deterministic mode (no LLM)", value=False)
    deterministic_amount = None
    deterministic_date = None
    deterministic_partial = False
    if deterministic_mode:
        deterministic_amount = st.number_input("Amount", min_value=0.01, value=100.0)
        deterministic_date = st.date_input("Desired payment date", value=as_of_date)
        deterministic_partial = st.checkbox("Partial payment is acceptable")

assistant_key = (user_id, as_of_date.isoformat(), deterministic_mode,
                deterministic_amount, str(deterministic_date), deterministic_partial)
if st.session_state.get("assistant_key") != assistant_key:
    st.session_state.assistant_key = assistant_key
    st.session_state.assistant = Assistant(
        user_id, as_of_date.isoformat(), no_llm=deterministic_mode,
        deterministic_amount=deterministic_amount,
        desired_date=deterministic_date.isoformat() if deterministic_date else None,
        partial=deterministic_partial,
    )
    st.session_state.chat = []
    st.session_state.latest_decision = None

left, right = st.columns([3, 2])
with left:
    st.subheader("Chat")
    for item in st.session_state.chat:
        with st.chat_message(item["role"]):
            st.write(item["content"])
    prompt = st.chat_input("Ask about a planned payment")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.chat_message("assistant"):
                with st.spinner("Reviewing the forecast..."):
                    reply, structured = st.session_state.assistant.ask(prompt)
                st.write(reply)
                if structured.get("model_unavailable"):
                    st.warning("The assistant model is unavailable. Select deterministic mode in the sidebar to continue without it.")
                st.session_state.chat.append({"role": "assistant", "content": reply})
                if not structured.get("clarification"):
                    st.session_state.latest_decision = structured
        except Exception as exc:
            st.error("I couldn't complete that request. You can continue in deterministic mode using the sidebar controls.")
            st.session_state.last_error = str(exc)

with right:
    st.subheader("Decision details")
    latest_decision = st.session_state.get("latest_decision")
    if not latest_decision:
        st.caption("A decision summary will appear here after your first complete request.")
    else:
        row = latest_decision["decision"]
        evidence = latest_decision["evidence"]
        currency = latest_decision["decision"].get("currency") or profile.home_currency
        status = row["affordability_status"]
        if status == "not_affordable":
            st.error(status.replace("_", " ").title())
        elif status == "affordable_later":
            st.warning(status.replace("_", " ").title())
        else:
            st.success(status.replace("_", " ").title())
        st.metric("Amount safe to pay now", f"{currency} {float(row['amount_safe_to_pay']):,.2f}")
        st.write("**Payment plan**")
        plan = row.get("payment_plan", "none")
        if plan == "none":
            st.write("No safe plan found.")
        else:
            plan_rows = []
            for part in plan.split("|"):
                day, amount = part.split(":", 1)
                plan_rows.append({"Date": day, "Amount": f"{currency} {float(amount):,.2f}"})
            st.dataframe(pd.DataFrame(plan_rows), hide_index=True, use_container_width=True)
        st.write("**Earliest full-payment date:**", row.get("earliest_date_for_full_payment") or "No safe date in forecast without changes")
        st.write("**Spending changes:**", row.get("spending_changes_needed", "none"))

        balances = evidence.get("balances", {})
        if balances:
            chart = pd.DataFrame({"date": list(balances), "Projected balance": list(balances.values())})
            minimum = float(evidence["minimum_balance"])
            chart["Minimum balance"] = minimum
            payment_dates = set(evidence.get("plan", {}))
            chart["Payment date"] = [value if day in payment_dates else None
                                     for day, value in zip(chart.date, chart["Projected balance"])]
            try:
                import altair as alt
                base = alt.Chart(chart).encode(x=alt.X("date:T", title="Date"))
                lines = base.transform_fold(
                    ["Projected balance", "Minimum balance"],
                    as_=["series", "balance"],
                ).mark_line().encode(y=alt.Y("balance:Q", title=f"Balance ({currency})"), color="series:N")
                points = base.mark_point(color="red", size=70, filled=True).encode(
                    y=alt.Y("Payment date:Q"), tooltip=["date:T", "Payment date:Q"]
                )
                st.altair_chart(lines + points, use_container_width=True)
            except ImportError:
                st.line_chart(chart.set_index("date")[["Projected balance", "Minimum balance"]])

        with st.expander("Evidence"):
            breakdown = evidence.get("breakdown", {})
            st.write("Largest forecast outflows", breakdown.get("largest_outflow_drivers", []))
            st.write("Retrieved message IDs", latest_decision.get("retrieved_messages", []))
            st.write("Income counted", latest_decision.get("evidence", {}).get("explanation_evidence", {}).get("income_counted", []))
