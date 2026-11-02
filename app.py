"""Material-style Streamlit financial planning interface."""
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
events = pd.read_csv(os.path.join(DATASET, "patched_financial_events.csv"), dtype={"user_id": str})

st.set_page_config(page_title="Buy or Wait?", page_icon=":material/account_balance_wallet:", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap');
:root{--surface:#FEF7FF;--low:#F7F2FA;--container:#F3EDF7;--high:#ECE6F0;--primary:#6750A4;--primary-container:#EADDFF;--secondary-container:#E8DEF8;--outline:#CAC4D0;--error-container:#F9DEDC;--success-container:#C8F0D0;--warning-container:#FFE8B3;--text:#1D1B20;--radius:22px}
html,body,[class*="css"]{font-family:Roboto,system-ui,sans-serif;color:var(--text)}
.block-container{max-width:1200px;padding-top:1.3rem}
[data-testid="stMetric"]{background:var(--container);border-radius:18px;padding:16px;box-shadow:0 1px 2px #0002}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--radius);background:var(--low);box-shadow:0 1px 3px #0002;padding:1rem}
div.stButton>button[kind="primary"],button[kind="primary"]{background:var(--primary);border:0;border-radius:999px;min-height:48px;color:white;font-weight:600}
div.stButton>button:not([kind="primary"]){background:var(--secondary-container);border:0;border-radius:999px}
.status-chip{display:inline-block;padding:6px 12px;border-radius:999px;background:var(--primary-container);font-weight:600}
h1,h2,h3{letter-spacing:-.02em} h1{font-size:2rem} h2{font-size:1.5rem} h3{font-size:1.25rem}
</style>""", unsafe_allow_html=True)

st.markdown("## :material/account_balance_wallet: Buy or Wait?")
st.caption("A clear view of what a payment could mean for your next 90 days.")

with st.sidebar:
    st.markdown("### :material/person: Your profile")
    user_ids = profiles.user_id.astype(str).tolist()
    user_id = st.selectbox("User", user_ids, format_func=lambda value: f"{value} · {profiles.loc[profiles.user_id.astype(str).eq(value), 'home_currency'].iloc[0]}")
    profile = profiles[profiles.user_id.astype(str).eq(user_id)].iloc[0]
    user_events = events[events.user_id.astype(str).eq(user_id)]
    user_requests = requests[requests.user_id.astype(str).eq(user_id)]
    dates = pd.concat([pd.to_datetime(user_events.event_date, errors="coerce"), pd.to_datetime(user_requests.request_date, errors="coerce")]).dropna()
    low_date = dates.min().date() if not dates.empty else dt.date.today()
    high_date = dates.max().date() if not dates.empty else dt.date.today()
    latest = max(pd.to_datetime(user_requests.request_date, errors="coerce").max() if not user_requests.empty else pd.NaT,
                 pd.to_datetime(user_events.event_date, errors="coerce").max() if not user_events.empty else pd.NaT)
    default_date = latest.date() if not pd.isna(latest) else high_date
    as_of_date = st.date_input("As-of date", value=default_date, min_value=low_date, max_value=high_date)
    st.caption(f"Data range: {low_date:%d %b %Y} – {high_date:%d %b %Y}.")
    st.caption("This is an information tool, not financial advice.")

profile_key = (user_id, as_of_date.isoformat())
if st.session_state.get("assistant_key") != profile_key:
    st.session_state.assistant_key = profile_key
    st.session_state.assistant = Assistant(user_id, as_of_date.isoformat(), dataset_dir=DATASET)
    st.session_state.chat = []
    st.session_state.advice = None

assistant = st.session_state.assistant
advice = st.session_state.get("advice")
situation = advice.get("situation", {}) if advice else {}
st.markdown("### :material/monitoring: Your situation")
tiles = st.columns(4)
tile_data = [
    ("Available", f"{profile.home_currency} {float(profile.current_available_balance):,.0f}"),
    ("Minimum to keep", f"{profile.home_currency} {float(profile.minimum_balance_to_keep):,.0f}"),
    ("Lowest projected balance", f"{profile.home_currency} {float(situation.get('lowest_projected_balance', profile.current_available_balance)):,.0f}"),
    ("Expected monthly income", f"{profile.home_currency} {float(situation.get('monthly_income_total', 0)):,.0f}" if advice else "Run a forecast"),
]
for col, (label, value) in zip(tiles, tile_data):
    col.metric(label, value)

st.markdown("### :material/payments: Plan a payment")
with st.container(border=True):
    with st.form("payment_form"):
        a, b, c = st.columns([1, 1, 2])
        amount = a.number_input("Amount", min_value=0.01, value=100.0, step=10.0)
        currency = b.selectbox("Currency", [profile.home_currency])
        item = c.text_input("What is it for? (optional)")
        no_deadline = st.checkbox("No deadline", value=True)
        pay_by_date = None if no_deadline else st.date_input("Pay by date", value=as_of_date)
        p, i, n = st.columns([1, 1, 1])
        allow_partial = p.toggle("Partial payment", value=False)
        allow_installments = i.toggle("Installments", value=False)
        installment_count = n.selectbox("Installment count", [3, 6, 12], disabled=not allow_installments)
        installment_fee_pct = st.number_input("Installment fee (%)", min_value=0.0, value=0.0, step=0.5, disabled=not allow_installments)
        has_key = any(os.getenv(k) for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"))
        use_ai = st.toggle("Use AI wording", value=has_key)
        submit = st.form_submit_button("Get advice", type="primary", icon=":material/auto_awesome:", use_container_width=True)
        if submit:
            form = {"amount": amount, "currency": currency, "item": item or None,
                    "pay_by_date": pay_by_date, "allow_partial": allow_partial,
                    "allow_installments": allow_installments, "installment_count": installment_count,
                    "installment_fee_pct": installment_fee_pct}
            try:
                with st.spinner("Checking your forecast…"):
                    result = assistant.advise(form)
                if use_ai:
                    try:
                        # The plan is complete before any optional wording call.
                        decision = {"row": result["decision"], "evidence": result["evidence"]}
                        result["rationale"] = assistant._reply("Explain this recommendation", decision, [])
                        result["model_unavailable"] = assistant.model_unavailable
                    except Exception:
                        result["model_unavailable"] = True
                        result["rationale"] = result["decision"]["decision_explanation"]
                st.session_state.advice = result
            except Exception as exc:
                st.error(f"Could not calculate the plan: {exc}")

advice = st.session_state.get("advice")
if advice:
    chosen = advice.get("recommended_option")
    tint = "success" if chosen and chosen["safe"] else "warning"
    st.markdown("### :material/lightbulb: Recommendation")
    if advice.get("model_unavailable"):
        st.info("AI wording is unavailable; showing the deterministic recommendation.")
    (st.success if tint == "success" else st.warning)(f"**{advice['headline']}**\n\n{advice.get('rationale') or advice['decision']['decision_explanation']}")
    st.markdown("### :material/compare_arrows: Compare options")
    options = advice.get("ranked_options", [])
    if options:
        st.dataframe(pd.DataFrame([{"Option": o["type"].replace("_", " ").title(),
                                    "Dates / amounts": " · ".join(f"{d}: {v:,.2f}" for d,v in o["amounts"].items()),
                                    "Total cost": f"{o['total_cost']:,.2f}",
                                    "Lowest balance": f"{o['lowest_projected_balance']:,.2f}",
                                    "Meets deadline": "—" if o["meets_deadline"] is None else ("Yes" if o["meets_deadline"] else "No")}
                                   for o in options]), hide_index=True, use_container_width=True)
    st.markdown("### :material/event_note: Plan details")
    st.dataframe(pd.DataFrame([{"Date": day, "Amount": f"{value:,.2f}"}
                              for day, value in (chosen or {}).get("amounts", {}).items()]), hide_index=True, use_container_width=True)
    evidence = advice.get("evidence", {})
    balances = evidence.get("balances", {})
    if balances:
        st.markdown("### :material/show_chart: Projected balance")
        chart = pd.DataFrame({"Date": pd.to_datetime(list(balances)), "Balance": list(balances.values())}).set_index("Date")
        chart["Minimum balance"] = float(evidence["minimum_balance"])
        payment_dates = set((chosen or {}).get("amounts", {}))
        chart["Payment marker"] = [float(balances[d]) if d in payment_dates else None for d in balances]
        st.line_chart(chart, color=["#6750A4", "#CAC4D0", "#E8DEF8"])
    st.markdown(f"Committed outflows · 30 days **{advice['situation']['committed_outflows_next_30_days']:,.2f}** · 90 days **{advice['situation']['committed_outflows_next_90_days']:,.2f}**")
    with st.expander("Evidence"):
        st.write("Evidence IDs", advice.get("evidence_ids", []))
        st.write("Breakdown", evidence.get("breakdown", {}))

st.markdown("### :material/chat: Ask a follow-up")
with st.container(border=True):
    for item in st.session_state.chat:
        with st.chat_message(item["role"]):
            st.write(item["content"])
    followup = st.chat_input("For example: what if I pay half now?")
    if followup:
        st.session_state.chat.append({"role": "user", "content": followup})
        current = st.session_state.get("advice")
        if current:
            reply = "The current recommendation is based on the form values above. Update the payment form and select Get advice to compare a different amount or schedule."
        else:
            reply = "Enter an amount in Plan a payment, then select Get advice."
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()
