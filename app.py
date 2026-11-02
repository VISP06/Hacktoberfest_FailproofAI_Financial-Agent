"""Chat-first deterministic payment advisor."""
import datetime as dt
import html
import os
import re
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


def money(value, currency):
    """Format money compactly at 10M+, consistently throughout the UI."""
    amount = float(value)
    if abs(amount) >= 10_000_000:
        compact = f"{amount / 1_000_000:.1f} M"
    else:
        compact = f"{amount:,.2f}"
    return f"{currency} {compact}"


def amount_from_text(text):
    matches = re.findall(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w.])", text or "")
    for raw in matches:
        try:
            amount = float(raw.replace(",", ""))
            if amount > 0:
                return amount
        except ValueError:
            continue
    return None


def _date_override(text, as_of):
    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if iso:
        try:
            return dt.date.fromisoformat(iso.group(1))
        except ValueError:
            pass
    if re.search(r"\b(next month|in a month|wait a month)\b", text, re.I):
        month = as_of.month % 12 + 1
        year = as_of.year + (as_of.month == 12)
        return dt.date(year, month, min(as_of.day, 28))
    return None


def _tile(label, value, currency, exact_value=None):
    if isinstance(value, str):
        return (f'<div class="stat-tile"><div class="stat-label">{html.escape(label)}</div>'
                f'<div class="stat-number">{html.escape(value)}</div></div>')
    formatted = money(value, currency)
    exact_amount = float(exact_value if exact_value is not None else value)
    exact = f"{currency} {exact_amount:,.2f}"
    return (f'<div class="stat-tile" title="{html.escape(exact)}">'
            f'<div class="stat-label">{html.escape(label)}</div>'
            f'<div class="stat-currency">{html.escape(currency)}</div>'
            f'<div class="stat-number">{html.escape(formatted[len(currency)+1:])}</div>'
            f'<div class="stat-exact">{html.escape(exact)}</div></div>')


st.set_page_config(page_title="Buy or Wait?", page_icon=":material/account_balance_wallet:", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Lato:wght@400;700&family=Oswald:wght@500;600&display=swap');
:root{--cream:#FBE9D0;--cream-card:#FFF4E5;--teal:#244855;--red:#E64833;--brown:#874F41;--sage:#90AEAD;--sage-light:#E5EFEC;--brown-light:#F0DFD8;--red-light:#F9DED7;--radius:22px}
html,body,[class*="css"]{font-family:'Lato',Arial,sans-serif;color:var(--teal);background:var(--cream)}
.block-container{max-width:1280px;padding-top:1rem}
[data-testid="stSidebar"]{background:var(--sage-light)}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--radius);background:var(--cream-card);box-shadow:0 1px 3px #24485520;padding:1rem}
h1,h2,h3,.stat-number,.status-chip{font-family:'Bebas Neue','Oswald',Impact,sans-serif;text-transform:uppercase;letter-spacing:.04em;color:var(--teal)}
h1{font-size:2.4rem}h2{font-size:1.7rem}h3{font-size:1.4rem}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:12px 0 22px}
.stat-tile{box-sizing:border-box;min-width:0;overflow:visible;white-space:normal;background:#FFF8EF;border:1px solid #90AEAD80;border-radius:20px;padding:15px 18px;box-shadow:0 1px 3px #24485520}
.stat-label{font-size:.9rem;color:var(--brown);overflow-wrap:anywhere}.stat-currency{font-size:.78rem;font-weight:700;color:var(--brown);margin-top:8px}
.stat-number{font-size:clamp(1.3rem,2.2vw,2rem);line-height:1.05;overflow-wrap:anywhere}.stat-exact{font-size:.72rem;color:var(--brown);margin-top:5px;overflow-wrap:anywhere}
.status-chip{display:inline-block;padding:5px 12px;border-radius:999px;background:var(--sage-light);color:var(--teal);font-size:1rem}
.advice-card{border-radius:22px;background:var(--cream-card);box-shadow:0 1px 4px #24485525;padding:18px;margin:8px 0 14px}
.advice-card.affordable{background:var(--sage-light)}.advice-card.later{background:var(--brown-light)}
.advice-card.unaffordable{background:var(--red-light);border-left:5px solid var(--red)}
div.stButton>button,button[kind="primary"]{background:var(--teal);color:var(--cream);border:1px solid var(--teal);border-radius:999px;min-height:42px;font-weight:700;transition:.15s}
div.stButton>button:hover,button[kind="primary"]:hover{background:#183640;color:white;box-shadow:0 2px 5px #24485540}
div.stButton>button:focus-visible,button[kind="primary"]:focus-visible{outline:3px solid var(--red);outline-offset:2px}
[data-testid="stChatMessage"]{border-radius:20px;background:var(--cream-card);box-shadow:0 1px 3px #2448551c}
small,.secondary{color:var(--brown)}
</style>""", unsafe_allow_html=True)

st.markdown("# :material/account_balance_wallet: Buy or Wait?")
st.caption("Grounded payment decisions, with your next 90 days in view.")

with st.sidebar:
    st.markdown("### :material/person: Your profile")
    user_ids = profiles.user_id.astype(str).tolist()
    user_id = st.selectbox("User", user_ids,
                           format_func=lambda value: f"{value} · {profiles.loc[profiles.user_id.astype(str).eq(value), 'home_currency'].iloc[0]}")
    profile = profiles[profiles.user_id.astype(str).eq(user_id)].iloc[0]
    user_events = events[events.user_id.astype(str).eq(user_id)]
    user_requests = requests[requests.user_id.astype(str).eq(user_id)]
    user_dates = pd.concat([pd.to_datetime(user_events.event_date, errors="coerce"),
                            pd.to_datetime(user_requests.request_date, errors="coerce")]).dropna()
    low_date = user_dates.min().date() if not user_dates.empty else dt.date.today()
    high_date = user_dates.max().date() if not user_dates.empty else dt.date.today()
    default_date = high_date
    as_of_date = st.date_input("As-of date", value=default_date, min_value=low_date, max_value=high_date)
    st.caption(f"Data range: {low_date:%d %b %Y} – {high_date:%d %b %Y}.")
    st.markdown("### :material/tune: Assumptions (optional)")
    no_deadline = st.checkbox("No deadline", value=True)
    pay_by_date = None if no_deadline else st.date_input("Pay by date", value=as_of_date)
    allow_partial = st.toggle("Partial payment", value=False)
    allow_installments = st.toggle("Installments", value=False)
    installment_count = st.selectbox("Installment count", [3, 6, 12], disabled=not allow_installments)
    installment_fee_pct = st.number_input("Optional fee (%)", min_value=0.0, value=0.0,
                                          step=0.5, disabled=not allow_installments)
    st.caption("This is an information tool, not financial advice.")

profile_key = (user_id, as_of_date.isoformat())
if st.session_state.get("assistant_key") != profile_key:
    st.session_state.assistant_key = profile_key
    st.session_state.assistant = Assistant(user_id, as_of_date.isoformat(), dataset_dir=DATASET)
    st.session_state.situation = st.session_state.assistant.situation()
    st.session_state.chat = []
    st.session_state.advice = None

assistant = st.session_state.assistant
situation = st.session_state.situation
api_key_set = any(os.getenv(key) for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"))
st.sidebar.caption(":material/check_circle: AI active" if api_key_set else ":material/info: AI unavailable · deterministic advice is active")

currency = str(profile.home_currency)
st.markdown("### :material/monitoring: Your situation")
tiles = [
    _tile("Available", situation["current_balance"], currency),
    _tile("Minimum to keep", situation["minimum_balance"], currency),
    _tile("Lowest projected balance", situation["lowest_projected_balance"], currency),
    _tile("Expected monthly income",
          money(situation["monthly_income_total"], currency)
          if situation["monthly_income_counted"] != "No recurring income counted"
          else "No recurring income counted", currency),
]
st.markdown('<div class="stats-grid">' + "".join(tiles) + "</div>", unsafe_allow_html=True)

chat_col, detail_col = st.columns([1.45, 1])
with chat_col:
    st.markdown("### :material/chat: Your payment plan")
    if not st.session_state.chat:
        st.caption("Try one of these")
        suggestions = st.columns(3)
        prompts = ["Can I afford a 12,000 laptop?", "Should I use installments?", "What if I wait a month?"]
        selected_prompt = None
        for col, prompt in zip(suggestions, prompts):
            if col.button(prompt, key="suggest_" + str(prompts.index(prompt))):
                selected_prompt = prompt
    else:
        selected_prompt = None
    typed_prompt = st.chat_input("Ask about a payment")
    message = typed_prompt or selected_prompt
    if message:
        st.session_state.chat.append({"role": "user", "content": message})
        parsed_amount = amount_from_text(message)
        if parsed_amount is None:
            reply = "What amount are you considering?"
            st.session_state.chat.append({"role": "assistant", "content": reply,
                                          "clarification": True})
        else:
            message_lower = message.lower()
            chosen_date = (None if re.search(r"\b(no deadline|no due date)\b", message_lower)
                           else _date_override(message, as_of_date) or pay_by_date)
            partial_no = any(phrase in message_lower for phrase in
                             ("no partial", "not partial", "full payment only", "pay in full"))
            partial_mentioned = partial_no or any(
                phrase in message_lower for phrase in ("partial", "half now", "pay half"))
            partial_value = (not partial_no) if partial_mentioned else allow_partial
            installments_no = any(phrase in message_lower for phrase in
                                  ("no installments", "not installments", "without installments"))
            installments_mentioned = not installments_no and "installment" in message_lower
            count_match = re.search(r"\b(3|three|6|six|12|twelve)\s+installments?\b", message_lower)
            counts = {"three": 3, "six": 6, "twelve": 12}
            terms_count = counts.get(count_match.group(1), None) if count_match else None
            if count_match and terms_count is None:
                terms_count = int(count_match.group(1))
            fee_match = re.search(r"(\d+(?:\.\d+)?)\s*%", message_lower)
            form = {"amount": parsed_amount, "currency": currency,
                    "item": None, "pay_by_date": chosen_date,
                    "allow_partial": partial_value,
                    "allow_installments": (False if installments_no else
                                           True if installments_mentioned else allow_installments),
                    "installment_count": terms_count or installment_count,
                    "installment_fee_pct": float(fee_match.group(1)) if fee_match else installment_fee_pct}
            try:
                with st.spinner("Checking the forecast…"):
                    result = assistant.advise(form)
                if api_key_set:
                    try:
                        result["rationale"] = assistant._reply(
                            message, {"row": result["decision"], "evidence": result["evidence"]}, [])
                        result["model_unavailable"] = assistant.model_unavailable
                    except Exception:
                        result["model_unavailable"] = True
                else:
                    result["model_unavailable"] = True
                st.session_state.advice = result
                st.session_state.chat.append({"role": "assistant", "content": result["headline"],
                                              "advice": result})
            except Exception as exc:
                st.session_state.chat.append({"role": "assistant", "content": f"I couldn't calculate that plan: {exc}"})
        st.rerun()

    for entry in st.session_state.chat:
        with st.chat_message(entry["role"]):
            if entry.get("clarification"):
                st.write(entry["content"])
            elif entry.get("advice"):
                result = entry["advice"]
                if result.get("model_unavailable"):
                    st.caption(":material/info: AI wording unavailable; showing deterministic advice.")
                chosen = result.get("recommended_option")
                status = result["decision"].get("affordability_status", "not_affordable")
                status_class = ("affordable" if status == "affordable_now" else
                                "unaffordable" if status == "not_affordable" or not chosen or not chosen.get("safe") else "later")
                rationale = result.get("rationale") or result["decision"]["decision_explanation"]
                st.markdown(f'<div class="advice-card {status_class}">'
                            f'<span class="status-chip">{html.escape(status.replace("_", " "))}</span>'
                            f'<h3>{html.escape(result["headline"])}</h3><p>{html.escape(rationale)}</p></div>',
                            unsafe_allow_html=True)
                if chosen:
                    st.dataframe(pd.DataFrame([{"Date": day, "Amount": money(value, currency)}
                                              for day, value in chosen["amounts"].items()]),
                                 hide_index=True, use_container_width=True)
                opts = result.get("ranked_options", [])
                if opts:
                    st.markdown("**Compared options**")
                    st.dataframe(pd.DataFrame([{
                        "Option": option["type"].replace("_", " ").title(),
                        "Plan": " · ".join(f"{day}: {money(value, currency)}" for day, value in option["amounts"].items()),
                        "Total cost": money(option["total_cost"], currency),
                        "Lowest balance": money(option["lowest_projected_balance"], currency),
                        "Deadline": "—" if option["meets_deadline"] is None else ("Yes" if option["meets_deadline"] else "No")}
                        for option in opts]), hide_index=True, use_container_width=True)
            else:
                st.write(entry["content"])

with detail_col:
    st.markdown("### :material/insights: Decision details")
    current_advice = st.session_state.get("advice")
    evidence = current_advice["evidence"] if current_advice else situation
    balances = evidence.get("balances", {})
    minimum = evidence.get("minimum_balance", situation["minimum_balance"])
    if balances:
        import altair as alt
        chart = pd.DataFrame({"date": pd.to_datetime(list(balances)),
                              "balance": list(balances.values())})
        chart["minimum"] = float(minimum)
        plan = (current_advice or {}).get("recommended_option") or {}
        payment_days = set(plan.get("amounts", {}))
        chart["payment"] = [balances[d] if d in payment_days else None for d in balances]
        base = alt.Chart(chart).encode(x=alt.X("date:T", title=None))
        area = base.mark_area(color="#D5E3DF", opacity=.65).encode(y=alt.Y("balance:Q", title=f"Balance ({currency})"))
        balance_line = base.mark_line(color="#244855", strokeWidth=3).encode(y="balance:Q")
        min_line = base.mark_line(color="#E64833", strokeDash=[7, 5], strokeWidth=2).encode(y="minimum:Q")
        markers = base.mark_point(color="#874F41", filled=True, size=95, shape="diamond").encode(
            y="payment:Q", tooltip=["date:T", "payment:Q"])
        st.altair_chart(area + balance_line + min_line + markers, use_container_width=True)
    situation_cols = st.columns(2)
    situation_cols[0].caption(f"30-day outflows · {money(situation['committed_outflows_next_30_days'], currency)}")
    situation_cols[1].caption(f"90-day outflows · {money(situation['committed_outflows_next_90_days'], currency)}")
    with st.expander("Evidence"):
        st.write("Evidence IDs", (current_advice or {}).get("evidence_ids", []))
        st.write("Pending items", situation.get("pending_items", {}))
        st.write("Income counted", situation.get("monthly_income_counted", "No recurring income counted"))
        st.write("Forecast breakdown", evidence.get("breakdown", {}))
