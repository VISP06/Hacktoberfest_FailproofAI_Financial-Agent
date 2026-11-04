"""Glacier browser terminal for deterministic payment advice."""

import datetime as dt
import os
import sys
import uuid

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from assistant import Assistant
from money import money
from terminal_flow import TerminalSession

APP_VERSION = "glacier-terminal-v2"
DATASET = os.path.join(ROOT, "dataset")
profiles = pd.read_csv(os.path.join(DATASET, "financial_profiles.csv"), dtype={"user_id": str})

st.set_page_config(page_title="Glacier Terminal", page_icon="❄", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root{--bg:#08141F;--panel:#0D2233;--border:#1F4660;--frost:#D8F3FF;--dim:#7FA6BD;--cyan:#5CE1E6;--ice:#8CC8FF;--mint:#7CF5C0;--amber:#FFD479;--coral:#FF7A90}
html,body,[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#08141F 0%,#0D2233 100%) fixed;color:var(--frost);font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,header{visibility:hidden;height:0}
.block-container{max-width:100%;padding:1.2rem 2rem 5rem}.status{color:#7FA6BD;border-bottom:1px solid #1F4660;padding:0 0 1rem;margin-bottom:1.25rem;font-size:.9rem}
.output{background:#0D2233;border:1px solid #1F4660;border-radius:8px;padding:1.2rem;color:#D8F3FF;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.55;min-height:55vh}
.boot{color:#5CE1E6;text-shadow:0 0 10px #5CE1E655}.prompt{color:#5CE1E6}.report{color:#D8F3FF}
div[data-testid="stChatInput"]{border:1px solid #1F4660;background:#0D2233}div[data-testid="stChatInput"] textarea{font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;color:#D8F3FF;caret-color:#5CE1E6}
div.stButton>button,div[data-testid="stDownloadButton"] button{background:#0D2233;color:#5CE1E6;border:1px solid #1F4660;font-family:inherit}
h1,h2,h3{color:#5CE1E6;text-shadow:0 0 10px #5CE1E644;font-family:inherit}
</style>""", unsafe_allow_html=True)

user_ids = profiles.user_id.astype(str).tolist()
user_id = st.session_state.get("terminal_user", "user_05" if "user_05" in user_ids else user_ids[0])
date_key = st.session_state.get("terminal_date")
session_key = f"terminal_session_{APP_VERSION}"
session = st.session_state.get(session_key)
if (not isinstance(session, TerminalSession) or not hasattr(getattr(session, "assistant", None), "situation")
        or session.user_id != user_id or (date_key and session.as_of_date != date_key)):
    if date_key is None:
        assistant = Assistant(user_id, dt.date.today().isoformat(), dataset_dir=DATASET, no_llm=True)
        dates = assistant.engine.events[assistant.engine.events.user_id.astype(str).eq(user_id)].event_date.astype(str)
        date_key = max(dates) if len(dates) else dt.date.today().isoformat()
        st.session_state.terminal_date = date_key
    session = TerminalSession(user_id, date_key, dataset_dir=DATASET,
                              assistant=Assistant(user_id, date_key, dataset_dir=DATASET, no_llm=True))
    st.session_state[session_key] = session
    st.session_state.terminal_output = [{"kind": "text", "plain_text": "glacier terminal ready · type /help for commands"}]

try:
    situation = session.assistant.situation()
    currency = str(session.assistant.engine.get_user_home_currency(user_id))
    status_line = f"glacier > {user_id} | {currency} | balance {money(situation['current_balance'],currency)} | minimum {money(situation['minimum_balance'],currency)} | as-of {session.as_of_date}"
except Exception:
    currency = "INR"
    status_line = f"glacier > {user_id} | {currency} | as-of {session.as_of_date}"
st.markdown(f'<div class="status">{status_line}</div>', unsafe_allow_html=True)
output = st.session_state.get("terminal_output", [])
st.markdown('<div class="output">' + "\n\n".join(str(x["plain_text"]) for x in output) + '</div>', unsafe_allow_html=True)
reports = [x["plain_text"] for x in output if x.get("kind") == "report"]
if reports:
    st.download_button("/export · download plain-text report", reports[-1], file_name="glacier-report.txt", mime="text/plain")
line = st.chat_input(" >  enter a request or command")
if line is not None:
    blocks = session.handle(line)
    output = list(output)
    output.append({"kind": "text", "plain_text": f"> {line}"})
    output.extend(blocks)
    st.session_state.terminal_output = output
    if line.startswith("/user "):
        candidate = line.split(maxsplit=1)[1]
        if candidate in user_ids: st.session_state.terminal_user = candidate
    if line.startswith("/date "):
        try: st.session_state.terminal_date = dt.date.fromisoformat(line.split(maxsplit=1)[1]).isoformat()
        except ValueError: pass
    st.rerun()
