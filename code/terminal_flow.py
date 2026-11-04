"""Deterministic conversational terminal flow; deliberately has no Streamlit dependency."""

import datetime as dt
import os
import re
import uuid

import failproofai_sdk as fp

from assistant import Assistant
from money import money


_CURRENCIES = {"INR": "INR", "USD": "USD", "EUR": "EUR", "ZAR": "ZAR", "IDR": "IDR"}
_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "R": "ZAR"}
_AMOUNT_RE = re.compile(r"(?<![\w.])(?:\d+(?:[, ]\d{3})+|\d+(?:\.\d+)?)(?![\d.])", re.I)


def parse_amount(text):
    """Return the first positive amount and explicit currency, if supplied."""
    value = str(text or "")
    currency = None
    for code in _CURRENCIES:
        if re.search(rf"\b{code}\b", value, re.I):
            currency = code
            break
    for symbol, code in _SYMBOLS.items():
        found = (symbol in value if symbol != "R" else bool(re.search(r"(?<!\w)R(?=\s*\d)", value, re.I)))
        if found:
            currency = code
            break
    for match in _AMOUNT_RE.finditer(value):
        raw = match.group(0).replace(",", "").replace(" ", "")
        try:
            amount = float(raw)
        except ValueError:
            continue
        suffix = value[match.end():]
        prefix = value[:match.start()]
        unit = re.match(r"\s*(k|m|million|lakh|lakhs|crore|crores)(?![a-z])", suffix, re.I)
        if unit:
            u = unit.group(1).lower()
            amount *= 1_000 if u == "k" else 1_000_000 if u in {"m", "million"} else 100_000 if u in {"lakh", "lakhs"} else 10_000_000
        elif re.search(r"(?:\d)\s*$", prefix) and " " in prefix[-2:]:
            continue
        if amount > 0:
            return amount, currency
    return None, currency


class TerminalSession:
    def __init__(self, user_id="user_05", as_of_date=None, dataset_dir=None, assistant=None):
        self.dataset_dir = dataset_dir or os.path.join(os.path.dirname(__file__), "..", "dataset")
        self.assistant = assistant
        if assistant is None:
            assistant = Assistant(user_id, as_of_date or self._latest_date(user_id),
                                  dataset_dir=self.dataset_dir, no_llm=True)
        self.assistant = assistant
        self.user_id = str(user_id)
        self.as_of_date = str(as_of_date or assistant.as_of_date)[:10]
        if self.user_id not in self.assistant.engine.profiles.index.astype(str):
            raise ValueError(f"Unknown user id: {self.user_id}")
        self.run_id = assistant.run_id
        self.session_id = f"chat-{self.run_id}"
        self.phase = "ready"
        self.request_text = ""
        self.form = {}
        self.questions = []
        self.question_index = 0
        self.invalid_attempts = 0
        self.report = None
        self.last_blocks = []

    def _latest_date(self, user_id):
        import pandas as pd
        engine = Assistant(user_id, dt.date.today().isoformat(), dataset_dir=self.dataset_dir, no_llm=True).engine
        dates = pd.to_datetime(engine.events[engine.events.user_id.astype(str).eq(str(user_id))].event_date, errors="coerce").dropna()
        return dates.max().date().isoformat() if not dates.empty else dt.date.today().isoformat()

    def _block(self, kind, text):
        block = {"kind": kind, "text": str(text), "plain_text": str(text)}
        self.last_blocks.append(block)
        return block

    def _prompt(self):
        question = self.questions[self.question_index]
        options = {
            "deadline": "When do you need to pay? [1 none (default), 2 within 14 days, 3 end of this month, 4 type YYYY-MM-DD]",
            "partial": "Is partial payment acceptable? [1 no (default), 2 yes]",
            "installments": "Use installments? [1 no (default), 2 3 months, 3 6 months, 4 12 months]",
            "fee": "Optional installment fee percentage? [Enter for 0%]",
        }
        return self._block("prompt", options[question])

    def _start(self, line):
        amount, currency = parse_amount(line)
        if amount is None:
            return [self._block("prompt", "What amount are you considering?")]
        self.request_text = line
        home = str(self.assistant.engine.get_user_home_currency(self.user_id))
        self.form = {"amount": amount, "currency": currency or home,
                     "item": re.sub(r"\b(?:can i afford|should i buy|buy|costs?)\b", "", line, flags=re.I).strip(" ?.!$") or "payment",
                     "allow_partial": False, "allow_installments": False,
                     "installment_count": 3, "installment_fee_pct": 0.0}
        lowered = line.lower()
        self.questions = []
        if not self._parse_deadline(line):
            self.questions.append("deadline")
        if re.search(r"pay half now|partial payment", lowered):
            self.form["allow_partial"] = True
        elif not re.search(r"\b(no partial|pay in full|full payment only)\b", lowered):
            self.questions.append("partial")
        count = re.search(r"\b(\d+)\s+installments?\b", lowered)
        if count:
            self.form.update(allow_installments=True, installment_count=int(count.group(1)))
        elif "installment" not in lowered:
            self.questions.append("installments")
        else:
            self.form.update(allow_installments=True)
            self.questions.append("fee")
        self.phase = "intake"
        self.question_index = 0
        if re.search(r"\bjust advise\b", lowered):
            self.questions = []
        if self.questions:
            return [self._prompt()]
        return self._make_report()

    def _parse_deadline(self, text):
        if re.search(r"\b(no deadline|no due date|just advise)\b", text, re.I):
            self.form["pay_by_date"] = None
            return True
        iso = re.search(r"\b(20\d\d-\d\d-\d\d)\b", text)
        if iso:
            try:
                self.form["pay_by_date"] = dt.date.fromisoformat(iso.group(1)).isoformat()
                return True
            except ValueError:
                pass
        if re.search(r"\bwithin 14 days\b", text, re.I):
            self.form["pay_by_date"] = (dt.date.fromisoformat(self.as_of_date) + dt.timedelta(days=14)).isoformat()
            return True
        if re.search(r"\bend of this month\b", text, re.I):
            day = dt.date.fromisoformat(self.as_of_date)
            next_month = day.replace(day=28) + dt.timedelta(days=4)
            self.form["pay_by_date"] = (next_month.replace(day=1) - dt.timedelta(days=1)).isoformat()
            return True
        return False

    def _accept_answer(self, line):
        key = self.questions[self.question_index]
        value = line.strip()
        choices = {"deadline": {"1": None, "2": "14", "3": "month", "4": "date"},
                   "partial": {"1": False, "2": True},
                   "installments": {"1": False, "2": 3, "3": 6, "4": 12}}
        valid = value == "" or (key in choices and value in choices[key]) or (key == "fee" and bool(re.fullmatch(r"\d+(?:\.\d+)?", value)))
        if key == "date" and value:
            try:
                dt.date.fromisoformat(value)
                valid = True
            except ValueError:
                valid = False
        if not valid:
            self.invalid_attempts += 1
            if self.invalid_attempts == 1:
                return [self._prompt()]
            value = ""
        self.invalid_attempts = 0
        if key == "deadline":
            if value == "2":
                self.form["pay_by_date"] = (dt.date.fromisoformat(self.as_of_date) + dt.timedelta(days=14)).isoformat()
            elif value == "3":
                d = dt.date.fromisoformat(self.as_of_date); n = d.replace(day=28) + dt.timedelta(days=4)
                self.form["pay_by_date"] = (n.replace(day=1) - dt.timedelta(days=1)).isoformat()
            elif value == "4":
                self.questions[self.question_index] = "date"
                return [self._block("prompt", "Type a date in YYYY-MM-DD format.")]
            else:
                self.form["pay_by_date"] = None
        elif key == "date":
            self.form["pay_by_date"] = dt.date.fromisoformat(value).isoformat() if value else None
        elif key == "partial": self.form["allow_partial"] = value == "2"
        elif key == "installments":
            self.form["allow_installments"] = value in {"2", "3", "4"}
            if value in {"2", "3", "4"}: self.form["installment_count"] = choices[key][value]
            if self.form["allow_installments"]: self.questions.insert(self.question_index + 1, "fee")
        elif key == "fee": self.form["installment_fee_pct"] = float(value) if value else 0.0
        self.question_index += 1
        if self.question_index < len(self.questions): return [self._prompt()]
        return self._make_report()

    def _make_report(self):
        self.phase = "reported"
        self.form.setdefault("pay_by_date", None)
        try:
            with fp.session(self.session_id):
                with fp.agent("assistant", goal="Produce deterministic payment advice"):
                    with fp.tool_call("advise", input={"user_id": self.user_id, "amount": self.form["amount"]}) as tool:
                        result = self.assistant.advise(self.form)
                        tool.output = result
            self.report = result
            text = self.render_report(result)
            return [self._block("report", text)]
        except Exception as exc:
            fp.event.error(session_id=self.session_id, agent_id="assistant", error_type=type(exc).__name__, message=str(exc)[:300])
            return [self._block("error", f"Unable to produce advice: {exc}")]

    def render_report(self, result):
        currency = self.form["currency"]
        options = result.get("ranked_options", [])
        chosen = result.get("recommended_option")
        if chosen is None: verdict = "NOT RECOMMENDED"
        elif chosen["type"] == "pay_in_full_now": verdict = "PAY IN FULL TODAY"
        elif chosen["type"] == "installments": verdict = f"PAY IN {self.form.get('installment_count', 3)} INSTALLMENTS"
        elif chosen["type"] == "partial_then_remainder": verdict = f"PAY {money(next(iter(chosen['amounts'].values())), currency)} NOW AND THE REST ON {list(chosen['amounts'])[-1]}"
        elif chosen["type"] == "wait": verdict = f"WAIT UNTIL {max(chosen['dates'])}"
        else: verdict = "NOT RECOMMENDED"
        situation = result["situation"]
        why = (f"The current balance is {money(situation['current_balance'], currency)}, and the minimum to keep is {money(situation['minimum_balance'], currency)}. "
               f"The lowest projected balance is {money(situation['lowest_projected_balance'], currency)} on {situation['lowest_projected_date']}. "
               f"Monthly counted income is {money(situation['monthly_income_total'], currency)}; only validated recurring income is counted. "
                f"Pending items considered or ignored: {', '.join(map(str, result.get('evidence_ids', []))) or 'none' }.")
        pending = situation.get("pending_items_considered_or_ignored", {})
        why += (f" Pending items considered: {', '.join(map(str, pending.get('considered', []))) or 'none'}; "
                f"ignored: {', '.join(map(str, pending.get('ignored', []))) or 'none'}.")
        balances = result.get("evidence", {}).get("balances", {})
        vals = list(balances.values())
        levels = "▁▂▃▄▅▆▇█"
        spark = "" if not vals else "".join(levels[round((v-min(vals))/(max(vals)-min(vals))*7)] if max(vals) != min(vals) else "▄" for v in vals)
        minimum = situation["minimum_balance"]
        rows = ["OPTION                 WHEN                 TOTAL COST       LOWEST BALANCE   MEETS DEADLINE"]
        for option in options:
            rows.append(f"{option['type'].replace('_',' '):22} {', '.join(option['dates']):20} {money(option['total_cost'],currency):16} {money(option['lowest_projected_balance'],currency):16} {option['meets_deadline']}")
        alts = [f"Best other options: {', '.join(x['type'].replace('_',' ') for x in options[1:]) or 'none' }.",
                f"Earliest safe date to wait for: {situation.get('earliest_date_for_full_payment') or 'not available'}.",
                f"Amount safe to pay today: {money(result['decision'].get('amount_safe_to_pay', 0),currency)}.",
                f"Spending changes that would free cash: {result['decision'].get('spending_changes_needed', 'none')}."]
        steps = [f"Confirm the selected plan for {money(self.form['amount'],currency)}.", "Keep the minimum balance available.", "Recheck the forecast before making the payment."]
        return (f"╭─ {verdict} ─╮\nWHY\n{why}\n\nNUMBERS\n90-day balance: {spark}\nMinimum-balance level: {money(minimum,currency)}\n\nOPTIONS\n" + "\n".join(rows) + "\n\nALTERNATIVES\n" + "\n".join("• "+s for s in alts) + "\n\nNEXT STEPS\n" + "\n".join("• "+s for s in steps))

    def handle(self, line):
        self.last_blocks = []
        text = str(line or "").strip()
        command = text.lower()
        if command.startswith("/help"):
            self._block("text", "/user <id> · /date YYYY-MM-DD · /situation · /reset · /export · /help · /skip")
        elif command == "/export": self._block("text", self.report or "No report to export yet.")
        elif command == "/situation":
            s = self.assistant.situation(); c = self.assistant.engine.get_user_home_currency(self.user_id)
            self._block("text", f"Available {money(s['current_balance'],c)}; minimum {money(s['minimum_balance'],c)}; as of {self.as_of_date}.")
        elif command == "/reset":
            self.phase = "ready"; self.form = {}; self.questions = []; self.report = None
            self._block("text", "Session reset. What are you considering?")
        elif command.startswith("/user "):
            new_user = text.split(maxsplit=1)[1]
            if new_user not in self.assistant.engine.profiles.index.astype(str): self._block("error", f"Unknown user id: {new_user}")
            else:
                self.__init__(new_user, self.as_of_date, self.dataset_dir)
                self._block("text", f"User set to {new_user}.")
        elif command.startswith("/date "):
            value = text.split(maxsplit=1)[1]
            try:
                date = dt.date.fromisoformat(value)
                all_dates = self.assistant.engine.events[self.assistant.engine.events.user_id.astype(str).eq(self.user_id)].event_date.dropna().astype(str)
                lo, hi = sorted(map(str, all_dates))[0], sorted(map(str, all_dates))[-1]
                if not lo <= value <= hi: raise ValueError
                self.__init__(self.user_id, date.isoformat(), self.dataset_dir)
                self._block("text", f"As-of date set to {value}.")
            except Exception: self._block("error", "Invalid date or date outside this user's data range.")
        elif self.phase == "intake" and command == "/skip":
            self.question_index = len(self.questions)
            self._make_report()
        elif self.phase == "intake": self._accept_answer(text)
        elif self.phase == "reported":
            if re.search(r"\bhalf\b", text, re.I):
                self.form["allow_partial"] = True
                self._make_report()
            elif re.search(r"\bwhy\b|\bnot now\b", text, re.I) and self.report:
                report_text = self.render_report(self.report)
                why = report_text.split("WHY\n", 1)[-1].split("\n\nNUMBERS", 1)[0]
                self._block("text", why)
            elif self.report:
                self._block("text", self.report["decision"].get("decision_explanation", "The previous recommendation remains based on the same forecast."))
        else:
            self._start(text)
        return self.last_blocks
