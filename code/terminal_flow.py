"""Deterministic conversational terminal flow; deliberately has no Streamlit dependency."""

import datetime as dt
import os
import re

import failproofai_sdk as fp

from assistant import Assistant
from money import money
from plan_dates import installment_completion_date


_CURRENCIES = {"INR": "INR", "USD": "USD", "EUR": "EUR", "ZAR": "ZAR", "IDR": "IDR"}
_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "R": "ZAR"}
_AMOUNT_RE = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\d{1,3}(?: \d{3})+|\d+(?:\.\d+)?)(?![\d,])", re.I
)
BANNER = ("Glacier checks whether a payment fits your next 90 days: your balance, income, bills and pending items, "
          "and tells you whether to pay now, split it, or wait.")


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
    if re.search(r"(?<!\w)rs\.?(?!\w)|\brupees?\b", value, re.I):
        currency = "INR"
    amount_text = re.sub(r"\b20\d{2}-\d{2}-\d{2}\b", " ", value)
    amount_text = re.sub(
        r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"(?:\s+20\d{2})?\b", " ", amount_text, flags=re.I,
    )
    amount_text = re.sub(r"\bwithin\s+\d+\s+days\b", " ", amount_text, flags=re.I)
    matches = list(_AMOUNT_RE.finditer(amount_text))
    # Prefer explicit monetary notation and grouped/decimal amounts over stray
    # dates, installment counts, or other plain integers in a question.
    def priority(match):
        raw = match.group(0)
        prefix = amount_text[max(0, match.start() - 8):match.start()]
        suffix = amount_text[match.end():match.end() + 8]
        explicit = (bool(re.search(r"(?:INR|USD|EUR|ZAR|IDR|Rs\.?|₹|\$|€|R)\s*$", prefix, re.I))
                    or bool(re.match(r"\s*(?:INR|USD|EUR|ZAR|IDR)\b", suffix, re.I)))
        marked = bool(re.search(r"[, ]|\.", raw)) or bool(re.match(r"\s*(?:k|m|million|lakh|lakhs|crore|crores)\b", suffix, re.I))
        return (0 if explicit else 1 if marked else 2, match.start())
    for match in sorted(matches, key=priority):
        raw = match.group(0).replace(",", "").replace(" ", "")
        try:
            amount = float(raw)
        except ValueError:
            continue
        suffix = amount_text[match.end():]
        prefix = amount_text[:match.start()]
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
    def __init__(self, user_id="user_07", as_of_date=None, dataset_dir=None, assistant=None):
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
        self.latest_result = None
        self.latest_wording = "plain text"
        self.last_blocks = []

    def _earliest_installment_completion(self):
        return installment_completion_date(self.as_of_date, min(self.installment_counts))

    @property
    def installment_counts(self):
        return (3, 6, 12)

    def _eligible_installment_counts(self):
        deadline = self.form.get("pay_by_date")
        if not deadline:
            return list(self.installment_counts)
        return [count for count in self.installment_counts
                if installment_completion_date(self.as_of_date, count) <= dt.date.fromisoformat(str(deadline)[:10])]

    def _installment_options_text(self):
        counts = self._eligible_installment_counts()
        if not self.form.get("pay_by_date"):
            counts = list(self.installment_counts)
        return "1 no (default)" + "".join(f", {idx + 2} {count} months" for idx, count in enumerate(counts))

    def _installment_deadline_message(self, deadline):
        earliest = installment_completion_date(self.as_of_date, min(self.installment_counts))
        return (f"No installment option completes by {deadline}; the shortest plan ends "
                f"on {earliest}, after the deadline.")

    def _has_installment_conflict(self, deadline):
        count = self.form.get("installment_count") if self.form.get("allow_installments") else None
        return bool(count and deadline and installment_completion_date(self.as_of_date, count) > dt.date.fromisoformat(str(deadline)[:10]))

    def _latest_date(self, user_id):
        import pandas as pd
        engine = Assistant(user_id, dt.date.today().isoformat(), dataset_dir=self.dataset_dir, no_llm=True).engine
        dates = pd.to_datetime(engine.events[engine.events.user_id.astype(str).eq(str(user_id))].event_date, errors="coerce").dropna()
        return dates.max().date().isoformat() if not dates.empty else dt.date.today().isoformat()

    def _block(self, kind, text, data=None):
        block = {"kind": kind, "text": str(text), "plain_text": str(text), "data": data or {}}
        self.last_blocks.append(block)
        return block

    def _prompt(self):
        question = self.questions[self.question_index]
        options = {
            "deadline": "When do you need to pay? [1 none (default), 2 within 14 days, 3 end of this month, 4 type YYYY-MM-DD]",
            "partial": "Is partial payment acceptable? [1 no (default), 2 yes]",
            "installments": f"Use installments? [{self._installment_options_text()}]",
            "fee": "Optional installment fee percentage? [Enter for 0%]",
        }
        if question == "deadline_conflict":
            earliest = installment_completion_date(self.as_of_date, self.form["installment_count"])
            text = f"The selected installment plan ends {earliest}, after that deadline. Choose 1 none, or type a date on or after {earliest}."
            return self._block("prompt", text, {"question": text, "options": [{"key": "1", "label": "none", "default": True}, {"key": "", "label": f"date on/after {earliest}", "default": False}]})
        if question == "date":
            text = "Type a date in YYYY-MM-DD format."
            return self._block("prompt", text, {"question": text, "options": []})
        text = options[question]
        if question == "deadline": choices = [("1", "none", True), ("2", "within 14 days", False), ("3", "end of this month", False), ("4", "type YYYY-MM-DD", False)]
        elif question == "partial": choices = [("1", "no", True), ("2", "yes", False)]
        elif question == "installments": choices = [(part.strip().split()[0], part.strip()[len(part.strip().split()[0]):].strip().strip("()"), idx == 0) for idx, part in enumerate(self._installment_options_text().split(","))]
        else: choices = [("", "0%", True)]
        return self._block("prompt", text, {"question": text, "options": [{"key": k, "label": label, "default": default} for k, label, default in choices]})

    def _start(self, line):
        amount, currency = parse_amount(line)
        if amount is None:
            text = "What amount are you considering?"
            return [self._block("prompt", text, {"question": text, "options": []})]
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
            fee_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lowered)
            if fee_match:
                self.form["installment_fee_pct"] = float(fee_match.group(1))
            else:
                self.questions.append("fee")
        elif "installment" not in lowered:
            self.questions.append("installments")
        else:
            self.questions.append("installments")
        if self.form.get("allow_installments") and self.form.get("pay_by_date") and self._has_installment_conflict(self.form["pay_by_date"]):
            deadline = self.form["pay_by_date"]
            self.questions = [q for q in self.questions if q != "deadline"]
            self.questions.insert(0, "deadline_conflict")
            self._block("text", f"The selected {self.form['installment_count']}-installment plan conflicts with the deadline {deadline}.")
        self.phase = "intake"
        self.question_index = 0
        if re.search(r"\bjust advise\b", lowered):
            self.questions = []
        if self.form.get("pay_by_date") and not self._eligible_installment_counts() and "installments" in self.questions:
            self.questions.remove("installments")
            self._block("text", self._installment_deadline_message(self.form["pay_by_date"]))
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
        dynamic_counts = self._eligible_installment_counts()
        installment_choices = {str(idx + 2): count for idx, count in enumerate(dynamic_counts)}
        choices = {"deadline": {"1": None, "2": "14", "3": "month", "4": "date"},
                   "deadline_conflict": {"1": None}, "partial": {"1": False, "2": True},
                   "installments": {"1": False, **installment_choices}}
        is_date = key == "date" or (key == "deadline_conflict" and value not in {"", "1"})
        valid = value == "" or (key in choices and value in choices[key]) or (key == "fee" and bool(re.fullmatch(r"\d+(?:\.\d+)?", value)))
        if is_date and value:
            try:
                parsed_date = dt.date.fromisoformat(value)
                valid = True
                if key == "deadline_conflict":
                    valid = parsed_date >= installment_completion_date(self.as_of_date, self.form["installment_count"])
            except ValueError:
                valid = False
        if not valid:
            self.invalid_attempts += 1
            if self.invalid_attempts == 1:
                return [self._prompt()]
            value = ""
        self.invalid_attempts = 0
        if key in {"deadline", "deadline_conflict"}:
            if value == "2":
                self.form["pay_by_date"] = (dt.date.fromisoformat(self.as_of_date) + dt.timedelta(days=14)).isoformat()
            elif value == "3":
                d = dt.date.fromisoformat(self.as_of_date); n = d.replace(day=28) + dt.timedelta(days=4)
                self.form["pay_by_date"] = (n.replace(day=1) - dt.timedelta(days=1)).isoformat()
            elif value == "4":
                self.questions[self.question_index] = "date"
                text = "Type a date in YYYY-MM-DD format."
                return [self._block("prompt", text, {"question": text, "options": []})]
            elif key == "deadline_conflict" and value not in {"", "1"}:
                self.form["pay_by_date"] = dt.date.fromisoformat(value).isoformat()
            else:
                self.form["pay_by_date"] = None
        elif key == "date":
            self.form["pay_by_date"] = dt.date.fromisoformat(value).isoformat() if value else None
        elif key == "partial": self.form["allow_partial"] = value == "2"
        elif key == "installments":
            self.form["allow_installments"] = bool(value and value != "1")
            if value and value != "1": self.form["installment_count"] = choices[key][value]
            if self.form["allow_installments"]: self.questions.insert(self.question_index + 1, "fee")
        elif key == "fee": self.form["installment_fee_pct"] = float(value) if value else 0.0
        self.question_index += 1
        blocks = []
        if key in {"deadline", "deadline_conflict"} and self.form.get("pay_by_date") and self.form.get("allow_installments") and self._has_installment_conflict(self.form["pay_by_date"]):
            conflict_index = max(0, self.question_index - 1)
            self.questions[conflict_index] = "deadline_conflict"
            self.question_index = conflict_index
            self._block("text", f"The selected installment plan conflicts with the deadline {self.form['pay_by_date']}.")
            self._prompt()
            return self.last_blocks
        if key in {"deadline", "deadline_conflict"} and self.form.get("pay_by_date") and not self._eligible_installment_counts() and "installments" in self.questions[self.question_index:]:
            self.questions.remove("installments")
            blocks.append(self._block("text", self._installment_deadline_message(self.form["pay_by_date"])))
        if self.question_index < len(self.questions):
            blocks.append(self._prompt())
            return blocks
        return self._make_report()

    def _make_report(self):
        self.phase = "reported"
        self.form.setdefault("pay_by_date", None)
        try:
            home = str(self.assistant.engine.get_user_home_currency(self.user_id))
            converted, _ = self.assistant._home_amount(self.form["amount"], self.form["currency"])
            if converted is None:
                return [self._block("error", f"Your account is in {home} and I have no {self.form['currency']} to {home} rate. Type the amount in {home}, or switch accounts with /user.")]
            with fp.session(self.session_id):
                with fp.agent("assistant", goal="Produce deterministic payment advice"):
                    with fp.tool_call("advise", input={"user_id": self.user_id, "amount": self.form["amount"]}) as tool:
                        result = self.assistant.advise(self.form)
                        tool.output = result
            self.report = result
            self.latest_result = result
            self.latest_wording = "plain text"
            text = self.render_report(result)
            return [self._block("report", text, self.report_data(result))]
        except Exception as exc:
            fp.event.error(session_id=self.session_id, agent_id="assistant", error_type=type(exc).__name__, message=str(exc)[:300])
            return [self._block("error", f"Unable to produce advice: {exc}")]

    def render_report(self, result):
        currency = self.form["currency"]
        options = result.get("ranked_options", [])
        chosen = result.get("recommended_option")
        if chosen is not None and not chosen.get("safe", False):
            chosen = None
        not_affordable = str(result.get("decision", {}).get("affordability_status")) == "not_affordable"
        if not_affordable or chosen is None:
            verdict = "NOT RECOMMENDED: no option keeps your balance above the minimum"
        elif chosen["type"] == "pay_in_full_now": verdict = "PAY IN FULL TODAY"
        elif chosen["type"] == "installments": verdict = f"PAY IN {self.form.get('installment_count', 3)} INSTALLMENTS"
        elif chosen["type"] == "partial_then_remainder": verdict = f"PAY {money(next(iter(chosen['amounts'].values())), currency)} NOW AND THE REST ON {list(chosen['amounts'])[-1]}"
        elif chosen["type"] == "wait": verdict = f"WAIT UNTIL {max(chosen['dates'])}"
        else: verdict = "NOT RECOMMENDED"
        situation = result["situation"]
        why = (f"The current balance is {money(situation['current_balance'], currency)}, and the minimum to keep is {money(situation['minimum_balance'], currency)}. "
               f"The lowest projected balance is {money(situation['lowest_projected_balance'], currency)} on {situation['lowest_projected_date']}. "
               f"Monthly counted income is {money(situation['monthly_income_total'], currency)}; only validated recurring income is counted.")
        pending = situation.get("pending_items_considered_or_ignored", {})
        why += (f" Pending items: considered {', '.join(map(str, pending.get('considered', []))) or 'none'}; "
                f"ignored {', '.join(map(str, pending.get('ignored', []))) or 'none'}.")
        balances = result.get("evidence", {}).get("balances", {})
        vals = list(balances.values())
        levels = "▁▂▃▄▅▆▇█"
        spark = "" if not vals else "".join(levels[round((v-min(vals))/(max(vals)-min(vals))*7)] if max(vals) != min(vals) else "▄" for v in vals)
        minimum = situation["minimum_balance"]
        rows = ["OPTION                 WHEN                 TOTAL COST       LOWEST BALANCE   MEETS DEADLINE   SAFE"]
        for option in options:
            safe_mark = "✓" if option.get("safe") else "✗"
            safety = "SAFE" if option.get("safe") else (
                f"(drops to {money(option['lowest_projected_balance'],currency)} on "
                f"{option['lowest_projected_date']}, {money(max(0, minimum-option['lowest_projected_balance']),currency)} below the minimum)"
            )
            meets = option.get("meets_deadline")
            meets = "n/a" if meets is None else str(meets)
            rows.append(f"{option['type'].replace('_',' '):22} {', '.join(option['dates']):20} {money(option['total_cost'],currency):16} {money(option['lowest_projected_balance'],currency):16} {meets}   {safe_mark} {'SAFE' if option.get('safe') else 'UNSAFE'}")
            if not option.get("safe"):
                rows.append(f"                         Note: {safety}")
        other_safe = [x["type"].replace("_", " ") for x in options if x.get("safe") and x is not chosen]
        alts = [f"Best other safe options: {', '.join(other_safe) or 'none'}.",
                f"Earliest safe date to wait for: {situation.get('earliest_date_for_full_payment') or 'not available'}.",
                f"Amount safe to pay today: {money(result['decision'].get('amount_safe_to_pay', 0),currency)}.",
                f"Spending changes that would free cash: {result['decision'].get('spending_changes_needed', 'none')}."]
        if verdict.startswith("NOT RECOMMENDED"):
            steps = ["Do not make this payment now"]
            safe_today = float(result["decision"].get("amount_safe_to_pay", 0) or 0)
            if safe_today > 0:
                steps.append(f"Amount safe to pay today: {money(safe_today, currency)}.")
            earliest = situation.get("earliest_date_for_full_payment")
            if earliest:
                steps.append(f"Earliest safe date: {earliest}.")
            spending_changes = result["decision"].get("spending_changes_needed", "none")
            steps.append(
                "This answer could change if income resumes, you choose a smaller amount, "
                f"or spending changes ({spending_changes})."
            )
        else:
            steps = [f"Confirm the selected plan for {money(self.form['amount'],currency)}.", "Keep the minimum balance available.", "Recheck the forecast before making the payment."]
        dates = list(balances)
        start_date = dates[0] if dates else self.as_of_date
        end_date = dates[-1] if dates else (dt.date.fromisoformat(self.as_of_date) + dt.timedelta(days=90)).isoformat()
        start_balance = vals[0] if vals else situation["current_balance"]
        end_balance = vals[-1] if vals else situation["current_balance"]
        return (f"╭─ {verdict} ─╮\nWHY\n{why}\n\nNUMBERS\n90-day balance: {spark}\nMinimum-balance level: {money(minimum,currency)}\n{start_date} {money(start_balance,currency)} → {end_date} {money(end_balance,currency)}\n\nOPTIONS\n" + "\n".join(rows) + "\n\nALTERNATIVES\n" + "\n".join("• "+s for s in alts) + "\n\nNEXT STEPS\n" + "\n".join("• "+s for s in steps) + "\n╰" + "─" * max(10, len(verdict) + 4) + "╯")

    def report_data(self, result):
        rendered = self.render_report(result)
        verdict = rendered.split("╭─ ", 1)[1].split(" ─╮", 1)[0]
        chosen = result.get("recommended_option")
        if not_affordable := (str(result.get("decision", {}).get("affordability_status")) == "not_affordable"):
            status = "bad"
        elif chosen is None or not chosen.get("safe", False):
            status = "bad"
        elif chosen.get("type") == "wait":
            status = "caution"
        else:
            status = "good"
        situation = result["situation"]
        balances = result.get("evidence", {}).get("balances", {})
        values = list(balances.values())
        dates = list(balances)
        options = []
        for item in result.get("ranked_options", []):
            low = item.get("lowest_projected_balance")
            options.append({"name": item.get("type", "").replace("_", " "), "when": ", ".join(item.get("dates", [])),
                            "cost": money(item.get("total_cost", 0), self.form["currency"]),
                            "lowest": money(low, self.form["currency"]),
                            "meets_deadline": "n/a" if item.get("meets_deadline") is None else str(item.get("meets_deadline")),
                            "safe": bool(item.get("safe")),
                            "note": "" if item.get("safe") else f"Unsafe: drops to {money(low, self.form['currency'])} on {item.get('lowest_projected_date')}"})
        sections = rendered.split("\n\nALTERNATIVES\n")
        alternatives = sections[-1].split("\n\nNEXT STEPS\n")[0].splitlines() if len(sections) > 1 else []
        next_steps = rendered.split("\n\nNEXT STEPS\n", 1)[-1].split("\n╰", 1)[0].splitlines()
        return {"verdict": verdict, "status": status,
                "why": rendered.split("WHY\n", 1)[-1].split("\n\nNUMBERS", 1)[0],
                "numbers": {"sparkline_values": values, "minimum": situation["minimum_balance"],
                            "start_date": dates[0] if dates else self.as_of_date,
                            "end_date": dates[-1] if dates else self.as_of_date,
                            "start_balance": values[0] if values else situation["current_balance"],
                            "end_balance": values[-1] if values else situation["current_balance"]},
                "options": options, "alternatives": alternatives, "next_steps": next_steps}

    def handle(self, line):
        self.last_blocks = []
        text = str(line or "").strip()
        command = text.lower()
        if command.startswith("/help"):
            self._block("text", "Start with an amount and optional currency. Intake asks only unanswered questions; Enter accepts defaults and /skip uses defaults. Commands: /user <id>, /users [currency], /date YYYY-MM-DD, /situation, /why, /reset, /export, /help, /skip. Use /export after a report to get plain text.")
        elif command == "/why":
            self._block("text", self._why_text())
        elif command == "/users" or command.startswith("/users "):
            requested = text.split(maxsplit=1)[1].upper() if len(text.split()) > 1 else None
            profiles = self.assistant.engine.profiles
            if requested:
                profiles = profiles[profiles.home_currency.astype(str).str.upper().eq(requested)]
            lines = [f"{uid} · {row.home_currency} · {money(row.current_available_balance, row.home_currency)}"
                     for uid, row in profiles.head(8).iterrows()]
            self._block("text", "Accounts" + (f" in {requested}" if requested else "") + ":\n" + ("\n".join(lines) or "No matching accounts."))
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
                report_text = self.render_report(self.report)
                self._block("text", report_text.split("\n\nNUMBERS", 1)[0])
        else:
            self._start(text)
        return self.last_blocks

    def _why_text(self):
        result = self.latest_result
        if not result:
            return "No completed answer yet. Ask an amount first."
        start = dt.date.fromisoformat(self.as_of_date)
        events = self.assistant.engine.events
        user_events = events[events.user_id.astype(str).eq(self.user_id)]
        evidence = result.get("evidence", {})
        balance_dates = sorted(evidence.get("balances", {}))
        window_start = balance_dates[0] if balance_dates else start.isoformat()
        window_end = balance_dates[-1] if balance_dates else (start + dt.timedelta(days=90)).isoformat()
        breakdown = evidence.get("breakdown", {})
        decisions = result.get("income_stream_classes", [])
        if not decisions:
            decisions = breakdown.get("income_decisions", [])
        if not decisions:
            decisions = evidence.get("explanation_evidence", {}).get("income_decisions", [])
        income_lines = [f"{entry.get('series_key', 'unknown stream')}: {entry.get('decision', 'unknown')} — {entry.get('reason', 'reason unavailable')}"
                        for entry in decisions]
        messages = evidence.get("retrieved_message_ids", [])
        if not messages:
            messages = evidence.get("explanation_evidence", {}).get("retrieved_message_ids", [])
        option_names = [option.get("type", "unknown") for option in result.get("ranked_options", [])]
        return (f"Data loaded: {len(user_events)} events for {self.user_id}; as-of {self.as_of_date}; "
                f"90-day window {window_start} through {window_end}.\n"
                f"Income streams/classes: {'; '.join(income_lines) if income_lines else 'No stream-class detail in engine evidence.'}\n"
                f"Retrieved message IDs: {', '.join(map(str, messages)) if messages else 'none in engine evidence.'}\n"
                f"Options considered: {', '.join(option_names) if option_names else 'none'}.\n"
                f"Wording: {self.latest_wording}.\nFailproof session: {self.session_id}")
