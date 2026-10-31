"""Optional, strictly validated language polish for deterministic explanations."""
import hashlib
import json
import os
import re
import time

import failproofai_sdk as fp
from litellm import completion
from income_classifier import IncomeClassifier

PROMPT_VERSION = "explain-polish-v1"


def _numbers(value, in_date=False):
    found = set()
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        found.add(float(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found |= _numbers(item, in_date or "date" in str(key).lower())
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= _numbers(item, in_date)
    elif in_date and isinstance(value, str):
        found.update(float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", value))
    return found


def _ids(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("id") or key.endswith("ids"):
                if isinstance(item, list):
                    found.update(map(str, item))
                elif item is not None:
                    found.add(str(item))
            else:
                found |= _ids(item)
    elif isinstance(value, list):
        for item in value:
            found |= _ids(item)
    return found


def validate_polish(text, evidence):
    if not isinstance(text, str) or not 2 <= len(re.findall(r"[^.!?]+[.!?](?:\s|$)", text.strip())) <= 3:
        return False
    allowed_numbers = _numbers(evidence)
    for token in re.findall(r"(?<![\w-])(?:\d[\d,]*(?:\.\d+)?)(?![\w-])", text):
        if float(token.replace(",", "")) not in allowed_numbers:
            return False
    allowed_ids = _ids(evidence)
    cited = set(re.findall(r"\b(?:event|message|pending|income)[-_][\w-]*\d[\w-]*\b", text, re.I))
    return cited <= allowed_ids


def polish_explanations(rows, model_name=None, cache_path=None, client=None):
    """Polish rows in groups of five; failures preserve their deterministic text."""
    model_name = model_name or os.getenv("EXPLAIN_MODEL") or os.getenv("BUY_OR_WAIT_MODEL", "gemini/gemini-1.5-flash")
    cache_path = cache_path or os.path.join("dataset", "explanation_cache.json")
    try:
        with open(cache_path, encoding="utf-8") as stream:
            cache = json.load(stream)
    except (OSError, ValueError):
        cache = {}
    completion_fn = client or completion
    daily_quota_exhausted = False
    for batch_start in range(0, len(rows), 5):
        if daily_quota_exhausted:
            break
        batch = rows[batch_start:batch_start + 5]
        pending = []
        for row in batch:
            evidence = row["explanation_evidence"]
            key = hashlib.sha256((PROMPT_VERSION + json.dumps(evidence, sort_keys=True, default=str)).encode()).hexdigest()
            row["_polish_cache_key"] = key
            if key in cache and validate_polish(cache[key], evidence):
                row["decision_explanation"] = cache[key]
            else:
                pending.append(row)
        for start in range(0, len(pending), 5):
            group = pending[start:start + 5]
            session_id = f"explain-batch-{batch_start // 5 + 1}"
            last_exc = None
            for attempt in range(4):
                try:
                    fp.event.model_request(session_id=session_id, agent_id="explanation-polisher", model=model_name, request_id=session_id)
                    response = completion_fn(model=model_name, messages=[{"role": "user", "content":
                        "Write 2-3 plain-language sentences per item, grounding every fact in its evidence. Return JSON only as {\"explanations\":[\"...\"]}. Input contains evidence only; do not add facts.\n" +
                        json.dumps([row["explanation_evidence"] for row in group], sort_keys=True, default=str)}],
                        temperature=0, response_format={"type": "json_object"})
                    usage = getattr(response, "usage", None)
                    fp.event.model_response(session_id=session_id, agent_id="explanation-polisher", model=model_name, request_id=session_id,
                        input_tokens=getattr(usage, "prompt_tokens", None), output_tokens=getattr(usage, "completion_tokens", None))
                    results = json.loads(response.choices[0].message.content).get("explanations", [])
                    if len(results) != len(group):
                        raise ValueError("wrong explanation count")
                    for row, text in zip(group, results):
                        if validate_polish(text, row["explanation_evidence"]):
                            row["decision_explanation"] = text
                            cache[row["_polish_cache_key"]] = text
                    break
                except Exception as exc:
                    last_exc = exc
                    fp.event.error(session_id=session_id, agent_id="explanation-polisher", error_type=type(exc).__name__, message=str(exc)[:300])
                    if IncomeClassifier._is_daily_quota(exc):
                        daily_quota_exhausted = True
                        break
                    if not (IncomeClassifier._is_429(exc) or IncomeClassifier._is_transient(exc)) or attempt == 3:
                        break
                    time.sleep(IncomeClassifier._retry_delay_429(exc) if IncomeClassifier._is_429(exc) else (5, 15, 45)[min(attempt, 2)])
    for row in rows:
        row.pop("_polish_cache_key", None)
    if cache:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as stream:
            json.dump(cache, stream, sort_keys=True)
    return rows
