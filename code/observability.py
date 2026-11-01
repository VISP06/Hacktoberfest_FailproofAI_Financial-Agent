"""Shared local usage logging and Failproof session identity helpers."""

import json
import os
import threading
from datetime import datetime, timezone


_usage_lock = threading.Lock()


def new_run_id():
    """Return the configured run id or a UTC timestamp suitable for a run id."""
    return os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def make_session_id(request_id, run_id):
    return f"req-{request_id}-{run_id}"


def _token_count(response, attribute):
    usage = getattr(response, "usage", None) if response is not None else None
    value = getattr(usage, attribute, None)
    return int(value) if value is not None else None


def log_model_usage(purpose, model, response=None, success=False):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "model": model,
        "input_tokens": _token_count(response, "prompt_tokens"),
        "output_tokens": _token_count(response, "completion_tokens"),
        "success": bool(success),
    }
    try:
        os.makedirs("logs", exist_ok=True)
        with _usage_lock, open("logs/usage.jsonl", "a", encoding="utf-8") as usage_file:
            usage_file.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        # Local usage accounting must never break the model-backed workflow.
        pass


def logged_completion(completion_fn, *, purpose, model, **kwargs):
    """Call a model and log exactly one usage record, including failed calls."""
    try:
        response = completion_fn(model=model, **kwargs)
    except Exception:
        log_model_usage(purpose, model, success=False)
        raise
    log_model_usage(purpose, model, response=response, success=True)
    return response
