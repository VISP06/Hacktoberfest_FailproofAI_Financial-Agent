import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "code"))
from observability import make_session_id


def test_session_id_uses_request_and_run_id():
    assert make_session_id("abc-123", "20261101T0830") == "req-abc-123-20261101T0830"
