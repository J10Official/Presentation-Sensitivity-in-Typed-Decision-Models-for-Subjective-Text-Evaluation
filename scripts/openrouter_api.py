"""Small OpenRouter Decisions API transport. Never edits the supplied payload."""

import json
import random
import time
from base64 import b64encode
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def request_body(payload):
    """The exact bytes sent, preserving supplied field and option order."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def read_key(env_path):
    """Use the process environment first, then a plain OPENROUTER_API_KEY=.env line."""
    import os

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"\'')
    return ""


def http_transport(request, timeout):
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def retry_delay(headers, attempt):
    """Honor numeric or HTTP-date Retry-After, otherwise use bounded jittered backoff."""
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), "")
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            delay = min(2 ** (attempt - 1), 30) * random.uniform(0.8, 1.2)
    return max(0.0, min(delay, 60.0))


def send_decision(payload, api_key, transport=http_transport, max_attempts=6, timeout=60, log_event=None):
    """Retry transient failures; log exact bodies through an optional callback."""
    if not api_key:
        return {"ok": False, "attempts": 0, "status_code": None,
                "response": None, "error": {"kind": "missing_key", "message": "Set OPENROUTER_API_KEY in .env or environment."}}

    body = request_body(payload)
    started_at, started = utc_now(), time.perf_counter()
    emit = log_event or (lambda event: None)
    for attempt in range(1, max_attempts + 1):
        request = Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        emit({"event": "request", "timestamp": utc_now(), "attempt": attempt,
              "method": "POST", "url": ENDPOINT,
              "headers": {"Content-Type": "application/json", "Authorization": "Bearer <redacted>"},
              "body_utf8": body.decode("utf-8"), "body_bytes": len(body)})
        attempt_started = time.perf_counter()
        try:
            status, raw, headers = transport(request, timeout)
        except (URLError, TimeoutError, OSError, ConnectionError,
                IncompleteRead, RemoteDisconnected) as exc:
            status, raw, headers = None, b"", {}
            error = {"kind": "transport_error", "message": str(exc), "retryable": True}
        else:
            error = None

        # Base64 preserves even a non-UTF-8 error response byte for byte.
        emit({"event": "response", "timestamp": utc_now(), "attempt": attempt,
              "status_code": status, "headers": headers,
              "latency_seconds": time.perf_counter() - attempt_started,
              "body_utf8": raw.decode("utf-8", errors="replace"),
              "body_base64": b64encode(raw).decode("ascii"), "error": error})
        try:
            response = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            response = None
        if status is not None and 200 <= status < 300 and isinstance(response, dict):
            return {"ok": True, "attempts": attempt, "status_code": status,
                    "response": response, "error": None, "started_at": started_at,
                    "finished_at": utc_now(), "latency_seconds": time.perf_counter() - started}
        if error is None:
            error = {"kind": "http_error" if status >= 300 else "invalid_json",
                     "message": raw.decode("utf-8", errors="replace")[:1000],
                     "retryable": status in RETRYABLE or 200 <= status < 300}
        retryable = error["retryable"]

        if not retryable or attempt == max_attempts:
            return {"ok": False, "attempts": attempt, "status_code": status,
                    "response": response, "error": error, "started_at": started_at,
                    "finished_at": utc_now(), "latency_seconds": time.perf_counter() - started}
        delay = retry_delay(headers, attempt)
        emit({"event": "retry", "timestamp": utc_now(), "attempt": attempt,
              "delay_seconds": delay, "error": error})
        time.sleep(delay)
