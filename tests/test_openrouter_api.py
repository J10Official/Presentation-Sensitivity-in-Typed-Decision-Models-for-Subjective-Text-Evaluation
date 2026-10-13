"""Offline transport contracts, independent of the excluded pilot runners."""
import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from openrouter_api import request_body, send_decision

PAYLOAD = {"model": "test/decision", "state": "Unicode: café",
           "questions": {"judgment": {"type": "choice", "instructions": "Choose.",
                                       "criteria": {"B": "First", "A": "Second"}}}}


class ApiTransportChecks(unittest.TestCase):
    def test_preserves_exact_bytes_order_unicode_and_redacts_authorization(self):
        events, received = [], []
        def transport(request, timeout):
            received.append(request.data)
            self.assertEqual(request.get_header("Authorization"), "Bearer test-private-key")
            return 200, b'{"ok":true}', {"X-Test": "header"}
        result = send_decision(PAYLOAD, "test-private-key", transport=transport, log_event=events.append)
        self.assertTrue(result["ok"])
        self.assertEqual(received, [request_body(PAYLOAD)])
        self.assertIn('"criteria":{"B":"First","A":"Second"}', received[0].decode())
        self.assertIn("café", received[0].decode())
        self.assertNotIn("test-private-key", json.dumps(events))
        self.assertEqual(events[0]["body_utf8"].encode(), received[0])

    def test_transient_http_failure_retries_and_keeps_response_bytes(self):
        responses = iter([(429, b'\xffbusy', {"Retry-After": "0"}), (200, b'{"value":1}', {})])
        events = []
        with patch("openrouter_api.time.sleep") as sleep:
            result = send_decision(PAYLOAD, "dummy", transport=lambda *_: next(responses), log_event=events.append)
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(base64.b64decode(events[1]["body_base64"]), b'\xffbusy')
        sleep.assert_called_once_with(0.0)

    def test_permanent_http_failure_does_not_retry(self):
        with patch("openrouter_api.time.sleep") as sleep:
            result = send_decision(PAYLOAD, "dummy", transport=lambda *_: (401, b'Unauthorized', {}))
        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 1)
        self.assertFalse(result["error"]["retryable"])
        sleep.assert_not_called()

    def test_transport_failure_exhaustion_returns_failure_signal(self):
        def transport(*_):
            raise TimeoutError("Synthetic timeout")
        with patch("openrouter_api.time.sleep"):
            result = send_decision(PAYLOAD, "dummy", transport=transport, max_attempts=2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["error"]["kind"], "transport_error")

    def test_invalid_success_body_is_retryable(self):
        responses = iter([(200, b'not-json', {}), (200, b'{"value":2}', {})])
        with patch("openrouter_api.time.sleep"):
            result = send_decision(PAYLOAD, "dummy", transport=lambda *_: next(responses), max_attempts=2)
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)

    def test_missing_key_never_calls_transport(self):
        def transport(*_):
            self.fail("Missing credentials must not reach transport")
        result = send_decision(PAYLOAD, "", transport=transport)
        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["error"]["kind"], "missing_key")


if __name__ == "__main__":
    unittest.main()
