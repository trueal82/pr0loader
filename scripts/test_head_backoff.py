"""Smoke-test for polite HEAD verification with Fibonacci backoff.

This script does NOT require real 429s from pr0gramm.
It monkeypatches the requests session's head() call to simulate:
- two 429 responses
- then one successful 200 with Content-Length

Run:
    uv run python scripts/test_head_backoff.py

Expected:
- get_remote_file_size() retries
- uses Fibonacci backoff (attempt increments)
- returns the mocked size
"""

from __future__ import annotations

import time

import requests

from pr0loader.api.client import APIClient
from pr0loader.config import load_settings


class _MockResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.reason = "mock"

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err


def main():
    settings = load_settings()
    settings.max_retries = 5
    settings.max_backoff_seconds = 3
    settings.request_timeout = 2

    api = APIClient(settings)

    calls = {"n": 0}

    def fake_head(url, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _MockResponse(429, headers={"Retry-After": "1"})
        return _MockResponse(200, headers={"Content-Length": "12345"})

    api.session.head = fake_head  # type: ignore

    t0 = time.time()
    size = api.get_remote_file_size("does-not-matter.jpg")
    dt = time.time() - t0

    assert size == 12345, size
    assert calls["n"] == 3, calls
    print(f"OK: size={size}, calls={calls['n']}, elapsed={dt:.2f}s")


if __name__ == "__main__":
    main()

