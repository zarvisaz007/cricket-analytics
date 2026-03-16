"""
http_client.py — Shared HTTP infrastructure for all Cricket Analytics scrapers.

Provides a `requests.Session`-based client with:
  - 12 rotating User-Agent strings (consistent per URL via hash mod)
  - Rate limiting via `scrape_delay()` (SCRAPE_DELAY_SECONDS env var, default 2.0)
  - `tenacity`-powered retry logic: exponential back-off, retries on
    ConnectionError, Timeout, and HTTP 429 / 502 / 503 responses
  - Standard request headers (Referer, Accept-Language)

Public API
----------
get_page(url, params=None) -> requests.Response
    Make a rate-limited, retried GET request. Raises after exhausted retries.

scrape_delay(url)
    Sleep for SCRAPE_DELAY_SECONDS before making a request. Called internally
    by get_page(); exposed for callers that need to pace non-get_page requests.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Dict, Optional

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRAPE_DELAY_SECONDS: float = float(os.getenv("SCRAPE_DELAY_SECONDS", "2.0"))

# 12 real-world User-Agent strings covering major browsers / OSes
_USER_AGENTS: list[str] = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.3; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    # Firefox on Linux
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3.1 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    # Chrome on Android
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.6261.119 Mobile Safari/537.36",
    # Safari on iPhone
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 "
    "Mobile/15E148 Safari/604.1",
    # Chrome on Windows (older)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Opera on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 OPR/108.0.0.0",
]

_NUM_AGENTS: int = len(_USER_AGENTS)  # 12

# Shared session — created once, reused across all requests
_session: Optional[requests.Session] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """Return the shared requests.Session, creating it on first call."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Session-level defaults (overridden per-request for UA)
        _session.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                          "image/avif,image/webp,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )
    return _session


def _user_agent_for_url(url: str) -> str:
    """
    Select a User-Agent deterministically from the pool by hashing the URL.

    The same URL always maps to the same UA, giving consistent browser
    fingerprinting while distributing load across multiple agents.
    """
    digest = hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()
    index = int(digest, 16) % _NUM_AGENTS
    return _USER_AGENTS[index]


# ---------------------------------------------------------------------------
# Public rate-limiting helper
# ---------------------------------------------------------------------------

def scrape_delay(url: str = "") -> None:
    """
    Sleep for `SCRAPE_DELAY_SECONDS` to honour site rate limits.

    Parameters
    ----------
    url:
        The target URL (used only for log context).
    """
    delay = SCRAPE_DELAY_SECONDS
    if delay > 0:
        logger.debug("Rate-limit delay %.1fs before %s", delay, url or "(unknown)")
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Retry predicate helpers
# ---------------------------------------------------------------------------

class _RetryableHTTPError(requests.HTTPError):
    """Sentinel exception raised when a response has a retryable HTTP status."""


def _check_response_status(response: requests.Response) -> requests.Response:
    """
    Raise `_RetryableHTTPError` for HTTP 429, 502, 503 so tenacity can retry.
    Raise `requests.HTTPError` for other 4xx/5xx (no retry).
    """
    retryable_codes = {429, 502, 503}
    if response.status_code in retryable_codes:
        logger.warning(
            "Received HTTP %s for %s — will retry",
            response.status_code,
            response.url,
        )
        raise _RetryableHTTPError(
            f"HTTP {response.status_code}", response=response
        )
    # Let other error codes surface as plain HTTPError (not retried)
    response.raise_for_status()
    return response


# ---------------------------------------------------------------------------
# Core fetch function with retry
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            _RetryableHTTPError,
        )
    ),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch(
    url: str,
    params: Optional[Dict] = None,
    timeout: int = 30,
) -> requests.Response:
    """
    Internal fetch with tenacity retry logic.

    Retries on `ConnectionError`, `Timeout`, and `_RetryableHTTPError`
    (HTTP 429 / 502 / 503).  Other HTTP errors are raised immediately.
    """
    session = _get_session()
    ua = _user_agent_for_url(url)
    headers = {"User-Agent": ua}

    logger.debug("GET %s  UA-index=%d", url, _USER_AGENTS.index(ua))
    response = session.get(url, params=params, headers=headers, timeout=timeout)
    return _check_response_status(response)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_page(
    url: str,
    params: Optional[Dict] = None,
    timeout: int = 30,
) -> requests.Response:
    """
    Fetch *url* with rate-limiting, consistent UA rotation, and automatic
    retry on transient errors.

    The function always calls `scrape_delay(url)` **before** issuing the
    network request so that every caller automatically respects the
    configured rate limit without needing any extra code.

    Parameters
    ----------
    url:
        Fully-qualified URL to fetch.
    params:
        Optional query-string parameters dict forwarded to `requests.get`.
    timeout:
        Per-request timeout in seconds (default 30).

    Returns
    -------
    requests.Response
        The successful HTTP response object.

    Raises
    ------
    requests.exceptions.ConnectionError / Timeout
        Re-raised after all retry attempts are exhausted.
    requests.HTTPError
        Raised immediately for non-retryable 4xx / 5xx responses, or after
        retries are exhausted for 429 / 502 / 503.
    """
    scrape_delay(url)
    return _fetch(url, params=params, timeout=timeout)


def reset_session() -> None:
    """
    Discard the shared session and create a fresh one on the next request.

    Useful in tests or after a long-running process where the TCP pool may
    have accumulated stale connections.
    """
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None
    logger.debug("HTTP session reset.")
