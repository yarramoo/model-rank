"""Small HTTP helpers on top of the standard library (no requests dependency)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

USER_AGENT = "param-budget-site/1.0 (static leaderboard build; +https://docs.gitlab.com/user/project/pages/)"


class FetchError(RuntimeError):
    pass


def get_bytes(url: str, headers: dict | None = None, timeout: int = 60, retries: int = 2) -> bytes:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    hdrs.update(headers or {})
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = e
            # 4xx other than 429 will not get better on retry
            if e.code != 429 and 400 <= e.code < 500:
                raise FetchError(f"HTTP {e.code} for {url}") from e
            if e.code == 429 and attempt < retries:
                try:
                    wait = min(float(e.headers.get("Retry-After", "20")), 90.0)
                except ValueError:
                    wait = 20.0
                time.sleep(wait)
                continue
        except Exception as e:  # timeouts, DNS, TLS, proxy refusals
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"{type(last).__name__}: {last} for {url}")


def get_json(url: str, headers: dict | None = None, timeout: int = 60, retries: int = 2):
    raw = get_bytes(url, headers=headers, timeout=timeout, retries=retries)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise FetchError(f"Not JSON from {url}: {e}") from e


def get_text(url: str, headers: dict | None = None, timeout: int = 60, retries: int = 2) -> str:
    return get_bytes(url, headers=headers, timeout=timeout, retries=retries).decode("utf-8", "replace")
