"""Shared HTTP helpers for the e2e provisioner.

Plain `requests`-like wrappers around httpx with logging and retry. We use
httpx because it's already a project dep; we use the *sync* client here
because the provisioner is a one-shot script — async buys nothing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

_log = logging.getLogger(__name__)


class ServiceUnreachable(RuntimeError):
    pass


def wait_for_endpoint(
    url: str,
    *,
    timeout_sec: float = 240.0,
    interval_sec: float = 2.0,
    expect_status: tuple[int, ...] = (200,),
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Poll `url` until it returns one of `expect_status` or timeout elapses.

    Returns the successful response. Raises ServiceUnreachable on timeout.
    """
    deadline = time.monotonic() + timeout_sec
    last_err: str = "never attempted"
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url, headers=headers or {})
            if resp.status_code in expect_status:
                _log.info("ready after %d attempt(s): %s", attempts, url)
                return resp
            last_err = f"HTTP {resp.status_code}"
        except httpx.TransportError as exc:
            last_err = f"transport: {exc}"
        time.sleep(interval_sec)

    raise ServiceUnreachable(f"{url} not ready after {timeout_sec:.0f}s (last: {last_err})")


class ApiClient:
    """Thin sync wrapper around httpx.Client with logging + JSON shortcuts."""

    def __init__(
        self, base_url: str, *, headers: dict[str, str] | None = None, timeout: float = 60.0
    ) -> None:
        self._client = httpx.Client(base_url=base_url, headers=headers or {}, timeout=timeout)
        self._base_url = base_url

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._raise(self._client.get(path, **kwargs))

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._raise(self._client.post(path, **kwargs))

    def put(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._raise(self._client.put(path, **kwargs))

    def patch(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._raise(self._client.patch(path, **kwargs))

    def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._raise(self._client.delete(path, **kwargs))

    def get_json(self, path: str, **kwargs: Any) -> Any:
        return self.get(path, **kwargs).json()

    def _raise(self, resp: httpx.Response) -> httpx.Response:
        if resp.is_success:
            return resp
        # Tag the URL + body in the error so failed provisioning shows context.
        try:
            body = resp.text[:500]
        except Exception:
            body = "<unreadable>"
        raise httpx.HTTPStatusError(
            f"{resp.request.method} {resp.request.url} -> HTTP {resp.status_code}: {body}",
            request=resp.request,
            response=resp,
        )
