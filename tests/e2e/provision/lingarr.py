"""Lingarr REST provisioner.

API quirks (verified against Lingarr 1.2.4):

- Settings live at `/api/setting/{key}` (singular). GET returns the bare
  value as JSON text; POST `/api/setting` takes `{"key":"...","value":"..."}`.
- Before any setting endpoint accepts writes, onboarding must be completed
  via POST `/api/auth/onboarding` with body
  `{"enableUserAuth":"false","request":{"username":"...","password":"..."}}`
  (yes, even when `enableUserAuth` is "false" a username/password is required —
  the controller validates them before checking the auth flag).
- The translate endpoint Bazarr-bridge calls is `/api/Translate/content`.
"""

from __future__ import annotations

import logging
from typing import Any

from tests.e2e.provision._http import ApiClient, wait_for_endpoint

_log = logging.getLogger(__name__)

# Username/password we throw at onboarding to satisfy the validator. We
# immediately disable user-auth in the same call so these credentials are
# never actually load-bearing.
_ONBOARDING_USERNAME = "e2e-admin"
_ONBOARDING_PASSWORD = "e2e-onboarding-throwaway-password"


class LingarrProvisioner:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = ApiClient(self._base_url, headers={"Accept": "application/json"})

    def wait_ready(self, timeout_sec: float = 240.0) -> None:
        wait_for_endpoint(f"{self._base_url}/health", timeout_sec=timeout_sec)

    def ensure_libretranslate_backend(self, libretranslate_url: str) -> None:
        """Complete onboarding (idempotent) + point service_type at libretranslate."""
        self._complete_onboarding_if_needed()

        cur_service = self._get_setting("service_type")
        target_service = "libretranslate"
        if cur_service != target_service:
            self._put_setting("service_type", target_service)
            _log.info("Lingarr service_type -> %s", target_service)
        else:
            _log.info("Lingarr service_type already %s", target_service)

        cur_url = self._get_setting("libretranslate_url")
        if cur_url != libretranslate_url:
            self._put_setting("libretranslate_url", libretranslate_url)
            _log.info("Lingarr libretranslate_url -> %s", libretranslate_url)
        else:
            _log.info("Lingarr libretranslate_url already %s", libretranslate_url)

        # Pretend we've finished the radarr/sonarr setup screens too, so
        # other parts of the Lingarr UI don't nag.
        if self._get_setting("radarr_settings_completed") != "true":
            self._put_setting("radarr_settings_completed", "true")
        if self._get_setting("sonarr_settings_completed") != "true":
            self._put_setting("sonarr_settings_completed", "true")

    def _complete_onboarding_if_needed(self) -> None:
        """Idempotent: re-POSTing onboarding once already complete is a no-op."""
        # If onboarding is required, settings endpoints return 403 with
        # `onboardingRequired: true`. We probe by GETting a known setting.
        probe = self._client._client.get("/api/setting/service_type")
        if probe.status_code == 200:
            _log.info("Lingarr onboarding already complete")
            return

        if probe.status_code != 403:
            probe.raise_for_status()

        _log.info("completing Lingarr onboarding (auth disabled)")
        self._client.post(
            "/api/auth/onboarding",
            json={
                "enableUserAuth": "false",
                "request": {
                    "username": _ONBOARDING_USERNAME,
                    "password": _ONBOARDING_PASSWORD,
                },
            },
        )

    # --- Low-level setting helpers -------------------------------------

    def _get_setting(self, key: str) -> Any:
        resp = self._client.get(f"/api/setting/{key}")
        # Body is either a bare JSON string ("…") or empty for missing values.
        text = resp.text.strip()
        if not text:
            return None
        try:
            return resp.json()
        except Exception:
            return text

    def _put_setting(self, key: str, value: Any) -> None:
        self._client.post("/api/setting", json={"key": key, "value": value})
