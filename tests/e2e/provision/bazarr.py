"""Bazarr REST provisioner.

API quirks discovered against Bazarr 1.5.6:

- Auth header is `X-API-KEY` (uppercase).
- `/api/system/settings` accepts form-urlencoded POSTs with keys shaped
  `settings-<section>-<field>`, e.g. `settings-general-use_radarr=true`.
  JSON POSTs are silently ignored.
- Booleans are accepted as the strings "true"/"false".
- Lists pass through as repeated form keys (multiple
  `settings-general-enabled_providers=whisperai` entries).
- Language profiles are created via the same settings endpoint, with a
  JSON-encoded array under `languages-profiles`.
- Tasks fire via `POST /api/system/tasks` with `taskid=<job_id>`.
- Subtitle download triggers via
  `PATCH /api/movies/subtitles?radarrid=X&language=Y&forced=False&hi=False`.

The API key is read from `/config/config/config.yaml` inside the container
via `docker compose exec`.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from typing import Any

from tests.e2e.provision._http import ApiClient, wait_for_endpoint

_log = logging.getLogger(__name__)


_BAZARR_SERVICE = "bazarr"
_BAZARR_CONFIG_PATH = "/config/config/config.yaml"
_PROVIDER_NAME = "whisperai"
_COMPOSE_FILE = "compose.e2e.yml"


class BazarrProvisioner:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key: str | None = None
        self._client: ApiClient | None = None

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            raise RuntimeError("bootstrap() not called yet")
        return self._api_key

    def wait_ready(self, timeout_sec: float = 240.0) -> None:
        # /api/system/status is auth-gated; 401 means "Bazarr is up but our
        # request lacks an X-API-KEY". For readiness probing that's fine —
        # we just need the HTTP stack to answer.
        wait_for_endpoint(
            f"{self._base_url}/api/system/status",
            timeout_sec=timeout_sec,
            expect_status=(200, 401),
        )

    def bootstrap(self) -> str:
        api_key = self._read_apikey_from_container()
        self._api_key = api_key
        self._client = ApiClient(
            self._base_url, headers={"X-API-KEY": api_key, "Accept": "application/json"}
        )
        return api_key

    def _read_apikey_from_container(self, retries: int = 30) -> str:
        docker = shutil.which("docker") or "docker"
        for attempt in range(retries):
            try:
                out = subprocess.run(
                    [
                        docker,
                        "compose",
                        "-f",
                        _COMPOSE_FILE,
                        "exec",
                        "-T",
                        _BAZARR_SERVICE,
                        "cat",
                        _BAZARR_CONFIG_PATH,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                _log.debug("attempt %d: cat config failed: %s", attempt + 1, exc.stderr.strip())
                time.sleep(2)
                continue

            match = re.search(r"^\s*apikey:\s*([0-9a-f]+)", out.stdout, re.MULTILINE)
            if match:
                return match.group(1)
            _log.debug("attempt %d: apikey not yet written", attempt + 1)
            time.sleep(2)

        raise RuntimeError("Bazarr api key not found in config.yaml after retries")

    def _api(self) -> ApiClient:
        if self._client is None:
            raise RuntimeError("bootstrap() not called yet")
        return self._client

    # --- Settings -------------------------------------------------------

    def fetch_settings(self) -> dict[str, Any]:
        return dict(self._api().get_json("/api/system/settings"))

    def _post_settings_form(self, fields: list[tuple[str, str]]) -> None:
        """Form-encoded settings POST. Bazarr expects keys like settings-section-field."""
        from urllib.parse import urlencode

        body = urlencode(fields)
        self._api().post(
            "/api/system/settings",
            content=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    # --- Radarr connection ---------------------------------------------

    def ensure_radarr_connection(self, radarr_url: str, radarr_api_key: str) -> None:
        settings = self.fetch_settings()
        radarr_cfg = settings.get("radarr", {})
        host, port = _split_host_port(radarr_url, default_port=7878)
        already = (
            radarr_cfg.get("ip") == host
            and radarr_cfg.get("port") == port
            and radarr_cfg.get("apikey") == radarr_api_key
            and settings.get("general", {}).get("use_radarr") is True
        )
        if already:
            _log.info("Bazarr -> Radarr connection already configured")
            return

        self._post_settings_form(
            [
                ("settings-general-use_radarr", "true"),
                ("settings-radarr-ip", host),
                ("settings-radarr-port", str(port)),
                ("settings-radarr-base_url", "/"),
                ("settings-radarr-ssl", "false"),
                ("settings-radarr-apikey", radarr_api_key),
                ("settings-radarr-only_monitored", "false"),
            ]
        )
        _log.info("configured Bazarr -> Radarr (%s:%s)", host, port)

    def trigger_radarr_sync(self) -> None:
        """Kick the 'Sync with Radarr' background task. Idempotent (no-op if already running)."""
        from urllib.parse import urlencode

        self._api().post(
            "/api/system/tasks",
            content=urlencode([("taskid", "update_movies")]),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        _log.info("triggered Bazarr 'update_movies' task")

    # --- Whisper provider ----------------------------------------------

    def ensure_whisper_provider(self, bridge_url: str) -> None:
        settings = self.fetch_settings()
        general = settings.get("general", {})
        providers = list(general.get("enabled_providers", []))
        whisperai = settings.get("whisperai", {})

        already = _PROVIDER_NAME in providers and whisperai.get("endpoint", "").rstrip(
            "/"
        ) == bridge_url.rstrip("/")
        if already:
            _log.info("whisperai provider already configured at %s", bridge_url)
            return

        if _PROVIDER_NAME not in providers:
            providers.append(_PROVIDER_NAME)

        fields: list[tuple[str, str]] = []
        for p in providers:
            fields.append(("settings-general-enabled_providers", p))
        fields.extend(
            [
                ("settings-whisperai-endpoint", bridge_url),
                # Bazarr's own client-side read timeout (seconds) — well above
                # what we expect a 10-30s clip to need but below our own
                # OPENARC_READ_TIMEOUT so Bazarr doesn't time out first.
                ("settings-whisperai-response", "1800"),
                ("settings-whisperai-timeout", "1800"),
                ("settings-whisperai-loglevel", "INFO"),
                # Forward the video_file query param so our bridge can pass it
                # through to Lingarr for cross-cue context.
                ("settings-whisperai-pass_video_name", "true"),
            ]
        )
        self._post_settings_form(fields)
        _log.info("enabled whisperai provider, endpoint=%s", bridge_url)

    # --- Language profiles ---------------------------------------------

    def ensure_language_profile(
        self,
        profile_id: int,
        name: str,
        language_alpha2: str,
    ) -> None:
        """Idempotently create a single-language profile (movie-side)."""
        existing = self._api().get_json("/api/system/languages/profiles")
        for prof in existing:
            if prof.get("profileId") == profile_id:
                items = prof.get("items", [])
                if items and items[0].get("language") == language_alpha2:
                    _log.info("language profile %s already present", name)
                    return
                break

        # Strings are JSON-encoded literals because Bazarr's parser inspects
        # them character-by-character.
        # Bazarr 1.5.6+ added 'audio_only_include' (per-language gate that
        # only fetches subs when audio matches). Older Bazarr used
        # 'audio_exclude' (the inverse). Send both so older / newer Bazarr
        # both accept the profile.
        profile = {
            "profileId": profile_id,
            "name": name,
            "items": [
                {
                    "id": 1,
                    "language": language_alpha2,
                    "forced": "False",
                    "hi": "False",
                    "audio_exclude": "False",
                    "audio_only_include": "False",
                }
            ],
            "cutoff": None,
            "mustContain": [],
            "mustNotContain": [],
            "originalFormat": False,
            "tag": "",
        }
        # Always send the full profile list (the setting is replace-all).
        all_profiles = [p for p in existing if p.get("profileId") != profile_id] + [profile]
        self._post_settings_form(
            [
                ("languages-profiles", json.dumps(all_profiles)),
            ]
        )
        _log.info(
            "created/updated language profile %s (id=%d, lang=%s)",
            name,
            profile_id,
            language_alpha2,
        )

    def assign_movie_profile(self, radarr_id: int, profile_id: int) -> None:
        """Assign a language profile to a movie. POST /api/movies?radarrid=&profileid=.

        Bazarr's POST /api/movies uses Flask reqparse with `action='append'`,
        which expects repeated bare keys (`profileid=1`), NOT bracket arrays
        (`profileid[]=1`) — the latter quietly returns 204 without doing
        anything.
        """
        self._api().post(
            "/api/movies",
            params=[("radarrid", str(radarr_id)), ("profileid", str(profile_id))],
        )
        _log.info("assigned profile %d to movie radarrid=%d", profile_id, radarr_id)

    # --- Provider throttle management ----------------------------------

    def reset_provider_throttle(self) -> None:
        """Clear Bazarr's in-memory provider throttle list.

        Bazarr throttles a provider for 24 h after a connection failure.  In
        the e2e suite this can be triggered by Bazarr's background subtitle
        scan racing against a slow/busy OpenArc response.  Calling this
        resets the throttle so the next explicit provider call succeeds.

        API: ``POST /api/providers?action=reset`` → 204.
        """
        self._api().post("/api/providers", params=[("action", "reset")])
        _log.info("reset Bazarr provider throttle")

    # --- Subtitle search trigger ---------------------------------------

    def trigger_subtitle_download(
        self,
        radarr_id: int,
        language_alpha2: str,
        *,
        forced: bool = False,
        hi: bool = False,
        provider: str = _PROVIDER_NAME,
    ) -> None:
        """Synchronously search providers + download the first match.

        Uses the two-step ``GET /api/providers/movies → POST /api/providers/movies``
        flow rather than ``PATCH /api/movies/subtitles``. The latter queues a
        background job and returns 204 immediately; for our setup it completes
        the job without ever calling the provider (Bazarr's `movies_scan_subtitles`
        helper short-circuits when there are no embedded subs and the audio
        language matches the wanted language). The GET+POST path forces a
        real provider call.

        If the first candidate lookup returns empty (Bazarr's background scan
        may have raced and throttled the provider), we reset the throttle and
        retry once before raising.
        """
        api = self._api()

        def _get_matches() -> list[dict]:
            body = api.get_json(f"/api/providers/movies?radarrid={radarr_id}")
            return [
                c
                for c in body["data"]
                if c.get("provider") == provider and c.get("language") == language_alpha2
            ]

        matches = _get_matches()
        if not matches:
            # Bazarr's periodic background scan can race here: it detects a
            # newly-assigned profile, tries the provider, gets a transient
            # connection failure, and throttles it for 24 h — all before this
            # explicit trigger fires.  Reset and retry once.
            _log.warning(
                "no %s candidate for radarrid=%d lang=%s — resetting provider throttle and retrying",
                provider,
                radarr_id,
                language_alpha2,
            )
            self.reset_provider_throttle()
            matches = _get_matches()

        if not matches:
            raise RuntimeError(
                f"no {provider} candidate for radarrid={radarr_id} language={language_alpha2}"
                " (even after throttle reset — check Bazarr logs)"
            )
        chosen = matches[0]
        params = [
            ("radarrid", str(radarr_id)),
            ("hi", "True" if hi else "False"),
            ("forced", "True" if forced else "False"),
            ("original_format", "False"),
            ("provider", provider),
            ("subtitle", chosen["subtitle"]),
        ]
        api.post("/api/providers/movies", params=params)
        _log.info(
            "triggered subtitle download radarrid=%d lang=%s provider=%s",
            radarr_id,
            language_alpha2,
            provider,
        )

    # --- Inventory snapshot --------------------------------------------

    def list_movies(self) -> list[dict[str, Any]]:
        body = self._api().get_json("/api/movies")
        if isinstance(body, dict) and "data" in body:
            return list(body["data"])
        return list(body)


def _split_host_port(url: str, default_port: int) -> tuple[str, int]:
    stripped = re.sub(r"^https?://", "", url).split("/")[0]
    if ":" in stripped:
        host, port = stripped.split(":", 1)
        return host, int(port)
    return stripped, default_port
