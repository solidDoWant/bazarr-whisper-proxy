"""Radarr REST provisioner.

Strategy (per spec 16 implementer notes and verified against Radarr 6.1.1):

  1. Wait until /initialize.json returns 200; read apiKey from that JSON.
  2. Add a root folder pointing at the shared media volume's container path.
  3. POST /api/v3/movie with a *real* tmdbId. We can't use a synthetic id
     because Radarr cross-checks against TMDB before accepting the record.
     The fixture set uses public-domain films whose tmdbIds are stable
     forever (Plan 9 / Night of the Living Dead).
  4. Pre-stage the fixture .mkv at <rootFolder>/<folderName>/<filename>.
  5. POST RescanMovie command; movie.hasFile flips to true.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from tests.e2e.fixtures.build import MANIFEST_PATH
from tests.e2e.provision._http import ApiClient, wait_for_endpoint

_log = logging.getLogger(__name__)


class RadarrProvisioner:
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
        wait_for_endpoint(f"{self._base_url}/initialize.json", timeout_sec=timeout_sec)

    def bootstrap(self) -> str:
        """Fetch /initialize.json (no auth required) to learn the API key."""
        resp = wait_for_endpoint(f"{self._base_url}/initialize.json")
        body = resp.json()
        api_key = body.get("apiKey")
        if not api_key:
            raise RuntimeError(f"Radarr /initialize.json missing apiKey: {body!r}")
        self._api_key = api_key
        self._client = ApiClient(
            self._base_url, headers={"X-Api-Key": api_key, "Accept": "application/json"}
        )
        return api_key

    def _api(self) -> ApiClient:
        if self._client is None:
            raise RuntimeError("bootstrap() not called yet")
        return self._client

    # --- Manifest -------------------------------------------------------

    def load_manifest(self) -> list[dict[str, Any]]:
        raw = json.loads(MANIFEST_PATH.read_text())
        return list(raw["clips"])

    # --- Root folder ----------------------------------------------------

    def ensure_root_folder(self, container_path: str) -> None:
        existing = self._api().get_json("/api/v3/rootfolder")
        for rf in existing:
            if rf.get("path", "").rstrip("/") == container_path.rstrip("/"):
                _log.info("root folder %s already present", container_path)
                return
        self._api().post("/api/v3/rootfolder", json={"path": container_path})
        _log.info("added root folder %s", container_path)

    # --- Movies ---------------------------------------------------------

    def list_movie_ids(self) -> list[int]:
        body = self._api().get_json("/api/v3/movie")
        return [m["id"] for m in body]

    def list_movies(self) -> list[dict[str, Any]]:
        return list(self._api().get_json("/api/v3/movie"))

    def ensure_movie_imported(self, clip: dict[str, Any], media_root: str) -> dict[str, Any]:
        api = self._api()
        tmdb_id = int(clip["tmdb_id"])
        title = clip["movie_title"]
        year = int(clip["movie_year"])
        folder_name = clip["radarr_folder"]
        filename = clip["filename"]

        # Idempotency: if a movie with our tmdbId already exists, skip add.
        existing = api.get_json("/api/v3/movie")
        for m in existing:
            if m.get("tmdbId") == tmdb_id:
                _log.info("movie %s already present (id=%s)", title, m["id"])
                self._stage_fixture_file(media_root, folder_name, filename)
                self._trigger_rescan(m["id"])
                self._wait_movie_has_file(m["id"], filename)
                return m

        profiles = api.get_json("/api/v3/qualityprofile")
        if not profiles:
            raise RuntimeError("Radarr has no quality profiles configured")
        profile_id = profiles[0]["id"]

        body = {
            "title": title,
            "year": year,
            "tmdbId": tmdb_id,
            "qualityProfileId": profile_id,
            "rootFolderPath": media_root,
            "monitored": False,
            "minimumAvailability": "released",
            "folder": folder_name,
            "path": f"{media_root.rstrip('/')}/{folder_name}",
            "addOptions": {"searchForMovie": False},
        }
        resp = api.post("/api/v3/movie", json=body)
        movie = resp.json()
        _log.info("added movie %s (id=%s, tmdbId=%s)", title, movie["id"], tmdb_id)

        self._stage_fixture_file(media_root, folder_name, filename)
        self._trigger_rescan(movie["id"])
        self._wait_movie_has_file(movie["id"], filename)
        return movie

    def _stage_fixture_file(self, media_root: str, folder_name: str, filename: str) -> None:
        host_root_str = os.environ.get("MEDIA_HOST_ROOT")
        if not host_root_str:
            raise RuntimeError("MEDIA_HOST_ROOT must be set so we can stage files into the volume")
        host_root = Path(host_root_str)
        src = Path(__file__).resolve().parent.parent / "fixtures" / "media" / filename
        if not src.exists():
            raise FileNotFoundError(f"fixture media not built: {src}")

        dst_dir = host_root / folder_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / filename
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            _log.info("fixture file already staged at %s", dst)
            return
        dst.write_bytes(src.read_bytes())
        _log.info("staged fixture file %s -> %s", src.name, dst)

    def _trigger_rescan(self, movie_id: int) -> None:
        self._api().post(
            "/api/v3/command",
            json={"name": "RescanMovie", "movieId": movie_id},
        )

    def _wait_movie_has_file(self, movie_id: int, filename: str, timeout_sec: float = 60.0) -> None:
        api = self._api()
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            movie = api.get_json(f"/api/v3/movie/{movie_id}")
            mf = movie.get("movieFile") or {}
            if movie.get("hasFile") and mf.get("relativePath", "").endswith(filename):
                _log.info("movie %s now hasFile=true (file=%s)", movie_id, mf.get("relativePath"))
                return
            time.sleep(2)
        raise TimeoutError(f"movie {movie_id} did not report hasFile within {timeout_sec}s")
