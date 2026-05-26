#!/usr/bin/env bash
# End-to-end test harness — brings up the compose stack, runs provisioning,
# executes the pytest e2e suite, and tears down.
#
# Usage:
#   e2e                  # default: full suite against a working OpenArc
#   e2e failure-modes    # extra mode for criterion 16 (black-hole OpenArc)
#   e2e --keep-up        # don't tear the stack down on exit (debugging)
#
# Required env var:
#   OPENARC_E2E_BASE_URL   URL of a reachable remote OpenArc (e.g.
#                          http://bazarr-openarc.media.svc.cluster.local:8000)

set -euo pipefail

MODE="${1:-default}"
KEEP_UP=false
if [[ "${1:-}" == "--keep-up" || "${2:-}" == "--keep-up" ]]; then
  KEEP_UP=true
  if [[ "${1:-}" == "--keep-up" ]]; then
    MODE="default"
  fi
fi

# ---- Repo root --------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

# ---- Sanity checks ----------------------------------------------------
if [[ "${MODE}" != "failure-modes" && -z "${OPENARC_E2E_BASE_URL:-}" ]]; then
  echo "ERROR: OPENARC_E2E_BASE_URL is not set." >&2
  echo "  Point it at a reachable remote OpenArc, e.g.:" >&2
  echo "    export OPENARC_E2E_BASE_URL=http://your-openarc-host:8000" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not on PATH. Enter \`nix develop\` first." >&2
  exit 2
fi

# ---- Mode-specific env ------------------------------------------------
COMPOSE_FILE="compose.e2e.yml"
case "${MODE}" in
  default)
    : "${OPENARC_E2E_BASE_URL:?}"
    ;;
  failure-modes)
    # Black-hole IP: TEST-NET-1 (RFC 5737), never routable on the internet.
    export OPENARC_E2E_BASE_URL="http://192.0.2.42:8000"
    export E2E_OPENARC_BLACKHOLE=1
    echo "[e2e] failure-modes: OpenArc is black-holed at ${OPENARC_E2E_BASE_URL}"
    ;;
  *)
    echo "ERROR: unknown mode ${MODE}" >&2
    exit 2
    ;;
esac

# ---- Build + load bridge image ----------------------------------------
echo "[e2e] building whisper-proxy image via nix"
IMAGE_STREAM="$(nix --extra-experimental-features 'nix-command flakes' \
  build --print-out-paths --no-link .#dockerImage)"
echo "[e2e] loading image into the local Docker daemon"
"${IMAGE_STREAM}" | docker load

# ---- Volume host paths ------------------------------------------------
# Compose names volumes like <project>_<volume>. Resolve the host path so
# we can stage fixture files and inspect produced .srt files from pytest.
COMPOSE_PROJECT="bazarr-whisper-proxy-e2e"

# ---- Bring stack up ---------------------------------------------------
cleanup() {
  if [[ "${KEEP_UP}" == "true" ]]; then
    echo "[e2e] --keep-up specified; leaving stack running."
    return
  fi
  echo "[e2e] tearing stack down (down -v)"
  docker compose -f "${COMPOSE_FILE}" down -v --remove-orphans || true
}
trap cleanup EXIT

echo "[e2e] starting compose stack"
docker compose -f "${COMPOSE_FILE}" up -d

# Resolve the host path of the media volume so the provisioner can stage
# fixture files into it.
MEDIA_VOLUME="${COMPOSE_PROJECT}_media"
# In some container environments (e.g. k8s pods with a Docker socket) the
# volume root is owned by root with 0700.  Open traversal before the -d check
# so the existence test doesn't false-negative.
sudo chmod -R o+rx /tmp/docker-data 2>/dev/null || true
MEDIA_HOST_ROOT="$(docker volume inspect "${MEDIA_VOLUME}" -f '{{ .Mountpoint }}')"
if [[ -z "${MEDIA_HOST_ROOT}" || ! -d "${MEDIA_HOST_ROOT}" ]]; then
  echo "ERROR: could not resolve host path of media volume ${MEDIA_VOLUME}" >&2
  exit 1
fi
echo "[e2e] media volume host path: ${MEDIA_HOST_ROOT}"
# Docker creates anonymous volumes root-owned; the linuxserver images run
# as PUID=1000 ('abc') and need to write into /media. Chown from inside any
# container that mounts it (Radarr is up first via depends_on).
docker compose -f "${COMPOSE_FILE}" exec -u 0 -T radarr chown -R abc:abc /media
docker compose -f "${COMPOSE_FILE}" exec -u 0 -T radarr chmod -R 0775 /media
# Re-open traversal after the in-container chown in case it tightened perms.
( sudo chmod -R o+rx /tmp/docker-data 2>/dev/null \
  || chmod -R o+rx "${MEDIA_HOST_ROOT}" 2>/dev/null \
  || true )

# ---- Provision --------------------------------------------------------
export RADARR_URL="http://127.0.0.1:7878"
export BAZARR_URL="http://127.0.0.1:6767"
# LINGARR_URL may be set by the caller (external Lingarr); fall back to in-compose default.
export LINGARR_URL="${LINGARR_URL:-http://127.0.0.1:9876}"
export BRIDGE_URL="http://127.0.0.1:9000"
export RADARR_IN_COMPOSE_URL="http://radarr:7878"
export BRIDGE_IN_COMPOSE_URL="http://host.docker.internal:9000"
export LIBRETRANSLATE_IN_COMPOSE_URL="http://libretranslate:5000"
export MEDIA_HOST_ROOT
export MEDIA_CONTAINER_ROOT="/media"

echo "[e2e] running provisioning step"
python -m tests.e2e.provision

# ---- Pytest -----------------------------------------------------------
echo "[e2e] running pytest suite"
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}" pytest -v tests/e2e/

# ---- Criterion 16: restart bridge against a black-hole OpenArc -------
# test_16 requires the bridge to be up against an unreachable OpenArc so
# /status returns 503.  We restart only the whisper-proxy service here
# (other services keep running) to avoid re-provisioning.
if [[ "${MODE}" == "default" ]]; then
  echo "[e2e] criterion 16: restarting whisper-proxy with black-hole OpenArc"
  OPENARC_E2E_BASE_URL="http://192.0.2.42:8000" \
    docker compose -f "${COMPOSE_FILE}" up -d --no-deps --force-recreate whisper-proxy

  echo "[e2e] waiting for whisper-proxy to become healthy (blackhole mode)"
  _bh_deadline=$(( $(date +%s) + 60 ))
  until curl -fsS --max-time 2 "http://127.0.0.1:${WHISPER_PROXY_PORT:-9000}/healthz" >/dev/null 2>&1; do
    if (( $(date +%s) > _bh_deadline )); then
      echo "ERROR: whisper-proxy did not become healthy in 60 s (blackhole mode)" >&2
      exit 1
    fi
    sleep 2
  done

  echo "[e2e] running criterion 16 (blackhole)"
  E2E_OPENARC_BLACKHOLE=1 PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}" \
    pytest -v tests/e2e/test_failure_modes.py::test_16_status_503_when_openarc_unreachable
fi

echo "[e2e] suite complete"
