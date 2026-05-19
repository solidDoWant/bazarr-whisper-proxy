# bazarr-whisper-proxy

Drop-in replacement for `bazarr-openai-whisperbridge`. Sits between
[Bazarr](https://github.com/morpheus65535/bazarr) and a self-hosted
[OpenArc](https://github.com/morioka/openarc) (Qwen3-ASR) backend, adding
forced alignment for accurate per-cue timing and valid SRT output.

Speaks the `whisper-asr-webservice` protocol to Bazarr and the OpenAI-compat
`/v1/audio/transcriptions` API to OpenArc.

## How to run

### Prerequisites

- [Nix](https://nixos.org/download/) with flakes enabled

### Development

```sh
# Enter the dev shell — provides Python 3.15, uv, ruff, mypy, pytest
nix develop

# Install Python dependencies into a local .venv
uv sync

# Start the server (binds to 0.0.0.0:9000)
uv run python -m whisper_proxy

# Verify
curl -sf http://localhost:9000/healthz
# → {"status":"ok"}
```

### Lint, type-check, and test

```sh
# Inside nix develop:
ruff check src tests
ruff format --check src tests
mypy --strict src
pytest tests/

# Or run everything via Nix (hermetic):
nix flake check
```

### Configuration

Copy `config.example.env` to `.env` and adjust. All values have defaults
suitable for local development (OpenArc expected at `http://localhost:8000`).

```sh
cp config.example.env .env
# edit .env as needed
uv run python -m whisper_proxy
```

## Architecture

See [SPEC.md](./SPEC.md) for the full specification and
[whisperbridge-api-contract.md](./whisperbridge-api-contract.md) for the
HTTP contract.
