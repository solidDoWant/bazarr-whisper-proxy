# bazarr-whisper-proxy — Specification

A replacement for `bazarr-openai-whisperbridge` that sits between Bazarr and a
self-hosted OpenArc (Qwen3-ASR) backend. The proxy owns audio handling, the
call to OpenArc for transcript text, **forced alignment to recover word-level
timestamps**, and SRT cue assembly. It speaks the
`whisper-asr-webservice` protocol to Bazarr and OpenAI-compatible
`/v1/audio/transcriptions` to OpenArc.

The normative HTTP contracts are defined in
[`whisperbridge-api-contract.md`](./whisperbridge-api-contract.md). This
document specifies *how we implement* them.

---

## 1. Goals & non-goals

### Goals

- Drop-in replacement for the existing `bazarr-openarc-asr` service on TCP
  `9000`. Bazarr config and `NetworkPolicy` stay untouched.
- Produce **valid SRT** that `pysrt.from_string(..., ERROR_RAISE)` accepts —
  fixing the failure mode `BAZARR Downloaded subtitles isn't valid for this
  file` that motivated this replacement.
- Recover per-cue timing locally via forced alignment, because OpenArc
  surfaces no per-segment timing in its response.
- Pure-Nix reproducible build of a small OCI image.
- Observable: structured logs, request-correlated, with timing breakdowns
  for ingest / OpenArc / align / format stages.

### Non-goals (Phase 1)

- `task=translate` handling. Phase 1 rejects with HTTP 422; Phase 2 spec'd in
  §11 once a translation backend is chosen.
- Authentication / TLS. The pod-level `NetworkPolicy` is the security
  boundary; this is unchanged from the existing bridge.
- Multi-backend support. We target OpenArc specifically. Abstracting over
  ASR backends is YAGNI until a second backend exists.
- Retry / queue / persistence. Bazarr is the orchestrator; the proxy is
  stateless and synchronous.

---

## 2. Architecture

### 2.1 Topology

The proxy runs as a **sidecar in the same Pod as OpenArc**, reaching it
over `http://localhost:8000`. Bazarr connects via the existing
`bazarr-openarc-asr:9000` `Service`.

```
                   ┌──────────────────────── Pod ────────────────────────┐
┌────────┐         │  ┌────────────────────┐    ┌────────────────────┐   │
│ Bazarr │ ──9000──┼─►│ whisper-proxy      │───►│ OpenArc            │   │
└────────┘         │  │  - FastAPI         │    │  :8000             │   │
                   │  │  - ctc-aligner     │    │                    │   │
                   │  │  - SRT assembler   │    │                    │   │
                   │  └────────────────────┘    └────────────────────┘   │
                   └─────────────────────────────────────────────────────┘
```

### 2.2 Request flow — `/asr` (transcribe)

```
Bazarr ── multipart PCM s16le 16kHz mono ──►  whisper-proxy
                                              │
                                              ├─ 1. parse query, validate task
                                              │      (task=translate → 422)
                                              │
                                              ├─ 2. buffer audio → numpy int16
                                              │      (single allocation; freed
                                              │       once aligner is done)
                                              │
                                              ├─ 3. WAV-wrap → POST /v1/audio/
                                              │      transcriptions to OpenArc
                                              │      (response_format=verbose_json)
                                              │
                                              ├─ 4. ctc-forced-aligner on
                                              │      (float32 audio, transcript)
                                              │      → list of (word, t_start, t_end)
                                              │
                                              ├─ 5. segment words → SRT cues
                                              │      per §4.5 policy
                                              │
                                              └─ 6. respond text/plain SRT
```

### 2.3 Request flow — `/detect-language`

Phase 1: re-use OpenArc. Trim to the first `LANG_DETECT_HEAD_SEC` (default
30 s), POST as transcription with `response_format=verbose_json`, read
`metrics.language`, map to ISO 639-1 alpha-2, return.

The audio is not aligned here — language detection consumes no aligner
work. Failure to map a returned name falls back to `"und"`.

---

## 3. Tech stack

| Concern        | Choice                                                                            | Reason                                                                                                |
| -------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Language       | Python 3.14                                                                       | Native fit for ASR/aligner ecosystem; matches Bazarr-side ergonomics.                                 |
| HTTP framework | FastAPI + Uvicorn (uvloop, httptools)                                             | Async, multipart-native, well-trodden.                                                                |
| HTTP client    | `httpx` (async)                                                                   | First-class async; matches FastAPI request lifecycle.                                                 |
| Config         | `pydantic-settings`                                                               | Env-var driven, validated at startup.                                                                 |
| Aligner        | `ctc-forced-aligner` + MMS (`mms-1b-all`)                                         | Multilingual (1000+), CTC-based, decoupled from ASR. We don't pull faster-whisper just for alignment. |
| ML runtime     | `torch` (CPU build by default; CUDA optional via Nix overlay)                     | Required by the aligner.                                                                              |
| Audio          | `numpy` + `soundfile`                                                             | `numpy` for in-memory PCM; `soundfile` only when we need to write WAV bytes to feed OpenArc.          |
| SRT            | Hand-rolled writer + `pysrt` for validation                                       | SRT is trivial to write; pysrt round-trip is the contract gate.                                       |
| Logging        | stdlib `logging` + `python-json-logger`                                           | Structured JSON, no heavyweight framework.                                                            |
| Tests          | `pytest`, `pytest-asyncio`, `respx` (httpx mock), `httpx.AsyncClient` for FastAPI | Standard.                                                                                             |
| Lint / format  | `ruff` (lint + format)                                                            | One tool, fast.                                                                                       |
| Type-check     | `mypy --strict` on `src/`                                                         | Catches contract drift in dataclasses / pydantic models.                                              |
| Packaging      | `pyproject.toml` (PEP 621), `uv` for dev locks                                    | `uv` is fast and Nix-friendly.                                                                        |
| Build          | **Nix flake** → OCI image via `dockerTools.streamLayeredImage`                    | Reproducible image, no Dockerfile.                                                                    |

### Why not the obvious alternatives

- **WhisperX alignment**: drags faster-whisper into the dep tree even when
  used "alignment-only," and its language coverage is narrower than MMS.
- **stable-ts / whisper-timestamped**: duplicates ASR work that OpenArc
  already does and is doing well.
- **aeneas**: sentence-level only, no word timestamps — and the cue-quality
  bar is what we're here to fix.

---

## 4. Component design

```
src/whisper_proxy/
├── __init__.py
├── __main__.py          # uvicorn entrypoint
├── app.py               # FastAPI app factory, lifespan, route wiring
├── config.py            # pydantic-settings Settings
├── logging_setup.py     # JSON logging, request-id contextvar
├── routes/
│   ├── asr.py           # POST /asr
│   ├── detect.py        # POST /detect-language
│   └── status.py        # GET /status, GET /healthz
├── audio/
│   ├── pcm.py           # PCM s16le → numpy / WAV bytes
│   └── trim.py          # head-clip helper for lang detect
├── openarc/
│   ├── client.py        # httpx.AsyncClient, /v1/audio/transcriptions, /openarc/status
│   └── models.py        # pydantic request/response shapes
├── align/
│   ├── aligner.py       # ctc-forced-aligner wrapper, lazy-loaded model
│   └── types.py         # Word, Segment dataclasses
├── srt/
│   ├── segment.py       # word → cue grouping (see §4.5)
│   └── writer.py        # cue list → SRT text, validated against pysrt
└── lang/
    └── map.py           # OpenArc name → ISO 639-1 alpha-2
```

Tests mirror the layout under `tests/`.

### 4.1 HTTP layer (`routes/`)

- `POST /asr` — parses query into a `pydantic` `AsrParams` model. On
  `task == "translate"`, returns `422` with a structured body
  (`{"detail": "translate not implemented", "code": "translate_unsupported"}`).
  On success, streams the request body into a `bytes` buffer, drives the
  pipeline, returns SRT with `Content-Type: text/plain; charset=utf-8` and
  a `Source:` header for parity with the upstream bridge.
- `POST /detect-language` — same audio handling, calls `OpenArcClient.detect_language(audio)`,
  returns `{"language_code": ..., "detected_language": ...}` per contract.
  On any failure returns `language_code: "und"`.
- `GET /status` — returns liveness JSON `{"status": "ok", "model": "<openarc-model>", "model_state": "<loaded|...>"}`.
  Used by Kubernetes probes. If OpenArc is not `loaded`, returns `503` so
  the probe restarts the Pod (matches existing behavior).
- `GET /healthz` — cheap liveness, no OpenArc call.

Bazarr never reads response status beyond connection errors, but a clean
`200` SRT vs `5xx` matters for the throttle path — see §6.4.

### 4.2 Audio (`audio/`)

- Input: raw `pcm_s16le @ 16000 Hz mono`. ~32 kB / sec. A 45-min episode is
  ~85 MB.
- Strategy:
  1. Read the full multipart field into a `bytes` buffer (single alloc;
     FastAPI uses `python-multipart` to spool to memory under the configured
     threshold, to a temp file beyond it — we set the spool threshold
     above 100 MB via `MAX_AUDIO_BYTES` to keep it in memory).
  2. Build a `numpy.ndarray[int16]` view over the bytes (no copy) for the
     aligner path (which expects `float32` — we cast once).
  3. Build WAV bytes for OpenArc by writing a 44-byte WAV header + the
     PCM bytes. No re-encoding. `soundfile.write(BytesIO, ...)` is used to
     get an exactly-correct header; benchmarked at <5 ms for 100 MB.
  4. Release the WAV `bytes` after the OpenArc POST completes; release the
     float32 array after alignment completes. Goal: peak RSS ≈ raw audio
     size + WAV bytes + float32 buffer, transient.

### 4.3 OpenArc client (`openarc/`)

- One process-wide `httpx.AsyncClient` with `timeout=httpx.Timeout(connect=5,
  read=OPENARC_READ_TIMEOUT)` where `OPENARC_READ_TIMEOUT` defaults to
  `3000` seconds (well under Bazarr's 3600).
- `transcribe(audio_wav: bytes, language: str | None) -> TranscriptionResponse`
  posts `multipart/form-data` with `file`, `model`, `response_format=verbose_json`,
  and `openarc_asr={"qwen3_asr": {"language": ...}}` when language is provided.
- `detect_language(audio_wav: bytes) -> str` — same call on a head-clipped
  audio with no language hint, returns `metrics.language`.
- `model_state() -> Literal["loaded", "loading", "unloaded", "unknown"]`
  — GET `/openarc/status`, parses the array, returns the matching model's
  state.

The client converts OpenArc's `4xx`/`5xx`/transport errors into a small
exception hierarchy (`OpenArcUnavailable`, `OpenArcBadRequest`,
`OpenArcInferenceError`); the route layer maps these into the Bazarr-facing
response per §6.4.

### 4.4 Aligner (`align/`)

- The MMS model is loaded **lazily on first request** and cached on the
  app state. Cold start is ~3–5 s; subsequent requests pay only inference.
- For each request:
  1. Convert `int16` PCM → `float32 / 32768.0`, shape `(N,)`.
  2. Run `ctc_forced_aligner.alignment(audio, transcript, language=alpha2)`.
  3. Get back per-word `(token, start, end)` tuples in seconds.
- Long-form audio: ctc-forced-aligner chunks internally; we set
  `batch_size` and `window_sec` via env config with safe defaults
  (`batch_size=16`, `window_sec=30`).
- If alignment fails (model error, all-silence audio, or a transcript that
  can't be matched), we fall back to a **single cue spanning the audio**
  with the full transcript. This keeps the output a valid SRT and lets
  Bazarr accept the subtitle rather than throttling.

### 4.5 SRT assembly (`srt/`)

Forced alignment produces word-level timing; the bridge owns the policy
for grouping words into displayable cues. Default heuristic:

1. Break on **sentence-final punctuation** (`.`, `!`, `?`, `…`, plus locale
   variants `。`, `！`, `？`).
2. Within a sentence, break on **inter-word silence ≥ `cue_silence_ms`**
   (default 700 ms) when the gap straddles a comma / clause boundary.
3. Otherwise, force-break when the running cue exceeds **`max_cue_chars`**
   (default 84 chars, two lines × 42 chars per BBC subtitle guidelines) or
   **`max_cue_sec`** (default 6 s).
4. Enforce a **minimum cue duration of `min_cue_sec`** (default 1.0 s),
   extending the trailing edge into the next gap if needed (clamped to
   the next cue's start).
5. Line-wrap within a cue at word boundaries to keep ≤2 lines.

The writer emits canonical SRT:

```
<index>
HH:MM:SS,mmm --> HH:MM:SS,mmm
<line 1>
<line 2 (optional)>

```

Validation gate: after writing, call `pysrt.from_string(text, error_handling=pysrt.ERROR_RAISE)`
in tests and in a debug-mode runtime check. CI fails on a regression.

All policy knobs are env-driven (§5).

### 4.6 Language map (`lang/map.py`)

OpenArc returns `metrics.language` as a human-readable English name
(`"English"`, `"Spanish"`, …). We translate via a static map (the
ISO 639 short-name table). Unknown names → `"und"`. The map is committed
as data (`lang/names_iso639.json`), not generated at runtime.

---

## 5. Configuration

All config via env vars, parsed by `pydantic-settings` at startup.
Validation errors abort boot.

| Env var                   | Default                    | Notes                                       |
| ------------------------- | -------------------------- | ------------------------------------------- |
| `OPENARC_BASE_URL`        | `http://localhost:8000`    | OpenArc reachable URL.                      |
| `OPENARC_MODEL`           | `qwen3-asr-0_6b-int8-asym` | Must match autoloaded model.                |
| `OPENARC_READ_TIMEOUT`    | `3000`                     | Seconds. Must be < Bazarr's 3600.           |
| `OPENARC_CONNECT_TIMEOUT` | `5`                        | Seconds.                                    |
| `ALIGNER_MODEL`           | `mms-1b-all`               | HF id, downloaded once at image build (§7). |
| `ALIGNER_BATCH_SIZE`      | `16`                       | Inference batch.                            |
| `ALIGNER_WINDOW_SEC`      | `30`                       | Chunk size for long-form.                   |
| `LANG_DETECT_HEAD_SEC`    | `30`                       | Audio prefix length for `/detect-language`. |
| `CUE_MAX_CHARS`           | `84`                       | Hard cue char cap.                          |
| `CUE_MAX_SEC`             | `6.0`                      | Hard cue duration cap.                      |
| `CUE_MIN_SEC`             | `1.0`                      | Soft cue duration floor.                    |
| `CUE_SILENCE_MS`          | `700`                      | Inter-word gap that allows a clause break.  |
| `MAX_AUDIO_BYTES`         | `200_000_000`              | Reject larger uploads with 413.             |
| `LOG_LEVEL`               | `INFO`                     | Stdlib level name.                          |
| `LOG_FORMAT`              | `json`                     | `json` or `text`.                           |

A `config.example.env` ships in the repo for local dev.

---

## 6. Cross-cutting behavior

### 6.1 Logging & correlation

- Each inbound request gets a `request_id` (UUID4) on a contextvar; it's
  attached to every log record and returned in `X-Request-Id`.
- Per-stage timings logged at INFO once per request:
  `ingest_ms / openarc_ms / align_ms / format_ms / total_ms`.
- `video_file` (when present in query) is logged at INFO so operators can
  tie a request to a specific media file (per contract §4).

### 6.2 Concurrency

- FastAPI / Uvicorn handles concurrency. The aligner is CPU-heavy and
  PyTorch holds the GIL during inference; we run **`uvicorn --workers 1
  --threads 1`** by default and rely on Bazarr's single-flight behavior
  (it does not parallelize provider calls per item).
- An asyncio `Semaphore(1)` around alignment guards against concurrent
  request bursts overloading CPU/RAM. Queue depth is logged.

### 6.3 Resource budget

- Audio buffer: ≤ `MAX_AUDIO_BYTES` (200 MB default).
- WAV bytes: same.
- Float32 audio array: ~2 × audio bytes (transient).
- Aligner model: ~1 GB resident.
- Set Pod requests/limits to roughly `1.5 GB` RAM, `2 CPU` request, `4 CPU`
  limit. Exact values out of scope here.

### 6.4 Error mapping

Bazarr's failure modes are asymmetric: **5xx triggers a 24-hour throttle**;
4xx does not. We map deliberately:

| Condition                                    | Bazarr-facing status               | Body                                                                       |
| -------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------- |
| `task=translate`                             | `422`                              | `{"detail": "translate not implemented", "code": "translate_unsupported"}` |
| `MAX_AUDIO_BYTES` exceeded                   | `413`                              | `{"detail": "audio too large"}`                                            |
| OpenArc model not loaded (`/openarc/status`) | `503` (with `Retry-After: 30`)     | `{"detail": "model loading"}`                                              |
| OpenArc 4xx                                  | `502`                              | Pass-through detail. (Acceptable to throttle — operator config error.)     |
| OpenArc 5xx / transport error                | `502`                              | But include `Retry-After`; this *will* throttle Bazarr. Logged as ERROR.   |
| Aligner failure                              | `200` with fallback single-cue SRT | Logged WARN; valid SRT keeps the user moving.                              |
| Invalid SRT produced internally (sanity)     | `500`                              | Logged ERROR. Should never happen — gated in tests.                        |

The `Retry-After: 30` on `503` is intentional: Bazarr treats 503 as a
transient failure without applying the throttle, so a model-cold restart
won't lock the provider out for 24 hours.

### 6.5 Streaming the response

The contract notes Bazarr reads `r.content` synchronously and gives us up
to 3600 s. We **buffer the full SRT** before responding — it's tiny
(KB-scale) and streaming would complicate the validator gate. The
*request* side may already be buffered (multipart spool), but the
processing happens once the body is fully received.

---

## 7. Build & deploy — Nix flake

The repo ships a `flake.nix` (no Dockerfile).

### 7.1 Outputs

```
flake outputs:
  packages.${system}.default        # the Python app as a Nix package
  packages.${system}.dockerImage    # streamLayeredImage tarball
  packages.${system}.aligner-model  # MMS weights, fetchurl with sha256
  devShells.${system}.default       # full dev env: python, uv, ruff, mypy, ffmpeg, …
  checks.${system}.tests            # pytest run inside Nix
  checks.${system}.lint             # ruff + mypy
```

### 7.2 Approach

- Use **`uv2nix`** (or `poetry2nix` fallback) to derive the Python env from
  `pyproject.toml` + `uv.lock`. Pin `nixpkgs` via flake input.
- The aligner model weights are a separate Nix derivation (`fetchurl` +
  `sha256`), placed at a fixed path inside the image. The runtime config
  points `ALIGNER_MODEL_PATH` at that path so the image needs no internet
  at start.
- The image is built with `pkgs.dockerTools.streamLayeredImage`:
  - base layer: glibc + tini + ca-certificates
  - python layer: interpreter + site-packages (mostly torch — biggest)
  - model layer: aligner weights (rarely changes; cached)
  - app layer: our source (changes every commit)
- Entrypoint: `tini -- python -m whisper_proxy`. Exposed port `9000`.
- Image size target: **< 3 GB** (CPU-only torch is the biggest contributor;
  GPU image is a separate output gated by `cuda` flag).

### 7.3 CI

GitHub Actions workflow:

1. `nix flake check` — runs `checks.tests` and `checks.lint`.
2. `nix build .#dockerImage` — produces the image tarball.
3. On `main`, push to GHCR with tags `:latest`, `:<short-sha>`,
   `:<conventional-commit-version>` (if release).

Local dev: `nix develop`, then `uv run python -m whisper_proxy`. The
devshell also provides an `mock-openarc` script that runs a minimal
FastAPI stub on `:8000` returning canned `verbose_json` — lets you
iterate without OpenArc up.

---

## 8. Testing strategy

Three layers, all run by `nix flake check`:

### 8.1 Unit

- `srt/segment.py`: grouping heuristic against fixture word-streams. Cases:
  short sentences, very long sentences, all-silence input, no punctuation
  input (e.g., `qwen3` output without final punctuation), CJK punctuation.
- `srt/writer.py`: output round-trips through `pysrt.from_string(...,
  ERROR_RAISE)`.
- `lang/map.py`: known names → alpha-2, unknown → `"und"`, case-insensitive.
- `audio/pcm.py`: WAV header byte-exact; round-trip via `soundfile.read`
  matches input samples.

### 8.2 Integration (in-process)

- `httpx.AsyncClient` against the FastAPI app + `respx`-mocked OpenArc.
- Fixtures:
  - Small (~5 s) PCM clip of public-domain speech, committed under
    `tests/fixtures/`.
  - Canned OpenArc `verbose_json` responses for that clip.
- Tests:
  - `POST /asr` with `task=transcribe` → SRT parses cleanly, cue count > 0,
    timing within audio duration.
  - `POST /asr` with `task=translate` → `422`.
  - `POST /detect-language` → expected alpha-2 code.
  - `GET /status` → 200 when OpenArc reports loaded; 503 when not.
  - Aligner failure path → 200 with single-cue fallback.
  - OpenArc 503 from `/openarc/status` → bridge 503 with `Retry-After`.

The aligner **is real** in integration tests (CPU, ~3 s overhead — fine).
This is the most important behavioral surface and worth exercising end-to-end.

### 8.3 Contract

- A `tests/contract/` set replays the exact request shape Bazarr sends
  (per `whisperbridge-api-contract.md` §1.1 / §1.2) byte-for-byte and
  asserts response shape per §1.1 / §1.2. These are the "if these fail,
  Bazarr will reject the subtitle" tests.

### 8.4 Manual smoke

Documented in `docs/smoke.md` (created later): with the cluster up, run a
single Bazarr provider search against a known short episode and verify
the SRT lands.

---

## 9. Observability hooks

Lightweight, no Prometheus dep in Phase 1:

- Structured JSON logs to stdout (caught by the cluster log pipeline).
- A `GET /metrics` endpoint is **out of scope** for Phase 1 (revisit if
  operators ask). The per-stage timings in logs cover the immediate need.

---

## 10. Repo layout (top-level)

```
.
├── SPEC.md                       # this document
├── whisperbridge-api-contract.md
├── README.md                     # short: what it is, how to run via nix
├── flake.nix
├── flake.lock
├── pyproject.toml
├── uv.lock
├── config.example.env
├── src/whisper_proxy/...         # see §4
├── tests/...
├── docs/
│   ├── smoke.md                  # manual smoke recipe
│   └── adr/                      # architecture decision records as they accrue
└── .github/workflows/
    ├── ci.yaml
    └── release.yaml
```

Kubernetes manifests are **not** committed here. The existing OpenArc
`HelmRelease` lives in the GitOps repo; the proxy is added to that Pod
spec there. We can revisit if operations diverge.

---

## 11. Phase 2 — `task=translate`

Phase 1 rejects with 422. Phase 2 will:

1. Transcribe in source language via OpenArc (Phase 1 path).
2. POST the resulting transcript to a configurable translation backend
   (env `TRANSLATE_BASE_URL`, OpenAI-compat chat completions).
3. Re-run the forced aligner with the **translated** text against the
   **original audio** — wrong, since words don't line up.
   So actually: **align in source language first**, then translate cue-by-cue,
   keeping the source-language timing. This preserves cue boundaries that
   match what's actually said; translated cues will be slightly off
   length-wise but timing stays grounded in audio.
4. Add tests with a known source-EN / target-EN translation pair (trivial),
   then a source-ES / target-EN pair using a small fixture.

Open question for Phase 2: which translation backend? Not decided. Likely
candidates: an in-cluster LiteLLM proxy, a local LLM via vLLM, or the
existing OpenArc instance with a different model loaded. Tracked as ADR
when we get there.

---

## 12. Open questions / explicit deferrals

1. **CPU-only image size**. Torch CPU wheels are ~700 MB. If we need a
   smaller image, we can swap to `torch-cpu-slim` builds or vendor a
   pruned wheel. Defer until image size becomes a problem.
2. **GPU image**. The aligner runs fine on CPU at this audio scale (3–10 s
   per episode). A GPU variant is a `cuda = true` flake input away. Build
   it on demand, not by default.
3. **Per-language SRT line-wrap rules**. The 42-char default is BBC-Latin.
   CJK has different conventions (char count, not char width). Phase 1
   ships the Latin defaults; revisit if/when users complain about Japanese
   or Chinese cues.
4. **OpenArc model warm-up on container start**. Today, the model loads on
   the first request to `/v1/audio/transcriptions`. We could prime it from
   the proxy's `lifespan` (call `/openarc/load`) to avoid first-request
   latency, but the contract says we shouldn't touch those endpoints. Leave
   as-is; rely on OpenArc's `OPENARC_AUTOLOAD_MODEL`.
5. **Streaming response to Bazarr**. Currently buffered. Streaming is
   trivial to add but earns us nothing measurable. Defer indefinitely.

---

## 13. What "done" looks like for Phase 1

- `nix build .#dockerImage` produces an image tagged `:0.1.0`.
- Bazarr, pointed at the new sidecar, downloads SRT for a known 45-minute
  episode without `Downloaded subtitles isn't valid for this file` showing
  up in Bazarr's logs.
- `pysrt.from_string` accepts every SRT the proxy produces across the
  integration test corpus.
- `nix flake check` passes locally and in CI.
- Logs show `ingest_ms / openarc_ms / align_ms / format_ms / total_ms`
  for every request.

Phase 2 (`translate`) and any additional observability bits land
incrementally on top.
