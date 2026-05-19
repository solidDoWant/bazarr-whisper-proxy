# Whisper-bridge replacement: API contract reference

A reference for the inbound (Bazarr → bridge) and outbound (bridge → OpenArc)
HTTP contracts the replacement tool must implement. Pulled from the actual
deployed code:

- Bazarr provider: `subliminal_patch/providers/whisperai.py` shipped in the
  `bazarr-9cfc79c64-fp25x` pod.
- Validity check: `subliminal_patch/subtitle.py::Subtitle.is_valid` and
  `bazarr/subtitles/manual.py:183`.
- OpenArc: tag `main-8856d1d9-r4` (upstream rev `8856d1d9`).

## Topology

```
┌────────┐       multipart POST       ┌──────────────────┐       multipart POST       ┌─────────┐
│ Bazarr │ ─────────────────────────► │     Bridge       │ ─────────────────────────► │ OpenArc │
│ whisper│      /asr  /detect-        │   (this tool)    │      /v1/audio/            │  :8000  │
│provider│      language              │                  │      transcriptions        │         │
└────────┘ ◄───────────────────────── │  + forced        │ ◄───────────────────────── └─────────┘
            text/plain SRT body       │    alignment     │  application/json
                                      └──────────────────┘
```

The bridge replaces the upstream `bazarr-openai-whisperbridge` image. It owns
audio decoding for the aligner, the call to OpenArc for the transcript text,
and SRT assembly. The Bazarr- and OpenArc-facing contracts are independent —
this doc fully specifies both.

## Service expectations

- Listens on TCP `9000` (Bazarr's whisper provider is configured with this
  endpoint; matches the existing service port `bazarr-openarc-asr:9000`).
- Plain HTTP — no TLS termination required in-cluster.
- No authentication. The bridge sits behind a `NetworkPolicy` that only
  permits Bazarr ingress.

## 1. Inbound: Bazarr → bridge

Bazarr's `WhisperAIProvider` uses two endpoints. Field semantics below follow
the `whisper-asr-webservice` API (the protocol Bazarr was originally written
against); paths and parameter names are normative and cannot be changed.

### 1.1 POST `/asr` — transcription / translation

Issued from `WhisperAIProvider.download_subtitle`:

```python
self.session.post(
    f"{self.endpoint}/asr",
    params={
        "task": subtitle.task,        # "transcribe" or "translate"
        "language": input_language,   # ISO 639-1 alpha-2 (e.g. "en"); see notes
        "output": "srt",              # always "srt" — hardcoded by Bazarr
        "encode": "false",            # always "false"
        "video_file": video_name,     # absolute video path or None
    },
    files={"audio_file": out},        # raw PCM s16le, 16 kHz, mono
    timeout=(self.response, self.timeout),
)
```

**Query parameters**

| Name         | Required | Values                                  | Notes |
|--------------|----------|-----------------------------------------|-------|
| `task`       | yes      | `transcribe` \| `translate`             | `translate` means "source audio → English subtitles". Always English target. |
| `language`   | yes      | ISO 639-1 alpha-2 (`en`, `es`, `fr`, …) | The audio's spoken language. For `task=translate` this is the source. |
| `output`     | yes      | `srt`                                   | Bazarr always sends `srt`. Accept the value but produce SRT regardless. |
| `encode`     | yes      | `false`                                 | Bazarr always sends `false` and pre-decodes audio. Accept `true` only if you want to add server-side ffmpeg later. |
| `video_file` | optional | absolute path string or omitted         | Bazarr sets this only when its "Pass video file name" toggle is on. Logging / heuristics only — the file is NOT accessible from inside the bridge pod. Do not rely on it. |

**Body**: `multipart/form-data` with a single field:

- `audio_file` — the raw audio bytes. With `encode=false` the contents are:
  - Container: **none** (raw stream, not WAV)
  - Codec: **PCM signed 16-bit little-endian** (`pcm_s16le`)
  - Sample rate: **16 000 Hz**
  - Channels: **1** (mono)
  - Bazarr produces this with `ffmpeg -f s16le -acodec pcm_s16le -ac 1 -ar 16000`.
  - Approx size: `bytes ≈ duration_sec × 32 000`. A 45-min episode is ~85 MB.
  - The bridge must add a WAV header (or feed `pcm_s16le` directly with rate/channel metadata) before handing the audio to OpenArc or the aligner.

**Headers**: `User-Agent: Subliminal/<version>`. No auth headers.

**Timeouts**: Bazarr applies a `(connect, read)` tuple sourced from its provider
config (`Response` / `Timeout` settings, defaults 5 / 3600 seconds). Transcription
+ alignment of a 45-min episode can run several minutes — the bridge should not
buffer the request for so long that it triggers Bazarr's read timeout. Stream
or chunk-encode the response if needed; Bazarr reads `r.content` so a held
response works as long as the body eventually arrives before the deadline.

**Response**

- Status: `200 OK` on success. Bazarr does not inspect the status code beyond
  the HTTP exception layer in `requests`; a `4xx`/`5xx` propagates as a
  provider failure.
- `Content-Type`: `text/plain; charset=utf-8` (any text/* works; Bazarr just
  reads `r.content`).
- Body: **valid SRT**. Specifically:

  ```
  1
  00:00:00,000 --> 00:00:04,120
  First cue text.

  2
  00:00:04,120 --> 00:00:07,640
  Second cue text.
  ```

  CRLF or LF both accepted by `pysrt`. Trailing blank line is conventional.

  **Validity is enforced** by `Subtitle.is_valid`
  (`subliminal_patch/subtitle.py:283`):

  1. `pysrt.from_string(text, error_handling=pysrt.ERROR_RAISE)` must succeed.
  2. If that fails, `pysubs2.SSAFile.from_string(text)` is tried as a fallback
     (handles SSA/ASS/MicroDVD/VTT).
  3. If both fail, `manual.py:183` logs:
     ```
     BAZARR Downloaded subtitles isn't valid for this file: <path>
     ```
     This is the failure mode that motivated this replacement.

  Empty body, plain-text body without timestamp cues, and JSON-wrapped strings
  all fail this check.

- Optional custom headers: the upstream bridge sets
  `Source: Transcribed using Bazarr to OpenAI Whisper Bridge!`. Bazarr does
  not read it. Set or don't.

**Bazarr-side language handling worth knowing**

- `input_language` is converted from alpha-3 to alpha-2 in
  `download_subtitle`. For languages without an alpha-2 code, Bazarr falls
  back to `en` if the target is English; otherwise it short-circuits before
  hitting `/asr`. So the bridge will only ever see alpha-2 codes.
- `task=translate` is only ever called with target English (Bazarr rejects
  other targets before the request).
- Bazarr does not retry on 4xx. A 5xx with a `requests.exceptions` cascade
  triggers Bazarr's throttle path:
  ```
  Throttling whisperai for 24 hours … because of: ConnectionError.
  ```
  Avoid surfacing transient OpenArc errors as 5xx if a retry would succeed.

### 1.2 POST `/detect-language` — language ID

Issued from `WhisperAIProvider.detect_language`:

```python
self.session.post(
    f"{self.endpoint}/detect-language",
    params={
        "encode": "false",
        "video_file": video_name,
    },
    files={"audio_file": out},
    timeout=(self.response, self.timeout),
)
```

Same audio format and `encode` semantics as `/asr`. No `language` parameter.

**Response**

- Status: `200 OK`.
- `Content-Type`: `application/json`.
- Body shape:
  ```json
  {
    "language_code": "en",
    "detected_language": "english"
  }
  ```
  - `language_code` is ISO 639-1 alpha-2, lowercase. Return `"und"` to signal
    "could not detect" — Bazarr handles this explicitly
    (`whisperai.py:499`) and treats the file as undetectable.
  - `detected_language` is a human-readable English name passed through to
    Bazarr's language-name mapper. Keep it lowercase.
- On JSON parse error Bazarr logs `Invalid JSON response in language
  detection` and treats detection as failed.

**OpenArc does not natively provide language detection**: it always returns
the detected language as part of the transcription metrics. The bridge has
two realistic implementations:

1. Run a transcription pass on the audio (or a head, e.g. first 30 seconds)
   and pull `metrics.language` from `verbose_json` — expensive but accurate.
2. Use a small dedicated language-ID model (e.g. `speechbrain/lang-id-voxlingua107-ecapa`
   or `facebook/mms-lid-126`) — cheap.

Either way the bridge owns the implementation; OpenArc is not in this path
unless option 1 is chosen.

### 1.3 Health / liveness

Bazarr does **not** call a health endpoint. The current pod's Kubernetes
probes hit `GET /status` on port 9000 (configured in `hr.yaml`); expose at
minimum:

- `GET /status` → `200 OK`, body shape free. The existing bridge returns
  arbitrary JSON; treat as a liveness signal only.
- Optional: `GET /docs` (FastAPI default). Useful for human debugging, not
  contractual.

## 2. Outbound: bridge → OpenArc

OpenArc is reachable in-pod at `http://localhost:8000` (the bridge runs as a
sidecar). With the existing `openarc-config-path.patch` applied — and **without**
the SRT-output patch we drafted (that's superseded by this bridge) — the
following endpoints are available.

### 2.1 POST `/v1/audio/transcriptions`

The bridge will use this to obtain the transcript text. It is OpenAI-compatible
on the wire but with one OpenArc extension and one important caveat about
response formats.

**Request**: `multipart/form-data`.

| Field            | Required | Value |
|------------------|----------|-------|
| `file`           | yes      | The audio file. OpenArc accepts any format `soundfile` can decode — WAV, OGG, FLAC, MP3, OPUS. **PCM s16le raw must be wrapped in a WAV container** before posting — the upstream bridge transcodes to OGG/Opus first and that works fine; cheaper is to slap a 44-byte WAV header on the raw PCM Bazarr sent. |
| `model`          | yes      | Model name as known to OpenArc, e.g. `qwen3-asr-0_6b-int8-asym`. Must match the autoloaded model. |
| `response_format`| optional | One of `json`, `verbose_json`, `srt`, `vtt`, `text`. **Always use `json` or `verbose_json`** — see caveat below. |
| `openarc_asr`    | optional | JSON-stringified `OpenArcASRConfig`. Required for qwen3_asr models in upstream; the `openarc-config-path.patch` already in tree relaxes this to "all-defaults if absent". Omit to use defaults. |

**`openarc_asr` shape** (`src/server/models/requests_openai.py:13`):

```json
{
  "qwen3_asr": {
    "language": "en",            // optional, ISO 639-1 or English name
    "max_tokens": 1024,          // per-chunk decode cap
    "max_chunk_sec": 30.0,       // upper bound on internal chunk size
    "search_expand_sec": 5.0,    // silence-search window around chunk boundary
    "min_window_ms": 100.0       // energy analysis window
  }
}
```

Field meanings (`OV_Qwen3ASRGenConfig` in `src/server/models/openvino.py:135`):
- `language` — forced language hint. If omitted, the model auto-detects.
- `max_tokens` — token cap per internal chunk. Don't go below ~256 unless the
  audio is very short.
- `max_chunk_sec`, `search_expand_sec`, `min_window_ms` — control OpenArc's
  internal silence-aware chunker (`split_audio_into_chunks`). These shape the
  inference-time chunking, not anything the bridge will see in the response.
  Defaults are sensible; leave alone unless tuning for quality, not for cues.

**Response — important caveat**

Upstream OpenArc's response handling (`src/server/routes/openai.py:531`) is:

```python
if response_format == "json":
    return {"text": ...}
elif response_format == "verbose_json":
    return {"text": ..., "language": ..., "duration": ..., "metrics": ...}
else:
    return result.get("text", "")
```

That final `else` returns a bare Python string from a FastAPI route, which
FastAPI encodes as a **JSON-quoted string** (`"transcript"`) — not plain
text. `response_format=srt`/`vtt`/`text` all land here and are all broken
for direct consumption. **The bridge must request `json` or `verbose_json`**
to get a usable response.

Use **`verbose_json`** unless you're sure you don't want the duration /
language info:

```json
{
  "text": "Full transcript text as one string. No cue boundaries. No per-segment timing.",
  "language": "English",
  "duration": null,
  "metrics": {
    "feature_sec": 0.37,
    "encoder_sec": 115.53,
    "prefill_sec": 2.22,
    "prefill_tok_s": 4294.72,
    "decode_sec": 52.87,
    "decode_tok_s": 37.15,
    "detok_sec": 1.58,
    "prompt_tokens": 9553,
    "generated_tokens": 1964,
    "encoder_tokens": 9193,
    "audio_duration_sec": 706.72,
    "model_load_sec": 1.97,
    "end_to_end_sec": 173.86,
    "rtf": 0.246,
    "language": "English"
  }
}
```

Notes:

- The top-level `duration` is the OpenAI-shaped field and is currently `null`
  in upstream; `metrics.audio_duration_sec` has the real value.
- `metrics.language` is the human-readable name OpenArc inferred. It does not
  follow ISO codes — translate to alpha-2 yourself before returning to Bazarr
  from `/detect-language` (e.g. `"English"` → `"en"`).
- **No per-chunk timing is exposed**. The forced aligner must recover all
  timing from `(audio, text)` on its own. This is the design assumption that
  makes this bridge able to skip the OpenArc patch.

**Errors**

- `400 Bad Request` with `{"detail": "..."}` for validation errors (e.g.
  model not loaded).
- `500 Internal Server Error` with `{"detail": "Transcription failed: ..."}`
  on inference failure.
- Connection-level errors (model reloading, OOM kill, GPU hang) drop the
  connection — the bridge sees a transport error, not a structured response.

### 2.2 GET `/openarc/status` — model load state

Used by the existing liveness probe and useful for the bridge to fail fast
if the model isn't ready:

```json
[
  {
    "model_name": "qwen3-asr-0_6b-int8-asym",
    "status": "loaded",
    "model_type": "qwen3_asr",
    ...
  }
]
```

`status` values include `loaded`, `loading`, `unloaded`. The bridge can
return `503 Service Unavailable` to Bazarr if the model isn't `loaded`,
which surfaces in Bazarr as a transient failure (no 24h throttle) rather
than a hard error.

The model can silently unload on certain malformed transcription requests —
the existing liveness probe is what restarts the pod when that happens.
The bridge does not need to handle this itself; let Kubernetes restart it.

### 2.3 Endpoints the bridge does NOT need

- `POST /openarc/load`, `/openarc/unload` — handled by autoload
  (`OPENARC_AUTOLOAD_MODEL` env). The bridge should not touch these.
- `GET /v1/models` — debugging aid, not needed at runtime.
- `POST /openarc/bench`, `/openarc/metrics` — telemetry, optional.

## 3. Field translation cheat sheet

For the bridge's request mapping:

| Bazarr `/asr` query                | OpenArc `/v1/audio/transcriptions` |
|------------------------------------|------------------------------------|
| `task=transcribe`                  | (no field — default behavior)      |
| `task=translate`                   | Not directly supported. OpenArc only transcribes in the source language. To honor `translate`, either run transcribe and then translate the text via a separate model, or refuse with 4xx and let Bazarr surface the failure. |
| `language=en` (alpha-2)            | `openarc_asr.qwen3_asr.language` — pass through as-is, OpenArc accepts both alpha-2 codes and English names. |
| `output=srt`                       | `response_format=verbose_json` (the bridge formats SRT itself). |
| `encode=false`                     | Bridge wraps PCM in WAV and uploads as `file`. |
| `video_file=<path>`                | No equivalent. Log it for debugging if you want. |

For the bridge's response assembly:

| Bridge needs                     | Source |
|----------------------------------|--------|
| Transcript text                  | OpenArc `text` field. |
| Audio duration (for fallback / sanity) | OpenArc `metrics.audio_duration_sec`. |
| Detected language (for `/detect-language`) | OpenArc `metrics.language` → translate to alpha-2 via a language map. |
| Word-level timestamps            | Forced aligner output — NOT OpenArc. |
| SRT cue boundaries               | Bridge's segmentation policy applied to aligner output. |

## 4. Behaviors to preserve from the upstream bridge

A few that aren't strictly part of the contract but match the working
behavior of the existing image:

- A single `/asr` request takes minutes. Don't impose internal timeouts
  smaller than Bazarr's `(response, timeout)` tuple (default `(5, 3600)`).
- Free the audio buffer between Bazarr ingress and OpenArc upload; full-episode
  PCM is ~100 MB and gets wasteful if duplicated.
- Streaming the response back to Bazarr is fine but not required — Bazarr
  reads `r.content` synchronously.
- Log the video file name if `video_file` is set; it's the only context tying
  a request back to a specific media file when debugging across services.
