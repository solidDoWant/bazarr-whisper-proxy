# Fixture media provenance

These media files are **not committed**. They are built on-demand by
`tests/e2e/fixtures/build.py` (invoked from `scripts/e2e.sh`) on first
harness run, into `tests/e2e/fixtures/media/`. That directory is
`.gitignore`d.

## Source

Audio is synthesized with [espeak-ng](https://github.com/espeak-ng/espeak-ng)
(GPL-3.0 tool; algorithmic TTS output is non-copyrightable). The transcript
text lives in `manifest.json` and is verbatim public-domain English / Spanish
pangrams.

The wrapping into MKV uses [ffmpeg](https://ffmpeg.org/) with a black
placeholder video stream (Radarr expects a video container).

## Why synthesized, not Common Voice

Spec 16 calls for Common Voice (CC0) clips. We tried two ways to fetch CV
clips lazily without committing them to the repo:

- The `/api/v1/{locale}/clips/...` endpoint requires a session token even
  for "anonymous" downloads.
- The HuggingFace mirror (`mozilla-foundation/common_voice_*`) requires
  accepting a per-dataset agreement before the auth-gated download URL is
  unlocked.

Neither path is hermetic without baking credentials into the harness.
Synthesized speech is:

- **Deterministic** — the same input text always produces the same audio.
  This makes the criterion-8 "first cue contains an expected word" check
  reliable across runs.
- **License-clean** — algorithmic output is not copyrightable; the
  underlying TTS is GPL-licensed but we're not redistributing it.
- **Trivially extensible** — add a new entry to `manifest.json`.

Tradeoff: synthesized speech is easier for ASR than natural speech, so this
suite **does not** exercise the OpenArc model's robustness to accents /
noise. That's outside the scope of correctness-gate e2e tests anyway
(spec 16 §out-of-scope, "Load / soak / chaos testing").

## License of the synthesized audio

The transcript texts in `manifest.json` are public-domain pangrams. The
rendered audio inherits no copyright from espeak-ng (algorithmic output of
a deterministic TTS engine, applied to public-domain text).
