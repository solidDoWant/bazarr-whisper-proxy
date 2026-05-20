# End-to-end test harness

The e2e suite brings up real Bazarr, Radarr, Lingarr, and LibreTranslate
containers, plus the bridge image built from this repo, and drives the
whole topology through its REST APIs. OpenArc itself is **not**
containerised — it requires a model + budget tuning that doesn't belong in
an ephemeral test harness. Point the suite at a remote OpenArc via the
`OPENARC_E2E_BASE_URL` env var.

## How to run

```sh
# 1. Enter the dev shell (provides docker, docker-compose, ffmpeg,
#    espeak-ng, python with our deps, and the e2e script on PATH).
nix develop

# 2. Point at a reachable OpenArc.
export OPENARC_E2E_BASE_URL=http://your-openarc-host:8000

# 3. Run the suite. Builds the bridge image, brings up compose, provisions
#    the services, runs pytest, tears the stack down on exit.
e2e
```

Useful flags:

| Invocation | Purpose |
|------------|---------|
| `e2e` | Default suite (criteria 6–14, 17–22). |
| `e2e failure-modes` | Bring the stack up with a black-hole OpenArc to validate criterion 16 (`/status` returns 503 + `Retry-After`). |
| `e2e --keep-up` | Don't tear the stack down at the end — useful when debugging a failure. Subsequent runs will see the previous state because provisioning is idempotent. |

## What fails the build

The suite maps directly to spec 16's acceptance criteria. Each test is
named `test_<N>_…` where N is the spec criterion number.

- **Criteria 6-9** (`tests/e2e/test_transcribe.py`) — Bazarr's
  `task=transcribe` path produces a valid SRT and Bazarr does not log the
  "Downloaded subtitles isn't valid for this file" regression marker.
- **Criteria 10-11** (`tests/e2e/test_detect_language.py`) — the bridge's
  `/detect-language` route returns the contract-mandated JSON shape and the
  right alpha-2 code.
- **Criteria 12-14** (`tests/e2e/test_translate.py`) — Bazarr's
  `task=translate` path through Lingarr → LibreTranslate produces English
  subs from Spanish audio, with cue count and timing identical to a
  transcribe pass.
- **Criteria 15-16** (`tests/e2e/test_failure_modes.py`) — bridge
  unavailable → no 24-hour throttle; OpenArc unreachable → 503 + Retry-After.
- **Criteria 18-22** (`tests/e2e/test_provisioning.py`) — Radarr/Bazarr
  inventory shape, settings shape, idempotency of provisioning, image-tag
  pinning.
- **Criterion 17** — wall-clock sanity check; logged as a WARN if exceeded,
  does not fail the suite.

## Fixture media

The fixture media is *not* committed. `tests/e2e/fixtures/build.py`
synthesises it lazily on first run via `espeak-ng`, then wraps each clip
into a minimal MKV (black 320x240 video + the synthesised audio stream).
The manifest lives at `tests/e2e/fixtures/manifest.json` and pairs each
clip with a real-world public-domain film's TMDB id (Radarr cross-checks
TMDB before accepting a movie record, so synthetic ids don't work).

See `tests/e2e/fixtures/PROVENANCE.md` for why we diverged from spec 16's
"Mozilla Common Voice" call-out (CV's per-clip URLs gate downloads behind
session auth).

### Extending the fixture set

To add a third language:

1. Append a new clip entry to `tests/e2e/fixtures/manifest.json`. Required
   fields: `slug`, `language` (alpha-2), `espeak_voice`, `espeak_speed`,
   `text`, `expected_words`, `movie_title`, `movie_year`, `tmdb_id`,
   `radarr_folder`, `filename`.
2. Add the new language code to `LT_LOAD_ONLY` in `compose.e2e.yml` so
   LibreTranslate loads its model.
3. Add a corresponding `espeak-ng` voice (most are bundled — check `espeak-ng --voices`).
4. Bump `OPENARC_E2E_MODEL` if you need a model variant that supports the
   new language.

## Pinned image tags

| Service | Tag | Bumped |
|---|---|---|
| `linuxserver/bazarr` | `1.5.6` | 2026-05-20 |
| `linuxserver/radarr` | `6.1.1` | 2026-05-20 |
| `lingarr/lingarr` | `1.2.4` | 2026-05-20 |
| `libretranslate/libretranslate` | `v1.9.5` | 2026-05-20 |

### How to bump a pinned tag

1. Check the upstream registry for the latest stable tag (`docker search`,
   the registry's tags API, or the project's GitHub releases).
2. Update the tag in `compose.e2e.yml`.
3. Run `e2e` — if the provisioner trips on a new API contract, the
   provisioning code in `tests/e2e/provision/{radarr,bazarr,lingarr}.py`
   may need updates.
4. Update the table above with the new tag and today's date.

## Known limitations

- **Not hermetic w.r.t. OpenArc.** The suite needs a reachable OpenArc to
  do anything meaningful. Different OpenArc deployments may load different
  models — the suite's "expected words" are a loose check against speech
  that the model both understands and transcribes recognisably. If you see
  criterion 8 fail, check that the upstream OpenArc actually loaded the
  expected model (`/v1/models`).
- **Not hermetic w.r.t. TMDB.** Radarr cross-checks every movie record's
  tmdbId against TMDB before accepting it. Provisioning fails if the
  Radarr container can't reach `api.themoviedb.org`.
- **No CI integration.** Per spec 16 §out-of-scope, the remote-OpenArc
  dependency makes per-PR CI infeasible. A separate manually-triggered
  workflow or self-hosted runner is the follow-up.
- **Synthesised speech, not real speech.** Easier for the ASR than natural
  audio; this is a correctness gate, not a robustness/SLA benchmark.
