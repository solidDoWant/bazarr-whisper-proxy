# bazarr-whisper-proxy

Proxies transcription requests from Bazarr to Whisper API services (OpenAI or self-hosted). This has several advantages over other similar projects:
* It's fully compatible with [OpenArc](https://github.com/morioka/openarc) with Qwen3-ASR-0.6B, which requires a non-standard `openarc_asr` parameter
* All transcribed text is force-aligned via [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) so that text lines up with spoken audio
* Transcribed audio is chunked/split into cues based upon language heuristics with word-level timing 
* Language detection works rather than returning a static, pre-set value
* Translation is fully supported when a [Lingarr](https://github.com/lingarr-translate/lingarr) endpoint is provided

Downstream API contract matches [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) (`/asr`), which Bazarr uses instead of OpenAI's Whisper contract (`/v1/audio/transcription`).
Upstream translation service is expected to match OpenAI's Whisper API.

## How to run

Container images are available at [gchr.io/soliddowant/bazarr-whisper-proxy](https://gchr.io/soliddowant/bazarr-whisper-proxy). For an example deployment, see [this Docker compose file](./compose.example.yml).

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
