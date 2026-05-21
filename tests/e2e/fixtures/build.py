"""Build e2e fixture media: espeak-ng -> WAV -> MKV (black video + audio).

The e2e harness needs real speech audio with a known transcript, wrapped in
a container the *arr tools recognize as a movie. Mozilla Common Voice doesn't
expose stable per-clip URLs without auth (we tried; their API gates downloads
behind a session token), so instead we synthesize speech locally with
espeak-ng. Algorithmically-generated audio is non-copyrightable and the
transcript content is deterministic, which makes criterion-8 ("first cue
contains an expected word") trivial to validate.

Idempotent: if the target .mkv already exists and matches the manifest
content hash, the build is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
DEFAULT_OUT_DIR = FIXTURES_DIR / "media"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    clips = raw.get("clips")
    if not isinstance(clips, list):
        raise SystemExit(f"manifest {path} missing 'clips' array")
    return list(clips)


# Minimum raw-PCM size the bridge must handle: 1 MiB + headroom.
# At 16 kHz mono s16le, 35 s ≈ 1 120 000 bytes.  Clips shorter than this
# are padded with silence so every e2e fixture exercises the > 1 MiB path.
_MIN_CLIP_DURATION_SEC: int = 35


def _content_hash(clip: dict[str, Any]) -> str:
    """Stable hash of the fields that affect the rendered audio.

    Bumping the espeak version may change the output bit-for-bit; that's fine —
    the hash isn't a strict integrity check, just a "rebuild on change" gate.
    Changing _MIN_CLIP_DURATION_SEC is included so cached MKVs are rebuilt.
    """
    fields = ("text", "espeak_voice", "espeak_speed", "language")
    payload = (
        "|".join(str(clip.get(k, "")) for k in fields)
        + f"|min_dur={_MIN_CLIP_DURATION_SEC}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _ensure_tools() -> None:
    # ffmpeg uses -version (single dash); espeak-ng uses --version.
    probes = (("espeak-ng", "--version"), ("ffmpeg", "-version"), ("ffprobe", "-version"))
    for tool, flag in probes:
        try:
            subprocess.run([tool, flag], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise SystemExit(
                f"required tool '{tool}' not on PATH. Enter `nix develop` first."
            ) from exc


def _synth_wav(clip: dict[str, Any], wav_path: Path) -> None:
    cmd = [
        "espeak-ng",
        "-v",
        str(clip["espeak_voice"]),
        "-s",
        str(clip["espeak_speed"]),
        "-w",
        str(wav_path),
        clip["text"],
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _pad_wav(wav_path: Path, min_duration_sec: int) -> None:
    """Extend wav_path in-place with silence to reach at least min_duration_sec.

    apad=whole_dur is a no-op when the input is already long enough, so this
    is safe to call unconditionally. The padded WAV replaces the original.
    """
    padded = wav_path.with_suffix(".padded.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-af",
            f"apad=whole_dur={min_duration_sec}",
            str(padded),
        ],
        check=True,
        capture_output=True,
    )
    padded.replace(wav_path)


_ALPHA2_TO_ISO639_2: dict[str, str] = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "ja": "jpn",
    "ko": "kor",
    "zh": "zho",
}


def _wrap_mkv(wav_path: Path, mkv_path: Path, hash_tag: str, audio_lang_alpha2: str) -> None:
    """Wrap a WAV into an MKV with a black placeholder video stream.

    The video is required because *arr tools expect a video container; without
    it Radarr's media-info probe rejects the file.

    Tags the audio stream with its ISO 639-2 language code so Bazarr's
    ffprobe-based audio-language detection picks it up — otherwise Bazarr
    defaults to ``default_und_audio_lang`` (English), which breaks the
    translate path tests for non-English fixtures.
    """
    iso639_2 = _ALPHA2_TO_ISO639_2.get(audio_lang_alpha2, "und")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=size=320x240:rate=1:color=black",
        "-i",
        str(wav_path),
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-tune",
        "stillimage",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-metadata:s:a:0",
        f"language={iso639_2}",
        "-metadata",
        f"comment=bazarr-whisper-proxy-e2e fixture {hash_tag}",
        str(mkv_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _mkv_matches_hash(mkv_path: Path, expected_tag: str) -> bool:
    if not mkv_path.exists():
        return False
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=comment",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(mkv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return False
    return expected_tag in out


def build_all(manifest_path: Path = MANIFEST_PATH, out_dir: Path = DEFAULT_OUT_DIR) -> list[Path]:
    """Synthesize + wrap every clip in the manifest. Returns the list of MKV paths."""
    _ensure_tools()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    clips = _load_manifest(manifest_path)
    produced: list[Path] = []
    for clip in clips:
        slug = clip["slug"]
        tag = _content_hash(clip)
        mkv_path = out_dir / clip["filename"]
        if _mkv_matches_hash(mkv_path, tag):
            print(f"[fixtures] skip (cached) {mkv_path.name}", file=sys.stderr)
            produced.append(mkv_path)
            continue

        wav_path = cache_dir / f"{slug}.wav"
        print(f"[fixtures] synthesizing {slug} → {mkv_path.name}", file=sys.stderr)
        _synth_wav(clip, wav_path)
        _pad_wav(wav_path, _MIN_CLIP_DURATION_SEC)
        _wrap_mkv(wav_path, mkv_path, tag, clip["language"])
        produced.append(mkv_path)

    return produced


def main() -> int:
    parser = argparse.ArgumentParser(description="Build e2e fixture media")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    paths = build_all(args.manifest, args.out)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
