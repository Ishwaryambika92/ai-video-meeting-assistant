"""
Audio acquisition and preprocessing.

Responsibilities:
    - Download audio from a YouTube URL (yt-dlp)
    - Convert any local audio/video file to a clean mono 16kHz WAV
    - Split long audio into fixed-length chunks for transcription
    - Clean up temporary/partial files safely

Design notes:
    - No code runs at import time (the original file called
      download/convert/chunk functions at module scope, which meant simply
      `import utils.audio_processor` triggered a real network download).
    - Every function is a pure operation on paths in/out; the Streamlit UI
      or a script decides when to call them.
    - Filenames coming from YouTube titles can contain characters that are
      invalid on Windows (: / \\ * ? " < > |) or awkward Unicode - these are
      sanitized before use.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Optional, cast

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
except ImportError:
    AudioSegment = None  # type: ignore
    CouldntDecodeError = Exception  # type: ignore

from config import DOWNLOAD_DIR, get_logger, settings

logger = get_logger(__name__)

_INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class AudioProcessingError(Exception):
    """Raised for any recoverable failure in the audio pipeline."""


def _sanitize_filename(name: str, max_len: int = 150) -> str:
    """Make a string safe to use as a filename on Windows/macOS/Linux."""
    name = unicodedata.normalize("NFKC", name)
    name = _INVALID_WINDOWS_CHARS.sub("_", name).strip(" .")
    if not name:
        name = "audio"
    return name[:max_len]


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise AudioProcessingError(
            "FFmpeg was not found on PATH. Install it and make sure it is "
            "accessible from the command line (e.g. `winget install ffmpeg` "
            "on Windows, `brew install ffmpeg` on macOS, "
            "`apt install ffmpeg` on Linux)."
        )


def is_youtube_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source.strip(), re.IGNORECASE)) and (
        "youtube.com" in source or "youtu.be" in source
    )


# --------------------------------------------------------------------------
# YouTube download
# --------------------------------------------------------------------------
def download_youtube_audio(url: str, output_dir: Path = DOWNLOAD_DIR) -> str:
    """Download the best available audio track for a YouTube URL and
    convert it to WAV. Returns the absolute path to the WAV file.

    Uses the video ID (not the raw title) as the filename so re-processing
    the same video does not create duplicate downloads, and so Unicode /
    emoji-heavy titles never touch the filesystem.
    """
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise AudioProcessingError(
            "yt-dlp is not installed. Install it with `pip install yt-dlp` to download and process YouTube videos."
        ) from exc

    _ensure_ffmpeg_available()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    node_candidates = [
        shutil.which("node"),
        shutil.which("deno"),
        "D:\\ISHU\\node.exe",
        "C:\\Program Files\\nodejs\\node.exe",
        "C:\\Program Files (x86)\\nodejs\\node.exe",
        "/usr/bin/node",
        "/usr/local/bin/node",
    ]
    node_exe = next((p for p in node_candidates if p and Path(p).exists()), None)
    js_runtimes: dict[str, Any] = {}
    if node_exe and Path(node_exe).exists():
        js_runtimes["node"] = {"path": node_exe}

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "restrictfilenames": True,  # ASCII-only, Windows-safe filenames
        "nopart": True,  # Prevent Windows file lock errors during .part rename
        "socket_timeout": 30,  # Prevent hanging indefinitely on dropped connections
        "retries": 3,
        "fragment_retries": 3,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes


    try:
        with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get("id", uuid.uuid4().hex) if info else uuid.uuid4().hex
            expected_wav = output_dir / f"{video_id}.wav"

            if expected_wav.exists() and expected_wav.stat().st_size > 0:
                logger.info("Reusing previously downloaded audio for %s", video_id)
                return str(expected_wav)

            ydl.download([url])

            if not expected_wav.exists():
                raise AudioProcessingError(
                    "Download finished but the expected WAV file was not "
                    "found. FFmpeg post-processing may have failed."
                )
            return str(expected_wav)

    except DownloadError as exc:
        message = str(exc)
        # Clean up any partial files left behind safely on Windows.
        for leftover in output_dir.glob("*.part"):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass

        if "HTTP Error 403" in message:
            raise AudioProcessingError(
                "YouTube returned HTTP 403 (Forbidden). This usually means "
                "YouTube is blocking the request - try updating yt-dlp "
                "(`pip install -U yt-dlp`) or wait and retry."
            ) from exc
        if "Sign in to confirm" in message or "not a bot" in message.lower():
            raise AudioProcessingError(
                "YouTube is requesting bot verification for this video. "
                "Try a different video or supply cookies to yt-dlp."
            ) from exc
        if "Requested format is not available" in message:
            raise AudioProcessingError(
                "No downloadable audio track was found for this URL."
            ) from exc
        raise AudioProcessingError(f"Failed to download YouTube audio: {message}") from exc

    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        msg = str(exc).lower()
        if "jsinterp" in msg or "javascript" in msg or "deno" in msg or "nodejs" in msg:
            raise AudioProcessingError(
                "yt-dlp needs a JavaScript runtime (Deno or Node.js) to "
                "decode this video's signature and could not find one. "
                "Install Node.js or Deno and try again."
            ) from exc
        raise AudioProcessingError(f"Unexpected error downloading video: {exc}") from exc


# --------------------------------------------------------------------------
# Local file -> WAV
# --------------------------------------------------------------------------
def convert_to_wav(input_path: str, output_dir: Optional[Path] = None) -> str:
    """Convert any audio/video file to a mono, 16kHz WAV suitable for
    speech recognition. Returns the absolute path to the new WAV file.
    """
    _ensure_ffmpeg_available()

    src = Path(input_path)
    if not src.exists():
        raise AudioProcessingError(f"Input file not found: {input_path}")
    if src.stat().st_size == 0:
        raise AudioProcessingError(f"Input file is empty: {input_path}")

    out_dir = output_dir or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _sanitize_filename(src.stem)
    output_path = out_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}_converted.wav"

    if AudioSegment is None:
        raise AudioProcessingError("pydub is not installed. Install requirements.txt to convert audio.")

    try:
        audio = cast(Any, AudioSegment.from_file(src))
    except CouldntDecodeError as exc:
        raise AudioProcessingError(
            f"Could not decode '{src.name}'. The file may be corrupted or "
            "in an unsupported format."
        ) from exc

    if len(audio) == 0:
        raise AudioProcessingError(f"'{src.name}' contains no audio.")

    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return str(output_path)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_audio(
    wav_path: str,
    chunk_seconds: int = 30,
    min_chunk_ms: int = 250,
) -> list[str]:
    """Split a WAV file into fixed-length chunks."""

    src = Path(wav_path)

    if not src.exists():
        raise AudioProcessingError(f"WAV file not found: {wav_path}")

    if AudioSegment is None:
        raise AudioProcessingError("pydub is not installed. Install requirements.txt to split audio into chunks.")

    try:
        audio = cast(Any, AudioSegment.from_wav(src))
    except CouldntDecodeError as exc:
        raise AudioProcessingError(
            f"Could not read WAV file: {wav_path}"
        ) from exc

    if len(audio) == 0:
        raise AudioProcessingError(
            f"WAV file has zero duration: {wav_path}"
        )

    # Sarvam API maximum is 30 seconds per request
    chunk_ms = chunk_seconds * 1000

    chunk_dir = src.parent / f"{src.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[str] = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        segment = cast(Any, audio[start:start + chunk_ms])

        if len(segment) < min_chunk_ms:
            logger.debug(
                "Skipping trailing chunk %d (%dms, too short)",
                i,
                len(segment),
            )
            continue

        chunk_path = chunk_dir / f"chunk_{i:04d}.wav"

        segment.export(
            chunk_path,
            format="wav"
        )

        if chunk_path.stat().st_size == 0:
            logger.warning(
                "Chunk %d exported as empty, discarding",
                i
            )
            chunk_path.unlink(missing_ok=True)
            continue

        chunk_paths.append(str(chunk_path))

    if not chunk_paths:
        raise AudioProcessingError(
            "Chunking produced no usable audio segments."
        )

    return chunk_paths

def cleanup_chunks(chunk_paths: list[str]) -> None:
    """Remove chunk files (and their parent chunk directory if now empty)."""
    parents = set()
    for path in chunk_paths:
        p = Path(path)
        parents.add(p.parent)
        p.unlink(missing_ok=True)
    for parent in parents:
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def process_input(
    source: str,
    chunk_minutes: int = settings.audio_chunk_minutes,
) -> list[str]:
    """Turn a YouTube URL or local file path into 30-second WAV chunks."""

    if is_youtube_url(source):
        logger.info("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        logger.info("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    logger.info(
        "Chunking audio into 30-second chunks..."
    )

    chunks = chunk_audio(
        wav_path,
        chunk_seconds=30,
    )

    # For local uploaded files, clean up intermediate converted WAV now that chunks exist
    if not is_youtube_url(source):
        try:
            p = Path(wav_path)
            if p.exists() and "_converted.wav" in p.name:
                p.unlink(missing_ok=True)
                logger.debug("Cleaned up intermediate converted WAV: %s", wav_path)
        except OSError:
            pass

    logger.info(
        "Audio ready - %d chunk(s) created.",
        len(chunks),
    )

    return chunks