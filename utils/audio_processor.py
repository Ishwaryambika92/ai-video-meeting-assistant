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


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        r"D:\ISHU\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _ensure_ffmpeg_available() -> str:
    ffmpeg_bin = _find_ffmpeg()
    if not ffmpeg_bin:
        raise AudioProcessingError(
            "FFmpeg was not found on PATH. Install it and make sure it is "
            "accessible from the command line (e.g. `winget install ffmpeg` "
            "on Windows, `brew install ffmpeg` on macOS, "
            "`apt install ffmpeg` on Linux)."
        )
    return ffmpeg_bin


def _detect_js_runtime() -> dict[str, Any]:
    """Detect available JS runtimes (preferring Deno, then Node.js 22+)."""
    # 1. Check for Deno
    deno_candidates = [
        shutil.which("deno"),
        "C:\\Program Files\\deno\\deno.exe",
        "/usr/bin/deno",
        "/usr/local/bin/deno",
    ]
    deno_exe = next((p for p in deno_candidates if p and Path(p).exists()), None)
    if deno_exe:
        try:
            ver_out = subprocess.check_output([deno_exe, "--version"], text=True, timeout=5)
            logger.info("Found Deno runtime: %s", ver_out.splitlines()[0] if ver_out else "")
            return {"deno": {"path": deno_exe}}
        except Exception:
            pass

    # 2. Check for Node.js
    node_candidates = [
        shutil.which("node"),
        "D:\\ISHU\\node.exe",
        "C:\\Program Files\\nodejs\\node.exe",
        "C:\\Program Files (x86)\\nodejs\\node.exe",
        "/usr/bin/node",
        "/usr/local/bin/node",
    ]
    node_exe = next((p for p in node_candidates if p and Path(p).exists()), None)
    if node_exe:
        try:
            ver_out = subprocess.check_output([node_exe, "--version"], text=True, timeout=5).strip()
            logger.info("Found Node.js runtime (%s) at %s", ver_out, node_exe)
            return {"node": {"path": node_exe}}
        except Exception:
            pass

    return {}


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

    js_runtimes = _detect_js_runtime()

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
                "player_client": ["mweb", "web", "android", "ios"],
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
        # Clean up any partial files left behind safely on Windows/Linux.
        for leftover in output_dir.glob("*.part"):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass

        lower_msg = message.lower()
        if "http error 403" in lower_msg or "403: forbidden" in lower_msg:
            raise AudioProcessingError(
                "YouTube returned HTTP 403 (Forbidden). YouTube frequently restricts requests "
                "from cloud hosting providers (e.g., Streamlit Community Cloud) or datacenter IPs. "
                "Please use the 'Upload file' tab to upload your recording directly."
            ) from exc
        if "sign in to confirm" in lower_msg or "not a bot" in lower_msg:
            raise AudioProcessingError(
                "YouTube requires bot verification/sign-in for this video. "
                "Please download the file locally and use the 'Upload file' tab."
            ) from exc
        if "requested format is not available" in lower_msg or "no suitable format" in lower_msg:
            raise AudioProcessingError(
                "No compatible audio track was found for this YouTube video. "
                "The video may be private, members-only, a livestream, or region-restricted."
            ) from exc
        if any(err_kw in lower_msg for err_kw in ["timed out", "connection reset", "network is unreachable", "getaddrinfo failed", "temporary failure in name resolution"]):
            raise AudioProcessingError(
                f"Network connection error contacting YouTube: {message}. "
                "Please check your internet connection and retry."
            ) from exc
        if any(kw in lower_msg for kw in ["jsinterp", "javascript", "ejs", "deno", "challenge"]):
            raise AudioProcessingError(
                "YouTube extraction failed due to a JavaScript challenge failure. "
                "A supported JavaScript runtime (Deno or Node.js 22+) is required for signature decoding."
            ) from exc
        raise AudioProcessingError(f"Failed to download YouTube audio: {message}") from exc

    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        msg = str(exc).lower()
        if any(kw in msg for kw in ["jsinterp", "javascript", "ejs", "deno", "challenge"]):
            raise AudioProcessingError(
                "YouTube extraction failed due to a JavaScript challenge failure. "
                "A supported JavaScript runtime (Deno or Node.js 22+) is required for signature decoding."
            ) from exc
        if any(err_kw in msg for err_kw in ["timed out", "connection reset", "network is unreachable", "getaddrinfo failed"]):
            raise AudioProcessingError(
                f"Network connection error contacting YouTube: {exc}. Please check your connection."
            ) from exc
        raise AudioProcessingError(f"Unexpected error downloading video: {exc}") from exc


# --------------------------------------------------------------------------
# Local file -> WAV
# --------------------------------------------------------------------------
def convert_to_wav(input_path: str, output_dir: Optional[Path] = None) -> str:
    """Convert any audio/video file to a mono, 16kHz WAV suitable for
    speech recognition using memory-efficient FFmpeg direct streaming.
    Returns the absolute path to the new WAV file.
    """
    ffmpeg_bin = _ensure_ffmpeg_available()

    src = Path(input_path)
    if not src.exists():
        raise AudioProcessingError(f"Input file not found: {input_path}")
    if src.stat().st_size == 0:
        raise AudioProcessingError(f"Input file is empty: {input_path}")

    out_dir = output_dir or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _sanitize_filename(src.stem)
    output_path = out_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}_converted.wav"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-nostdin",
        "-loglevel", "error",
        "-i", str(src.resolve()),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path.resolve()),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise AudioProcessingError(f"FFmpeg execution failed: {exc}") from exc

    if res.returncode != 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        err = res.stderr.strip()
        if "matches no streams" in err or "does not contain any stream" in err:
            raise AudioProcessingError(f"'{src.name}' contains no audio track.")
        raise AudioProcessingError(
            f"Could not convert '{src.name}' to WAV: {err or 'FFmpeg conversion error'}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise AudioProcessingError(f"'{src.name}' contains no audio or output is empty.")

    return str(output_path.resolve())



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