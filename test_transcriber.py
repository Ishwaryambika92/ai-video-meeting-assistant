"""
Speech-to-text abstraction.

Two engines are supported:
    - Whisper (local, runs on GPU if available, else CPU) - for English
    - Sarvam AI Saaras (cloud) - for Hindi / Hinglish

Both are exposed through a single `transcribe_chunk()` router and a
`transcribe_all()` helper that also preserves per-chunk timestamps so the
rest of the pipeline (RAG, reports) can cite "around 12:34" style answers.

Sarvam API notes (verified against current docs, Aug 2026):
    - The old `/speech-to-text-translate` endpoint used by the original
      tutorial is now LEGACY (saaras:v2.5) and being phased out.
    - The current endpoint is POST /speech-to-text with:
        - model: "saaras:v3" (default) or "saaras:v4"
        - mode: "transcribe" (native language) or "translate" (-> English)
        - with_timestamps: bool
    - Auth header is unchanged: `api-subscription-key: <SARVAM_API_KEY>`
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
import whisper

from config import get_device, get_logger, settings

logger = get_logger(__name__)

CHUNK_SECONDS_DEFAULT = 30

# Cloud transcription can safely process a small number of chunks in parallel.
# Keep this low to avoid API rate limits.
SARVAM_MAX_WORKERS = 2


class TranscriptionError(Exception):
    """Raised for any recoverable transcription failure."""


@dataclass
class TranscriptSegment:
    """A single transcribed chunk with its position in the full recording."""

    text: str
    start_seconds: float
    end_seconds: float
    engine: str
    language_code: Optional[str] = None

    @property
    def start_timestamp(self) -> str:
        return _format_timestamp(self.start_seconds)


@dataclass
class TranscriptResult:
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    engine: str = "whisper"


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# Whisper (English, local)
# --------------------------------------------------------------------------
_model = None
_model_device = None


def load_model():
    """Load (and cache) the local Whisper model on the best available device."""
    global _model, _model_device

    device = get_device()
    if _model is None or _model_device != device:
        logger.info("Loading Whisper model '%s' on %s...", settings.whisper_model, device)
        try:
            _model = whisper.load_model(settings.whisper_model, device=device)
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Failed to load Whisper model: {exc}") from exc
        _model_device = device
        logger.info("Whisper model loaded on %s.", device)

    return _model


def detect_language(chunk_path: str) -> str:
    """Detect the spoken language of an audio chunk using Whisper."""

    if not Path(chunk_path).exists():
        raise TranscriptionError(
            f"Audio chunk not found: {chunk_path}"
        )

    model = load_model()

    try:
        audio = whisper.load_audio(chunk_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=model.dims.n_mels,
        ).to(model.device)

        _, probabilities = model.detect_language(mel)

        detected_code = max(
            probabilities,
            key=probabilities.get,
        )

        logger.info(
            "Auto-detected language: %s",
            detected_code,
        )

        return detected_code

    except Exception as exc:
        raise TranscriptionError(
            f"Language detection failed: {exc}"
        ) from exc


def transcribe_chunk_whisper(chunk_path: str, translate: bool = False) -> str:
    """Transcribe a single audio chunk with local Whisper."""
    if not Path(chunk_path).exists():
        raise TranscriptionError(f"Audio chunk not found: {chunk_path}")

    model = load_model()
    task = "translate" if translate else "transcribe"

    try:
        import torch

        result = model.transcribe(
            chunk_path,
            task=task,
            fp16=torch.cuda.is_available(),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Whisper failed on '{chunk_path}': {exc}") from exc

    return result.get("text", "").strip()


# --------------------------------------------------------------------------
# Sarvam AI (Hindi / Hinglish, cloud)
# --------------------------------------------------------------------------
def transcribe_chunk_sarvam(
    chunk_path: str,
    mode: str = "translate",
    max_retries: int = 3,
) -> tuple[str, Optional[str]]:
    """Transcribe a single audio chunk with Sarvam AI Saaras.

    mode="translate" returns clean English text directly (source speech in
    any supported Indic language / Hinglish -> English).
    mode="transcribe" returns text in the original spoken language/script.

    Returns (text, language_code).
    """
    if not settings.sarvam_api_key:
        raise TranscriptionError(
            "SARVAM_API_KEY is not set. Add it to your .env file to use "
            "Hindi/Hinglish transcription."
        )
    if not Path(chunk_path).exists():
        raise TranscriptionError(f"Audio chunk not found: {chunk_path}")

    url = f"{settings.sarvam_base_url}/speech-to-text"
    headers = {"api-subscription-key": settings.sarvam_api_key}
    data = {
        "model": settings.sarvam_stt_model,
        "mode": mode,
        "with_timestamps": "false",
    }

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            with open(chunk_path, "rb") as f:
                files = {"file": (Path(chunk_path).name, f, "audio/wav")}
                response = requests.post(
                    url, headers=headers, files=files, data=data, timeout=300
                )
        except requests.exceptions.RequestException as exc:
            last_error = f"Network error contacting Sarvam AI: {exc}"
            time.sleep(min(2 ** attempt, 8))
            continue

        if response.status_code == 200:
            payload = response.json()
            return payload.get("transcript", "").strip(), payload.get("language_code")

        if response.status_code in (401, 403):
            raise TranscriptionError(
                "Sarvam AI rejected the request (auth error). Check that "
                "SARVAM_API_KEY is correct and active."
            )
        if response.status_code == 404:
            raise TranscriptionError(
                "Sarvam AI endpoint not found (404). The API may have "
                "changed - verify SARVAM_STT_MODEL and the API version."
            )
        if response.status_code == 422:
            raise TranscriptionError(
                f"Sarvam AI rejected the audio/parameters (422): {response.text[:300]}"
            )
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
            logger.warning("Sarvam AI rate limit hit, retrying in %ds...", retry_after)
            time.sleep(retry_after)
            last_error = "Sarvam AI rate limit (429) exceeded retries."
            continue
        if response.status_code >= 500:
            logger.warning(
                "Sarvam AI server error %d, retrying (%d/%d)...",
                response.status_code, attempt, max_retries,
            )
            last_error = f"Sarvam AI server error {response.status_code}."
            time.sleep(min(2 ** attempt, 8))
            continue

        raise TranscriptionError(
            f"Sarvam AI returned unexpected status {response.status_code}: "
            f"{response.text[:300]}"
        )

    raise TranscriptionError(last_error or "Sarvam AI request failed after retries.")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
) -> tuple[str, Optional[str]]:
    """Transcribe one audio chunk using the appropriate engine.

    Routing:
    - Auto Detect: Whisper detects the language, then English uses Whisper
      and non-English speech uses Sarvam.
    - Indic languages: Sarvam AI.
    - English: local Whisper.
    """
    lang = language.lower().strip()

    indic_languages = {
        "telugu",
        "hindi",
        "hinglish",
        "tamil",
        "kannada",
        "malayalam",
        "marathi",
        "bengali",
        "gujarati",
        "punjabi",
    }

    # Auto Detect must keep `detected` inside this branch. The previous
    # version used `detected` outside the branch, causing UnboundLocalError.
    if lang in {"auto", "auto detect", "automatic"}:
        detected = detect_language(chunk_path)

        logger.info(
            "Auto-detected language for %s: %s",
            Path(chunk_path).name,
            detected,
        )

        if detected == "en":
            return (
                transcribe_chunk_whisper(
                    chunk_path,
                    translate=False,
                ),
                "en",
            )

        # Sarvam can transcribe/translate supported Indic languages.
        text, sarvam_language = transcribe_chunk_sarvam(
            chunk_path,
            mode="translate",
        )
        return text, sarvam_language or detected

    # Indic languages -> Sarvam AI -> English output.
    if lang in indic_languages:
        return transcribe_chunk_sarvam(
            chunk_path,
            mode="translate",
        )

    # English (and the default fallback) -> local Whisper.
    return (
        transcribe_chunk_whisper(
            chunk_path,
            translate=False,
        ),
        "en",
    )


def transcribe_all(
    chunks: list[str],
    language: str = "english",
    chunk_seconds: int = CHUNK_SECONDS_DEFAULT,
    keep_original: bool = True,
) -> TranscriptResult:
    """Transcribe all chunks while preserving original chunk order.

    Sarvam cloud requests use limited parallelism. Whisper local inference
    remains sequential because multiple simultaneous GPU model calls can
    increase memory usage and reduce stability.

    For Auto Detect, the first chunk is used to choose the pipeline for the
    whole recording. This keeps processing fast and avoids repeatedly loading
    Whisper language-detection work for every chunk.
    """
    if not chunks:
        raise TranscriptionError("No audio chunks were provided.")

    lang = language.lower().strip()

    indic_languages = {
        "telugu",
        "hindi",
        "hinglish",
        "tamil",
        "kannada",
        "malayalam",
        "marathi",
        "bengali",
        "gujarati",
        "punjabi",
    }

    # Decide the engine once for the recording.
    if lang in {"auto", "auto detect", "automatic"}:
        detected_code = detect_language(chunks[0])

        logger.info(
            "Auto Detect selected. First chunk language: %s",
            detected_code,
        )

        is_sarvam = detected_code != "en"

        # After detection, route every chunk through one stable engine.
        language_for_chunks = (
            "english" if detected_code == "en" else "hinglish"
        )
    else:
        is_sarvam = lang in indic_languages
        language_for_chunks = language

    engine = "Sarvam AI" if is_sarvam else "Whisper"

    logger.info(
        "Transcribing %d chunk(s) with %s...",
        len(chunks),
        engine,
    )

    def transcribe_one(index: int, chunk: str):
        logger.info(
            "Transcribing chunk %d/%d...",
            index + 1,
            len(chunks),
        )

        text, lang_code = transcribe_chunk(
            chunk,
            language=language_for_chunks,
        )

        return index, text, lang_code

    # Sarvam is a cloud API, so keep concurrency deliberately low.
    if is_sarvam and len(chunks) > 1:
        with ThreadPoolExecutor(
            max_workers=min(SARVAM_MAX_WORKERS, len(chunks))
        ) as executor:
            futures = [
                executor.submit(transcribe_one, i, chunk)
                for i, chunk in enumerate(chunks)
            ]

            results = [future.result() for future in futures]

        # Threads can complete in a different order.
        results.sort(key=lambda item: item[0])

    else:
        # Whisper local GPU/CPU inference remains sequential.
        results = [
            transcribe_one(i, chunk)
            for i, chunk in enumerate(chunks)
        ]

    segments: list[TranscriptSegment] = []
    full_text_parts: list[str] = []

    for i, text, lang_code in results:
        start = i * chunk_seconds
        end = start + chunk_seconds

        segments.append(
            TranscriptSegment(
                text=text,
                start_seconds=start,
                end_seconds=end,
                engine=engine,
                language_code=lang_code,
            )
        )

        if text:
            full_text_parts.append(text)

    logger.info(
        "Transcription completed: %d/%d chunks produced.",
        sum(1 for _, text, _ in results if text),
        len(results),
    )

    return TranscriptResult(
        full_text=" ".join(full_text_parts).strip(),
        segments=segments,
        engine=engine,
    )


def transcribe_all_original(
    chunks: list[str],
    language: str = "english",
) -> str:
    """Transcribe every chunk in the ORIGINAL spoken language."""
    
    lang = language.lower().strip()
    parts: list[str] = []

    indic_languages = {
        "telugu",
        "hindi",
        "hinglish",
        "tamil",
        "kannada",
        "malayalam",
        "marathi",
        "bengali",
        "gujarati",
        "punjabi",
    }

    for i, chunk in enumerate(chunks):
        logger.info(
            "Transcribing original-language chunk %d/%d...",
            i + 1,
            len(chunks),
        )

        if lang in indic_languages:
            text, _ = transcribe_chunk_sarvam(
                chunk,
                mode="transcribe",
            )
        else:
            text = transcribe_chunk_whisper(
                chunk,
                translate=False,
            )

        if text:
            parts.append(text)

    return " ".join(parts).strip()
