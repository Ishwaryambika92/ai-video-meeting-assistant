"""
Fast speech-to-text engine for the AI Video/Meeting Assistant.

Supported languages:
    English -> Whisper
    Telugu -> Sarvam AI
    Hindi -> Sarvam AI
    Hinglish -> Sarvam AI
    Tamil -> Sarvam AI
    Kannada -> Sarvam AI
    Malayalam -> Sarvam AI
    Marathi -> Sarvam AI
    Bengali -> Sarvam AI
    Gujarati -> Sarvam AI
    Punjabi -> Sarvam AI

Performance:
    - Whisper model loaded only once
    - Sarvam chunks processed concurrently
    - Auto Detect performed only once
    - Results restored to original chunk order
    - Retry handling for temporary API errors
    - Worker count configurable through .env

Language behavior:
    - Specific language selection keeps the original language
    - Auto Detect detects the language and keeps that language
    - Hinglish keeps the original spoken language
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
_whisper_import_error: Optional[Exception] = None
try:
    import whisper
except ImportError as err:
    _whisper_import_error = err
    # Auto-resolve from local virtual environment if invoked via global/external Python
    _venv_site_pkgs = Path(_PROJECT_ROOT) / ".venv" / "Lib" / "site-packages"
    if _venv_site_pkgs.exists() and str(_venv_site_pkgs) not in sys.path:
        sys.path.insert(0, str(_venv_site_pkgs))
        try:
            import whisper
            _whisper_import_error = None
        except ImportError as sub_err:
            _whisper_import_error = sub_err
            whisper = None  # type: ignore
    else:
        whisper = None  # type: ignore

from config import (
    get_device,
    get_logger,
    settings,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

CHUNK_SECONDS_DEFAULT = 30


# Number of simultaneous Sarvam requests.
#
# 2 = safer for rate limits
# 4 = faster
#
# Configure in .env:
#
# SARVAM_MAX_WORKERS=4

SARVAM_MAX_WORKERS = int(
    os.getenv(
        "SARVAM_MAX_WORKERS",
        "4",
    )
)


# Whisper is kept sequential.
#
# Running multiple Whisper inference jobs simultaneously can consume
# significant RAM/VRAM.

WHISPER_MAX_WORKERS = 1


# ============================================================================
# ERRORS
# ============================================================================

class TranscriptionError(Exception):
    """Raised when transcription fails."""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TranscriptSegment:
    """
    One transcript chunk with its approximate position
    in the recording.
    """

    text: str

    start_seconds: float

    end_seconds: float

    engine: str

    language_code: Optional[str] = None

    @property
    def start_timestamp(self) -> str:
        return _format_timestamp(
            self.start_seconds
        )

    @property
    def end_timestamp(self) -> str:
        return _format_timestamp(
            self.end_seconds
        )


@dataclass
class TranscriptResult:
    """Complete transcription result."""

    full_text: str

    segments: list[TranscriptSegment] = field(
        default_factory=list
    )

    engine: str = "whisper"


# ============================================================================
# TIMESTAMP
# ============================================================================

def _format_timestamp(
    seconds: float,
) -> str:
    """
    Convert seconds into:

        MM:SS

    or:

        HH:MM:SS
    """

    seconds = max(
        0,
        int(seconds),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, secs = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================================
# LANGUAGE SETS
# ============================================================================

INDIC_LANGUAGES = {
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


# Whisper language code -> application language name.
#
# This is important for Auto Detect.

LANGUAGE_NAMES = {
    "en": "english",
    "te": "telugu",
    "hi": "hindi",
    "ta": "tamil",
    "kn": "kannada",
    "ml": "malayalam",
    "mr": "marathi",
    "bn": "bengali",
    "gu": "gujarati",
    "pa": "punjabi",
}


LANGUAGE_CODES = {
    "english": "en",
    "telugu": "te",
    "hindi": "hi",
    "hinglish": "hi",
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "bengali": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
}


# ============================================================================
# WHISPER
# ============================================================================

_model = None
_model_device = None


def load_model():
    """
    Load Whisper once and reuse it.

    This is important for long videos because the model
    should not be loaded again for every chunk.
    """

    global _model
    global _model_device

    device = get_device()

    if (
        _model is not None
        and _model_device == device
    ):
        return _model

    logger.info(
        "Loading Whisper model '%s' on %s...",
        settings.whisper_model,
        device,
    )

    global whisper
    global _whisper_import_error

    if whisper is None:
        try:
            import whisper
            _whisper_import_error = None
        except Exception as err:
            _whisper_import_error = err
            whisper = None

    if whisper is None:
        err_msg = (
            f"OpenAI Whisper is not installed or failed to initialize ({_whisper_import_error}). "
            "Please verify requirements.txt to run local Whisper transcription."
        ) if _whisper_import_error else "OpenAI Whisper is not installed. Install requirements.txt to run local Whisper transcription."
        raise TranscriptionError(err_msg)

    try:

        _model = whisper.load_model(
            settings.whisper_model,
            device=device,
        )

        _model_device = device

        logger.info(
            "Whisper model loaded successfully on %s.",
            device,
        )

        return _model

    except Exception as exc:

        raise TranscriptionError(
            f"Failed to load Whisper model: {exc}"
        ) from exc


# ============================================================================
# AUTO LANGUAGE DETECTION
# ============================================================================

def detect_language(
    chunk_path: str,
) -> str:
    """
    Detect the spoken language using one audio chunk.

    Auto Detect is intentionally performed only once
    for the entire recording.

    Returns a Whisper language code such as:

        en
        te
        hi
        ta
        kn
    """

    if not Path(chunk_path).exists():

        raise TranscriptionError(
            f"Audio chunk not found: {chunk_path}"
        )

    model = load_model()
    assert whisper is not None

    try:

        audio = whisper.load_audio(
            chunk_path
        )

        audio = whisper.pad_or_trim(
            audio
        )

        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=model.dims.n_mels,
        ).to(
            model.device
        )

        _, probabilities = (
            model.detect_language(
                mel
            )
        )

        probs: dict[str, float] = (
            probabilities[0] if isinstance(probabilities, list) else probabilities  # pyright: ignore[reportAssignmentType]
        )

        detected_code = max(
            probs,
            key=lambda k: probs.get(k, 0.0),
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


# ============================================================================
# WHISPER TRANSCRIPTION
# ============================================================================

def transcribe_chunk_whisper(
    chunk_path: str,
    translate: bool = False,
) -> str:
    """
    Transcribe one chunk using Whisper.

    translate=False:
        Keep original language.

    translate=True:
        Translate to English.
    """

    if not Path(chunk_path).exists():

        raise TranscriptionError(
            f"Audio chunk not found: {chunk_path}"
        )

    model = load_model()

    task = (
        "translate"
        if translate
        else "transcribe"
    )

    try:

        import torch

        use_fp16 = (
            torch.cuda.is_available()
            and str(
                model.device
            ).startswith("cuda")
        )

        result = model.transcribe(
            chunk_path,
            task=task,
            fp16=use_fp16,
            verbose=False,
        )

        raw_text = result.get("text", "") if isinstance(result, dict) else ""
        return str(raw_text or "").strip()

    except Exception as exc:

        raise TranscriptionError(
            f"Whisper failed on "
            f"'{chunk_path}': {exc}"
        ) from exc


# ============================================================================
# SARVAM AI
# ============================================================================

def transcribe_chunk_sarvam(
    chunk_path: str,
    mode: str = "transcribe",
    max_retries: int = 3,
) -> tuple[str, Optional[str]]:
    """
    Transcribe one chunk using Sarvam AI.

    mode="transcribe":
        Keep source language.

    mode="translate":
        Translate source language to English.

    For the main application we use "transcribe"
    so Telugu/Hindi/etc. remain in their original language.
    """

    if not settings.sarvam_api_key:

        raise TranscriptionError(
            "SARVAM_API_KEY is not set. "
            "Add it to your .env file."
        )

    if not Path(chunk_path).exists():

        raise TranscriptionError(
            f"Audio chunk not found: {chunk_path}"
        )

    url = (
        f"{settings.sarvam_base_url}"
        "/speech-to-text"
    )

    headers = {
        "api-subscription-key":
            settings.sarvam_api_key,
    }

    data = {
        "model":
            settings.sarvam_stt_model,

        "mode":
            mode,

        "with_timestamps":
            "false",
    }

    last_error: Optional[str] = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        # --------------------------------------------------------------------
        # Request
        # --------------------------------------------------------------------

        try:

            with open(
                chunk_path,
                "rb",
            ) as audio_file:

                files = {
                    "file": (
                        Path(
                            chunk_path
                        ).name,

                        audio_file,

                        "audio/wav",
                    )
                }

                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300,
                )

        except requests.exceptions.RequestException as exc:

            last_error = (
                "Network error contacting "
                f"Sarvam AI: {exc}"
            )

            wait_time = min(
                2 ** attempt,
                8,
            )

            logger.warning(
                "Sarvam network error. "
                "Retrying in %ds...",
                wait_time,
            )

            time.sleep(
                wait_time
            )

            continue

        # --------------------------------------------------------------------
        # Success
        # --------------------------------------------------------------------

        if response.status_code == 200:

            try:

                payload = response.json()

            except Exception as exc:

                raise TranscriptionError(
                    f"Invalid JSON response from "
                    f"Sarvam AI: {exc}"
                ) from exc

            transcript = (
                payload.get(
                    "transcript",
                    "",
                )
                or ""
            ).strip()

            language_code = payload.get(
                "language_code"
            )

            return (
                transcript,
                language_code,
            )

        # --------------------------------------------------------------------
        # Authentication
        # --------------------------------------------------------------------

        if response.status_code in (
            401,
            403,
        ):

            raise TranscriptionError(
                "Sarvam AI rejected the request. "
                "Check that SARVAM_API_KEY is "
                "correct and active."
            )

        # --------------------------------------------------------------------
        # Endpoint / model
        # --------------------------------------------------------------------

        if response.status_code == 404:

            raise TranscriptionError(
                "Sarvam AI endpoint/model was "
                "not found. Check "
                "SARVAM_BASE_URL and "
                "SARVAM_STT_MODEL."
            )

        # --------------------------------------------------------------------
        # Invalid request
        # --------------------------------------------------------------------

        if response.status_code == 422:

            raise TranscriptionError(
                "Sarvam AI rejected the "
                "audio/parameters: "
                f"{response.text[:300]}"
            )

        # --------------------------------------------------------------------
        # Rate limit
        # --------------------------------------------------------------------

        if response.status_code == 429:

            retry_after_header = (
                response.headers.get(
                    "Retry-After"
                )
            )

            try:

                retry_after = int(
                    retry_after_header
                ) if retry_after_header else min(2 ** attempt, 10)

            except (
                TypeError,
                ValueError,
            ):

                retry_after = min(
                    2 ** attempt,
                    10,
                )

            logger.warning(
                "Sarvam rate limit hit. "
                "Retrying in %ds...",
                retry_after,
            )

            time.sleep(
                retry_after
            )

            last_error = (
                "Sarvam AI rate limit exceeded."
            )

            continue

        # --------------------------------------------------------------------
        # Server errors
        # --------------------------------------------------------------------

        if response.status_code >= 500:

            last_error = (
                "Sarvam AI server error "
                f"{response.status_code}."
            )

            wait_time = min(
                2 ** attempt,
                8,
            )

            logger.warning(
                "Sarvam server error %d. "
                "Retrying in %ds...",
                response.status_code,
                wait_time,
            )

            time.sleep(
                wait_time
            )

            continue

        # --------------------------------------------------------------------
        # Unknown error
        # --------------------------------------------------------------------

        raise TranscriptionError(
            "Sarvam AI returned unexpected "
            f"status {response.status_code}: "
            f"{response.text[:300]}"
        )

    raise TranscriptionError(
        last_error
        or "Sarvam request failed "
           "after retries."
    )


# ============================================================================
# SINGLE CHUNK ROUTER
# ============================================================================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
) -> tuple[str, Optional[str]]:
    """
    Route one chunk to Whisper or Sarvam.

    IMPORTANT:
        Specific Indic language selection keeps
        the original language.

        Telugu -> Telugu
        Hindi -> Hindi
        Tamil -> Tamil
        etc.
    """

    lang = (
        language
        .lower()
        .strip()
    )

    # ------------------------------------------------------------------------
    # Auto Detect
    # ------------------------------------------------------------------------

    if lang in {
        "auto",
        "auto detect",
        "automatic",
    }:

        detected = detect_language(
            chunk_path
        )

        logger.info(
            "Auto Detect detected: %s",
            detected,
        )

        # English -> Whisper
        if detected == "en":

            return (
                transcribe_chunk_whisper(
                    chunk_path,
                    translate=False,
                ),
                "en",
            )

        # Non-English -> Sarvam
        # IMPORTANT: keep original language.
        text, sarvam_language = (
            transcribe_chunk_sarvam(
                chunk_path,
                mode="transcribe",
            )
        )

        return (
            text,
            sarvam_language or detected,
        )

    # ------------------------------------------------------------------------
    # Indic languages
    # ------------------------------------------------------------------------

    if lang in INDIC_LANGUAGES:

        # IMPORTANT:
        #
        # Use "transcribe", NOT "translate".
        #
        # This keeps:
        #
        # Telugu -> Telugu
        # Hindi -> Hindi
        # Tamil -> Tamil
        # Kannada -> Kannada
        # etc.

        return transcribe_chunk_sarvam(
            chunk_path,
            mode="transcribe",
        )

    # ------------------------------------------------------------------------
    # English / fallback
    # ------------------------------------------------------------------------

    return (
        transcribe_chunk_whisper(
            chunk_path,
            translate=False,
        ),
        "en",
    )


# ============================================================================
# TRANSCRIBE ALL CHUNKS
# ============================================================================

def transcribe_all(
    chunks: list[str],
    language: str = "english",
    chunk_seconds: int = CHUNK_SECONDS_DEFAULT,
    keep_original: bool = True,
) -> TranscriptResult:
    """
    Transcribe all chunks efficiently.

    Sarvam:
        Parallel cloud requests.

    Whisper:
        Sequential processing.

    Auto Detect:
        Detect language only once.
    """

    if not chunks:

        raise TranscriptionError(
            "No audio chunks were provided."
        )

    lang = (
        language
        .lower()
        .strip()
    )

    # ------------------------------------------------------------------------
    # Decide engine once
    # ------------------------------------------------------------------------

    if lang in {
        "auto",
        "auto detect",
        "automatic",
    }:

        detected_code = detect_language(
            chunks[0]
        )

        logger.info(
            "Auto Detect result: %s",
            detected_code,
        )

        # English
        if detected_code == "en":

            is_sarvam = False

            language_for_chunks = (
                "english"
            )

        # Non-English
        else:

            is_sarvam = True

            # Convert Whisper code to application
            # language name.
            #
            # Example:
            # te -> telugu
            # hi -> hindi
            # ta -> tamil

            language_for_chunks = (
                LANGUAGE_NAMES.get(
                    detected_code
                )
            )

            if not language_for_chunks:

                raise TranscriptionError(
                    "Auto Detect found an "
                    f"unsupported language code: "
                    f"{detected_code}"
                )

    else:

        is_sarvam = (
            lang in INDIC_LANGUAGES
        )

        language_for_chunks = lang

    engine = (
        "Sarvam AI"
        if is_sarvam
        else "Whisper"
    )

    logger.info(
        "Starting transcription: "
        "%d chunks using %s.",
        len(chunks),
        engine,
    )

    logger.info(
        "Selected transcription language: %s",
        language_for_chunks,
    )

    # ------------------------------------------------------------------------
    # One chunk
    # ------------------------------------------------------------------------

    if len(chunks) == 1:

        text, lang_code = (
            transcribe_chunk(
                chunks[0],
                language=language_for_chunks,
            )
        )

        results = [
            (
                0,
                text,
                lang_code,
            )
        ]

    # ------------------------------------------------------------------------
    # Sarvam parallel
    # ------------------------------------------------------------------------

    elif is_sarvam:

        worker_count = min(
            SARVAM_MAX_WORKERS,
            len(chunks),
        )

        logger.info(
            "Using %d parallel Sarvam workers.",
            worker_count,
        )

        results = []

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:

            future_map = {
                executor.submit(
                    _transcribe_one,
                    index,
                    chunk,
                    language_for_chunks,
                    len(chunks),
                ): index

                for index, chunk
                in enumerate(chunks)
            }

            for future in as_completed(
                future_map
            ):

                result = future.result()

                results.append(
                    result
                )

        # Threads finish in arbitrary order.
        # Restore original chunk order.

        results.sort(
            key=lambda item: item[0]
        )

    # ------------------------------------------------------------------------
    # Whisper sequential
    # ------------------------------------------------------------------------

    else:

        results = []

        for index, chunk in enumerate(
            chunks
        ):

            results.append(
                _transcribe_one(
                    index,
                    chunk,
                    language_for_chunks,
                    len(chunks),
                )
            )

    # ------------------------------------------------------------------------
    # Build segments
    # ------------------------------------------------------------------------

    segments: list[
        TranscriptSegment
    ] = []

    full_text_parts: list[str] = []

    for (
        index,
        text,
        lang_code,
    ) in results:

        start_seconds = (
            index * chunk_seconds
        )

        end_seconds = (
            start_seconds
            + chunk_seconds
        )

        segments.append(
            TranscriptSegment(
                text=text,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                engine=engine,
                language_code=lang_code,
            )
        )

        if text:

            full_text_parts.append(
                text
            )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    successful_chunks = sum(
        1
        for _, text, _ in results
        if text
    )

    logger.info(
        "Transcription completed: "
        "%d/%d chunks produced text.",
        successful_chunks,
        len(results),
    )

    # ------------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------------

    return TranscriptResult(
        full_text=" ".join(
            full_text_parts
        ).strip(),

        segments=segments,

        engine=engine,
    )


# ============================================================================
# CHUNK WORKER
# ============================================================================

def _transcribe_one(
    index: int,
    chunk: str,
    language: str,
    total_chunks: int,
):
    """
    Process one chunk.

    Returns:
        (index, text, language_code)
    """

    logger.info(
        "Transcribing chunk %d/%d...",
        index + 1,
        total_chunks,
    )

    text, language_code = (
        transcribe_chunk(
            chunk,
            language=language,
        )
    )

    logger.info(
        "Finished chunk %d/%d.",
        index + 1,
        total_chunks,
    )

    return (
        index,
        text,
        language_code,
    )


# ============================================================================
# ORIGINAL-LANGUAGE TRANSCRIPTION
# ============================================================================

def transcribe_all_original(
    chunks: list[str],
    language: str = "english",
) -> str:
    """
    Transcribe all chunks in their original language.

    Mainly used when the application needs to preserve
    the original Hinglish / Indic transcript separately.
    """

    if not chunks:

        return ""

    lang = (
        language
        .lower()
        .strip()
    )

    # ------------------------------------------------------------------------
    # English
    # ------------------------------------------------------------------------

    if lang == "english":

        whisper_results = [
            transcribe_chunk_whisper(
                chunk,
                translate=False,
            )

            for chunk in chunks
        ]

        return " ".join(
            text
            for text in whisper_results
            if text
        ).strip()

    # ------------------------------------------------------------------------
    # Indic languages
    # ------------------------------------------------------------------------

    if lang in INDIC_LANGUAGES:

        worker_count = min(
            SARVAM_MAX_WORKERS,
            len(chunks),
        )

        logger.info(
            "Transcribing original "
            "language using %d Sarvam workers.",
            worker_count,
        )

        indic_results: list[
            tuple[int, str]
        ] = []

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:

            future_map = {
                executor.submit(
                    _transcribe_original_one,
                    index,
                    chunk,
                    len(chunks),
                    lang,
                ): index

                for index, chunk
                in enumerate(chunks)
            }

            for future in as_completed(
                future_map
            ):

                indic_results.append(
                    future.result()
                )

        indic_results.sort(
            key=lambda item: item[0]
        )

        return " ".join(
            text
            for _, text in indic_results
            if text
        ).strip()

    # ------------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------------

    fallback_results = [
        transcribe_chunk_whisper(
            chunk,
            translate=False,
        )

        for chunk in chunks
    ]

    return " ".join(
        text
        for text in fallback_results
        if text
    ).strip()


# ============================================================================
# ORIGINAL-LANGUAGE WORKER
# ============================================================================

def _transcribe_original_one(
    index: int,
    chunk: str,
    total_chunks: int,
    language: str,
):
    """
    Transcribe one chunk in its original language.
    """

    logger.info(
        "Original transcript chunk %d/%d...",
        index + 1,
        total_chunks,
    )

    text, _ = transcribe_chunk_sarvam(
        chunk,
        mode="transcribe",
    )

    logger.info(
        "Finished original chunk %d/%d.",
        index + 1,
        total_chunks,
    )

    return (
        index,
        text,
    )