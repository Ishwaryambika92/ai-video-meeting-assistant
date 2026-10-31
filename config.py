"""
Centralized configuration for the AI Video/Meeting Assistant.

Loads environment variables once, exposes typed settings, sets up
application-wide logging, and resolves runtime hardware (CPU/GPU).

Every other module should import from here instead of calling
os.getenv() / load_dotenv() directly, so configuration stays in one
place and is easy to audit.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure module is safely registered in sys.modules during Streamlit hot-reloads
if __name__ in sys.modules:
    sys.modules.setdefault("config", sys.modules[__name__])

try:
    import pyarrow
except ImportError:
    pass

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DOWNLOAD_DIR = PROJECT_ROOT / "downloads"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _dir in (DOWNLOAD_DIR, CHROMA_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def _setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quiet down noisy third-party loggers by default.
    for noisy in (
        "httpx",
        "urllib3",
        "sentence_transformers",
        "chromadb",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_setup_logging()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:

    # ----------------------------------------------------------------------
    # Whisper
    # ----------------------------------------------------------------------
    # Local English transcription

    whisper_model: str = os.getenv(
        "WHISPER_MODEL",
        "base",
    )

    # ----------------------------------------------------------------------
    # Sarvam AI
    # ----------------------------------------------------------------------
    # Hindi / Hinglish transcription

    sarvam_api_key: str = os.getenv(
        "SARVAM_API_KEY",
        "",
    )

    sarvam_stt_model: str = os.getenv(
        "SARVAM_STT_MODEL",
        "saaras:v3",
    )

    sarvam_base_url: str = os.getenv(
        "SARVAM_BASE_URL",
        "https://api.sarvam.ai",
    )

    # ----------------------------------------------------------------------
    # Gemini
    # ----------------------------------------------------------------------
    # Summarization, extraction and RAG chat

    gemini_api_key: str = os.getenv(
        "GEMINI_API_KEY",
        "",
    )

    gemini_model: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    # ----------------------------------------------------------------------
    # Embeddings
    # ----------------------------------------------------------------------

    hf_token: str = os.getenv(
        "HF_TOKEN",
        "",
    )

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # ----------------------------------------------------------------------
    # Chunking
    # ----------------------------------------------------------------------

    audio_chunk_minutes: int = int(
        os.getenv(
            "AUDIO_CHUNK_MINUTES",
            "10",
        )
    )

    transcript_chunk_size: int = int(
        os.getenv(
            "TRANSCRIPT_CHUNK_SIZE",
            "1200",
        )
    )

    transcript_chunk_overlap: int = int(
        os.getenv(
            "TRANSCRIPT_CHUNK_OVERLAP",
            "150",
        )
    )

    # ----------------------------------------------------------------------
    # RAG
    # ----------------------------------------------------------------------

    rag_top_k: int = int(
        os.getenv(
            "RAG_TOP_K",
            "4",
        )
    )

    # ----------------------------------------------------------------------
    # Authentication & Supabase
    # ----------------------------------------------------------------------

    supabase_url: str = os.getenv("SUPABASE_URL", "").strip()
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "").strip()
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    auth_secret_key: str = os.getenv("AUTH_SECRET_KEY", "video-agent-auth-secret-key-change-in-prod").strip()
    require_auth: bool = os.getenv("REQUIRE_AUTH", "true").lower() in ("true", "1", "yes")


settings = Settings()


def get_settings() -> Settings:
    """Return the global Settings singleton instance."""
    return settings


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------

def validate_settings(
    require_gemini: bool = True,
    require_sarvam: bool = False,
) -> list[str]:
    """
    Return a list of human-readable configuration problems.
    """

    problems: list[str] = []

    # ----------------------------------------------------------------------
    # Gemini
    # ----------------------------------------------------------------------

    if require_gemini and not settings.gemini_api_key:
        problems.append(
            "GEMINI_API_KEY is not set. "
            "Summarization, extraction and chat will not work."
        )

    # ----------------------------------------------------------------------
    # Sarvam
    # ----------------------------------------------------------------------

    if require_sarvam and not settings.sarvam_api_key:
        problems.append(
            "SARVAM_API_KEY is not set. "
            "Hindi/Hinglish transcription will not work."
        )

    return problems

# --------------------------------------------------------------------------
# Hardware detection
# --------------------------------------------------------------------------

def get_device() -> str:
    """
    Return 'cuda' if a usable NVIDIA GPU is available,
    otherwise return 'cpu'.
    """

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except ImportError:
        pass

    return "cpu"