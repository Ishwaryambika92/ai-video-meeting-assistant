"""
Translation & transcript cleanup engine.

Responsibilities:
    - Maintains two distinct transcript streams:
        * original transcript -> exact verbatim spoken audio record
        * clean English transcript -> polished, highly readable, structured text
          used for executive reporting, summarization, action item extraction, and RAG.
    - Deep speech-to-text cleanup:
        * Strips filler words (um, uh, er, ah, you know, like, basically, actually).
        * Removes stuttered word repetitions and false starts.
        * Contextually repairs speech recognition phonetic errors (e.g., 'cooper nettles' -> 'Kubernetes',
          'sink' -> 'sync', 'cash' -> 'cache', 'data base' -> 'database', 'api end points' -> 'API endpoints').
        * Corrects grammar, punctuation, sentence boundaries, and capitalization.
        * Strictly preserves 100% of factual details: names, numbers, dates, technical terms,
          commitments, and decisions.
        * Strictly preserves speaker labels and timestamps when present, and never invents them when absent.
        * Performs intelligent semantic chunking for long transcripts with retry and graceful heuristic fallbacks.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from google import genai
from google.genai import errors

from config import get_logger, settings

logger = get_logger(__name__)

_MAX_MT_CHARS = 4500
_MAX_EDITORIAL_CHUNK_CHARS = 4000

# Supported language name to ISO 639-1 code mapping
LANGUAGE_MAP: dict[str, str] = {
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
    "auto": "auto",
    "auto detect": "auto",
    "automatic": "auto",
}


class TranslationError(Exception):
    """Raised when translation or transcript cleanup fails."""


def _get_gemini_client() -> genai.Client:
    """Create and return a configured Gemini client."""
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise TranslationError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )
    return genai.Client(api_key=api_key)


def _heuristic_clean(text: str) -> str:
    """Fast local fallback to strip common verbal fillers and stutters if AI fails."""
    if not text:
        return ""

    # Remove standalone filler sounds (um, uh, er, ah, etc.)
    cleaned = re.sub(r"\b(um+|uh+|er+|ah+)\b[,.]?\s*", "", text, flags=re.IGNORECASE)

    # Remove immediately repeated words (e.g. "we we", "to to", "in in")
    cleaned = re.sub(r"\b([A-Za-z]{2,})\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)

    # Normalize excessive spaces and clean punctuation spacing
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.:;?!])", r"\1", cleaned)

    return cleaned.strip()


def _split_into_editorial_chunks(text: str, max_chars: int = _MAX_EDITORIAL_CHUNK_CHARS) -> list[str]:
    """Split long transcripts into coherent chunks along speaker, paragraph, or sentence boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # If transcript has line breaks or speaker tags, split by lines/paragraphs
    if "\n" in text:
        units = [u.strip() for u in text.split("\n") if u.strip()]
        delim = "\n\n"
    else:
        # Split along sentence boundaries (?<=[.?!])\s+
        units = re.split(r"(?<=[.?!])\s+", text)
        delim = " "

    chunks: list[str] = []
    current_units: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        if current_len + unit_len + len(delim) <= max_chars:
            current_units.append(unit)
            current_len += unit_len + len(delim)
        else:
            if current_units:
                chunks.append(delim.join(current_units))
                current_units = []
                current_len = 0

            if unit_len > max_chars:
                # Fallback for unusually long unbroken block: split by word
                words = unit.split(" ")
                sub_chunk: list[str] = []
                sub_len = 0
                for w in words:
                    if sub_len + len(w) + 1 <= max_chars:
                        sub_chunk.append(w)
                        sub_len += len(w) + 1
                    else:
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk))
                        sub_chunk = [w]
                        sub_len = len(w)
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
            else:
                current_units.append(unit)
                current_len = unit_len

    if current_units:
        chunks.append(delim.join(current_units))

    return chunks


CLEANUP_SYSTEM_PROMPT = """You are an elite executive speech transcript editor and linguist.
Your objective is to transform raw speech-to-text transcripts into clean, highly readable, and professional English while faithfully preserving 100% of the original meaning, speaker intent, and facts.

STRICT EDITING RULES:
1. REMOVE FILLER WORDS & VOCAL DISFLUENCIES:
   - Strip filler words: 'um', 'uh', 'ah', 'er', 'you know', 'sort of', 'kind of', 'basically', 'actually', 'I mean', and casual filler 'like' (retain 'like' only when used for comparison or preference, e.g., 'like a server' or 'I like Python').
   - Remove stuttered words and repeated phrases (e.g., 'we we need to to' -> 'we need to').
   - Clean up false starts without losing the speaker's intended thought.

2. CORRECT GRAMMAR, SYNTAX & PUNCTUATION:
   - Correct grammatical mistakes, tense inconsistencies, and subject-verb agreements (e.g., 'the customers was complaining' -> 'the customers were complaining', 'me and Sarah was' -> 'Sarah and I were').
   - Add proper capitalization, commas, periods, apostrophes, and question marks.
   - Break run-on spoken speech into natural, readable, fluent written sentences.

3. REPAIR CONTEXTUAL SPEECH-TO-TEXT ERRORS:
   - Intelligently fix obvious speech recognition phonetic errors using surrounding context (e.g., 'cooper nettles' -> 'Kubernetes', 'sink between database' -> 'sync', 'cash server' -> 'cache server', 'data base' -> 'database', 'api end points' -> 'API endpoints', 'fast a p i' -> 'FastAPI').
   - Correct homophones and common transcription slips ('to' -> 'too', 'their' -> 'there', 'then' -> 'than', 'won\\'t' -> 'want') when context makes the intent obvious.

4. PRESERVE ALL FACTS, DETAILS & STRUCTURE:
   - If speaker labels and timestamps are present, PRESERVE them exactly in their original position (e.g., 'Speaker 1 (00:00):', 'Alex:').
   - If speaker labels or timestamps are NOT present in the raw transcript, NEVER invent or add speaker labels or timestamps.
   - PRESERVE all factual details: names, numbers, dates, technical terms, metrics, decisions, and commitments with 100% fidelity.
   - NEVER summarize, shorten, or omit content. Every spoken thought and detail must remain.
   - NEVER hallucinate or invent new information.
   - Maintain clear paragraph breaks between speaker turns or natural topic shifts.

Return ONLY the clean transcript. Do not include any explanations, greetings, preamble, or markdown code fences."""


def _clean_chunk_with_gemini(
    client: genai.Client,
    chunk: str,
    max_retries: int = 2,
) -> str:
    """Clean a single transcript chunk using Gemini with retry handling and heuristic fallback."""
    if not chunk.strip():
        return ""

    prompt = f"{CLEANUP_SYSTEM_PROMPT}\n\nRaw Transcript:\n{chunk}"

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            if response.text and response.text.strip():
                return response.text.strip()
            logger.warning("Gemini returned empty text for chunk cleanup (attempt %d).", attempt + 1)
        except Exception as exc:
            logger.warning(
                "Gemini cleanup attempt %d failed (%s).",
                attempt + 1,
                exc,
            )
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))

    logger.warning("Gemini cleanup exhausted retries; applying local heuristic cleaner to chunk.")
    return _heuristic_clean(chunk)


def _llm_cleanup(text: str) -> str:
    """Clean speech transcript using Gemini with semantic chunking and graceful fallbacks."""
    if not text.strip():
        return ""

    try:
        client = _get_gemini_client()
    except TranslationError:
        logger.warning("GEMINI_API_KEY not configured; using local heuristic cleaner.")
        return _heuristic_clean(text)

    chunks = _split_into_editorial_chunks(text, max_chars=_MAX_EDITORIAL_CHUNK_CHARS)
    logger.info("Cleaning transcript in %d chunk(s) using model %s...", len(chunks), settings.gemini_model)

    cleaned_chunks: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            logger.info("Cleaning transcript chunk %d/%d (%d chars)...", i, len(chunks), len(chunk))
        cleaned_piece = _clean_chunk_with_gemini(client, chunk)
        cleaned_chunks.append(cleaned_piece)

    return "\n\n".join(cleaned_chunks).strip()


def _gemini_translate(text: str, source_language: str = "auto") -> str:
    """Translate text directly to English using Gemini."""
    if not text.strip():
        return ""

    prompt = f"""You are a professional translator and editor.
Translate the following speech transcript into clean, natural English.
Rules:
- Translate accurately to English.
- Fix grammar, punctuation, and capitalization.
- Remove filler words (um, uh, etc.).
- Preserve all facts, names, numbers, dates, and meaning.
- Return ONLY the clean English translation without any explanation, preamble, or notes.

Transcript:
{text}
"""

    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        if response.text and response.text.strip():
            return response.text.strip()
    except Exception as exc:
        logger.warning(
            "Gemini translation failed (%s); keeping original text.",
            exc,
        )

    return text


def _machine_translate(text: str, source_language: str = "auto") -> str:
    """Fast bulk translation to English using Google Translate, falling back to Gemini."""
    if not text.strip():
        return ""

    # Normalize language identifier
    clean_lang = source_language.lower().strip()
    normalized_source = LANGUAGE_MAP.get(clean_lang, clean_lang)

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(
            source=normalized_source,
            target="en",
        )

        chunks = [
            text[i : i + _MAX_MT_CHARS]
            for i in range(0, len(text), _MAX_MT_CHARS)
        ]

        translated_parts = []
        for chunk in chunks:
            translated = translator.translate(chunk)
            if not translated:
                raise ValueError("Empty translation chunk received")
            translated_parts.append(translated)

        return " ".join(translated_parts).strip()

    except Exception as exc:
        logger.warning(
            "Machine translation failed (%s); falling back to Gemini translation.",
            exc,
        )
        return _gemini_translate(text, source_language=normalized_source)


def translate_to_clean_english(
    original_text: str,
    source_language: str = "hi",
) -> str:
    """
    Translate an Indic/multilingual transcript into clean, natural English.
    The original text is never modified.
    """
    if not original_text.strip():
        return ""

    logger.info("Translating transcript to English (source: %s)...", source_language)
    machine_translated = _machine_translate(
        original_text,
        source_language=source_language,
    )

    logger.info("Running Gemini editorial cleanup pass on translated transcript...")
    return _llm_cleanup(machine_translated)


def clean_transcript(text: str) -> str:
    """
    Polish an English transcript: remove filler words, correct grammar/punctuation,
    fix speech-to-text recognition errors, and preserve meaning, speaker tags, and facts.
    """
    return _llm_cleanup(text)