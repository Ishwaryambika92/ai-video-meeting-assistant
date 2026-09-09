"""
Translation & transcript cleanup.

The pipeline keeps two distinct pieces of data:
    - original transcript -> exactly what was spoken
    - clean English transcript -> used for summarization/RAG

For Hindi/Hinglish:
    Google Translate -> Gemini cleanup

For English:
    Gemini cleanup only.
"""

from __future__ import annotations

import os

from google import genai

from config import get_logger

logger = get_logger(__name__)

_MAX_MT_CHARS = 4500


class TranslationError(Exception):
    pass


def _get_gemini_client():
    """Create the Gemini client."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise TranslationError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    return genai.Client(api_key=api_key)


def _machine_translate(text: str, source_language: str = "auto") -> str:
    """Fast bulk translation to English using Google Translate."""

    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import RequestError, TranslationNotFound

    if not text.strip():
        return ""

    translator = GoogleTranslator(
        source=source_language,
        target="en",
    )

    chunks = [
        text[i : i + _MAX_MT_CHARS]
        for i in range(0, len(text), _MAX_MT_CHARS)
    ]

    translated_parts = []

    for chunk in chunks:
        try:
            translated_parts.append(
                translator.translate(chunk) or ""
            )

        except (RequestError, TranslationNotFound) as exc:
            logger.warning(
                "Machine translation chunk failed (%s); "
                "keeping original text.",
                exc,
            )
            translated_parts.append(chunk)

    return " ".join(translated_parts).strip()



def _llm_cleanup(text: str) -> str:
    """Clean transcript using Gemini."""

    if not text.strip():
        return ""

    prompt = f"""
You are an expert transcript editor.

Clean the following speech-to-text transcript.

Rules:
- Fix grammar.
- Fix punctuation.
- Fix capitalization.
- Remove filler words such as um, uh, and repeated words.
- Remove obvious speech-to-text artifacts.
- Do NOT summarize.
- Do NOT shorten the content.
- Do NOT translate the content.
- Do NOT add new information.
- Do NOT remove factual information.
- Preserve all names, numbers, facts, events and ideas.
- Preserve the original meaning.
- Return ONLY the cleaned transcript.
- Do not add explanations or a preamble.

Transcript:

{text}
"""

    try:
        client = _get_gemini_client()

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        if not response.text:
            logger.warning(
                "Gemini returned an empty cleanup response; "
                "using original text."
            )
            return text

        return response.text.strip()

    except TranslationError:
        raise

    except Exception as exc:
        logger.warning(
            "Gemini cleanup failed (%s); "
            "falling back to un-cleaned text.",
            exc,
        )
        return text


def translate_to_clean_english(
    original_text: str,
    source_language: str = "hi",
) -> str:
    """
    Translate a Hindi/Hinglish transcript into clean English.

    The original text is never modified.
    """

    if not original_text.strip():
        return ""

    logger.info(
        "Machine-translating transcript to English..."
    )

    machine_translated = _machine_translate(
        original_text,
        source_language=source_language,
    )

    logger.info(
        "Running Gemini cleanup pass on translated transcript..."
    )

    return _llm_cleanup(machine_translated)


def clean_transcript(text: str) -> str:
    """
    Tidy up an already-English transcript without translating it.
    """

    return _llm_cleanup(text)