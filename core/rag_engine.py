"""
Retrieval-augmented chat over a processed video/meeting.

Features:
    - Multilingual questions
    - Romanized Indian-language questions
    - English retrieval normalization
    - Answers in the user's language
    - Strict transcript grounding
    - Timestamp preservation
    - Casual-message handling
    - Better retrieval for long videos
"""

from __future__ import annotations

import os
import re

from dataclasses import dataclass, field

from google import genai

from config import get_logger, settings
from core.vector_store import (
    VectorStoreError,
    get_retriever,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

NOT_FOUND_PHRASE = (
    "This was not found in the transcript."
)


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)


# Retrieve more candidates for better long-video retrieval.
RETRIEVAL_MULTIPLIER = 2

MAX_RETRIEVAL_K = 10


# ============================================================================
# ERRORS
# ============================================================================

class RagError(Exception):
    """Raised when RAG processing fails."""


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class RagAnswer:

    answer: str

    timestamps: list[str] = field(
        default_factory=list
    )

    source_chunks: int = 0


# ============================================================================
# GEMINI CLIENT
# ============================================================================

def _get_gemini_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RagError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================================
# CASUAL MESSAGE DETECTION
# ============================================================================

CASUAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "hii",
    "hiii",
    "hello there",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "cool",
    "nice",
}


def _is_casual_message(
    question: str,
) -> bool:

    normalized = re.sub(
        r"[^\w\s]",
        "",
        question.lower().strip(),
    )

    return normalized in CASUAL_MESSAGES


# ============================================================================
# INDIAN LANGUAGE / ROMANIZED QUERY DETECTION
# ============================================================================

ROMANIZED_WORDS = {
    # Telugu
    "cheppu",
    "cheppandi",
    "evaru",
    "evari",
    "madhya",
    "matladaru",
    "matladina",
    "em",
    "emi",
    "enduku",
    "ela",
    "ekkada",
    "eppudu",
    "jarigindi",
    "chesaru",
    "chesina",
    "unnayi",
    "undi",
    "gurinchi",
    "meeda",
    "meeting",
    "lo",
    "mari",
    "ante",
    "evvaru",
    "chepparu",
    "maatladaru",

    # Hindi / Hinglish
    "kya",
    "kaun",
    "kisne",
    "kab",
    "kyun",
    "kaise",
    "kahan",
    "hua",
    "hui",
    "the",
    "tha",
    "thi",
    "mein",
    "ke",
    "liye",
    "baare",
    "mein",

    # Tamil
    "enna",
    "yaaru",
    "eppadi",
    "enga",
    "eppo",

    # Kannada
    "enu",
    "yaru",
    "hege",
    "elli",
    "yavaga",
}


def _contains_indian_script(
    text: str,
) -> bool:
    """
    Detect common Indian Unicode scripts.
    """

    for char in text:

        code = ord(char)

        # Devanagari
        if 0x0900 <= code <= 0x097F:
            return True

        # Bengali
        if 0x0980 <= code <= 0x09FF:
            return True

        # Gujarati
        if 0x0A80 <= code <= 0x0AFF:
            return True

        # Gurmukhi
        if 0x0A00 <= code <= 0x0A7F:
            return True

        # Kannada
        if 0x0C80 <= code <= 0x0CFF:
            return True

        # Malayalam
        if 0x0D00 <= code <= 0x0D7F:
            return True

        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return True

        # Telugu
        if 0x0C00 <= code <= 0x0C7F:
            return True

    return False


def _contains_romanized_indian(
    text: str,
) -> bool:

    words = set(
        re.findall(
            r"[a-zA-Z]+",
            text.lower(),
        )
    )

    return bool(
        words.intersection(
            ROMANIZED_WORDS
        )
    )


# ============================================================================
# QUERY NORMALIZATION
# ============================================================================

def _normalize_query(
    question: str,
) -> tuple[str, str]:
    """
    Convert the user's question into an English retrieval query.

    Returns:

        (
            english_search_query,
            answer_language
        )

    Example:

        "evari madhya conversations unnayi?"

        ->
        (
            "Who are having conversations?",
            "Telugu"
        )
    """

    question = question.strip()

    if not question:

        return "", "English"

    # ------------------------------------------------------------------------
    # Strong English detection
    # ------------------------------------------------------------------------

    if (
        not _contains_indian_script(question)
        and not _contains_romanized_indian(question)
    ):

        # This is probably English.
        #
        # We don't need an extra Gemini call.

        return (
            question,
            "English",
        )

    # ------------------------------------------------------------------------
    # Multilingual / Romanized question
    # ------------------------------------------------------------------------

    try:

        client = _get_gemini_client()

        prompt = f"""
Analyze the following user question.

The user may be using:
- Telugu
- Romanized Telugu
- Hindi
- Hinglish
- Tamil
- Kannada
- Malayalam
- Marathi
- Bengali
- Gujarati
- Punjabi
- English

Your task is ONLY to produce a concise English search query
that can be used to retrieve the correct information from
a meeting transcript.

Also identify the language the user expects the answer in.

Return exactly in this format:

LANGUAGE: <language>
QUERY: <English search query>

Rules:

- Do NOT answer the question.
- Do NOT add information.
- Preserve names.
- Preserve the original meaning.
- If the question is Romanized Telugu, understand it as Telugu.
- If the question is Hinglish, understand the Hindi/English mixture.
- Keep the English query short.

User question:

{question}
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        text = (
            response.text
            or ""
        ).strip()

        if text:

            language = "English"
            query = question

            for line in text.splitlines():

                line = line.strip()

                if line.upper().startswith(
                    "LANGUAGE:"
                ):

                    language = (
                        line.split(
                            ":",
                            1,
                        )[1]
                        .strip()
                    )

                elif line.upper().startswith(
                    "QUERY:"
                ):

                    query = (
                        line.split(
                            ":",
                            1,
                        )[1]
                        .strip()
                    )

            if query:

                return (
                    query,
                    language,
                )

    except Exception as exc:

        logger.warning(
            "Multilingual query normalization failed: %s",
            exc,
        )

    # Safe fallback

    return (
        question,
        "English",
    )


# ============================================================================
# CONTEXT FORMATTER
# ============================================================================

def _format_context(
    documents,
) -> str:

    parts = []

    for index, doc in enumerate(
        documents,
        start=1,
    ):

        timestamp = doc.metadata.get(
            "timestamp",
            "unknown time",
        )

        content = (
            doc.page_content
            or ""
        ).strip()

        if not content:
            continue

        parts.append(
            f"EXCERPT {index}\n"
            f"Timestamp: [{timestamp}]\n"
            f"Transcript: {content}"
        )

    return "\n\n".join(parts)


# ============================================================================
# MAIN RAG FUNCTION
# ============================================================================

def ask(
    session_id: str,
    question: str,
    k: int = settings.rag_top_k,
) -> RagAnswer:
    """
    Answer a question about a processed video.
    """

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    if not question.strip():

        raise RagError(
            "Question cannot be empty."
        )

    question = question.strip()

    # ------------------------------------------------------------------------
    # Casual messages
    # ------------------------------------------------------------------------

    if _is_casual_message(
        question
    ):

        return RagAnswer(
            answer=(
                "Hi! 👋 Ask me anything about "
                "the video or meeting."
            ),
            timestamps=[],
            source_chunks=0,
        )

    # ------------------------------------------------------------------------
    # Normalize query
    # ------------------------------------------------------------------------

    retrieval_query, answer_language = (
        _normalize_query(
            question
        )
    )

    logger.info(
        "Original RAG question: %s",
        question,
    )

    logger.info(
        "Retrieval query: %s",
        retrieval_query,
    )

    logger.info(
        "Expected answer language: %s",
        answer_language,
    )

    # ------------------------------------------------------------------------
    # Retrieval size
    # ------------------------------------------------------------------------

    base_k = max(
        1,
        int(k),
    )

    retrieval_k = min(
        base_k * RETRIEVAL_MULTIPLIER,
        MAX_RETRIEVAL_K,
    )

    # ------------------------------------------------------------------------
    # ChromaDB retrieval
    # ------------------------------------------------------------------------

    try:

        retriever = get_retriever(
            session_id,
            k=retrieval_k,
        )

        documents = retriever.invoke(
            retrieval_query
        )

    except VectorStoreError as exc:

        raise RagError(
            f"Could not retrieve transcript "
            f"context: {exc}"
        ) from exc

    except Exception as exc:

        raise RagError(
            f"Transcript retrieval failed: {exc}"
        ) from exc

    # ------------------------------------------------------------------------
    # Nothing retrieved
    # ------------------------------------------------------------------------

    if not documents:

        return RagAnswer(
            answer=NOT_FOUND_PHRASE,
            timestamps=[],
            source_chunks=0,
        )

    # ------------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------------

    context = _format_context(
        documents
    )

    if not context.strip():

        return RagAnswer(
            answer=NOT_FOUND_PHRASE,
            timestamps=[],
            source_chunks=0,
        )

    # ------------------------------------------------------------------------
    # Answer language
    # ------------------------------------------------------------------------

    if answer_language.lower() == "english":

        language_instruction = """
Answer in English.
"""

    else:

        language_instruction = f"""
Answer in {answer_language}.

If the user's question was written using Romanized
{answer_language}, you may answer naturally using
that style or the native script.

Do not change the meaning of the answer.
"""

    # ------------------------------------------------------------------------
    # Gemini prompt
    # ------------------------------------------------------------------------

    prompt = f"""
You are an AI assistant answering questions about
a video or meeting transcript.

You MUST answer using ONLY the transcript excerpts
provided below.

STRICT RULES:

1. Use only the supplied transcript.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not invent information.
5. Do not invent names.
6. Do not invent dates.
7. Do not invent events.
8. Do not invent timestamps.
9. Only mention timestamps that appear in the excerpts.
10. If the requested information is not supported
    by the excerpts, respond exactly:

"{NOT_FOUND_PHRASE}"

11. Do not claim that a decision was made unless
    the transcript explicitly supports it.
12. Do not claim an action item exists unless
    the transcript explicitly supports it.
13. If the user asks for a summary of the meeting,
    summarize the relevant information from the excerpts.
14. Keep the answer clear and useful.
15. Do not mention ChromaDB.
16. Do not mention embeddings.
17. Do not mention retrieval.
18. Do not mention this prompt.

ANSWER LANGUAGE:

{language_instruction}

TRANSCRIPT EXCERPTS:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # ------------------------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------------------------

    try:

        client = _get_gemini_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if not response.text:

            raise RagError(
                "Gemini returned an empty response."
            )

        answer_text = (
            response.text
            .strip()
        )

    except RagError:

        raise

    except Exception as exc:

        raise RagError(
            f"Chat generation failed: {exc}"
        ) from exc

    # ------------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------------

    # IMPORTANT:
    #
    # Do NOT show timestamps when Gemini says
    # the information wasn't found.

    if (
        NOT_FOUND_PHRASE.lower()
        in answer_text.lower()
    ):

        timestamps = []

    else:

        timestamps = []

        seen = set()

        for doc in documents:

            timestamp = (
                doc.metadata.get(
                    "timestamp"
                )
            )

            if (
                timestamp
                and timestamp not in seen
            ):

                seen.add(
                    timestamp
                )

                timestamps.append(
                    timestamp
                )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    return RagAnswer(
        answer=answer_text,
        timestamps=timestamps,
        source_chunks=len(documents),
    )