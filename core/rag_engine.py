"""
Retrieval-augmented chat over a processed video/meeting.

Features:
    - Persistent language preference handling (English, Telugu script, Telugu-English mix, Hindi, Tamil)
    - Pre-retrieval language command detection
    - Resolution of short follow-up messages ("cheppu", "continue", "explain", "k tell me", "inka cheppu")
    - Dynamic language-specific system prompts with strict script enforcement
    - Strict transcript grounding without hallucinations
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from google import genai
from google.genai import errors
import time

from config import get_logger, settings

from core.vector_store import (
    VectorStoreError,
    get_retriever,
)

logger = get_logger(__name__)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

NOT_FOUND_PHRASE = "This was not found in the transcript."


RETRIEVAL_MULTIPLIER = 2
MAX_RETRIEVAL_K = 10


class ResponseLanguage:
    ENGLISH = "english"
    TELUGU = "telugu"                 # Telugu script (తెలుగు లిపి)
    TELUGU_ENGLISH = "telugu_english" # Roman Telugu / Tanglish (Latin letters + English terms)
    HINDI = "hindi"                   # Hindi in Devanagari script (हिंदी)
    TAMIL = "tamil"                   # Tamil in Tamil script (தமிழ்)

    # Aliases
    ROMAN_TELUGU = "telugu_english"
    TELUGU_SCRIPT = "telugu"


# ============================================================================
# ERRORS & DATA STRUCTURES
# ============================================================================

class RagError(Exception):
    """Raised when RAG processing fails."""


@dataclass
class RagAnswer:
    answer: str
    timestamps: list[str] = field(default_factory=list)
    source_chunks: int = 0
    language: str = ResponseLanguage.ENGLISH


# ============================================================================
# GEMINI CLIENT
# ============================================================================

def _get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RagError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )
    return genai.Client(api_key=api_key, http_options={"timeout": 60000})


# ============================================================================
# SCRIPT & UNICODE DETECTION HELPERS
# ============================================================================

def contains_tamil_script(text: str) -> bool:
    """Detect if any Tamil Unicode characters (0x0B80 - 0x0BFF) are present."""
    return any(0x0B80 <= ord(c) <= 0x0BFF for c in text)


def contains_telugu_script(text: str) -> bool:
    """Detect if any Telugu Unicode characters (0x0C00 - 0x0C7F) are present."""
    return any(0x0C00 <= ord(c) <= 0x0C7F for c in text)


def contains_devanagari_script(text: str) -> bool:
    """Detect if any Devanagari Unicode characters (0x0900 - 0x097F) are present."""
    return any(0x0900 <= ord(c) <= 0x097F for c in text)


def _parse_conversation_history(conversation_history: Any) -> list[dict[str, str]]:
    """Normalize conversation history from various formats (tuples, dicts) into standard dicts."""
    parsed: list[dict[str, str]] = []
    if not conversation_history:
        return parsed

    for item in conversation_history:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            role = str(item[0]).lower()
            text = str(item[1])
            parsed.append({"role": "user" if "user" in role else "assistant", "content": text})
        elif isinstance(item, dict):
            role = str(item.get("role", "user")).lower()
            text = str(item.get("content", item.get("text", "")))
            parsed.append({"role": "user" if "user" in role else "assistant", "content": text})
    return parsed


# ============================================================================
# 1. LANGUAGE COMMAND DETECTION (RUNS BEFORE RETRIEVAL)
# ============================================================================

TEL_ENG_PATTERNS = [
    r"\b(telugu[\s\-_]?english|tel[\s\-_]?eng|tanglish|roman(?:ized)?[\s\-_]?telugu|english[\s\-_]?telugu)\b",
    r"\b(telugu\s+(?:and\s+)?english\s+mix|tel\s+eng\s+mix|telugu\s+mix)\b",
    r"\bchange\s+to\s+telugu[\s\-_]?english(?:\s+mix)?\b",
    r"\bnaku\s+tel(?:ugu)?[\s\-_]?eng(?:lish)?\s+mix\s+lo\s+cheppu\b",
    r"\benglish\s+and\s+telugu\s+mix\b",
]

HINDI_PATTERNS = [
    r"\b(hindi\s+lo\s+cheppu|naku\s+hindi\s+lo\s+cheppu|hindi\s+lo|hindi\s+mein\s+batao|hindi\s+me\s+batao|hindi\s+me\s+bolo)\b",
    r"\b(change\s+(?:language\s+)?to\s+hindi|switch\s+to\s+hindi|answer\s+in\s+hindi|in\s+hindi)\b",
    r"\b(from\s+now\s+on,?\s+(?:answer\s+in\s+)?hindi)\b",
]

TELUGU_PATTERNS = [
    r"\b(naku\s+eppudu\s+telugu\s+lo\s+cheppu|naku\s+telugu\s+lo\s+cheppu|telugu\s+lo\s+cheppu|telugulo\s+cheppu)\b",
    r"\b(change\s+(?:language\s+)?to\s+telugu|switch\s+to\s+telugu|answer\s+in\s+telugu|in\s+telugu|telugu\s+script)\b",
    r"\b(from\s+now\s+on,?\s+(?:answer\s+in\s+)?telugu)\b",
    r"\b(eppudu\s+telugu\s+lo\s+cheppu|telugu\s+lo)\b",
]

ENGLISH_PATTERNS = [
    r"\b(tell\s+me\s+in\s+english|answer\s+in\s+english|in\s+english|only\s+english|english\s+lo\s+cheppu|english\s+lo)\b",
    r"\b(change\s+(?:language\s+)?to\s+english|switch\s+to\s+english)\b",
    r"\b(from\s+now\s+on,?\s+(?:answer\s+in\s+)?english)\b",
]

TAMIL_PATTERNS = [
    r"\b(tamil\s+la\s+sollu|tamil\s+lo\s+cheppu|tamilil\s+solla|in\s+tamil|tamil\s+script)\b",
    r"\b(change\s+(?:language\s+)?to\s+tamil|switch\s+to\s+tamil|answer\s+in\s+tamil)\b",
]


def detect_language_command(text: str) -> tuple[Optional[str], bool]:
    """
    Detect language preference commands in user query before transcript retrieval.

    Returns:
        (detected_language_or_None, is_pure_command)
    """
    t_clean = text.strip()
    t_lower = t_clean.lower()

    detected_lang: Optional[str] = None

    # Priority 1: Telugu-English mix (must be checked before generic Telugu to avoid misclassification)
    if any(re.search(p, t_lower) for p in TEL_ENG_PATTERNS):
        detected_lang = ResponseLanguage.TELUGU_ENGLISH

    # Priority 2: Hindi (e.g. "naku hindi lo cheppu")
    elif any(re.search(p, t_lower) for p in HINDI_PATTERNS):
        detected_lang = ResponseLanguage.HINDI

    # Priority 3: Telugu script (e.g. "naku eppudu telugu lo cheppu")
    elif any(re.search(p, t_lower) for p in TELUGU_PATTERNS):
        detected_lang = ResponseLanguage.TELUGU

    # Priority 4: English
    elif any(re.search(p, t_lower) for p in ENGLISH_PATTERNS):
        detected_lang = ResponseLanguage.ENGLISH

    # Priority 5: Tamil
    elif any(re.search(p, t_lower) for p in TAMIL_PATTERNS):
        detected_lang = ResponseLanguage.TAMIL

    if not detected_lang:
        return (None, False)

    # Determine if it's purely a command without an accompanying meeting inquiry
    all_patterns = TEL_ENG_PATTERNS + HINDI_PATTERNS + TELUGU_PATTERNS + ENGLISH_PATTERNS + TAMIL_PATTERNS
    residual = t_lower
    for p in all_patterns:
        residual = re.sub(p, " ", residual)

    words = [
        w for w in re.findall(r"[a-zA-Z]+", residual)
        if w not in (
            "naku", "naaku", "eppudu", "ippudu", "please", "can", "you",
            "cheppu", "batao", "sollu", "to", "in", "from", "now", "on",
            "change", "language", "preference", "mix", "always",
        )
    ]

    meeting_terms = {
        "meeting", "technique", "techniques", "discuss", "discussed",
        "decision", "decisions", "summary", "summarize", "action", "tasks",
        "what", "who", "when", "why", "how", "fab", "bscq", "feature",
        "advantage", "benefit", "whisper", "sarvam", "chromadb", "database",
        "agenda", "details", "points",
    }

    has_question_content = any(w in meeting_terms for w in words) or len(words) >= 4
    is_pure_command = not has_question_content

    return (detected_lang, is_pure_command)


def get_language_acknowledgment(language: str) -> str:
    """Generate a natural acknowledgment confirming the language preference switch."""
    if language == ResponseLanguage.TELUGU:
        return "సరే! ఇకపై నేను మీటింగ్‌కు సంబంధించిన అన్ని వివరాలను తెలుగు లిపిలోనే సమాధానాలు ఇస్తాను. మీ ప్రశ్న అడగండి."
    elif language == ResponseLanguage.HINDI:
        return "ठीक है! अब से मैं मीटिंग के सभी उत्तर हिंदी (देवनागरी लिपि) में दूंगा। कृपया मीटिंग के बारे में अपना प्रश्न पूछें।"
    elif language == ResponseLanguage.TELUGU_ENGLISH:
        return "Sure! Ippati nunchi nenu responses anni natural Telugu-English mix (Roman Telugu) lo English technical terms tho isthanu. Meeting gurinchi mee question adagandi."
    elif language == ResponseLanguage.TAMIL:
        return "சரி! இனி நான் கூட்டத்தின் அனைத்து விவரங்களையும் தமிழிலேயே பதிலளிக்கிறேன். உங்கள் கேள்வியைக் கேளுங்கள்."
    else:  # ENGLISH
        return "Understood! From now on, I will answer all questions about the meeting in English. What would you like to know?"


def detect_response_language(
    question: str,
    conversation_history: Any = None,
    preferred_language: Optional[str] = None,
) -> str:
    """
    Determine the active response language:
    1. Check for explicit language commands in the current question.
    2. Check native Unicode scripts.
    3. Check persistent preferred_language.
    4. Contextual inheritance from history.
    5. Default to English.
    """
    cmd_lang, _ = detect_language_command(question)
    if cmd_lang:
        return cmd_lang

    # Native scripts in input
    if contains_telugu_script(question):
        return ResponseLanguage.TELUGU
    if contains_devanagari_script(question):
        return ResponseLanguage.HINDI
    if contains_tamil_script(question):
        return ResponseLanguage.TAMIL

    # Persistent preferred language
    if preferred_language and preferred_language in (
        ResponseLanguage.ENGLISH,
        ResponseLanguage.TELUGU,
        ResponseLanguage.TELUGU_ENGLISH,
        ResponseLanguage.HINDI,
        ResponseLanguage.TAMIL,
    ):
        return preferred_language

    # Heuristic for Roman Telugu in question
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    roman_telugu_markers = {"teesukunnaro", "theesukunnaru", "matladaru", "jarigindi", "gurinchi", "cheyyi"}
    if q_words.intersection(roman_telugu_markers):
        return ResponseLanguage.TELUGU_ENGLISH

    return ResponseLanguage.ENGLISH


# ============================================================================
# 2. SHORT FOLLOW-UP & CONTEXT RESOLUTION ("cheppu", "continue", "explain")
# ============================================================================

SHORT_FOLLOWUP_SET = {
    "cheppu", "cheppandi", "cheppava", "cheppu bro", "inka cheppu", "malli cheppu",
    "continue", "explain", "tell me", "k tell me", "ok tell me", "okay tell me",
    "tell me more", "explain this", "explain more", "details", "more details",
    "go on", "elaborate", "what else", "and then", "next", "batao", "bataiye",
    "aur batao", "samjhao", "sollu", "sollunga", "k", "ok", "okay", "yes",
}


def is_short_followup(question: str) -> bool:
    """Check if the user message is a short continuation command."""
    q_clean = re.sub(r"[^\w\s]", " ", question.lower()).strip()
    words = q_clean.split()
    if not words:
        return True
    if q_clean in SHORT_FOLLOWUP_SET:
        return True
    if len(words) == 1 and words[0] in ("cheppu", "continue", "explain", "details", "next", "batao", "sollu"):
        return True
    if len(words) <= 3 and any(w in ("cheppu", "tell", "explain", "continue", "more", "details", "batao", "sollu") for w in words):
        # Exclude questions that have interrogatives or distinct topics
        if not any(w in ("what", "who", "when", "why", "how", "em", "emi", "kya", "enna", "fab", "bscq") for w in words):
            return True
    return False


def resolve_followup_subject(parsed_history: list[dict[str, str]]) -> str:
    """
    Search backwards through conversation history to find the last substantive meeting question,
    skipping pure language command turns.
    """
    for turn in reversed(parsed_history):
        if turn["role"] == "user":
            content = turn["content"].strip()
            _, is_pure = detect_language_command(content)
            if not is_pure and not is_short_followup(content):
                # Clean prompt artifacts
                clean_subj = re.sub(r"(?i)\b(naku|naaku|cheppu|batao|sollu|lo|mein|la|in)\b", " ", content).strip()
                return clean_subj or content

    # If no previous user question found, check last assistant turn
    for turn in reversed(parsed_history):
        if turn["role"] == "assistant":
            content = turn["content"].strip()
            # If not an acknowledgment, take key terms
            if not any(ack_word in content.lower() for ack_word in ("సరే", "ठीक है", "sure", "understood", "சரி")):
                words = [w for w in re.findall(r"\w+", content) if len(w) > 4][:6]
                if words:
                    return " ".join(words)

    return "meeting summary discussion techniques decisions"


# ============================================================================
# 3. TRANSCRIPT RETRIEVAL (CHROMA & CLEAN TRANSCRIPT)
# ============================================================================

def _format_document_context(documents) -> str:
    parts = []
    for index, doc in enumerate(documents, start=1):
        timestamp = doc.metadata.get("timestamp", "unknown time")
        content = (doc.page_content or "").strip()
        if not content:
            continue
        parts.append(
            f"EXCERPT {index}\n"
            f"Timestamp: [{timestamp}]\n"
            f"Transcript: {content}"
        )
    return "\n\n".join(parts)


def retrieve_transcript_evidence(
    session_id: str,
    retrieval_query: str,
    clean_transcript: Optional[str] = None,
    k: int = settings.rag_top_k,
    is_followup: bool = False,
) -> tuple[str, list[str], int]:
    """Retrieve timestamped excerpts from ChromaDB and incorporate clean transcript."""
    retrieval_k = min(max(1, k) * RETRIEVAL_MULTIPLIER, MAX_RETRIEVAL_K)
    documents = []
    timestamps: list[str] = []

    try:
        retriever = get_retriever(session_id, k=retrieval_k)
        documents = retriever.invoke(retrieval_query)
    except Exception as exc:
        logger.warning("Vector store retrieval for session '%s' returned: %s", session_id, exc)
        documents = []

    seen_ts = set()
    for doc in documents:
        ts = doc.metadata.get("timestamp")
        if ts and ts not in seen_ts:
            seen_ts.add(ts)
            timestamps.append(ts)

    context_parts: list[str] = []
    if documents:
        context_parts.append("--- TIMESTAMPED TRANSCRIPT EXCERPTS ---")
        context_parts.append(_format_document_context(documents))

    if clean_transcript and clean_transcript.strip():
        ct_clean = clean_transcript.strip()
        if len(ct_clean) <= 15000:
            context_parts.append("\n--- FULL CLEAN TRANSCRIPT ---")
            context_parts.append(ct_clean)
        else:
            context_parts.append("\n--- CLEAN TRANSCRIPT OVERVIEW ---")
            context_parts.append(ct_clean[:8000] + "\n\n[...]\n\n" + ct_clean[-4000:])

    combined_context = "\n\n".join(context_parts).strip()
    return (combined_context, timestamps, len(documents))


# ============================================================================
# 4. DYNAMIC LANGUAGE-SPECIFIC PROMPTS & GENERATION
# ============================================================================

def _build_language_directive(language: str) -> str:
    """Build dynamic system instruction tailored to the stored language preference."""
    if language == ResponseLanguage.TELUGU:
        return """
CRITICAL LANGUAGE DIRECTIVE - TELUGU SCRIPT:
1. You MUST write your ENTIRE response in clear, natural Telugu script (తెలుగు లిపి).
2. All headings, bullet points, explanations, and follow-ups MUST be in Telugu script.
3. Standard technical and framework abbreviations (e.g. BSCQ, FAB, Feature, Advantage, Benefit, Whisper, ChromaDB, Sarvam AI) may be preserved in uppercase English or transliterated.
4. ABSOLUTE PROHIBITION: UNDER NO CIRCUMSTANCES OUTPUT TAMIL SCRIPT (தமிழ் / Unicode U+0B80-U+0BFF) OR TAMIL WORDS!
5. ABSOLUTE PROHIBITION: DO NOT write in Roman Telugu (English alphabet) when Telugu script is selected.
"""
    elif language == ResponseLanguage.TELUGU_ENGLISH:
        return """
CRITICAL LANGUAGE DIRECTIVE - TELUGU-ENGLISH MIX (ROMAN TELUGU / TANGLISH):
1. You MUST write your ENTIRE response in natural, conversational Roman Telugu (Telugu words spelled with English/Latin letters).
2. Mix natural English words for technical, business, and framework terms (e.g. BSCQ, FAB, Feature, Advantage, Benefit, architecture, decisions, database, Whisper, Sarvam AI, ChromaDB, tasks, assigned, integration, deadline).
3. Example conversational phrasing:
   - "Meeting lo main ga sales and technical techniques gurinchi discuss chesaru. FAB technique lo Feature, Advantage, Benefit untayi..."
4. ABSOLUTE PROHIBITION: UNDER NO CIRCUMSTANCES OUTPUT TAMIL SCRIPT (தமிழ் / Unicode U+0B80-U+0BFF) OR TAMIL WORDS!
5. ABSOLUTE PROHIBITION: DO NOT use Telugu script characters. Use purely the English alphabet (Latin letters).
"""
    elif language == ResponseLanguage.HINDI:
        return """
CRITICAL LANGUAGE DIRECTIVE - HINDI (DEVANAGARI SCRIPT):
1. You MUST write your ENTIRE response in natural Hindi using Devanagari script (हिंदी / देवनागरी लिपि).
2. All headings, bullet points, and explanations MUST be in Hindi.
3. Standard technical and framework abbreviations (e.g. BSCQ, FAB, Feature, Advantage, Benefit, Whisper, ChromaDB, Sarvam AI) may be preserved in uppercase English.
4. ABSOLUTE PROHIBITION: DO NOT output Tamil script, Telugu script, or Roman Telugu.
"""
    elif language == ResponseLanguage.TAMIL:
        return """
CRITICAL LANGUAGE DIRECTIVE - TAMIL SCRIPT:
1. You MUST write your ENTIRE response in clear, natural Tamil script (தமிழ்).
2. All headings, bullet points, and explanations MUST be in Tamil script.
3. Standard technical terms (BSCQ, FAB, Whisper, ChromaDB) may be preserved in uppercase English.
4. ABSOLUTE PROHIBITION: DO NOT output Telugu script or Hindi script.
"""
    else:  # ENGLISH
        return """
CRITICAL LANGUAGE DIRECTIVE - ENGLISH:
1. You MUST answer in clear, concise, professional English.
2. ABSOLUTE PROHIBITION: DO NOT output Indian scripts (Tamil, Telugu, Devanagari) or Roman Telugu.
"""


def _format_conversation_history_block(parsed_history: list[dict[str, str]]) -> str:
    if not parsed_history:
        return ""
    lines = ["CONVERSATION HISTORY (Recent turns):"]
    for turn in parsed_history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"][:400].replace("\n", " ")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def generate_grounded_answer(
    question: str,
    context: str,
    language: str,
    conversation_history: Any = None,
    is_followup: bool = False,
    followup_topic: str = "",
) -> str:
    """Generate evidence-grounded answer using Gemini with dynamic language directives."""
    parsed_history = _parse_conversation_history(conversation_history)
    history_block = _format_conversation_history_block(parsed_history)
    language_directive = _build_language_directive(language)

    followup_instruction = ""
    if is_followup:
        followup_instruction = f"""
NOTE ON FOLLOW-UP REQUEST:
- The user's query is a short continuation / follow-up ('{question}').
- The underlying topic being continued from previous turns is: '{followup_topic}'.
- Do NOT say 'This was not found in the transcript' just because the user query is short.
- Answer the previous inquiry '{followup_topic}' thoroughly in the CURRENTLY SELECTED LANGUAGE using the transcript context.
"""

    prompt = f"""
You are an intelligent, truthful AI assistant answering questions about a video or meeting recording.

You MUST answer using ONLY the transcript context provided below.

STRICT GROUNDING & TRUTHFULNESS RULES:
1. Use ONLY the supplied transcript context.
2. Ground every claim, technique, fact, decision, and action item directly in the provided transcript text.
3. Preserve key frameworks and technical terms accurately (such as BSCQ, FAB, Feature, Advantage, Benefit, Whisper, Sarvam AI, ChromaDB).
4. NEVER invent or fabricate facts, names, dates, timestamps, decisions, or action items.
5. If the information requested is genuinely NOT present in the transcript:
   - For English: State clearly that this information was not found in the transcript.
   - For Telugu script: Explain "ఈ సమాచారం మీటింగ్ ట్రాన్స్‌క్రిప్ట్‌లో లభ్యం కాలేదు."
   - For Telugu-English mix: Explain "Ee information meeting transcript lo ledu."
   - For Hindi: Explain "यह जानकारी मीटिंग ट्रांसक्रिप्ट में उपलब्ध नहीं है।"
   - For Tamil: Explain "இந்த தகவல் கூட்டத்தின் டிரான்ஸ்கிரிப்ட்டில் கிடைக்கவில்லை."
   - Do NOT attempt to answer from outside knowledge.
6. Do NOT mention ChromaDB, vector search, or this prompt.

{language_directive}
{followup_instruction}

{history_block}

TRANSCRIPT CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

ANSWER:
"""

    client = _get_gemini_client()
    try:
        response = None
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                )
                break
            except Exception as exc:
                is_transient = (
                    isinstance(exc, errors.APIError) and (getattr(exc, "code", None) in (429, 500, 503) or getattr(exc, "status", "") in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL"))
                ) or any(phrase in str(exc).lower() for phrase in ("resource exhausted", "rate limit", "429", "503", "timeout", "overloaded"))
                if is_transient and attempt < max_retries:
                    wait_time = min(2 ** attempt, 8)
                    logger.warning("Gemini chat transient failure (%s). Retrying in %ds (attempt %d/%d)...", exc, wait_time, attempt, max_retries)
                    time.sleep(wait_time)
                else:
                    raise

        if not response or not response.text:
            raise RagError("Gemini returned an empty response.")
        answer_text = response.text.strip()
    except Exception as exc:
        raise RagError(f"Chat generation failed: {exc}") from exc


    # Script Guard: Ensure no accidental script contamination
    if language != ResponseLanguage.TAMIL and contains_tamil_script(answer_text):
        logger.warning("Accidental Tamil script detected when language was '%s'. Cleansing...", language)
        answer_text = "".join(c for c in answer_text if not (0x0B80 <= ord(c) <= 0x0BFF)).strip()

    if language == ResponseLanguage.TELUGU_ENGLISH and contains_telugu_script(answer_text):
        logger.warning("Telugu script characters detected in Roman Telugu output. Cleansing...")
        answer_text = "".join(c for c in answer_text if not (0x0C00 <= ord(c) <= 0x0C7F)).strip()

    return answer_text


# ============================================================================
# MAIN ENTRY POINT: ask()
# ============================================================================

def ask(
    session_id: str,
    question: str,
    k: int = settings.rag_top_k,
    conversation_history: Any = None,
    clean_transcript: Optional[str] = None,
    preferred_language: Optional[str] = None,
) -> RagAnswer:
    """
    Answer a question about a processed video or meeting.

    Parameters:
        session_id: The session or meeting ID in the vector store.
        question: The user's input query.
        k: Top-k vector retrieval count.
        conversation_history: Optional list of past chat turns for follow-up resolution.
        clean_transcript: Optional full cleaned transcript string for direct grounding.
        preferred_language: Persistent user response language preference.

    Returns:
        RagAnswer containing the generated answer, referenced timestamps, chunk count, and active language.
    """
    if not question or not question.strip():
        raise RagError("Question cannot be empty.")

    q = question.strip()
    parsed_history = _parse_conversation_history(conversation_history)

    # 1. Check for Language Command BEFORE Retrieval
    cmd_lang, is_pure_command = detect_language_command(q)
    if cmd_lang:
        active_language = cmd_lang
        logger.info("Language command detected: '%s' -> language: '%s', is_pure=%s", q, active_language, is_pure_command)
        if is_pure_command:
            # Immediate acknowledgment without unnecessary transcript search
            ack_message = get_language_acknowledgment(active_language)
            return RagAnswer(
                answer=ack_message,
                timestamps=[],
                source_chunks=0,
                language=active_language,
            )
    else:
        # Determine language from persistent preference or question
        active_language = detect_response_language(
            question=q,
            conversation_history=parsed_history,
            preferred_language=preferred_language,
        )

    # 2. Check for Short Follow-Up ("cheppu", "continue", "explain", "k tell me")
    is_followup = is_short_followup(q) and bool(parsed_history)
    followup_topic = ""
    if is_followup:
        followup_topic = resolve_followup_subject(parsed_history)
        retrieval_query = f"{followup_topic} details explanation".strip()
        logger.info("Follow-up resolved: '%s' -> subject: '%s', retrieval query: '%s', lang: '%s'",
                    q, followup_topic, retrieval_query, active_language)
    else:
        # Normalize standalone query to English keywords for ChromaDB retrieval
        retrieval_query = re.sub(r"[^\w\s]", " ", q).strip()
        # Remove common language command words from retrieval query
        retrieval_query = re.sub(r"(?i)\b(naku|naaku|cheppu|batao|sollu|lo|mein|la|in|telugu|hindi|english|tamil|mix|script)\b", " ", retrieval_query).strip()
        if not retrieval_query:
            retrieval_query = "meeting summary discussion"

    # 3. Retrieve Transcript Evidence
    context, timestamps, chunk_count = retrieve_transcript_evidence(
        session_id=session_id,
        retrieval_query=retrieval_query,
        clean_transcript=clean_transcript,
        k=k,
        is_followup=is_followup,
    )

    if not context.strip():
        return RagAnswer(
            answer=NOT_FOUND_PHRASE,
            timestamps=[],
            source_chunks=0,
            language=active_language,
        )

    # 4. Generate Grounded Answer with Dynamic Language Prompt
    answer_text = generate_grounded_answer(
        question=q,
        context=context,
        language=active_language,
        conversation_history=parsed_history,
        is_followup=is_followup,
        followup_topic=followup_topic,
    )

    if NOT_FOUND_PHRASE.lower() in answer_text.lower():
        timestamps = []

    return RagAnswer(
        answer=answer_text,
        timestamps=timestamps,
        source_chunks=chunk_count,
        language=active_language,
    )