"""
Cross-Meeting Memory and Intelligence Engine.

Enables queries across multiple previously processed meetings by combining
authoritative SQLite records (decisions, summaries, real-time action item statuses)
with ChromaDB semantic transcript retrieval and Gemini evidence-grounded generation.
"""

from __future__ import annotations

try:
    import pyarrow
except ImportError:
    pass

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from google import genai
from google.genai import errors
import time

from config import get_logger, settings

from core.meeting_store import (
    MeetingRecord,
    TaskRecord,
    get_action_items,
    get_all_decisions,
    get_all_meetings,
    get_meeting_by_id,
    parse_iso_date,
)
from core.rag_engine import (
    ResponseLanguage,
    _build_language_directive,
    _parse_conversation_history,
    contains_tamil_script,
    detect_language_command,
    detect_response_language,
    get_language_acknowledgment,
)
from core.vector_store import get_retriever

logger = get_logger(__name__)

NO_EVIDENCE_PHRASE = "The available meeting records do not contain enough evidence to answer this question."



@dataclass
class CrossMeetingSource:
    meeting_id: str
    meeting_title: str
    meeting_date: str
    source_type: str  # "decision", "action_item", "transcript", "summary"
    content: str
    timestamp: str = ""


@dataclass
class CrossMeetingAnswer:
    answer: str
    sources: list[CrossMeetingSource] = field(default_factory=list)
    meetings_referenced: list[str] = field(default_factory=list)
    language: str = ResponseLanguage.ENGLISH


def _get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=api_key, http_options={"timeout": 60000})


def _format_meeting_date(iso_str: str) -> str:
    """Format ISO timestamp into human-readable date string."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %Y (%I:%M %p)")
    except Exception:
        return iso_str[:16] if iso_str else "Unknown Date"


def ask_cross_meetings(
    question: str,
    selected_meeting_ids: list[str] | None = None,
    filter_participant: str | None = None,
    filter_date_start: str | None = None,
    filter_date_end: str | None = None,
    conversation_history: list | None = None,
    preferred_language: str | None = None,
    user_id: str | None = None,
) -> CrossMeetingAnswer:

    """
    Retrieve and synthesize information across multiple past meetings.

    Combines:
    1. Authoritative SQLite meeting summaries, decisions, and participants.
    2. Authoritative real-time action items (status, assignee, deadlines, priority).
    3. ChromaDB vector semantic transcript chunks with timestamps.
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        return CrossMeetingAnswer(
            answer="Please enter a question to search across past meetings.",
            sources=[],
            meetings_referenced=[],
            language=preferred_language or ResponseLanguage.ENGLISH,
        )

    # Check for pure language preference commands
    cmd_lang, is_pure_command = detect_language_command(cleaned_question)
    if cmd_lang:
        active_lang = cmd_lang
        if is_pure_command:
            return CrossMeetingAnswer(
                answer=get_language_acknowledgment(active_lang),
                sources=[],
                meetings_referenced=[],
                language=active_lang,
            )
    else:
        parsed_history = _parse_conversation_history(conversation_history)
        active_lang = detect_response_language(
            question=cleaned_question,
            conversation_history=parsed_history,
            preferred_language=preferred_language,
        )

    # 1. Fetch available meetings from SQLite (scoped by user_id)
    all_meetings = get_all_meetings(limit=100, user_id=user_id)

    if not all_meetings:
        return CrossMeetingAnswer(
            answer="No meetings have been saved to memory yet. Once you process or load meetings, cross-meeting memory will be available.",
            sources=[],
            meetings_referenced=[],
        )

    # 2. Filter candidate meetings
    candidate_meetings = all_meetings
    if selected_meeting_ids:
        candidate_meetings = [m for m in candidate_meetings if m.id in selected_meeting_ids]

    if filter_participant and filter_participant.strip() and filter_participant != "All":
        p_norm = filter_participant.strip().lower()
        candidate_meetings = [
            m for m in candidate_meetings
            if any(p_norm in p.lower() for p in m.participants)
        ]

    # Date filters
    start_d = parse_iso_date(filter_date_start)
    end_d = parse_iso_date(filter_date_end)
    if start_d:
        candidate_meetings = [
            m for m in candidate_meetings
            if (parse_iso_date(m.created_at[:10]) or date.min) >= start_d
        ]
    if end_d:
        candidate_meetings = [
            m for m in candidate_meetings
            if (parse_iso_date(m.created_at[:10]) or date.max) <= end_d
        ]

    if not candidate_meetings:
        return CrossMeetingAnswer(
            answer="No meetings matched the specified filters (meeting, participant, or date range).",
            sources=[],
            meetings_referenced=[],
        )

    candidate_meeting_ids = {m.id for m in candidate_meetings}

    # 3. Gather authoritative action items from database (scoped by user_id)
    all_tasks: list[TaskRecord] = []
    for m in candidate_meetings:
        all_tasks.extend(get_action_items(meeting_id=m.id, user_id=user_id))


    # 4. Gather transcript excerpts from ChromaDB where available
    sources: list[CrossMeetingSource] = []
    transcript_excerpts_by_meeting: dict[str, list[str]] = {}

    for m in candidate_meetings:
        # Check ChromaDB for transcript chunks
        try:
            retriever = get_retriever(m.id, k=3)
            docs = retriever.invoke(cleaned_question)
            excerpts = []
            for doc in docs:
                ts = doc.metadata.get("timestamp", "")
                content = doc.page_content.strip()
                excerpts.append(f"[{ts}] {content}" if ts else content)
                sources.append(
                    CrossMeetingSource(
                        meeting_id=m.id,
                        meeting_title=m.title,
                        meeting_date=_format_meeting_date(m.created_at),
                        source_type="transcript",
                        content=content,
                        timestamp=ts,
                    )
                )
            if excerpts:
                transcript_excerpts_by_meeting[m.id] = excerpts
        except Exception:
            # Fall back to clean transcript snippet if vector store is unavailable
            if m.clean_transcript:
                clean_lines = [l.strip() for l in m.clean_transcript.splitlines() if l.strip()]
                # Search for keyword matches
                kw_matches = [
                    line for line in clean_lines
                    if any(w.lower() in line.lower() for w in cleaned_question.split() if len(w) > 3)
                ][:3]
                if kw_matches:
                    transcript_excerpts_by_meeting[m.id] = kw_matches
                    for line in kw_matches:
                        sources.append(
                            CrossMeetingSource(
                                meeting_id=m.id,
                                meeting_title=m.title,
                                meeting_date=_format_meeting_date(m.created_at),
                                source_type="transcript",
                                content=line,
                                timestamp="",
                            )
                        )

        # Also collect authoritative decisions matching the query into sources
        q_lower_terms = [w.lower() for w in re.findall(r"\w+", cleaned_question) if len(w) > 3]
        is_dec_q = any(k in cleaned_question.lower() for k in ("decis", "decide", "agreed", "launch", "design", "cache", "option"))
        for d in (m.decisions or []):
            if not d or not d.strip() or d.strip().lower() in ("none found.", "none."):
                continue
            d_str = d.strip()
            matches_d = any(qt in d_str.lower() for qt in q_lower_terms) if q_lower_terms else True
            if is_dec_q and (matches_d or "all" in cleaned_question.lower() or "across" in cleaned_question.lower()):
                sources.append(
                    CrossMeetingSource(
                        meeting_id=m.id,
                        meeting_title=m.title,
                        meeting_date=_format_meeting_date(m.created_at),
                        source_type="decision",
                        content=d_str,
                        timestamp="",
                    )
                )

    # Deduplicate sources by (meeting_id, content)
    dedup_sources: list[CrossMeetingSource] = []
    seen_src: set[tuple[str, str]] = set()
    for s in sources:
        key = (s.meeting_id, s.content.strip())
        if key not in seen_src:
            seen_src.add(key)
            dedup_sources.append(s)
    sources = dedup_sources

    # 5. Assemble structured meeting records context
    meeting_context_blocks: list[str] = []
    for m in candidate_meetings:
        date_str = _format_meeting_date(m.created_at)
        participants_str = ", ".join(m.participants) if m.participants else "Not specified"
        decisions_str = "\n".join(f"- {d}" for d in m.decisions) if m.decisions else "None recorded."
        takeaways_str = "\n".join(f"- {t}" for t in m.key_takeaways) if m.key_takeaways else "None."

        # Filter tasks for this meeting
        m_tasks = [t for t in all_tasks if t.meeting_id == m.id]
        if m_tasks:
            tasks_lines = []
            for t in m_tasks:
                due_info = f"Due: {t.due_date}" if t.due_date else "No due date"
                overdue_info = " [OVERDUE]" if t.is_overdue else ""
                tasks_lines.append(
                    f"- [{t.status}{overdue_info}] {t.description} (Assignee: {t.assignee}, Priority: {t.priority}, {due_info})"
                )
            tasks_str = "\n".join(tasks_lines)
        else:
            tasks_str = "No action items recorded."

        transcript_snippets = transcript_excerpts_by_meeting.get(m.id, [])
        transcript_str = "\n".join(f"  {s}" for s in transcript_snippets) if transcript_snippets else "No matching transcript excerpts."

        block = f"""
=== MEETING RECORD ===
Meeting ID: {m.id}
Title: {m.title}
Date/Time: {date_str}
Participants: {participants_str}
Overview: {m.overview or m.summary or 'None provided.'}
Decisions Made:
{decisions_str}
Action Items (Authoritative Current Status):
{tasks_str}
Key Takeaways:
{takeaways_str}
Relevant Transcript Excerpts:
{transcript_str}
======================
"""
        meeting_context_blocks.append(block.strip())

    full_context = "\n\n".join(meeting_context_blocks)

    # 6. Formulate evidence-grounded prompt for Gemini
    language_directive = _build_language_directive(active_lang)

    prompt = f"""
You are an intelligent executive meeting assistant answering questions across historical meetings.

You have access to authoritative meeting records and transcript excerpts from {len(candidate_meetings)} meeting(s).

STRICT RULES:
1. Answer using ONLY the supplied meeting records and transcript excerpts below.
2. Ground every claim in the provided evidence. Cite the specific meeting title and date for each fact.
3. For task statuses, deadlines, and assignees, treat the "Action Items (Authoritative Current Status)" from the database as 100% authoritative. Never contradict the database status (Pending, In Progress, Completed).
4. If a task is marked [OVERDUE], acknowledge it as overdue.
5. If the records do NOT contain enough information or evidence to answer the question, state clearly:
   "{NO_EVIDENCE_PHRASE}"
   Explain what records were checked and what was missing.
6. NEVER invent dates, participants, decisions, or action items that do not appear in the context.
7. Format your response cleanly using markdown (bullet points, bold meeting titles, concise tables if helpful).

{language_directive}

HISTORICAL MEETING RECORDS:
{full_context}

USER QUESTION:
{cleaned_question}

ANSWER:
"""

    # 7. Call Gemini model with exponential backoff retry for transient errors
    try:
        client = _get_gemini_client()
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
                    logger.warning("Gemini cross-meeting transient failure (%s). Retrying in %ds (attempt %d/%d)...", exc, wait_time, attempt, max_retries)
                    time.sleep(wait_time)
                else:
                    raise

        answer_text = response.text.strip() if (response and response.text) else NO_EVIDENCE_PHRASE

        # Tamil script guard
        if active_lang != ResponseLanguage.TAMIL and contains_tamil_script(answer_text):
            answer_text = "".join(c for c in answer_text if not (0x0B80 <= ord(c) <= 0x0BFF)).strip()

    except Exception as exc:
        logger.error("Error in cross-meeting Gemini call: %s", exc)
        # Fallback to direct structured answering if API fails or is offline
        answer_text = _fallback_structured_answer(cleaned_question, candidate_meetings, all_tasks)

    # Find referenced meetings in the answer text
    referenced_meetings: list[str] = []
    for m in candidate_meetings:
        if m.title.lower() in answer_text.lower() or m.id in answer_text:
            referenced_meetings.append(m.title)

    if not referenced_meetings:
        referenced_meetings = [m.title for m in candidate_meetings[:3]]

    return CrossMeetingAnswer(
        answer=answer_text,
        sources=sources,
        meetings_referenced=referenced_meetings,
        language=active_lang,
    )


def _fallback_structured_answer(
    question: str,
    meetings: list[MeetingRecord],
    tasks: list[TaskRecord],
) -> str:
    """Offline / fallback answer generator for common query intents."""
    q_lower = question.lower()

    # Intent 1: Pending or overdue tasks
    if any(k in q_lower for k in ("pending", "task", "action item", "todo", "overdue")):
        pending_tasks = [t for t in tasks if t.status in ("Pending", "In Progress")]
        if not pending_tasks:
            return "There are currently no pending action items across the selected meetings."

        lines = ["### Pending Action Items Across Meetings:\n"]
        for t in pending_tasks:
            overdue_tag = " ⚠️ **OVERDUE**" if t.is_overdue else ""
            due_str = f"Due: {t.due_date}" if t.due_date else "No deadline"
            lines.append(
                f"- **{t.meeting_title}**: {t.description} — *Assignee:* {t.assignee} | *Status:* {t.status}{overdue_tag} | *{due_str}*"
            )
        return "\n".join(lines)

    # Intent 2: Decisions
    if any(k in q_lower for k in ("decid", "decision", "agree", "agreed", "choice", "chose")):
        decisions_found = []
        for m in meetings:
            for d in m.decisions:
                if d and d.strip() and d.lower() != "none found.":
                    decisions_found.append((m.title, _format_meeting_date(m.created_at), d.strip()))

        if not decisions_found:
            return "No recorded decisions found across the selected meetings."

        stop_words = {
            "what", "were", "about", "which", "decide", "decided", "decision", "decisions",
            "meetings", "meeting", "previous", "across", "all", "made", "summary", "summarize",
            "list", "show", "every", "have", "been", "recorded", "from", "past", "history", "our"
        }
        query_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 2 and w not in stop_words]
        if query_words:
            matched_decisions = [
                item for item in decisions_found
                if any(qw in item[0].lower() or qw in item[2].lower() for qw in query_words)
            ]
            if matched_decisions:
                decisions_found = matched_decisions
            else:
                return f"{NO_EVIDENCE_PHRASE}\n\nNo recorded decisions found matching '{question}' across historical meetings."

        lines = ["### Recorded Decisions Across Meetings:\n"]
        for m_title, m_date, dec in decisions_found:
            lines.append(f"- **{m_title}** ({m_date}): {dec}")
        return "\n".join(lines)

    # Intent 3: Deadlines
    if any(k in q_lower for k in ("deadline", "due date", "due")):
        dated_tasks = [t for t in tasks if t.due_date]
        if not dated_tasks:
            return "No deadlines are currently recorded for action items in the selected meetings."

        lines = ["### Deadlines Discussed Across Meetings:\n"]
        for t in dated_tasks:
            overdue_tag = " ⚠️ **OVERDUE**" if t.is_overdue else ""
            lines.append(
                f"- **{t.due_date}** ({t.meeting_title}): {t.description} (Assigned to {t.assignee}, Status: {t.status}{overdue_tag})"
            )
        return "\n".join(lines)

    # Default fallback
    return f"{NO_EVIDENCE_PHRASE}\n\nPlease check that your Gemini API key is valid, or refine your query."
