"""
Structured extraction of action items, decisions, open questions and
follow-ups from a clean transcript.

Owners/deadlines are NEVER invented: if the transcript doesn't state them,
the field is set to the literal string "Not specified".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field
import time

from config import get_logger, settings

logger = get_logger(__name__)


NOT_SPECIFIED = "Not specified"


class ActionItem(BaseModel):
    action: str = Field(
        description="The concrete task or action to be done."
    )

    owner: str = Field(
        default=NOT_SPECIFIED,
        description=f"Who is responsible. '{NOT_SPECIFIED}' if not mentioned.",
    )

    deadline: str = Field(
        default=NOT_SPECIFIED,
        description=f"When it is due. '{NOT_SPECIFIED}' if not mentioned.",
    )

    status: str = Field(
        default=NOT_SPECIFIED,
        description=f"Current status if mentioned, else '{NOT_SPECIFIED}'.",
    )


class ExtractionResult(BaseModel):
    action_items: list[ActionItem] = Field(
        default_factory=list
    )

    decisions: list[str] = Field(
        default_factory=list,
        description="Concrete decisions that were made.",
    )

    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions raised but not resolved.",
    )

    follow_ups: list[str] = Field(
        default_factory=list,
        description="Things explicitly flagged for later follow-up.",
    )


class ExtractionError(Exception):
    pass


_SYSTEM_PROMPT = f"""
You are an expert meeting/video transcript analyst.

Extract structured information from the transcript into:

1. Action items
2. Decisions
3. Open questions
4. Follow-ups

ACTION ITEMS:
Identify concrete work that needs to be done or was committed to.

Include:
- Explicitly assigned tasks
- Personal commitments such as "I will..."
- Team commitments such as "we need to..."
- Requests such as "please complete..."
- Required work such as "we have to..."
- Tasks with or without a named owner
- Tasks with or without a deadline

Do NOT invent information.

If the transcript clearly states that a task needs to be done but
does not state who is responsible, use exactly:

"{NOT_SPECIFIED}"

If a deadline is not mentioned, use exactly:

"{NOT_SPECIFIED}"

If status is not mentioned, use exactly:

"{NOT_SPECIFIED}"

IMPORTANT:

A statement such as:
"I will prepare the report"

is an action item.

A statement such as:
"We need to finish the backend"

is an action item.

A statement such as:
"Please test the application"

is an action item.

Do not convert general discussion or opinions into action items.

DECISIONS:
Extract concrete decisions that were actually made.
Do not invent decisions.

OPEN QUESTIONS:
Extract questions that were raised but not resolved.
Do not create questions that were not present.

FOLLOW-UPS:
Extract things explicitly mentioned for later discussion,
review, checking, or completion.

GENERAL RULE:
Use ONLY information contained in the transcript.

Never invent names, owners, deadlines, status, decisions,
questions, or follow-ups.

If a category has no valid items, return an empty list.
"""


def _get_gemini_client():
    """Create the Gemini client."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ExtractionError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    return genai.Client(api_key=api_key)


def extract_action_items(transcript: str) -> ExtractionResult:
    """Extract action items, decisions, open questions and follow-ups."""

    if not transcript.strip():
        raise ExtractionError(
            "Cannot extract from an empty transcript."
        )

    client = _get_gemini_client()

    prompt = f"""
{_SYSTEM_PROMPT}

Transcript:

{transcript}
"""

    try:
        response = None
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractionResult,
                    ),
                )
                break
            except Exception as exc:
                is_transient = (
                    isinstance(exc, errors.APIError) and (getattr(exc, "code", None) in (429, 500, 503) or getattr(exc, "status", "") in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL"))
                ) or any(phrase in str(exc).lower() for phrase in ("resource exhausted", "rate limit", "429", "503", "timeout", "overloaded"))
                if is_transient and attempt < max_retries:
                    wait_time = min(2 ** attempt, 8)
                    logger.warning("Gemini extraction transient failure (%s). Retrying in %ds (attempt %d/%d)...", exc, wait_time, attempt, max_retries)
                    time.sleep(wait_time)
                else:
                    raise

        if not response or not response.text:
            raise ExtractionError(
                "Gemini returned an empty response."
            )

        result = ExtractionResult.model_validate_json(
            response.text
        )

    except ExtractionError:
        raise

    except Exception as exc:
        raise ExtractionError(
            f"Gemini extraction failed: {exc}"
        ) from exc


    if not isinstance(result, ExtractionResult):
        raise ExtractionError(
            "Gemini did not return valid structured extraction data."
        )

    # Normalize blank values.
    for item in result.action_items:
        item.owner = item.owner.strip() or NOT_SPECIFIED
        item.deadline = item.deadline.strip() or NOT_SPECIFIED
        item.status = item.status.strip() or NOT_SPECIFIED

    return result