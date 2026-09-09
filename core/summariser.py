"""
Structured summarization of a clean transcript.

Produces a title, overview, summary, important points and key takeaways
as a validated Pydantic object using Google Gemini structured output.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import get_logger

logger = get_logger(__name__)

# Guard against extremely long transcripts.
_SINGLE_PASS_CHAR_LIMIT = 60_000


class MeetingSummary(BaseModel):
    """Structured summary of a video/meeting transcript."""

    title: str = Field(
        description="A concise, descriptive title for this video/meeting (max 12 words)."
    )

    overview: str = Field(
        description="One or two sentence high-level overview of what was discussed."
    )

    summary: str = Field(
        description="A well-organized paragraph-form summary of the full content."
    )

    important_points: list[str] = Field(
        default_factory=list,
        description="The most important individual points raised, as short bullet strings.",
    )

    key_takeaways: list[str] = Field(
        default_factory=list,
        description="The main conclusions or lessons a listener should walk away with.",
    )


class SummarizationError(Exception):
    pass


_SYSTEM_PROMPT = """
You are an expert meeting and video summarizer.

Read the transcript carefully and summarize ONLY the information
that is explicitly present in the transcript.

Return the answer according to the provided structured schema.

Rules:
- title: maximum 12 words.
- overview: 1-2 sentences.
- summary: clear paragraph covering the main content.
- important_points: important points from the transcript.
- key_takeaways: useful conclusions or lessons.
- Never invent names, numbers, facts, events or conclusions.
- If there are no important points, return an empty list.
- If there are no key takeaways, return an empty list.
"""


def _get_gemini_client():
    """Create the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise SummarizationError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    return genai.Client(api_key=api_key)


def summarize_transcript(transcript: str) -> MeetingSummary:
    """Generate a structured summary for a clean English transcript."""

    if not transcript.strip():
        raise SummarizationError("Cannot summarize an empty transcript.")

    text = transcript

    if len(text) > _SINGLE_PASS_CHAR_LIMIT:
        logger.warning(
            "Transcript is %d chars, truncating to %d for summarization.",
            len(text),
            _SINGLE_PASS_CHAR_LIMIT,
        )
        text = text[:_SINGLE_PASS_CHAR_LIMIT]

    client = _get_gemini_client()

    prompt = f"""
{_SYSTEM_PROMPT}

Transcript:

{text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=MeetingSummary,
            ),
        )

        if not response.text:
            raise SummarizationError(
                "Gemini returned an empty response."
            )

        result = MeetingSummary.model_validate_json(response.text)

    except SummarizationError:
        raise

    except Exception as exc:
        raise SummarizationError(
            f"Gemini summarization failed: {exc}"
        ) from exc

    if not isinstance(result, MeetingSummary):
        raise SummarizationError(
            "Gemini did not return a valid structured summary."
        )

    return result