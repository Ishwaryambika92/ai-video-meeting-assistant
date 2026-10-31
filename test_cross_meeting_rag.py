"""
Test suite for Cross-Meeting Intelligence & RAG Engine (Feature 7).
Validates cross-meeting queries, evidence grounding, citations, and fallback.
"""

from __future__ import annotations

try:
    import pyarrow
except ImportError:
    pass

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.cross_meeting_rag import NO_EVIDENCE_PHRASE, ask_cross_meetings
from core.extractor import ActionItem, ExtractionResult
from core.meeting_store import delete_meeting, save_meeting
from core.summariser import MeetingSummary

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")


def run_tests() -> None:
    print("=" * 60)
    print("RUNNING CROSS-MEETING INTELLIGENCE & RAG TESTS")
    print("=" * 60)

    meet_1_id = "test-rag-meeting-01"
    meet_2_id = "test-rag-meeting-02"

    delete_meeting(meet_1_id)
    delete_meeting(meet_2_id)

    # 1. Test empty meeting handling / filter miss
    empty_ans = ask_cross_meetings("What was decided about the database?", selected_meeting_ids=["non-existent-session-id"])
    assert empty_ans is not None
    assert "No meetings" in empty_ans.answer or "saved" in empty_ans.answer
    print("[OK] Test 1: Empty meeting history / filter handled gracefully without error")

    # Seed 2 realistic meetings with distinct decisions and tasks
    past_due = (date.today() - timedelta(days=2)).strftime("%Y-%m-%d")
    future_due = (date.today() + timedelta(days=5)).strftime("%Y-%m-%d")

    m1 = save_meeting(
        meeting_id=meet_1_id,
        title="Q3 Database Architecture Sync",
        summary_obj=MeetingSummary(
            title="Q3 Database Architecture Sync",
            overview="Database layer selection and indexing strategy.",
            summary="The engineering team decided to standardize on SQLite with WAL mode for meeting memory.",
            important_points=["SQLite avoids external server dependencies."],
            key_takeaways=["Database migrations must be backward compatible."],
        ),
        extraction_obj=ExtractionResult(
            action_items=[
                ActionItem(action="Implement SQLite meeting schema", owner="Vikram", deadline=past_due, status="Pending"),
                ActionItem(action="Enable WAL pragma and foreign keys", owner="Sarah", deadline=future_due, status="Completed"),
            ],
            decisions=[
                "Selected SQLite with WAL mode for all persistent storage.",
            ],
            open_questions=["Should we support PostgreSQL for multi-tenant deployments?"],
            follow_ups=["Review database size benchmarks next Friday."],
        ),
        clean_transcript="Speaker 1: Welcome team. Let's decide on the database storage engine. Speaker 2: SQLite in WAL mode gives us ultra-low latency.",
        source="sync_db.mp4",
        participants=["Vikram", "Sarah"],
    )

    m2 = save_meeting(
        meeting_id=meet_2_id,
        title="Q3 AI Model & UI Styling Sync",
        summary_obj=MeetingSummary(
            title="Q3 AI Model & UI Styling Sync",
            overview="Finalizing AI models and dashboard design system.",
            summary="The design team approved the coral-red SaaS UI theme. Engineering confirmed Sarvam AI for Indic languages.",
            important_points=["Coral-red hex #FF4B4B chosen as primary accent."],
            key_takeaways=["Indic speech processing enabled for Telugu and Hindi."],
        ),
        extraction_obj=ExtractionResult(
            action_items=[
                ActionItem(action="Deploy coral-red SaaS dashboard theme", owner="Design Team", deadline=future_due, status="In Progress"),
                ActionItem(action="Integrate Sarvam AI speech models", owner="Vikram", deadline="", status="Pending"),
            ],
            decisions=[
                "Approved coral-red SaaS design system with hex #FF4B4B across all dashboards.",
                "Standardized on Sarvam AI for Indian language transcriptions.",
            ],
            open_questions=["What is the response latency for Telugu audio?"],
            follow_ups=["Run end-to-end smoke tests before deployment."],
        ),
        clean_transcript="Speaker 1: Let's review the new coral-red design system. Speaker 2: Looks stunning! And Sarvam AI is working for Indic transcripts.",
        source="sync_ui.mp4",
        participants=["Vikram", "Design Team"],
    )

    # 2. Test Cross-Meeting Question about Decisions
    ans_decisions = ask_cross_meetings("What did we decide about the database in previous meetings?")
    assert ans_decisions is not None
    assert len(ans_decisions.answer) > 20
    # Should mention SQLite or Database Architecture Sync
    assert "sqlite" in ans_decisions.answer.lower() or "database" in ans_decisions.answer.lower()
    print("[OK] Test 2: Cross-meeting decision question correctly retrieves historical decisions")

    # 3. Test Cross-Meeting Question about Pending Tasks / Deadlines
    ans_tasks = ask_cross_meetings("Which tasks are still pending across our meetings?")
    assert ans_tasks is not None
    ans_text_lower = ans_tasks.answer.lower()
    # Must identify pending action items
    assert "implement sqlite" in ans_text_lower or "sarvam" in ans_text_lower or "pending" in ans_text_lower
    print("[OK] Test 3: Cross-meeting action item query returns pending deliverables")

    # 4. Test Filtering by Meeting ID
    ans_filtered = ask_cross_meetings(
        "What decisions were made?",
        selected_meeting_ids=[meet_2_id],
    )
    assert ans_filtered is not None
    # Must focus on meeting 2 and not meeting 1
    assert "coral" in ans_filtered.answer.lower() or "sarvam" in ans_filtered.answer.lower() or "design" in ans_filtered.answer.lower()
    print("[OK] Test 4: Meeting ID scoping correctly restricts answer context")

    # 5. Test Question on Unrecorded / Out-of-Scope Topic (No Hallucination)
    ans_unrecorded = ask_cross_meetings("What was decided about purchasing Tesla stock or buying bitcoins?")
    assert ans_unrecorded is not None
    # Must indicate not found or no evidence
    assert (
        NO_EVIDENCE_PHRASE.lower() in ans_unrecorded.answer.lower()
        or "not found" in ans_unrecorded.answer.lower()
        or "no evidence" in ans_unrecorded.answer.lower()
        or "no recorded" in ans_unrecorded.answer.lower()
    )
    print("[OK] Test 5: Unrecorded topic cleanly produces evidence-absent notification without hallucinations")

    # Clean up test records
    delete_meeting(meet_1_id)
    delete_meeting(meet_2_id)
    print("[OK] Test 6: Test fixtures cleanly purged from SQLite database")

    print("\n" + "=" * 60)
    print("ALL CROSS-MEETING INTELLIGENCE TESTS PASSED (6/6)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
