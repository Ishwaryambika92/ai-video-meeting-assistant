"""
Test suite specifically validating the Decisions Repository and Decision Retrieval logic:
- Multiple decisions from the same meeting
- Decisions from different meetings
- Empty repository and filter misses
- Missing meeting metadata resiliency
- Duplicate records & repeated processing deduplication
- Normalization (bullet markers, numbered lists, hyphens)
- Meeting and participant scoping filters
- Phantom decision filtering ("None found.", "No decisions made.")
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.extractor import ActionItem, ExtractionResult
from core.meeting_store import (
    delete_meeting,
    get_all_decisions,
    normalize_decision_text,
    save_meeting,
)
from core.summariser import MeetingSummary

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")


def run_tests() -> None:
    print("=" * 60)
    print("RUNNING DECISIONS REPOSITORY & RETRIEVAL VALIDATION TESTS")
    print("=" * 60)

    # 1. Normalization tests (stripping bullets, numbers, hyphens)
    assert normalize_decision_text("1. Proceed with Option B for the product launch.") == "Proceed with Option B for the product launch."
    assert normalize_decision_text("- Approved coral-red SaaS design system.") == "Approved coral-red SaaS design system."
    assert normalize_decision_text("• Continue monitoring the supply chain issue.") == "Continue monitoring the supply chain issue."
    assert normalize_decision_text("* Retain local embedding caches.") == "Retain local embedding caches."
    assert normalize_decision_text("2) Standardize on SQLite in WAL mode.") == "Standardize on SQLite in WAL mode."
    assert normalize_decision_text("(3) Deploy Sarvam AI speech models.") == "Deploy Sarvam AI speech models."
    assert normalize_decision_text("Clean decision without prefix.") == "Clean decision without prefix."
    print("[OK] Test 1: Normalization correctly strips leading bullets, numbers, hyphens, and whitespace")

    # Fixture meeting IDs
    m_alpha = "test-dec-meet-alpha"
    m_beta = "test-dec-meet-beta"
    m_empty = "test-dec-meet-empty"
    m_sparse = "test-dec-meet-sparse"

    delete_meeting(m_alpha)
    delete_meeting(m_beta)
    delete_meeting(m_empty)
    delete_meeting(m_sparse)

    try:
        # 2. Test Empty Repository / Filter Miss
        empty_res = get_all_decisions(meeting_id="non-existent-meeting-id")
        assert len(empty_res) == 0
        print("[OK] Test 2: Non-existent meeting returns clean empty decision list without error")

        # Seed meeting Alpha with multiple decisions (including duplicates and bullet points)
        save_meeting(
            meeting_id=m_alpha,
            title="Sprint Alpha Product Review",
            summary_obj=MeetingSummary(
                title="Sprint Alpha Product Review",
                overview="Discussion on product launch options and team roles.",
                summary="The committee selected Option B for rollout and assigned project roles.",
                important_points=["Option B minimizes customer migration disruption."],
                key_takeaways=["Rollout commences on Monday."],
            ),
            extraction_obj=ExtractionResult(
                action_items=[
                    ActionItem(action="Finalize launch plan", owner="Dev Lead", deadline="Tomorrow", status="In Progress"),
                ],
                decisions=[
                    "1. Proceed with Option B for the product launch.",
                    "- Assign roles for the upcoming project.",
                    "• Continue monitoring the supply chain issue.",
                    "Proceed with Option B for the product launch.",  # Duplicate with identical normalized text!
                    "None found.",  # Phantom decision to be ignored!
                ],
                open_questions=[],
                follow_ups=[],
            ),
            clean_transcript="Speaker 1: We will proceed with Option B. Speaker 2: Agreed.",
            source="alpha.mp4",
            participants=["Vikram_DecTest", "Sarah_DecTest"],
        )

        # Seed meeting Beta with decisions from a distinct meeting
        save_meeting(
            meeting_id=m_beta,
            title="Design System & Architecture Alignment",
            summary_obj=MeetingSummary(
                title="Design System & Architecture Alignment",
                overview="Approval of coral-red UI system and embedding cache policies.",
                summary="UI design approved and local caching retained.",
                important_points=["Coral-red provides consistent branding."],
                key_takeaways=["Embedding caches reduce API costs."],
            ),
            extraction_obj=ExtractionResult(
                action_items=[
                    ActionItem(action="Update CSS variables", owner="Frontend Lead", deadline="Friday", status="Pending"),
                ],
                decisions=[
                    "Approved coral-red SaaS design system across all client dashboards.",
                    "Retain local embedding caches to reduce cold-start latency.",
                    "No decisions made.",  # Phantom decision to be ignored!
                ],
                open_questions=[],
                follow_ups=[],
            ),
            clean_transcript="Speaker 1: Coral-red theme is approved. Speaker 2: We will retain local embedding caches.",
            source="beta.mp4",
            participants=["Vikram_DecTest", "Elena_DecTest"],
        )

        # Seed meeting with NO decisions
        save_meeting(
            meeting_id=m_empty,
            title="Casual Standup No Decisions",
            summary_obj=MeetingSummary(
                title="Casual Standup No Decisions",
                overview="General standup updates without any decisions.",
                summary="No decisions were taken.",
                important_points=[],
                key_takeaways=[],
            ),
            extraction_obj=ExtractionResult(
                action_items=[],
                decisions=["None found.", ""],
                open_questions=[],
                follow_ups=[],
            ),
            clean_transcript="Speaker 1: Just giving updates. No decisions today.",
            source="empty.mp4",
            participants=["Bob"],
        )

        # Seed meeting with sparse metadata (e.g. empty title, no participants)
        save_meeting(
            meeting_id=m_sparse,
            title="",
            summary_obj=MeetingSummary(
                title="",
                overview="",
                summary="Sparse meeting summary.",
                important_points=[],
                key_takeaways=[],
            ),
            extraction_obj=ExtractionResult(
                action_items=[],
                decisions=["Standardize on UTF-8 encoding across all log outputs."],
                open_questions=[],
                follow_ups=[],
            ),
            clean_transcript="Speaker: Logging must use UTF-8.",
            source="",
            participants=[],
        )

        # 3. Test Multiple decisions from the same meeting & deduplication
        alpha_decs = get_all_decisions(meeting_id=m_alpha)
        assert len(alpha_decs) == 3, f"Expected 3 deduplicated decisions from Alpha, got {len(alpha_decs)}"
        expected_alpha_texts = {
            "Proceed with Option B for the product launch.",
            "Assign roles for the upcoming project.",
            "Continue monitoring the supply chain issue.",
        }
        retrieved_texts = {d["decision"] for d in alpha_decs}
        assert retrieved_texts == expected_alpha_texts, f"Mismatch in Alpha decisions: {retrieved_texts}"
        print("[OK] Test 3: Multiple decisions from same meeting retrieved, normalized, and deduplicated (phantom skipped)")

        # 4. Test Decisions from different meetings correctly partitioned
        beta_decs = get_all_decisions(meeting_id=m_beta)
        assert len(beta_decs) == 2, f"Expected 2 decisions from Beta, got {len(beta_decs)}"
        assert all(d["meeting_id"] == m_beta for d in beta_decs)
        assert all(d["meeting_title"] == "Design System & Architecture Alignment" for d in beta_decs)
        print("[OK] Test 4: Decisions from different meetings strictly isolated with verified meeting provenance")

        # 5. Test Empty repository for meeting without decisions
        empty_decs = get_all_decisions(meeting_id=m_empty)
        assert len(empty_decs) == 0, f"Expected 0 decisions for meeting without decisions, got {len(empty_decs)}"
        print("[OK] Test 5: Meeting without decisions produces zero phantom decisions")

        # 6. Test Missing meeting metadata resilience
        sparse_decs = get_all_decisions(meeting_id=m_sparse)
        assert len(sparse_decs) == 1
        assert sparse_decs[0]["meeting_title"] == f"Meeting ({m_sparse})"
        assert sparse_decs[0]["participants"] == []
        assert sparse_decs[0]["decision"] == "Standardize on UTF-8 encoding across all log outputs."
        print("[OK] Test 6: Missing metadata handled gracefully with sensible defaults")

        # 7. Test Meeting scoping by list of IDs
        multi_decs = get_all_decisions(meeting_id=[m_alpha, m_beta])
        assert len(multi_decs) == 5, f"Expected 5 decisions across Alpha & Beta, got {len(multi_decs)}"
        print("[OK] Test 7: Multi-meeting ID scoping correctly returns decisions across target meetings")

        # 8. Test Participant filtering
        elena_decs = get_all_decisions(participant="Elena_DecTest")
        assert len(elena_decs) == 2
        assert all(d["meeting_id"] == m_beta for d in elena_decs)

        sarah_decs = get_all_decisions(participant="Sarah_DecTest")
        assert len(sarah_decs) == 3
        assert all(d["meeting_id"] == m_alpha for d in sarah_decs)
        print("[OK] Test 8: Participant filter accurately scopes decisions to attending members")

        # 9. Test Keyword Search filtering
        search_launch = get_all_decisions(search_term="Option B")
        assert len(search_launch) >= 1
        assert all("option b" in d["decision"].lower() for d in search_launch)

        search_cache = get_all_decisions(search_term="embedding cache")
        assert len(search_cache) >= 1
        assert all("embedding cache" in d["decision"].lower() for d in search_cache)
        print("[OK] Test 9: Decision text search returns exact matching items")

        # 10. Test Search by meeting title
        search_title = get_all_decisions(meeting_id=m_beta, search_term="Architecture Alignment")
        assert len(search_title) == 2
        assert all(d["meeting_id"] == m_beta for d in search_title)
        print("[OK] Test 10: Search by meeting title retrieves all decisions belonging to that session")

    finally:
        # Clean up test records
        delete_meeting(m_alpha)
        delete_meeting(m_beta)
        delete_meeting(m_empty)
        delete_meeting(m_sparse)

    print("\n" + "=" * 60)
    print("ALL DECISIONS REPOSITORY & RETRIEVAL VALIDATION TESTS PASSED (10/10)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
