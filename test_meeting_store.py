"""
Test suite for Meeting Store & Action Item Database (Feature 7).
Validates persistence, overdue calculation, task status updates, deduplication, and metrics.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.extractor import ActionItem, ExtractionResult
from core.meeting_store import (
    add_custom_task,
    calculate_dashboard_metrics,
    delete_meeting,
    delete_task,
    get_action_items,
    get_all_decisions,
    get_all_meetings,
    get_meeting_by_id,
    is_task_overdue,
    save_meeting,
    update_task_details,
    update_task_status,
)
from core.summariser import MeetingSummary


if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")


def run_tests() -> None:
    print("=" * 60)
    print("RUNNING MEETING STORE & PROGRESS DASHBOARD TESTS")
    print("=" * 60)

    # Clean up test meetings if existing
    test_id_1 = "test-session-001"
    test_id_2 = "test-session-002"
    delete_meeting(test_id_1)
    delete_meeting(test_id_2)

    # 1. Save new meeting with action items and decisions
    past_due = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
    future_due = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")

    summary_1 = MeetingSummary(
        title="Q3 Core Architecture Review",
        overview="Reviewing SQLite persistence and RAG grounding architecture.",
        summary="Engineering decided to implement WAL-mode SQLite database for persistent meeting memory and action item tracking.",
        important_points=["WAL mode enables concurrent reads without database locking."],
        key_takeaways=["SQLite database provides single source of truth for task statuses."],
    )

    extraction_1 = ExtractionResult(
        action_items=[
            ActionItem(action="Implement SQLite meeting store tables", owner="Alex", deadline=past_due, status="Pending"),
            ActionItem(action="Deploy cross-meeting intelligence engine", owner="Priya", deadline=future_due, status="In Progress"),
            ActionItem(action="Audit database indexes and vacuum", owner="DevOps", deadline="", status="Pending"),
            ActionItem(action="Write master integration test suite", owner="Alex", deadline=past_due, status="Completed"),
        ],
        decisions=[
            "Approved SQLite as authoritative store for meeting records and action items.",
            "Enforce foreign key constraints with ON DELETE CASCADE.",
        ],
        open_questions=["What is the max size before archiving old transcripts?"],
        follow_ups=["Review sprint deliverables next Monday."],
    )

    saved_record = save_meeting(
        meeting_id=test_id_1,
        title=summary_1.title,
        summary_obj=summary_1,
        extraction_obj=extraction_1,
        clean_transcript="Clean test transcript content.",
        original_transcript="Original test transcript content.",
        source="test_video.mp4",
        participants=["Alex", "Priya", "DevOps"],
    )

    assert saved_record is not None, "Failed to save meeting record"
    assert saved_record.id == test_id_1, f"Expected ID {test_id_1}, got {saved_record.id}"
    assert saved_record.title == "Q3 Core Architecture Review"
    print("[OK] Test 1: New meeting successfully saved to database")

    # 2. Existing meetings remain accessible
    fetched = get_meeting_by_id(test_id_1)
    assert fetched is not None, "Failed to retrieve meeting by ID"
    assert len(fetched.decisions) == 2, f"Expected 2 decisions, got {len(fetched.decisions)}"
    assert "Alex" in fetched.participants, "Expected Alex in participants"
    print("[OK] Test 2: Meeting record accessible with full structured metadata")

    # 3. Action items saved correctly
    tasks = get_action_items(meeting_id=test_id_1)
    assert len(tasks) == 4, f"Expected 4 action items, got {len(tasks)}"
    print("[OK] Test 3: Extracted action items persisted correctly")

    # 4. Overdue detection logic verification
    # Task 1: past due date + Pending => MUST be overdue
    task_1 = next(t for t in tasks if t.description == "Implement SQLite meeting store tables")
    assert task_1.is_overdue is True, "Task with past deadline and Pending status must be overdue"

    # Task 2: future due date + In Progress => MUST NOT be overdue
    task_2 = next(t for t in tasks if t.description == "Deploy cross-meeting intelligence engine")
    assert task_2.is_overdue is False, "Task with future deadline must not be overdue"

    # Task 3: no due date => MUST NOT be overdue
    task_3 = next(t for t in tasks if t.description == "Audit database indexes and vacuum")
    assert task_3.is_overdue is False, "Task without due date must never be overdue"

    # Task 4: past due date + Completed => MUST NOT be overdue
    task_4 = next(t for t in tasks if t.description == "Write master integration test suite")
    assert task_4.is_overdue is False, "Completed task must NEVER be marked overdue"
    print("[OK] Test 4: Overdue detection correctly evaluated for all deadline & status rules")

    # 5. Task status changes persist
    assert task_1.id is not None
    update_task_status(task_1.id, "Completed")
    tasks_after_update = get_action_items(meeting_id=test_id_1)
    updated_t1 = next(t for t in tasks_after_update if t.id == task_1.id)
    assert updated_t1.status == "Completed", "Task status change failed to persist"
    # Once marked completed, it must no longer be overdue!
    assert updated_t1.is_overdue is False, "Completed task must no longer be overdue"
    print("[OK] Test 5: Task status update persisted and removed from overdue status")

    # 6. Task details editing
    update_task_details(
        task_id=task_3.id,  # type: ignore[arg-type]
        description=task_3.description,
        assignee="Lead DBA",
        due_date=future_due,
        priority="High",
        status="In Progress",
    )
    tasks_after_edit = get_action_items(meeting_id=test_id_1)
    edited_t3 = next(t for t in tasks_after_edit if t.id == task_3.id)
    assert edited_t3.assignee == "Lead DBA"
    assert edited_t3.priority == "High"
    assert edited_t3.status == "In Progress"
    print("[OK] Test 6: Task details editing (assignee, deadline, priority, status) persisted")

    # 7. Deduplication on reprocessing
    # Calling save_meeting again with same meeting ID must NOT create duplicate tasks
    save_meeting(
        meeting_id=test_id_1,
        title=summary_1.title,
        summary_obj=summary_1,
        extraction_obj=extraction_1,
        clean_transcript="Updated clean transcript.",
        original_transcript="Updated original transcript.",
        source="test_video.mp4",
    )
    tasks_reprocessed = get_action_items(meeting_id=test_id_1)
    assert len(tasks_reprocessed) == 4, f"Reprocessing created duplicates! Expected 4, got {len(tasks_reprocessed)}"
    # And edited values must be preserved!
    rechecked_t3 = next(t for t in tasks_reprocessed if t.id == task_3.id)
    assert rechecked_t3.assignee == "Lead DBA", "Reprocessing overwrote user-modified assignee"
    print("[OK] Test 7: Reprocessing meeting prevents duplicate tasks and preserves user edits")

    # 8. Dashboard metrics calculation
    metrics = calculate_dashboard_metrics()
    assert metrics.total_meetings >= 1
    assert metrics.total_tasks >= 4
    # Calculate expected completion rate
    expected_pct = round((metrics.completed_tasks / metrics.total_tasks) * 100.0, 1)
    assert abs(metrics.completion_percentage - expected_pct) < 0.1
    print("[OK] Test 8: Dashboard metrics accurately computed from database records")

    # 9. Search and filtering
    search_results = get_all_meetings(search_term="Architecture")
    assert len(search_results) >= 1
    assert any(m.id == test_id_1 for m in search_results)
    print("[OK] Test 9: Meeting history search by keyword returned matching records")

    # 10. Clean up test record
    delete_meeting(test_id_1)
    assert get_meeting_by_id(test_id_1) is None
    assert len(get_action_items(meeting_id=test_id_1)) == 0, "Action items should cascade delete"
    print("[OK] Test 10: Meeting and associated action items safely deleted via cascade")

    print("\n" + "=" * 60)
    print("ALL MEETING STORE & ACTION ITEM TESTS PASSED (10/10)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
