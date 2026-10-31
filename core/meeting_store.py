"""
Meeting Store & Action Item Database.

Provides persistent SQLite storage for meetings, structured summaries,
decisions, and action items. Handles task status updates, overdue detection,
progress metric calculation, and meeting search.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import get_logger

logger = get_logger(__name__)

DB_PATH = Path(_PROJECT_ROOT) / "data" / "meetings.db"


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class MeetingRecord:
    id: str
    title: str
    created_at: str
    source: str = ""
    overview: str = ""
    summary: str = ""
    important_points: list[str] = field(default_factory=list)
    key_takeaways: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    clean_transcript: str = ""
    original_transcript: str = ""
    status: str = "Completed"
    user_id: str = "default_user"


@dataclass
class TaskRecord:
    id: int | None
    meeting_id: str
    meeting_title: str
    description: str
    assignee: str = "Unassigned"
    due_date: str = ""  # YYYY-MM-DD or ""
    priority: str = "Medium"  # Low, Medium, High
    status: str = "Pending"  # Pending, In Progress, Completed
    created_at: str = ""
    updated_at: str = ""
    is_overdue: bool = False
    user_id: str = "default_user"


@dataclass
class DashboardMetrics:
    total_meetings: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    pending_tasks: int = 0
    in_progress_tasks: int = 0
    overdue_tasks: int = 0
    high_priority_tasks: int = 0
    completion_percentage: float = 0.0
    upcoming_tasks_count: int = 0


# -----------------------------------------------------------------------------
# Database Initialization & Connection
# -----------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return an SQLite connection with Row factory and foreign keys enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn



def init_db() -> None:
    """Create database tables and indices safely if they do not exist."""
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT DEFAULT '',
                overview TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                important_points TEXT DEFAULT '[]',
                key_takeaways TEXT DEFAULT '[]',
                decisions TEXT DEFAULT '[]',
                open_questions TEXT DEFAULT '[]',
                follow_ups TEXT DEFAULT '[]',
                participants TEXT DEFAULT '[]',
                clean_transcript TEXT DEFAULT '',
                original_transcript TEXT DEFAULT '',
                status TEXT DEFAULT 'Completed',
                user_id TEXT DEFAULT 'default_user'
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                meeting_title TEXT NOT NULL,
                description TEXT NOT NULL,
                assignee TEXT DEFAULT 'Unassigned',
                due_date TEXT DEFAULT '',
                priority TEXT DEFAULT 'Medium',
                status TEXT DEFAULT 'Pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id TEXT DEFAULT 'default_user',
                FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE
            );
            """
        )

        # Migration check: Ensure user_id column exists on existing pre-auth tables
        for table in ("meetings", "action_items"):
            try:
                cur = conn.execute(f"PRAGMA table_info({table})")
                rows = cur.fetchall()
                if rows:
                    cols = [r[1] if isinstance(r, (tuple, list)) else r["name"] for r in rows]
                    if "user_id" not in cols:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT 'default_user';")
                        logger.info("Migrated table '%s' to include user_id column.", table)
            except Exception as mig_err:
                logger.warning("Migration check for table '%s' skipped: %s", table, mig_err)

        # Indices for efficient querying
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meetings_created ON meetings(created_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meetings_user ON meetings(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_meeting ON action_items(meeting_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON action_items(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_due ON action_items(due_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user ON action_items(user_id);")

        conn.commit()
    finally:
        conn.close()

    logger.info("Initialized Meeting database at '%s'", DB_PATH)


# Initialize schema upon module load
init_db()


# -----------------------------------------------------------------------------
# Helper Functions: Date Parsing & Overdue Detection
# -----------------------------------------------------------------------------

def parse_iso_date(date_str: str | None) -> date | None:
    """Parse string date (YYYY-MM-DD) safely; return None if invalid or unassigned."""
    if not date_str:
        return None
    cleaned = date_str.strip()
    if not cleaned or cleaned.lower() in ("not specified", "unassigned", "none", "n/a", "tbd"):
        return None
    try:
        # Try direct ISO format (YYYY-MM-DD)
        return datetime.strptime(cleaned[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        pass

    # Try common formats
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def is_task_overdue(due_date_str: str | None, status: str) -> bool:
    """
    Check if a task is overdue.
    Rules:
    - Status == 'Completed' is NEVER overdue.
    - Missing / unassigned due date is NEVER overdue.
    - Overdue if parsed due date < current date and status != 'Completed'.
    """
    if status == "Completed":
        return False
    parsed = parse_iso_date(due_date_str)
    if not parsed:
        return False
    return parsed < date.today()


# -----------------------------------------------------------------------------
# Meeting CRUD Operations
# -----------------------------------------------------------------------------

def save_meeting(
    meeting_id: str,
    title: str,
    summary_obj: Any,
    extraction_obj: Any,
    clean_transcript: str = "",
    original_transcript: str = "",
    source: str = "",
    participants: list[str] | None = None,
    user_id: str = "default_user",
) -> MeetingRecord:
    """
    Save or update a meeting record and its extracted action items.
    Prevents duplicates on reprocessing.
    """
    now_iso = datetime.now().isoformat()

    # Extract fields from MeetingSummary / extraction objects safely
    overview = getattr(summary_obj, "overview", "")
    summary_text = getattr(summary_obj, "summary", "")
    important_points = list(getattr(summary_obj, "important_points", []) or [])
    key_takeaways = list(getattr(summary_obj, "key_takeaways", []) or [])

    decisions = list(getattr(extraction_obj, "decisions", []) or [])
    open_questions = list(getattr(extraction_obj, "open_questions", []) or [])
    follow_ups = list(getattr(extraction_obj, "follow_ups", []) or [])
    extracted_actions = list(getattr(extraction_obj, "action_items", []) or [])

    # Infer participants from action item owners if not explicitly supplied
    if participants is None:
        inferred = set()
        for it in extracted_actions:
            owner = getattr(it, "owner", "")
            if owner and owner.lower() not in ("not specified", "unassigned", "none"):
                inferred.add(owner.strip())
        participants = sorted(list(inferred))

    with get_connection() as conn:
        # Check if meeting already exists
        cur = conn.execute("SELECT created_at FROM meetings WHERE id = ?", (meeting_id,))
        existing = cur.fetchone()
        created_at = existing["created_at"] if existing else now_iso

        conn.execute(
            """
            INSERT INTO meetings (
                id, title, created_at, source, overview, summary,
                important_points, key_takeaways, decisions, open_questions,
                follow_ups, participants, clean_transcript, original_transcript, status, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                source = excluded.source,
                overview = excluded.overview,
                summary = excluded.summary,
                important_points = excluded.important_points,
                key_takeaways = excluded.key_takeaways,
                decisions = excluded.decisions,
                open_questions = excluded.open_questions,
                follow_ups = excluded.follow_ups,
                participants = excluded.participants,
                clean_transcript = excluded.clean_transcript,
                original_transcript = excluded.original_transcript,
                status = excluded.status,
                user_id = excluded.user_id;
            """,
            (
                meeting_id,
                title,
                created_at,
                source,
                overview,
                summary_text,
                json.dumps(important_points),
                json.dumps(key_takeaways),
                json.dumps(decisions),
                json.dumps(open_questions),
                json.dumps(follow_ups),
                json.dumps(participants),
                clean_transcript,
                original_transcript,
                "Completed",
                user_id,
            ),
        )

        # Sync action items: Avoid duplicating tasks when reprocessed
        # Query existing task descriptions for this meeting
        cur = conn.execute("SELECT description, status, priority, due_date, assignee FROM action_items WHERE meeting_id = ?", (meeting_id,))
        existing_tasks = {row["description"].strip(): row for row in cur.fetchall()}

        for item in extracted_actions:
            desc = getattr(item, "action", "").strip()
            if not desc:
                continue

            owner = getattr(item, "owner", "").strip()
            if not owner or owner.lower() in ("not specified", "none", "unknown"):
                owner = "Unassigned"

            deadline = getattr(item, "deadline", "").strip()
            if deadline.lower() in ("not specified", "none", "unknown"):
                deadline = ""

            status = getattr(item, "status", "").strip()
            if not status or status.lower() in ("not specified", "none", "pending", "open"):
                status = "Pending"
            elif status.lower() in ("in progress", "active"):
                status = "In Progress"
            elif status.lower() in ("completed", "done", "closed"):
                status = "Completed"
            else:
                status = "Pending"

            if desc in existing_tasks:
                # Existing task: preserve user-modified status/priority
                continue
            else:
                conn.execute(
                    """
                    INSERT INTO action_items (
                        meeting_id, meeting_title, description, assignee,
                        due_date, priority, status, created_at, updated_at, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        meeting_id,
                        title,
                        desc,
                        owner,
                        deadline,
                        "Medium",
                        status,
                        now_iso,
                        now_iso,
                        user_id,
                    ),
                )

    logger.info("Saved meeting '%s' (%s) with %d action items", title, meeting_id, len(extracted_actions))
    return get_meeting_by_id(meeting_id)  # type: ignore[return-value]


def get_meeting_by_id(meeting_id: str) -> MeetingRecord | None:
    """Retrieve full meeting record by meeting ID."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
        row = cur.fetchone()
        if not row:
            return None

        return MeetingRecord(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            source=row["source"],
            overview=row["overview"],
            summary=row["summary"],
            important_points=json.loads(row["important_points"] or "[]"),
            key_takeaways=json.loads(row["key_takeaways"] or "[]"),
            decisions=json.loads(row["decisions"] or "[]"),
            open_questions=json.loads(row["open_questions"] or "[]"),
            follow_ups=json.loads(row["follow_ups"] or "[]"),
            participants=json.loads(row["participants"] or "[]"),
            clean_transcript=row["clean_transcript"],
            original_transcript=row["original_transcript"],
            status=row["status"],
            user_id=row["user_id"] if "user_id" in row.keys() else "default_user",
        )


def get_all_meetings(search_term: str = "", limit: int = 100, user_id: str | None = None) -> list[MeetingRecord]:
    """Retrieve meetings list, optionally filtered by user_id and search keyword."""
    query = "SELECT * FROM meetings"
    conditions: list[str] = []
    params: list[Any] = []

    if user_id:
        conditions.append("(user_id = ? OR user_id = 'default_user')")
        params.append(user_id)

    if search_term.strip():
        term = f"%{search_term.strip()}%"
        conditions.append("""(
            title LIKE ?
            OR overview LIKE ?
            OR summary LIKE ?
            OR decisions LIKE ?
            OR participants LIKE ?
            OR created_at LIKE ?
        )""")
        params.extend([term, term, term, term, term, term])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    records: list[MeetingRecord] = []
    with get_connection() as conn:
        cur = conn.execute(query, params)
        for row in cur.fetchall():
            records.append(
                MeetingRecord(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    source=row["source"],
                    overview=row["overview"],
                    summary=row["summary"],
                    important_points=json.loads(row["important_points"] or "[]"),
                    key_takeaways=json.loads(row["key_takeaways"] or "[]"),
                    decisions=json.loads(row["decisions"] or "[]"),
                    open_questions=json.loads(row["open_questions"] or "[]"),
                    follow_ups=json.loads(row["follow_ups"] or "[]"),
                    participants=json.loads(row["participants"] or "[]"),
                    clean_transcript=row["clean_transcript"],
                    original_transcript=row["original_transcript"],
                    status=row["status"],
                    user_id=row["user_id"] if "user_id" in row.keys() else "default_user",
                )
            )
    return records


def delete_meeting(meeting_id: str) -> bool:
    """Delete a meeting and its associated action items and vector collection."""
    # Clean up associated vector collection from ChromaDB
    try:
        from core.vector_store import delete_session
        delete_session(meeting_id)
    except Exception as exc:
        logger.debug("Could not delete vector collection for '%s': %s", meeting_id, exc)

    with get_connection() as conn:
        cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        return cur.rowcount > 0



# -----------------------------------------------------------------------------
# Action Item Operations
# -----------------------------------------------------------------------------

def get_action_items(
    meeting_id: str | None = None,
    status: str | None = None,
    assignee: str | None = None,
    priority: str | None = None,
    overdue_only: bool = False,
    user_id: str | None = None,
) -> list[TaskRecord]:
    """Retrieve action items with optional multi-criteria filtering and user isolation."""
    query = "SELECT * FROM action_items WHERE 1=1"
    params: list[Any] = []

    if user_id:
        query += " AND (user_id = ? OR user_id = 'default_user')"
        params.append(user_id)

    if meeting_id:
        query += " AND meeting_id = ?"
        params.append(meeting_id)

    if status and status != "All":
        query += " AND status = ?"
        params.append(status)

    if assignee and assignee != "All":
        query += " AND assignee = ?"
        params.append(assignee)

    if priority and priority != "All":
        query += " AND priority = ?"
        params.append(priority)

    query += " ORDER BY id DESC"

    tasks: list[TaskRecord] = []
    with get_connection() as conn:
        cur = conn.execute(query, params)
        for row in cur.fetchall():
            is_over = is_task_overdue(row["due_date"], row["status"])
            if overdue_only and not is_over:
                continue

            tasks.append(
                TaskRecord(
                    id=row["id"],
                    meeting_id=row["meeting_id"],
                    meeting_title=row["meeting_title"],
                    description=row["description"],
                    assignee=row["assignee"],
                    due_date=row["due_date"],
                    priority=row["priority"],
                    status=row["status"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    is_overdue=is_over,
                    user_id=row["user_id"] if "user_id" in row.keys() else "default_user",
                )
            )
    return tasks


def update_task_status(task_id: int, new_status: str) -> bool:
    """Update task status (Pending, In Progress, Completed)."""
    if new_status not in ("Pending", "In Progress", "Completed"):
        raise ValueError(f"Invalid task status: '{new_status}'")

    now_iso = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE action_items SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, now_iso, task_id),
        )
        return cur.rowcount > 0


def update_task_details(
    task_id: int,
    description: str,
    assignee: str,
    due_date: str,
    priority: str,
    status: str,
) -> bool:
    """Update full task details (description, assignee, due date, priority, status)."""
    now_iso = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE action_items
            SET description = ?,
                assignee = ?,
                due_date = ?,
                priority = ?,
                status = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (
                description.strip(),
                assignee.strip() or "Unassigned",
                due_date.strip(),
                priority,
                status,
                now_iso,
                task_id,
            ),
        )
        return cur.rowcount > 0


def add_custom_task(
    meeting_id: str,
    meeting_title: str,
    description: str,
    assignee: str = "Unassigned",
    due_date: str = "",
    priority: str = "Medium",
    status: str = "Pending",
    user_id: str = "default_user",
) -> int:
    """Add a new task manually associated with a meeting."""
    now_iso = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO action_items (
                meeting_id, meeting_title, description, assignee,
                due_date, priority, status, created_at, updated_at, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                meeting_id,
                meeting_title,
                description.strip(),
                assignee.strip() or "Unassigned",
                due_date.strip(),
                priority,
                status,
                now_iso,
                now_iso,
                user_id,
            ),
        )
        return cur.lastrowid or 0


def delete_task(task_id: int) -> bool:
    """Delete a task by ID."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM action_items WHERE id = ?", (task_id,))
        return cur.rowcount > 0


# -----------------------------------------------------------------------------
# Progress Dashboard Metrics & Aggregations
# -----------------------------------------------------------------------------

def calculate_dashboard_metrics(
    meeting_id_filter: str | None = None,
    user_id: str | None = None,
) -> DashboardMetrics:
    """Compute aggregate progress metrics from real database records."""
    # Fast O(1) count without loading full meeting transcripts into memory
    with get_connection() as conn:
        if user_id:
            cur = conn.execute("SELECT COUNT(*) FROM meetings WHERE (user_id = ? OR user_id = 'default_user')", (user_id,))
        else:
            cur = conn.execute("SELECT COUNT(*) FROM meetings")
        row = cur.fetchone()
        total_meetings = row[0] if row else 0

    tasks = get_action_items(meeting_id=meeting_id_filter, user_id=user_id)
    total_tasks = len(tasks)


    completed = sum(1 for t in tasks if t.status == "Completed")
    in_progress = sum(1 for t in tasks if t.status == "In Progress")
    pending = sum(1 for t in tasks if t.status == "Pending")
    overdue = sum(1 for t in tasks if t.is_overdue)

    completion_pct = round((completed / total_tasks * 100.0), 1) if total_tasks > 0 else 0.0

    # Count upcoming tasks (due within next 7 days, not completed)
    today = date.today()
    upcoming_count = 0
    for t in tasks:
        if t.status != "Completed":
            parsed = parse_iso_date(t.due_date)
            if parsed and today <= parsed:
                delta = (parsed - today).days
                if delta <= 7:
                    upcoming_count += 1

    high_priority = sum(1 for t in tasks if t.priority.strip().lower() == "high" and t.status != "Completed")

    return DashboardMetrics(
        total_meetings=total_meetings,
        total_tasks=total_tasks,
        completed_tasks=completed,
        pending_tasks=pending,
        in_progress_tasks=in_progress,
        overdue_tasks=overdue,
        high_priority_tasks=high_priority,
        completion_percentage=completion_pct,
        upcoming_tasks_count=upcoming_count,
    )


def normalize_decision_text(text: str) -> str:
    """
    Clean and normalize decision strings by stripping leading bullets, numbers, hyphens,
    and whitespace, while preserving original phrasing and capitalized acronyms.
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Strip leading markdown list markers, bullets, and numbering:
    # e.g. "1. ", "1) ", "- ", "• ", "* ", "– ", "— ", "(1) ", "1: "
    cleaned = re.sub(r"^(?:[-*•–—]|\d+[\.\)\-:]|\([0-9]+\))\s*", "", cleaned).strip()
    return cleaned


INVALID_DECISION_PHRASES = {
    "none",
    "none.",
    "none found.",
    "none found",
    "no decisions",
    "no decisions made.",
    "no decisions made",
    "no decisions recorded.",
    "no decision recorded.",
    "n/a",
    "not applicable",
    "-",
}


def get_all_decisions(
    meeting_id: str | list[str] | set[str] | None = None,
    search_term: str = "",
    participant: str | None = None,
    limit: int = 100,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve deduplicated, normalized decisions across all meetings with rich meeting context and metadata.
    Supports filtering by meeting_id (single ID or list/set of IDs), search_term, participant, and user_id.
    """
    meetings = get_all_meetings(limit=100, user_id=user_id)
    results: list[dict[str, Any]] = []
    seen_decisions_per_meeting: set[tuple[str, str]] = set()

    search_norm = search_term.strip().lower()
    p_norm = participant.strip().lower() if participant and participant != "All" else ""

    # Normalize target meeting IDs filter
    target_meeting_ids: set[str] | None = None
    if meeting_id:
        if isinstance(meeting_id, (list, set, tuple)):
            target_meeting_ids = set(meeting_id)
        elif isinstance(meeting_id, str):
            target_meeting_ids = {meeting_id}

    for m in meetings:
        # Filter by meeting_id if specified
        if target_meeting_ids is not None and m.id not in target_meeting_ids:
            continue

        # Filter by participant if specified
        participants = m.participants or []
        if p_norm and not any(p_norm in p.lower() for p in participants):
            continue

        meeting_title = m.title.strip() if (m.title and m.title.strip()) else f"Meeting ({m.id})"
        created_at = m.created_at or ""

        # Process decisions from this meeting
        for idx, dec in enumerate(m.decisions or []):
            if not dec:
                continue

            dec_text = normalize_decision_text(dec)
            if not dec_text or dec_text.lower() in INVALID_DECISION_PHRASES:
                continue

            # Deduplicate per meeting (avoid repeated processing artifacts)
            dedup_key = (m.id, dec_text.lower())
            if dedup_key in seen_decisions_per_meeting:
                continue
            seen_decisions_per_meeting.add(dedup_key)

            # Filter by search term if provided
            if search_norm:
                matches_p = any(search_norm in p.lower() for p in participants)
                if (
                    search_norm not in dec_text.lower()
                    and search_norm not in meeting_title.lower()
                    and search_norm not in (m.id or "").lower()
                    and not matches_p
                ):
                    continue

            results.append({
                "decision_id": f"{m.id}-dec-{idx + 1}",
                "meeting_id": m.id,
                "meeting_title": meeting_title,
                "created_at": created_at,
                "meeting_date": created_at,
                "decision": dec_text,
                "status": "Adopted",
                "participants": participants,
                "source": m.source or "",
                "clean_transcript": m.clean_transcript or "",
                "overview": m.overview or "",
            })

            if len(results) >= limit:
                break

        if len(results) >= limit:
            break

    return results

