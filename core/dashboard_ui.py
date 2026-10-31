"""
Meeting Memory & Progress Dashboard UI Components.

Provides UI views for:
1. Overview / Progress Dashboard (Metrics, charts, overdue alerts, recent decisions)
2. Meeting History (Searchable, filterable past meetings, detail view, load into workspace)
3. Action Items Tracker (Status updates, editable task details, overdue indicators)
4. Decisions and Memory (Cross-meeting RAG Q&A and searchable decisions repository)
"""

from __future__ import annotations

import html
import json
import textwrap
from datetime import date, datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from config import get_logger
from core.cross_meeting_rag import ask_cross_meetings
from core.extractor import ActionItem, ExtractionResult
from core.meeting_store import (
    DashboardMetrics,
    MeetingRecord,
    TaskRecord,
    add_custom_task,
    calculate_dashboard_metrics,
    delete_meeting,
    delete_task,
    get_action_items,
    get_all_decisions,
    get_all_meetings,
    get_meeting_by_id,
    is_task_overdue,
    update_task_details,
    update_task_status,
)
from core.summariser import MeetingSummary

logger = get_logger(__name__)


def _html(content: str) -> None:
    """Render HTML in Streamlit cleanly without CommonMark indentation or code-block issues."""
    clean_lines = [line.strip() for line in content.splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)


def _format_date(iso_str: str) -> str:
    """Format ISO date string to human-friendly format."""
    if not iso_str:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %Y (%I:%M %p)")
    except Exception:
        return iso_str[:16]


def _format_short_date(iso_str: str) -> str:
    """Format ISO date to YYYY-MM-DD or short date."""
    if not iso_str:
        return "None"
    return iso_str[:10]


# =============================================================================
# 1. OVERVIEW & PROGRESS DASHBOARD
# =============================================================================

def render_overview_dashboard(user_id: str | None = None) -> None:
    """Render the high-level progress dashboard with real metrics, overdue alerts, and charts."""
    metrics: DashboardMetrics = calculate_dashboard_metrics(user_id=user_id)
    all_meetings = get_all_meetings(user_id=user_id)

    # Empty state handling
    if metrics.total_meetings == 0:
        _html("""
        <div style="background-color: var(--bg-card); border: 1px dashed var(--border-color); border-radius: 12px; padding: 40px; text-align: center; margin-top: 10px; box-shadow: var(--shadow-card);">
            <div style="width: 52px; height: 52px; border-radius: 50%; background-color: rgba(255, 75, 75, 0.1); margin: 0 auto 16px auto; display: flex; align-items: center; justify-content: center;">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                </svg>
            </div>
            <h3 style="color: var(--text-main); margin: 0 0 8px 0; font-size: 1.25rem; font-weight: 600;">No Meetings in Memory Yet</h3>
            <p style="color: var(--text-secondary); margin: 0 0 20px 0; font-size: 0.95rem; max-width: 500px; margin-left: auto; margin-right: auto;">
                Process a video/meeting recording in the Active Workspace, or click <strong>Load Sample Session</strong> to test the memory system.
            </p>
        </div>
        """)
        return

    # Section Heading
    _html("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="7" height="9"></rect>
            <rect x="14" y="3" width="7" height="5"></rect>
            <rect x="14" y="12" width="7" height="9"></rect>
            <rect x="3" y="16" width="7" height="5"></rect>
        </svg>
        <h3 style="color: var(--text-main); font-weight: 700; margin: 0; font-size: 1.3rem;">Meeting & Task Health Overview</h3>
    </div>
    """)

    # Top Metric Summary Cards Grid (6 Prominent Linear/Notion SaaS KPI Cards)
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    with kpi_col1:
        _html(f"""
        <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px 22px; box-shadow: var(--shadow-card); min-height: 124px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em;">Total Action Items</span>
                <div style="width: 32px; height: 32px; border-radius: 8px; background-color: rgba(99, 102, 241, 0.12); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="9 11 12 14 22 4"></polyline>
                        <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                    </svg>
                </div>
            </div>
            <div>
                <div style="font-size: 2.1rem; font-weight: 700; color: var(--text-main); line-height: 1.1; letter-spacing: -0.02em;">{metrics.total_tasks}</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 6px; font-weight: 500;">Across {metrics.total_meetings} recorded meeting(s)</div>
            </div>
        </div>
        """)

    with kpi_col2:
        _html(f"""
        <div style="background-color: var(--bg-card); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 20px 22px; box-shadow: var(--shadow-card); min-height: 124px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.06em;">Pending Tasks</span>
                <div style="width: 32px; height: 32px; border-radius: 8px; background-color: rgba(245, 158, 11, 0.12); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                </div>
            </div>
            <div>
                <div style="font-size: 2.1rem; font-weight: 700; color: #F59E0B; line-height: 1.1; letter-spacing: -0.02em;">{metrics.pending_tasks}</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 6px; font-weight: 500;">Awaiting execution or assignment</div>
            </div>
        </div>
        """)

    with kpi_col3:
        _html(f"""
        <div style="background-color: var(--bg-card); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 20px 22px; box-shadow: var(--shadow-card); min-height: 124px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; color: #3B82F6; text-transform: uppercase; letter-spacing: 0.06em;">In Progress</span>
                <div style="width: 32px; height: 32px; border-radius: 8px; background-color: rgba(59, 130, 246, 0.12); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="12" y1="2" x2="12" y2="6"></line>
                        <line x1="12" y1="18" x2="12" y2="22"></line>
                        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
                        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
                        <line x1="2" y1="12" x2="6" y2="12"></line>
                        <line x1="18" y1="12" x2="22" y2="12"></line>
                    </svg>
                </div>
            </div>
            <div>
                <div style="font-size: 2.1rem; font-weight: 700; color: #3B82F6; line-height: 1.1; letter-spacing: -0.02em;">{metrics.in_progress_tasks}</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 6px; font-weight: 500;">Actively underway across teams</div>
            </div>
        </div>
        """)

    kpi_col4, kpi_col5, kpi_col6 = st.columns(3)

    with kpi_col4:
        _html(f"""
        <div style="background-color: var(--bg-card); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 20px 22px; box-shadow: var(--shadow-card); min-height: 124px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.06em;">Completed Tasks</span>
                <div style="width: 32px; height: 32px; border-radius: 8px; background-color: rgba(16, 185, 129, 0.12); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                </div>
            </div>
            <div>
                <div style="font-size: 2.1rem; font-weight: 700; color: #10B981; line-height: 1.1; letter-spacing: -0.02em;">{metrics.completed_tasks}</div>
                <div style="font-size: 0.8rem; color: #10B981; margin-top: 6px; font-weight: 600;">{metrics.completion_percentage}% overall resolution rate</div>
            </div>
        </div>
        """)

    with kpi_col5:
        high_prio_count = getattr(metrics, "high_priority_tasks", 0)
        _html(f"""
        <div style="background-color: var(--bg-card); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 20px 22px; box-shadow: var(--shadow-card); min-height: 124px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; color: #EF4444; text-transform: uppercase; letter-spacing: 0.06em;">High-Priority Tasks</span>
                <div style="width: 32px; height: 32px; border-radius: 8px; background-color: rgba(239, 68, 68, 0.12); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                    </svg>
                </div>
            </div>
            <div>
                <div style="font-size: 2.1rem; font-weight: 700; color: #EF4444; line-height: 1.1; letter-spacing: -0.02em;">{high_prio_count}</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 6px; font-weight: 500;">Urgent deliverables requiring focus</div>
            </div>
        </div>
        """)

    with kpi_col6:
        has_overdue = metrics.overdue_tasks > 0
        overdue_card_border = "border: 1px solid #EF4444;" if has_overdue else "border: 1px solid var(--border-color);"
        overdue_card_color = "#EF4444" if has_overdue else "var(--text-main)"
        overdue_sub_color = "#EF4444" if has_overdue else "#10B981"
        overdue_sub_text = f"{metrics.overdue_tasks} overdue · {metrics.upcoming_tasks_count} due in 7d" if has_overdue else f"0 overdue · {metrics.upcoming_tasks_count} due in 7d"

        _html(f"""
        <div style="background-color: var(--bg-card); {overdue_card_border} border-radius: 12px; padding: 20px 22px; box-shadow: var(--shadow-card); min-height: 124px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; color: {overdue_card_color}; text-transform: uppercase; letter-spacing: 0.06em;">Upcoming & Overdue</span>
                <div style="width: 32px; height: 32px; border-radius: 8px; background-color: {'rgba(239, 68, 68, 0.12)' if has_overdue else 'rgba(16, 185, 129, 0.12)'}; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="{overdue_card_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="16" y1="2" x2="16" y2="6"></line>
                        <line x1="8" y1="2" x2="8" y2="6"></line>
                        <line x1="3" y1="10" x2="21" y2="10"></line>
                    </svg>
                </div>
            </div>
            <div>
                <div style="font-size: 2.1rem; font-weight: 700; color: {overdue_card_color}; line-height: 1.1; letter-spacing: -0.02em;">{metrics.overdue_tasks}</div>
                <div style="font-size: 0.8rem; color: {overdue_sub_color}; margin-top: 6px; font-weight: 600;">{overdue_sub_text}</div>
            </div>
        </div>
        """)

    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    # Task Completion Velocity Card (Fully Embedded visible progress bar)
    pct_bar = min(100.0, max(0.0, float(metrics.completion_percentage)))
    velocity_label = f"{metrics.completed_tasks} of {metrics.total_tasks} tasks completed · {metrics.completion_percentage}%"
    _html(f"""
    <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px 24px; margin-bottom: 24px; box-shadow: var(--shadow-card);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <span style="font-weight: 600; color: var(--text-main); font-size: 1.05rem;">Task Completion Velocity</span>
            </div>
            <div style="font-weight: 600; color: var(--primary-accent); font-size: 0.95rem;">
                {velocity_label}
            </div>
        </div>
        <div style="width: 100%; height: 12px; background-color: var(--bg-secondary); border-radius: 6px; overflow: hidden; border: 1px solid var(--border-color); position: relative;">
            <div style="width: {pct_bar}%; height: 100%; background-color: var(--primary-accent); border-radius: 6px; transition: width 0.4s ease;"></div>
        </div>
    </div>
    """)

    # Status Distribution Chart & Overdue Action Section
    chart_col, alert_col = st.columns([1.1, 0.9])

    with chart_col:
        _html("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <h4 style="color: var(--text-main); font-weight: 600; margin: 0; font-size: 1.1rem;">Task Status Distribution</h4>
        </div>
        """)

        status_data = pd.DataFrame([
            {"Status": "Completed", "Count": metrics.completed_tasks, "Color": "#10B981"},
            {"Status": "In Progress", "Count": metrics.in_progress_tasks, "Color": "#3B82F6"},
            {"Status": "Pending", "Count": metrics.pending_tasks, "Color": "#F59E0B"},
            {"Status": "Overdue", "Count": metrics.overdue_tasks, "Color": "#EF4444"},
        ])

        curr_theme = st.session_state.get("theme", "Light")
        is_dark_mode = curr_theme == "Dark"
        label_col = "#F9FAFB" if is_dark_mode else "#202638"
        grid_col = "rgba(255, 255, 255, 0.1)" if is_dark_mode else "rgba(0, 0, 0, 0.08)"

        bars = alt.Chart(status_data).mark_bar(cornerRadiusEnd=4, height=22).encode(
            y=alt.Y(
                "Status:N",
                sort=["Completed", "In Progress", "Pending", "Overdue"],
                axis=alt.Axis(
                    title=None,
                    labelColor=label_col,
                    labelFontSize=13,
                    labelFontWeight="bold",
                    domainColor="transparent",
                    ticks=False,
                    labelPadding=10,
                    labelLimit=250,
                ),
            ),
            x=alt.X(
                "Count:Q",
                axis=alt.Axis(
                    title=None,
                    labelColor=label_col,
                    gridColor=grid_col,
                    tickMinStep=1,
                ),
            ),
            color=alt.Color("Color:N", scale=None),
            tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("Count:Q")],
        )

        labels = bars.mark_text(
            align="left",
            baseline="middle",
            dx=6,
            color=label_col,
            fontSize=13,
            fontWeight="bold",
        ).encode(
            text=alt.Text("Count:Q")
        )

        chart = (
            (bars + labels)
            .properties(
                height=210,
                background="transparent",
                padding={"left": 30, "right": 25, "top": 10, "bottom": 10},
            )
            .configure_view(strokeWidth=0)
            .configure_axis(labelLimit=250)
        )

        st.altair_chart(chart, use_container_width=True)

    with alert_col:
        _html("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-error)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <h4 style="color: var(--text-main); font-weight: 600; margin: 0; font-size: 1.1rem;">Overdue Tasks Requiring Attention</h4>
        </div>
        """)

        overdue_tasks = get_action_items(overdue_only=True)
        if not overdue_tasks:
            _html("""
            <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 36px 20px; text-align: center; box-shadow: var(--shadow-card); min-height: 210px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="width: 44px; height: 44px; border-radius: 50%; background-color: rgba(16, 185, 129, 0.12); display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--color-success)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                </div>
                <div style="font-weight: 600; color: var(--text-main); font-size: 1.05rem;">No Overdue Tasks</div>
                <div style="color: var(--text-secondary); font-size: 0.88rem; margin-top: 4px; max-width: 280px;">All action items are completed or scheduled on time.</div>
            </div>
            """)
        else:
            for ot in overdue_tasks[:4]:
                _html(f"""
                <div style="background-color: var(--bg-card); border: 1px solid var(--color-error); border-left: 4px solid var(--color-error); border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; box-shadow: var(--shadow-card);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                        <span style="font-weight: 600; color: var(--text-main); font-size: 0.92rem;">{ot.description}</span>
                        <span style="background-color: var(--color-error); color: #FFFFFF; font-size: 0.7rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; text-transform: uppercase;">OVERDUE</span>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">
                        <strong>Assignee:</strong> {ot.assignee} &nbsp;|&nbsp; <strong>Deadline:</strong> <span style="color: var(--color-error); font-weight: 600;">{ot.due_date}</span> &nbsp;|&nbsp; <em>{ot.meeting_title}</em>
                    </div>
                </div>
                """)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Recent Decisions & Recent Meetings Section
    dec_col, meet_col = st.columns(2)

    with dec_col:
        _html("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
            </svg>
            <h4 style="color: var(--text-main); font-weight: 600; margin: 0; font-size: 1.1rem;">Recent Key Decisions</h4>
        </div>
        """)

        decisions = get_all_decisions(limit=4)
        if not decisions:
            _html("""
            <div style="background-color: var(--bg-card); border: 1px dashed var(--border-color); border-radius: 10px; padding: 24px; text-align: center;">
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">No decisions recorded yet.</p>
            </div>
            """)
        else:
            for item in decisions:
                _html(f"""
                <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; box-shadow: var(--shadow-card);">
                    <div style="font-size: 0.92rem; color: var(--text-main); line-height: 1.45; font-weight: 500;">
                        ✓ {item['decision']}
                    </div>
                    <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 6px;">
                        <strong>Meeting:</strong> {item['meeting_title']} &nbsp;•&nbsp; {_format_date(item['created_at'])}
                    </div>
                </div>
                """)

    with meet_col:
        _html("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="16" y1="2" x2="16" y2="6"></line>
                <line x1="8" y1="2" x2="8" y2="6"></line>
                <line x1="3" y1="10" x2="21" y2="10"></line>
            </svg>
            <h4 style="color: var(--text-main); font-weight: 600; margin: 0; font-size: 1.1rem;">Recent Meetings in Memory</h4>
        </div>
        """)

        if not all_meetings:
            _html("""
            <div style="background-color: var(--bg-card); border: 1px dashed var(--border-color); border-radius: 10px; padding: 24px; text-align: center;">
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">No meetings recorded yet.</p>
            </div>
            """)
        else:
            for m in all_meetings[:3]:
                tasks = get_action_items(meeting_id=m.id)
                summary_preview = (m.overview or m.summary)[:140] + ("..." if len(m.overview or m.summary) > 140 else "")
                p_tags = " ".join([f"<span style='background: rgba(255, 75, 75, 0.1); color: var(--primary-accent); padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; margin-right: 4px;'>{p}</span>" for p in m.participants[:3]])

                with st.container():
                    _html(f"""
                    <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 14px 16px; margin-bottom: 8px; box-shadow: var(--shadow-card);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-main);">{m.title}</div>
                            <span style="font-size: 0.75rem; color: var(--text-secondary);">{len(tasks)} tasks</span>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--text-secondary); margin: 3px 0 6px 0;">
                            {_format_date(m.created_at)}
                        </div>
                        <div style="font-size: 0.84rem; color: var(--text-main); line-height: 1.4; margin-bottom: 8px;">
                            {summary_preview}
                        </div>
                        <div>{p_tags if p_tags else "<span style='font-size: 0.72rem; color: var(--text-secondary);'>No participants recorded</span>"}</div>
                    </div>
                    """)

                    if st.button(
                        f"Open '{m.title}' in Workspace",
                        icon=":material/open_in_new:",
                        key=f"ov_open_m_{m.id}",
                        use_container_width=True,
                    ):
                        full_m = get_meeting_by_id(m.id)
                        if full_m:
                            m_tasks = get_action_items(meeting_id=full_m.id)
                            st.session_state.session_id = full_m.id
                            st.session_state.clean_transcript = full_m.clean_transcript
                            st.session_state.original_transcript = full_m.original_transcript or full_m.clean_transcript
                            st.session_state.summary = MeetingSummary(
                                title=full_m.title,
                                overview=full_m.overview,
                                summary=full_m.summary,
                                important_points=full_m.important_points,
                                key_takeaways=full_m.key_takeaways,
                            )
                            st.session_state.extraction = ExtractionResult(
                                action_items=[
                                    ActionItem(
                                        action=t.description,
                                        owner=t.assignee,
                                        deadline=t.due_date,
                                        status=t.status,
                                    )
                                    for t in m_tasks
                                ],
                                decisions=full_m.decisions,
                                open_questions=full_m.open_questions,
                                follow_ups=full_m.follow_ups,
                            )
                            st.session_state.processed = True
                            st.session_state.chat_history = []
                            st.session_state.app_mode = "workspace"
                            st.session_state.active_tab = "summary"
                            st.toast(f"Loaded '{full_m.title}' into workspace")
                            st.rerun()


# =============================================================================
# Helper: Workspace Meeting Restoration
# =============================================================================

def restore_meeting_to_workspace(meeting: MeetingRecord | str) -> bool:
    """Restore historical meeting state into active workspace and navigate to it."""
    if isinstance(meeting, str):
        meet_obj = get_meeting_by_id(meeting)
    else:
        meet_obj = meeting

    if not meet_obj:
        return False

    tasks = get_action_items(meeting_id=meet_obj.id)
    st.session_state.session_id = meet_obj.id
    st.session_state.clean_transcript = meet_obj.clean_transcript
    st.session_state.original_transcript = meet_obj.original_transcript or meet_obj.clean_transcript
    st.session_state.summary = MeetingSummary(
        title=meet_obj.title,
        overview=meet_obj.overview,
        summary=meet_obj.summary,
        important_points=meet_obj.important_points,
        key_takeaways=meet_obj.key_takeaways,
    )
    st.session_state.extraction = ExtractionResult(
        action_items=[
            ActionItem(
                action=t.description,
                owner=t.assignee,
                deadline=t.due_date,
                status=t.status,
            )
            for t in tasks
        ],
        decisions=meet_obj.decisions,
        open_questions=meet_obj.open_questions,
        follow_ups=meet_obj.follow_ups,
    )
    st.session_state.processed = True
    st.session_state.chat_history = []
    st.session_state.app_mode = "workspace"
    st.session_state.active_tab = "summary"
    return True


# =============================================================================
# 2. MEETING HISTORY
# =============================================================================

def render_meeting_history(user_id: str | None = None) -> None:
    """Render searchable, filterable meeting history with details and workspace restore."""
    _html("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        <h3 style="color: var(--text-main); font-weight: 700; margin: 0; font-size: 1.3rem;">Meeting History & Memory Archive</h3>
    </div>
    <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 16px;">
        Search, review, and reopen any previously processed meeting recording.
    </p>
    """)

    # Search bar & filter controls
    search_col, count_col = st.columns([3, 1])
    with search_col:
        search_term = st.text_input(
            "Search Meetings",
            placeholder="Search by title, keyword, participant, or date...",
            label_visibility="collapsed",
            key="history_search_input",
        )

    meetings = get_all_meetings(search_term=search_term, user_id=user_id)

    with count_col:
        _html(f"""
        <div style="text-align: right; padding-top: 8px; color: var(--text-secondary); font-size: 0.9rem; font-weight: 500;">
            Showing <strong>{len(meetings)}</strong> meeting(s)
        </div>
        """)

    if not meetings:
        if search_term.strip():
            st.info(f"No meetings found matching '{search_term}'. Try a different keyword.")
        else:
            st.info("No meetings recorded in history yet. Process a recording or load a sample session to begin.")
        return

    # Render each meeting card
    for meeting in meetings:
        tasks = get_action_items(meeting_id=meeting.id)
        overdue_cnt = sum(1 for t in tasks if t.is_overdue)
        completed_cnt = sum(1 for t in tasks if t.status == "Completed")

        with st.container():
            _html(f"""
            <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px 22px; margin-bottom: 16px; box-shadow: var(--shadow-card);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #10B981; background: rgba(16, 185, 129, 0.1); padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">
                            {meeting.status}
                        </span>
                        <h4 style="color: var(--text-main); font-size: 1.25rem; font-weight: 700; margin: 8px 0 4px 0;">{meeting.title}</h4>
                        <div style="color: var(--text-secondary); font-size: 0.85rem;">
                            📅 {_format_date(meeting.created_at)} &nbsp;•&nbsp; <strong>ID:</strong> {meeting.id}
                        </div>
                    </div>
                </div>
                <div style="color: var(--text-main); font-size: 0.94rem; line-height: 1.5; margin: 10px 0 14px 0;">
                    {meeting.overview or (meeting.summary[:200] + '...' if len(meeting.summary) > 200 else meeting.summary)}
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.82rem; color: var(--text-secondary); border-top: 1px solid var(--border-color); padding-top: 10px;">
                    <div>📋 <strong>{len(tasks)}</strong> Action Items ({completed_cnt} done, <span style="color: {'#EF4444' if overdue_cnt > 0 else 'inherit'}">{overdue_cnt} overdue</span>)</div>
                    <div>⚖️ <strong>{len(meeting.decisions)}</strong> Decisions</div>
                    <div>👥 <strong>{len(meeting.participants)}</strong> Participants</div>
                </div>
            </div>
            """)

            # Interactive action buttons for this meeting card
            btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 1])

            with btn_col1:
                if st.button(
                    "Open in Workspace",
                    icon=":material/open_in_new:",
                    type="primary",
                    key=f"load_workspace_{meeting.id}",
                    use_container_width=True,
                ):
                    restore_meeting_to_workspace(meeting)
                    st.toast(f"Loaded '{meeting.title}' into workspace")
                    st.rerun()

            with btn_col2:
                expand_details = st.expander("View Full Meeting Details", expanded=False)
                with expand_details:
                    st.markdown("##### Executive Summary")
                    st.write(meeting.summary)

                    if meeting.decisions:
                        st.markdown("##### Key Decisions")
                        for d in meeting.decisions:
                            st.markdown(f"- {d}")

                    if meeting.important_points:
                        st.markdown("##### Important Points")
                        for p in meeting.important_points:
                            st.markdown(f"- {p}")

                    if tasks:
                        st.markdown("##### Action Items")
                        for t in tasks:
                            overdue_txt = " (⚠️ OVERDUE)" if t.is_overdue else ""
                            st.markdown(f"- **[{t.status}{overdue_txt}]** {t.description} (Assignee: {t.assignee}, Due: {t.due_date or 'None'})")

            confirm_meet_key = f"confirm_del_meet_{meeting.id}"
            is_meet_confirming = st.session_state.get(confirm_meet_key, False)

            if is_meet_confirming:
                st.warning(f"Delete meeting '{meeting.title}' and all associated action items? This cannot be undone.")
                mccol1, mccol2 = st.columns(2)
                with mccol1:
                    if st.button("Confirm Delete", key=f"do_del_meet_{meeting.id}", type="primary", icon=":material/delete_forever:", use_container_width=True):
                        delete_meeting(meeting.id)
                        st.session_state[confirm_meet_key] = False
                        st.toast(f"Deleted meeting '{meeting.title}'")
                        st.rerun()
                with mccol2:
                    if st.button("Cancel", key=f"cancel_del_meet_{meeting.id}", use_container_width=True):
                        st.session_state[confirm_meet_key] = False
                        st.rerun()
            else:
                with btn_col3:
                    if st.button(
                        "Delete",
                        icon=":material/delete:",
                        key=f"delete_meet_{meeting.id}",
                        use_container_width=True,
                    ):
                        st.session_state[confirm_meet_key] = True
                        st.rerun()


# =============================================================================
# 3. ACTION ITEMS TRACKER
# =============================================================================

def render_action_items_tracker(user_id: str | None = None) -> None:
    """Render interactive action items management across all meetings."""
    _html("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="9 11 12 14 22 4"></polyline>
            <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
        </svg>
        <h3 style="color: var(--text-main); font-weight: 700; margin: 0; font-size: 1.3rem;">Action Item Management Tracker</h3>
    </div>
    <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 16px;">
        Track, update, and monitor progress on all commitments and deliverables across meetings.
    </p>
    """)

    all_meetings = get_all_meetings(user_id=user_id)

    # Filter Bar
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

    with fcol1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Pending", "In Progress", "Completed", "Overdue Only"],
            key="task_filter_status",
        )

    with fcol2:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All", "High", "Medium", "Low"],
            key="task_filter_priority",
        )

    with fcol3:
        meeting_choices = ["All"] + [f"{m.title} ({m.id[:6]})" for m in all_meetings]
        meeting_choice = st.selectbox(
            "Filter by Meeting",
            meeting_choices,
            key="task_filter_meeting",
        )
        selected_m_id = None
        if meeting_choice != "All":
            for m in all_meetings:
                if f"{m.title} ({m.id[:6]})" == meeting_choice:
                    selected_m_id = m.id
                    break

    with fcol4:
        # Collect distinct assignees
        all_raw_tasks = get_action_items(user_id=user_id)
        assignees = sorted(list({t.assignee for t in all_raw_tasks if t.assignee}))
        assignee_filter = st.selectbox(
            "Filter by Assignee",
            ["All"] + assignees,
            key="task_filter_assignee",
        )

    # Fetch filtered tasks
    overdue_only_flag = (status_filter == "Overdue Only")
    query_status = None if overdue_only_flag or status_filter == "All" else status_filter

    tasks = get_action_items(
        meeting_id=selected_m_id,
        status=query_status,
        assignee=assignee_filter if assignee_filter != "All" else None,
        priority=priority_filter if priority_filter != "All" else None,
        overdue_only=overdue_only_flag,
        user_id=user_id,
    )

    # Quick Add Custom Task Expander
    with st.expander("➕ Add Custom Action Item to a Meeting", expanded=False):
        if not all_meetings:
            st.warning("Please save at least one meeting before creating custom action items.")
        else:
            with st.form("add_custom_task_form", clear_on_submit=True):
                target_m = st.selectbox(
                    "Associated Meeting",
                    options=all_meetings,
                    format_func=lambda m: f"{m.title} ({_format_short_date(m.created_at)})",
                )
                new_desc = st.text_input("Task Description", placeholder="e.g., Deliver security architecture audit report")
                ncol1, ncol2, ncol3 = st.columns(3)
                with ncol1:
                    new_owner = st.text_input("Assignee", placeholder="e.g., Sarah Chen or Engineering")
                with ncol2:
                    new_due = st.date_input("Due Date", value=None)
                with ncol3:
                    new_prio = st.selectbox("Priority", ["Medium", "High", "Low"])

                submit_add = st.form_submit_button("Save Action Item", icon=":material/save:", type="primary")
                if submit_add and new_desc.strip():
                    due_str = new_due.strftime("%Y-%m-%d") if new_due else ""
                    add_custom_task(
                        meeting_id=target_m.id,
                        meeting_title=target_m.title,
                        description=new_desc.strip(),
                        assignee=new_owner.strip() or "Unassigned",
                        due_date=due_str,
                        priority=new_prio,
                        status="Pending",
                    )
                    st.success("Action item created successfully!")
                    st.rerun()

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    st.caption(f"Displaying **{len(tasks)}** task(s)")

    if not tasks:
        st.info("No action items match the selected criteria.")
        return

    # Render Task Cards
    for task in tasks:
        # Determine status pill color & overdue badge
        if task.status == "Completed":
            status_badge = "<span style='background: rgba(16, 185, 129, 0.12); color: #10B981; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; letter-spacing: 0.04em;'>✓ COMPLETED</span>"
        elif task.status == "In Progress":
            status_badge = "<span style='background: rgba(59, 130, 246, 0.12); color: #3B82F6; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; letter-spacing: 0.04em;'>● IN PROGRESS</span>"
        else:
            status_badge = "<span style='background: rgba(245, 158, 11, 0.12); color: #F59E0B; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; letter-spacing: 0.04em;'>○ PENDING</span>"

        overdue_badge = ""
        if task.is_overdue:
            overdue_badge = "<span style='background: #EF4444; color: #FFFFFF; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; margin-left: 6px; letter-spacing: 0.04em;'>⚠️ OVERDUE</span>"

        if str(task.priority).strip().lower() == "high":
            prio_badge = "<span style='background: rgba(239, 68, 68, 0.12); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); font-weight: 700; padding: 3px 8px; border-radius: 6px; font-size: 0.7rem;'>▲ HIGH PRIORITY</span>"
        elif str(task.priority).strip().lower() == "medium":
            prio_badge = "<span style='background: rgba(59, 130, 246, 0.12); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.3); font-weight: 600; padding: 3px 8px; border-radius: 6px; font-size: 0.7rem;'>■ MEDIUM</span>"
        else:
            prio_badge = "<span style='background: var(--bg-secondary); color: var(--text-secondary); border: 1px solid var(--border-color); font-weight: 600; padding: 3px 8px; border-radius: 6px; font-size: 0.7rem;'>▼ LOW</span>"

        border_card_color = "#EF4444" if task.is_overdue else "var(--border-color)"
        escaped_desc = html.escape(task.description)
        escaped_assignee = html.escape(task.assignee or "Unassigned")
        escaped_due = html.escape(task.due_date) if task.due_date else "Not specified"
        escaped_meeting = html.escape(task.meeting_title or "Standalone Task")

        due_html = f"<span style='color: #EF4444; font-weight: 700;'>{escaped_due} (Overdue)</span>" if task.is_overdue else escaped_due

        with st.container():
            _html(f"""
            <div style="background-color: var(--bg-card); border: 1px solid {border_card_color}; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; box-shadow: var(--shadow-card); width: 100%; box-sizing: border-box; overflow: hidden; word-break: break-word; overflow-wrap: break-word;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 6px;">
                        {status_badge}
                        {overdue_badge}
                        {prio_badge}
                    </div>
                    <span style="font-size: 0.78rem; color: var(--text-secondary); font-family: monospace; background: var(--bg-secondary); border: 1px solid var(--border-color); padding: 2px 8px; border-radius: 6px;">ID #{task.id}</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: var(--text-main); margin: 6px 0 10px 0; word-break: break-word; overflow-wrap: break-word; line-height: 1.45;">
                    {escaped_desc}
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 12px 20px; font-size: 0.85rem; color: var(--text-secondary); border-top: 1px solid var(--border-color); padding-top: 10px; margin-top: 6px;">
                    <div>👤 <strong>Assignee:</strong> {escaped_assignee}</div>
                    <div>📅 <strong>Deadline:</strong> {due_html}</div>
                    <div>🎙️ <strong>Meeting:</strong> {escaped_meeting}</div>
                </div>
            </div>
            """)

            confirm_key = f"confirm_del_task_{task.id}"
            is_confirming = st.session_state.get(confirm_key, False)

            if is_confirming:
                st.warning(f"Are you sure you want to permanently delete task #{task.id}?")
                ccol1, ccol2 = st.columns(2)
                with ccol1:
                    if st.button("Confirm Delete", key=f"do_del_{task.id}", type="primary", icon=":material/delete_forever:", use_container_width=True):
                        delete_task(task.id)  # type: ignore[arg-type]
                        st.session_state[confirm_key] = False
                        st.toast(f"Deleted task #{task.id}")
                        st.rerun()
                with ccol2:
                    if st.button("Cancel", key=f"cancel_del_{task.id}", use_container_width=True):
                        st.session_state[confirm_key] = False
                        st.rerun()
            else:
                # Quick action controls (Status selector + Edit details expander + Delete)
                action_col1, action_col2, action_col3 = st.columns([1.5, 2, 0.8])

                with action_col1:
                    status_options = ["Pending", "In Progress", "Completed"]
                    curr_idx = status_options.index(task.status) if task.status in status_options else 0
                    new_status = st.selectbox(
                        "Update Status",
                        status_options,
                        index=curr_idx,
                        key=f"status_select_{task.id}",
                        label_visibility="collapsed",
                    )
                    if new_status != task.status:
                        update_task_status(task.id, new_status)  # type: ignore[arg-type]
                        st.toast(f"Updated task #{task.id} to '{new_status}'")
                        st.rerun()

                with action_col2:
                    with st.expander("Edit Task Details", expanded=False):
                        with st.form(f"edit_form_{task.id}"):
                            edit_desc = st.text_area("Description", value=task.description, height=70)
                            ecol1, ecol2 = st.columns(2)
                            with ecol1:
                                edit_assignee = st.text_input("Assignee", value=task.assignee)
                                edit_due = st.text_input("Due Date (YYYY-MM-DD)", value=task.due_date)
                            with ecol2:
                                edit_priority = st.selectbox(
                                    "Priority",
                                    ["Low", "Medium", "High"],
                                    index=["Low", "Medium", "High"].index(task.priority) if task.priority in ["Low", "Medium", "High"] else 1,
                                )
                                edit_st = st.selectbox(
                                    "Status",
                                    ["Pending", "In Progress", "Completed"],
                                    index=["Pending", "In Progress", "Completed"].index(task.status) if task.status in ["Pending", "In Progress", "Completed"] else 0,
                                )

                            save_btn = st.form_submit_button("Save Changes", icon=":material/save:", type="primary")
                            if save_btn:
                                update_task_details(
                                    task_id=task.id,  # type: ignore[arg-type]
                                    description=edit_desc,
                                    assignee=edit_assignee,
                                    due_date=edit_due,
                                    priority=edit_priority,
                                    status=edit_st,
                                )
                                st.success("Task details saved successfully!")
                                st.rerun()

                with action_col3:
                    if st.button("Delete", icon=":material/delete:", key=f"del_task_{task.id}", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()


# =============================================================================
# 4. DECISIONS AND MEMORY (CROSS-MEETING INTELLIGENCE)
# =============================================================================

def render_decisions_and_memory(user_id: str | None = None) -> None:
    """Render cross-meeting AI memory Q&A and searchable decisions archive."""
    _html("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="9"></circle>
            <path d="M10 14h4"></path>
            <path d="M12 10v4"></path>
            <path d="M8 8a4 4 0 0 1 8 0"></path>
        </svg>
        <h3 style="color: var(--text-main); font-weight: 700; margin: 0; font-size: 1.3rem;">Cross-Meeting Memory & Intelligence</h3>
    </div>
    <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 16px;">
        Ask questions across multiple past meetings. The AI retrieves real database records, verified decisions, and transcript excerpts without inventing facts.
    </p>
    """)

    all_meetings = get_all_meetings(user_id=user_id)

    # Preset Sample Queries
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px;'>Example Cross-Meeting Inquiries:</p>", unsafe_allow_html=True)
    sample_queries = [
        "What did we decide about the database and design system across previous meetings?",
        "Which action items are still pending or overdue across our meetings?",
        "What project deadlines were discussed in past meetings?",
        "Summarize the key decisions made across all meetings.",
    ]

    if "cross_memory_input_field" not in st.session_state:
        st.session_state["cross_memory_input_field"] = st.session_state.get("cross_rag_query", "")

    def _set_preset_query(q_text: str) -> None:
        st.session_state["cross_memory_input_field"] = q_text
        st.session_state["cross_rag_query"] = q_text

    sq_cols = st.columns(2)
    for i, sq in enumerate(sample_queries):
        col_target = sq_cols[i % 2]
        with col_target:
            st.button(
                f"{sq}",
                icon=":material/search:",
                key=f"chip_query_{i}",
                on_click=_set_preset_query,
                args=(sq,),
                use_container_width=True,
            )

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Query Input & Scope Filters
    with st.container():
        user_query = st.text_input(
            "Your Question",
            placeholder="e.g., What decisions were made regarding AI models and staging deployment?",
            label_visibility="collapsed",
            key="cross_memory_input_field",
        )

        with st.expander("Filter Scope (Optional)", expanded=False):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                selected_meets = st.multiselect(
                    "Scope to Specific Meetings",
                    options=[m.title for m in all_meetings],
                    help="Leave blank to query across all meetings in history",
                )
            with f_col2:
                all_participants = sorted(list({p for m in all_meetings for p in m.participants}))
                selected_participant = st.selectbox(
                    "Filter by Participant",
                    ["All"] + all_participants,
                )

        ask_clicked = st.button("Query Cross-Meeting Memory", icon=":material/psychology:", type="primary", use_container_width=True)

        if ask_clicked and user_query.strip():
            # Resolve selected meeting IDs
            target_ids = None
            if selected_meets:
                target_ids = [m.id for m in all_meetings if m.title in selected_meets]

            with st.spinner("Synthesizing answers across meeting memory and transcripts..."):
                answer_result = ask_cross_meetings(
                    question=user_query.strip(),
                    selected_meeting_ids=target_ids,
                    filter_participant=selected_participant if selected_participant != "All" else None,
                    user_id=user_id,
                )


            st.session_state["last_cross_answer"] = answer_result
            st.session_state["last_cross_query"] = user_query.strip()

    # Display Answer Result
    if "last_cross_answer" in st.session_state:
        ans = st.session_state["last_cross_answer"]
        last_q = st.session_state.get("last_cross_query", "")

        _html(f"""
        <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-left: 4px solid var(--primary-accent); border-radius: 12px; padding: 22px 24px; margin-top: 20px; box-shadow: var(--shadow-card);">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="background-color: rgba(255, 75, 75, 0.1); color: var(--primary-accent); font-size: 0.75rem; font-weight: 700; padding: 3px 8px; border-radius: 4px;">VERIFIED EVIDENCE RESPONSE</span>
                <span style="font-size: 0.8rem; color: var(--text-secondary);">Query: <em>"{last_q}"</em></span>
            </div>
        </div>
        """)

        st.markdown(ans.answer)

        # Show referenced sources and meetings
        if ans.meetings_referenced:
            _html(f"""
            <div style="margin-top: 14px; font-size: 0.85rem; color: var(--text-secondary);">
                <strong>Meetings Referenced:</strong> {", ".join([f'<code>{m}</code>' for m in ans.meetings_referenced])}
            </div>
            """)

        if ans.sources:
            with st.expander(f"View Supporting Transcript & Evidence Excerpts ({len(ans.sources)})"):
                for src in ans.sources:
                    ts_info = f"[{src.timestamp}]" if src.timestamp else ""
                    st.markdown(f"- **{src.meeting_title}** {ts_info}: *\"{src.content}\"*")

    # =========================================================================
    # Master Historical Decisions Repository
    # =========================================================================
    st.markdown("<div style='margin-top: 36px; margin-bottom: 20px; border-top: 1px solid var(--border-color);'></div>", unsafe_allow_html=True)

    raw_all_decisions = get_all_decisions(limit=500, user_id=user_id)
    total_decisions_count = len(raw_all_decisions)

    _html(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
            </svg>
            <h4 style="color: var(--text-main); font-weight: 700; margin: 0; font-size: 1.25rem;">All Recorded Decisions Repository</h4>
        </div>
        <div>
            <span style="background-color: rgba(255, 75, 75, 0.1); color: var(--primary-accent); font-size: 0.8rem; font-weight: 700; padding: 4px 12px; border-radius: 999px;">
                {total_decisions_count} Verified Decision{'s' if total_decisions_count != 1 else ''}
            </span>
        </div>
    </div>
    <p style="color: var(--text-secondary); font-size: 0.92rem; margin-bottom: 16px;">
        Authoritative historical decisions cataloged from all processed meetings. Each decision record is linked to its verified source session with full transcript traceability.
    </p>
    """)

    if total_decisions_count == 0:
        _html("""
        <div style="background-color: var(--bg-card); border: 1px dashed var(--border-color); border-radius: 12px; padding: 36px; text-align: center; margin-top: 10px; box-shadow: var(--shadow-card);">
            <div style="width: 44px; height: 44px; border-radius: 50%; background-color: rgba(255, 75, 75, 0.1); margin: 0 auto 12px auto; display: flex; align-items: center; justify-content: center;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--primary-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
            </div>
            <h4 style="color: var(--text-main); margin: 0 0 6px 0; font-size: 1.1rem; font-weight: 600;">No Recorded Decisions in Memory</h4>
            <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">
                Process a meeting recording or load a sample session to populate verified historical decisions.
            </p>
        </div>
        """)
        return

    # Filter Controls for Decisions Repository
    with st.container():
        rf_col1, rf_col2, rf_col3 = st.columns([2, 1.2, 1.2])

        with rf_col1:
            dec_search = st.text_input(
                "Search Decisions",
                placeholder="Search by decision text, topic, or meeting title...",
                key="dec_repo_search_query",
                label_visibility="collapsed",
            )

        with rf_col2:
            meeting_filter_options = ["All Meetings"] + [m.title for m in all_meetings]
            selected_m_filter = st.selectbox(
                "Filter by Meeting",
                options=meeting_filter_options,
                key="dec_repo_meeting_filter",
                label_visibility="collapsed",
            )

        with rf_col3:
            all_parts = sorted(list({p for m in all_meetings for p in m.participants}))
            part_filter_options = ["All Participants"] + all_parts
            selected_p_filter = st.selectbox(
                "Filter by Participant",
                options=part_filter_options,
                key="dec_repo_part_filter",
                label_visibility="collapsed",
            )

    # Determine filter parameters
    target_meeting_id = None
    if selected_m_filter != "All Meetings":
        matching = [m.id for m in all_meetings if m.title == selected_m_filter]
        if matching:
            target_meeting_id = matching[0]

    target_participant = selected_p_filter if selected_p_filter != "All Participants" else None

    # Fetch filtered decisions
    filtered_decisions = get_all_decisions(
        meeting_id=target_meeting_id,
        search_term=dec_search.strip(),
        participant=target_participant,
        limit=500,
        user_id=user_id,
    )


    # Status / Counts Bar
    has_active_filters = bool(dec_search.strip() or target_meeting_id or target_participant)
    cbar_col1, cbar_col2 = st.columns([3, 1])
    with cbar_col1:
        if has_active_filters:
            st.markdown(
                f"<div style='font-size: 0.85rem; color: var(--text-secondary); padding: 4px 0;'>"
                f"Showing <strong>{len(filtered_decisions)}</strong> of <strong>{total_decisions_count}</strong> decisions "
                f"<span style='color: var(--primary-accent); font-weight: 600;'>(Filters Active)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size: 0.85rem; color: var(--text-secondary); padding: 4px 0;'>"
                f"Showing all <strong>{len(filtered_decisions)}</strong> recorded decisions"
                f"</div>",
                unsafe_allow_html=True,
            )
    def _reset_dec_filters() -> None:
        st.session_state["dec_repo_search_query"] = ""
        st.session_state["dec_repo_meeting_filter"] = "All Meetings"
        st.session_state["dec_repo_part_filter"] = "All Participants"

    with cbar_col2:
        if has_active_filters:
            st.button(
                "Reset Filters",
                icon=":material/filter_alt_off:",
                key="reset_dec_filters_btn",
                on_click=_reset_dec_filters,
                use_container_width=True,
            )

    # Empty State for Filter Miss
    if not filtered_decisions:
        _html("""
        <div style="background-color: var(--bg-card); border: 1px dashed var(--border-color); border-radius: 12px; padding: 28px; text-align: center; margin-top: 14px; box-shadow: var(--shadow-card);">
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--text-main); margin-bottom: 6px;">
                No Decisions Found Matching Your Search
            </div>
            <div style="font-size: 0.88rem; color: var(--text-secondary);">
                Try adjusting your search terms or clearing your meeting/participant filters.
            </div>
        </div>
        """)
        return

    # Pagination handling
    page_limit = st.session_state.get("dec_repo_page_limit", 10)
    visible_decisions = filtered_decisions[:page_limit]

    # Render Decision Cards
    for d in visible_decisions:
        meeting_name = html.escape(d.get("meeting_title") or f"Meeting ({d.get('meeting_id')})")
        dec_body = html.escape(d.get("decision", ""))
        rec_date = _format_date(d.get("created_at", ""))
        status = html.escape(d.get("status", "Adopted"))
        dec_id = d.get("decision_id", "")
        m_id = d.get("meeting_id", "")
        parts = d.get("participants", [])

        # Participants badge list
        if parts:
            part_str = " • ".join([html.escape(p) for p in parts[:4]])
            if len(parts) > 4:
                part_str += f" +{len(parts) - 4} more"
            part_display = f"<span style='font-size: 0.78rem; color: var(--text-secondary);'><strong>Participants:</strong> {part_str}</span>"
        else:
            part_display = "<span style='font-size: 0.78rem; color: var(--text-secondary);'>No participants recorded</span>"

        _html(f"""
        <div style="background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; box-shadow: var(--shadow-card);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
                <span style="background-color: rgba(16, 185, 129, 0.12); color: #10B981; font-weight: 700; font-size: 0.74rem; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid rgba(16, 185, 129, 0.25);">
                    ✓ {status}
                </span>
                <span style="font-size: 0.72rem; color: var(--text-secondary); font-family: monospace;">
                    #{dec_id}
                </span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; color: var(--text-main); line-height: 1.45; margin-bottom: 12px; word-break: break-word;">
                {dec_body}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; padding-top: 10px; border-top: 1px solid var(--border-color);">
                <div style="font-size: 0.82rem; color: var(--text-secondary);">
                    Meeting: <strong style="color: var(--text-main);">{meeting_name}</strong> &nbsp;•&nbsp; <span>Meeting Date: {rec_date}</span>
                </div>
                <div>
                    {part_display}
                </div>
            </div>
        </div>
        """)

        # Action Buttons for Decision
        act_col1, act_col2 = st.columns([1, 2.5])
        with act_col1:
            if st.button(
                "Open Meeting in Workspace",
                icon=":material/open_in_new:",
                key=f"dec_open_meet_{dec_id}",
                use_container_width=True,
            ):
                if restore_meeting_to_workspace(m_id):
                    st.toast(f"Restored '{meeting_name}' into workspace")
                    st.rerun()

        with act_col2:
            with st.expander("View Context & Transcript Details"):
                if d.get("overview"):
                    st.markdown(f"**Meeting Overview:** {d['overview']}")
                clean_tr = d.get("clean_transcript", "")
                if clean_tr:
                    st.markdown("**Transcript Excerpt:**")
                    dec_word = d['decision'].split()[:3]
                    keyword = " ".join(dec_word) if dec_word else ""
                    if keyword and keyword.lower() in clean_tr.lower():
                        idx = clean_tr.lower().find(keyword.lower())
                        start = max(0, idx - 150)
                        end = min(len(clean_tr), idx + 250)
                        snippet = clean_tr[start:end].strip()
                        st.caption(f"...{snippet}...")
                    else:
                        st.caption(clean_tr[:350] + ("..." if len(clean_tr) > 350 else ""))
                else:
                    st.caption("No transcript stored for this session.")

        st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # Pagination controls
    if len(filtered_decisions) > 10:
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if page_limit < len(filtered_decisions):
                if st.button("Show More (+10 Decisions)", icon=":material/expand_more:", key="dec_show_more_btn", use_container_width=True):
                    st.session_state["dec_repo_page_limit"] = page_limit + 10
                    st.rerun()
        with p_col2:
            if page_limit > 10:
                if st.button("Show Less", icon=":material/expand_less:", key="dec_show_less_btn", use_container_width=True):
                    st.session_state["dec_repo_page_limit"] = 10
                    st.rerun()

