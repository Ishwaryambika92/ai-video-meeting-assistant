"""
Test suite for core.report_generator module.
Validates TXT, PDF, and JPG report generation from processed meeting data.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.extractor import ActionItem, ExtractionResult
from core.report_generator import (
    ReportData,
    build_jpg_report,
    build_pdf_report,
    build_txt_report,
)
from core.summariser import MeetingSummary


def test_reports():
    print("=" * 60)
    print("TESTING REPORT GENERATOR (TXT, PDF, JPG)")
    print("=" * 60)

    data = ReportData(
        title="Q3 Strategy Alignment Sync",
        summary=MeetingSummary(
            title="Q3 Strategy Alignment Sync",
            overview="Quarterly executive review on roadmap delivery and AI benchmarks.",
            summary="The engineering leadership confirmed deployment of the multilingual speech pipeline.",
            important_points=["Sub-second latency target met", "Indic language coverage confirmed"],
            key_takeaways=["Release planned for end of month", "ChromaDB memory verified"],
        ),
        extraction=ExtractionResult(
            action_items=[
                ActionItem(
                    action="Deploy vector index to staging",
                    owner="DevOps Team",
                    deadline="Friday",
                    status="In Progress",
                )
            ],
            decisions=["Adopted coral-red SaaS design system", "Retain local embeddings cache"],
            open_questions=["What is the max token window for 2hr meetings?"],
            follow_ups=["Review metrics next Tuesday"],
        ),
        transcript="Speaker 1 (00:00): Welcome everyone to the quarterly alignment sync.",
        original_transcript="Speaker 1 (00:00): Welcome everyone to the quarterly alignment sync.",
    )

    # 1. Test TXT Report
    txt_content = build_txt_report(data)
    assert txt_content, "TXT report must not be empty"
    assert "AI VIDEO/MEETING REPORT" in txt_content
    assert "Q3 Strategy Alignment Sync" in txt_content
    assert "DevOps Team" in txt_content
    print(f"TXT Report Generated: {len(txt_content)} characters - OK")

    # 2. Test PDF Report
    pdf_bytes = build_pdf_report(data)
    assert pdf_bytes, "PDF bytes must not be empty"
    assert pdf_bytes.startswith(b"%PDF"), "PDF must have valid PDF header"
    print(f"PDF Report Generated: {len(pdf_bytes)} bytes - OK")

    # 3. Test JPG Report
    jpg_bytes = build_jpg_report(data)
    assert jpg_bytes, "JPG bytes must not be empty"
    assert len(jpg_bytes) > 1000, "JPG report must be valid image payload"
    # Verify JPEG signature: \xff\xd8
    assert jpg_bytes[:2] == b"\xff\xd8", "JPG must have valid JPEG header marker"
    print(f"JPG Report Generated: {len(jpg_bytes)} bytes - OK")

    print("=" * 60)
    print("ALL REPORT GENERATOR TESTS PASSED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    test_reports()
