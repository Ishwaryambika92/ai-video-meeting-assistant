"""
Build downloadable TXT and PDF reports from a processed session.

Kept dependency-light and presentation-agnostic: this module only builds
bytes/strings, the Streamlit UI decides how to offer them for download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

from core.extractor import ExtractionResult
from core.summariser import MeetingSummary


def _safe_pdf_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "•": "-",
        "→": "->",
        "←": "<-",
        "✓": "[OK]",
        "✔": "[OK]",
        "⚠": "[WARNING]",
        "🚀": "",
        "⬇": "",
        "📌": "",
        "🎯": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("latin-1", errors="replace").decode("latin-1")

@dataclass
class ReportData:
    title: str
    summary: MeetingSummary
    extraction: ExtractionResult
    transcript: str
    original_transcript: str | None = None
    generated_at: datetime = field(default_factory=datetime.now)


def build_txt_report(data: ReportData) -> str:
    lines: list[str] = []
    lines.append(f"AI VIDEO/MEETING REPORT")
    lines.append(f"Generated: {data.generated_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"TITLE: {data.summary.title}")
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 70)
    lines.append(data.summary.overview)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(data.summary.summary)
    lines.append("")

    lines.append("IMPORTANT POINTS")
    lines.append("-" * 70)
    lines += [f"- {p}" for p in data.summary.important_points] or ["(none)"]
    lines.append("")

    lines.append("KEY TAKEAWAYS")
    lines.append("-" * 70)
    lines += [f"- {t}" for t in data.summary.key_takeaways] or ["(none)"]
    lines.append("")

    lines.append("ACTION ITEMS")
    lines.append("-" * 70)
    if data.extraction.action_items:
        for item in data.extraction.action_items:
            lines.append(f"- {item.action}")
            lines.append(f"    Owner: {item.owner} | Deadline: {item.deadline} | Status: {item.status}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("DECISIONS")
    lines.append("-" * 70)
    lines += [f"- {d}" for d in data.extraction.decisions] or ["(none)"]
    lines.append("")

    lines.append("OPEN QUESTIONS")
    lines.append("-" * 70)
    lines += [f"- {q}" for q in data.extraction.open_questions] or ["(none)"]
    lines.append("")

    lines.append("FOLLOW-UPS")
    lines.append("-" * 70)
    lines += [f"- {f}" for f in data.extraction.follow_ups] or ["(none)"]
    lines.append("")

    if data.original_transcript and data.original_transcript.strip() != data.transcript.strip():
        lines.append("ORIGINAL TRANSCRIPT")
        lines.append("-" * 70)
        lines.append(data.original_transcript)
        lines.append("")

    lines.append("CLEAN TRANSCRIPT")
    lines.append("-" * 70)
    lines.append(data.transcript)

    return "\n".join(lines)


def _add_section(
    pdf: FPDF,
    heading: str,
    body_lines: list[str],
) -> None:

    # Always start from left margin
    pdf.set_x(pdf.l_margin)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(
        0,
        8,
        _safe_pdf_text(heading),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Helvetica", "", 10)

    # Explicit usable page width
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    for line in body_lines or ["(none)"]:

        text = _safe_pdf_text(line)

        # Split extremely long words/URLs
        chunks = []

        while len(text) > 70:
            part = text[:70]

            break_pos = part.rfind(" ")

            if break_pos > 0:
                part = text[:break_pos]
                text = text[break_pos + 1:]
            else:
                part = text[:60]
                text = text[60:]

            chunks.append(part)

        if text:
            chunks.append(text)

        for chunk in chunks:
            pdf.set_x(pdf.l_margin)

            pdf.multi_cell(
                usable_width,
                5,
                chunk,
                new_x="LMARGIN",
                new_y="NEXT",
            )

        pdf.ln(1)

    pdf.ln(3)


def build_pdf_report(data: ReportData) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_x(pdf.l_margin)

    pdf.multi_cell(
    pdf.w - pdf.l_margin - pdf.r_margin,
    10,
    _safe_pdf_text(data.summary.title),
    new_x="LMARGIN",
    new_y="NEXT",
)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(pdf.l_margin)

    pdf.multi_cell(
    pdf.w - pdf.l_margin - pdf.r_margin,
    6,
    _safe_pdf_text(
        f"Generated: {data.generated_at.strftime('%Y-%m-%d %H:%M')}"
    ),
    new_x="LMARGIN",
    new_y="NEXT",
    )

    pdf.ln(4)

    # Summary sections
    _add_section(
        pdf,
        "Overview",
        [data.summary.overview],
    )

    _add_section(
        pdf,
        "Summary",
        [data.summary.summary],
    )

    _add_section(
        pdf,
        "Important Points",
        [f"- {p}" for p in data.summary.important_points],
    )

    _add_section(
        pdf,
        "Key Takeaways",
        [f"- {t}" for t in data.summary.key_takeaways],
    )

    # Action Items
    action_lines = []

    for item in data.extraction.action_items:
        action_lines.append(f"- {item.action}")
        action_lines.append(
            f"  Owner: {item.owner} | "
            f"Deadline: {item.deadline} | "
            f"Status: {item.status}"
        )

    _add_section(
        pdf,
        "Action Items",
        action_lines,
    )

    _add_section(
        pdf,
        "Decisions",
        [f"- {d}" for d in data.extraction.decisions],
    )

    _add_section(
        pdf,
        "Open Questions",
        [f"- {q}" for q in data.extraction.open_questions],
    )

    _add_section(
        pdf,
        "Follow-ups",
        [f"- {f}" for f in data.extraction.follow_ups],
    )

    # Clean transcript
    pdf.add_page()

    transcript_lines = data.transcript.splitlines()

    _add_section(
        pdf,
        "Clean Transcript",
        transcript_lines,
    )

    return bytes(pdf.output())

def build_jpg_report(data: ReportData) -> bytes:
    """Build a compact shareable JPG report."""

    width = 1200
    padding = 60

    title = data.summary.title
    overview = data.summary.overview

    important_points = data.summary.important_points[:5]
    key_takeaways = data.summary.key_takeaways[:5]
    action_items = data.extraction.action_items[:5]
    decisions = data.extraction.decisions[:5]

    # Try common Windows fonts.
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]

    bold_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]

    font_path = next((p for p in font_paths if Path(p).exists()), None)
    bold_path = next((p for p in bold_paths if Path(p).exists()), None)

    if font_path:
        body_font = ImageFont.truetype(font_path, 28)
        small_font = ImageFont.truetype(font_path, 24)
    else:
        body_font = ImageFont.load_default()
        small_font = body_font

    if bold_path:
        heading_font = ImageFont.truetype(bold_path, 34)
        title_font = ImageFont.truetype(bold_path, 42)
    else:
        heading_font = body_font
        title_font = body_font

    sections = []

    def add_text(text: str, font):
        sections.append((text, font))

    add_text("AI VIDEO / MEETING REPORT", title_font)
    add_text(title, heading_font)
    add_text("OVERVIEW", heading_font)
    add_text(overview, body_font)

    if important_points:
        add_text("IMPORTANT POINTS", heading_font)
        for point in important_points:
            add_text(f"• {point}", body_font)

    if action_items:
        add_text("ACTION ITEMS", heading_font)
        for item in action_items:
            add_text(
                f"• {item.action} | Owner: {item.owner} | Deadline: {item.deadline}",
                body_font,
            )

    if decisions:
        add_text("DECISIONS", heading_font)
        for decision in decisions:
            add_text(f"• {decision}", body_font)

    if key_takeaways:
        add_text("KEY TAKEAWAYS", heading_font)
        for takeaway in key_takeaways:
            add_text(f"• {takeaway}", body_font)

    # Calculate approximate height.
    height = 180

    for text, font in sections:
        bbox = ImageDraw.Draw(
            Image.new("RGB", (1, 1))
        ).textbbox((0, 0), text, font=font)

        text_width = max(1, bbox[2] - bbox[0])
        lines = max(1, (text_width // (width - 2 * padding)) + 1)

        height += lines * 45 + 25

    height = max(height, 900)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = padding

    for text, font in sections:
        max_width = width - 2 * padding

        # Simple word wrapping.
        words = text.split()
        current = ""

        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)

            if bbox[2] - bbox[0] <= max_width:
                current = test
            else:
                if current:
                    draw.text((padding, y), current, font=font, fill="black")
                    y += 45
                current = word

        if current:
            draw.text((padding, y), current, font=font, fill="black")
            y += 45

        y += 20

    # Crop unused space.
    image = image.crop((0, 0, width, min(y + padding, height)))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)

    return output.getvalue()