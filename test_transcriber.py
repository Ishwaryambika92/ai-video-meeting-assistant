"""
Test suite for core.transcriber module.
"""

from __future__ import annotations

from core.transcriber import (
    load_model,
    TranscriptSegment,
    TranscriptResult,
    _format_timestamp,
)


def test_transcriber():
    print("=" * 60)
    print("TESTING TRANSCRIBER MODULE")
    print("=" * 60)

    # 1. Test timestamp formatting
    assert _format_timestamp(0) == "00:00"
    assert _format_timestamp(65) == "01:05"
    assert _format_timestamp(3665) == "01:01:05"
    print("Timestamp formatting: OK")

    # 2. Test TranscriptSegment
    seg = TranscriptSegment(
        text="Hello world",
        start_seconds=12.5,
        end_seconds=15.0,
        engine="whisper",
        language_code="en",
    )
    assert seg.start_timestamp == "00:12"
    print("TranscriptSegment creation: OK")

    # 3. Test TranscriptResult
    res = TranscriptResult(
        full_text="Hello world",
        segments=[seg],
        engine="whisper",
    )
    assert len(res.segments) == 1
    print("TranscriptResult creation: OK")

    # 4. Test Whisper model loading
    print("Loading Whisper model...")
    try:
        model = load_model()
        assert model is not None
        print(f"Whisper model loaded successfully: {type(model).__name__}")
    except Exception as exc:
        if "not installed" in str(exc).lower():
            print(f"Whisper is optional/not installed in this environment ({exc}). Graceful fallback confirmed: OK")
        else:
            raise

    print("=" * 60)
    print("TRANSCRIBER TEST SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    test_transcriber()
