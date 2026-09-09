"""
Simple end-to-end smoke test for the audio -> transcription pipeline.

Run:

python test.py "https://www.youtube.com/watch?v=3PzKNC1jqkI" --language hinglish

or:

python test.py "path/to/local/file.mp4" --language english
"""

from __future__ import annotations

import argparse

from core.transcriber import transcribe_all
from utils.audio_processor import cleanup_chunks, process_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the transcription pipeline."
    )

    parser.add_argument(
        "source",
        help="YouTube URL or local audio/video file path",
    )

    parser.add_argument(
        "--language",
        default="english",
        choices=["english", "hinglish"],
        help="Transcription language/engine",
    )

    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=10,
        help="Audio processing chunk size in minutes",
    )

    args = parser.parse_args()

    chunks = process_input(
        args.source,
        chunk_minutes=args.chunk_minutes,
    )

    try:
        result = transcribe_all(
            chunks,
            language=args.language,
            chunk_seconds=30,
        )

        print("=" * 60)
        print(f"Engine: {result.engine}")
        print(f"Segments: {len(result.segments)}")
        print("=" * 60)

        print(result.full_text)

        print("=" * 60)

    finally:
        cleanup_chunks(chunks)


if __name__ == "__main__":
    main()