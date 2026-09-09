from core.vector_store import index_transcript
from core.transcriber import TranscriptSegment


segments = [
    TranscriptSegment(
        text="The team decided to use Python and Streamlit for the AI Video Assistant.",
        start_seconds=0,
        end_seconds=30,
        engine="Sarvam AI",
        language_code="en",
    ),

    TranscriptSegment(
        text="Deepak will complete the backend integration by Friday.",
        start_seconds=30,
        end_seconds=60,
        engine="Sarvam AI",
        language_code="en",
    ),

    TranscriptSegment(
        text="The team still needs to decide which database should be used.",
        start_seconds=60,
        end_seconds=90,
        engine="Sarvam AI",
        language_code="en",
    ),
]


try:
    store = index_transcript(
        session_id="test_session",
        segments=segments,
    )

    print("=" * 60)
    print("VECTOR STORE TEST SUCCESS")
    print("=" * 60)

    data = store.get()

    print(f"Documents stored: {len(data.get('ids', []))}")
    print("ChromaDB vector store created successfully.")

    print("=" * 60)

except Exception as e:
    print("=" * 60)
    print("VECTOR STORE TEST FAILED")
    print("=" * 60)
    print(e)