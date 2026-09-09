from core.summariser import summarize_transcript


transcript = """
Today we discussed the AI Video Assistant project.

The team decided to use Python and Streamlit for the application.
Whisper will be used for English transcription and Sarvam AI will
be used for Hindi and Hinglish transcription.

Deepak will complete the backend integration by Friday.
Rahul will work on the Streamlit interface.

The team still needs to decide which database should be used
for storing application data.
"""


try:
    result = summarize_transcript(transcript)

    print("=" * 60)
    print("SUMMARISER TEST SUCCESS")
    print("=" * 60)

    print("\nTITLE:")
    print(result.title)

    print("\nOVERVIEW:")
    print(result.overview)

    print("\nSUMMARY:")
    print(result.summary)

    print("\nIMPORTANT POINTS:")
    for point in result.important_points:
        print("-", point)

    print("\nKEY TAKEAWAYS:")
    for takeaway in result.key_takeaways:
        print("-", takeaway)

    print("=" * 60)

except Exception as e:
    print("=" * 60)
    print("SUMMARISER TEST FAILED")
    print("=" * 60)
    print(e)