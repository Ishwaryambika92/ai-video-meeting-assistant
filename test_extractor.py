from core.extractor import extract_action_items
transcript = """
Today we discussed the AI Video Assistant project.

The team decided to use Python and Streamlit for the application.

Deepak will complete the backend integration by Friday.
Rahul will work on the Streamlit interface by Monday.

The team decided that Whisper will be used for English
transcription and Sarvam AI will be used for Hindi and
Hinglish transcription.

We still need to decide which database should be used
for storing application data.

The team will discuss the database selection again next week.
"""


try:
    result = extract_action_items(transcript)

    print("=" * 60)
    print("EXTRACTOR TEST SUCCESS")
    print("=" * 60)

    print("\nACTION ITEMS:")

    for item in result.action_items:
        print(f"Task     : {item.action}")
        print(f"Owner    : {item.owner}")
        print(f"Deadline : {item.deadline}")
        print(f"Status   : {item.status}")
        print("-" * 40)

    print("\nDECISIONS:")

    for decision in result.decisions:
        print("-", decision)

    print("\nOPEN QUESTIONS:")

    for question in result.open_questions:
        print("-", question)

    print("\nFOLLOW-UPS:")

    for follow_up in result.follow_ups:
        print("-", follow_up)

    print("=" * 60)

except Exception as e:
    print("=" * 60)
    print("EXTRACTOR TEST FAILED")
    print("=" * 60)
    print(e)