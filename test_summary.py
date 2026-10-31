import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from google import genai
from config import settings

client = genai.Client(api_key=settings.gemini_api_key)

transcript = """
Today we discussed our AI Video Assistant project.

The team decided to use Python and Streamlit for the application.
Whisper will be used for English transcription and Sarvam AI will
be used for Hindi and Hinglish transcription.

Deepak will complete the backend integration by Friday.
Rahul will work on the Streamlit interface.

The team still needs to decide which database should be used
for storing application data.
"""

prompt = f"""
You are an AI meeting assistant.

Summarize the following transcript in a clear and concise way.

Give the output in this format:

SUMMARY:
- Point 1
- Point 2
- Point 3

KEY DECISIONS:
- Decision 1
- Decision 2

ACTION ITEMS:
- Person: Task - Deadline

OPEN QUESTIONS:
- Question 1

TRANSCRIPT:
{transcript}
"""

response = client.models.generate_content(
    model=settings.gemini_model,
    contents=prompt,
)

print("=" * 60)
print("GEMINI SUMMARY TEST SUCCESS")
print("=" * 60)
print(response.text)
print("=" * 60)