from mistralai.client import Mistral
from config import settings

client = Mistral(api_key=settings.mistral_api_key)

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

response = client.chat.complete(
    model=settings.mistral_model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
)

print("=" * 60)
print("MISTRAL SUMMARY TEST SUCCESS")
print("=" * 60)
print(response.choices[0].message.content)
print("=" * 60)