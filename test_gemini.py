from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found")
    exit()

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain artificial intelligence in one simple sentence."
)

print("=" * 60)
print("GEMINI TEST SUCCESS")
print("=" * 60)
print(response.text)