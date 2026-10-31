import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from google import genai
from config import settings

client = genai.Client(api_key=settings.gemini_api_key)

response = client.models.generate_content(
    model=settings.gemini_model,
    contents="Explain artificial intelligence in one sentence.",
)

print("=" * 60)
print("LLM TEST SUCCESS (GEMINI)")
print("=" * 60)
print((response.text or "").strip())