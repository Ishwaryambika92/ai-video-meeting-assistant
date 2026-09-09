from mistralai.client import Mistral

from config import settings

client = Mistral(api_key=settings.mistral_api_key)

response = client.chat.complete(
    model="mistral-small-2603",
    messages=[
        {
            "role": "user",
            "content": "Explain artificial intelligence in one sentence."
        }
    ],
)

print("=" * 60)
print("MISTRAL TEST SUCCESS")
print("=" * 60)

print(response.choices[0].message.content)