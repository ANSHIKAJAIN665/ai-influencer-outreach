import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say hello in one short sentence."
)

print("Gemini response:")
print(response.text)