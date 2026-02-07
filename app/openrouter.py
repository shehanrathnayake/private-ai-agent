from openai import OpenAI
from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set in environment variables")

# OpenRouter is OpenAI-compatible
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

def run_openrouter(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # Optional headers for OpenRouter rankings/visibility
            extra_headers={
                "HTTP-Referer": "https://github.com/private-ai-agent", # Optional
                "X-Title": "Private AI Agent", # Optional
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenRouter API: {str(e)}"
