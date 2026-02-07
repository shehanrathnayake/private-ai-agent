import google.generativeai as genai
from app.config import GEMINI_API_KEY

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Use gemini-1.5-flash as default for better speed and reliability in free tier
# or gemini-pro if preferred by user's original plan. 
# I'll stick to gemini-1.5-flash for better free-tier experience.
model = genai.GenerativeModel("gemini-2.0-flash")

def run_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"
