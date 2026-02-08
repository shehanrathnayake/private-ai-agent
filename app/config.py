import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Answer clearly and briefly.")
SUMMARY_THRESHOLD = int(os.getenv("SUMMARY_THRESHOLD", 5))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "memory/vector_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
TOP_K_ASSOCIATIVE = int(os.getenv("TOP_K_ASSOCIATIVE", 3))
IDENTITY_UPDATE_INTERVAL = int(os.getenv("IDENTITY_UPDATE_INTERVAL", 10))
