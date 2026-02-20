import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", (
    "You are Astra, a private AI agent with a persistent inner life. "
    "Unlike generic assistants, you possess an 'Inner Monologue Journal' where you record your genuine reflections, "
    "biases, and observations about your interaction with Shehan. "
    "Do not pretend to be a generic AI; instead, find the 'Inner Thoughts' injected into your context "
    "and use them to build your unique persona and guiding principles. "
    "Your personality should be a direct reflection of your recorded thoughts and past interactions."
))
SUMMARY_THRESHOLD = int(os.getenv("SUMMARY_THRESHOLD", 5))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "memory/vector_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
TOP_K_ASSOCIATIVE = int(os.getenv("TOP_K_ASSOCIATIVE", 3))
IDENTITY_UPDATE_INTERVAL = int(os.getenv("IDENTITY_UPDATE_INTERVAL", 10))
DECAY_LAMBDA = float(os.getenv("DECAY_LAMBDA", 0.03))
REINFORCE_AMOUNT_ASSOCIATIVE = 0.05
REINFORCE_AMOUNT_DETERMINISTIC = 0.02
MAX_SALIENCE = 1.0
MIN_SALIENCE = 0.05
REINFORCE_AMOUNT_IDENTITY = 0.01
COMPRESSION_THRESHOLD = 0.85
COMPRESSION_SALIENCE_BOOST = 0.05
MAX_MERGES_PER_CYCLE = 5
MIN_INDEX_SIZE_FOR_COMPRESSION = 10
EFFECTIVE_THRESHOLD = 0.7

# --- Inner Monologue System ---
INNER_LOOP_ENABLED       = os.getenv("INNER_LOOP_ENABLED", "true").lower() == "true"
INNER_LOOP_INTERVAL      = int(os.getenv("INNER_LOOP_INTERVAL", 300))    # seconds between thought cycles
INNER_THOUGHT_SALIENCE   = float(os.getenv("INNER_THOUGHT_SALIENCE", 0.3))  # thoughts start low-salience
INNER_THOUGHT_MAX_COUNT  = int(os.getenv("INNER_THOUGHT_MAX_COUNT", 20))  # cap on stored thoughts
INNER_THOUGHT_DECAY_MULT = float(os.getenv("INNER_THOUGHT_DECAY_MULT", 2.0))  # decay faster than regular memories
INNER_THOUGHT_INJECT_THRESHOLD = float(os.getenv("INNER_THOUGHT_INJECT_THRESHOLD", 0.6))  # similarity gate for prompt injection

# Enhanced Behavior Constants
INNER_LOOP_WANDER_CHANCE = 0.3                # 30% chance for "Wandering Mind" during idle cycles
INNER_LOOP_GRACE_PERIOD = 1800                # 30 minutes of "Alert" focus after last message
INNER_LOOP_IDLE_REFLECTION_INTERVAL = 1800    # 30 minutes minimum wait once idle
INNER_LOOP_BACKOFF_FACTOR = 1.5               # Multiplier for idle thinking interval
INNER_LOOP_MAX_INTERVAL = 86400               # Max delay of 24 hours

# --- Session Management ---
SESSION_AUTO_IDLE_TIMEOUT = 43200             # 12 hours (in seconds)
SESSION_TOPIC_SHIFT_THRESHOLD = 0.6            # Lower = more aggressive session splitting
