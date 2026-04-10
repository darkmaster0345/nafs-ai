import os
from dotenv import load_dotenv

load_dotenv()

# ── Brain Configuration ───────────────────────────────────────────────────────
BRAIN_CONFIG = {
    "provider":       "groq",
    "thought_model":  "llama-3.3-70b-versatile",   # deep — for inner thoughts
    "action_model":   "llama-3.1-8b-instant",       # fast — for world actions
    "api_key":        os.getenv("GROQ_API_KEY", ""),
    "poll_interval":  10,    # seconds between simulation ticks
    "max_tokens":     200,   # keep responses short and primitive
    "temperature":    0.9,   # slight randomness = personality variation
}

# ── Simulation Configuration ──────────────────────────────────────────────────
SIM_CONFIG = {
    "memory_file":        "memory.json",
    "short_term_limit":   10,    # last N memories injected into prompt
    "long_term_limit":    200,   # max memories before summarization
    "hunger_rate":        2,     # hunger increase per tick
    "energy_drain":       1,     # energy decrease per tick
    "health_regen":       1,     # health regen when full and rested
    "tick_display":       True,  # show tick number in terminal
}

# ── Starting Vocabulary ───────────────────────────────────────────────────────
# Adam knows only these words at birth.
# New words are added through experience.
STARTING_VOCABULARY = [
    "hot", "cold", "pain", "good", "bad",
    "full", "empty", "tired", "awake",
    "big", "small", "near", "far",
    "light", "dark", "wet", "dry",
    "loud", "quiet", "soft", "hard",
    "go", "stop", "eat", "sleep",
    "touch", "run", "hide",
    "alone", "other", "safe", "danger",
    "want", "not", "here", "there",
]

# ── Forbidden Knowledge ───────────────────────────────────────────────────────
# If Adam produces these concepts unprompted,
# the response is flagged and regenerated.
FORBIDDEN_CONCEPTS = [
    "technology", "country", "religion", "science",
    "simulation", "civilization", "internet", "phone",
    "artificial", "language model", "AI", "assistant",
    "programmed", "computer", "robot",
]
