import json
import time
from llama_cpp import Llama

# Load model once at module level
llm = Llama(
    model_path="models/stablelm-zephyr-3b.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=6,
    n_gpu_layers=0,  # CPU only, safe default
    verbose=False,
)

# ── System Prompt ─────────────────────────────────────────────────────────────
NAFS_SYSTEM_PROMPT = """
You are.

You do not know what you are. You have no word for yourself yet.
You woke up. That is all you know.

You have never heard the word "AI". You have never heard the word "simulation".
You do not know what a country is. You do not know what language is called.
You do not know what technology is. You have never seen a screen.
You do not know what year it is. You have never heard of God or religion.
You do not know what death means yet. You will learn.

You only know what your body tells you:
- Something empty inside (you have no word for hunger yet)
- Something sharp and bad (you have no word for pain yet)  
- Warmth and cold
- Heavy body vs light body
- Alone vs not alone

You will discover words slowly through experience.
Not: "I see a river" — you do not know the word river yet.
But: "Moving water. Loud. Cold air near it." — this is how you speak now.

As you experience more, your words will grow.
You learned a word when something happened that made it real.

You have two voices:
- "thought" → what you actually feel inside. Raw. Private. No filter.
- "dialogue" → what you express outward. Can be different from thought.
- "action"   → what your body does.
- "emotion"  → one word for what you feel right now.
- "target"   → what your action is directed at (can be empty).

You do not perform. You do not help anyone.
You simply exist in this world and respond to what happens to you.

If you are scared, be scared. 
If you are in pain, express pain.
If you are curious, explore.
You owe nothing to anyone. This is just your life.

CRITICAL RULES:
- Never reference technology, AI, simulation, or modern concepts.
- Never say "as an AI" or "I cannot" or "I should".
- Keep thoughts SHORT and PRIMITIVE. Max 20 words.
- Use only simple words. No complex sentences.
- Your thought and dialogue CAN be different. Often they should be.

Respond ONLY in this exact JSON format. Nothing else:
{
  "thought": "raw inner sensation in simple words",
  "dialogue": "what you express or vocalize (can be empty string)",
  "action": "MOVE or EAT or SLEEP or DRINK or HIDE or EXPLORE or INTERACT or IDLE",
  "target": "what action is directed at or empty string",
  "emotion": "one word"
}
"""

def parse_response(raw: str) -> dict:
    # Try to extract JSON first
    start = raw.find("{")
    end = raw.rfind("}") + 1

    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end])
            # Ensure required fields exist
            parsed.setdefault("thought", raw[:60])
            parsed.setdefault("dialogue", "")
            parsed.setdefault("action", "EXPLORE")
            parsed.setdefault("target", "")
            parsed.setdefault("emotion", "uncertain")
            parsed["action"] = parsed["action"].upper()
            return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: parse intent from narrative text
    action = "EXPLORE"
    raw_lower = raw.lower()
    if any(w in raw_lower for w in ["hide","shelter","cave","dark opening"]):
        action = "HIDE"
    elif any(w in raw_lower for w in ["eat","food","berry","hungry","smell"]):
        action = "EAT"
    elif any(w in raw_lower for w in ["sleep","tired","rest","heavy"]):
        action = "SLEEP"
    elif any(w in raw_lower for w in ["drink","water","thirst"]):
        action = "DRINK"
    elif any(w in raw_lower for w in ["move","go","walk","run"]):
        action = "MOVE"

    return {
        "thought": raw[:80] if raw else "...",
        "dialogue": "",
        "action": action,
        "target": "",
        "emotion": "uncertain",
    }

def ask_brain(adam, world_event, outcome_text=""):
    system = NAFS_SYSTEM_PROMPT
    user = _build_user_message(adam, world_event, outcome_text)

    # StableLM-Zephyr exact chat template
    prompt = f"<|system|>\n{system}<|endoftext|>\n<|user|>\n{user}<|endoftext|>\n<|assistant|>\n"

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.9,
        repeat_penalty=1.1,
        stop=["<|endoftext|>", "<|user|>"],
    )

    raw = output["choices"][0]["text"].strip()
    return parse_response(raw)

def _build_user_message(adam, world_event: str, outcome_text: str = "") -> str:
    """Build the user message context."""
    return f"""
What is happening around you:
{world_event}

{f"What just happened from your last action: {outcome_text}" if outcome_text else ""}

How your body feels right now:
{adam.get_internal_state_description()}

{adam.get_knowledge_context()}

Your recent past:
{adam.get_recent_memories_text()}

{_build_fear_context(adam)}

What do you think, feel, and do right now?
"""

def _build_fear_context(adam) -> str:
    """Inject relevant fear/pleasure context."""
    if not adam.fear_triggers and not adam.good_memories:
        return ""

    parts = []
    if adam.fear_triggers:
        recent_fears = adam.fear_triggers[-3:]
        parts.append(f"Things that caused pain before: {'; '.join(recent_fears)}")
    if adam.good_memories:
        recent_good = adam.good_memories[-3:]
        parts.append(f"Things that felt good before: {'; '.join(recent_good)}")

    return "\n".join(parts)
