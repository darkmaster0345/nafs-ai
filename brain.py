import json
import time
from groq import Groq
from config import BRAIN_CONFIG

client = Groq(api_key=BRAIN_CONFIG["api_key"])

# ── System Prompt ─────────────────────────────────────────────────────────────
# This is the soul of the experiment.
# Adam is not told he is an AI.
# Adam is not told this is a simulation.
# Adam is given only sensation.

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


def ask_brain(adam, world_event: str, outcome_text: str = "") -> dict:
    """
    Send Adam's current state to the LLM.
    Returns parsed response dict.
    """

    # Build the user message
    user_message = f"""
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

    # Try thought model first, fall back to action model
    for model in [BRAIN_CONFIG["thought_model"], BRAIN_CONFIG["action_model"]]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NAFS_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=BRAIN_CONFIG["max_tokens"],
                temperature=BRAIN_CONFIG["temperature"],
            )

            raw = response.choices[0].message.content.strip()
            parsed = _parse_response(raw)

            if parsed:
                return parsed

        except Exception as e:
            print(f"[Brain] Model {model} failed: {e}")
            time.sleep(2)

    # Fallback if everything fails
    return {
        "thought":  "confused. nothing working.",
        "dialogue": "",
        "action":   "IDLE",
        "target":   "",
        "emotion":  "confused",
    }


def _parse_response(raw: str) -> dict | None:
    """Parse JSON from LLM response. Handles messy output."""
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Find first { and last }
    start = raw.find("{")
    end   = raw.rfind("}") + 1

    if start == -1 or end == 0:
        return None

    try:
        parsed = json.loads(raw[start:end])

        # Validate required fields
        required = ["thought", "dialogue", "action", "emotion"]
        for field in required:
            if field not in parsed:
                parsed[field] = "?"

        # Normalize action to uppercase
        parsed["action"] = parsed.get("action", "IDLE").upper()

        # Ensure target exists
        if "target" not in parsed:
            parsed["target"] = ""

        return parsed

    except json.JSONDecodeError:
        return None


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
