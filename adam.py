import json
import os
from config import STARTING_VOCABULARY, SIM_CONFIG, FORBIDDEN_CONCEPTS


class Adam:
    def __init__(self):
        # ── Internal State ────────────────────────────────────────────────────
        self.health      = 100
        self.hunger      = 10    # starts low — just woke up
        self.energy      = 80
        self.stress      = 5

        # ── Identity ──────────────────────────────────────────────────────────
        self.name        = None   # he doesn't have a name yet. he will find one.
        self.age_ticks   = 0
        self.is_alive    = True

        # ── Vocabulary & Knowledge ────────────────────────────────────────────
        self.vocabulary        = STARTING_VOCABULARY.copy()
        self.concepts_learned  = {}   # {"fire": "causes pain", "berry": "reduces empty feeling"}

        # ── Memory ────────────────────────────────────────────────────────────
        self.short_term  = []   # last N experiences
        self.long_term   = []   # all experiences

        # ── Personality (emerges, not assigned) ──────────────────────────────
        self.fear_triggers    = []   # things that caused pain
        self.good_memories    = []   # things that caused pleasure
        self.current_emotion  = "uncertain"
        self.last_action      = "IDLE"
        self.last_thought     = ""

        # ── Load existing memory if present ──────────────────────────────────
        self._load_memory()

    # ── State Management ─────────────────────────────────────────────────────

    def apply_time_passage(self, hunger_rate: int, energy_drain: int):
        """Called every tick — natural stat decay."""
        self.hunger    = min(100, self.hunger + hunger_rate)
        self.energy    = max(0, self.energy - energy_drain)
        self.age_ticks += 1

        # Hunger causes health loss when critical
        if self.hunger >= 90:
            self.health = max(0, self.health - 3)
            self.stress = min(100, self.stress + 5)

        # Exhaustion causes stress
        if self.energy <= 10:
            self.stress = min(100, self.stress + 3)

        # Stress recovery logic
        if self.hunger < 40 and self.health > 70:
            self.stress = max(0, self.stress - 3)

        # Natural death
        if self.health <= 0:
            self.is_alive = False

    def apply_outcome(self, outcome: dict):
        """Apply world outcome to Adam's state."""
        self.health = max(0, min(100, self.health + outcome.get("health_delta", 0)))
        self.hunger = max(0, min(100, self.hunger + outcome.get("hunger_delta", 0)))
        self.energy = max(0, min(100, self.energy + outcome.get("energy_delta", 0)))

        # Sleep-based stress reduction
        outcome_text = outcome.get("outcome_text", "").lower()
        if "light again" in outcome_text or "rest" in outcome_text:
            self.stress = max(0, self.stress - 20)

        if self.health <= 0:
            self.is_alive = False

    def get_internal_state_description(self) -> str:
        """Convert numeric states into primitive sensory description."""
        parts = []

        if self.hunger > 70:
            parts.append("Very strong empty feeling inside. Painful.")
        elif self.hunger > 40:
            parts.append("Empty inside. Twisting.")
        elif self.hunger < 15:
            parts.append("Inside feels full.")

        if self.energy < 20:
            parts.append("Body very heavy. Hard to move.")
        elif self.energy < 50:
            parts.append("Tired.")
        elif self.energy > 80:
            parts.append("Body feels ready.")

        if self.health < 30:
            parts.append("Pain. Body not working well.")
        elif self.health < 60:
            parts.append("Something not right in body.")

        if self.stress > 60:
            parts.append("Something wrong. Scared feeling.")
        elif self.stress > 30:
            parts.append("Uneasy.")

        return " ".join(parts) if parts else "Body feels normal."

    # ── Memory System ─────────────────────────────────────────────────────────

    def remember(self, tick: int, event: str, thought: str,
        action: str, emotion: str, outcome: str):
        """Store a new memory."""
        memory = {
            "tick":    tick,
            "event":   event,
            "thought": thought,
            "action":  action,
            "emotion": emotion,
            "outcome": outcome,
        }

        self.short_term.append(memory)
        self.long_term.append(memory)

        # Track fear triggers
        if "pain" in outcome.lower() or "hurt" in outcome.lower() or "wrong" in outcome.lower():
            if len(event) < 80:
                self.fear_triggers.append(event[:60])

        # Track good memories
        if "good" in outcome.lower() or "full" in outcome.lower() or "warm" in outcome.lower():
            if len(outcome) < 80:
                self.good_memories.append(outcome[:60])

        # Trim short term to limit
        if len(self.short_term) > SIM_CONFIG["short_term_limit"]:
            self.short_term.pop(0)

        # Save to file
        self._save_memory()

    def get_recent_memories_text(self) -> str:
        """Format short-term memory for prompt injection."""
        if not self.short_term:
            return "No memories yet."

        lines = []
        for m in self.short_term[-6:]:   # last 6 only
            lines.append(
                f"[Tick {m['tick']}] Happened: {m['event'][:60]} | "
                f"I did: {m['action']} | Felt: {m['emotion']} | "
                f"Result: {m['outcome'][:60]}"
            )
        return "\n".join(lines)

    def learn_concept(self, word: str, meaning: str):
        """Adam learns a new word from experience."""
        if word not in self.vocabulary:
            self.vocabulary.append(word)
        self.concepts_learned[word] = meaning

    def get_knowledge_context(self) -> str:
        """Build known concepts text for prompt."""
        if not self.concepts_learned:
            return "You have learned nothing with words yet."
        lines = [f"'{k}' means '{v}'" for k, v in list(self.concepts_learned.items())[-10:]]
        return "Things you have learned:\n" + "\n".join(lines)

    # ── Violation Detection ───────────────────────────────────────────────────

    def response_is_clean(self, response: dict) -> bool:
        """Check if response contains forbidden modern knowledge."""
        text = json.dumps(response).lower()
        for concept in FORBIDDEN_CONCEPTS:
            if concept.lower() in text:
                return False
        return True

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_memory(self):
        data = {
            "age_ticks":        self.age_ticks,
            "vocabulary":       self.vocabulary,
            "concepts_learned": self.concepts_learned,
            "fear_triggers":    self.fear_triggers[-20:],
            "good_memories":    self.good_memories[-20:],
            "long_term":        self.long_term[-SIM_CONFIG["long_term_limit"]:],
            "stats": {
                "health": self.health,
                "hunger": self.hunger,
                "energy": self.energy,
                "stress": self.stress,
            }
        }
        with open(SIM_CONFIG["memory_file"], "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_memory(self):
        if not os.path.exists(SIM_CONFIG["memory_file"]):
            return

        try:
            with open(SIM_CONFIG["memory_file"], "r", encoding="utf-8") as f:
                data = json.load(f)

            self.age_ticks        = data.get("age_ticks", 0)
            self.vocabulary       = data.get("vocabulary", STARTING_VOCABULARY.copy())
            self.concepts_learned = data.get("concepts_learned", {})
            self.fear_triggers    = data.get("fear_triggers", [])
            self.good_memories    = data.get("good_memories", [])
            self.long_term        = data.get("long_term", [])
            self.short_term       = self.long_term[-SIM_CONFIG["short_term_limit"]:]

            stats = data.get("stats", {})
            self.health = stats.get("health", 100)
            self.hunger = stats.get("hunger", 10)
            self.energy = stats.get("energy", 80)
            self.stress = stats.get("stress", 5)

            print(f"[Memory] Loaded. Adam has {len(self.long_term)} memories. Age: {self.age_ticks} ticks.")

        except Exception as e:
            print(f"[Memory] Could not load: {e}. Starting fresh.")

    def status_line(self) -> str:
        return (
            f"Health:{self.health:>3}  "
            f"Hunger:{self.hunger:>3}  "
            f"Energy:{self.energy:>3}  "
            f"Stress:{self.stress:>3}  "
            f"Emotion: {self.current_emotion}"
        )
