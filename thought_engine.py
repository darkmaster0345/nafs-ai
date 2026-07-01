"""
Nafs AI — Thought Engine (Phase 1: The Inner Voice)
"What does Adam feel?"

Generates primitive thoughts and emotions from raw sensory experience.
No LLM. No pretrained models. No external knowledge.

The thought IS the sensation, expressed in Adam's limited vocabulary.
The emotion IS the bodily state, classified into a single word.

Philosophy:
  - Thoughts are not "generated" by a model — they ARE the raw sensations
  - The vocabulary constraint is key: Adam can only think in words he knows
  - As vocabulary grows (Phase 2), thoughts become richer — naturally
  - Emotions are direct bodily responses, not learned patterns
  - The gap between thought and dialogue (Phase 3) IS the subconscious
"""

import random


# ═══════════════════════════════════════════════════════════════════════════════
# World Event Describer — Converts numeric world state to primitive text
# ═══════════════════════════════════════════════════════════════════════════════

def describe_world_event(world_state: dict, adam_stats: dict) -> str:
    """
    Generate a primitive text description of the current world moment.

    This is what Adam "sees" and "feels" from the environment.
    Uses only sensory language — no concepts, no naming, no interpretation.
    Now includes biome and weather descriptions.

    Returns: str — primitive description of the current moment
    """
    parts = []

    # Biome description (primitive — Adam describes what he feels)
    biome = world_state.get('biome', '')
    if biome == 'desert':
        parts.append("Sand. Dry ground. Hot wind.")
    elif biome == 'forest':
        parts.append("Trees. Green. Cool shade.")
    elif biome == 'tundra':
        parts.append("White ground. Freezing. Nothing grows.")
    elif biome == 'plains':
        parts.append("Open grass. Wind. Can see far.")
    elif biome == 'mountain':
        parts.append("Steep rocks. Thin air. Cold.")
    elif biome == 'swamp':
        parts.append("Wet mud. Smell of rot. Bugs.")
    elif biome == 'ocean':
        parts.append("Water everywhere. Waves. Salt.")
    elif biome == 'jungle':
        parts.append("Dense green. Humid. Something watching.")
    elif biome == 'cave':
        parts.append("Dark stone. Echoes. Dripping.")
    elif biome == 'volcano':
        parts.append("Hot ground. Smoke. Smell of fire.")

    # Weather description
    weather = world_state.get('weather', '')
    if weather == 'rain':
        parts.append("Rain. Cold drops on skin. Wet.")
    elif weather == 'snow':
        parts.append("Snow. White flakes. Cold.")
    elif weather == 'storm':
        parts.append("Thunder. Lightning. Wind. Scared.")
    elif weather == 'heatwave':
        parts.append("Scorching. Air shimmers. Dry heat.")
    elif weather == 'fog':
        parts.append("Fog. Cannot see. White mist.")
    elif weather == 'sandstorm':
        parts.append("Sand in eyes. Cannot see. Wind.")
    elif weather == 'blizzard':
        parts.append("Whiteout. Wind screaming. Cannot move.")

    # Temperature
    temp = world_state.get('temperature', 20)
    if temp < 2:
        parts.append("Freezing. Cannot feel hands.")
    elif temp < 8:
        parts.append("Very cold. Shivering.")
    elif temp < 15:
        parts.append("Cold ground. Cold air.")
    elif temp > 38:
        parts.append("Burning hot. Skin hurts.")
    elif temp > 30:
        parts.append("Hot. Heat on skin.")
    elif temp > 22:
        parts.append("Warm. Comfortable.")

    # Light
    light = world_state.get('light_level', 0.5)
    if light < 0.15:
        parts.append("No light. Cannot see anything.")
    elif light < 0.3:
        parts.append("Very dark. Hard to see.")
    elif light < 0.5:
        parts.append("Dim. Getting dark.")
    elif light > 0.9:
        parts.append("Bright. Light everywhere.")
    elif light > 0.7:
        parts.append("Light. Can see far.")

    # Food
    if world_state.get('smell_food', 0) > 0.5:
        parts.append("Sweet smell. Something good near.")
    elif world_state.get('smell_food', 0) > 0.2:
        parts.append("Faint smell. Maybe something near.")

    # Danger
    if world_state.get('smell_danger', 0) > 0.5:
        parts.append("Something big and dark. Not safe here.")
    elif world_state.get('smell_danger', 0) > 0.2:
        parts.append("Something feels wrong. Not sure what.")

    # Water
    if world_state.get('wetness', 0) > 0.5:
        parts.append("Wet. Sound of water.")
    elif world_state.get('wetness', 0) > 0.2:
        parts.append("Damp. Smell of water somewhere.")

    # Sound
    sound = world_state.get('sound_level', 0.1)
    if sound > 0.7:
        parts.append("Loud sound. Close.")
    elif sound > 0.4:
        parts.append("Something making sound.")

    # Internal sensations
    hunger = adam_stats.get('hunger', 0)
    if hunger > 80:
        parts.append("Stomach hurts. Very empty inside. Painful.")
    elif hunger > 50:
        parts.append("Empty feeling inside. Twisting.")
    elif hunger > 30:
        parts.append("Slightly empty.")

    energy = adam_stats.get('energy', 100)
    if energy < 10:
        parts.append("Body very heavy. Cannot move well.")
    elif energy < 25:
        parts.append("Tired. Body heavy.")
    elif energy < 50:
        parts.append("Getting tired.")

    pain = adam_stats.get('pain', 0)
    if pain > 5:
        parts.append("Sharp pain. Hurts everywhere.")
    elif pain > 2:
        parts.append("Something hurts.")

    stress = adam_stats.get('stress', 0)
    if stress > 70:
        parts.append("Something very wrong. Cannot think. Scared.")
    elif stress > 40:
        parts.append("Uneasy. Something not right.")

    if not parts:
        parts.append("Quiet. Nothing happens.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Thought Generator — Converts sensory state into primitive inner speech
# ═══════════════════════════════════════════════════════════════════════════════

class ThoughtGenerator:
    """
    Converts Adam's raw sensory experience into primitive thoughts.

    The thought IS the sensation. Not an interpretation — the raw feeling
    expressed in the only words Adam knows.

    Process:
      1. Extract active sensations (what Adam feels RIGHT NOW)
      2. Rank by intensity (strongest feeling first)
      3. Add action awareness (what am I doing?)
      4. Add memory connection (does this feel familiar?)
      5. Compose in primitive syntax (word. word. word.)
    """

    # Maps sensation conditions to vocabulary words
    # Each entry: (word, intensity_compute_function)
    SENSATION_MAP = {
        "cold": lambda ws, s: max(0, (10 - ws.get('temperature', 20)) / 20),
        "hot": lambda ws, s: max(0, (ws.get('temperature', 20) - 30) / 10),
        "pain": lambda ws, s: s.get('pain', 0) / 10.0,
        "empty": lambda ws, s: s.get('hunger', 0) / 100.0,
        "tired": lambda ws, s: 1.0 - s.get('energy', 100) / 100.0,
        "dark": lambda ws, s: 1.0 - ws.get('light_level', 0.5),
        "wet": lambda ws, s: ws.get('wetness', 0),
        "loud": lambda ws, s: ws.get('sound_level', 0),
        "near": lambda ws, s: ws.get('smell_food', 0),
        "bad": lambda ws, s: ws.get('smell_danger', 0),
        "scared": lambda ws, s: s.get('stress', 0) / 100.0,
        # "full" and "awake" only register as thoughts when they're NOTABLE
        # — meaning Adam was recently empty/tired and now feels the contrast.
        # Early on when everything is fine, these are not notable sensations.
        "full": lambda ws, s: 0,   # Phase 4 will make this context-aware (full after being empty)
        "awake": lambda ws, s: 0,  # Phase 4 will make this context-aware (awake after being tired)
        "good": lambda ws, s: 0,   # Phase 4 will make this context-aware (good after being bad)
        "dry": lambda ws, s: 1.0 - ws.get('wetness', 0) if ws.get('wetness', 0) < 0.2 else 0,
        "quiet": lambda ws, s: 1.0 - ws.get('sound_level', 0) if ws.get('sound_level', 0) < 0.2 else 0,
        "hard": lambda ws, s: 1.0 if ws.get('touch_softness', 0.5) < 0.3 else 0,
        "soft": lambda ws, s: ws.get('touch_softness', 0.5) if ws.get('touch_softness', 0.5) > 0.7 else 0,
    }

    # Maps action names to thought-words
    ACTION_WORDS = {
        "EXPLORE": "look",
        "EAT": "eat",
        "DRINK": "drink",
        "SLEEP": "rest",
        "HIDE": "hide",
        "MOVE": "go",
        "FLEE": "run",
        "IDLE": "still",
    }

    def __init__(self, vocabulary: list = None):
        self.vocabulary = vocabulary or [
            "hot", "cold", "pain", "good", "bad",
            "full", "empty", "tired", "awake",
            "big", "small", "near", "far",
            "here", "there", "wet", "dry",
            "loud", "quiet", "soft", "hard",
            "light", "dark",
        ]
        # Words discovered through experience (Phase 2 will expand this)
        self.discovered_words = []
        self.total_thoughts = 0

    def generate(self, world_state: dict, adam_stats: dict,
                 action: str, recent_memories: list = None) -> str:
        """
        Generate a primitive thought from Adam's current experience.

        The thought is composed from:
          1. Dominant sensations (what is strongest RIGHT NOW)
          2. Action awareness (what am I doing?)
          3. Memory echo (does this connect to something before?)

        Returns: str — 2-6 word thought, period-separated
        """
        thought_parts = []
        active_sensations = []

        # Step 1: Find all active sensations above threshold
        for word, intensity_fn in self.SENSATION_MAP.items():
            if word not in self.vocabulary and word not in self.discovered_words:
                continue
            try:
                intensity = intensity_fn(world_state, adam_stats)
                if intensity > 0.15:  # threshold — only noticeable sensations
                    active_sensations.append((word, float(intensity)))
            except Exception:
                continue

        # Sort by intensity (strongest first)
        active_sensations.sort(key=lambda x: x[1], reverse=True)

        # Take top sensations (2-4 depending on intensity spread)
        if active_sensations:
            # Always include the strongest
            thought_parts.append(active_sensations[0][0])

            # Include others if they're strong enough relative to the top
            for word, intensity in active_sensations[1:4]:
                if intensity > 0.25:
                    thought_parts.append(word)

        # Step 2: Action awareness
        action_word = self.ACTION_WORDS.get(action)
        if action_word:
            # Only add if not already expressing the same thing
            if action_word not in thought_parts and len(thought_parts) < 5:
                thought_parts.append(action_word)

        # Step 3: Memory echo
        if recent_memories and len(recent_memories) > 0 and len(thought_parts) < 5:
            memory_word = self._find_memory_echo(world_state, adam_stats, recent_memories)
            if memory_word and memory_word not in thought_parts:
                thought_parts.append(memory_word)

        # Step 4: Compose thought
        self.total_thoughts += 1

        if not thought_parts:
            # No strong sensation — just awareness of existing
            return "quiet. still."

        # Primitive syntax: word. word. word.
        return ". ".join(thought_parts) + "."

    def _find_memory_echo(self, world_state: dict, adam_stats: dict,
                          recent_memories: list) -> str:
        """
        Find a word that connects the current experience to a past memory.

        This is the simplest form of memory influence on thought.
        Phase 4 will make this sophisticated with proper memory retrieval.
        """
        # Check if danger connects to a painful memory
        if world_state.get('smell_danger', 0) > 0.3:
            for mem in recent_memories[-3:]:
                outcome = mem.get('outcome', '').lower()
                if 'pain' in outcome or 'hurt' in outcome or 'damage' in outcome:
                    return "bad"

        # Check if food connects to a good memory
        if world_state.get('smell_food', 0) > 0.3:
            for mem in recent_memories[-3:]:
                outcome = mem.get('outcome', '').lower()
                if 'good' in outcome or 'full' in outcome or 'eat' in outcome:
                    return "good"

        # Check if cold connects to a cold memory
        if world_state.get('temperature', 20) < 10:
            for mem in recent_memories[-3:]:
                event = mem.get('event', '').lower()
                if 'cold' in event or 'freezing' in event:
                    return "cold"

        return None

    def add_word(self, word: str):
        """Add a new word to Adam's discovered vocabulary."""
        if word not in self.vocabulary and word not in self.discovered_words:
            self.discovered_words.append(word)

    def get_all_vocabulary(self) -> list:
        """Return complete vocabulary (starting + discovered)."""
        return self.vocabulary + self.discovered_words


# ═══════════════════════════════════════════════════════════════════════════════
# Emotion Classifier — Maps bodily state to single emotion word
# ═══════════════════════════════════════════════════════════════════════════════

class EmotionClassifier:
    """
    Classifies Adam's emotional state from his bodily state and environment.

    Emotions are NOT learned — they are direct responses to bodily conditions.
    This is philosophically correct: a primitive being doesn't "learn" to feel
    pain or fear. The body responds. The emotion IS the response.

    Rules are priority-ordered: first match wins.
    """

    EMOTION_RULES = [
        # (emotion_name, condition_function, description)
        # Priority-ordered: first match wins. Strong negative emotions first.
        ("desperate", lambda ws, s: s.get('health', 100) < 20 and s.get('hunger', 0) > 60,
         "near death and starving"),
        ("pained", lambda ws, s: s.get('pain', 0) > 4,
         "physical pain is dominant"),
        ("terrified", lambda ws, s: s.get('stress', 0) > 70 and ws.get('smell_danger', 0) > 0.3,
         "extreme fear with danger present"),
        ("scared", lambda ws, s: s.get('stress', 0) > 55,
         "high stress dominates experience"),
        ("cautious", lambda ws, s: ws.get('smell_danger', 0) > 0.3 or (s.get('stress', 0) > 35 and s.get('hunger', 0) > 50),
         "danger or stress requires caution"),
        ("exhausted", lambda ws, s: s.get('energy', 100) < 12,
         "body has no energy left"),
        ("hungry", lambda ws, s: s.get('hunger', 0) > 65,
         "hunger dominates all feeling"),
        # Positive emotions — these require contrast with a recent negative state.
        # "relieved" means: stress WAS high but is now low — recovery from stress.
        # Detect recovery: low current stress + good energy (implies recent rest/recovery).
        ("relieved", lambda ws, s: (
            s.get('stress', 0) < 15 and s.get('energy', 100) > 50
        ),
         "stress was high, now recovered — relief"),
        # "satisfied" means: hunger WAS high but is now low — recovery from hunger.
        ("satisfied", lambda ws, s: (
            s.get('hunger', 0) < 20 and s.get('energy', 100) > 30
        ),
         "hunger was high, now fed — satisfaction"),
        ("calm", lambda ws, s: s.get('hunger', 0) > 10 and s.get('hunger', 0) < 35 and s.get('energy', 100) > 50 and s.get('stress', 0) < 15,
         "slight hunger but managing"),
        ("curious", lambda ws, s: s.get('energy', 100) > 50 and s.get('stress', 0) < 25 and ws.get('smell_food', 0) > 0.2,
         "energy and interest in surroundings"),
        # "uncertain" is the default when nothing strong is felt
        # This is the correct emotion for a newborn entity with no experience
        ("uncertain", lambda ws, s: True,
         "default — no strong emotion"),
    ]

    def classify(self, world_state: dict, adam_stats: dict) -> str:
        """
        Classify Adam's current emotional state.

        Returns: str — single emotion word
        """
        for emotion, condition, _ in self.EMOTION_RULES:
            try:
                if condition(world_state, adam_stats):
                    return emotion
            except Exception:
                continue
        return "uncertain"

    def classify_with_reason(self, world_state: dict, adam_stats: dict) -> tuple:
        """
        Classify emotion and return the reason.

        Returns: (emotion, reason) tuple
        """
        for emotion, condition, reason in self.EMOTION_RULES:
            try:
                if condition(world_state, adam_stats):
                    return (emotion, reason)
            except Exception:
                continue
        return ("uncertain", "no strong signals")


# ═══════════════════════════════════════════════════════════════════════════════
# Episode Memory — Rolling buffer for experiences within one episode
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodeMemory:
    """
    Rolling memory buffer for a single training episode.

    Stores the rich experience data (thoughts, emotions, events, outcomes)
    that the PPO buffer doesn't need but Adam's inner life requires.

    This is NOT the PPO rollout buffer — that stores obs/actions/rewards.
    This stores the EXPERIENTIAL data for thought generation and display.
    """

    def __init__(self, max_size: int = 10):
        self.memories = []
        self.max_size = max_size

    def add(self, tick: int, event: str, thought: str,
            action: str, emotion: str, outcome: str):
        """Store a new experience."""
        memory = {
            "tick": tick,
            "event": event,
            "thought": thought,
            "action": action,
            "emotion": emotion,
            "outcome": outcome,
        }
        self.memories.append(memory)
        if len(self.memories) > self.max_size:
            self.memories.pop(0)

    def get_recent(self, n: int = 3) -> list:
        """Get the N most recent memories."""
        return self.memories[-n:]

    def clear(self):
        """Reset for a new episode."""
        self.memories = []

    def __len__(self):
        return len(self.memories)


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent Memory — Survives across episodes (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════════

class PersistentMemory:
    """
    Memory that survives across episodes. This IS Adam's personality.

    Philosophy:
      - Short-term memory resets each episode (Adam is born anew)
      - But long-term memory persists (Adam remembers past lives)
      - This is what creates PERSONALITY — consistent patterns across episodes
      - Adam who learned "water near river" in episode 50 remembers it in episode 200
      - Fear triggers accumulate: Adam becomes more cautious over time
      - Good memories accumulate: Adam becomes more confident

    What persists:
      - Key experiences (high reward or high punishment events)
      - Fear triggers (things that caused pain)
      - Good memories (things that gave reward)
      - Discovered vocabulary and their meanings
      - Behavioral patterns (what worked, what didn't)
    """

    def __init__(self, max_experiences: int = 100):
        self.key_experiences = []   # Significant experiences across episodes
        self.fear_triggers = []     # World states that led to pain
        self.good_memories = []     # World states that led to reward
        self.pattern_memory = {}    # {situation_hash: best_action, avg_reward}
        self.episodes_survived = 0
        self.total_ticks_lived = 0
        self.max_experiences = max_experiences
        self._discovered_vocabulary = {}  # Phase 2 vocabulary

    def store_experience(self, experience: dict, reward: float, tick: int):
        """
        Store a significant experience if it's worth remembering.

        Criteria for "worth remembering":
          - High positive reward (> 1.0) — something good happened
          - High negative reward (< -1.0) — something bad happened
          - Death — the ultimate negative experience
          - First time experiencing something (discovery events)
        """
        is_significant = False

        # High reward events — only store GENUINE successes.
        # A "good memory" must be an action that actually produced a positive outcome.
        # We check the outcome text to confirm the action succeeded, not just the
        # reward number (which can be inflated by survival_tick + reflection bonuses).
        outcome_text = experience.get('outcome', '').lower()
        action = experience.get('action', '')
        is_genuine_success = False

        if reward > 0.8:
            # Verify the action actually succeeded by checking outcome text
            if action == "EAT" and ("empty feeling goes away" in outcome_text or "found something to eat" in outcome_text):
                is_genuine_success = True
            elif action == "DRINK" and ("found water" in outcome_text or "less empty" in outcome_text):
                is_genuine_success = True
            elif action == "SLEEP" and ("rested" in outcome_text or "body feels better" in outcome_text):
                is_genuine_success = True
            elif action == "HIDE" and "danger passed" in outcome_text:
                is_genuine_success = True
            elif action == "FLEE" and "safe now" in outcome_text:
                is_genuine_success = True
            elif action == "EXPLORE" and ("found something" in outcome_text or "good smell" in outcome_text):
                is_genuine_success = True

        if is_genuine_success:
            is_significant = True
            self.good_memories.append({
                "event": experience.get('event', ''),
                "action": experience.get('action', ''),
                "outcome": experience.get('outcome', ''),
                "reward": reward,
                "tick": tick,
            })
            if len(self.good_memories) > 20:
                self.good_memories.pop(0)

        # High punishment events
        if reward < -1.0:
            is_significant = True
            self.fear_triggers.append({
                "event": experience.get('event', ''),
                "action": experience.get('action', ''),
                "outcome": experience.get('outcome', ''),
                "reward": reward,
                "tick": tick,
            })
            if len(self.fear_triggers) > 20:
                self.fear_triggers.pop(0)

        # Discovery events
        if 'new_words' in experience:
            is_significant = True

        if is_significant:
            record = {
                "tick": tick,
                "event": experience.get('event', ''),
                "thought": experience.get('thought', ''),
                "dialogue": experience.get('dialogue', ''),
                "emotion": experience.get('emotion', ''),
                "action": experience.get('action', ''),
                "outcome": experience.get('outcome', ''),
                "reward": reward,
            }
            self.key_experiences.append(record)
            if len(self.key_experiences) > self.max_experiences:
                self.key_experiences.pop(0)

    def update_pattern(self, situation_key: str, action: str, reward: float):
        """
        Update behavioral pattern memory.

        situation_key: a hash of the current world state
        action: what Adam did
        reward: what happened as a result

        Over time, Adam learns: "In THIS situation, THIS action works best"
        """
        if situation_key not in self.pattern_memory:
            self.pattern_memory[situation_key] = {}

        if action not in self.pattern_memory[situation_key]:
            self.pattern_memory[situation_key][action] = {
                "count": 0, "total_reward": 0.0, "avg_reward": 0.0
            }

        pattern = self.pattern_memory[situation_key][action]
        pattern["count"] += 1
        pattern["total_reward"] += reward
        pattern["avg_reward"] = pattern["total_reward"] / pattern["count"]

    def get_best_action_for(self, situation_key: str) -> str:
        """
        Get the best action Adam has learned for a given situation.

        Returns None if Adam hasn't experienced this situation enough.
        """
        if situation_key not in self.pattern_memory:
            return None

        actions = self.pattern_memory[situation_key]
        best_action = None
        best_avg = -float('inf')

        for action, data in actions.items():
            if data["count"] >= 3 and data["avg_reward"] > best_avg:
                best_avg = data["avg_reward"]
                best_action = action

        return best_action

    def get_relevant_fears(self, world_state: dict) -> list:
        """Get fear triggers that are similar to the current world state."""
        relevant = []
        for fear in self.fear_triggers[-5:]:
            # Handle both dict and string fear entries (legacy data may store strings)
            if isinstance(fear, dict):
                event = fear.get('event', '').lower()
            elif isinstance(fear, str):
                event = fear.lower()
            else:
                continue
            # Simple keyword matching for relevance
            if world_state.get('smell_danger', 0) > 0.3 and ('danger' in event or 'bad' in event or 'dark' in event):
                relevant.append(fear)
            elif world_state.get('temperature', 20) < 10 and ('cold' in event or 'freezing' in event):
                relevant.append(fear)
        return relevant

    def get_relevant_good_memories(self, world_state: dict) -> list:
        """Get good memories that are similar to the current world state."""
        relevant = []
        for mem in self.good_memories[-5:]:
            # Handle both dict and string memory entries (legacy data may store strings)
            if isinstance(mem, dict):
                event = mem.get('event', '').lower()
            elif isinstance(mem, str):
                event = mem.lower()
            else:
                continue
            if world_state.get('smell_food', 0) > 0.3 and ('food' in event or 'good' in event or 'near' in event):
                relevant.append(mem)
            elif world_state.get('wetness', 0) > 0.3 and ('water' in event or 'wet' in event):
                relevant.append(mem)
        return relevant

    def end_episode(self, ticks_survived: int):
        """Called at the end of each episode."""
        self.episodes_survived += 1
        self.total_ticks_lived += ticks_survived

    def make_situation_key(self, world_state: dict, adam_stats: dict) -> str:
        """
        Create a simplified hash of the current situation for pattern matching.

        Discretizes continuous values into buckets.
        """
        health_bucket = "h_high" if adam_stats.get('health', 100) > 60 else "h_low"
        hunger_bucket = "hu_low" if adam_stats.get('hunger', 0) < 30 else ("hu_mid" if adam_stats.get('hunger', 0) < 65 else "hu_high")
        energy_bucket = "e_high" if adam_stats.get('energy', 100) > 40 else "e_low"
        danger = "d_yes" if world_state.get('smell_danger', 0) > 0.3 else "d_no"
        food = "f_yes" if world_state.get('smell_food', 0) > 0.3 else "f_no"
        temp = "t_cold" if world_state.get('temperature', 20) < 10 else ("t_hot" if world_state.get('temperature', 20) > 30 else "t_ok")
        light = "l_dark" if world_state.get('light_level', 0.5) < 0.3 else ("l_dim" if world_state.get('light_level', 0.5) < 0.7 else "l_light")

        return f"{health_bucket}_{hunger_bucket}_{energy_bucket}_{danger}_{food}_{temp}_{light}"

    def get_personality_summary(self) -> dict:
        """
        Get a summary of Adam's developing personality.

        This IS the Nafs — the pattern of being that emerges from experience.
        """
        fear_count = len(self.fear_triggers)
        good_count = len(self.good_memories)

        # Personality traits emerge from experience ratio
        if fear_count > good_count * 2:
            disposition = "fearful"
        elif good_count > fear_count * 2:
            disposition = "confident"
        elif fear_count > 0 and good_count > 0:
            disposition = "cautious"
        else:
            disposition = "uncertain"

        return {
            "episodes_survived": self.episodes_survived,
            "total_ticks_lived": self.total_ticks_lived,
            "key_experiences": len(self.key_experiences),
            "fear_triggers": fear_count,
            "good_memories": good_count,
            "patterns_learned": len(self.pattern_memory),
            "disposition": disposition,
        }

    # ── Phase 5: Fear/Pleasure signal computation ──────────────────────────

    def compute_fear_signal(self, world_state: dict) -> float:
        """
        Compute a fear signal [0, 1] based on how many relevant fear triggers
        match the current world state.

        This is injected into the PPO observation vector so Adam's brain
        can learn from past fears, not just current danger.
        """
        if not self.fear_triggers:
            return 0.0

        relevant = self.get_relevant_fears(world_state)
        # Signal strength scales with number of relevant fears
        return min(1.0, len(relevant) * 0.25)

    def compute_pleasure_signal(self, world_state: dict) -> float:
        """
        Compute a pleasure signal [0, 1] based on how many relevant good
        memories match the current world state.

        This is injected into the PPO observation vector so Adam's brain
        can learn from past rewards, not just current food/water.
        """
        if not self.good_memories:
            return 0.0

        relevant = self.get_relevant_good_memories(world_state)
        # Signal strength scales with number of relevant good memories
        return min(1.0, len(relevant) * 0.25)

    def compute_pattern_confidence(self, world_state: dict, adam_stats: dict) -> float:
        """
        Compute pattern confidence [0, 1] — how confident Adam is about
        the best action for the current situation.

        Based on:
          - Whether a pattern exists for this situation
          - How many times it's been observed
          - How consistent the reward has been

        Returns 0 if no pattern exists yet (Adam hasn't learned this situation).
        """
        situation_key = self.make_situation_key(world_state, adam_stats)
        if situation_key not in self.pattern_memory:
            return 0.0

        # Find the best action for this situation
        best_action = self.get_best_action_for(situation_key)
        if best_action is None:
            return 0.0

        # Confidence = min(1.0, count/10) — need 10 observations for full confidence
        pattern_data = self.pattern_memory[situation_key][best_action]
        confidence = min(1.0, pattern_data["count"] / 10.0)

        return confidence

    # ── Phase 4: Persistence to disk ───────────────────────────────────────

    def save_to_disk(self, filepath: str = "memory.json"):
        """Save persistent memory to disk so it survives training restarts."""
        import json
        data = {
            "episodes_survived": self.episodes_survived,
            "total_ticks_lived": self.total_ticks_lived,
            "fear_triggers": self.fear_triggers[-20:],
            "good_memories": self.good_memories[-20:],
            "key_experiences": self.key_experiences[-50:],
            "pattern_memory": {
                k: {a: d for a, d in v.items() if d["count"] >= 2}
                for k, v in self.pattern_memory.items()
            },
            # Phase 2: Persist discovered vocabulary so it survives restarts
            "discovered_vocabulary": getattr(self, '_discovered_vocabulary', {}),
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [Memory] Save failed: {e}", flush=True)

    def load_from_disk(self, filepath: str = "memory.json"):
        """Load persistent memory from disk (previous training runs)."""
        import json
        import os
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.episodes_survived = data.get("episodes_survived", 0)
            self.total_ticks_lived = data.get("total_ticks_lived", 0)
            # Sanitize: ensure fear_triggers and good_memories are dicts, not strings
            raw_fears = data.get("fear_triggers", [])
            self.fear_triggers = [
                f if isinstance(f, dict) else {"event": str(f), "action": "", "outcome": "", "reward": -1.0, "tick": 0}
                for f in raw_fears
            ]
            raw_good = data.get("good_memories", [])
            self.good_memories = [
                m if isinstance(m, dict) else {"event": str(m), "action": "", "outcome": "", "reward": 1.0, "tick": 0}
                for m in raw_good
            ]
            self.key_experiences = data.get("key_experiences", [])
            self.pattern_memory = data.get("pattern_memory", {})
            # Phase 2: Restore discovered vocabulary
            self._discovered_vocabulary = data.get("discovered_vocabulary", {})
            print(f"  [Memory] Loaded: {len(self.fear_triggers)} fears, "
                  f"{len(self.good_memories)} joys, "
                  f"{len(self.pattern_memory)} patterns, "
                  f"{len(self._discovered_vocabulary)} discovered words, "
                  f"{self.episodes_survived} past episodes", flush=True)
        except Exception as e:
            self._discovered_vocabulary = {}
            print(f"  [Memory] Load failed: {e}. Starting fresh.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome Describer — Converts action result to primitive text
# ═══════════════════════════════════════════════════════════════════════════════

def describe_action_outcome(action: str, world_state: dict,
                            adam_stats: dict, prev_stats: dict) -> str:
    """
    Describe what happened as a result of Adam's action.

    Compares before/after stats to detect what changed.

    Returns: str — primitive description of the outcome
    """
    parts = []

    health_delta = adam_stats.get('health', 100) - prev_stats.get('health', 100)
    hunger_delta = adam_stats.get('hunger', 0) - prev_stats.get('hunger', 0)
    energy_delta = adam_stats.get('energy', 100) - prev_stats.get('energy', 100)
    stress_delta = adam_stats.get('stress', 0) - prev_stats.get('stress', 0)

    if action == "EAT":
        if hunger_delta < -10:
            parts.append("Found something to eat. Empty feeling goes away.")
        else:
            parts.append("Nothing to eat here. Still empty.")

    elif action == "DRINK":
        if hunger_delta < -3:
            parts.append("Found water. Wet. Less empty.")
        else:
            parts.append("No water here.")

    elif action == "SLEEP":
        if energy_delta > 15:
            parts.append("Rested. Body feels better. Light again.")
        elif energy_delta > 5:
            parts.append("Some rest. Not enough.")
        else:
            parts.append("Could not rest well.")

    elif action == "HIDE":
        if world_state.get('smell_danger', 0) > 0.3:
            if stress_delta < -3:
                parts.append("Hiding. Danger passed.")
            else:
                parts.append("Hiding. Still scared.")
        else:
            parts.append("Hiding. Nothing to hide from.")

    elif action == "EXPLORE":
        if world_state.get('smell_food', 0) > 0.3:
            parts.append("Found something. Good smell.")
        elif world_state.get('smell_danger', 0) > 0.3:
            parts.append("Something bad here. Pain.")
        else:
            parts.append("Looking around. Nothing yet.")

    elif action == "MOVE":
        if world_state.get('smell_food', 0) > 0.3 or world_state.get('wetness', 0) > 0.3:
            parts.append("Went somewhere new. Found something.")
        else:
            parts.append("Moved. Same as before.")

    elif action == "FLEE":
        if stress_delta < -5:
            parts.append("Ran away. Safe now.")
        else:
            parts.append("Running. Still scared.")

    elif action == "IDLE":
        parts.append("Did nothing. Waited.")

    # Add pain/stress changes
    if health_delta < -3:
        parts.append("Pain. Body hurting.")
    elif health_delta < -1:
        parts.append("Something not right.")

    if stress_delta > 10:
        parts.append("More scared now.")
    elif stress_delta < -10:
        parts.append("Feeling better. Less scared.")

    if not parts:
        parts.append("Nothing changed.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary Discovery — Adam names things from experience (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

class VocabularyDiscovery:
    """
    Adam discovers new words by combining existing vocabulary when he
    experiences something significant for the first time.

    Philosophy:
      - Adam doesn't "learn words" from a dictionary — he NAMES things
      - The naming comes from combining what he already knows:
        "cold" + "pain" = "cold pain" (freezing)
        "near" + "good" = "near good" (food is close)
        "big" + "bad" = "big bad" (danger)
      - Each discovered word carries MEANING — the context in which it was born
      - This is genuine language emergence: no LLM, no pretrained knowledge

    Discovery triggers:
      1. First successful EAT → name the food experience
      2. First pain from danger → name the danger
      3. First extreme cold → name the freezing
      4. First water found → name the water
      5. First shelter found → name the shelter
      6. Recovery from near-death → name survival
    """

    # Discovery rules: (trigger_name, condition, word_parts, meaning)
    DISCOVERY_RULES = [
        # Food-related discoveries
        ("first_eat_success",
         lambda ws, s, action, prev: (
             action == "EAT" and
             s.get('hunger', 0) < prev.get('hunger', 0) - 10
         ),
         ("near", "good"),
         "something to eat when hungry"),

        ("first_food_found",
         lambda ws, s, action, prev: (
             ws.get('smell_food', 0) > 0.5 and
             action in ("EXPLORE", "MOVE")
         ),
         ("good", "near"),
         "smelled something good nearby"),

        # Danger-related discoveries
        ("first_danger_pain",
         lambda ws, s, action, prev: (
             ws.get('smell_danger', 0) > 0.3 and
             s.get('pain', 0) > 2
         ),
         ("big", "bad"),
         "something big and dangerous causes pain"),

        ("first_danger_seen",
         lambda ws, s, action, prev: (
             ws.get('smell_danger', 0) > 0.5 and
             action == "HIDE"
         ),
         ("bad", "far"),
         "something bad, need to stay far"),

        # Temperature-related discoveries
        ("first_freezing",
         lambda ws, s, action, prev: (
             ws.get('temperature', 20) < 5 and
             s.get('health', 100) < prev.get('health', 100)
         ),
         ("cold", "pain"),
         "extreme cold hurts the body"),

        ("first_hot_damage",
         lambda ws, s, action, prev: (
             ws.get('temperature', 20) > 35 and
             s.get('health', 100) < prev.get('health', 100)
         ),
         ("hot", "pain"),
         "extreme heat hurts the body"),

        # Water-related discoveries
        ("first_water_found",
         lambda ws, s, action, prev: (
             ws.get('wetness', 0) > 0.5 and
             action in ("EXPLORE", "MOVE")
         ),
         ("wet", "good"),
         "found water, wet and good"),

        ("first_drink_success",
         lambda ws, s, action, prev: (
             action == "DRINK" and
             s.get('hunger', 0) < prev.get('hunger', 0) - 3
         ),
         ("wet", "near"),
         "found water to drink"),

        # Shelter-related discoveries
        ("first_shelter_rest",
         lambda ws, s, action, prev: (
             action == "SLEEP" and
             s.get('energy', 100) > prev.get('energy', 100) + 20
         ),
         ("soft", "good"),
         "found a good place to rest"),

        # Darkness-related discoveries
        ("first_dark_fear",
         lambda ws, s, action, prev: (
             ws.get('light_level', 0.5) < 0.15 and
             s.get('stress', 0) > prev.get('stress', 0) + 3
         ),
         ("dark", "bad"),
         "darkness brings fear"),

        # Survival-related discoveries
        ("near_death_survival",
         lambda ws, s, action, prev: (
             s.get('health', 100) < 25 and
             prev.get('health', 100) < 25 and
             s.get('health', 100) > prev.get('health', 100)
         ),
         ("pain", "good"),
         "survived despite the pain"),
    ]

    def __init__(self):
        self.discovered = {}  # {word: {"meaning": str, "tick": int, "trigger": str}}
        self.triggered = set()  # Track which triggers have fired (first-time only)
        self.discovery_log = []  # Log of all discoveries

    def check_for_discovery(self, world_state: dict, adam_stats: dict,
                            action: str, prev_stats: dict,
                            tick: int, vocabulary: list) -> list:
        """
        Check if Adam's current experience triggers a new word discovery.

        Each trigger fires ONLY ONCE — the first time it's encountered.
        This mirrors real learning: you only need to discover fire burns once.

        Returns: list of (new_word, meaning) tuples discovered this tick
        """
        new_discoveries = []

        for trigger_name, condition, word_parts, meaning in self.DISCOVERY_RULES:
            # Skip if already discovered this trigger
            if trigger_name in self.triggered:
                continue

            # Check if condition is met
            try:
                if condition(world_state, adam_stats, action, prev_stats):
                    # Create new word by combining existing vocabulary
                    part1, part2 = word_parts

                    # Only create the word if both parts are in vocabulary
                    if part1 in vocabulary and part2 in vocabulary:
                        new_word = f"{part1} {part2}"

                        # Check if we already have this exact word
                        if new_word not in self.discovered:
                            self.discovered[new_word] = {
                                "meaning": meaning,
                                "tick": tick,
                                "trigger": trigger_name,
                            }
                            self.triggered.add(trigger_name)

                            discovery_record = {
                                "tick": tick,
                                "word": new_word,
                                "meaning": meaning,
                                "trigger": trigger_name,
                                "context": {
                                    "health": adam_stats.get('health', 100),
                                    "hunger": adam_stats.get('hunger', 0),
                                    "energy": adam_stats.get('energy', 100),
                                    "stress": adam_stats.get('stress', 0),
                                    "pain": adam_stats.get('pain', 0),
                                    "temperature": world_state.get('temperature', 20),
                                    "action": action,
                                }
                            }
                            self.discovery_log.append(discovery_record)
                            new_discoveries.append((new_word, meaning))
            except Exception:
                continue

        return new_discoveries

    def get_discovered_words(self) -> list:
        """Return list of all discovered words."""
        return list(self.discovered.keys())

    def get_vocabulary_with_meanings(self) -> dict:
        """Return all discovered words with their meanings."""
        return {word: info["meaning"] for word, info in self.discovered.items()}

    def reset_for_episode(self):
        """Reset trigger tracking for a new episode.
        Discoveries persist across episodes (Adam remembers what he learned),
        but triggers can fire again in new episodes (reinforcing the concept).
        """
        self.triggered = set()


# ═══════════════════════════════════════════════════════════════════════════════
# Dialogue Generator — The Subconscious Gap (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class DialogueGenerator:
    """
    Generates Adam's outward expression (dialogue) from his inner thought.

    The gap between thought and dialogue IS the subconscious.
    Nobody programmed that gap. It emerges from constraint.

    Rules:
      - When calm (stress < 20): dialogue ≈ thought (Adam speaks his mind)
      - When stressed (stress 20-50): dialogue is simplified (fewer words)
      - When highly stressed (stress > 50): dialogue is minimal or absent
      - When in pain: dialogue is just the pain word, nothing else
      - When scared: dialogue hides vulnerability ("not bad" instead of "scared")

    This creates the subconscious: Adam thinks "cold. empty. scared."
    but says "cold." — hiding his fear because his stressed mind
    can't articulate it. The Observer sees both. The world sees only dialogue.
    """

    # Emotion → dialogue override mappings
    # When Adam feels these, he expresses them differently
    EMOTION_DIALOGUE_MAP = {
        "terrified": "bad.",           # Can't articulate terror, just "bad"
        "scared": "not good.",        # Downplays fear
        "desperate": "...",           # Too overwhelmed to speak
        "pained": "pain.",            # Only pain comes through
        "exhausted": "tired.",        # Simple statement
        "hungry": "empty.",           # Focus on the physical sensation
        "cautious": None,            # No special mapping — use thought
        "uncertain": None,           # No special mapping — use thought
        "calm": None,                # Calm = thought ≈ dialogue
        "curious": None,             # Curious = thought ≈ dialogue
        "relieved": None,
        "satisfied": None,
    }

    def generate(self, thought: str, emotion: str,
                 adam_stats: dict, world_state: dict) -> str:
        """
        Generate dialogue from thought, filtered through emotion and stress.

        The subconscious gap widens under stress:
          - Calm Adam: says what he thinks
          - Stressed Adam: says less, hides vulnerability
          - Terrified Adam: barely speaks

        Returns: str — what Adam expresses outwardly
        """
        stress = adam_stats.get('stress', 0)
        pain = adam_stats.get('pain', 0)

        # Step 1: Check for emotion-specific dialogue override
        emotion_dialogue = self.EMOTION_DIALOGUE_MAP.get(emotion)

        # Step 2: If in extreme pain, dialogue is just "pain."
        if pain > 5:
            return "pain."

        # Step 3: If emotion has a specific override, use it when stressed
        if emotion_dialogue is not None and stress > 40:
            return emotion_dialogue

        # Step 4: Filter the thought based on stress level
        thought_words = [w.strip() for w in thought.replace(".", " ").split() if w.strip()]

        if stress < 15:
            # Calm: dialogue ≈ thought (Adam speaks freely)
            return thought

        elif stress < 35:
            # Mild stress: drop the last 1-2 words (can't complete thoughts)
            keep = max(2, len(thought_words) - 1)
            filtered = thought_words[:keep]
            return ". ".join(filtered) + "."

        elif stress < 55:
            # Moderate stress: keep only the strongest 1-2 words
            keep = min(2, len(thought_words))
            filtered = thought_words[:keep]
            return ". ".join(filtered) + "."

        else:
            # High stress: minimal dialogue (1 word or silence)
            if emotion_dialogue:
                return emotion_dialogue
            if thought_words:
                return thought_words[0] + "."
            return "..."

    def measure_subconscious_gap(self, thought: str, dialogue: str) -> dict:
        """
        Measure the gap between thought and dialogue.

        This IS the subconscious — the difference between what Adam
        thinks and what he expresses.

        Returns: dict with gap metrics
        """
        thought_words = set(thought.replace(".", " ").split())
        dialogue_words = set(dialogue.replace(".", " ").split())

        # Words Adam thought but didn't say
        hidden = thought_words - dialogue_words

        # Words Adam said but didn't think (emotion overrides)
        added = dialogue_words - thought_words

        return {
            "hidden_words": list(hidden),
            "added_words": list(added),
            "thought_length": len(thought_words),
            "dialogue_length": len(dialogue_words),
            "gap_ratio": len(hidden) / max(len(thought_words), 1),
            "subconscious_active": len(hidden) > 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════

class ThoughtEngine:
    """
    Unified interface for Adam's inner experience.

    Combines:
      - ThoughtGenerator (what Adam thinks)
      - EmotionClassifier (what Adam feels)
      - EpisodeMemory (what Adam remembers)
      - WorldDescriber (what Adam perceives)

    Usage:
      engine = ThoughtEngine()
      experience = engine.experience(world_state, adam_stats, action, prev_stats)
      print(experience['thought'])   # "cold. empty. look."
      print(experience['emotion'])   # "hungry"
    """

    def __init__(self, vocabulary: list = None, memory_size: int = 10,
                 memory_filepath: str = None):
        self.thought_gen = ThoughtGenerator(vocabulary)
        self.emotion_cls = EmotionClassifier()
        self.memory = EpisodeMemory(max_size=memory_size)
        self.vocab_discovery = VocabularyDiscovery()  # Phase 2
        self.dialogue_gen = DialogueGenerator()  # Phase 3
        self.persistent_memory = PersistentMemory()  # Phase 4
        self.memory_filepath = memory_filepath

        # In single-life mode, Adam starts with NO memory from past lives.
        # Only load if a filepath is explicitly provided (for backward compat).
        if memory_filepath is not None:
            self.persistent_memory.load_from_disk(memory_filepath)

        # Restore discovered vocabulary from persistent memory
        if self.persistent_memory._discovered_vocabulary:
            for word, info in self.persistent_memory._discovered_vocabulary.items():
                self.thought_gen.add_word(word)
                # Also restore to vocab_discovery so it knows these were already discovered
                self.vocab_discovery.discovered[word] = info
                self.vocab_discovery.triggered.add(info.get('trigger', ''))

    def experience(self, world_state: dict, adam_stats: dict,
                   action: str, prev_stats: dict = None,
                   tick: int = 0, reward: float = 0.0) -> dict:
        """
        Generate a complete inner experience for one moment.

        This is the core call — it produces everything Adam "experiences"
        in a single tick: what he perceives, what he thinks, what he feels,
        and what happened as a result of his action.

        Phase 4 addition: reward is now passed in for persistent memory storage.

        Returns: dict with keys:
          - event: what's happening in the world
          - thought: Adam's inner speech
          - dialogue: what Adam says outwardly
          - emotion: Adam's emotional state
          - action: what Adam did
          - outcome: what happened as a result
          - subconscious_gap: the gap between thought and dialogue
        """
        prev = prev_stats or adam_stats

        # Phase 4: Context-aware thought generation
        # Check if Adam was recently empty/tired and now feels the contrast
        prev_hunger = prev.get('hunger', 0)
        prev_energy = prev.get('energy', 100)
        curr_hunger = adam_stats.get('hunger', 0)
        curr_energy = adam_stats.get('energy', 100)

        # Enable "full" when Adam was hungry and just ate
        # Use default argument to capture the VALUE, not the closure variable
        if prev_hunger > 30 and curr_hunger < prev_hunger - 10:
            captured_prev = prev_hunger
            self.thought_gen.SENSATION_MAP["full"] = lambda ws, s, ph=captured_prev: max(0, (ph - s.get('hunger', 0)) / ph) if s.get('hunger', 0) < 25 else 0

        # Enable "awake" when Adam was tired and just rested
        if prev_energy < 40 and curr_energy > prev_energy + 15:
            captured_prev_e = prev_energy
            self.thought_gen.SENSATION_MAP["awake"] = lambda ws, s, pe=captured_prev_e: max(0, (s.get('energy', 100) - pe) / 60) if s.get('energy', 100) > 65 else 0

        # Enable "good" when things improved
        if prev.get('hunger', 0) > 30 and adam_stats.get('hunger', 0) < 20 and adam_stats.get('stress', 0) < 20:
            self.thought_gen.SENSATION_MAP["good"] = lambda ws, s: 0.5 if s.get('hunger', 0) < 25 and s.get('stress', 0) < 20 else 0

        # What's happening in the world
        event = describe_world_event(world_state, adam_stats)

        # What Adam thinks (inner speech)
        # Phase 5: Include persistent memory context (fear/pleasure maps)
        recent = self.memory.get_recent(3)
        thought = self.thought_gen.generate(world_state, adam_stats, action, recent)

        # Phase 5: Enhance thought with fear/pleasure map signals
        relevant_fears = self.persistent_memory.get_relevant_fears(world_state)
        relevant_goods = self.persistent_memory.get_relevant_good_memories(world_state)

        if relevant_fears and "bad" not in thought and "scared" not in thought:
            # Adam remembers this situation was bad before
            thought = thought.rstrip(".") + ". bad."
        elif relevant_goods and "good" not in thought and "near" not in thought:
            # Adam remembers this situation was good before
            thought = thought.rstrip(".") + ". good."

        # What Adam feels (emotion)
        emotion, emotion_reason = self.emotion_cls.classify_with_reason(world_state, adam_stats)

        # Phase 3: What Adam says (dialogue) — filtered through stress
        # This creates the subconscious gap
        dialogue = self.dialogue_gen.generate(thought, emotion, adam_stats, world_state)
        subconscious_gap = self.dialogue_gen.measure_subconscious_gap(thought, dialogue)

        # What happened
        outcome = describe_action_outcome(action, world_state, adam_stats, prev)

        # Phase 2: Check for vocabulary discovery
        new_words = self.vocab_discovery.check_for_discovery(
            world_state, adam_stats, action, prev,
            tick, self.thought_gen.get_all_vocabulary()
        )
        for word, meaning in new_words:
            self.thought_gen.add_word(word)

        # Store in episode memory
        self.memory.add(tick, event, thought, action, emotion, outcome)

        result = {
            "event": event,
            "thought": thought,
            "dialogue": dialogue,
            "emotion": emotion,
            "emotion_reason": emotion_reason,
            "action": action,
            "outcome": outcome,
            "subconscious_gap": subconscious_gap,
        }

        # Include discovery info if any new words were found
        if new_words:
            # Enrich with trigger + context (Phase 0.3: vocab_log.jsonl spec)
            enriched = []
            for word, meaning in new_words:
                meta = self.vocab_discovery.discovered.get(word, {})
                # Find matching discovery_record for full context
                ctx = {}
                for rec in reversed(self.vocab_discovery.discovery_log):
                    if rec.get("word") == word:
                        ctx = rec.get("context", {})
                        break
                enriched.append({
                    "word": word,
                    "meaning": meaning,
                    "trigger": meta.get("trigger", ""),
                    "context": ctx,
                })
            result["new_words"] = enriched
            # Backward-compat: also expose as list of (word, meaning) tuples
            result["new_words_tuples"] = [(w, m) for w, m in new_words]
            # Sync discovered vocabulary to persistent memory for persistence
            for word, meaning in new_words:
                self.persistent_memory._discovered_vocabulary[word] = {
                    "meaning": meaning,
                    "tick": tick,
                    "trigger": self.vocab_discovery.discovered.get(word, {}).get("trigger", ""),
                }

        # Phase 4: Store in persistent memory
        self.persistent_memory.store_experience(result, reward, tick)

        # Phase 4: Update behavioral patterns
        situation_key = self.persistent_memory.make_situation_key(world_state, adam_stats)
        self.persistent_memory.update_pattern(situation_key, action, reward)

        return result

    def reset_episode(self):
        """Clear episode memory for a new episode."""
        self.memory.clear()
        self.vocab_discovery.reset_for_episode()

    def add_word(self, word: str):
        """Add a discovered word to Adam's vocabulary."""
        self.thought_gen.add_word(word)

    def get_vocabulary(self) -> list:
        """Return Adam's complete vocabulary."""
        return self.thought_gen.get_all_vocabulary()

    def get_discovered_vocabulary(self) -> dict:
        """Return discovered words with their meanings."""
        return self.vocab_discovery.get_vocabulary_with_meanings()

    def get_discovery_log(self) -> list:
        """Return the full discovery log."""
        return self.vocab_discovery.discovery_log

    def end_episode(self, ticks_survived: int, save: bool = True):
        """End the current episode and update persistent memory."""
        self.persistent_memory.end_episode(ticks_survived)
        # Save to disk periodically (not every episode — too much I/O)
        # Save every 50 episodes for efficiency
        if save and self.persistent_memory.episodes_survived % 50 == 0:
            self.persistent_memory.save_to_disk(self.memory_filepath)

    def get_phase5_signals(self, world_state: dict, adam_stats: dict) -> dict:
        """
        Compute Phase 5 signals for the PPO observation vector.

        Returns dict with:
          - fear_signal: float [0, 1]
          - pleasure_signal: float [0, 1]
          - pattern_confidence: float [0, 1]
          - suggested_action: str or None (the best action for this situation)
        """
        fear = self.persistent_memory.compute_fear_signal(world_state)
        pleasure = self.persistent_memory.compute_pleasure_signal(world_state)
        confidence = self.persistent_memory.compute_pattern_confidence(world_state, adam_stats)

        # Also provide the suggested action for reflection feedback
        situation_key = self.persistent_memory.make_situation_key(world_state, adam_stats)
        suggested = self.persistent_memory.get_best_action_for(situation_key)

        return {
            "fear_signal": fear,
            "pleasure_signal": pleasure,
            "pattern_confidence": confidence,
            "suggested_action": suggested,
        }

    def get_personality(self) -> dict:
        """Get Adam's developing personality summary."""
        return self.persistent_memory.get_personality_summary()

    def get_persistent_memory(self) -> PersistentMemory:
        """Access the persistent memory directly."""
        return self.persistent_memory

    def reflect(self, world_state: dict, adam_stats: dict,
                recent_actions: list = None) -> dict:
        """
        Phase 6: Deep reflection — Adam thinks about what he just did.

        This is the SLOW path. While PPO selects actions instantly (fast path),
        reflection happens AFTER the action, generating deeper insight.

        Dual-speed processing:
          - Fast path: PPO selects action (instant, survival-critical)
          - Slow path: Reflection generates insight (delayed, meaning-making)

        The reflection doesn't change what Adam DID — it changes what he
        UNDERSTANDS about what he did. This is the deepest layer of the Nafs.
        """
        # What pattern does Adam see in his recent actions?
        reflection_parts = []

        if recent_actions and len(recent_actions) >= 3:
            # Check if Adam has been repeating actions
            action_counts = {}
            for a in recent_actions[-5:]:
                action_counts[a] = action_counts.get(a, 0) + 1

            most_common = max(action_counts, key=action_counts.get)
            if action_counts[most_common] >= 3:
                # Adam notices he's been doing the same thing
                reflection_parts.append(f"again. {most_common.lower()}.")

        # Does Adam's current state make sense given his persistent memory?
        personality = self.persistent_memory.get_personality_summary()
        if personality['disposition'] == 'fearful':
            reflection_parts.append("scared. always scared.")
        elif personality['disposition'] == 'confident':
            reflection_parts.append("know this. can do.")

        # What does the persistent memory suggest Adam should do?
        situation_key = self.persistent_memory.make_situation_key(world_state, adam_stats)
        best_action = self.persistent_memory.get_best_action_for(situation_key)
        if best_action:
            action_word = self.thought_gen.ACTION_WORDS.get(best_action, best_action.lower())
            reflection_parts.append(f"before. {action_word}. worked.")

        if not reflection_parts:
            return {
                "reflection": "",
                "has_reflection": False,
            }

        return {
            "reflection": " ".join(reflection_parts),
            "has_reflection": True,
            "suggested_action": best_action,
            "personality": personality['disposition'],
        }

    def format_experience(self, exp: dict, tick: int = 0,
                          adam_stats: dict = None, compact: bool = False) -> str:
        """
        Format an experience dict into a readable string.

        Args:
            exp: experience dict from self.experience()
            tick: current tick number
            adam_stats: Adam's stats for display
            compact: if True, use single-line format

        Returns: str — formatted experience
        """
        if compact:
            thought = exp['thought']
            dialogue = exp.get('dialogue', thought)
            emotion = exp['emotion']
            action = exp['action']
            gap = exp.get('subconscious_gap', {})
            gap_str = f" [gap:{gap.get('gap_ratio', 0):.0%}]" if gap.get('subconscious_active') else ""
            return f'  \U0001f4ad "{thought}" | \U0001f5e3\ufe0f "{dialogue}"{gap_str} | \U0001fac0 {emotion} | \u26a1 {action}'

        # Full format
        lines = []

        # Status line
        if adam_stats:
            h = adam_stats.get('health', 100)
            hu = adam_stats.get('hunger', 0)
            e = adam_stats.get('energy', 100)
            s = adam_stats.get('stress', 0)
            lines.append(f"  Health:{h:>5.0f}  Hunger:{hu:>5.0f}  Energy:{e:>5.0f}  Stress:{s:>5.0f}")

        # World event
        lines.append(f"  \U0001f30d {exp['event']}")

        # Thought
        lines.append(f"  \U0001f4ad {exp['thought']}")

        # Dialogue (what Adam says outwardly)
        dialogue = exp.get('dialogue', exp['thought'])
        if dialogue != exp['thought']:
            gap = exp.get('subconscious_gap', {})
            hidden = gap.get('hidden_words', [])
            lines.append(f"  \U0001f5e3\ufe0f {dialogue}")
            if hidden:
                lines.append(f"  \U0001f535 hidden: {', '.join(hidden)}")
        else:
            lines.append(f"  \U0001f5e3\ufe0f {dialogue}")

        # Action
        lines.append(f"  \u26a1 {exp['action']}")

        # Emotion
        lines.append(f"  \U0001fac0 {exp['emotion']}")

        # Outcome
        lines.append(f"  \u2192 {exp['outcome']}")

        return "\n".join(lines)
