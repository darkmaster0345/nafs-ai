"""
Nafs AI — Culture & Transmission (Phase 9)
==========================================

Implements MD Phase 9: culture is information that spreads between
individuals through behaviour, not genetics. When a baby watches its
parent avoid red berries and learns to avoid them without tasting them
— that is culture.

Covers:
  9.1 Observational Learning
      - Baby watching parent EAT + positive reward → baby's food_preference
        for that item increases
      - Baby watching parent FLEE + positive reward → baby's fear_trigger
        for that tile type increases
      - Rate proportional to TRUST level between observer and observed

  9.2 Cultural Drift
      - Different families develop different food preferences based on
        local environment
      - Cultural signature per agent: weighted vector of preferences/fears
      - Cultural distance between agents

  9.3 Tool Proto-Behaviour
      - Shelter discovery: agent sleeps near mountain → learns shelter reduces cold
      - Fire use: agent near fire at night → positive thermal reward
      - Log proto-tool events (first time any agent exhibits these)

  9.4 Vocabulary Culture
      - Words invented by parents pass to children through dialogue exposure
      - Words not passed (parent dies before child reaches language stage)
        → extinct vocabulary
      - Track vocabulary_lineage: which words survived to Generation 3, 4, 5
      - Vocabulary extinction events logged as cultural losses

Design constraints:
  - Does NOT modify base rewards
  - Standalone module
  - Builds on Phase 7 (FirstContactEngine) + Phase 8 (SocialEngine)

Usage:
    from culture import CultureEngine
    culture = CultureEngine()
    culture.record_observation(observer_id, parent_id, 'EAT', 'blue_berry', reward=0.5, trust=0.7)
    culture.update_cultural_signature(agent_id, biome, action)
    distance = culture.compute_cultural_distance(agent_a, agent_b)
    culture.log_proto_tool_event(agent_id, 'shelter', tick)
"""

import json
import os
import math
import random
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Observational learning
OBSERVATION_LEARNING_RATE = 0.1  # base rate (scaled by trust)
OBSERVATION_MAX_PREFERENCE = 1.0
OBSERVATION_MIN_PREFERENCE = -1.0

# Cultural signature weights
SIGNATURE_FOOD_WEIGHT = 1.0
SIGNATURE_FEAR_WEIGHT = 1.0
SIGNATURE_BEHAVIOR_WEIGHT = 0.5

# Proto-tool event types
PROTO_TOOL_EVENTS = ["shelter_discovery", "fire_use", "cooking_discovery", "tool_use"]

# Vocabulary lineage
LANGUAGE_STAGE_TICKS = 800  # child must reach this age to absorb vocabulary
VOCAB_EXTINCTION_LOG = "vocab_extinctions.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# CultureEngine
# ═══════════════════════════════════════════════════════════════════════════════

class CultureEngine:
    """
    Master culture engine for the Nafs AI world.

    Holds:
      - food_preferences: {agent_id: {food_type: -1.0 to +1.0}}
      - fear_triggers: {agent_id: {biome/danger: 0.0 to 1.0}}
      - behavior_patterns: {agent_id: {action: count}}
      - cultural_signatures: {agent_id: dict} (computed vector)
      - proto_tool_events: list of first-time tool discoveries
      - vocabulary_lineage: {word: {originator, generation, ticks_survived, extinct}}
    """

    def __init__(self, log_path: str = "culture_events.jsonl",
                 seed: Optional[int] = None):
        self.log_path = log_path
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Observational learning state
        self.food_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.fear_triggers: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.behavior_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Cultural signatures (computed on demand)
        self.cultural_signatures: Dict[str, Dict[str, float]] = {}

        # Proto-tool events
        self.proto_tool_events: List[Dict] = []
        self._proto_tool_firsts: Set[str] = set()  # track which types have been logged

        # Vocabulary lineage
        # {word: {originator, origin_tick, generation, last_seen_tick, extinct}}
        self.vocabulary_lineage: Dict[str, Dict] = {}
        self.vocab_extinctions: List[Dict] = []

        # Truncate log
        try:
            with open(self.log_path, "w") as f:
                f.write("")
        except Exception:
            pass

        self.current_tick = 0

    # ─────────────────────────────────────────────────────────────────────────
    # 9.1 Observational Learning
    # ─────────────────────────────────────────────────────────────────────────

    def record_observation(self, observer_id: str, observed_id: str,
                              action: str, target: str,
                              reward: float, trust: float,
                              tick: int) -> Dict:
        """
        Record that observer watched observed perform an action with outcome.

        Updates observer's food_preferences (for EAT) or fear_triggers (for FLEE)
        proportional to trust level.

        Returns dict describing what was learned.
        """
        self.current_tick = tick
        learned = {"learned": False, "type": None, "target": target, "delta": 0.0}

        # Learning rate scales with trust
        rate = OBSERVATION_LEARNING_RATE * max(0.0, trust)

        if action == "EAT" and reward > 0:
            # Positive food preference
            current = self.food_preferences[observer_id].get(target, 0.0)
            new_val = min(OBSERVATION_MAX_PREFERENCE, current + rate)
            self.food_preferences[observer_id][target] = new_val
            learned = {"learned": True, "type": "food_preference",
                       "target": target, "delta": new_val - current}
        elif action == "EAT" and reward < 0:
            # Negative food preference (avoid)
            current = self.food_preferences[observer_id].get(target, 0.0)
            new_val = max(OBSERVATION_MIN_PREFERENCE, current - rate)
            self.food_preferences[observer_id][target] = new_val
            learned = {"learned": True, "type": "food_aversion",
                       "target": target, "delta": new_val - current}
        elif action == "FLEE" and reward > 0:
            # Fear trigger for this target (biome/danger type)
            current = self.fear_triggers[observer_id].get(target, 0.0)
            new_val = min(1.0, current + rate)
            self.fear_triggers[observer_id][target] = new_val
            learned = {"learned": True, "type": "fear_trigger",
                       "target": target, "delta": new_val - current}

        return learned

    def get_food_preference(self, agent_id: str, food_type: str) -> float:
        """Get agent's learned preference for a food type (-1 to +1)."""
        return self.food_preferences.get(agent_id, {}).get(food_type, 0.0)

    def get_fear_trigger(self, agent_id: str, target: str) -> float:
        """Get agent's learned fear for a target (0 to 1)."""
        return self.fear_triggers.get(agent_id, {}).get(target, 0.0)

    def record_behavior(self, agent_id: str, action: str) -> None:
        """Record an agent's behavior for cultural signature computation."""
        self.behavior_patterns[agent_id][action] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # 9.2 Cultural Drift
    # ─────────────────────────────────────────────────────────────────────────

    def update_cultural_signature(self, agent_id: str) -> Dict:
        """
        Compute and store the cultural signature for an agent.

        Signature is a vector of:
          - food_preferences (normalized)
          - fear_triggers (normalized)
          - behavior_patterns (normalized)
        """
        food = self.food_preferences.get(agent_id, {})
        fears = self.fear_triggers.get(agent_id, {})
        behaviors = dict(self.behavior_patterns.get(agent_id, {}))

        # Normalize behaviors to proportions
        total_behavior = sum(behaviors.values()) or 1
        behavior_norm = {k: v / total_behavior for k, v in behaviors.items()}

        signature = {
            "food_preferences": dict(food),
            "fear_triggers": dict(fears),
            "behavior_patterns": behavior_norm,
        }
        self.cultural_signatures[agent_id] = signature
        return signature

    def compute_cultural_distance(self, agent_a: str, agent_b: str) -> float:
        """
        Compute cultural distance between two agents.
        Returns 0.0 (identical) to 1.0 (completely different).
        """
        sig_a = self.cultural_signatures.get(agent_a) or self.update_cultural_signature(agent_a)
        sig_b = self.cultural_signatures.get(agent_b) or self.update_cultural_signature(agent_b)

        # Compare food preferences
        all_foods = set(sig_a.get("food_preferences", {}).keys()) | \
                    set(sig_b.get("food_preferences", {}).keys())
        food_dist = 0.0
        for food in all_foods:
            va = sig_a.get("food_preferences", {}).get(food, 0.0)
            vb = sig_b.get("food_preferences", {}).get(food, 0.0)
            food_dist += abs(va - vb)
        food_dist = food_dist / max(1, len(all_foods))

        # Compare fear triggers
        all_fears = set(sig_a.get("fear_triggers", {}).keys()) | \
                    set(sig_b.get("fear_triggers", {}).keys())
        fear_dist = 0.0
        for fear in all_fears:
            va = sig_a.get("fear_triggers", {}).get(fear, 0.0)
            vb = sig_b.get("fear_triggers", {}).get(fear, 0.0)
            fear_dist += abs(va - vb)
        fear_dist = fear_dist / max(1, len(all_fears))

        # Compare behavior patterns
        all_behaviors = set(sig_a.get("behavior_patterns", {}).keys()) | \
                        set(sig_b.get("behavior_patterns", {}).keys())
        behavior_dist = 0.0
        for b in all_behaviors:
            va = sig_a.get("behavior_patterns", {}).get(b, 0.0)
            vb = sig_b.get("behavior_patterns", {}).get(b, 0.0)
            behavior_dist += abs(va - vb)
        behavior_dist = behavior_dist / max(1, len(all_behaviors))

        # Weighted average
        total_dist = (SIGNATURE_FOOD_WEIGHT * food_dist +
                       SIGNATURE_FEAR_WEIGHT * fear_dist +
                       SIGNATURE_BEHAVIOR_WEIGHT * behavior_dist) / \
                      (SIGNATURE_FOOD_WEIGHT + SIGNATURE_FEAR_WEIGHT + SIGNATURE_BEHAVIOR_WEIGHT)
        return min(1.0, total_dist)

    def get_cultural_signature(self, agent_id: str) -> Dict:
        """Get the cached cultural signature for an agent."""
        return self.cultural_signatures.get(agent_id, {})

    # ─────────────────────────────────────────────────────────────────────────
    # 9.3 Tool Proto-Behaviour
    # ─────────────────────────────────────────────────────────────────────────

    def log_proto_tool_event(self, agent_id: str, event_type: str,
                                tick: int, location: Tuple[int, int] = (0, 0),
                                details: Optional[Dict] = None) -> Optional[Dict]:
        """
        Log a proto-tool discovery event.

        Only logs the FIRST time an event_type is observed.
        Returns the event dict if this was a first, None otherwise.
        """
        if event_type not in PROTO_TOOL_EVENTS:
            return None

        # Only log first occurrence
        if event_type in self._proto_tool_firsts:
            return None

        self._proto_tool_firsts.add(event_type)
        event = {
            "event": "PROTO_TOOL_DISCOVERY",
            "event_type": event_type,
            "agent_id": agent_id,
            "tick": tick,
            "location": list(location),
            "details": details or {},
        }
        self.proto_tool_events.append(event)
        self._log_event(event)
        return event

    def detect_shelter_use(self, agent_id: str, biome: str, action: str,
                             body_temp: float, tick: int,
                             location: Tuple[int, int] = (0, 0)) -> Optional[Dict]:
        """
        Detect shelter use: agent sleeps in mountain/cave biome while cold.
        Returns event dict if first discovery, None otherwise.
        """
        if action == "SLEEP" and biome in ("mountain", "cave") and body_temp < 35:
            return self.log_proto_tool_event(
                agent_id, "shelter_discovery", tick, location,
                {"biome": biome, "body_temp": body_temp},
            )
        return None

    def detect_fire_use(self, agent_id: str, adjacent_fire_count: int,
                          body_temp: float, time_of_day: int,
                          tick: int, location: Tuple[int, int] = (0, 0)) -> Optional[Dict]:
        """
        Detect fire use: agent adjacent to fire at night while cold.
        """
        is_night = time_of_day >= 18 or time_of_day < 6
        if adjacent_fire_count > 0 and is_night and body_temp < 35:
            return self.log_proto_tool_event(
                agent_id, "fire_use", tick, location,
                {"adjacent_fires": adjacent_fire_count, "body_temp": body_temp,
                 "time_of_day": time_of_day},
            )
        return None

    def detect_cooking(self, agent_id: str, food_type: str,
                         tick: int, location: Tuple[int, int] = (0, 0)) -> Optional[Dict]:
        """
        Detect cooking discovery: agent eats cooked food for the first time.
        """
        if food_type.startswith("cooked_"):
            return self.log_proto_tool_event(
                agent_id, "cooking_discovery", tick, location,
                {"food_type": food_type},
            )
        return None

    def get_proto_tool_events(self) -> List[Dict]:
        return list(self.proto_tool_events)

    # ─────────────────────────────────────────────────────────────────────────
    # 9.4 Vocabulary Culture
    # ─────────────────────────────────────────────────────────────────────────

    def record_word_invention(self, word: str, originator_id: str,
                                tick: int, generation: int = 1) -> None:
        """Record that an agent invented a new word."""
        if word not in self.vocabulary_lineage:
            self.vocabulary_lineage[word] = {
                "originator": originator_id,
                "origin_tick": tick,
                "generation": generation,
                "last_seen_tick": tick,
                "extinct": False,
                "extinct_tick": None,
            }
            self._log_event({
                "event": "WORD_INVENTED",
                "word": word,
                "originator": originator_id,
                "tick": tick,
                "generation": generation,
            })
        else:
            # Word already exists — update last seen
            self.vocabulary_lineage[word]["last_seen_tick"] = tick

    def record_word_used(self, word: str, tick: int) -> None:
        """Record that a word was used (keeps it alive)."""
        if word in self.vocabulary_lineage:
            self.vocabulary_lineage[word]["last_seen_tick"] = tick

    def check_vocab_extinction(self, current_tick: int,
                                 extinction_threshold: int = 5000) -> List[Dict]:
        """
        Check for vocabulary extinction.
        A word is extinct if it hasn't been used in `extinction_threshold` ticks.
        """
        newly_extinct = []
        for word, info in self.vocabulary_lineage.items():
            if info["extinct"]:
                continue
            if current_tick - info["last_seen_tick"] > extinction_threshold:
                info["extinct"] = True
                info["extinct_tick"] = current_tick
                event = {
                    "event": "VOCAB_EXTINCTION",
                    "word": word,
                    "originator": info["originator"],
                    "origin_tick": info["origin_tick"],
                    "extinct_tick": current_tick,
                    "ticks_survived": current_tick - info["origin_tick"],
                    "generation_originated": info["generation"],
                }
                self.vocab_extinctions.append(event)
                self._log_event(event)
                newly_extinct.append(event)
        return newly_extinct

    def transfer_vocabulary_to_child(self, parent_id: str, child_id: str,
                                       parent_vocab: List[str],
                                       child_age_ticks: int, tick: int) -> Dict:
        """
        Transfer parent's vocabulary to child through dialogue exposure.

        Only transfers if child has reached language stage (LANGUAGE_STAGE_TICKS).
        Returns dict with transfer stats.
        """
        if child_age_ticks < LANGUAGE_STAGE_TICKS:
            return {
                "transferred": False,
                "reason": "child_too_young",
                "words_transferred": 0,
            }

        # Transfer all parent words to child's lineage
        transferred = 0
        for word in parent_vocab:
            if word in self.vocabulary_lineage:
                # Mark word as seen (kept alive by transmission)
                self.record_word_used(word, tick)
                transferred += 1

        return {
            "transferred": True,
            "words_transferred": transferred,
            "child_id": child_id,
            "parent_id": parent_id,
        }

    def get_vocab_lineage_summary(self) -> Dict:
        """Get summary of vocabulary lineage."""
        total = len(self.vocabulary_lineage)
        extinct = sum(1 for info in self.vocabulary_lineage.values() if info["extinct"])
        alive = total - extinct
        return {
            "total_words": total,
            "alive": alive,
            "extinct": extinct,
            "extinction_rate": extinct / max(1, total),
        }

    def get_surviving_words_by_generation(self) -> Dict[int, List[str]]:
        """Group surviving words by the generation that originated them."""
        result = defaultdict(list)
        for word, info in self.vocabulary_lineage.items():
            if not info["extinct"]:
                result[info["generation"]].append(word)
        return dict(result)

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_event(self, event: Dict) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[Culture] Failed to write log: {e}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        return {
            "total_agents_with_preferences": len(self.food_preferences),
            "total_proto_tool_events": len(self.proto_tool_events),
            "vocab_total": len(self.vocabulary_lineage),
            "vocab_extinct": len(self.vocab_extinctions),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing CultureEngine...")

    culture = CultureEngine(seed=42, log_path="/tmp/test_culture.jsonl")

    # Test 9.1: Observational learning
    learned = culture.record_observation(
        observer_id="baby_1", observed_id="adam",
        action="EAT", target="blue_berry",
        reward=0.5, trust=0.7, tick=100,
    )
    print(f"  Observation (EAT blue_berry, reward +, trust 0.7): {learned}")
    assert learned["learned"]
    assert learned["type"] == "food_preference"
    pref = culture.get_food_preference("baby_1", "blue_berry")
    print(f"  Baby's preference for blue_berry: {pref:.3f}")
    assert pref > 0

    # Negative food preference (avoid)
    culture.record_observation(
        "baby_1", "adam", "EAT", "red_berry",
        reward=-0.5, trust=0.7, tick=110,
    )
    pref_red = culture.get_food_preference("baby_1", "red_berry")
    print(f"  Baby's preference for red_berry: {pref_red:.3f}")
    assert pref_red < 0

    # Fear trigger
    culture.record_observation(
        "baby_1", "adam", "FLEE", "desert",
        reward=0.3, trust=0.5, tick=120,
    )
    fear = culture.get_fear_trigger("baby_1", "desert")
    print(f"  Baby's fear of desert: {fear:.3f}")
    assert fear > 0

    # Trust affects learning rate
    culture2 = CultureEngine(seed=42, log_path="/tmp/test_culture2.jsonl")
    culture2.record_observation("a", "b", "EAT", "berry", 0.5, trust=0.9, tick=100)
    culture2.record_observation("c", "b", "EAT", "berry", 0.5, trust=0.1, tick=100)
    pref_high_trust = culture2.get_food_preference("a", "berry")
    pref_low_trust = culture2.get_food_preference("c", "berry")
    print(f"  High trust pref: {pref_high_trust:.3f}, Low trust: {pref_low_trust:.3f}")
    assert pref_high_trust > pref_low_trust

    # Test 9.2: Cultural drift + distance
    culture3 = CultureEngine(seed=42, log_path="/tmp/test_culture3.jsonl")
    # Family A: forest, likes berries
    for _ in range(10):
        culture3.record_observation("adam", "forest_parent", "EAT", "blue_berry", 0.5, 0.9, 100)
    # Family B: swamp, likes fish
    for _ in range(10):
        culture3.record_observation("eve", "swamp_parent", "EAT", "fish", 0.5, 0.9, 100)
    culture3.update_cultural_signature("adam")
    culture3.update_cultural_signature("eve")
    distance = culture3.compute_cultural_distance("adam", "eve")
    print(f"  Cultural distance adam↔eve: {distance:.3f}")
    assert distance > 0

    # Same agent → distance 0
    assert culture3.compute_cultural_distance("adam", "adam") == 0.0

    # Test 9.3: Proto-tool events
    culture4 = CultureEngine(seed=42, log_path="/tmp/test_culture4.jsonl")
    # Shelter discovery: cold agent sleeps in cave
    event = culture4.detect_shelter_use(
        agent_id="adam", biome="cave", action="SLEEP",
        body_temp=28.0, tick=500, location=(5, 5),
    )
    print(f"  Shelter discovery: {event}")
    assert event is not None
    assert event["event_type"] == "shelter_discovery"

    # Second shelter discovery → None (already logged)
    event2 = culture4.detect_shelter_use("eve", "cave", "SLEEP", 28.0, 600)
    assert event2 is None

    # Fire use
    event = culture4.detect_fire_use(
        agent_id="adam", adjacent_fire_count=1,
        body_temp=30.0, time_of_day=22, tick=700,
    )
    print(f"  Fire use: {event}")
    assert event is not None

    # Cooking
    event = culture4.detect_cooking("adam", "cooked_meat", 800)
    print(f"  Cooking: {event}")
    assert event is not None

    # Test 9.4: Vocabulary lineage
    culture5 = CultureEngine(seed=42, log_path="/tmp/test_culture5.jsonl")
    culture5.record_word_invention("cold pain", "adam", tick=100, generation=1)
    culture5.record_word_invention("warm good", "eve", tick=110, generation=1)
    culture5.record_word_invention("big bad", "adam", tick=200, generation=1)

    # Word used recently — survives (used at tick 5000, well within threshold)
    culture5.record_word_used("cold pain", tick=5000)

    # Check extinction (no use in 1000 ticks)
    # "cold pain": last seen at 5000 → 5500-5000 = 500 < 1000 → survives
    # "warm good": last seen at 110 → 5500-110 > 1000 → extinct
    # "big bad": last seen at 200 → 5500-200 > 1000 → extinct
    extinctions = culture5.check_vocab_extinction(current_tick=5500, extinction_threshold=1000)
    print(f"  Extinctions at tick 5500: {len(extinctions)}")
    assert len(extinctions) == 2

    # Transfer to child
    culture6 = CultureEngine(seed=42, log_path="/tmp/test_culture6.jsonl")
    culture6.record_word_invention("test_word", "adam", tick=100, generation=1)
    result = culture6.transfer_vocabulary_to_child(
        parent_id="adam", child_id="baby",
        parent_vocab=["test_word"],
        child_age_ticks=900,  # > LANGUAGE_STAGE_TICKS (800)
        tick=1000,
    )
    print(f"  Vocab transfer: {result}")
    assert result["transferred"]
    assert result["words_transferred"] == 1

    # Too young
    result = culture6.transfer_vocabulary_to_child(
        "adam", "baby2", ["test_word"], child_age_ticks=500, tick=1000,
    )
    assert not result["transferred"]

    # Lineage summary
    summary = culture6.get_vocab_lineage_summary()
    print(f"  Vocab lineage: {summary}")
    assert summary["total_words"] >= 1

    print("\n✓ CultureEngine self-test passed")
