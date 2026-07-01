"""
Nafs AI — First Contact & Interaction (Phase 7)
================================================

Implements MD Phase 7: Adam and Eve have been developing separately. Now
they meet. This is the most important moment in the simulation — two
different minds, different vocabularies, different experiences, encountering
each other for the first time.

Covers:
  7.1 First Contact Event
      - Detect first tick where Adam and Eve are within 5 tiles
      - Log: {tick, adam_position, eve_position, adam_vocab_size, eve_vocab_size}
      - FIRST_CONTACT event
      - Other_agent sensory signal: distance, direction, size, movement_state

  7.2 Interaction Actions
      - OBSERVE: watch another agent (no energy cost, gives info)
      - APPROACH: move toward another agent
      - FLEE_AGENT: flee from another agent
      - SHARE: drop food item at current tile (other agent can pick up)
      - FOLLOW: trail another agent at 2-tile distance
      - Unlocked at adolescent stage — babies cannot interact

  7.3 Social Reward Shaping
      - Sharing food when other agent is hungry → +reward for sharer
      - Following agent who finds food → +reward for follower
      - Flee response to aggressive approach → -reward for aggressor
      - Trust metric: accumulated positive/negative interactions per agent pair

  7.4 Vocabulary Contact
      - When agents are within 2 tiles, their DIALOGUE outputs are in each
        other's sensory vector
      - Dialogue is just a token sequence — other agent cannot 'understand'
        it initially
      - Over hundreds of ticks of proximity: reward shaping creates
        correlation between heard tokens and outcomes
      - Primitive communication emerges: agent A says word → agent B learns
        word precedes food

Design constraints:
  - Does NOT modify base rewards in world_sim.py
  - Standalone module
  - Adds new interaction actions + reward signals on top of existing base

Usage:
    from first_contact import FirstContactEngine
    fc = FirstContactEngine()
    fc.check_first_contact(adam_pos, eve_pos, adam_vocab, eve_vocab, tick)
    signal = fc.get_other_agent_signal(observer_pos, other_pos, other_state)
    reward = fc.shape_social_reward(action, observer_state, other_state)
"""

import json
import os
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

FIRST_CONTACT_DISTANCE = 5     # tiles — triggers FIRST_CONTACT event
VOCAB_CONTACT_DISTANCE = 2     # tiles — dialogue in other's sensory vector
FOLLOW_DISTANCE = 2            # tiles — trail distance for FOLLOW action
SHARE_HUNGER_THRESHOLD = 50.0  # other agent hunger > 50 → SHARE reward
TRUST_GAIN_POSITIVE = 0.1      # per positive interaction
TRUST_GAIN_NEGATIVE = -0.3     # per negative interaction (asymmetric — trust hard to build, easy to lose)
TRUST_INITIAL = 0.0            # strangers start at 0 trust
TRUST_MAX = 1.0
TRUST_MIN = -1.0

# Interaction actions (Phase 7.2)
INTERACTION_ACTIONS = ["OBSERVE", "APPROACH", "FLEE_AGENT", "SHARE", "FOLLOW"]

# Action unlock thresholds (by life stage)
ACTION_UNLOCK_STAGE = {
    "OBSERVE": "child",        # children can observe
    "APPROACH": "adolescent",  # adolescents can approach
    "FLEE_AGENT": "adolescent",
    "SHARE": "adolescent",
    "FOLLOW": "adolescent",
}


# ═══════════════════════════════════════════════════════════════════════════════
# FirstContactEngine
# ═══════════════════════════════════════════════════════════════════════════════

class FirstContactEngine:
    """
    Master first contact + interaction engine for the Nafs AI world.

    Holds:
      - first_contact_event: dict or None (first time within 5 tiles)
      - trust_scores: {(agent_a, agent_b): float -1 to +1}
      - interaction_history: per-pair list of recent interactions
      - vocabulary_contact_log: when agents were within 2 tiles

    Each agent pair has a bidirectional trust score.
    """

    def __init__(self, log_path: str = "first_contact.jsonl",
                 seed: Optional[int] = None):
        self.log_path = log_path
        self.rng = random.Random(seed or random.randint(0, 999999))

        # First contact event
        self.first_contact_event: Optional[Dict] = None

        # Trust scores: {(agent_a, agent_b): float}
        # Always stored with agents in sorted order for consistency
        self.trust_scores: Dict[Tuple[str, str], float] = defaultdict(lambda: TRUST_INITIAL)

        # Interaction history per pair: {(a, b): [interactions]}
        self.interaction_history: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

        # Vocabulary contact tracking
        self.vocab_contact_log: List[Dict] = []
        self.vocab_contact_ticks: Dict[Tuple[str, str], int] = defaultdict(int)  # total ticks in contact

        # Heard dialogues: {agent_id: [list of (tick, speaker, dialogue)]}
        self.heard_dialogues: Dict[str, List[Dict]] = defaultdict(list)

        # Truncate log at start
        try:
            with open(self.log_path, "w") as f:
                f.write("")
        except Exception:
            pass

        self.current_tick = 0

    def _pair_key(self, a: str, b: str) -> Tuple[str, str]:
        """Sort agent IDs for consistent pair key."""
        return (a, b) if a <= b else (b, a)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.1 First Contact Event
    # ─────────────────────────────────────────────────────────────────────────

    def check_first_contact(self, adam_pos: Tuple[int, int],
                              eve_pos: Tuple[int, int],
                              adam_vocab_size: int,
                              eve_vocab_size: int,
                              tick: int,
                              adam_id: str = "adam",
                              eve_id: str = "eve") -> Optional[Dict]:
        """
        Check if this is the first time Adam and Eve are within 5 tiles.
        Returns the FIRST_CONTACT event dict, or None if no event.
        """
        if self.first_contact_event is not None:
            return None  # already happened

        dist = abs(adam_pos[0] - eve_pos[0]) + abs(adam_pos[1] - eve_pos[1])
        if dist > FIRST_CONTACT_DISTANCE:
            return None

        event = {
            "event": "FIRST_CONTACT",
            "tick": tick,
            "adam_id": adam_id,
            "eve_id": eve_id,
            "adam_position": list(adam_pos),
            "eve_position": list(eve_pos),
            "distance": dist,
            "adam_vocabulary_size": adam_vocab_size,
            "eve_vocabulary_size": eve_vocab_size,
        }
        self.first_contact_event = event
        self._log_event(event)
        return event

    def get_first_contact_event(self) -> Optional[Dict]:
        return self.first_contact_event

    # ─────────────────────────────────────────────────────────────────────────
    # Other-agent sensory signal
    # ─────────────────────────────────────────────────────────────────────────

    def get_other_agent_signal(self, observer_pos: Tuple[int, int],
                                 other_pos: Tuple[int, int],
                                 other_state: Optional[Dict] = None
                                 ) -> Dict:
        """
        Returns sensory signal about another agent.

        Fields:
          - other_presence: 0-1 (1 if within sight range, decays with distance)
          - other_distance: tile distance
          - other_direction: N/S/E/W or none
          - other_size: 0-1 (proxy for life stage — adult=1.0, baby=0.3)
          - other_movement_state: 'idle' / 'moving' / 'sleeping' / 'fleeing'
        """
        dx = other_pos[0] - observer_pos[0]
        dy = other_pos[1] - observer_pos[1]
        dist = abs(dx) + abs(dy)

        if dist > FIRST_CONTACT_DISTANCE:
            return {
                "other_presence": 0.0,
                "other_distance": dist,
                "other_direction": "none",
                "other_size": 0.0,
                "other_movement_state": "unknown",
            }

        # Presence decays with distance
        presence = 1.0 - (dist / float(FIRST_CONTACT_DISTANCE + 1))

        # Direction
        if abs(dx) > abs(dy):
            direction = "E" if dx > 0 else "W"
        elif dy != 0:
            direction = "S" if dy > 0 else "N"
        else:
            direction = "here"  # same tile

        # Size proxy from life stage (default 1.0 if unknown)
        size = 1.0
        if other_state:
            stage = other_state.get("life_stage", "adult")
            size = {"newborn": 0.3, "child": 0.5, "adolescent": 0.7,
                    "adult": 1.0, "elder": 0.9, "ancient": 0.8}.get(stage, 1.0)

        movement = "idle"
        if other_state:
            movement = other_state.get("movement_state", "idle")

        return {
            "other_presence": round(presence, 3),
            "other_distance": dist,
            "other_direction": direction,
            "other_size": size,
            "other_movement_state": movement,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 7.2 Interaction Actions
    # ─────────────────────────────────────────────────────────────────────────

    def is_action_unlocked(self, action: str, life_stage: str) -> bool:
        """Check if an interaction action is unlocked for this life stage."""
        required = ACTION_UNLOCK_STAGE.get(action)
        if required is None:
            return True  # not an interaction action — always available

        stage_order = ["newborn", "child", "adolescent", "adult", "elder", "ancient"]
        try:
            return stage_order.index(life_stage) >= stage_order.index(required)
        except ValueError:
            return False

    def execute_interaction(self, action: str,
                              actor_id: str, actor_pos: Tuple[int, int],
                              actor_state: Dict,
                              target_id: str, target_pos: Tuple[int, int],
                              target_state: Dict,
                              tick: int) -> Dict:
        """
        Execute an interaction action.

        Returns dict with:
          - success: bool
          - reward: float (social reward for actor)
          - target_reward: float (reward for target, if any)
          - new_position: Optional[Tuple] (if action moves the actor)
          - info: Dict (extra info for logging)
        """
        self.current_tick = tick

        if action not in INTERACTION_ACTIONS:
            return {"success": False, "reward": 0.0, "target_reward": 0.0,
                    "new_position": None, "info": {"reason": "invalid_action"}}

        # Check action unlocked
        actor_stage = actor_state.get("life_stage", "adult")
        if not self.is_action_unlocked(action, actor_stage):
            return {"success": False, "reward": 0.0, "target_reward": 0.0,
                    "new_position": None, "info": {"reason": "action_locked"}}

        # Distance check
        dist = abs(actor_pos[0] - target_pos[0]) + abs(actor_pos[1] - target_pos[1])
        if dist > FIRST_CONTACT_DISTANCE:
            return {"success": False, "reward": 0.0, "target_reward": 0.0,
                    "new_position": None, "info": {"reason": "too_far"}}

        if action == "OBSERVE":
            return self._do_observe(actor_id, actor_pos, actor_state,
                                     target_id, target_pos, target_state, dist, tick)
        elif action == "APPROACH":
            return self._do_approach(actor_id, target_id, actor_pos, target_pos, dist, tick)
        elif action == "FLEE_AGENT":
            return self._do_flee_agent(actor_id, target_id, actor_pos, target_pos, dist, tick)
        elif action == "SHARE":
            return self._do_share(actor_id, actor_state, target_id, target_state, dist, tick)
        elif action == "FOLLOW":
            return self._do_follow(actor_id, target_id, actor_pos, target_pos, dist, tick)

        return {"success": False, "reward": 0.0, "target_reward": 0.0,
                "new_position": None, "info": {"reason": "unknown_action"}}

    def _do_observe(self, actor_id, actor_pos, actor_state,
                     target_id, target_pos, target_state, dist, tick):
        """OBSERVE: watch another agent. No energy cost, gives info."""
        # Recording this interaction
        self._record_interaction(actor_id, target_id, "OBSERVE", +0.05, tick)
        return {
            "success": True,
            "reward": 0.05,  # small positive for gaining info
            "target_reward": 0.0,
            "new_position": None,
            "info": {"observed_target_state": target_state},
        }

    def _do_approach(self, actor_id, target_id, actor_pos, target_pos, dist, tick):
        """APPROACH: move toward another agent."""
        if dist <= 1:
            return {"success": False, "reward": 0.0, "target_reward": 0.0,
                    "new_position": None, "info": {"reason": "already_adjacent"}}
        # Move one tile toward target
        dx = target_pos[0] - actor_pos[0]
        dy = target_pos[1] - actor_pos[1]
        if abs(dx) >= abs(dy):
            new_pos = (actor_pos[0] + (1 if dx > 0 else -1), actor_pos[1])
        else:
            new_pos = (actor_pos[0], actor_pos[1] + (1 if dy > 0 else -1))
        self._record_interaction(actor_id, target_id, "APPROACH", 0.0, tick)
        return {
            "success": True,
            "reward": 0.0,
            "target_reward": 0.0,
            "new_position": new_pos,
            "info": {"approached": True},
        }

    def _do_flee_agent(self, actor_id, target_id, actor_pos, target_pos, dist, tick):
        """FLEE_AGENT: flee from another agent."""
        if dist >= FIRST_CONTACT_DISTANCE:
            return {"success": False, "reward": 0.0, "target_reward": 0.0,
                    "new_position": None, "info": {"reason": "already_far"}}
        # Move one tile away from target
        dx = actor_pos[0] - target_pos[0]
        dy = actor_pos[1] - target_pos[1]
        if abs(dx) >= abs(dy):
            new_pos = (actor_pos[0] + (1 if dx > 0 else -1 if dx < 0 else 0), actor_pos[1])
        else:
            new_pos = (actor_pos[0], actor_pos[1] + (1 if dy > 0 else -1 if dy < 0 else 0))
        self._record_interaction(actor_id, target_id, "FLEE_AGENT", -0.1, tick)
        return {
            "success": True,
            "reward": -0.05,  # small cost for fleeing
            "target_reward": -0.1,  # target gets negative signal (aggressor penalty if target was approaching)
            "new_position": new_pos,
            "info": {"fled": True},
        }

    def _do_share(self, actor_id, actor_state, target_id, target_state, dist, tick):
        """SHARE: drop food item at current tile (other agent can pick up)."""
        target_hunger = target_state.get("hunger", 0)
        if target_hunger > SHARE_HUNGER_THRESHOLD:
            # Positive reward for sharer when target is hungry
            self._record_interaction(actor_id, target_id, "SHARE", +0.5, tick)
            return {
                "success": True,
                "reward": 0.5,  # altruism reward
                "target_reward": 0.3,  # target benefits
                "new_position": None,
                "info": {"shared_with_hungry": True, "target_hunger": target_hunger},
            }
        else:
            # Small reward even if target not hungry (still a positive gesture)
            self._record_interaction(actor_id, target_id, "SHARE", +0.1, tick)
            return {
                "success": True,
                "reward": 0.1,
                "target_reward": 0.1,
                "new_position": None,
                "info": {"shared_with_hungry": False, "target_hunger": target_hunger},
            }

    def _do_follow(self, actor_id, target_id, actor_pos, target_pos, dist, tick):
        """FOLLOW: trail another agent at 2-tile distance."""
        if dist < FOLLOW_DISTANCE:
            # Too close — back off
            dx = actor_pos[0] - target_pos[0]
            dy = actor_pos[1] - target_pos[1]
            new_pos = (actor_pos[0] + (1 if dx >= 0 else -1), actor_pos[1])
            return {
                "success": True,
                "reward": 0.0,
                "target_reward": 0.0,
                "new_position": new_pos,
                "info": {"backing_off": True},
            }
        elif dist > FOLLOW_DISTANCE + 1:
            # Too far — move closer
            dx = target_pos[0] - actor_pos[0]
            dy = target_pos[1] - actor_pos[1]
            if abs(dx) >= abs(dy):
                new_pos = (actor_pos[0] + (1 if dx > 0 else -1), actor_pos[1])
            else:
                new_pos = (actor_pos[0], actor_pos[1] + (1 if dy > 0 else -1))
            self._record_interaction(actor_id, target_id, "FOLLOW", +0.05, tick)
            return {
                "success": True,
                "reward": 0.05,
                "target_reward": 0.0,
                "new_position": new_pos,
                "info": {"following": True},
            }
        else:
            # At follow distance — just observe
            self._record_interaction(actor_id, target_id, "FOLLOW", +0.05, tick)
            return {
                "success": True,
                "reward": 0.05,
                "target_reward": 0.0,
                "new_position": None,
                "info": {"at_follow_distance": True},
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 7.3 Social Reward Shaping + Trust
    # ─────────────────────────────────────────────────────────────────────────

    def _record_interaction(self, actor_id: str, target_id: str,
                              action: str, reward_delta: float, tick: int) -> None:
        """Record an interaction and update trust."""
        pair = self._pair_key(actor_id, target_id)
        self.interaction_history[pair].append({
            "tick": tick,
            "actor": actor_id,
            "target": target_id,
            "action": action,
            "reward_delta": reward_delta,
        })

        # Update trust score
        if reward_delta > 0:
            self.trust_scores[pair] = min(TRUST_MAX,
                                           self.trust_scores[pair] + TRUST_GAIN_POSITIVE)
        elif reward_delta < 0:
            self.trust_scores[pair] = max(TRUST_MIN,
                                           self.trust_scores[pair] + TRUST_GAIN_NEGATIVE)

    def get_trust(self, agent_a: str, agent_b: str) -> float:
        """Get trust score between two agents (-1 to +1)."""
        pair = self._pair_key(agent_a, agent_b)
        return self.trust_scores[pair]

    def get_trust_label(self, trust: float) -> str:
        """Convert trust score to human-readable label."""
        if trust < -0.5: return "HOSTILE"
        if trust < -0.1: return "DISTRUSTED"
        if trust < 0.1: return "STRANGER"
        if trust < 0.5: return "FAMILIAR"
        return "TRUSTED"

    def get_relationship_summary(self, agent_a: str, agent_b: str) -> Dict:
        """Get full relationship summary for a pair."""
        pair = self._pair_key(agent_a, agent_b)
        trust = self.trust_scores[pair]
        history = self.interaction_history.get(pair, [])
        return {
            "trust": round(trust, 3),
            "trust_label": self.get_trust_label(trust),
            "total_interactions": len(history),
            "positive_interactions": sum(1 for h in history if h["reward_delta"] > 0),
            "negative_interactions": sum(1 for h in history if h["reward_delta"] < 0),
            "last_interaction_tick": history[-1]["tick"] if history else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 7.4 Vocabulary Contact
    # ─────────────────────────────────────────────────────────────────────────

    def check_vocabulary_contact(self, adam_pos: Tuple[int, int],
                                   eve_pos: Tuple[int, int],
                                   adam_dialogue: str,
                                   eve_dialogue: str,
                                   tick: int,
                                   adam_id: str = "adam",
                                   eve_id: str = "eve") -> Dict:
        """
        Check if agents are within vocab contact distance.
        If so, exchange dialogue tokens in each other's sensory vector.

        Returns dict with:
          - in_contact: bool
          - adam_heard: str (dialogue Adam heard from Eve)
          - eve_heard: str (dialogue Eve heard from Adam)
        """
        dist = abs(adam_pos[0] - eve_pos[0]) + abs(adam_pos[1] - eve_pos[1])
        if dist > VOCAB_CONTACT_DISTANCE:
            return {
                "in_contact": False,
                "adam_heard": "",
                "eve_heard": "",
            }

        # In contact — exchange dialogues
        self.vocab_contact_ticks[self._pair_key(adam_id, eve_id)] += 1

        # Record heard dialogues
        self.heard_dialogues[adam_id].append({
            "tick": tick,
            "speaker": eve_id,
            "dialogue": eve_dialogue,
        })
        self.heard_dialogues[eve_id].append({
            "tick": tick,
            "speaker": adam_id,
            "dialogue": adam_dialogue,
        })

        # Log vocabulary contact
        self._log_event({
            "event": "VOCAB_CONTACT",
            "tick": tick,
            "adam_id": adam_id,
            "eve_id": eve_id,
            "distance": dist,
            "adam_dialogue": adam_dialogue,
            "eve_dialogue": eve_dialogue,
        })

        return {
            "in_contact": True,
            "adam_heard": eve_dialogue,
            "eve_heard": adam_dialogue,
        }

    def get_heard_dialogues(self, agent_id: str, last_n: int = 10) -> List[Dict]:
        """Get recent dialogues heard by an agent."""
        return self.heard_dialogues.get(agent_id, [])[-last_n:]

    def get_total_vocab_contact_ticks(self, agent_a: str, agent_b: str) -> int:
        """Total ticks the two agents have been in vocab contact."""
        return self.vocab_contact_ticks.get(self._pair_key(agent_a, agent_b), 0)

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_event(self, event: Dict) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[FirstContact] Failed to write log: {e}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Sensory extensions
    # ─────────────────────────────────────────────────────────────────────────

    def get_sensory_extensions(self, observer_id: str,
                                 observer_pos: Tuple[int, int],
                                 other_id: str,
                                 other_pos: Tuple[int, int],
                                 other_state: Optional[Dict] = None) -> Dict:
        """
        Returns Phase 7 sensory extensions:
          - other_presence: 0-1
          - other_distance: tiles
          - other_direction: N/S/E/W/none/here
          - other_size: 0-1
          - other_movement_state: str
          - trust_score: -1 to +1
          - trust_label: str
          - vocab_contact_ticks: int (total time in vocab contact)
        """
        signal = self.get_other_agent_signal(observer_pos, other_pos, other_state)
        trust = self.get_trust(observer_id, other_id)
        return {
            **signal,
            'trust_score': round(trust, 3),
            'trust_label': self.get_trust_label(trust),
            'vocab_contact_ticks': self.get_total_vocab_contact_ticks(observer_id, other_id),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        return {
            "first_contact_happened": self.first_contact_event is not None,
            "first_contact_tick": (self.first_contact_event or {}).get("tick"),
            "total_pairs": len(self.trust_scores),
            "total_interactions": sum(len(h) for h in self.interaction_history.values()),
            "total_vocab_contact_ticks": sum(self.vocab_contact_ticks.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing FirstContactEngine...")

    fc = FirstContactEngine(seed=42, log_path="/tmp/test_fc.jsonl")

    # Test 7.1: First contact detection
    event = fc.check_first_contact(
        adam_pos=(5, 5), eve_pos=(7, 7),
        adam_vocab_size=24, eve_vocab_size=20,
        tick=100,
    )
    print(f"  First contact (dist=4): {event}")
    assert event is not None
    assert event["distance"] == 4
    assert event["adam_vocabulary_size"] == 24

    # Second call — already happened, returns None
    event2 = fc.check_first_contact((5, 5), (6, 6), 25, 21, tick=110)
    assert event2 is None

    # Test too far
    fc2 = FirstContactEngine(seed=42, log_path="/tmp/test_fc2.jsonl")
    event = fc2.check_first_contact((0, 0), (10, 10), 20, 18, tick=100)
    assert event is None  # too far

    # Test other-agent sensory signal
    signal = fc.get_other_agent_signal(
        observer_pos=(5, 5), other_pos=(5, 7),
        other_state={"life_stage": "adult", "movement_state": "moving"},
    )
    print(f"  Other agent signal: {signal}")
    assert signal["other_presence"] > 0
    assert signal["other_direction"] == "S"
    assert signal["other_size"] == 1.0

    # Test out of range
    signal = fc.get_other_agent_signal((0, 0), (20, 20))
    assert signal["other_presence"] == 0.0

    # Test 7.2: Action unlocking
    assert not fc.is_action_unlocked("APPROACH", "newborn")
    assert not fc.is_action_unlocked("APPROACH", "child")
    assert fc.is_action_unlocked("APPROACH", "adolescent")
    assert fc.is_action_unlocked("APPROACH", "adult")
    assert fc.is_action_unlocked("OBSERVE", "child")  # children can observe

    # Test 7.2: OBSERVE interaction
    result = fc.execute_interaction(
        action="OBSERVE",
        actor_id="adam", actor_pos=(5, 5),
        actor_state={"life_stage": "adult"},
        target_id="eve", target_pos=(7, 7),
        target_state={"life_stage": "adult", "hunger": 60},
        tick=200,
    )
    print(f"  OBSERVE: success={result['success']}, reward={result['reward']}")
    assert result["success"]
    assert result["reward"] > 0

    # Test 7.2: APPROACH
    result = fc.execute_interaction(
        action="APPROACH",
        actor_id="adam", actor_pos=(5, 5),
        actor_state={"life_stage": "adult"},
        target_id="eve", target_pos=(7, 7),
        target_state={"life_stage": "adult"},
        tick=210,
    )
    print(f"  APPROACH: new_pos={result['new_position']}")
    assert result["success"]
    assert result["new_position"] is not None

    # Test 7.2: SHARE when target hungry
    result = fc.execute_interaction(
        action="SHARE",
        actor_id="adam", actor_pos=(5, 5),
        actor_state={"life_stage": "adult"},
        target_id="eve", target_pos=(5, 6),
        target_state={"life_stage": "adult", "hunger": 70},
        tick=220,
    )
    print(f"  SHARE with hungry target: actor_reward={result['reward']}, target_reward={result['target_reward']}")
    assert result["reward"] == 0.5  # altruism reward
    assert result["target_reward"] == 0.3

    # Test 7.3: Trust tracking
    trust = fc.get_trust("adam", "eve")
    print(f"  Trust after positive interactions: {trust:.3f}")
    assert trust > 0  # should have built up from OBSERVE + SHARE

    # Test negative interaction (FLEE_AGENT)
    fc.execute_interaction(
        action="FLEE_AGENT",
        actor_id="eve", actor_pos=(5, 6),
        actor_state={"life_stage": "adult"},
        target_id="adam", target_pos=(5, 5),
        target_state={"life_stage": "adult"},
        tick=230,
    )
    trust_after_flee = fc.get_trust("adam", "eve")
    print(f"  Trust after FLEE: {trust_after_flee:.3f} (was {trust:.3f})")
    assert trust_after_flee < trust  # trust dropped

    # Test trust labels
    assert fc.get_trust_label(0.7) == "TRUSTED"
    assert fc.get_trust_label(0.3) == "FAMILIAR"
    assert fc.get_trust_label(0.0) == "STRANGER"
    assert fc.get_trust_label(-0.3) == "DISTRUSTED"
    assert fc.get_trust_label(-0.7) == "HOSTILE"

    # Test 7.4: Vocabulary contact
    result = fc.check_vocabulary_contact(
        adam_pos=(5, 5), eve_pos=(5, 6),  # adjacent
        adam_dialogue="cold. tired.", eve_dialogue="hungry. bad.",
        tick=300,
    )
    print(f"  Vocab contact: in_contact={result['in_contact']}, adam_heard='{result['adam_heard']}'")
    assert result["in_contact"]
    assert result["adam_heard"] == "hungry. bad."

    # Out of range
    result = fc.check_vocabulary_contact(
        (0, 0), (10, 10), "test", "test", tick=310
    )
    assert not result["in_contact"]

    # Check heard dialogues
    heard = fc.get_heard_dialogues("adam")
    print(f"  Adam heard {len(heard)} dialogues")
    assert len(heard) >= 1

    # Test summary
    summary = fc.get_summary()
    print(f"  Summary: {summary}")
    assert summary["first_contact_happened"]
    assert summary["total_interactions"] >= 3

    print("\n✓ FirstContactEngine self-test passed")
