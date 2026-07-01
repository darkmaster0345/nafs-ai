"""
Nafs AI — Godot Bridge (Phase 11)
=================================

Python bridge that prepares simulation state for Godot consumption.

The Godot client polls an HTTP endpoint (default: http://localhost:5000/state)
to get the current world state. This module provides:

  - GodotBridge: Collects state from all engines (world_sim, physics,
    chemistry, biology, reproduction, etc.) into a single dict
  - Serializes to JSON for HTTP response
  - Tracks milestones (first contact, first birth, first death, etc.)
    and includes them in the state for Godot to display as banners

Usage:
    from godot_bridge import GodotBridge
    bridge = GodotBridge(world_sim, physics, chemistry, biology, reproduction,
                          first_contact, social, evolution)
    state = bridge.build_state(tick, agent_positions)
    # Serve this via Flask/HTTP server for Godot to fetch
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
# GodotBridge
# ═══════════════════════════════════════════════════════════════════════════════

class GodotBridge:
    """
    Collects state from all engines into a dict that Godot can consume.

    The Godot client polls /state and receives:
      - biome_map: {"x,y": "forest", ...} — 64x64 grid
      - agents: [{id, x, y, type, action, thought, dialogue, life_stage, ...}]
      - fire_tiles: [[x, y], ...]
      - water_tiles: [[x, y], ...]
      - time_of_day: int
      - weather: str
      - milestones: [{type, details}] — events Godot should banner
    """

    # Milestone types that Godot knows how to display
    MILESTONE_TYPES = {
        "FIRST_CONTACT", "FIRST_BIRTH", "FIRST_DEATH", "FIRST_WORD",
        "NEW_GENERATION", "EXTINCTION", "COOKING_DISCOVERY",
    }

    def __init__(self, world_sim=None, physics=None, chemistry=None,
                 biology=None, reproduction=None, first_contact=None,
                 social=None, evolution=None):
        self.world_sim = world_sim
        self.physics = physics
        self.chemistry = chemistry
        self.biology = biology
        self.reproduction = reproduction
        self.first_contact = first_contact
        self.social = social
        self.evolution = evolution

        # Milestone tracking — each milestone fires only once
        self.milestones_fired: set = set()
        self.pending_milestones: deque = deque(maxlen=50)  # queued for Godot

        # History buffer for scrubbing (last 1000 ticks)
        self.history: deque = deque(maxlen=1000)

    # ─────────────────────────────────────────────────────────────────────────
    # Build state for Godot
    # ─────────────────────────────────────────────────────────────────────────

    def build_state(self, tick: int,
                     agent_positions: Optional[Dict[str, Tuple[int, int]]] = None,
                     agent_data: Optional[List[Dict]] = None) -> Dict:
        """
        Build the complete state dict for Godot consumption.

        Args:
            tick: current simulation tick
            agent_positions: {agent_id: (x, y)} — where each agent is
            agent_data: list of per-agent dicts with stats, vocab, etc.

        Returns:
            Dict ready to be JSON-serialized and sent to Godot.
        """
        state = {
            "tick": tick,
            "biome_map": self._build_biome_map(),
            "agents": self._build_agents_state(agent_positions, agent_data, tick),
            "fire_tiles": self._build_fire_tiles(),
            "water_tiles": self._build_water_tiles(),
            "time_of_day": self._get_time_of_day(),
            "weather": self._get_weather(),
            "milestones": self._drain_pending_milestones(),
            "population": self._get_population(),
            "summary": self._build_summary(tick),
        }
        # Store in history
        self.history.append(state)
        return state

    def _build_biome_map(self) -> Dict[str, str]:
        """Build {x,y: biome_name} for all 64x64 tiles."""
        if not self.world_sim:
            return {}
        world_map = self.world_sim.world_map
        result = {}
        for x in range(world_map.width):
            for y in range(world_map.height):
                biome = world_map.get_biome(x, y)
                result[f"{x},{y}"] = biome
        return result

    def _build_agents_state(self, agent_positions, agent_data, tick) -> List[Dict]:
        """Build list of agent state dicts."""
        if not agent_positions and not agent_data:
            return []

        agents = []
        # If agent_data provided, use it; otherwise build from positions
        if agent_data:
            for data in agent_data:
                agent_id = data.get("id", "")
                pos = agent_positions.get(agent_id, (0, 0)) if agent_positions else (0, 0)
                agent = {
                    "id": agent_id,
                    "x": pos[0],
                    "y": pos[1],
                    "type": data.get("type", "adam"),
                    "action": data.get("action", "IDLE"),
                    "thought": data.get("thought", ""),
                    "dialogue": data.get("dialogue", ""),
                    "life_stage": data.get("life_stage", "adult"),
                    "age_ticks": data.get("age_ticks", 0),
                    "health": data.get("health", 100),
                    "glucose": data.get("glucose", 80),
                    "hydration": data.get("hydration", 80),
                    "body_temp": data.get("body_temp", 37),
                    "vocabulary_size": data.get("vocabulary_size", 0),
                    "vocabulary": data.get("vocabulary", []),
                    "generation": data.get("generation", 1),
                    "parents": data.get("parents", []),
                    "injury_name": data.get("injury_name", "NONE"),
                    "sleep_debt": data.get("sleep_debt", 0),
                }
                agents.append(agent)
        elif agent_positions:
            # Just positions, no detailed data
            for agent_id, pos in agent_positions.items():
                agents.append({
                    "id": agent_id,
                    "x": pos[0],
                    "y": pos[1],
                    "type": "adam" if agent_id == "adam" else "eve" if agent_id == "eve" else "baby",
                })
        return agents

    def _build_fire_tiles(self) -> List[List[int]]:
        """Get list of [x, y] for all fire tiles."""
        if not self.physics:
            return []
        return [[x, y] for (x, y) in self.physics.fire_tiles.keys()]

    def _build_water_tiles(self) -> List[List[int]]:
        """Get list of [x, y] for all temporary water tiles."""
        if not self.physics:
            return []
        return [[x, y] for (x, y) in self.physics.water_tiles]

    def _get_time_of_day(self) -> int:
        if self.world_sim:
            return self.world_sim.time_of_day
        return 12

    def _get_weather(self) -> str:
        if self.world_sim:
            return self.world_sim.weather_system.current
        return "clear"

    def _get_population(self) -> int:
        if self.reproduction:
            return sum(1 for a in self.reproduction.agents.values() if a.death_tick is None)
        return 0

    def _drain_pending_milestones(self) -> List[Dict]:
        """Drain the pending milestones queue (Godot displays each once)."""
        milestones = []
        while self.pending_milestones:
            milestones.append(self.pending_milestones.popleft())
        return milestones

    def _build_summary(self, tick: int) -> Dict:
        """High-level summary for dashboard."""
        summary = {
            "tick": tick,
            "time_of_day": self._get_time_of_day(),
            "weather": self._get_weather(),
        }
        if self.reproduction:
            summary["reproduction"] = self.reproduction.get_summary()
        if self.first_contact:
            summary["first_contact"] = self.first_contact.get_summary()
        if self.social:
            summary["social"] = self.social.get_summary()
        if self.evolution:
            summary["evolution"] = self.evolution.get_summary()
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # Milestone tracking
    # ─────────────────────────────────────────────────────────────────────────

    def fire_milestone(self, milestone_type: str, details: Optional[Dict] = None,
                        unique: bool = True) -> bool:
        """
        Fire a milestone event for Godot to display.

        Args:
            milestone_type: one of MILESTONE_TYPES
            details: optional dict with extra info (agent_id, position, etc.)
            unique: if True, only fire once per milestone_type

        Returns:
            True if milestone was fired (not deduplicated), False otherwise.
        """
        if milestone_type not in self.MILESTONE_TYPES:
            return False

        if unique and milestone_type in self.milestones_fired:
            return False

        if unique:
            self.milestones_fired.add(milestone_type)

        event = {
            "type": milestone_type,
            "details": details or {},
        }
        self.pending_milestones.append(event)
        return True

    def check_milestones(self, tick: int, agent_data: List[Dict],
                          agent_positions: Dict[str, Tuple[int, int]]) -> None:
        """
        Check for milestone conditions and fire events.

        Called each tick by the trainer to detect:
          - First Contact (Phase 7)
          - First Birth (Phase 5)
          - First Death (any agent dies)
          - First Word (Phase 0.3 vocab logger)
          - New Generation (Phase 5)
          - Extinction (Phase 10)
          - Cooking Discovery (Phase 9)
        """
        # First Contact
        if self.first_contact and "FIRST_CONTACT" not in self.milestones_fired:
            fc_event = self.first_contact.get_first_contact_event()
            if fc_event:
                self.fire_milestone("FIRST_CONTACT", {
                    "tick": fc_event["tick"],
                    "adam_position": fc_event["adam_position"],
                    "eve_position": fc_event["eve_position"],
                    "adam_vocab_size": fc_event["adam_vocabulary_size"],
                    "eve_vocab_size": fc_event["eve_vocabulary_size"],
                })

        # First Birth
        if self.reproduction and "FIRST_BIRTH" not in self.milestones_fired:
            if self.reproduction.total_births > 0:
                self.fire_milestone("FIRST_BIRTH", {
                    "tick": tick,
                    "total_births": self.reproduction.total_births,
                })

        # First Death
        if self.reproduction and "FIRST_DEATH" not in self.milestones_fired:
            if self.reproduction.total_deaths > 0:
                # Find the most recent death
                latest_death = None
                for agent in self.reproduction.agents.values():
                    if agent.death_tick is not None:
                        if latest_death is None or agent.death_tick > latest_death.death_tick:
                            latest_death = agent
                if latest_death:
                    self.fire_milestone("FIRST_DEATH", {
                        "agent_id": latest_death.agent_id,
                        "tick": latest_death.death_tick,
                        "cause": latest_death.death_cause,
                    })

        # New Generation
        if self.reproduction and "NEW_GENERATION" not in self.milestones_fired:
            summary = self.reproduction.get_summary()
            if summary["max_generation"] >= 2:
                self.fire_milestone("NEW_GENERATION", {
                    "generation": summary["max_generation"],
                    "tick": tick,
                })

        # Extinction Event
        if self.evolution and "EXTINCTION" not in self.milestones_fired:
            if self.evolution.cataclysms:
                latest = self.evolution.cataclysms[-1]
                self.fire_milestone("EXTINCTION", {
                    "tick": latest["tick"],
                    "cause": latest["cause"],
                    "agents_killed": latest["agents_killed"],
                })

    # ─────────────────────────────────────────────────────────────────────────
    # History (for scrubbing)
    # ─────────────────────────────────────────────────────────────────────────

    def get_history_at(self, index: int) -> Optional[Dict]:
        """Get historical state at index (for scrubbing)."""
        if index < 0 or index >= len(self.history):
            return None
        return self.history[index]

    def get_history_length(self) -> int:
        """Get number of stored historical states."""
        return len(self.history)

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def state_to_json(self, state: Dict) -> str:
        """Serialize state to JSON for HTTP response."""
        return json.dumps(state, default=str)

    def get_summary(self) -> Dict:
        return {
            "milestones_fired": list(self.milestones_fired),
            "pending_milestones": len(self.pending_milestones),
            "history_length": len(self.history),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing GodotBridge...")

    bridge = GodotBridge()

    # Test milestone firing
    fired = bridge.fire_milestone("FIRST_CONTACT", {"tick": 100, "adam_position": [5, 5]})
    print(f"  First milestone fire: {fired}")
    assert fired

    # Second fire should be deduplicated
    fired2 = bridge.fire_milestone("FIRST_CONTACT", {"tick": 110})
    print(f"  Duplicate fire: {fired2}")
    assert not fired2

    # Invalid milestone type
    fired3 = bridge.fire_milestone("INVALID_TYPE")
    assert not fired3

    # Test build_state with no engines
    state = bridge.build_state(tick=100, agent_positions={"adam": (5, 5), "eve": (7, 7)})
    print(f"  State keys: {list(state.keys())}")
    assert state["tick"] == 100
    assert len(state["agents"]) == 2
    assert state["agents"][0]["id"] == "adam"
    assert state["milestones"][0]["type"] == "FIRST_CONTACT"

    # Test with agent_data
    agent_data = [
        {"id": "adam", "type": "adam", "action": "MOVE", "life_stage": "adult",
         "age_ticks": 100, "health": 90, "vocabulary_size": 25},
        {"id": "eve", "type": "eve", "action": "EAT", "life_stage": "adult",
         "age_ticks": 100, "health": 85, "vocabulary_size": 20},
    ]
    state = bridge.build_state(tick=200, agent_positions={"adam": (5, 5), "eve": (7, 7)},
                                 agent_data=agent_data)
    print(f"  Agent state: {state['agents'][0]}")
    assert state["agents"][0]["action"] == "MOVE"
    assert state["agents"][0]["health"] == 90

    # Test history
    assert bridge.get_history_length() == 2
    hist = bridge.get_history_at(0)
    assert hist["tick"] == 100

    # Test JSON serialization
    json_str = bridge.state_to_json(state)
    assert isinstance(json_str, str)
    assert "\"tick\": 200" in json_str

    # Test summary
    summary = bridge.get_summary()
    print(f"  Summary: {summary}")
    assert "FIRST_CONTACT" in summary["milestones_fired"]

    print("\n✓ GodotBridge self-test passed")
