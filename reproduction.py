"""
Nafs AI — Reproduction Engine (Phase 5)
=======================================

Implements MD Phase 5: reproduction creates generations. Generations create
evolution. Evolution creates everything else.

Covers:
  5.1 Fertility Conditions
      - Both agents: health > 70, glucose > 60, hydration > 60, adult stage
      - Eve fertility cycle: 200 fertile / 800 infertile (1000-tick cycle)
      - Adam: always fertile if conditions met
      - Proximity required: within 2 tiles, no danger
      - Reproduction is NOT an explicit action — emergent

  5.2 Pregnancy (Eve)
      - Gestation: 600 ticks
      - Eve's metabolism +30% during pregnancy
      - Movement speed -20% in late pregnancy (last 200 ticks)
      - Miscarriage if Eve's health < 30 during pregnancy
      - Track pregnancy status + gestation progress

  5.3 Baby Agent Spawning
      - Baby spawns within 3 tiles of Eve at birth
      - Inherits biological traits from both parents (averaged + ±10% mutation)
      - Baby brain: 500 params MLP, blank vocabulary, blank memory
      - NO knowledge from parents — only biology
      - Warmth bonus when near Eve (maternal bond)

  5.4 Inherited Biological Traits
      - Metabolism rate, cold resistance, disease immunity, curiosity base rate
      - Vision range, max body mass, growth rate
      - Inheritance: avg(parents) + Gaussian mutation ±10%
      - Disease immunity: 0.5 * max(parents) (partial transfer)

  5.5 Generational Tracking
      - agent_id, parents (list), generation_number, birth_tick, death_tick
      - Adam and Eve = Generation 1
      - Children = Generation 2, grandchildren = Gen 3, etc.
      - generations.jsonl: permanent record of every agent ever born
      - Family tree + evolution tracker (avg traits per generation)

Design constraints:
  - Does NOT modify base rewards
  - Standalone module
  - Hooks into biology engine for trait inheritance

Usage:
    from reproduction import ReproductionEngine
    repro = ReproductionEngine(physics_engine)
    if repro.check_fertility(adam_bio, eve_bio, adam_pos, eve_pos, tick):
        repro.start_pregnancy(eve_id, adam_id, tick)
    pregnancy_update = repro.update_pregnancy(eve_id, eve_bio, tick)
    if pregnancy_update['birth']:
        baby = repro.spawn_baby(eve_pos, eve_id, adam_id, eve_bio, adam_bio, tick)
"""

import json
import os
import random
import math
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Fertility thresholds
FERTILE_HEALTH = 70.0
FERTILE_GLUCOSE = 60.0
FERTILE_HYDRATION = 60.0
FERTILE_PROXIMITY = 2  # tiles
EVE_FERTILE_DURATION = 200   # ticks per cycle
EVE_INFERTILE_DURATION = 800 # ticks per cycle
EVE_CYCLE_LENGTH = EVE_FERTILE_DURATION + EVE_INFERTILE_DURATION  # 1000

# Pregnancy
GESTATION_DURATION = 600       # ticks
PREGNANCY_METABOLISM_MULT = 1.3  # +30% metabolism
LATE_PREGNANCY_START = 400    # last 200 ticks
LATE_PREGNANCY_SPEED_MULT = 0.8  # -20% movement speed
MISCARRIAGE_HEALTH_THRESHOLD = 30.0

# Baby spawning
BABY_SPAWN_RADIUS = 3  # tiles from Eve
BABY_WARMTH_BONUS = 0.2  # positive reward when baby is near Eve
BABY_WARMTH_PROXIMITY = 3  # tiles

# Trait inheritance
MUTATION_STDDEV = 0.10  # 10% Gaussian noise on inherited traits

# Generational tracking
GENERATIONS_LOG_PATH = "generations.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# ReproductionEngine
# ═══════════════════════════════════════════════════════════════════════════════

class ReproductionEngine:
    """
    Master reproduction engine for the Nafs AI world.

    Holds:
      - fertility tracking (Eve's cycle)
      - active pregnancies (per-Eve)
      - agent registry (id, parents, generation, birth_tick, death_tick)
      - generations.jsonl log

    One instance per WorldSim. Adam is always male, Eve is always female
    (for v1.0; future phases may add more agents).
    """

    def __init__(self, physics_engine=None, seed: Optional[int] = None,
                 log_path: str = GENERATIONS_LOG_PATH):
        self.physics = physics_engine
        self.rng = random.Random(seed or random.randint(0, 999999))
        self.log_path = log_path

        # Eve's fertility cycle: tick when current cycle started
        self.eve_cycle_start = 0

        # Active pregnancies: {eve_id: Pregnancy}
        self.pregnancies: Dict[str, 'Pregnancy'] = {}

        # Agent registry: {agent_id: AgentRecord}
        self.agents: Dict[str, 'AgentRecord'] = {}

        # Next agent ID counter
        self._next_id = 1

        # Initialize log file (truncate at start)
        try:
            with open(self.log_path, "w") as f:
                f.write("")
        except Exception:
            pass

        # Statistics
        self.total_births = 0
        self.total_miscarriages = 0
        self.total_deaths = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Agent registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_agent(self, agent_id: str, parents: List[str],
                        generation: int, birth_tick: int,
                        traits: Optional[Dict] = None) -> 'AgentRecord':
        """Register a new agent in the lineage database."""
        record = AgentRecord(
            agent_id=agent_id,
            parents=list(parents),
            generation=generation,
            birth_tick=birth_tick,
            death_tick=None,
            traits=traits or {},
        )
        self.agents[agent_id] = record

        # Append to log
        self._log_event({
            "event": "BIRTH",
            "agent_id": agent_id,
            "parents": list(parents),
            "generation": generation,
            "birth_tick": birth_tick,
            "traits": traits or {},
        })

        return record

    def record_death(self, agent_id: str, death_tick: int,
                      cause: str = "") -> None:
        """Record an agent's death."""
        if agent_id in self.agents:
            self.agents[agent_id].death_tick = death_tick
            self.agents[agent_id].death_cause = cause
            self.total_deaths += 1
            self._log_event({
                "event": "DEATH",
                "agent_id": agent_id,
                "death_tick": death_tick,
                "cause": cause,
                "generation": self.agents[agent_id].generation,
            })

    def _log_event(self, event: Dict) -> None:
        """Append event to generations.jsonl log."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[Reproduction] Failed to write log: {e}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.1 Fertility checking
    # ─────────────────────────────────────────────────────────────────────────

    def check_fertility(self, adam_bio, eve_bio,
                         adam_pos: Tuple[int, int],
                         eve_pos: Tuple[int, int],
                         tick: int,
                         danger_present: bool = False) -> Dict:
        """
        Check if reproduction conditions are met.

        Returns dict with:
          - can_reproduce: bool
          - reason: str (why not, if can_reproduce is False)
          - eve_fertile: bool
        """
        result = {
            "can_reproduce": False,
            "reason": "",
            "eve_fertile": self.is_eve_fertile(tick),
        }

        # Check life stage (both must be adults)
        if not (hasattr(adam_bio, 'can_reproduce') and adam_bio.can_reproduce()):
            result["reason"] = "adam_not_adult"
            return result
        if not (hasattr(eve_bio, 'can_reproduce') and eve_bio.can_reproduce()):
            result["reason"] = "eve_not_adult"
            return result

        # Check stats thresholds
        if adam_bio.glucose < FERTILE_GLUCOSE:
            result["reason"] = "adam_low_glucose"
            return result
        if eve_bio.glucose < FERTILE_GLUCOSE:
            result["reason"] = "eve_low_glucose"
            return result
        if adam_bio.hydration < FERTILE_HYDRATION:
            result["reason"] = "adam_low_hydration"
            return result
        if eve_bio.hydration < FERTILE_HYDRATION:
            result["reason"] = "eve_low_hydration"
            return result

        # Health threshold (using metabolic health proxy)
        adam_health_proxy = (adam_bio.glucose + adam_bio.hydration) / 2.0
        eve_health_proxy = (eve_bio.glucose + eve_bio.hydration) / 2.0
        if adam_health_proxy < FERTILE_HEALTH:
            result["reason"] = "adam_low_health"
            return result
        if eve_health_proxy < FERTILE_HEALTH:
            result["reason"] = "eve_low_health"
            return result

        # Eve's fertility cycle
        if not self.is_eve_fertile(tick):
            result["reason"] = "eve_infertile_cycle"
            return result

        # Already pregnant?
        for eve_id, preg in self.pregnancies.items():
            if not preg.completed:
                result["reason"] = "eve_already_pregnant"
                return result

        # Proximity check
        dist = abs(adam_pos[0] - eve_pos[0]) + abs(adam_pos[1] - eve_pos[1])
        if dist > FERTILE_PROXIMITY:
            result["reason"] = "too_far_apart"
            return result

        # Danger check
        if danger_present:
            result["reason"] = "danger_present"
            return result

        result["can_reproduce"] = True
        return result

    def is_eve_fertile(self, tick: int) -> bool:
        """Check if Eve is in her fertile window."""
        cycle_pos = (tick - self.eve_cycle_start) % EVE_CYCLE_LENGTH
        return cycle_pos < EVE_FERTILE_DURATION

    def get_eve_cycle_info(self, tick: int) -> Dict:
        """Return info about Eve's current cycle state."""
        cycle_pos = (tick - self.eve_cycle_start) % EVE_CYCLE_LENGTH
        if cycle_pos < EVE_FERTILE_DURATION:
            return {
                "state": "fertile",
                "ticks_remaining": EVE_FERTILE_DURATION - cycle_pos,
            }
        else:
            return {
                "state": "infertile",
                "ticks_remaining": EVE_CYCLE_LENGTH - cycle_pos,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 5.2 Pregnancy
    # ─────────────────────────────────────────────────────────────────────────

    def start_pregnancy(self, eve_id: str, adam_id: str, tick: int) -> 'Pregnancy':
        """Start a new pregnancy for Eve."""
        pregnancy = Pregnancy(
            eve_id=eve_id,
            adam_id=adam_id,
            start_tick=tick,
            end_tick=tick + GESTATION_DURATION,
        )
        self.pregnancies[eve_id] = pregnancy
        self._log_event({
            "event": "PREGNANCY_START",
            "eve_id": eve_id,
            "adam_id": adam_id,
            "tick": tick,
            "expected_birth_tick": pregnancy.end_tick,
        })
        return pregnancy

    def update_pregnancy(self, eve_id: str, eve_bio, tick: int) -> Dict:
        """
        Update pregnancy state for an Eve.

        Returns dict with:
          - active: bool (is Eve currently pregnant?)
          - progress: float (0-1)
          - in_late_phase: bool (last 200 ticks)
          - metabolism_mult: float (1.3 if pregnant, 1.0 otherwise)
          - speed_mult: float (0.8 if late pregnancy, 1.0 otherwise)
          - miscarriage: bool (true if miscarried this tick)
          - birth: bool (true if baby born this tick)
        """
        pregnancy = self.pregnancies.get(eve_id)
        if pregnancy is None or pregnancy.completed:
            return {
                "active": False, "progress": 0.0, "in_late_phase": False,
                "metabolism_mult": 1.0, "speed_mult": 1.0,
                "miscarriage": False, "birth": False,
            }

        # Check miscarriage (health < 30)
        eve_health_proxy = (eve_bio.glucose + eve_bio.hydration) / 2.0
        if eve_health_proxy < MISCARRIAGE_HEALTH_THRESHOLD:
            pregnancy.completed = True
            pregnancy.miscarried = True
            self.total_miscarriages += 1
            self._log_event({
                "event": "MISCARRIAGE",
                "eve_id": eve_id,
                "tick": tick,
            })
            return {
                "active": False, "progress": 1.0, "in_late_phase": False,
                "metabolism_mult": 1.0, "speed_mult": 1.0,
                "miscarriage": True, "birth": False,
            }

        # Update progress
        elapsed = tick - pregnancy.start_tick
        pregnancy.progress = min(1.0, elapsed / GESTATION_DURATION)

        # Check if birth time
        if tick >= pregnancy.end_tick:
            pregnancy.completed = True
            return {
                "active": False, "progress": 1.0,
                "in_late_phase": elapsed >= LATE_PREGNANCY_START,
                "metabolism_mult": 1.0, "speed_mult": 1.0,
                "miscarriage": False, "birth": True,
            }

        # Active pregnancy
        in_late = elapsed >= LATE_PREGNANCY_START
        return {
            "active": True,
            "progress": pregnancy.progress,
            "in_late_phase": in_late,
            "metabolism_mult": PREGNANCY_METABOLISM_MULT,
            "speed_mult": LATE_PREGNANCY_SPEED_MULT if in_late else 1.0,
            "miscarriage": False,
            "birth": False,
        }

    def get_pregnancy_status(self, eve_id: str) -> Optional[Dict]:
        """Get current pregnancy status for dashboard display."""
        preg = self.pregnancies.get(eve_id)
        if preg is None or preg.completed:
            return None
        return {
            "active": True,
            "progress": preg.progress,
            "start_tick": preg.start_tick,
            "end_tick": preg.end_tick,
            "in_late_phase": (preg.progress * GESTATION_DURATION) >= LATE_PREGNANCY_START,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3 + 5.4 Baby spawning + trait inheritance
    # ─────────────────────────────────────────────────────────────────────────

    def spawn_baby(self, eve_pos: Tuple[int, int],
                    eve_id: str, adam_id: str,
                    eve_bio, adam_bio,
                    tick: int,
                    world_width: int = 64, world_height: int = 64) -> Dict:
        """
        Spawn a baby agent near Eve.

        Returns dict with:
          - baby_id: str
          - baby_position: (x, y)
          - inherited_traits: dict
          - parents: [eve_id, adam_id]
          - generation: int (max(parents' generations) + 1)
        """
        # Spawn position: within 3 tiles of Eve
        for _ in range(10):  # try 10 times
            dx = self.rng.randint(-BABY_SPAWN_RADIUS, BABY_SPAWN_RADIUS)
            dy = self.rng.randint(-BABY_SPAWN_RADIUS, BABY_SPAWN_RADIUS)
            baby_x = (eve_pos[0] + dx) % world_width
            baby_y = (eve_pos[1] + dy) % world_height
            if dx != 0 or dy != 0:  # don't spawn on top of Eve
                break

        # Determine generation
        eve_gen = self.agents.get(eve_id, AgentRecord("", [], 1, 0)).generation
        adam_gen = self.agents.get(adam_id, AgentRecord("", [], 1, 0)).generation
        baby_generation = max(eve_gen, adam_gen) + 1

        # Generate baby ID
        baby_id = f"agent_{self._next_id}"
        self._next_id += 1

        # Inherit traits
        inherited_traits = self._inherit_traits(eve_bio, adam_bio)

        # Register baby
        self.register_agent(
            agent_id=baby_id,
            parents=[eve_id, adam_id],
            generation=baby_generation,
            birth_tick=tick,
            traits=inherited_traits,
        )

        self.total_births += 1
        self._log_event({
            "event": "BABY_BORN",
            "baby_id": baby_id,
            "eve_id": eve_id,
            "adam_id": adam_id,
            "tick": tick,
            "generation": baby_generation,
            "position": [baby_x, baby_y],
        })

        return {
            "baby_id": baby_id,
            "baby_position": (baby_x, baby_y),
            "inherited_traits": inherited_traits,
            "parents": [eve_id, adam_id],
            "generation": baby_generation,
        }

    def _inherit_traits(self, eve_bio, adam_bio) -> Dict:
        """
        Inherit biological traits from both parents.

        Most traits: avg(parents) + Gaussian mutation ±10%
        Disease immunity: 0.5 * max(parents) (partial transfer)
        """
        # Helper: average + Gaussian mutation
        def inherit_value(val1, val2):
            avg = (val1 + val2) / 2.0
            mutated = avg + self.rng.gauss(0, MUTATION_STDDEV * max(0.1, abs(avg)))
            return max(0.0, mutated)

        # Get trait values from parents (use defaults if missing)
        def get_trait(bio, name, default):
            return getattr(bio, name, default) if bio else default

        traits = {
            # Metabolic
            "metabolism_rate": inherit_value(
                get_trait(eve_bio, 'metabolism_rate', 1.0),
                get_trait(adam_bio, 'metabolism_rate', 1.0),
            ),
            # Cold resistance (proxy: body_mass)
            "cold_resistance": inherit_value(
                get_trait(eve_bio, 'body_mass', 70.0),
                get_trait(adam_bio, 'body_mass', 70.0),
            ) / 70.0,  # normalize to ~1.0
            # Curiosity base rate
            "curiosity_base_rate": inherit_value(
                get_trait(eve_bio, 'curiosity_base_rate', 0.5),
                get_trait(adam_bio, 'curiosity_base_rate', 0.5),
            ),
            # Vision range
            "vision_range": inherit_value(
                get_trait(eve_bio, 'vision_range', 3.0),
                get_trait(adam_bio, 'vision_range', 3.0),
            ),
            # Max body mass
            "max_body_mass": inherit_value(
                get_trait(eve_bio, 'body_mass', 70.0) * 1.2,
                get_trait(adam_bio, 'body_mass', 70.0) * 1.2,
            ),
            # Growth rate (brain)
            "growth_rate": inherit_value(
                get_trait(eve_bio, 'growth_rate', 1.0),
                get_trait(adam_bio, 'growth_rate', 1.0),
            ),
        }

        # Disease immunity: 0.5 * max(parents) (partial transfer)
        eve_immunity = (eve_bio.get_average_immunity()
                        if hasattr(eve_bio, 'get_average_immunity') else 0.0)
        adam_immunity = (adam_bio.get_average_immunity()
                         if hasattr(adam_bio, 'get_average_immunity') else 0.0)
        traits["disease_immunity"] = 0.5 * max(eve_immunity, adam_immunity)

        return traits

    # ─────────────────────────────────────────────────────────────────────────
    # 5.5 Generational tracking + family tree
    # ─────────────────────────────────────────────────────────────────────────

    def get_agent(self, agent_id: str) -> Optional['AgentRecord']:
        return self.agents.get(agent_id)

    def get_generation(self, agent_id: str) -> int:
        agent = self.agents.get(agent_id)
        return agent.generation if agent else 1

    def get_children(self, agent_id: str) -> List[str]:
        """Return list of agent IDs who have this agent as a parent."""
        return [aid for aid, rec in self.agents.items()
                if agent_id in rec.parents]

    def get_family_tree(self, root_id: str, max_depth: int = 3) -> Dict:
        """Build a family tree starting from root_id."""
        def build_node(agent_id, depth):
            agent = self.agents.get(agent_id)
            if not agent or depth > max_depth:
                return None
            return {
                "agent_id": agent_id,
                "generation": agent.generation,
                "birth_tick": agent.birth_tick,
                "death_tick": agent.death_tick,
                "parents": [build_node(p, depth + 1) for p in agent.parents
                            if p in self.agents],
                "children": [build_node(c, depth + 1)
                             for c in self.get_children(agent_id)],
            }
        return build_node(root_id, 0)

    def get_generation_stats(self) -> Dict:
        """Get average traits per generation."""
        by_gen: Dict[int, List] = {}
        for agent in self.agents.values():
            by_gen.setdefault(agent.generation, []).append(agent)

        stats = {}
        for gen, agents in by_gen.items():
            trait_sums = {}
            for agent in agents:
                for trait, val in agent.traits.items():
                    if isinstance(val, (int, float)):
                        trait_sums[trait] = trait_sums.get(trait, 0) + val
            stats[gen] = {
                "count": len(agents),
                "avg_traits": {k: v / len(agents) for k, v in trait_sums.items()},
            }
        return stats

    def get_summary(self) -> Dict:
        """Return summary stats for dashboard."""
        living = sum(1 for a in self.agents.values() if a.death_tick is None)
        return {
            "total_agents": len(self.agents),
            "living": living,
            "deceased": self.total_deaths,
            "total_births": self.total_births,
            "total_miscarriages": self.total_miscarriages,
            "active_pregnancies": sum(1 for p in self.pregnancies.values() if not p.completed),
            "max_generation": max((a.generation for a in self.agents.values()), default=1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helper classes
# ═══════════════════════════════════════════════════════════════════════════════

class AgentRecord:
    """Record of an agent's life in the lineage database."""

    def __init__(self, agent_id: str, parents: List[str],
                 generation: int, birth_tick: int,
                 death_tick: Optional[int] = None,
                 traits: Optional[Dict] = None,
                 death_cause: str = ""):
        self.agent_id = agent_id
        self.parents = parents
        self.generation = generation
        self.birth_tick = birth_tick
        self.death_tick = death_tick
        self.traits = traits or {}
        self.death_cause = death_cause


class Pregnancy:
    """Tracks a single pregnancy."""

    def __init__(self, eve_id: str, adam_id: str,
                 start_tick: int, end_tick: int):
        self.eve_id = eve_id
        self.adam_id = adam_id
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.progress = 0.0
        self.completed = False
        self.miscarried = False


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing ReproductionEngine...")

    # Mock biology
    class MockBio:
        def __init__(self, glucose=80, hydration=80, body_mass=70):
            self.glucose = glucose
            self.hydration = hydration
            self.body_mass = body_mass
            self.metabolism_rate = 1.0
            self.curiosity_base_rate = 0.5
            self.vision_range = 3.0
            self.growth_rate = 1.0
            self._age_ticks = 3000  # adult
        def can_reproduce(self):
            return self._age_ticks >= 2000 and self._age_ticks < 6000
        def get_average_immunity(self):
            return 0.5

    repro = ReproductionEngine(seed=42, log_path="/tmp/test_generations.jsonl")

    # Register Adam and Eve (Gen 1)
    repro.register_agent("adam", [], 1, 0)
    repro.register_agent("eve", [], 1, 0)
    print(f"  Registered Adam + Eve (Gen 1)")

    # Test fertility check
    adam_bio = MockBio()
    eve_bio = MockBio()
    result = repro.check_fertility(adam_bio, eve_bio, (5, 5), (5, 5), tick=50)
    print(f"  Fertility check (close, fertile window): {result['can_reproduce']}, reason={result['reason']}")
    assert result["can_reproduce"] is True

    # Test too far apart
    result = repro.check_fertility(adam_bio, eve_bio, (5, 5), (10, 10), tick=50)
    assert not result["can_reproduce"]
    assert result["reason"] == "too_far_apart"

    # Test Eve infertile window (after 200 ticks)
    result = repro.check_fertility(adam_bio, eve_bio, (5, 5), (5, 5), tick=500)
    print(f"  Fertility at tick 500 (infertile window): {result['can_reproduce']}, reason={result['reason']}")
    assert not result["can_reproduce"]
    assert result["reason"] == "eve_infertile_cycle"

    # Test low glucose
    adam_low_glucose = MockBio(glucose=30)
    result = repro.check_fertility(adam_low_glucose, eve_bio, (5, 5), (5, 5), tick=50)
    assert not result["can_reproduce"]
    assert result["reason"] == "adam_low_glucose"

    # Start pregnancy
    preg = repro.start_pregnancy("eve", "adam", tick=50)
    print(f"  Started pregnancy: end_tick={preg.end_tick}, duration={preg.end_tick - preg.start_tick}")
    assert preg.end_tick == 50 + 600

    # Update pregnancy — early phase
    update = repro.update_pregnancy("eve", eve_bio, tick=100)
    print(f"  Early pregnancy: active={update['active']}, progress={update['progress']:.2f}, "
          f"metabolism_mult={update['metabolism_mult']}")
    assert update["active"] is True
    assert update["metabolism_mult"] == 1.3
    assert update["speed_mult"] == 1.0  # not late yet

    # Late pregnancy (tick 500, 450 elapsed > 400)
    update = repro.update_pregnancy("eve", eve_bio, tick=500)
    print(f"  Late pregnancy: in_late={update['in_late_phase']}, speed_mult={update['speed_mult']}")
    assert update["in_late_phase"] is True
    assert update["speed_mult"] == 0.8

    # Birth (tick 650, end_tick=650)
    update = repro.update_pregnancy("eve", eve_bio, tick=650)
    print(f"  Birth: birth={update['birth']}, progress={update['progress']:.2f}")
    assert update["birth"] is True

    # Spawn baby
    baby = repro.spawn_baby((5, 5), "eve", "adam", eve_bio, adam_bio, tick=650)
    print(f"  Baby born: id={baby['baby_id']}, gen={baby['generation']}, "
          f"pos={baby['baby_position']}")
    print(f"  Inherited traits: {baby['inherited_traits']}")
    assert baby["generation"] == 2
    assert baby["baby_id"] != "adam" and baby["baby_id"] != "eve"

    # Test miscarriage
    repro2 = ReproductionEngine(seed=42, log_path="/tmp/test_generations2.jsonl")
    repro2.register_agent("adam", [], 1, 0)
    repro2.register_agent("eve", [], 1, 0)
    repro2.start_pregnancy("eve", "adam", tick=50)
    eve_low_health = MockBio(glucose=15, hydration=15)  # health proxy = 15 < 30
    update = repro2.update_pregnancy("eve", eve_low_health, tick=100)
    print(f"  Miscarriage: {update['miscarriage']}")
    assert update["miscarriage"] is True

    # Test generational tracking
    summary = repro.get_summary()
    print(f"  Summary: {summary}")
    assert summary["total_births"] == 1
    assert summary["max_generation"] == 2

    # Test family tree
    tree = repro.get_family_tree("adam", max_depth=2)
    print(f"  Family tree from Adam: {tree['agent_id']}, children={len(tree['children'])}")
    # Adam should have the baby as a child
    assert len(tree["children"]) == 1

    # Test generation stats
    stats = repro.get_generation_stats()
    print(f"  Generation stats: {list(stats.keys())}")
    assert 1 in stats and 2 in stats

    # Test death recording
    repro.record_death("adam", death_tick=1000, cause="starvation")
    agent = repro.get_agent("adam")
    assert agent.death_tick == 1000
    assert agent.death_cause == "starvation"

    print("\n✓ ReproductionEngine self-test passed")
