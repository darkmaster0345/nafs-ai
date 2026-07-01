"""
Nafs AI — Open-Ended Extension Systems (Phase 13)
=================================================

Implements MD Phase 13: these are not features with a completion state.
They make the simulation truly open-ended — capable of producing novelty
indefinitely.

Covers:
  13.1 Procedural World Evolution
      - Biomes shift slowly over 50,000 ticks (climate change analogue)
      - New biomes emerge from combined conditions (flooded forest = mangrove)
      - Volcano eruptions: random, destroy 5-tile radius, create ash biome
      - Glacier advance: avg temp drops 3° for 5000 ticks → tundra expands

  13.2 Disease Evolution
      - Diseases mutate over time (more virulent after 10,000 ticks)
      - Agents with high immunity pressure disease to mutate (arms race)
      - New disease variants emerge every 5 generations
      - Ongoing selection pressure

  13.3 Emergent Complexity Detector
      - Automated novelty detector: compare behavior distribution to baseline
      - Flag when new behavior pattern appears
      - Measure complexity over time
      - Open-ended evolution metric

  13.4 World Seeding & Experiments
      - Save/load world state at any tick
      - Fork simulation: two branches from same tick with different seeds
      - Compare outcomes
      - Controlled experiments

Design constraints:
  - Does NOT modify base rewards
  - Standalone module
  - All systems are open-ended (no completion state)

Usage:
    from open_ended import WorldEvolution, DiseaseEvolution,
                            NoveltyDetector, WorldSeeding
    world_evol = WorldEvolution(world_map)
    world_evol.step(tick, avg_temp)
    disease_evol = DiseaseEvolution()
    disease_evol.maybe_mutate(tick, generation, avg_immunity)
    novelty = NoveltyDetector()
    is_novel = novelty.check_behavior(behavior_signature, tick)
    seeding = WorldSeeding()
    state = seeding.snapshot(all_engines)
    seeding.restore(state, all_engines)
"""

import json
import os
import random
import math
import copy
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, deque


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# 13.1 World evolution
BIOME_SHIFT_INTERVAL = 50000      # ticks between biome shifts
VOLCANO_ERUPTION_PROB = 0.0001    # per tick
VOLCANO_ERUPTION_RADIUS = 5       # tiles destroyed
GLACIER_TEMP_DROP = 3.0           # °C drop required for glacier advance
GLACIER_DURATION = 5000           # ticks of cold required

# Composite biomes (emerge from combined conditions)
COMPOSITE_BIOMES = {
    "flooded_forest": {
        "requires": ("forest", "high_water"),
        "becomes": "mangrove",
    },
    "snowed_desert": {
        "requires": ("desert", "cold_temp"),
        "becomes": "frozen_sand",
    },
}

# 13.2 Disease evolution
DISEASE_MUTATION_INTERVAL = 10000  # ticks between mutations
DISEASE_NEW_VARIANT_GENERATIONS = 5  # new variant every 5 generations
DISEASE_VIRULENCE_INCREASE = 0.1   # +10% virulence per mutation
DISEASE_MUTATION_PROB = 0.05       # per tick when high immunity pressure

# 13.3 Novelty detector
NOVELTY_BASELINE_SIZE = 1000       # behaviors in baseline
NOVELTY_WINDOW_SIZE = 100          # recent behaviors to compare
NOVELTY_MIN_OCCURRENCES = 3        # min count to be in baseline

# 13.4 World seeding
MAX_SNAPSHOTS = 10                 # max saved snapshots


# ═══════════════════════════════════════════════════════════════════════════════
# 13.1 WorldEvolution
# ═══════════════════════════════════════════════════════════════════════════════

class WorldEvolution:
    """
    Procedural world evolution: biomes shift, volcanoes erupt, glaciers advance.
    """

    def __init__(self, world_map, seed: Optional[int] = None):
        self.world_map = world_map
        self.width = world_map.width
        self.height = world_map.height
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Climate tracking
        # Long-term baseline temp (the world's "normal" temperature)
        self.baseline_temp = 20.0
        self.avg_temp_history: deque = deque(maxlen=GLACIER_DURATION)
        self.cold_streak_ticks = 0

        # Volcano eruption history
        self.eruptions: List[Dict] = []

        # Biome shifts
        self.biome_shifts: List[Dict] = []
        self.last_shift_tick = 0

        # Composite biome tracking
        self.composite_biome_tiles: Dict[Tuple[int, int], str] = {}

        # Ash tiles (from eruptions)
        self.ash_tiles: Set[Tuple[int, int]] = set()

    def step(self, tick: int, avg_temp: float,
              water_tiles: Optional[Set[Tuple[int, int]]] = None) -> Dict:
        """
        Per-tick world evolution update.

        Args:
            tick: current simulation tick
            avg_temp: current average world temperature
            water_tiles: set of temporarily-flooded tiles

        Returns:
            Dict of events that occurred this tick:
              - biome_shift: bool
              - volcano_eruption: Optional[Dict]
              - glacier_advance: bool
              - composite_biome_emergence: List[Dict]
        """
        events = {
            "biome_shift": False,
            "volcano_eruption": None,
            "glacier_advance": False,
            "composite_biome_emergence": [],
        }

        # Track temperature history
        self.avg_temp_history.append(avg_temp)
        # Compare against fixed baseline (world's "normal" temperature)
        if avg_temp < (self.baseline_temp - GLACIER_TEMP_DROP):
            self.cold_streak_ticks += 1
        else:
            self.cold_streak_ticks = 0

        # Biome shift (every 50,000 ticks)
        if tick - self.last_shift_tick >= BIOME_SHIFT_INTERVAL:
            self.last_shift_tick = tick
            self._shift_biomes(tick)
            events["biome_shift"] = True

        # Volcano eruption (random)
        if self.rng.random() < VOLCANO_ERUPTION_PROB:
            eruption = self._erupt_volcano(tick)
            events["volcano_eruption"] = eruption

        # Glacier advance
        if self.cold_streak_ticks >= GLACIER_DURATION:
            self._advance_glacier(tick)
            events["glacier_advance"] = True
            self.cold_streak_ticks = 0  # reset after advancing

        # Composite biomes
        if water_tiles:
            new_composites = self._check_composite_biomes(water_tiles, tick)
            events["composite_biome_emergence"] = new_composites

        return events

    def _get_baseline_temp(self) -> float:
        """Get the world's baseline temperature."""
        return self.baseline_temp

    def _shift_biomes(self, tick: int) -> None:
        """Shift biomes globally (climate change analogue)."""
        # Pick a random biome to expand and another to contract
        from world_sim import BIOMES
        biome_list = list(BIOMES.keys())
        expanding = self.rng.choice(biome_list)
        contracting = self.rng.choice([b for b in biome_list if b != expanding])

        # Convert some contracting tiles to expanding
        converted = 0
        for x in range(self.width):
            for y in range(self.height):
                if self.world_map.get_biome(x, y) == contracting:
                    if self.rng.random() < 0.05:  # 5% conversion chance
                        self.world_map.tiles[(x, y)] = expanding
                        converted += 1

        self.biome_shifts.append({
            "tick": tick,
            "expanding": expanding,
            "contracting": contracting,
            "tiles_converted": converted,
        })

    def _erupt_volcano(self, tick: int) -> Dict:
        """Trigger a volcano eruption at a random volcano tile."""
        # Find a volcano tile
        volcano_tiles = []
        for x in range(self.width):
            for y in range(self.height):
                if self.world_map.get_biome(x, y) == "volcano":
                    volcano_tiles.append((x, y))

        if not volcano_tiles:
            return {}

        # Pick a random volcano tile
        vx, vy = self.rng.choice(volcano_tiles)

        # Destroy all life in 5-tile radius
        destroyed = []
        for dx in range(-VOLCANO_ERUPTION_RADIUS, VOLCANO_ERUPTION_RADIUS + 1):
            for dy in range(-VOLCANO_ERUPTION_RADIUS, VOLCANO_ERUPTION_RADIUS + 1):
                if dx * dx + dy * dy <= VOLCANO_ERUPTION_RADIUS * VOLCANO_ERUPTION_RADIUS:
                    tx, ty = (vx + dx) % self.width, (vy + dy) % self.height
                    destroyed.append((tx, ty))
                    # Mark as ash (fertile)
                    self.ash_tiles.add((tx, ty))

        eruption = {
            "tick": tick,
            "location": [vx, vy],
            "radius": VOLCANO_ERUPTION_RADIUS,
            "tiles_destroyed": len(destroyed),
        }
        self.eruptions.append(eruption)
        return eruption

    def _advance_glacier(self, tick: int) -> None:
        """Expand tundra biome due to prolonged cold."""
        # Find tundra tiles and expand them
        for x in range(self.width):
            for y in range(self.height):
                if self.world_map.get_biome(x, y) == "tundra":
                    # Try to convert adjacent non-tundra tiles
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = (x + dx) % self.width, (y + dy) % self.height
                        if self.world_map.get_biome(nx, ny) != "tundra":
                            if self.rng.random() < 0.3:  # 30% conversion chance
                                self.world_map.tiles[(nx, ny)] = "tundra"

    def _check_composite_biomes(self, water_tiles: Set[Tuple[int, int]],
                                   tick: int) -> List[Dict]:
        """Check if any composite biomes should emerge."""
        new_composites = []
        for (x, y) in list(water_tiles):
            biome = self.world_map.get_biome(x, y)
            for composite_name, rule in COMPOSITE_BIOMES.items():
                required_biome = rule["requires"][0]
                required_condition = rule["requires"][1]
                if biome == required_biome:
                    if required_condition == "high_water":
                        if (x, y) not in self.composite_biome_tiles:
                            self.composite_biome_tiles[(x, y)] = rule["becomes"]
                            new_composites.append({
                                "tick": tick,
                                "tile": [x, y],
                                "from": biome,
                                "to": rule["becomes"],
                                "composite_type": composite_name,
                            })
        return new_composites

    def is_ash_tile(self, x: int, y: int) -> bool:
        """Check if tile is ash (post-volcano, fertile)."""
        return (x, y) in self.ash_tiles

    def get_summary(self) -> Dict:
        return {
            "biome_shifts": len(self.biome_shifts),
            "volcano_eruptions": len(self.eruptions),
            "ash_tiles": len(self.ash_tiles),
            "composite_biomes": len(self.composite_biome_tiles),
            "cold_streak_ticks": self.cold_streak_ticks,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 13.2 DiseaseEvolution
# ═══════════════════════════════════════════════════════════════════════════════

class DiseaseEvolution:
    """
    Disease evolution: diseases mutate over time, new variants emerge.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Disease registry: {disease_name: {virulence, mutation_count, first_tick}}
        self.diseases: Dict[str, Dict] = {
            "swamp_fever": {"virulence": 0.4, "mutation_count": 0, "first_tick": 0},
            "desert_plague": {"virulence": 0.3, "mutation_count": 0, "first_tick": 0},
        }

        self.mutation_events: List[Dict] = []
        self.last_mutation_check = 0
        self.last_variant_generation = 0

    def step(self, tick: int, generation: int,
              avg_immunity: float = 0.0) -> Dict:
        """
        Per-tick disease evolution update.

        Args:
            tick: current tick
            generation: current max generation
            avg_immunity: average immunity across population (0-1)

        Returns:
            Dict of events:
              - mutations: List[Dict]
              - new_variants: List[Dict]
        """
        events = {"mutations": [], "new_variants": []}

        # Periodic mutation (every 10,000 ticks)
        if tick - self.last_mutation_check >= DISEASE_MUTATION_INTERVAL:
            self.last_mutation_check = tick
            mutations = self._mutate_diseases(tick)
            events["mutations"].extend(mutations)

        # Immunity-pressure mutation
        if avg_immunity > 0.5 and self.rng.random() < DISEASE_MUTATION_PROB:
            mutation = self._pressure_mutate(tick, avg_immunity)
            if mutation:
                events["mutations"].append(mutation)

        # New variant every 5 generations
        if generation - self.last_variant_generation >= DISEASE_NEW_VARIANT_GENERATIONS:
            self.last_variant_generation = generation
            new_variant = self._spawn_new_variant(tick, generation)
            events["new_variants"].append(new_variant)

        return events

    def _mutate_diseases(self, tick: int) -> List[Dict]:
        """Increase virulence of all known diseases."""
        mutations = []
        for name, info in self.diseases.items():
            old_virulence = info["virulence"]
            info["virulence"] = min(1.0, old_virulence + DISEASE_VIRULENCE_INCREASE)
            info["mutation_count"] += 1
            mutation = {
                "tick": tick,
                "disease": name,
                "old_virulence": round(old_virulence, 3),
                "new_virulence": round(info["virulence"], 3),
                "mutation_count": info["mutation_count"],
                "reason": "periodic",
            }
            mutations.append(mutation)
            self.mutation_events.append(mutation)
        return mutations

    def _pressure_mutate(self, tick: int, avg_immunity: float) -> Optional[Dict]:
        """Mutate a disease in response to high immunity pressure."""
        # Pick a random disease to mutate
        if not self.diseases:
            return None
        name = self.rng.choice(list(self.diseases.keys()))
        info = self.diseases[name]
        old_virulence = info["virulence"]
        # Pressure mutation: larger increase based on immunity level
        increase = DISEASE_VIRULENCE_INCREASE * (1.0 + avg_immunity)
        info["virulence"] = min(1.0, old_virulence + increase)
        info["mutation_count"] += 1
        mutation = {
            "tick": tick,
            "disease": name,
            "old_virulence": round(old_virulence, 3),
            "new_virulence": round(info["virulence"], 3),
            "mutation_count": info["mutation_count"],
            "reason": "immunity_pressure",
            "immunity_level": round(avg_immunity, 3),
        }
        self.mutation_events.append(mutation)
        return mutation

    def _spawn_new_variant(self, tick: int, generation: int) -> Dict:
        """Create a new disease variant."""
        variant_name = f"variant_gen{generation}"
        self.diseases[variant_name] = {
            "virulence": self.rng.uniform(0.2, 0.6),
            "mutation_count": 0,
            "first_tick": tick,
            "parent_disease": self.rng.choice(list(self.diseases.keys())),
        }
        return {
            "tick": tick,
            "new_variant": variant_name,
            "generation": generation,
            "virulence": round(self.diseases[variant_name]["virulence"], 3),
        }

    def get_disease_virulence(self, name: str) -> float:
        return self.diseases.get(name, {}).get("virulence", 0.0)

    def get_all_diseases(self) -> Dict:
        return dict(self.diseases)

    def get_summary(self) -> Dict:
        return {
            "total_diseases": len(self.diseases),
            "total_mutations": len(self.mutation_events),
            "avg_virulence": (
                sum(d["virulence"] for d in self.diseases.values()) /
                max(1, len(self.diseases))
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 13.3 NoveltyDetector
# ═══════════════════════════════════════════════════════════════════════════════

class NoveltyDetector:
    """
    Automated novelty detector: flags new behaviors not in baseline.
    """

    def __init__(self):
        # Baseline behavior counts
        self.baseline: Dict[str, int] = defaultdict(int)
        self.recent_behaviors: deque = deque(maxlen=NOVELTY_WINDOW_SIZE)
        self.novel_events: List[Dict] = []
        self.baseline_established = False
        self.total_behaviors_recorded = 0

    def record_behavior(self, behavior_signature: str, tick: int) -> bool:
        """
        Record a behavior observation. Returns True if novel (not in baseline).
        """
        self.total_behaviors_recorded += 1
        self.recent_behaviors.append(behavior_signature)

        # Build baseline for first N observations
        if self.total_behaviors_recorded <= NOVELTY_BASELINE_SIZE:
            self.baseline[behavior_signature] += 1
            if self.total_behaviors_recorded == NOVELTY_BASELINE_SIZE:
                self.baseline_established = True
            return False

        # Check if novel (not in baseline with sufficient occurrences)
        if not self.baseline_established:
            return False

        is_novel = self.baseline.get(behavior_signature, 0) < NOVELTY_MIN_OCCURRENCES

        if is_novel:
            event = {
                "tick": tick,
                "behavior": behavior_signature,
                "baseline_count": self.baseline.get(behavior_signature, 0),
            }
            self.novel_events.append(event)
            # Add to baseline so we don't keep flagging it
            self.baseline[behavior_signature] += 1
            return True

        # Increment baseline count
        self.baseline[behavior_signature] += 1
        return False

    def measure_complexity(self) -> Dict:
        """
        Measure current complexity of behavior distribution.

        Higher entropy = more diverse behaviors = more complex.
        """
        if not self.recent_behaviors:
            return {"entropy": 0.0, "unique_behaviors": 0, "total": 0}

        # Count recent behaviors
        counts = defaultdict(int)
        for b in self.recent_behaviors:
            counts[b] += 1

        # Compute Shannon entropy
        total = len(self.recent_behaviors)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            "entropy": round(entropy, 3),
            "unique_behaviors": len(counts),
            "total": total,
        }

    def get_novel_events(self) -> List[Dict]:
        return list(self.novel_events)

    def get_baseline_size(self) -> int:
        return sum(self.baseline.values())

    def get_summary(self) -> Dict:
        complexity = self.measure_complexity()
        return {
            "baseline_established": self.baseline_established,
            "baseline_size": self.get_baseline_size(),
            "unique_baseline_behaviors": len(self.baseline),
            "novel_events_detected": len(self.novel_events),
            "current_entropy": complexity["entropy"],
            "current_unique_behaviors": complexity["unique_behaviors"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 13.4 WorldSeeding
# ═══════════════════════════════════════════════════════════════════════════════

class WorldSeeding:
    """
    Save/load world state + fork simulations for controlled experiments.
    """

    def __init__(self, snapshot_dir: str = "snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        self.snapshots: Dict[str, Dict] = {}

    def snapshot(self, name: str, engines: Dict, tick: int) -> str:
        """
        Save the state of all engines to a snapshot.

        Args:
            name: snapshot name (e.g., "before_extinction")
            engines: dict of {engine_name: engine_object}
            tick: current simulation tick

        Returns:
            Snapshot file path.
        """
        state = {
            "name": name,
            "tick": tick,
            "engines": {},
        }

        for engine_name, engine in engines.items():
            if hasattr(engine, "to_dict"):
                try:
                    state["engines"][engine_name] = engine.to_dict()
                except Exception as e:
                    state["engines"][engine_name] = {"error": str(e)}

        # Save to file
        filename = f"{name}_tick{tick}.json"
        filepath = os.path.join(self.snapshot_dir, filename)
        with open(filepath, "w") as f:
            json.dump(state, f, default=str, indent=2)

        # Keep in memory (max MAX_SNAPSHOTS)
        self.snapshots[name] = state
        if len(self.snapshots) > MAX_SNAPSHOTS:
            # Remove oldest (by tick)
            oldest = min(self.snapshots.items(), key=lambda x: x[1].get("tick", 0))
            del self.snapshots[oldest[0]]

        return filepath

    def restore(self, name: str, engines: Dict) -> Dict:
        """
        Restore engine states from a snapshot.

        Args:
            name: snapshot name
            engines: dict of {engine_name: engine_object}

        Returns:
            The restored state dict.
        """
        if name not in self.snapshots:
            # Try loading from file
            return {}

        state = self.snapshots[name]
        for engine_name, engine in engines.items():
            if engine_name in state["engines"]:
                engine_state = state["engines"][engine_name]
                if hasattr(engine, "load_state") and not isinstance(engine_state, dict) or "error" not in engine_state:
                    try:
                        engine.load_state(engine_state)
                    except Exception:
                        pass  # ignore restore errors
        return state

    def fork(self, name: str, fork_seed: int) -> Dict:
        """
        Fork a snapshot — create a copy with a different random seed
        for divergent evolution.

        Returns:
            The forked state dict (deep copy with seed annotation).
        """
        if name not in self.snapshots:
            return {}

        forked = copy.deepcopy(self.snapshots[name])
        forked["fork_seed"] = fork_seed
        forked["forked_from"] = name
        return forked

    def list_snapshots(self) -> List[Dict]:
        """List all saved snapshots."""
        return [
            {
                "name": name,
                "tick": state.get("tick", 0),
                "engines": list(state.get("engines", {}).keys()),
            }
            for name, state in self.snapshots.items()
        ]

    def compare_runs(self, run_a: Dict, run_b: Dict) -> Dict:
        """
        Compare two simulation runs (forked from same snapshot).

        Returns dict with differences in key metrics.
        """
        return {
            "run_a_tick": run_a.get("tick", 0),
            "run_b_tick": run_b.get("tick", 0),
            "engines_a": set(run_a.get("engines", {}).keys()),
            "engines_b": set(run_b.get("engines", {}).keys()),
            "common_engines": set(run_a.get("engines", {}).keys()) &
                              set(run_b.get("engines", {}).keys()),
        }

    def get_summary(self) -> Dict:
        return {
            "total_snapshots": len(self.snapshots),
            "snapshot_names": list(self.snapshots.keys()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    print("Testing Phase 13 — Open-Ended Extension Systems...")

    # Mock world map
    class MockWorldMap:
        def __init__(self):
            self.width = 16
            self.height = 16
            self.tiles = {}
            for x in range(self.width):
                for y in range(self.height):
                    self.tiles[(x, y)] = "plains"
            # Add some volcano and forest tiles
            self.tiles[(5, 5)] = "volcano"
            for x in range(8, 12):
                for y in range(8, 12):
                    self.tiles[(x, y)] = "forest"
            # Add some tundra
            for x in range(0, 4):
                for y in range(0, 4):
                    self.tiles[(x, y)] = "tundra"
        def get_biome(self, x, y):
            return self.tiles.get((x % self.width, y % self.height), "plains")

    # ── 13.1 WorldEvolution ─────────────────────────────────────────────────
    world_map = MockWorldMap()
    world_evol = WorldEvolution(world_map, seed=42)

    # Simulate some ticks
    for tick in range(100):
        events = world_evol.step(tick, avg_temp=20.0, water_tiles={(10, 10)})
    print(f"  World evolution after 100 ticks: {world_evol.get_summary()}")

    # Force biome shift
    world_evol.last_shift_tick = -50000  # allow shift
    events = world_evol.step(tick=100, avg_temp=20.0)
    print(f"  Biome shift event: {events['biome_shift']}")

    # Force volcano eruption (override prob)
    world_evol.rng = random.Random(0)  # deterministic
    for _ in range(10000):
        eruption = world_evol.step(tick=200, avg_temp=20.0)
        if eruption["volcano_eruption"]:
            print(f"  Volcano eruption: {eruption['volcano_eruption']}")
            break

    # Cold streak → glacier advance
    world_evol2 = WorldEvolution(MockWorldMap(), seed=42)
    glacier_advanced = False
    for tick in range(5001):
        events = world_evol2.step(tick, avg_temp=15.0)  # 5° below baseline 20
        if events.get("glacier_advance"):
            glacier_advanced = True
    print(f"  Glacier advanced during 5000 cold ticks: {glacier_advanced}")
    assert glacier_advanced

    # ── 13.2 DiseaseEvolution ───────────────────────────────────────────────
    disease_evol = DiseaseEvolution(seed=42)
    # Force mutation
    disease_evol.last_mutation_check = -10000
    events = disease_evol.step(tick=10000, generation=1, avg_immunity=0.3)
    print(f"  Disease mutations at t=10000: {len(events['mutations'])}")
    assert len(events["mutations"]) > 0

    # Pressure mutation
    disease_evol2 = DiseaseEvolution(seed=42)
    for _ in range(1000):
        events = disease_evol2.step(tick=100, generation=1, avg_immunity=0.8)
        if events["mutations"]:
            print(f"  Pressure mutation: {events['mutations'][0]['reason']}")
            break

    # New variant every 5 generations
    disease_evol3 = DiseaseEvolution(seed=42)
    events = disease_evol3.step(tick=100, generation=5, avg_immunity=0.3)
    print(f"  New variant at gen 5: {events['new_variants']}")
    assert len(events["new_variants"]) == 1

    # ── 13.3 NoveltyDetector ────────────────────────────────────────────────
    novelty = NoveltyDetector()

    # Build baseline (first 1000 behaviors)
    for i in range(1000):
        novelty.record_behavior(f"behavior_{i % 10}", tick=i)  # 10 unique behaviors

    print(f"  Baseline established: {novelty.baseline_established}")
    assert novelty.baseline_established

    # Now record a novel behavior
    is_novel = novelty.record_behavior("brand_new_behavior", tick=1001)
    print(f"  Novel behavior detected: {is_novel}")
    assert is_novel

    # Known behavior should not be novel
    is_novel2 = novelty.record_behavior("behavior_0", tick=1002)
    assert not is_novel2

    # Complexity measure
    complexity = novelty.measure_complexity()
    print(f"  Complexity: {complexity}")
    assert complexity["entropy"] > 0

    # ── 13.4 WorldSeeding ───────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        seeding = WorldSeeding(snapshot_dir=tmp)

        # Mock engine
        class MockEngine:
            def __init__(self, value=0):
                self.value = value
            def to_dict(self):
                return {"value": self.value}
            def load_state(self, state):
                self.value = state.get("value", 0)

        engine1 = MockEngine(42)
        path = seeding.snapshot("before_extinction", {"engine1": engine1}, tick=1000)
        print(f"  Snapshot saved: {path}")
        assert os.path.exists(path)

        # Modify engine
        engine1.value = 999

        # Restore
        seeding.restore("before_extinction", {"engine1": engine1})
        print(f"  After restore: engine1.value = {engine1.value}")
        assert engine1.value == 42

        # Fork
        forked = seeding.fork("before_extinction", fork_seed=12345)
        print(f"  Forked: seed={forked.get('fork_seed')}")
        assert forked["fork_seed"] == 12345
        assert forked["forked_from"] == "before_extinction"

        # List snapshots
        snaps = seeding.list_snapshots()
        print(f"  Snapshots: {snaps}")
        assert len(snaps) == 1

        # Compare runs
        comparison = seeding.compare_runs(forked, forked)
        print(f"  Comparison: {comparison}")

    print("\n✓ Phase 13 — Open-Ended Extension Systems self-test passed")
