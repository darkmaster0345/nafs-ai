"""
Nafs AI — Mathematical Intuition (Phase 6)
==========================================

Implements MD Phase 6: Adam does not learn mathematics. He develops quantity
intuition through reward signals. These are NOT taught — they emerge from
patterns in experience.

Covers:
  6.1 Quantity Sense
      - Tile food_density encoded in sensory vector as 0.0-1.0 float
      - Agent learns: higher food_density tile → better EAT reward
      - This IS counting without the concept of numbers

  6.2 Pattern & Cycle Recognition
      - Reflection engine tracks: action taken → outcome → tick_of_day
      - Agent gradually learns night = rest time, day = explore time
      - Seasonal cycle: every 5000 ticks, food density shifts across biomes
      - Agent can learn to migrate seasonally if it lives long enough

  6.3 Distance & Space
      - Spatial memory via visit counts per tile
      - Return to known food tiles — spatial navigation emerges
      - Territory behaviour: high-value tiles get defended (Phase 8)

  6.4 Time Sense
      - Internal clock: hunger rate creates natural time perception
      - Agent learns approximately when food respawns (every 200 ticks)
      - Sleep cycle creates circadian rhythm (1 day = 200 ticks)

Design constraints:
  - Does NOT modify base rewards
  - Standalone module
  - Provides sensory signals + tracks patterns for the agent to learn from

Usage:
    from math_intuition import MathIntuitionEngine
    math = MathIntuitionEngine(world_map)
    math.record_visit(x, y, tick)
    math.record_food_eaten(x, y, tick, reward)
    math.record_action_outcome(action, outcome, tick_of_day, reward)
    density = math.get_food_density(x, y)
    ext = math.get_sensory_extensions(x, y, tick)
"""

import random
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SEASONAL_CYCLE_TICKS = 5000      # food density shifts every 5000 ticks
FOOD_RESPAWN_TICKS = 200         # food respawns roughly every 200 ticks
DAY_LENGTH_TICKS = 200           # 1 day = 200 ticks (circadian rhythm)
VISIT_MEMORY_SIZE = 1000         # tiles to remember visits for
ACTION_OUTCOME_MEMORY = 500      # last 500 action-outcome pairs
TILE_VALUE_DECAY = 0.95          # tile value decays slowly when not visited
SEASONAL_SHIFT_MAGNITUDE = 0.3   # how much food density shifts per season


# ═══════════════════════════════════════════════════════════════════════════════
# MathIntuitionEngine
# ═══════════════════════════════════════════════════════════════════════════════

class MathIntuitionEngine:
    """
    Master mathematical intuition engine for the Nafs AI world.

    Provides emergent mathematical intuition through:
      - Per-tile food density tracking (quantity sense)
      - Action-outcome patterns by time of day (cycle recognition)
      - Visit counts per tile (spatial memory)
      - Food respawn timing (time sense)
      - Seasonal biome shifts (long-term pattern recognition)

    The agent does NOT see these directly as 'math' — they appear as
    sensory signals that correlate with reward, allowing the agent's
    PPO brain to develop intuitive quantity/time/space sense.
    """

    def __init__(self, world_map, seed: Optional[int] = None):
        self.world_map = world_map
        self.width = world_map.width
        self.height = world_map.height
        self.rng = random.Random(seed or random.randint(0, 999999))

        # 6.1 Quantity sense: per-tile food density (0.0-1.0)
        # Initialized based on biome food_chance
        self.food_density: Dict[Tuple[int, int], float] = {}
        self._init_food_density()

        # 6.2 Pattern recognition: action → outcome → tick_of_day → reward
        self.action_outcomes: deque = deque(maxlen=ACTION_OUTCOME_MEMORY)
        # Aggregated patterns: {(action, time_bin): {avg_reward, count}}
        self.action_time_patterns: Dict[Tuple[str, int], Dict] = defaultdict(
            lambda: {"total_reward": 0.0, "count": 0}
        )

        # 6.3 Spatial memory: visit counts + tile values
        self.visit_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.tile_values: Dict[Tuple[int, int], float] = defaultdict(float)
        self.food_eaten_at: Dict[Tuple[int, int], int] = {}  # last tick food was eaten at tile

        # 6.4 Time sense: food respawn timing
        self.food_respawn_events: List[Tuple[int, int]] = []  # (tick_spawned, tick_eaten)
        self.estimated_respawn_time = FOOD_RESPAWN_TICKS  # learned over time

        # Seasonal cycle tracking
        self.current_season = 0  # 0=spring, 1=summer, 2=autumn, 3=winter
        self.last_seasonal_shift = 0

        self.current_tick = 0

    def _init_food_density(self) -> None:
        """Initialize food density per tile based on biome."""
        from world_sim import BIOMES
        for x in range(self.width):
            for y in range(self.height):
                biome = self.world_map.get_biome(x, y)
                biome_data = BIOMES.get(biome, {})
                # Base density from biome food_chance
                base_density = min(1.0, biome_data.get("food_chance", 0.1) * 5)
                self.food_density[(x, y)] = base_density

    # ─────────────────────────────────────────────────────────────────────────
    # 6.1 Quantity Sense
    # ─────────────────────────────────────────────────────────────────────────

    def get_food_density(self, x: int, y: int) -> float:
        """Get food density at tile (0.0-1.0)."""
        wx = x % self.width
        wy = y % self.height
        return self.food_density.get((wx, wy), 0.0)

    def set_food_density(self, x: int, y: int, density: float) -> None:
        """Set food density at tile (used when food spawned/eaten)."""
        wx = x % self.width
        wy = y % self.height
        self.food_density[(wx, wy)] = max(0.0, min(1.0, density))

    def consume_food(self, x: int, y: int) -> None:
        """Mark food as consumed at tile (density drops to 0)."""
        self.set_food_density(x, y, 0.0)
        self.food_eaten_at[(x % self.width, y % self.height)] = self.current_tick

    def maybe_respawn_food(self, x: int, y: int) -> bool:
        """
        Check if food should respawn at tile.
        Returns True if food respawned this tick.
        """
        tile = (x % self.width, y % self.height)
        if self.food_density.get(tile, 0) > 0:
            return False  # already has food
        last_eaten = self.food_eaten_at.get(tile)
        if last_eaten is None:
            return False  # never had food
        if self.current_tick - last_eaten >= self.estimated_respawn_time:
            # Respawn food
            from world_sim import BIOMES
            biome = self.world_map.get_biome(x, y)
            base_density = min(1.0, BIOMES.get(biome, {}).get("food_chance", 0.1) * 5)
            self.food_density[tile] = base_density
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # 6.2 Pattern & Cycle Recognition
    # ─────────────────────────────────────────────────────────────────────────

    def record_action_outcome(self, action: str, outcome: str,
                                tick_of_day: int, reward: float) -> None:
        """
        Record an action-outcome pair for pattern learning.

        The agent's PPO brain will gradually learn which actions work best
        at which times of day by recognizing patterns in reward signals.
        """
        # Time bin: 0=night (18-6), 1=morning (6-12), 2=afternoon (12-18)
        if tick_of_day < 6 or tick_of_day >= 18:
            time_bin = 0  # night
        elif tick_of_day < 12:
            time_bin = 1  # morning
        else:
            time_bin = 2  # afternoon

        self.action_outcomes.append({
            "action": action,
            "outcome": outcome,
            "tick_of_day": tick_of_day,
            "time_bin": time_bin,
            "reward": reward,
            "tick": self.current_tick,
        })

        # Update aggregated pattern
        key = (action, time_bin)
        self.action_time_patterns[key]["total_reward"] += reward
        self.action_time_patterns[key]["count"] += 1

    def get_action_pattern(self, action: str, time_bin: int) -> Dict:
        """Get average reward for an action at a time of day."""
        key = (action, time_bin)
        pattern = self.action_time_patterns.get(key, {"total_reward": 0.0, "count": 0})
        if pattern["count"] == 0:
            return {"avg_reward": 0.0, "count": 0}
        return {
            "avg_reward": pattern["total_reward"] / pattern["count"],
            "count": pattern["count"],
        }

    def get_best_action_for_time(self, time_bin: int,
                                  available_actions: List[str]) -> Optional[str]:
        """Returns the action with highest historical avg reward at this time."""
        best_action = None
        best_reward = float('-inf')
        for action in available_actions:
            pattern = self.get_action_pattern(action, time_bin)
            if pattern["count"] >= 5 and pattern["avg_reward"] > best_reward:
                best_reward = pattern["avg_reward"]
                best_action = action
        return best_action

    def _check_seasonal_shift(self) -> None:
        """Check if a seasonal shift should occur (every 5000 ticks)."""
        if self.current_tick - self.last_seasonal_shift >= SEASONAL_CYCLE_TICKS:
            self.last_seasonal_shift = self.current_tick
            self.current_season = (self.current_season + 1) % 4
            self._apply_seasonal_shift()

    def _apply_seasonal_shift(self) -> None:
        """
        Shift food density across biomes based on season.
        This creates long-term patterns the agent can learn to anticipate.
        """
        from world_sim import BIOMES
        season_mult = {
            0: 1.0,   # spring: normal
            1: 1.2,   # summer: more food
            2: 0.9,   # autumn: less food
            3: 0.6,   # winter: much less food
        }[self.current_season]

        for x in range(self.width):
            for y in range(self.height):
                biome = self.world_map.get_biome(x, y)
                biome_data = BIOMES.get(biome, {})
                base = min(1.0, biome_data.get("food_chance", 0.1) * 5)
                # Apply seasonal multiplier with random variation
                shifted = base * season_mult * self.rng.uniform(0.8, 1.2)
                self.food_density[(x, y)] = max(0.0, min(1.0, shifted))

    def get_season_name(self) -> str:
        return ["spring", "summer", "autumn", "winter"][self.current_season]

    # ─────────────────────────────────────────────────────────────────────────
    # 6.3 Distance & Space (spatial memory)
    # ─────────────────────────────────────────────────────────────────────────

    def record_visit(self, x: int, y: int, tick: int) -> None:
        """Record that the agent visited a tile."""
        tile = (x % self.width, y % self.height)
        self.visit_counts[tile] += 1

    def get_visit_count(self, x: int, y: int) -> int:
        """Get number of times agent has visited a tile."""
        return self.visit_counts.get((x % self.width, y % self.height), 0)

    def record_tile_value(self, x: int, y: int, value: float) -> None:
        """Record the value of a tile (e.g., +1 for food, -1 for danger)."""
        tile = (x % self.width, y % self.height)
        # Weighted update: recent experiences matter more
        self.tile_values[tile] = self.tile_values[tile] * 0.7 + value * 0.3

    def get_tile_value(self, x: int, y: int) -> float:
        """Get the cached value of a tile (-1 to +1)."""
        return self.tile_values.get((x % self.width, y % self.height), 0.0)

    def find_known_food_tiles(self, max_distance: int = 10,
                                from_pos: Tuple[int, int] = (0, 0)
                                ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Find tiles the agent remembers having food, within max_distance.

        Returns list of ((x, y), last_known_density) sorted by distance.
        """
        results = []
        for tile, value in self.tile_values.items():
            if value > 0.1:  # positive value = good tile
                dist = abs(tile[0] - from_pos[0]) + abs(tile[1] - from_pos[1])
                if dist <= max_distance:
                    results.append((tile, value, dist))

        # Sort by distance
        results.sort(key=lambda r: r[2])
        return [(r[0], r[1]) for r in results[:10]]  # top 10 nearest

    def decay_tile_values(self) -> None:
        """Slowly decay tile values so old memories fade."""
        for tile in list(self.tile_values.keys()):
            self.tile_values[tile] *= TILE_VALUE_DECAY
            if abs(self.tile_values[tile]) < 0.01:
                del self.tile_values[tile]

    # ─────────────────────────────────────────────────────────────────────────
    # 6.4 Time Sense
    # ─────────────────────────────────────────────────────────────────────────

    def update_respawn_estimate(self, actual_respawn_time: int) -> None:
        """Update the estimated food respawn time based on observation."""
        # Exponential moving average
        alpha = 0.1
        self.estimated_respawn_time = (
            alpha * actual_respawn_time + (1 - alpha) * self.estimated_respawn_time
        )

    def get_internal_clock_signal(self, tick: int) -> Dict:
        """
        Return internal clock signals based on tick.

        The agent's PPO brain can learn these signals correlate with
        food availability, danger, sleep need, etc.
        """
        day_progress = (tick % DAY_LENGTH_TICKS) / DAY_LENGTH_TICKS  # 0-1
        is_night = (tick % DAY_LENGTH_TICKS) > (DAY_LENGTH_TICKS / 2)
        return {
            "day_progress": round(day_progress, 3),
            "is_night": is_night,
            "season": self.current_season,
            "season_name": self.get_season_name(),
            "estimated_respawn_time": round(self.estimated_respawn_time, 1),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, tick: int) -> None:
        """Per-tick update: check seasonal shifts, decay tile values."""
        self.current_tick = tick
        self._check_seasonal_shift()

        # Periodically decay tile values (every 100 ticks)
        if tick % 100 == 0:
            self.decay_tile_values()

    # ─────────────────────────────────────────────────────────────────────────
    # Sensory extensions
    # ─────────────────────────────────────────────────────────────────────────

    def get_sensory_extensions(self, x: int, y: int, tick: int) -> Dict:
        """
        Returns math-intuition sensory fields:
          - food_density (0-1): quantity sense
          - visit_count (int): how many times agent has been here
          - tile_value (-1 to +1): cached value of this tile
          - day_progress (0-1): circadian rhythm signal
          - is_night (bool): time of day
          - season (int 0-3): seasonal cycle
          - estimated_respawn_time (ticks): learned food timing
        """
        clock = self.get_internal_clock_signal(tick)
        return {
            'food_density': round(self.get_food_density(x, y), 3),
            'visit_count': self.get_visit_count(x, y),
            'tile_value': round(self.get_tile_value(x, y), 3),
            'day_progress': clock["day_progress"],
            'is_night': clock["is_night"],
            'season': clock["season"],
            'season_name': clock["season_name"],
            'estimated_respawn_time': clock["estimated_respawn_time"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialize for checkpointing."""
        return {
            'food_density': {f"{k[0]},{k[1]}": v for k, v in self.food_density.items()},
            'visit_counts': {f"{k[0]},{k[1]}": v for k, v in self.visit_counts.items()},
            'tile_values': {f"{k[0]},{k[1]}": v for k, v in self.tile_values.items()},
            'food_eaten_at': {f"{k[0]},{k[1]}": v for k, v in self.food_eaten_at.items()},
            'action_time_patterns': {
                f"{k[0]}|{k[1]}": v for k, v in self.action_time_patterns.items()
            },
            'current_season': self.current_season,
            'last_seasonal_shift': self.last_seasonal_shift,
            'estimated_respawn_time': self.estimated_respawn_time,
            'current_tick': self.current_tick,
        }

    def load_state(self, state: Dict) -> None:
        """Restore from checkpoint."""
        self.food_density = {
            tuple(int(c) for c in k.split(',')): v
            for k, v in state.get('food_density', {}).items()
        }
        self.visit_counts = defaultdict(int, {
            tuple(int(c) for c in k.split(',')): v
            for k, v in state.get('visit_counts', {}).items()
        })
        self.tile_values = defaultdict(float, {
            tuple(int(c) for c in k.split(',')): v
            for k, v in state.get('tile_values', {}).items()
        })
        self.food_eaten_at = {
            tuple(int(c) for c in k.split(',')): v
            for k, v in state.get('food_eaten_at', {}).items()
        }
        self.action_time_patterns = defaultdict(
            lambda: {"total_reward": 0.0, "count": 0}
        )
        for k, v in state.get('action_time_patterns', {}).items():
            action, time_bin = k.split('|')
            self.action_time_patterns[(action, int(time_bin))] = v
        self.current_season = state.get('current_season', 0)
        self.last_seasonal_shift = state.get('last_seasonal_shift', 0)
        self.estimated_respawn_time = state.get('estimated_respawn_time', FOOD_RESPAWN_TICKS)
        self.current_tick = state.get('current_tick', 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing MathIntuitionEngine...")

    # Mock world map
    class MockWorldMap:
        def __init__(self):
            self.width = 16
            self.height = 16
        def get_biome(self, x, y):
            return "forest"

    world_map = MockWorldMap()
    math_engine = MathIntuitionEngine(world_map, seed=42)

    # Test 6.1: Quantity sense
    density = math_engine.get_food_density(5, 5)
    print(f"  Initial food density at (5,5): {density:.3f}")
    assert 0 <= density <= 1

    # Consume food → density drops to 0
    math_engine.consume_food(5, 5)
    assert math_engine.get_food_density(5, 5) == 0.0
    print(f"  After consume: density = {math_engine.get_food_density(5, 5)}")

    # Test respawn timing
    math_engine.current_tick = 250  # 250 > 200 (respawn time)
    respawned = math_engine.maybe_respawn_food(5, 5)
    print(f"  Respawn at tick 250: {respawned}")
    assert respawned is True

    # Test 6.2: Pattern recognition
    for tick in range(100):
        math_engine.record_action_outcome(
            action="EXPLORE", outcome="found_food",
            tick_of_day=10, reward=0.5,
        )
    pattern = math_engine.get_action_pattern("EXPLORE", time_bin=1)  # morning
    print(f"  EXPLORE in morning: avg_reward={pattern['avg_reward']:.3f}, count={pattern['count']}")
    assert pattern["count"] == 100
    assert pattern["avg_reward"] == 0.5

    # Best action for morning
    best = math_engine.get_best_action_for_time(
        time_bin=1, available_actions=["EXPLORE", "SLEEP", "EAT"]
    )
    print(f"  Best action for morning: {best}")
    assert best == "EXPLORE"

    # Test seasonal shift
    math_engine2 = MathIntuitionEngine(world_map, seed=42)
    initial_density = math_engine2.get_food_density(5, 5)
    # Force seasonal shift (every 5000 ticks)
    math_engine2.step(tick=5000)
    new_density = math_engine2.get_food_density(5, 5)
    print(f"  After seasonal shift (tick 5000): {initial_density:.3f} → {new_density:.3f}")
    print(f"  Current season: {math_engine2.get_season_name()}")
    assert math_engine2.current_season == 1  # spring → summer

    # Test 6.3: Spatial memory
    math_engine3 = MathIntuitionEngine(world_map, seed=42)
    math_engine3.record_visit(5, 5, tick=10)
    math_engine3.record_visit(5, 5, tick=20)
    math_engine3.record_visit(5, 5, tick=30)
    assert math_engine3.get_visit_count(5, 5) == 3

    # Tile value recording
    math_engine3.record_tile_value(5, 5, 1.0)  # found food here
    value = math_engine3.get_tile_value(5, 5)
    print(f"  Tile value at (5,5): {value:.3f}")
    assert value > 0

    # Find known food tiles
    math_engine3.record_tile_value(6, 5, 0.8)
    math_engine3.record_tile_value(7, 5, 0.5)
    known = math_engine3.find_known_food_tiles(max_distance=5, from_pos=(5, 5))
    print(f"  Known food tiles: {len(known)}")
    assert len(known) >= 1

    # Test 6.4: Time sense
    clock = math_engine.get_internal_clock_signal(tick=100)
    print(f"  Internal clock at tick 100: {clock}")
    assert 0 <= clock["day_progress"] <= 1

    # Test respawn time update
    math_engine.update_respawn_estimate(220)
    print(f"  Estimated respawn time: {math_engine.estimated_respawn_time:.1f}")
    assert math_engine.estimated_respawn_time != FOOD_RESPAWN_TICKS  # updated

    # Test sensory extensions
    ext = math_engine.get_sensory_extensions(5, 5, tick=100)
    required = {"food_density", "visit_count", "tile_value", "day_progress",
                "is_night", "season", "season_name", "estimated_respawn_time"}
    print(f"  Sensory extensions: {list(ext.keys())}")
    assert set(ext.keys()) == required

    print("\n✓ MathIntuitionEngine self-test passed")
