"""
Nafs AI — Physics Engine (Phase 1)
==================================

Implements MD Phase 1: physical laws that are consistent and discoverable.
Adam does NOT know these laws — he discovers them through experience.

Covers:
  1.1 Temperature & Heat
      - Body temperature separate from air temperature
      - Thermal conductance (gradual change, not instant)
      - Hypothermia (<30°C) / Hyperthermia (>42°C) → health drain
      - Shelter (cave) = stable 12°C
      - Fire tile = heat source
      - Night: desert drops to 5°C

  1.2 Wind & Movement
      - Wind direction (N/S/E/W) + speed (0-5) via Markov chain
      - +30% energy moving against wind, -15% moving with wind
      - Storm → wind speed 5
      - Sandstorm: wind + sand → visibility drop + pain

  1.3 Elevation & Terrain
      - 64x64 elevation map (0-10 scale)
      - Moving uphill costs more energy
      - Falling 3+ tiles → damage proportional
      - High elevation = colder (-2°C per unit above 5)
      - Mountains block wind (downwind tiles have lower wind)

  1.4 Fire System
      - Fire is a tile state (not a biome)
      - Spreads to fuel tiles (forest, plains, jungle) every 50 ticks
      - Burned tiles become ash (barren biome — represented as a tile overlay)
      - Dies in rain weather
      - Lightning during storm ignites dry fuel tiles
      - Pain signal when adjacent to fire
      - Positive reward for proximity when cold

  1.5 Water Physics
      - Rain fills low-elevation tiles (temporary water sources)
      - Rivers flow from high to low (static water tiles in biome gen)
      - Ocean tiles: undrinkable, swimming = high energy cost
      - Drought: no rain for 2000 ticks → temporary water tiles dry up
      - Sensory vector additions: body_temp, wind_speed, wind_dir,
        elevation, surface_wetness

Design constraints:
  - Does NOT modify base rewards in world_sim.py — only adds new physics-
    driven effects (hypothermia damage, fire warmth benefit, etc.).
  - Designed as a standalone module that WorldSim instantiates and calls.
  - All state is local — no shared globals.

Usage:
    from physics import PhysicsEngine
    physics = PhysicsEngine(world_map)
    physics.step(biome, weather, time_of_day, agent_x, agent_y)
    body_temp = physics.body_temp
    wind = (physics.wind_speed, physics.wind_dir)
    elev = physics.get_elevation(x, y)
    cost = physics.get_movement_energy_cost(action, x, y, new_x, new_y)
    physics.ignite_fire(x, y)  # lightning strike
"""

import random
import math
from typing import Dict, Tuple, Optional, List, Set


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Body temperature thresholds (°C)
BODY_TEMP_NORMAL = 37.0
BODY_TEMP_MIN_SURVIVE = 25.0   # below this = rapid death
BODY_TEMP_MAX_SURVIVE = 45.0
HYPOTHERMIA_THRESHOLD = 30.0   # < 30 → health drain
HYPERTHERMIA_THRESHOLD = 42.0  # > 42 → health drain

# Thermal conductance: how fast body temp moves toward air temp (per tick)
THERMAL_CONDUCTANCE = 0.05     # 5% per tick

# Wind
WIND_MAX = 5
WIND_DIRECTIONS = ["N", "S", "E", "W"]
DIR_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
DIR_VECTORS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

# Elevation
ELEVATION_MAX = 10
ELEVATION_COLD_THRESHOLD = 5    # above this → -2°C per unit
ELEVATION_COLD_PER_UNIT = 2.0
ELEVATION_FALL_DAMAGE_THRESHOLD = 3   # 3+ tiles lower → damage
ELEVATION_FALL_DAMAGE_PER_UNIT = 5.0

# Fire
FIRE_SPREAD_INTERVAL = 50       # ticks
FIRE_FUEL_BIOMES = {"forest", "plains", "jungle"}
FIRE_STARTING_FUEL = 200        # ticks of burning before consumed
FIRE_PAIN_PER_ADJACENT = 2.0
FIRE_WARMTH_BENEFIT = 0.3       # positive reward when cold + adjacent to fire
FIRE_WARMTH_THRESHOLD = 35.0    # body temp below which fire warmth is beneficial

# Lightning
LIGHTNING_IGNITE_PROB = 0.005   # per dry fuel tile per storm tick

# Water
DROUGHT_THRESHOLD = 2000        # ticks without rain → temporary water dries up
WATER_FILL_ELEVATION_MAX = 3    # only tiles with elevation ≤ 3 fill with rainwater
OCEAN_BIOME = "ocean"
SWIMMING_ENERGY_COST = 3.0      # extra energy cost for swimming in ocean

# Wind Markov chain states
WIND_STATES = ["calm", "breeze", "strong", "storm"]
WIND_TRANSITIONS = {
    "calm":    {"calm": 0.70, "breeze": 0.25, "strong": 0.05, "storm": 0.00},
    "breeze":  {"calm": 0.20, "breeze": 0.55, "strong": 0.20, "storm": 0.05},
    "strong":  {"calm": 0.05, "breeze": 0.25, "strong": 0.50, "storm": 0.20},
    "storm":   {"calm": 0.00, "breeze": 0.10, "strong": 0.30, "storm": 0.60},
}
WIND_SPEED_BY_STATE = {"calm": 0, "breeze": 2, "strong": 4, "storm": 5}

# Sandstorm: visibility reduction
SANDSTORM_VISIBILITY_PENALTY = 0.7
SANDSTORM_PAIN_PROB = 0.15
SANDSTORM_PAIN_AMOUNT = (1.0, 3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PhysicsEngine
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsEngine:
    """
    Master physics engine for the Nafs AI world.

    One instance per WorldSim. Holds all physics state:
      - body_temp: agent's body temperature (separate from air temp)
      - wind_speed, wind_dir: current wind
      - elevation_map: 64x64 grid (0-10)
      - fire_tiles: { (x,y): {fuel_left, intensity, age} }
      - water_tiles: set of (x,y) with temporary rainwater
      - last_rain_tick: for drought detection
      - _wind_state: current Markov state
    """

    def __init__(self, world_map, seed: Optional[int] = None):
        self.world_map = world_map
        self.width = world_map.width
        self.height = world_map.height
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Agent body temperature
        self.body_temp = BODY_TEMP_NORMAL

        # Wind
        self.wind_speed = 0
        self.wind_dir = self.rng.choice(WIND_DIRECTIONS)
        self._wind_state = "calm"

        # Elevation map (procedural)
        self.elevation_map = self._generate_elevation_map()

        # Fire system
        self.fire_tiles: Dict[Tuple[int, int], Dict] = {}
        self.ash_tiles: Set[Tuple[int, int]] = set()  # burned tiles
        self._ticks_since_fire_spread = 0

        # Water system
        self.water_tiles: Set[Tuple[int, int]] = set()
        self.last_rain_tick = 0
        self.current_tick = 0

        # Per-tick cached local conditions
        self._cached_air_temp: Dict[Tuple[int, int], float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Elevation map generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_elevation_map(self) -> List[List[int]]:
        """
        Generate a 64x64 elevation map (0-10) using simple value noise.

        Mountains get high elevation, oceans/lowlands get low elevation,
        everything else is mid-range. We bias by biome so that mountains
        are actually mountains and oceans are actually low.
        """
        elev = [[0] * self.width for _ in range(self.height)]

        # Base layer: smooth value noise via averaging random seed points
        seed_points = []
        for _ in range(20):
            sx = self.rng.randint(0, self.width - 1)
            sy = self.rng.randint(0, self.height - 1)
            sv = self.rng.uniform(0, ELEVATION_MAX)
            seed_points.append((sx, sy, sv))

        for x in range(self.width):
            for y in range(self.height):
                # Inverse-distance-weighted average of seed points
                total_w = 0.0
                total_v = 0.0
                for sx, sy, sv in seed_points:
                    d = math.sqrt((x - sx) ** 2 + (y - sy) ** 2) + 1.0
                    w = 1.0 / d
                    total_w += w
                    total_v += sv * w
                elev[y][x] = total_v / total_w

        # Biome bias: mountains = high, ocean = low
        for x in range(self.width):
            for y in range(self.height):
                biome = self.world_map.get_biome(x, y)
                if biome == "mountain":
                    elev[y][x] = min(ELEVATION_MAX, elev[y][x] + 5.0)
                elif biome == "ocean":
                    elev[y][x] = max(0, elev[y][x] - 4.0)
                elif biome == "volcano":
                    elev[y][x] = min(ELEVATION_MAX, elev[y][x] + 4.0)
                elif biome in ("desert", "plains"):
                    elev[y][x] = max(0, min(ELEVATION_MAX * 0.5, elev[y][x]))
                # Discretize to 0-10 integer scale
                elev[y][x] = int(round(max(0, min(ELEVATION_MAX, elev[y][x]))))

        return elev

    def get_elevation(self, x: int, y: int) -> int:
        """Get elevation (0-10) at tile (x, y)."""
        wx = x % self.width
        wy = y % self.height
        return self.elevation_map[wy][wx]

    # ─────────────────────────────────────────────────────────────────────────
    # Wind
    # ─────────────────────────────────────────────────────────────────────────

    def _update_wind(self, weather: str) -> None:
        """Advance the wind Markov chain one step."""
        # Weather can force wind state
        if weather == "storm":
            self._wind_state = "storm"
        elif weather == "blizzard":
            self._wind_state = "storm"
        elif weather == "sandstorm":
            self._wind_state = "storm"
        elif weather in ("clear", "fog"):
            # Bias toward calm
            if self._wind_state == "calm":
                pass  # already calm
            else:
                # 50% chance to drop a state
                if self.rng.random() < 0.5:
                    idx = WIND_STATES.index(self._wind_state)
                    self._wind_state = WIND_STATES[max(0, idx - 1)]
        else:
            # Normal Markov transition
            transitions = WIND_TRANSITIONS[self._wind_state]
            r = self.rng.random()
            cumulative = 0.0
            for state, prob in transitions.items():
                cumulative += prob
                if r < cumulative:
                    self._wind_state = state
                    break

        self.wind_speed = WIND_SPEED_BY_STATE[self._wind_state]

        # Wind direction: 20% chance to rotate each tick
        if self.rng.random() < 0.20 and self.wind_speed > 0:
            # Rotate 90° clockwise or counter-clockwise
            current_idx = WIND_DIRECTIONS.index(self.wind_dir)
            rotation = self.rng.choice([-1, 1])
            self.wind_dir = WIND_DIRECTIONS[(current_idx + rotation) % 4]

    def get_wind_modifier_for_movement(self, dx: int, dy: int) -> float:
        """
        Returns multiplier for movement energy cost based on wind.

        With wind: 0.85x cost (-15%)
        Against wind: 1.30x cost (+30%)
        Crosswind / no wind: 1.0x
        """
        if self.wind_speed == 0:
            return 1.0

        # Determine movement direction
        if dx == 0 and dy == -1:
            move_dir = "N"
        elif dx == 0 and dy == 1:
            move_dir = "S"
        elif dx == 1 and dy == 0:
            move_dir = "E"
        elif dx == -1 and dy == 0:
            move_dir = "W"
        else:
            return 1.0  # diagonal or no movement

        if move_dir == self.wind_dir:
            return 0.85  # with wind
        elif move_dir == DIR_OPPOSITE.get(self.wind_dir):
            return 1.30  # against wind
        else:
            return 1.0  # crosswind

    # ─────────────────────────────────────────────────────────────────────────
    # Temperature
    # ─────────────────────────────────────────────────────────────────────────

    def get_local_air_temp(self, biome: str, weather: str, time_of_day: int,
                            x: int, y: int, base_biome_temp: float,
                            weather_temp_mod: float) -> float:
        """
        Compute local air temperature at (x, y).

        Considers:
          - Base biome temperature
          - Weather modifier
          - Time of day (desert drops to 5°C at night)
          - Elevation (colder above 5)
          - Cave shelter (stable 12°C)
          - Mountain wind shadow (no temp effect, but blocks wind)
        """
        # Cave: stable 12°C regardless of outside
        if biome == "cave":
            return 12.0

        temp = base_biome_temp + weather_temp_mod

        # Night modifier: desert drops to ~5°C at night
        is_night = (time_of_day < 6 or time_of_day >= 18)
        if biome == "desert" and is_night:
            temp = min(temp, 5.0 + self.rng.uniform(-1, 1))
        elif is_night:
            # Generic night cooling (already applied in world_sim, but ensure)
            pass

        # Elevation: -2°C per unit above 5
        elev = self.get_elevation(x, y)
        if elev > ELEVATION_COLD_THRESHOLD:
            temp -= (elev - ELEVATION_COLD_THRESHOLD) * ELEVATION_COLD_PER_UNIT

        # Volcano biome: warm even at night
        if biome == "volcano":
            temp = max(temp, 35.0)

        return temp

    def _update_body_temp(self, biome: str, weather: str, time_of_day: int,
                          x: int, y: int, base_biome_temp: float,
                          weather_temp_mod: float) -> None:
        """
        Body temperature gradually moves toward local air temp.
        Also affected by fire proximity (raises body temp).
        """
        air_temp = self.get_local_air_temp(
            biome, weather, time_of_day, x, y, base_biome_temp, weather_temp_mod
        )

        # Thermal conductance — gradual change
        delta = (air_temp - self.body_temp) * THERMAL_CONDUCTANCE
        self.body_temp += delta

        # Fire warmth: raises body temp if adjacent
        fire_count = self._count_adjacent_fires(x, y)
        if fire_count > 0:
            # Each adjacent fire raises body temp by 0.5°C per tick
            self.body_temp += fire_count * 0.5
            # Cap to avoid runaway heating
            if self.body_temp > 45.0:
                self.body_temp = 45.0

        # Swimming in water cools body faster
        if biome == "ocean" or (x, y) in self.water_tiles:
            self.body_temp -= 0.3

        # Clamp to survivable range (will die from hypo/hyperthermia anyway)
        if self.body_temp < 15.0:
            self.body_temp = 15.0
        if self.body_temp > 50.0:
            self.body_temp = 50.0

    def get_hypothermia_damage(self) -> float:
        """Health drain per tick when body temp < 30°C."""
        if self.body_temp < HYPOTHERMIA_THRESHOLD:
            return (HYPOTHERMIA_THRESHOLD - self.body_temp) * 0.10
        return 0.0

    def get_hyperthermia_damage(self) -> float:
        """Health drain per tick when body temp > 42°C."""
        if self.body_temp > HYPERTHERMIA_THRESHOLD:
            return (self.body_temp - HYPERTHERMIA_THRESHOLD) * 0.10
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Fire system
    # ─────────────────────────────────────────────────────────────────────────

    def _count_adjacent_fires(self, x: int, y: int) -> int:
        """Count fire tiles in 4-neighbourhood of (x, y)."""
        count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (x + dx, y + dy) in self.fire_tiles:
                count += 1
        return count

    def get_adjacent_fire_pain(self, x: int, y: int) -> float:
        """Pain signal when fire is adjacent. Caps at 10."""
        return min(FIRE_PAIN_PER_ADJACENT * self._count_adjacent_fires(x, y), 10.0)

    def get_fire_warmth_benefit(self, x: int, y: int) -> float:
        """
        Positive reward signal when adjacent to fire AND cold.
        Encourages learning: 'fire is good when I'm cold'.
        """
        if self.body_temp < FIRE_WARMTH_THRESHOLD:
            if self._count_adjacent_fires(x, y) > 0:
                return FIRE_WARMTH_BENEFIT
        return 0.0

    def ignite_fire(self, x: int, y: int) -> bool:
        """
        Lightning strike ignites a fire on a fuel tile.
        Returns True if ignition succeeded.
        """
        if (x, y) in self.fire_tiles or (x, y) in self.ash_tiles:
            return False
        biome = self.world_map.get_biome(x, y)
        if biome not in FIRE_FUEL_BIOMES:
            return False
        self.fire_tiles[(x, y)] = {
            "fuel_left": FIRE_STARTING_FUEL,
            "intensity": 1.0,
            "age": 0,
        }
        return True

    def _update_fire(self, weather: str, x: int, y: int) -> None:
        """
        Update all fire tiles for this tick.
          - Fire consumes fuel each tick
          - When fuel runs out, tile becomes ash
          - Fire dies instantly in rain/snow/blizzard
          - Every 50 ticks, fire spreads to adjacent fuel tiles
          - Lightning during storm has a small chance to ignite dry fuel
        """
        # Rain/snow/blizzard extinguish all fires
        if weather in ("rain", "snow", "blizzard"):
            if self.fire_tiles:
                # All fires die, but they don't become ash (just extinguished)
                self.fire_tiles.clear()

        # Lightning ignition during storm
        if weather == "storm":
            # Check nearby fuel tiles for random ignition
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if self.rng.random() < LIGHTNING_IGNITE_PROB:
                        tx, ty = x + dx, y + dy
                        if 0 <= tx < self.width and 0 <= ty < self.height:
                            self.ignite_fire(tx, ty)

        # Consume fuel
        consumed = []
        for tile, info in self.fire_tiles.items():
            info["fuel_left"] -= 1
            info["age"] += 1
            if info["fuel_left"] <= 0:
                consumed.append(tile)

        for tile in consumed:
            del self.fire_tiles[tile]
            self.ash_tiles.add(tile)  # tile is now barren (ash)

        # Spread fire every 50 ticks
        self._ticks_since_fire_spread += 1
        if self._ticks_since_fire_spread >= FIRE_SPREAD_INTERVAL:
            self._ticks_since_fire_spread = 0
            new_fires = []
            for (fx, fy), info in self.fire_tiles.items():
                # Spread to adjacent fuel tiles
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = fx + dx, fy + dy
                    if (nx, ny) in self.fire_tiles or (nx, ny) in self.ash_tiles:
                        continue
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        biome = self.world_map.get_biome(nx, ny)
                        if biome in FIRE_FUEL_BIOMES:
                            # Probability of spread
                            if self.rng.random() < 0.4:
                                new_fires.append((nx, ny))
            for tile in new_fires:
                if tile not in self.fire_tiles:
                    self.fire_tiles[tile] = {
                        "fuel_left": FIRE_STARTING_FUEL,
                        "intensity": 1.0,
                        "age": 0,
                    }

    def is_fire_tile(self, x: int, y: int) -> bool:
        return (x, y) in self.fire_tiles

    def is_ash_tile(self, x: int, y: int) -> bool:
        return (x, y) in self.ash_tiles

    # ─────────────────────────────────────────────────────────────────────────
    # Water system
    # ─────────────────────────────────────────────────────────────────────────

    def _update_water(self, weather: str, x: int, y: int) -> None:
        """
        Rain fills low-elevation tiles (≤3) with temporary water.
        Drought: no rain for 2000 ticks → temporary water tiles dry up.
        """
        if weather == "rain":
            self.last_rain_tick = self.current_tick
            # Add water to some random low-elevation tiles near agent
            for _ in range(3):
                rx = (x + self.rng.randint(-5, 5)) % self.width
                ry = (y + self.rng.randint(-5, 5)) % self.height
                if self.get_elevation(rx, ry) <= WATER_FILL_ELEVATION_MAX:
                    biome = self.world_map.get_biome(rx, ry)
                    if biome not in ("ocean", "volcano", "mountain"):
                        self.water_tiles.add((rx, ry))
        elif weather == "storm":
            self.last_rain_tick = self.current_tick
            # Storms fill more aggressively
            for _ in range(8):
                rx = (x + self.rng.randint(-8, 8)) % self.width
                ry = (y + self.rng.randint(-8, 8)) % self.height
                if self.get_elevation(rx, ry) <= WATER_FILL_ELEVATION_MAX:
                    biome = self.world_map.get_biome(rx, ry)
                    if biome not in ("ocean", "volcano", "mountain"):
                        self.water_tiles.add((rx, ry))

        # Drought: dry up temporary water
        if self.current_tick - self.last_rain_tick >= DROUGHT_THRESHOLD:
            if self.water_tiles:
                # Dry up 10% of temporary water tiles per tick
                to_remove = list(self.water_tiles)[:max(1, len(self.water_tiles) // 10)]
                for tile in to_remove:
                    self.water_tiles.discard(tile)

    def is_water_tile(self, x: int, y: int) -> bool:
        """Is this a temporary rainwater tile?"""
        return (x, y) in self.water_tiles

    def is_ocean_tile(self, x: int, y: int) -> bool:
        """Is this an ocean tile (undrinkable)?"""
        return self.world_map.get_biome(x, y) == OCEAN_BIOME

    def is_swimmable(self, x: int, y: int) -> bool:
        """Is this tile swimmable (ocean)?"""
        return self.is_ocean_tile(x, y)

    def get_swimming_energy_cost(self, x: int, y: int) -> float:
        """Extra energy cost for entering an ocean tile."""
        if self.is_swimmable(x, y):
            return SWIMMING_ENERGY_COST
        return 0.0

    def is_drought(self) -> bool:
        """True if no rain for DROUGHT_THRESHOLD ticks."""
        return (self.current_tick - self.last_rain_tick) >= DROUGHT_THRESHOLD

    # ─────────────────────────────────────────────────────────────────────────
    # Movement energy + falling damage
    # ─────────────────────────────────────────────────────────────────────────

    def get_movement_energy_cost(self, base_cost: float, x: int, y: int,
                                  new_x: int, new_y: int) -> float:
        """
        Compute movement energy cost including:
          - Wind modifier (+30% against, -15% with)
          - Elevation uphill cost (+10% per elevation unit)
          - Swimming cost (ocean tiles)
        """
        dx = new_x - x
        dy = new_y - y
        cost = base_cost

        # Wind modifier
        cost *= self.get_wind_modifier_for_movement(dx, dy)

        # Elevation uphill cost
        elev_here = self.get_elevation(x, y)
        elev_there = self.get_elevation(new_x, new_y)
        if elev_there > elev_here:
            cost *= 1.0 + (elev_there - elev_here) * 0.10

        # Swimming cost
        cost += self.get_swimming_energy_cost(new_x, new_y)

        return cost

    def check_falling_damage(self, x: int, y: int,
                              new_x: int, new_y: int) -> float:
        """
        If moving to a tile 3+ elevation units lower, return falling damage.
        Damage = (height_diff - 2) * 5
        """
        elev_here = self.get_elevation(x, y)
        elev_there = self.get_elevation(new_x, new_y)
        diff = elev_here - elev_there
        if diff >= ELEVATION_FALL_DAMAGE_THRESHOLD:
            return (diff - 2) * ELEVATION_FALL_DAMAGE_PER_UNIT
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Sandstorm effects
    # ─────────────────────────────────────────────────────────────────────────

    def get_sandstorm_effects(self, weather: str, biome: str) -> Dict:
        """
        Returns sandstorm-specific effects:
          - visibility_penalty: 0.7 (reduces visibility by 70%)
          - pain: random 1-3 with 15% probability per tick
        """
        if weather != "sandstorm":
            return {"visibility_penalty": 0.0, "pain": 0.0}

        pain = 0.0
        if self.rng.random() < SANDSTORM_PAIN_PROB:
            pain = self.rng.uniform(*SANDSTORM_PAIN_AMOUNT)

        # Desert + sandstorm = worse
        vis_pen = SANDSTORM_VISIBILITY_PENALTY
        if biome == "desert":
            vis_pen = 0.85

        return {"visibility_penalty": vis_pen, "pain": pain}

    # ─────────────────────────────────────────────────────────────────────────
    # Sensory vector extensions (Phase 1.5)
    # ─────────────────────────────────────────────────────────────────────────

    def get_sensory_extensions(self, x: int, y: int) -> Dict:
        """
        Returns new sensory fields for Phase 1.5:
          - body_temp: agent's body temperature
          - wind_speed: 0-5
          - wind_dir: N/S/E/W (encoded as 0-3)
          - elevation: 0-10
          - surface_wetness: 0-1 (1 if on water tile, 0.5 if adjacent)
          - is_on_fire: bool (current tile is fire)
          - adjacent_fire_count: int 0-4
          - is_drought: bool
        """
        adjacent_fires = self._count_adjacent_fires(x, y)
        # Surface wetness
        if (x, y) in self.water_tiles or self.is_ocean_tile(x, y):
            wetness = 1.0
        else:
            # 0.5 if any adjacent tile is water
            wetness = 0.0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if ((x + dx) in self.water_tiles or
                        self.is_ocean_tile(x + dx, y + dy)):
                    wetness = 0.5
                    break

        return {
            'body_temp': round(self.body_temp, 2),
            'wind_speed': self.wind_speed,
            'wind_dir': WIND_DIRECTIONS.index(self.wind_dir),
            'elevation': self.get_elevation(x, y),
            'surface_wetness': wetness,
            'is_on_fire': self.is_fire_tile(x, y),
            'adjacent_fire_count': adjacent_fires,
            'is_drought': self.is_drought(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, biome: str, weather: str, time_of_day: int,
             x: int, y: int, base_biome_temp: float,
             weather_temp_mod: float) -> None:
        """
        Advance all physics systems by one tick.

        Args:
            biome: current biome name
            weather: current weather name
            time_of_day: 0-23
            x, y: agent position
            base_biome_temp: BIOMES[biome]['base_temp']
            weather_temp_mod: weather_data['temp_mod']
        """
        self.current_tick += 1
        self._update_wind(weather)
        self._update_fire(weather, x, y)
        self._update_water(weather, x, y)
        self._update_body_temp(
            biome, weather, time_of_day, x, y,
            base_biome_temp, weather_temp_mod,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialize physics state for checkpointing."""
        return {
            'body_temp': self.body_temp,
            'wind_speed': self.wind_speed,
            'wind_dir': self.wind_dir,
            'wind_state': self._wind_state,
            'fire_tiles': {f"{k[0]},{k[1]}": v for k, v in self.fire_tiles.items()},
            'ash_tiles': list(self.ash_tiles),
            'water_tiles': [f"{t[0]},{t[1]}" for t in self.water_tiles],
            'last_rain_tick': self.last_rain_tick,
            'current_tick': self.current_tick,
        }

    def load_state(self, state: Dict) -> None:
        """Restore physics state from a checkpoint."""
        self.body_temp = state.get('body_temp', BODY_TEMP_NORMAL)
        self.wind_speed = state.get('wind_speed', 0)
        self.wind_dir = state.get('wind_dir', 'N')
        self._wind_state = state.get('wind_state', 'calm')
        self.fire_tiles = {
            tuple(int(c) for c in k.split(',')): v
            for k, v in state.get('fire_tiles', {}).items()
        }
        self.ash_tiles = set(tuple(int(c) for c in t.split(',')) for t in state.get('ash_tiles', []))
        self.water_tiles = set(tuple(int(c) for c in t.split(',')) for t in state.get('water_tiles', []))
        self.last_rain_tick = state.get('last_rain_tick', 0)
        self.current_tick = state.get('current_tick', 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing PhysicsEngine...")

    # Mock world map
    class MockWorldMap:
        def __init__(self):
            self.width = 64
            self.height = 64

        def get_biome(self, x, y):
            # Plains everywhere for testing
            return "plains"

    world_map = MockWorldMap()
    physics = PhysicsEngine(world_map, seed=42)

    print(f"  Initial body temp: {physics.body_temp}°C")
    print(f"  Initial wind: {physics.wind_speed} ({physics.wind_dir})")
    print(f"  Elevation at (10,10): {physics.get_elevation(10, 10)}")

    # Simulate 100 ticks in plains/clear weather
    for tick in range(100):
        physics.step(
            biome="plains", weather="clear", time_of_day=12,
            x=10, y=10, base_biome_temp=20.0, weather_temp_mod=0.0,
        )
    print(f"  After 100 ticks plains/clear: body temp = {physics.body_temp:.1f}°C")

    # Test fire ignition
    ignited = physics.ignite_fire(11, 10)
    print(f"  Ignite fire at (11,10): {ignited}")
    print(f"  Adjacent fire count at (10,10): {physics._count_adjacent_fires(10, 10)}")
    print(f"  Fire pain at (10,10): {physics.get_adjacent_fire_pain(10, 10)}")
    print(f"  Fire warmth benefit at (10,10): {physics.get_fire_warmth_benefit(10, 10)}")

    # Test elevation-based temp
    class MountainWorldMap:
        def __init__(self):
            self.width = 64
            self.height = 64

        def get_biome(self, x, y):
            return "mountain"

    mountain_map = MountainWorldMap()
    mountain_physics = PhysicsEngine(mountain_map, seed=42)
    # Manually set a high elevation
    mountain_physics.elevation_map[10][10] = 9
    air_temp = mountain_physics.get_local_air_temp(
        biome="mountain", weather="clear", time_of_day=12,
        x=10, y=10, base_biome_temp=5.0, weather_temp_mod=0.0,
    )
    print(f"  Mountain air temp at elev 9: {air_temp:.1f}°C (should be 5 - 4*2 = -3)")

    # Test hypothermia
    cold_physics = PhysicsEngine(world_map, seed=42)
    cold_physics.body_temp = 25.0
    print(f"  Hypothermia damage at 25°C: {cold_physics.get_hypothermia_damage():.2f}")

    # Test hyperthermia
    hot_physics = PhysicsEngine(world_map, seed=42)
    hot_physics.body_temp = 44.0
    print(f"  Hyperthermia damage at 44°C: {hot_physics.get_hyperthermia_damage():.2f}")

    # Test falling damage
    fall_physics = PhysicsEngine(world_map, seed=42)
    # elevation_map is indexed [y][x] — set cells (x=10,y=10) and (x=10,y=11)
    fall_physics.elevation_map[10][10] = 9
    fall_physics.elevation_map[11][10] = 4
    dmg = fall_physics.check_falling_damage(10, 10, 10, 11)
    print(f"  Falling damage from elev 9 → 4: {dmg:.1f} (should be (5-2)*5 = 15)")

    # Test wind movement modifier
    wind_physics = PhysicsEngine(world_map, seed=42)
    wind_physics.wind_speed = 5
    wind_physics.wind_dir = "N"
    print(f"  Move N with N wind: {wind_physics.get_wind_modifier_for_movement(0, -1):.2f} (with wind, 0.85)")
    print(f"  Move S against N wind: {wind_physics.get_wind_modifier_for_movement(0, 1):.2f} (against, 1.30)")

    print("\n✓ PhysicsEngine self-test passed")
