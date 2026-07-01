"""
Nafs AI — Chemistry Engine (Phase 2)
====================================

Implements MD Phase 2: substances have properties. Adam does NOT know these
properties — he discovers them through his body's reaction.

The chemistry engine runs silently; only the consequences appear in the
sensory vector.

Covers:
  2.1 Food Composition
      - 7+ food types with hidden properties: calories, protein, fat,
        water_content, toxicity, decay_rate
      - Body processes food over time (not instant nutrition)
      - Stomach contents tracked as rolling buffer (last 5 eaten items)

  2.2 Toxicity & Illness
      - Toxicity causes DELAYED sensory pain signal (100 ticks after eating)
      - Adam cannot know which food caused it — must learn by tracking
        what he ate
      - Illness state: vomiting action, health drain, reduced energy
      - Recovery: illness clears after 500 ticks or with medicine plant
      - Medicine plant: reduces illness intensity (discovered by eating
        when sick)

  2.3 Fire as Chemistry
      - Raw food near fire tile for 50 ticks → becomes cooked (toxicity removed)
      - Adam has no COOK action — discovers cooking by accident (food near fire)
      - Cooked food sensory signal: different smell indicator in sensory vector
      - This is one of the biggest emergent discoveries possible

  2.4 Water Quality
      - River/rain water: safe to drink
      - Swamp water: 0.4 toxicity — causes illness over time
      - Ocean water: immediate illness (salt poisoning)
      - Adam cannot see water quality — only experiences consequence

  2.5 Fermentation & Decay
      - Food stored on a tile decays over 500 ticks
      - Decayed food: higher toxicity than fresh
      - No explicit inventory system — food is only 'available' at tile
        Adam is on
      - Sensory vector additions: stomach_contents[5], toxicity_signal,
        illness_level, smell_intensity

Design constraints:
  - Does NOT modify base rewards in world_sim.py
  - Designed as a standalone module that WorldSim instantiates and calls
  - Chemistry is INVISIBLE — Adam only sees the consequences in his body
    (pain, illness_level, smell_intensity)

Usage:
    from chemistry import ChemistryEngine
    chemistry = ChemistryEngine(physics_engine)
    chemistry.spawn_food_at(biome, x, y)  # called during world gen
    chemistry.eat(food_type, tick)        # called when agent EATs
    chemistry.drink(water_source, tick)   # called when agent DRINKs
    chemistry.update(tick, x, y)          # called every tick
    chemistry.get_sensory_extensions()
"""

import random
from typing import Dict, List, Optional, Tuple, Set
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

STOMACH_BUFFER_SIZE = 5              # last 5 eaten items
TOXICITY_DELAY_TICKS = 100           # pain appears 100 ticks after eating toxic food
ILLNESS_DURATION = 500               # illness clears after 500 ticks
ILLNESS_HEALTH_DRAIN = 0.3           # health drain per tick when ill
ILLNESS_ENERGY_PENALTY = 0.5         # extra energy drain per tick when ill
COOK_TIME = 50                       # ticks near fire to cook
FOOD_DECAY_TIME = 500                # ticks before food fully decays
DECAY_TOXICITY_MULT = 2.0            # decayed food is 2x more toxic

# Biome → which food types can spawn there
BIOME_FOOD_SPAWNS = {
    "forest":  ["red_berry", "blue_berry", "mushroom", "roots"],
    "jungle":  ["red_berry", "blue_berry", "mushroom", "roots"],
    "plains":  ["roots", "blue_berry"],
    "swamp":   ["mushroom", "roots", "medicine_plant"],
    "cave":    ["mushroom", "roots", "medicine_plant"],
    "mountain":["roots"],
    "tundra":  ["roots"],
    "desert":  [],  # very rare food
    "ocean":   ["fish"],
    "volcano": [],
}

# Food types and their HIDDEN properties.
# Adam does NOT know these — he discovers them through body reactions.
FOOD_TYPES = {
    "red_berry": {
        "calories": 20, "protein": 0,  "fat": 0,  "water_content": 5,
        "toxicity": 0.8, "decay_rate": 0.002,
        "cooked_to": None,  # berries don't cook
        "smell": "sweet",
    },
    "blue_berry": {
        "calories": 15, "protein": 0,  "fat": 0,  "water_content": 8,
        "toxicity": 0.0, "decay_rate": 0.003,
        "cooked_to": None,
        "smell": "fresh",
    },
    "mushroom": {
        "calories": 40, "protein": 5,  "fat": 1,  "water_content": 10,
        "toxicity": 0.3, "decay_rate": 0.005,
        "cooked_to": None,  # mushrooms don't cook
        "smell": "earthy",
    },
    "fish": {
        "calories": 60, "protein": 30, "fat": 5,  "water_content": 20,
        "toxicity": 0.0, "decay_rate": 0.004,
        "cooked_to": "cooked_fish",
        "smell": "fishy",
    },
    "roots": {
        "calories": 25, "protein": 3,  "fat": 0,  "water_content": 15,
        "toxicity": 0.0, "decay_rate": 0.001,
        "cooked_to": None,
        "smell": "dirt",
    },
    "raw_meat": {
        "calories": 80, "protein": 40, "fat": 15, "water_content": 5,
        "toxicity": 0.2, "decay_rate": 0.004,
        "cooked_to": "cooked_meat",
        "smell": "iron",
    },
    "cooked_meat": {
        "calories": 80, "protein": 40, "fat": 15, "water_content": 0,
        "toxicity": 0.0, "decay_rate": 0.002,
        "cooked_to": None,
        "smell": "roast",
    },
    "cooked_fish": {
        "calories": 60, "protein": 30, "fat": 5,  "water_content": 0,
        "toxicity": 0.0, "decay_rate": 0.002,
        "cooked_to": None,
        "smell": "roast",
    },
    "medicine_plant": {
        "calories": 5, "protein": 0, "fat": 0, "water_content": 10,
        "toxicity": 0.0, "decay_rate": 0.001,
        "cooked_to": None,
        "heals_illness": True,
        "smell": "bitter",
    },
}

# Water quality by source type
WATER_QUALITY = {
    "river":  {"toxicity": 0.0, "hydration": 30, "immediate_illness": False},
    "rain":   {"toxicity": 0.0, "hydration": 20, "immediate_illness": False},
    "swamp":  {"toxicity": 0.4, "hydration": 15, "immediate_illness": False},
    "ocean":  {"toxicity": 1.0, "hydration": -10, "immediate_illness": True},
}


# ═══════════════════════════════════════════════════════════════════════════════
# ChemistryEngine
# ═══════════════════════════════════════════════════════════════════════════════

class ChemistryEngine:
    """
    Master chemistry engine for the Nafs AI world.

    Holds all chemistry state:
      - stomach: rolling buffer of last 5 (food_type, tick_eaten, toxicity)
      - pending_toxicity: list of (tick_due, amount, food_name) waiting to fire
      - illness_level: 0.0 (healthy) to 1.0 (severe)
      - illness_tick: when illness started
      - cooking_food: { (x,y): {food_type, ticks_cooked, cooked} }
      - food_on_tiles: { (x,y): {food_type, freshness, tick_spawned} }

    All chemistry is INVISIBLE to the agent — only consequences (pain, illness,
    smell) appear in sensory vector.
    """

    def __init__(self, physics_engine=None, seed: Optional[int] = None):
        self.physics = physics_engine
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Stomach: rolling buffer of last 5 items eaten
        self.stomach: deque = deque(maxlen=STOMACH_BUFFER_SIZE)

        # Pending toxicity events: [(tick_due, amount, food_name)]
        self.pending_toxicity: List[Tuple[int, float, str]] = []

        # Illness state
        self.illness_level = 0.0       # 0 = healthy, 1 = severe
        self.illness_tick = 0          # tick when illness started
        self.illness_cause = ""        # what food/water caused it

        # Food on tiles: { (x,y): {food_type, freshness, tick_spawned, decayed} }
        self.food_on_tiles: Dict[Tuple[int, int], Dict] = {}

        # Cooking tracker: { (x,y): {food_type, ticks_cooked, cooked} }
        self.cooking_food: Dict[Tuple[int, int], Dict] = {}

        # Water sources (chemistry tracks quality)
        # Ocean is determined by biome; swamp water by biome; rain/temp water
        # by physics.water_tiles
        self.current_tick = 0

        # Vomit action state (Phase 2.2)
        self.just_vomited = False
        self.vomit_cooldown = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Food spawning
    # ─────────────────────────────────────────────────────────────────────────

    def spawn_food_at(self, biome: str, x: int, y: int,
                       force_type: Optional[str] = None) -> Optional[str]:
        """
        Spawn a food item at (x, y) appropriate to the biome.
        Returns the food_type spawned, or None if no food spawned.
        """
        if force_type:
            food_type = force_type
        else:
            choices = BIOME_FOOD_SPAWNS.get(biome, [])
            if not choices:
                return None
            food_type = self.rng.choice(choices)

        if food_type not in FOOD_TYPES:
            return None

        self.food_on_tiles[(x, y)] = {
            "food_type": food_type,
            "freshness": 1.0,  # 1.0 = fresh, 0.0 = fully decayed
            "tick_spawned": self.current_tick,
            "decayed": False,
        }
        return food_type

    def get_food_at(self, x: int, y: int) -> Optional[Dict]:
        """Get food info at (x, y), or None if no food."""
        return self.food_on_tiles.get((x, y))

    def remove_food_at(self, x: int, y: int) -> None:
        """Remove food from tile (after eaten)."""
        self.food_on_tiles.pop((x, y), None)

    def has_food_at(self, x: int, y: int) -> bool:
        return (x, y) in self.food_on_tiles

    # ─────────────────────────────────────────────────────────────────────────
    # Eating
    # ─────────────────────────────────────────────────────────────────────────

    def eat(self, food_type: str, tick: int) -> Dict:
        """
        Agent eats food. Returns dict with nutrition info and effects.

        If the food is toxic, schedules a delayed pain event (100 ticks later).
        If the food heals illness (medicine_plant), reduces illness_level.

        Returns:
          {
            "calories": float,
            "protein": float,
            "fat": float,
            "water_content": float,
            "toxicity": float,        # raw food toxicity (delayed effect)
            "heals_illness": bool,
            "smell": str,
            "cooked": bool,
          }
        """
        food = FOOD_TYPES.get(food_type)
        if food is None:
            return {}

        self.current_tick = tick

        # Add to stomach buffer
        self.stomach.append({
            "food_type": food_type,
            "tick_eaten": tick,
            "toxicity": food["toxicity"],
        })

        # Schedule delayed toxicity pain (100 ticks later)
        if food["toxicity"] > 0:
            self.pending_toxicity.append(
                (tick + TOXICITY_DELAY_TICKS, food["toxicity"], food_type)
            )

        # Medicine plant heals illness
        heals_illness = food.get("heals_illness", False)
        if heals_illness and self.illness_level > 0:
            self.illness_level = max(0.0, self.illness_level - 0.5)
            if self.illness_level == 0.0:
                self.illness_tick = 0
                self.illness_cause = ""

        return {
            "calories": food["calories"],
            "protein": food["protein"],
            "fat": food["fat"],
            "water_content": food["water_content"],
            "toxicity": food["toxicity"],
            "heals_illness": heals_illness,
            "smell": food.get("smell", ""),
            "cooked": food_type.startswith("cooked_"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Drinking
    # ─────────────────────────────────────────────────────────────────────────

    def drink(self, water_source: str, tick: int) -> Dict:
        """
        Agent drinks water from a source.

        Args:
            water_source: "river", "rain", "swamp", or "ocean"
            tick: current simulation tick

        Returns:
            {
              "hydration": float,
              "toxicity": float,
              "immediate_illness": bool,
            }
        """
        quality = WATER_QUALITY.get(water_source, WATER_QUALITY["river"])
        self.current_tick = tick

        # Schedule delayed toxicity (for swamp water)
        if quality["toxicity"] > 0 and not quality["immediate_illness"]:
            self.pending_toxicity.append(
                (tick + TOXICITY_DELAY_TICKS, quality["toxicity"], f"water:{water_source}")
            )

        # Immediate illness (ocean — salt poisoning)
        if quality["immediate_illness"]:
            self.illness_level = min(1.0, self.illness_level + 0.6)
            self.illness_tick = tick
            self.illness_cause = f"water:{water_source}"

        return {
            "hydration": quality["hydration"],
            "toxicity": quality["toxicity"],
            "immediate_illness": quality["immediate_illness"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Cooking (Phase 2.3 — fire as chemistry)
    # ─────────────────────────────────────────────────────────────────────────

    def update_cooking(self, x: int, y: int) -> None:
        """
        Check if food on a tile near fire should cook.
        Raw food near fire for 50 ticks → becomes cooked (toxicity removed).
        """
        # Check all food tiles near fire
        if self.physics is None:
            return

        # For each food tile, check if adjacent to fire
        to_cook = []
        for (fx, fy), food_info in self.food_on_tiles.items():
            food_type = food_info["food_type"]
            food_def = FOOD_TYPES.get(food_type, {})
            if food_def.get("cooked_to") is None:
                continue  # can't be cooked

            # Check 4-neighborhood for fire tiles
            near_fire = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if self.physics.is_fire_tile(fx + dx, fy + dy):
                    near_fire = True
                    break

            if near_fire:
                # Track cooking progress
                key = (fx, fy)
                if key not in self.cooking_food:
                    self.cooking_food[key] = {
                        "food_type": food_type,
                        "ticks_cooked": 0,
                        "cooked": False,
                    }
                self.cooking_food[key]["ticks_cooked"] += 1

                if self.cooking_food[key]["ticks_cooked"] >= COOK_TIME:
                    # Transform the food to its cooked version
                    cooked_type = food_def["cooked_to"]
                    food_info["food_type"] = cooked_type
                    food_info["freshness"] = 1.0  # cooked food is "fresh"
                    food_info["decayed"] = False
                    self.cooking_food[key]["cooked"] = True
                    to_cook.append((fx, fy, cooked_type))

        # Clean up completed cooking entries
        for (fx, fy, _) in to_cook:
            self.cooking_food.pop((fx, fy), None)

    def is_cooking_at(self, x: int, y: int) -> bool:
        """Is food currently cooking at this tile?"""
        info = self.cooking_food.get((x, y))
        return info is not None and not info.get("cooked", False)

    def get_cook_progress(self, x: int, y: int) -> float:
        """Returns cooking progress as 0.0 to 1.0."""
        info = self.cooking_food.get((x, y))
        if not info:
            return 0.0
        return min(1.0, info["ticks_cooked"] / COOK_TIME)

    # ─────────────────────────────────────────────────────────────────────────
    # Food decay (Phase 2.5)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_decay(self) -> None:
        """Update freshness of all food on tiles. Decayed food has higher toxicity."""
        to_remove = []
        for (x, y), food_info in self.food_on_tiles.items():
            food_type = food_info["food_type"]
            food_def = FOOD_TYPES.get(food_type, {})
            decay_rate = food_def.get("decay_rate", 0.002)

            food_info["freshness"] -= decay_rate
            if food_info["freshness"] <= 0:
                food_info["freshness"] = 0
                food_info["decayed"] = True
                # Decayed food stays on tile but is now toxic
                # (toxicity applied at eat time)

            # Very old food disappears
            age = self.current_tick - food_info["tick_spawned"]
            if age > FOOD_DECAY_TIME * 2:
                to_remove.append((x, y))

        for tile in to_remove:
            self.food_on_tiles.pop(tile, None)
            self.cooking_food.pop(tile, None)

    def get_effective_toxicity(self, food_type: str, freshness: float,
                                 decayed: bool) -> float:
        """Get effective toxicity of a food item (decayed food is more toxic)."""
        base_tox = FOOD_TYPES.get(food_type, {}).get("toxicity", 0.0)
        if decayed:
            return min(1.0, base_tox * DECAY_TOXICITY_MULT)
        # Slight toxicity increase as freshness drops (but not decayed yet)
        if freshness < 0.5:
            return base_tox * (1.0 + (1.0 - freshness))
        return base_tox

    # ─────────────────────────────────────────────────────────────────────────
    # Pending toxicity & illness update
    # ─────────────────────────────────────────────────────────────────────────

    def _update_pending_toxicity(self, tick: int) -> List[Dict]:
        """
        Fire any pending toxicity events that are due.
        Returns list of pain events fired this tick.
        """
        fired = []
        still_pending = []
        for entry in self.pending_toxicity:
            tick_due, amount, food_name = entry
            if tick >= tick_due:
                # Fire pain event
                fired.append({
                    "tick": tick,
                    "amount": amount,
                    "food_name": food_name,
                    "pain": amount * 10.0,  # scale to 0-10 pain range
                })
                # Toxic food also triggers illness (gradual)
                if amount > 0.3:
                    self.illness_level = min(1.0, self.illness_level + amount * 0.5)
                    if self.illness_tick == 0:
                        self.illness_tick = tick
                    if not self.illness_cause:
                        self.illness_cause = food_name
            else:
                still_pending.append(entry)
        self.pending_toxicity = still_pending
        return fired

    def _update_illness(self, tick: int) -> Dict:
        """
        Update illness state. Returns dict with damage to apply this tick.

        Illness clears after ILLNESS_DURATION ticks.
        """
        if self.illness_level <= 0:
            return {"health_drain": 0.0, "energy_drain": 0.0, "recovered": False}

        # Check if illness should clear
        if tick - self.illness_tick >= ILLNESS_DURATION:
            self.illness_level = 0.0
            self.illness_tick = 0
            self.illness_cause = ""
            return {"health_drain": 0.0, "energy_drain": 0.0, "recovered": True}

        # Apply illness effects
        drain = ILLNESS_HEALTH_DRAIN * self.illness_level
        energy = ILLNESS_ENERGY_PENALTY * self.illness_level
        return {
            "health_drain": drain,
            "energy_drain": energy,
            "recovered": False,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Vomit action (Phase 2.2)
    # ─────────────────────────────────────────────────────────────────────────

    def vomit(self, tick: int) -> Dict:
        """
        Agent vomits — empties stomach, reduces illness slightly.
        Cooldown of 50 ticks between vomits.
        """
        if self.vomit_cooldown > 0:
            return {"success": False, "reason": "cooldown"}

        self.vomit_cooldown = 50
        self.just_vomited = True

        # Empty stomach
        emptied = list(self.stomach)
        self.stomach.clear()

        # Reduce illness by 0.2 (vomiting helps a bit)
        if self.illness_level > 0:
            self.illness_level = max(0.0, self.illness_level - 0.2)

        # Remove some pending toxicity (the most recent ones)
        if self.pending_toxicity:
            self.pending_toxicity = self.pending_toxicity[:-1]  # drop last

        return {
            "success": True,
            "emptied_count": len(emptied),
            "illness_reduction": 0.2 if self.illness_level > 0 else 0.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main update
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, tick: int, x: int, y: int) -> Dict:
        """
        Per-tick chemistry update.

        Returns dict with effects to apply this tick:
          - pain_events: list of delayed pain events fired
          - illness: dict with health_drain, energy_drain, recovered
        """
        self.current_tick = tick

        if self.vomit_cooldown > 0:
            self.vomit_cooldown -= 1

        pain_events = self._update_pending_toxicity(tick)
        illness = self._update_illness(tick)
        self._update_decay()
        self.update_cooking(x, y)

        return {
            "pain_events": pain_events,
            "illness": illness,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Sensory extensions (Phase 2.5)
    # ─────────────────────────────────────────────────────────────────────────

    def get_sensory_extensions(self, x: int, y: int) -> Dict:
        """
        Returns chemistry-related sensory fields:
          - stomach_contents: list of last 5 food names eaten
          - toxicity_signal: 0-1, current toxic load in stomach
          - illness_level: 0-1
          - smell_intensity: 0-1, intensity of nearby food smell
          - food_here: bool, is there food on current tile?
          - food_cooking: bool, is food cooking nearby?
        """
        # Stomach contents (last 5)
        stomach_contents = [s["food_type"] for s in self.stomach]

        # Toxicity signal: sum of toxicities in stomach
        toxicity_signal = sum(s["toxicity"] for s in self.stomach) / max(1, len(self.stomach))

        # Smell: is there food nearby?
        smell_intensity = 0.0
        food_here = self.has_food_at(x, y)
        if food_here:
            smell_intensity = 1.0
        else:
            # Check adjacent tiles for food
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if self.has_food_at(x + dx, y + dy):
                    smell_intensity = max(smell_intensity, 0.5)
                    break

        # Cooking smell
        food_cooking = self.is_cooking_at(x, y)

        return {
            'stomach_contents': stomach_contents,
            'toxicity_signal': round(toxicity_signal, 3),
            'illness_level': round(self.illness_level, 3),
            'smell_intensity': round(smell_intensity, 3),
            'food_here': food_here,
            'food_cooking': food_cooking,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def is_ill(self) -> bool:
        return self.illness_level > 0

    def get_stomach(self) -> List[Dict]:
        """Return list of stomach contents (newest last)."""
        return list(self.stomach)

    def get_pending_toxicity_count(self) -> int:
        """How many pending toxicity events are scheduled."""
        return len(self.pending_toxicity)

    def get_water_source_for(self, biome: str, x: int, y: int,
                              physics_water_tiles: set) -> str:
        """
        Determine the water source type for a tile.
        Used when agent DRINKs to look up water quality.
        """
        if biome == "ocean":
            return "ocean"
        if biome == "swamp":
            return "swamp"
        # If on a physics temp water tile, it's rainwater
        if (x, y) in physics_water_tiles:
            return "rain"
        # Otherwise, river (default safe water)
        return "river"

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialize chemistry state for checkpointing."""
        return {
            'stomach': list(self.stomach),
            'pending_toxicity': self.pending_toxicity,
            'illness_level': self.illness_level,
            'illness_tick': self.illness_tick,
            'illness_cause': self.illness_cause,
            'food_on_tiles': {
                f"{k[0]},{k[1]}": v for k, v in self.food_on_tiles.items()
            },
            'cooking_food': {
                f"{k[0]},{k[1]}": v for k, v in self.cooking_food.items()
            },
            'current_tick': self.current_tick,
            'vomit_cooldown': self.vomit_cooldown,
        }

    def load_state(self, state: Dict) -> None:
        """Restore chemistry state from a checkpoint."""
        self.stomach = deque(state.get('stomach', []), maxlen=STOMACH_BUFFER_SIZE)
        self.pending_toxicity = state.get('pending_toxicity', [])
        self.illness_level = state.get('illness_level', 0.0)
        self.illness_tick = state.get('illness_tick', 0)
        self.illness_cause = state.get('illness_cause', '')
        self.food_on_tiles = {
            tuple(int(c) for c in k.split(',')): v
            for k, v in state.get('food_on_tiles', {}).items()
        }
        self.cooking_food = {
            tuple(int(c) for c in k.split(',')): v
            for k, v in state.get('cooking_food', {}).items()
        }
        self.current_tick = state.get('current_tick', 0)
        self.vomit_cooldown = state.get('vomit_cooldown', 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing ChemistryEngine...")

    # Mock physics engine
    class MockPhysics:
        def __init__(self):
            self.fire_tiles = set()

        def is_fire_tile(self, x, y):
            return (x, y) in self.fire_tiles

    physics = MockPhysics()
    chem = ChemistryEngine(physics, seed=42)

    # Test food spawning
    food = chem.spawn_food_at("forest", 5, 5)
    print(f"  Spawned food in forest: {food}")
    assert food in ("red_berry", "blue_berry", "mushroom", "roots")

    # Test eating safe food
    chem2 = ChemistryEngine(physics, seed=42)
    result = chem2.eat("blue_berry", tick=10)
    print(f"  Eat blue_berry: calories={result['calories']}, toxicity={result['toxicity']}")
    assert result["calories"] == 15
    assert result["toxicity"] == 0.0

    # Test eating toxic food (delayed pain)
    chem3 = ChemistryEngine(physics, seed=42)
    chem3.eat("red_berry", tick=10)
    # No pain immediately
    update = chem3.update(tick=20, x=0, y=0)
    assert len(update["pain_events"]) == 0
    # Pain 100 ticks later
    update = chem3.update(tick=110, x=0, y=0)
    print(f"  Pain events at t110 (after eating red_berry at t10): {len(update['pain_events'])}")
    assert len(update["pain_events"]) == 1
    assert update["pain_events"][0]["pain"] > 0

    # Test illness from toxic food
    chem4 = ChemistryEngine(physics, seed=42)
    chem4.eat("red_berry", tick=10)
    # Wait for toxicity to fire
    chem4.update(tick=110, x=0, y=0)
    print(f"  Illness level after toxic food: {chem4.illness_level:.2f}")
    assert chem4.illness_level > 0

    # Test medicine plant heals illness
    chem5 = ChemistryEngine(physics, seed=42)
    chem5.illness_level = 0.8
    chem5.illness_tick = 100
    result = chem5.eat("medicine_plant", tick=110)
    print(f"  After medicine_plant: illness={chem5.illness_level:.2f}")
    assert chem5.illness_level < 0.8
    assert result["heals_illness"] is True

    # Test ocean water causes immediate illness
    chem6 = ChemistryEngine(physics, seed=42)
    result = chem6.drink("ocean", tick=10)
    print(f"  Drink ocean: hydration={result['hydration']}, immediate_illness={result['immediate_illness']}")
    assert result["immediate_illness"] is True
    assert chem6.illness_level > 0

    # Test swamp water causes delayed illness
    chem7 = ChemistryEngine(physics, seed=42)
    result = chem7.drink("swamp", tick=10)
    assert not result["immediate_illness"]
    chem7.update(tick=110, x=0, y=0)
    print(f"  Illness after swamp water (100t later): {chem7.illness_level:.2f}")
    assert chem7.illness_level > 0

    # Test cooking: raw_meat near fire → cooked_meat
    chem8 = ChemistryEngine(physics, seed=42)
    chem8.spawn_food_at("plains", 5, 5, force_type="raw_meat")
    physics.fire_tiles.add((6, 5))  # fire adjacent to food
    # Cook for 50 ticks
    for tick in range(50):
        chem8.update(tick=tick, x=5, y=5)
    food_at = chem8.get_food_at(5, 5)
    print(f"  After 50 ticks near fire: raw_meat → {food_at['food_type']}")
    assert food_at["food_type"] == "cooked_meat"
    assert FOOD_TYPES["cooked_meat"]["toxicity"] == 0.0

    # Test decay: food gets more toxic over time
    class MockPhysics2:
        def __init__(self):
            self.fire_tiles = set()
        def is_fire_tile(self, x, y):
            return (x, y) in self.fire_tiles
    chem9 = ChemistryEngine(MockPhysics2(), seed=42)
    chem9.spawn_food_at("plains", 5, 5, force_type="raw_meat")
    food_info = chem9.get_food_at(5, 5)
    initial_tox = chem9.get_effective_toxicity(
        food_info["food_type"], food_info["freshness"], food_info["decayed"]
    )
    # Decay for 1000 ticks
    for tick in range(1000):
        chem9.update(tick=tick, x=0, y=0)
    food_info = chem9.get_food_at(5, 5)
    if food_info:  # might have disappeared
        decayed_tox = chem9.get_effective_toxicity(
            food_info["food_type"], food_info["freshness"], food_info["decayed"]
        )
        print(f"  Fresh toxicity: {initial_tox:.2f}, Decayed: {decayed_tox:.2f}")
        assert decayed_tox >= initial_tox

    # Test vomit
    chem10 = ChemistryEngine(physics, seed=42)
    chem10.eat("red_berry", tick=10)
    chem10.eat("blue_berry", tick=20)
    assert len(chem10.stomach) == 2
    result = chem10.vomit(tick=25)
    print(f"  Vomit: success={result['success']}, emptied={result['emptied_count']}")
    assert result["success"]
    assert result["emptied_count"] == 2
    assert len(chem10.stomach) == 0

    # Test sensory extensions
    chem11 = ChemistryEngine(physics, seed=42)
    chem11.eat("blue_berry", tick=10)
    chem11.spawn_food_at("forest", 5, 5, force_type="mushroom")
    ext = chem11.get_sensory_extensions(5, 5)
    print(f"  Sensory extensions: {ext}")
    assert "stomach_contents" in ext
    assert "toxicity_signal" in ext
    assert "illness_level" in ext
    assert "smell_intensity" in ext
    assert ext["food_here"] is True

    print("\n✓ ChemistryEngine self-test passed")
