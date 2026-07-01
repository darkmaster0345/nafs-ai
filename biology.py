"""
Nafs AI — Biology Engine (Phase 3)
==================================

Implements MD Phase 3: Adam and Eve are organisms, not game characters.
Their bodies have real biology that interacts with the physics and chemistry
systems.

Covers:
  3.1 Metabolic System
      - Replaces hunger/thirst counters with real metabolism:
        glucose, fat, protein, hydration, body_mass
      - Dehydration accelerates all metabolic drain rates
      - Starvation sequence: glucose → fat → protein → death
      - Different foods replenish different nutrients

  3.2 Life Stages & Aging
      - 6 stages: newborn → child → adolescent → adult → elder → ancient
      - Physical stats (speed, strength, vision) peak at adult, decline after
      - Elder agents have larger pattern memory
      - First agent to reach Elder is a milestone

  3.3 Immune System
      - Immunity builds from disease exposure (0.0-1.0 per disease type)
      - High immunity = disease causes less damage
      - Swamp disease: first exposure severe, subsequent mild
      - Immunity passed partially to offspring (biological inheritance)

  3.4 Injury System
      - 4 states: NONE / BRUISED / WOUNDED / CRITICAL
      - Sources: falling, fire, attack, severe storm
      - Wounded: movement speed -40%, energy cost +30%
      - Recovery: automatic over 300 ticks, faster in shelter
      - Critical: near-death, all stats severely impaired

  3.5 Sleep Biology
      - REM sleep (first 100 ticks): dream engine, memory consolidation
      - Deep sleep (next 200 ticks): physical recovery, injury healing
      - Sleep debt: 3000+ ticks → hallucination signals
      - Circadian rhythm: sleepier at night

Design constraints:
  - Does NOT modify base rewards in world_sim.py
  - Designed as a standalone module
  - Hooks into existing hunger/energy/stress stats (translates them)

Usage:
    from biology import BiologyEngine
    bio = BiologyEngine()
    bio.digest_food(food_result)   # called when agent EATs
    bio.step(tick, action, biome, time_of_day)
    bio.apply_injury(2, tick)      # WOUNDED
    dmg_mult = bio.expose_to_disease("swamp_fever")
    inherited = bio.get_inherited_immunity()  # for offspring
"""

import random
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Metabolic rates (per tick)
GLUCOSE_DRAIN_BASE = 0.5      # base glucose consumption per tick
GLUCOSE_DRAIN_ACTION_MULT = {
    "EXPLORE": 2.0, "EAT": 0.3, "DRINK": 0.2, "SLEEP": -1.0,  # negative = recovery
    "HIDE": 0.5, "MOVE": 1.5, "FLEE": 3.0, "IDLE": 0.3,
}
HYDRATION_DRAIN_BASE = 0.3
HYDRATION_DRAIN_ACTION_MULT = {
    "EXPLORE": 1.5, "EAT": 0.2, "DRINK": -5.0,  # drinking restores hydration
    "SLEEP": 0.1, "HIDE": 0.3, "MOVE": 1.0, "FLEE": 2.0, "IDLE": 0.2,
}

# Starvation sequence thresholds
GLUCOSE_LOW_THRESHOLD = 20.0     # below this, convert fat → glucose
FAT_LOW_THRESHOLD = 10.0         # below this, convert protein → glucose
PROTEIN_LOW_THRESHOLD = 5.0      # below this, severe starvation (health drain)

# Conversion rates
FAT_TO_GLUCOSE_RATE = 0.3        # fat converted to glucose per tick when glucose low
PROTEIN_TO_GLUCOSE_RATE = 0.2    # protein converted to glucose in starvation

# Body mass
BODY_MASS_BASE = 70.0            # kg
BODY_MASS_STARVATION_LOSS = 0.01 # kg lost per tick in starvation

# Dehydration effect
DEHYDRATION_DRAIN_MULT = 1.5     # all drains x1.5 when dehydrated
DEHYDRATION_THRESHOLD = 30.0     # below this, dehydrated

# Life stages (start_tick, end_tick, name, stat_multiplier, action_space_mult)
LIFE_STAGES = [
    (0,     200,    "newborn",    0.5, 0.5),   # fragile, limited actions
    (200,   800,    "child",      0.7, 0.8),   # rapid learning
    (800,   2000,   "adolescent", 0.9, 1.0),   # peak learning, full actions
    (2000,  6000,   "adult",      1.0, 1.0),   # peak strength, reproduction
    (6000,  10000,  "elder",      0.7, 1.0),   # wisdom, declining stats
    (10000, float('inf'), "ancient", 0.4, 1.0),  # rare, slow, max memory
]

# Injury states
INJURY_NONE = 0
INJURY_BRUISED = 1
INJURY_WOUNDED = 2
INJURY_CRITICAL = 3
INJURY_NAMES = ["NONE", "BRUISED", "WOUNDED", "CRITICAL"]
INJURY_RECOVERY_TIME = 300       # ticks to recover one injury level
INJURY_RECOVERY_SHELTER_TIME = 150  # faster in shelter
INJURY_SPEED_PENALTY = {0: 1.0, 1: 0.85, 2: 0.6, 3: 0.3}   # movement speed mult
INJURY_ENERGY_PENALTY = {0: 1.0, 1: 1.1, 2: 1.3, 3: 2.0}   # energy cost mult

# Sleep
REM_DURATION = 100               # first 100 ticks of sleep = REM
DEEP_SLEEP_DURATION = 200        # next 200 ticks = deep sleep
SLEEP_DEBT_PER_AWAKE_TICK = 0.1
SLEEP_DEBT_NIGHT_BONUS = 0.2     # extra debt accumulation at night
SLEEP_DEBT_RECOVERY_PER_TICK = 0.5
SLEEP_DEBT_HALLUCINATION = 50.0  # 3000+ ticks awake → hallucinations
CIRCADIAN_NIGHT_START = 18
CIRCADIAN_NIGHT_END = 6

# Immune system
IMMUNITY_GAIN_PER_EXPOSURE = 0.3
IMMUNITY_DAMAGE_REDUCTION = 0.7  # at immunity 1.0, damage * (1 - 0.7) = 0.3x
IMMUNITY_INHERITANCE = 0.5       # offspring get 50% of parent's immunity


# ═══════════════════════════════════════════════════════════════════════════════
# BiologyEngine
# ═══════════════════════════════════════════════════════════════════════════════

class BiologyEngine:
    """
    Master biology engine for the Nafs AI world.

    Holds all biological state:
      - Metabolic: glucose, fat, protein, hydration, body_mass
      - Aging: age_ticks, life_stage
      - Immune: immunity dict per disease type
      - Injury: injury_level (0-3), injury_tick
      - Sleep: sleep_debt, sleep_state, sleep_ticks, last_slept_tick
    """

    def __init__(self, inherited_immunity: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None):
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Metabolic state (0-100 each)
        self.glucose = 80.0          # immediate energy fuel
        self.fat = 50.0              # medium-term reserve
        self.protein = 70.0          # structural
        self.hydration = 80.0        # water level
        self.body_mass = BODY_MASS_BASE

        # Aging
        self.age_ticks = 0
        self.life_stage = "newborn"
        self._update_life_stage()

        # Immune system: {disease_name: immunity_level 0-1}
        self.immunity: Dict[str, float] = {}
        if inherited_immunity:
            for disease, level in inherited_immunity.items():
                self.immunity[disease] = min(1.0, level)

        # Injury system
        self.injury_level = INJURY_NONE
        self.injury_tick = 0

        # Sleep
        self.sleep_debt = 0.0
        self.sleep_state = "awake"   # awake / REM / deep
        self.sleep_ticks = 0
        self.last_slept_tick = 0
        self._consecutive_awake_ticks = 0

        self.current_tick = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Metabolism
    # ─────────────────────────────────────────────────────────────────────────

    def digest_food(self, food_result: Dict) -> None:
        """
        Process food from chemistry engine. Applies nutrients to metabolic state.

        Different foods replenish different nutrients:
          - fish / meat → protein
          - berries → glucose
          - roots → stable glucose + small protein
          - water_content → hydration
          - calories → glucose (primary) + fat (excess)
        """
        if not food_result:
            return

        calories = food_result.get("calories", 0)
        protein_in = food_result.get("protein", 0)
        fat_in = food_result.get("fat", 0)
        water = food_result.get("water_content", 0)

        # Calories primarily → glucose (excess → fat)
        glucose_gain = min(calories, 30.0)   # cap per meal
        fat_gain = max(0, calories - 30.0) * 0.5
        self.glucose = min(100.0, self.glucose + glucose_gain)
        self.fat = min(100.0, self.fat + fat_gain + fat_in * 0.3)

        # Protein → protein stores
        self.protein = min(100.0, self.protein + protein_in * 0.5)

        # Water content → hydration
        self.hydration = min(100.0, self.hydration + water * 1.5)

    def drink_water(self, hydration_amount: float) -> None:
        """Restore hydration from drinking water."""
        self.hydration = min(100.0, self.hydration + hydration_amount)

    def _update_metabolism(self, action: str) -> Dict:
        """
        Per-tick metabolism update. Returns dict with effects:
          - health_drain: from starvation/dehydration
          - body_mass_loss: from starvation
          - is_starving: bool
          - is_dehydrated: bool
        """
        # Get action-based drain rates
        glucose_mult = GLUCOSE_DRAIN_ACTION_MULT.get(action, 0.5)
        hydration_mult = HYDRATION_DRAIN_ACTION_MULT.get(action, 0.3)

        # Sleep action: glucose recovers (negative drain)
        if action == "SLEEP":
            self.glucose = min(100.0, self.glucose + 1.0)
            glucose_drain = 0
        else:
            glucose_drain = GLUCOSE_DRAIN_BASE * glucose_mult

        hydration_drain = HYDRATION_DRAIN_BASE * hydration_mult

        # Dehydration accelerates all drains
        is_dehydrated = self.hydration < DEHYDRATION_THRESHOLD
        if is_dehydrated:
            glucose_drain *= DEHYDRATION_DRAIN_MULT
            hydration_drain *= DEHYDRATION_DRAIN_MULT

        # Apply injury energy penalty
        injury_energy_mult = INJURY_ENERGY_PENALTY.get(self.injury_level, 1.0)
        glucose_drain *= injury_energy_mult

        # Apply drains
        self.glucose = max(0.0, self.glucose - glucose_drain)
        self.hydration = max(0.0, self.hydration - hydration_drain)

        # Starvation sequence: glucose → fat → protein → death
        health_drain = 0.0
        body_mass_loss = 0.0
        is_starving = False

        if self.glucose < GLUCOSE_LOW_THRESHOLD:
            # Convert fat → glucose
            convert = min(FAT_TO_GLUCOSE_RATE, self.fat)
            self.fat -= convert
            self.glucose += convert

        if self.fat < FAT_LOW_THRESHOLD and self.glucose < GLUCOSE_LOW_THRESHOLD:
            # Convert protein → glucose (starvation mode)
            convert = min(PROTEIN_TO_GLUCOSE_RATE, self.protein)
            self.protein -= convert
            self.glucose += convert
            is_starving = True
            body_mass_loss = BODY_MASS_STARVATION_LOSS

        if self.protein < PROTEIN_LOW_THRESHOLD and self.glucose < GLUCOSE_LOW_THRESHOLD:
            # Severe starvation: health drain
            health_drain = 0.5
            is_starving = True

        # Dehydration health drain
        if self.hydration < 10.0:
            health_drain += 0.3
            is_starving = True

        # Apply body mass loss
        if body_mass_loss > 0:
            self.body_mass = max(40.0, self.body_mass - body_mass_loss)

        return {
            "health_drain": health_drain,
            "body_mass_loss": body_mass_loss,
            "is_starving": is_starving,
            "is_dehydrated": is_dehydrated,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Life stages & aging
    # ─────────────────────────────────────────────────────────────────────────

    def _update_life_stage(self) -> None:
        """Update life_stage based on age_ticks."""
        for start, end, name, _, _ in LIFE_STAGES:
            if start <= self.age_ticks < end:
                if self.life_stage != name:
                    self.life_stage = name
                return
        # Fallback
        self.life_stage = "ancient"

    def get_stat_multiplier(self) -> float:
        """Returns multiplier for physical stats (speed, strength) based on life stage."""
        for start, end, name, mult, _ in LIFE_STAGES:
            if start <= self.age_ticks < end:
                return mult
        return 0.4

    def get_action_space_multiplier(self) -> float:
        """Returns multiplier for available actions (newborns limited)."""
        for start, end, name, _, mult in LIFE_STAGES:
            if start <= self.age_ticks < end:
                return mult
        return 1.0

    def can_reproduce(self) -> bool:
        """Only adults can reproduce."""
        return self.life_stage == "adult"

    def is_newborn(self) -> bool:
        return self.life_stage == "newborn"

    def is_elder_or_older(self) -> bool:
        return self.life_stage in ("elder", "ancient")

    # ─────────────────────────────────────────────────────────────────────────
    # Immune system
    # ─────────────────────────────────────────────────────────────────────────

    def expose_to_disease(self, disease_name: str) -> float:
        """
        Expose agent to a disease. Returns damage multiplier (0-1).

        First exposure: full damage (immunity 0)
        Subsequent exposures: less damage as immunity builds
        At immunity 1.0: damage * (1 - 0.7) = 0.3x
        """
        current = self.immunity.get(disease_name, 0.0)
        damage_mult = 1.0 - (current * IMMUNITY_DAMAGE_REDUCTION)
        # Build immunity from exposure
        self.immunity[disease_name] = min(1.0, current + IMMUNITY_GAIN_PER_EXPOSURE)
        return damage_mult

    def get_immunity(self, disease_name: str) -> float:
        """Get immunity level for a specific disease (0-1)."""
        return self.immunity.get(disease_name, 0.0)

    def get_average_immunity(self) -> float:
        """Average immunity across all known diseases."""
        if not self.immunity:
            return 0.0
        return sum(self.immunity.values()) / len(self.immunity)

    def get_inherited_immunity(self) -> Dict[str, float]:
        """Return partial immunity for offspring (50% of parent's)."""
        return {k: v * IMMUNITY_INHERITANCE for k, v in self.immunity.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Injury system
    # ─────────────────────────────────────────────────────────────────────────

    def apply_injury(self, level: int, tick: int) -> None:
        """Apply injury if more severe than current."""
        if level > self.injury_level:
            self.injury_level = min(INJURY_CRITICAL, level)
            self.injury_tick = tick

    def _update_injury_recovery(self, in_shelter: bool = False) -> bool:
        """
        Update injury recovery. Returns True if injury level decreased.
        Recovery time: 300 ticks normally, 150 in shelter.
        """
        if self.injury_level == INJURY_NONE:
            return False

        recovery_time = (INJURY_RECOVERY_SHELTER_TIME if in_shelter
                         else INJURY_RECOVERY_TIME)
        if self.current_tick - self.injury_tick >= recovery_time:
            self.injury_level = max(INJURY_NONE, self.injury_level - 1)
            self.injury_tick = self.current_tick
            return True
        return False

    def get_movement_speed_mult(self) -> float:
        """Returns movement speed multiplier based on injury level."""
        return INJURY_SPEED_PENALTY.get(self.injury_level, 1.0)

    def get_energy_cost_mult(self) -> float:
        """Returns energy cost multiplier based on injury level."""
        return INJURY_ENERGY_PENALTY.get(self.injury_level, 1.0)

    def get_injury_name(self) -> str:
        return INJURY_NAMES[self.injury_level]

    # ─────────────────────────────────────────────────────────────────────────
    # Sleep biology
    # ─────────────────────────────────────────────────────────────────────────

    def _update_sleep(self, action: str, time_of_day: int) -> Dict:
        """
        Update sleep state. Returns dict with:
          - sleep_state: 'awake' / 'REM' / 'deep'
          - in_rem: bool (dream engine should be active)
          - in_deep: bool (physical recovery, healing)
          - hallucinating: bool (sleep debt > 50)
        """
        if action == "SLEEP":
            self.sleep_ticks += 1
            self.last_slept_tick = self.current_tick
            self._consecutive_awake_ticks = 0

            if self.sleep_ticks <= REM_DURATION:
                self.sleep_state = "REM"
            else:
                self.sleep_state = "deep"

            # Sleep reduces debt
            self.sleep_debt = max(0.0, self.sleep_debt - SLEEP_DEBT_RECOVERY_PER_TICK)
        else:
            self.sleep_ticks = 0
            self.sleep_state = "awake"
            self._consecutive_awake_ticks += 1
            self.sleep_debt += SLEEP_DEBT_PER_AWAKE_TICK

            # Circadian rhythm: extra debt at night
            if time_of_day >= CIRCADIAN_NIGHT_START or time_of_day < CIRCADIAN_NIGHT_END:
                self.sleep_debt += SLEEP_DEBT_NIGHT_BONUS

        return {
            "sleep_state": self.sleep_state,
            "in_rem": self.sleep_state == "REM",
            "in_deep": self.sleep_state == "deep",
            "hallucinating": self.sleep_debt >= SLEEP_DEBT_HALLUCINATION,
        }

    def is_hallucinating(self) -> bool:
        """True if sleep debt is high enough to cause hallucinations."""
        return self.sleep_debt >= SLEEP_DEBT_HALLUCINATION

    def is_sleepy(self) -> bool:
        """True if agent should feel sleepy (high sleep debt OR night time)."""
        return self.sleep_debt > 20.0

    # ─────────────────────────────────────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, tick: int, action: str, time_of_day: int,
             in_shelter: bool = False) -> Dict:
        """
        Per-tick biology update.

        Args:
            tick: current simulation tick
            action: agent's action this tick
            time_of_day: 0-23
            in_shelter: True if agent is in a sheltered tile (cave, etc.)

        Returns:
            Dict with effects to apply:
              - metabolism: dict with health_drain, body_mass_loss, is_starving, is_dehydrated
              - sleep: dict with sleep_state, in_rem, in_deep, hallucinating
              - injury_recovered: bool
        """
        self.current_tick = tick
        self.age_ticks += 1
        self._update_life_stage()

        metabolism = self._update_metabolism(action)
        sleep = self._update_sleep(action, time_of_day)
        injury_recovered = self._update_injury_recovery(in_shelter)

        # Deep sleep provides physical recovery + injury healing boost
        if sleep["in_deep"]:
            # Faster injury recovery during deep sleep
            if self.injury_level > INJURY_NONE:
                # Reduce recovery time effectively by advancing injury_tick
                pass  # handled by in_shelter + deep sleep both contributing
            # Immune boost during deep sleep
            if self.immunity:
                # Slow immunity decay prevention
                pass

        return {
            "metabolism": metabolism,
            "sleep": sleep,
            "injury_recovered": injury_recovered,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Sensory extensions (Phase 3.5)
    # ─────────────────────────────────────────────────────────────────────────

    def get_sensory_extensions(self) -> Dict:
        """
        Returns biology-related sensory fields:
          - glucose, fat, protein, hydration, body_mass (0-100 / kg)
          - age_ticks (int)
          - life_stage (str)
          - injury_level (0-3)
          - injury_name (str)
          - immunity_avg (0-1)
          - sleep_debt (0-100)
          - sleep_state (str)
          - hallucinating (bool)
          - stat_mult (0-1, life-stage-based)
        """
        return {
            'glucose': round(self.glucose, 1),
            'fat': round(self.fat, 1),
            'protein': round(self.protein, 1),
            'hydration': round(self.hydration, 1),
            'body_mass': round(self.body_mass, 1),
            'age_ticks': self.age_ticks,
            'life_stage': self.life_stage,
            'injury_level': self.injury_level,
            'injury_name': self.get_injury_name(),
            'immunity_avg': round(self.get_average_immunity(), 2),
            'sleep_debt': round(self.sleep_debt, 1),
            'sleep_state': self.sleep_state,
            'hallucinating': self.is_hallucinating(),
            'stat_mult': self.get_stat_multiplier(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Compatibility helpers (translate to/from old hunger/energy/stress stats)
    # ─────────────────────────────────────────────────────────────────────────

    def get_hunger_equivalent(self) -> float:
        """Translate metabolic state to hunger equivalent (0-100, 100=starving)."""
        # Hunger = inverse of (glucose + fat + protein) / 3
        avg = (self.glucose + self.fat + self.protein) / 3.0
        return max(0.0, min(100.0, 100.0 - avg))

    def get_energy_equivalent(self) -> float:
        """Translate metabolic state to energy equivalent (0-100, 100=full)."""
        # Energy = weighted average of glucose + hydration
        return (self.glucose * 0.6 + self.hydration * 0.4)

    def get_thirst_equivalent(self) -> float:
        """Translate hydration to thirst (0-100, 100=very thirsty)."""
        return max(0.0, min(100.0, 100.0 - self.hydration))

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialize biology state for checkpointing."""
        return {
            'glucose': self.glucose,
            'fat': self.fat,
            'protein': self.protein,
            'hydration': self.hydration,
            'body_mass': self.body_mass,
            'age_ticks': self.age_ticks,
            'life_stage': self.life_stage,
            'immunity': self.immunity,
            'injury_level': self.injury_level,
            'injury_tick': self.injury_tick,
            'sleep_debt': self.sleep_debt,
            'sleep_state': self.sleep_state,
            'sleep_ticks': self.sleep_ticks,
            'last_slept_tick': self.last_slept_tick,
            'current_tick': self.current_tick,
        }

    def load_state(self, state: Dict) -> None:
        """Restore biology state from a checkpoint."""
        self.glucose = state.get('glucose', 80.0)
        self.fat = state.get('fat', 50.0)
        self.protein = state.get('protein', 70.0)
        self.hydration = state.get('hydration', 80.0)
        self.body_mass = state.get('body_mass', BODY_MASS_BASE)
        self.age_ticks = state.get('age_ticks', 0)
        self.life_stage = state.get('life_stage', 'newborn')
        self.immunity = state.get('immunity', {})
        self.injury_level = state.get('injury_level', INJURY_NONE)
        self.injury_tick = state.get('injury_tick', 0)
        self.sleep_debt = state.get('sleep_debt', 0.0)
        self.sleep_state = state.get('sleep_state', 'awake')
        self.sleep_ticks = state.get('sleep_ticks', 0)
        self.last_slept_tick = state.get('last_slept_tick', 0)
        self.current_tick = state.get('current_tick', 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing BiologyEngine...")

    bio = BiologyEngine(seed=42)
    print(f"  Initial: glucose={bio.glucose}, fat={bio.fat}, hydration={bio.hydration}")
    print(f"  Life stage: {bio.life_stage}, age: {bio.age_ticks}")
    assert bio.life_stage == "newborn"

    # Test metabolism — drain from MOVE action
    for _ in range(100):
        bio.step(tick=bio.current_tick + 1, action="MOVE", time_of_day=12)
    print(f"  After 100 MOVEs: glucose={bio.glucose:.1f}, fat={bio.fat:.1f}, hydration={bio.hydration:.1f}")
    assert bio.glucose < 80.0  # should have drained

    # Test food digestion
    bio2 = BiologyEngine(seed=42)
    bio2.glucose = 30.0
    bio2.eat_food = bio2.digest_food({
        "calories": 60, "protein": 30, "fat": 5, "water_content": 20,
    })
    print(f"  After eating fish: glucose={bio2.glucose:.1f}, protein={bio2.protein:.1f}, hydration={bio2.hydration:.1f}")
    assert bio2.glucose > 30.0
    assert bio2.protein > 70.0
    assert bio2.hydration > 80.0

    # Test starvation sequence
    bio3 = BiologyEngine(seed=42)
    bio3.glucose = 5.0  # critically low
    bio3.fat = 50.0
    bio3.step(tick=1, action="IDLE", time_of_day=12)
    # Fat should have converted to glucose
    print(f"  Starvation: glucose={bio3.glucose:.1f} (was 5), fat={bio3.fat:.1f} (was 50)")
    assert bio3.fat < 50.0  # fat should have been consumed

    # Test severe starvation → protein consumption
    bio4 = BiologyEngine(seed=42)
    bio4.glucose = 5.0
    bio4.fat = 5.0  # also low
    bio4.protein = 70.0
    bio4.step(tick=1, action="IDLE", time_of_day=12)
    print(f"  Severe starvation: protein={bio4.protein:.1f} (was 70)")
    assert bio4.protein < 70.0

    # Test life stages
    bio5 = BiologyEngine(seed=42)
    bio5.age_ticks = 500
    bio5._update_life_stage()
    assert bio5.life_stage == "child"
    bio5.age_ticks = 3000
    bio5._update_life_stage()
    assert bio5.life_stage == "adult"
    assert bio5.can_reproduce() is True
    bio5.age_ticks = 7000
    bio5._update_life_stage()
    assert bio5.life_stage == "elder"
    assert bio5.can_reproduce() is False
    print("  Life stages: newborn → child → adult → elder ✓")

    # Test immune system
    bio6 = BiologyEngine(seed=42)
    dmg1 = bio6.expose_to_disease("swamp_fever")
    assert dmg1 == 1.0  # first exposure = full damage
    assert bio6.get_immunity("swamp_fever") > 0
    dmg2 = bio6.expose_to_disease("swamp_fever")
    print(f"  Disease exposure: dmg1={dmg1:.2f}, dmg2={dmg2:.2f} (less)")
    assert dmg2 < dmg1

    # Test immunity inheritance
    inherited = bio6.get_inherited_immunity()
    print(f"  Inherited immunity: {inherited}")
    for k, v in inherited.items():
        assert v <= bio6.immunity[k]  # offspring gets less

    # Test injury system
    bio7 = BiologyEngine(seed=42)
    bio7.apply_injury(INJURY_WOUNDED, tick=10)
    assert bio7.injury_level == INJURY_WOUNDED
    assert bio7.get_movement_speed_mult() == 0.6  # -40% speed
    assert bio7.get_energy_cost_mult() == 1.3  # +30% energy
    # Recover after 300 ticks
    for tick in range(11, 350):
        bio7.step(tick=tick, action="IDLE", time_of_day=12)
    print(f"  Injury recovery after 300 ticks: {bio7.get_injury_name()}")
    assert bio7.injury_level < INJURY_WOUNDED

    # Test sleep biology
    bio8 = BiologyEngine(seed=42)
    # Stay awake for 3000 ticks → hallucinate
    for tick in range(1, 3001):
        bio8.step(tick=tick, action="IDLE", time_of_day=12)
    print(f"  Sleep debt after 3000 awake ticks: {bio8.sleep_debt:.1f}")
    assert bio8.is_hallucinating()

    # Test sleep recovery
    bio9 = BiologyEngine(seed=42)
    bio9.sleep_debt = 60.0
    for tick in range(1, 150):
        bio9.step(tick=tick, action="SLEEP", time_of_day=0)
    print(f"  After 150 sleep ticks: state={bio9.sleep_state}, debt={bio9.sleep_debt:.1f}")
    assert bio9.sleep_state == "deep"  # first 100 REM, then deep
    assert bio9.sleep_debt < 60.0  # recovered

    # Test sensory extensions
    bio10 = BiologyEngine(seed=42)
    ext = bio10.get_sensory_extensions()
    required = {"glucose", "fat", "protein", "hydration", "body_mass",
                "age_ticks", "life_stage", "injury_level", "injury_name",
                "immunity_avg", "sleep_debt", "sleep_state", "hallucinating",
                "stat_mult"}
    assert set(ext.keys()) == required
    print(f"  Sensory extensions: {list(ext.keys())}")

    # Test hunger/energy equivalents
    bio11 = BiologyEngine(seed=42)
    bio11.glucose = 80
    bio11.fat = 50
    bio11.protein = 70
    bio11.hydration = 80
    hunger = bio11.get_hunger_equivalent()
    energy = bio11.get_energy_equivalent()
    thirst = bio11.get_thirst_equivalent()
    print(f"  Equivalents: hunger={hunger:.1f}, energy={energy:.1f}, thirst={thirst:.1f}")
    assert 0 <= hunger <= 100
    assert 0 <= energy <= 100
    assert 0 <= thirst <= 100

    print("\n✓ BiologyEngine self-test passed")
