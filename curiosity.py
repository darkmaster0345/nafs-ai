"""
Nafs AI — Curiosity Module (Intrinsic Motivation)
"Why does Adam explore? Because the unknown calls to him."

Philosophy:
  - Curiosity is the desire to reduce uncertainty
  - Adam doesn't explore because we tell him to
  - He explores because states he hasn't seen are INTRINSICLY rewarding
  - This is not extrinsic reward — it comes from WITHIN
  - As Adam learns the world, curiosity naturally fades (he's seen it all)
  - This creates the exploration-exploitation balance organically

Implementation:
  - Count-based curiosity: track how often Adam has visited each state
  - Intrinsic reward = 1 / sqrt(N(s)) where N(s) is visit count
  - Novel states give high curiosity, familiar states give low
  - Curiosity bonus is added to the PPO reward (small, not dominant)
  - State is discretized into buckets for counting (like pattern memory)

Tunable parameters:
  - curiosity_bonus: max intrinsic reward per step (default 0.15)
  - curiosity_decay: how fast curiosity fades per episode (default 0.98)
  - state_discretization: how coarsely to bucket the state space
"""

import numpy as np
from collections import defaultdict


class CuriosityModule:
    """
    Count-based intrinsic motivation.

    Adam receives bonus reward for visiting states he hasn't seen before.
    This is genuine curiosity: the unknown is rewarding in itself.

    The curiosity bonus naturally decays as Adam learns the world:
      - Episode 1: everything is new → high curiosity
      - Episode 100: most things seen → low curiosity (but still present)
      - Episode 500: curiosity is faint but never zero

    This means Adam will always have SOME drive to explore,
    preventing the common RL failure mode of getting stuck
    in a safe but uninteresting behavioral rut.
    """

    def __init__(self, curiosity_bonus: float = 0.15,
                 curiosity_decay: float = 0.98,
                 min_curiosity: float = 0.01):
        self.curiosity_bonus = curiosity_bonus
        self.curiosity_decay = curiosity_decay
        self.min_curiosity = min_curiosity

        # Visit counts for each discretized state
        # Key: state_hash string, Value: number of visits
        self.visit_counts = defaultdict(int)

        # Per-state curiosity value (cached for speed)
        # Recomputed periodically via decay
        self.state_curiosity = defaultdict(float)

        # Total intrinsic reward given this episode
        self.episode_intrinsic_reward = 0.0

        # Statistics
        self.total_states_seen = 0
        self.total_curiosity_given = 0.0
        self.episodes_processed = 0

    def discretize_state(self, world_state: dict, adam_stats: dict) -> str:
        """
        Convert continuous world state into a discrete hash.

        This is critical: too fine-grained and every state is "new",
        too coarse and Adam can't distinguish important differences.

        We use the same discretization as pattern_memory for consistency.
        """
        # Temperature: 5 buckets
        temp = world_state.get('temperature', 20)
        if temp < 0: temp_b = "frigid"
        elif temp < 10: temp_b = "cold"
        elif temp < 25: temp_b = "warm"
        elif temp < 35: temp_b = "hot"
        else: temp_b = "burning"

        # Light: 3 buckets
        light = world_state.get('light_level', 0.5)
        if light < 0.2: light_b = "dark"
        elif light < 0.7: light_b = "dim"
        else: light_b = "bright"

        # Food: present or not
        food_b = "food" if world_state.get('smell_food', 0) > 0.3 else "nofood"

        # Danger: present or not
        danger_b = "danger" if world_state.get('smell_danger', 0) > 0.3 else "safe"

        # Water: present or not
        water_b = "wet" if world_state.get('wetness', 0) > 0.3 else "dry"

        # Sound: 3 levels
        sound = world_state.get('sound_level', 0.1)
        if sound < 0.2: sound_b = "quiet"
        elif sound < 0.6: sound_b = "moderate"
        else: sound_b = "loud"

        # Health: 3 buckets
        health = adam_stats.get('health', 100)
        if health > 70: health_b = "healthy"
        elif health > 30: health_b = "hurt"
        else: health_b = "critical"

        # Hunger: 3 buckets
        hunger = adam_stats.get('hunger', 0)
        if hunger < 30: hunger_b = "fed"
        elif hunger < 65: hunger_b = "hungry"
        else: hunger_b = "starving"

        # Energy: 3 buckets
        energy = adam_stats.get('energy', 100)
        if energy > 50: energy_b = "energetic"
        elif energy > 20: energy_b = "tired"
        else: energy_b = "exhausted"

        # Stress: 2 buckets
        stress = adam_stats.get('stress', 0)
        stress_b = "stressed" if stress > 40 else "calm"

        return f"{temp_b}_{light_b}_{food_b}_{danger_b}_{water_b}_{sound_b}_{health_b}_{hunger_b}_{energy_b}_{stress_b}"

    def compute_intrinsic_reward(self, world_state: dict, adam_stats: dict) -> float:
        """
        Compute intrinsic curiosity reward for the current state.

        Uses count-based exploration: reward = curiosity_bonus / sqrt(N(s))
        - First visit: full bonus (1.0 * curiosity_bonus)
        - 4th visit: half bonus
        - 100th visit: 1/10th bonus
        - Never reaches zero (min_curiosity floor)

        This means:
          - Adam ALWAYS gets some reward for being curious
          - But the reward naturally diminishes as he learns
          - Completely novel states are strongly attractive
          - Well-known states are mildly interesting at best
        """
        state_key = self.discretize_state(world_state, adam_stats)

        # Increment visit count
        self.visit_counts[state_key] += 1
        count = self.visit_counts[state_key]

        if count == 1:
            # First time! Maximum curiosity
            self.total_states_seen += 1
            intrinsic = self.curiosity_bonus
        else:
            # Diminishing returns: 1/sqrt(N)
            intrinsic = self.curiosity_bonus / np.sqrt(count)

        # Apply minimum floor — curiosity never fully dies
        intrinsic = max(intrinsic, self.min_curiosity * self.curiosity_bonus)

        # Apply episode decay — as Adam gains experience, curiosity
        # becomes a smaller proportion of total reward
        decay_factor = self.curiosity_decay ** self.episodes_processed
        intrinsic *= max(decay_factor, 0.3)  # Never decay below 30%

        # Track stats
        self.episode_intrinsic_reward += intrinsic
        self.total_curiosity_given += intrinsic

        return intrinsic

    def reset_episode(self):
        """Called at the start of each new episode."""
        self.episode_intrinsic_reward = 0.0

    def end_episode(self):
        """Called at the end of each episode."""
        self.episodes_processed += 1

    def get_curiosity_stats(self) -> dict:
        """Get statistics about curiosity exploration."""
        total_visits = sum(self.visit_counts.values())
        avg_visits = total_visits / max(len(self.visit_counts), 1)

        # Curiosity distribution: how many states are "novel" vs "familiar"
        novel_states = sum(1 for v in self.visit_counts.values() if v <= 3)
        familiar_states = sum(1 for v in self.visit_counts.values() if v > 10)

        return {
            "total_states_discovered": len(self.visit_counts),
            "total_state_visits": total_visits,
            "avg_visits_per_state": round(avg_visits, 1),
            "novel_states": novel_states,       # Visited 1-3 times
            "familiar_states": familiar_states,  # Visited 10+ times
            "total_curiosity_given": round(self.total_curiosity_given, 2),
            "episodes_processed": self.episodes_processed,
        }

    def save_state(self) -> dict:
        """Serialize curiosity state for persistence."""
        return {
            "visit_counts": dict(self.visit_counts),
            "episodes_processed": self.episodes_processed,
            "total_states_seen": self.total_states_seen,
            "total_curiosity_given": self.total_curiosity_given,
        }

    def load_state(self, data: dict):
        """Restore curiosity state from persistence."""
        if "visit_counts" in data:
            self.visit_counts = defaultdict(int, data["visit_counts"])
        self.episodes_processed = data.get("episodes_processed", 0)
        self.total_states_seen = data.get("total_states_seen", 0)
        self.total_curiosity_given = data.get("total_curiosity_given", 0.0)
