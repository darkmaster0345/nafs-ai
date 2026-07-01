"""
Nafs AI — Eve Agent Module (v0.3)
==================================

The second agent. Separate brain. Separate memory. Separate personality.

Philosophy:
    Adam was alone. He learned the world through his own pain and pleasure.
    Now Eve wakes up in the same world — but with her own mind.

    They share the world but not memory.
    They share the same biomes, weather, and time — but not experience.
    They may encounter each other, but neither knows the other's thoughts.

    This is the experiment: what happens when two blank slates
    coexist in the same reality?

Implementation:
    Eve is a complete second agent with:
      - Her own BabyBrain (PPO+GRU)
      - Her own ThoughtEngine (6-phase consciousness)
      - Her own CuriosityModule
      - Her own DreamEngine
      - Her own LearnedThinker
      - Her own position in the world
      - Her own vital stats

    The WorldSim is shared — they live in the same procedural world,
    experience the same weather, and can see each other.

    Eve is born at a different location than Adam. They can move
    independently. When they're close enough, they can "see" each other
    (a sensory input), but they cannot communicate thoughts.

Usage:
    from eve_agent import EveAgent

    eve = EveAgent(world_sim, device='cpu')
    eve.birth()

    # Each tick:
    eve_world_state = eve.get_world_state(world_sim)
    action = eve.choose_action(eve_world_state)
    result = eve.step(action, world_sim)

    # When Eve dies:
    eve.die()
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from typing import Optional, Tuple, Dict, Any

from baby_brain_model import BabyBrain
from thought_engine import ThoughtEngine
from curiosity import CuriosityModule
from dreaming import DreamEngine
from sensory_encoder import encode_sensory_input, INPUT_DIM
from learned_thinking import LearnedThinker, blend_thoughts
from config import PPO_CONFIG, ACTION_NAMES
from world_sim import BIOMES, WEATHER_TYPES


class EveAgent:
    """
    The second agent — Eve.

    She has her own brain, her own memory, her own personality.
    She lives in the same world as Adam but experiences it independently.

    Key differences from Adam:
      - Different birth location (at least 10 tiles away from Adam)
      - Her own brain weights (initialized randomly, just like Adam)
      - Her own thought engine (separate vocabulary, memories, fears)
      - Her own curiosity module
      - Her own dream engine
      - Her own learned thinker
      - Her own vital stats

    What they share:
      - The world map (biomes, terrain)
      - Weather (same Markov chain)
      - Time of day
      - Each other's presence (when nearby)
    """

    def __init__(
        self,
        world_sim,
        device: str = 'cpu',
        learning_rate: float = None,
        name: str = "Eve",
        input_dim: int = None,
        first_contact: bool = False,
    ):
        self.name = name
        self.device = torch.device(device)
        self.world_sim = world_sim

        # v1.0: First-contact mode uses 23-dim sensory input (21 base + 2 multi-agent)
        # so Eve's PPO can observe Adam's last action.
        self.first_contact = first_contact
        if first_contact:
            from sensory_encoder_multi import INPUT_DIM_MULTI, encode_sensory_input_multi
            self._input_dim = INPUT_DIM_MULTI
            self._encode_fn = lambda ws, st, **kw: encode_sensory_input_multi(
                ws, st,
                fear_signal=kw.get('fear_signal', 0.0),
                pleasure_signal=kw.get('pleasure_signal', 0.0),
                pattern_confidence=kw.get('pattern_confidence', 0.0),
                other_presence=kw.get('other_presence', 0.0),
                other_last_action_idx=kw.get('other_last_action_idx', -1),
            )
        else:
            self._input_dim = input_dim or INPUT_DIM
            # Wrap to ignore multi-agent kwargs (backward compat with 21-dim encoder)
            self._encode_fn = lambda ws, st, **kw: encode_sensory_input(
                ws, st,
                fear_signal=kw.get('fear_signal', 0.0),
                pleasure_signal=kw.get('pleasure_signal', 0.0),
                pattern_confidence=kw.get('pattern_confidence', 0.0),
            )

        # Eve's brain — completely separate from Adam's
        HIDDEN_DIM = PPO_CONFIG["hidden_dim"]
        NUM_ACTIONS = len(ACTION_NAMES)
        self.model = BabyBrain(self._input_dim, HIDDEN_DIM, NUM_ACTIONS).to(self.device)
        lr = learning_rate or PPO_CONFIG["learning_rate"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        # Eve's consciousness — separate thought engine
        self.thought_engine = ThoughtEngine(memory_size=20)
        self.curiosity = CuriosityModule()
        self.dream_engine = DreamEngine()
        self.learned_thinker = LearnedThinker(sensory_dim=self._input_dim, device=str(self.device))
        self.learned_thinker.start()

        # Eve's position and stats
        self.x = 0
        self.y = 0
        self.facing = 'north'
        self.stats = {
            'health': 100.0,
            'hunger': 0.0,
            'thirst': 0.0,
            'energy': 100.0,
            'stress': 0.0,
            'pain': 0.0,
        }
        self.current_biome = 'plains'

        # State
        self.alive = True
        self.tick = 0
        self.total_reward = 0.0
        self.hidden_state = None
        self.sensory_input = None
        self.action_counts = {a: 0 for a in ACTION_NAMES}

        # Latest inner experience
        self.latest_thought = "quiet. still."
        self.latest_emotion = "uncertain"
        self.latest_dream = None
        self.latest_reflection = None
        self.latest_dialogue = ""
        self.recent_actions = []

        # PPO buffer
        self.obs_list = []
        self.actions_list = []
        self.old_log_probs = []
        self.rewards_list = []
        self.values_list = []
        self.masks_list = []
        self.episode_intrinsic_reward = 0.0

        # PPO config
        self.GAMMA = PPO_CONFIG["gamma"]
        self.GAE_LAMBDA = PPO_CONFIG["gae_lambda"]
        self.CLIP_EPSILON = PPO_CONFIG["clip_epsilon"]
        self.PPO_UPDATE_INTERVAL = PPO_CONFIG.get("update_interval", 64)
        self.REFLECTION_INTERVAL = 20

        # Stats for dashboard
        self.birth_tick = 0
        self.death_cause = None

    def birth(self, adam_x: int = None, adam_y: int = None):
        """
        Birth Eve at a location far from Adam.

        Args:
            adam_x, adam_y: Adam's position — Eve will spawn at least 10 tiles away
        """
        # Find a spawn location far from Adam
        if adam_x is not None and adam_y is not None:
            for _ in range(100):
                self.x = random.randint(0, self.world_sim.world_map.width - 1)
                self.y = random.randint(0, self.world_sim.world_map.height - 1)
                dist = abs(self.x - adam_x) + abs(self.y - adam_y)
                if dist >= 10:
                    break
        else:
            self.x = random.randint(0, self.world_sim.world_map.width - 1)
            self.y = random.randint(0, self.world_sim.world_map.height - 1)

        self.current_biome = self.world_sim.world_map.get_biome(self.x, self.y)

        # Initialize hidden state
        self.hidden_state = self.model.init_hidden(1).to(self.device)

        # Initial sensory input
        world_state = self.get_world_state()
        phase5 = self.thought_engine.get_phase5_signals(world_state, self.stats)
        self.sensory_input = self._encode_fn(
            world_state, self.stats,
            fear_signal=phase5['fear_signal'],
            pleasure_signal=phase5['pleasure_signal'],
            pattern_confidence=phase5['pattern_confidence'],
            other_presence=0.0,
            other_last_action_idx=-1,
        ).to(self.device)

        self.alive = True
        self.latest_thought = f"{self.name} wakes. strange. new."
        self.latest_emotion = "curious"

    def get_world_state(self, adam_x: int = None, adam_y: int = None) -> dict:
        """
        Get Eve's view of the world.

        Eve sees:
          - Her own biome (where she's standing)
          - The weather (shared with Adam)
          - Time of day (shared)
          - Her own position
          - Adam's position IF he's nearby (within sight range)

        Args:
            adam_x, adam_y: Adam's position (to calculate "other presence")

        Returns:
            World state dict (same format as WorldSim._get_world_state)
        """
        biome = self.world_sim.world_map.get_biome(self.x, self.y)
        biome_data = BIOMES.get(biome, {})
        weather_data = WEATHER_TYPES.get(self.world_sim.weather_system.current, {})

        # Time of day from world sim
        time_of_day = getattr(self.world_sim, 'time_of_day', 12)
        light_level = 1.0 if 6 <= time_of_day <= 18 else 0.3

        # Check if Adam is nearby (within sight range = 3 tiles)
        other_presence = 0.0
        other_direction = "none"
        if adam_x is not None and adam_y is not None:
            dx = adam_x - self.x
            dy = adam_y - self.y
            dist = abs(dx) + abs(dy)
            if dist <= 3:
                other_presence = 1.0 - (dist / 4.0)
                # Direction to other
                if abs(dx) > abs(dy):
                    other_direction = "east" if dx > 0 else "west"
                else:
                    other_direction = "south" if dy > 0 else "north"

        # Get nearby resources (same logic as WorldSim but from Eve's position)
        smell_food = 0.0
        smell_danger = 0.0
        nearby_biomes = self.world_sim.world_map.get_nearby_biomes(self.x, self.y, 2)
        for nb in nearby_biomes:
            if nb in ['forest', 'jungle', 'plains']:
                smell_food = max(smell_food, 0.3)
            if nb in ['volcano', 'swamp']:
                smell_danger = max(smell_danger, 0.4)

        return {
            'biome': biome,
            'weather': self.world_sim.weather_system.current,
            'temperature': biome_data.get('base_temp', 20) + weather_data.get('temp_mod', 0),
            'time_of_day': time_of_day,
            'light_level': light_level,
            'adam_x': self.x,  # Reuse field name for consistency
            'adam_y': self.y,
            'facing': self.facing,
            'smell_food': smell_food,
            'smell_danger': smell_danger,
            'wetness': 0.5 if self.world_sim.weather_system.current in ['rain', 'storm'] else 0.0,
            'visibility': weather_data.get('visibility', 1.0),
            'sound_level': 0.0,
            'touch_softness': 0.5,
            # Eve-specific: presence of the other agent
            'other_presence': other_presence,
            'other_direction': other_direction,
        }

    def choose_action(self, world_state: dict) -> Tuple[str, int, torch.Tensor, torch.Tensor]:
        """
        Eve chooses an action using her own brain.

        Returns:
            action_name: str (e.g., "EXPLORE")
            action_idx: int
            log_prob: torch.Tensor
            value: torch.Tensor
        """
        with torch.no_grad():
            action_logits, value, new_hidden = self.model(
                self.sensory_input.unsqueeze(0), self.hidden_state
            )
            # Convert logits to probabilities (BabyBrain returns logits, not probs)
            action_probs = torch.softmax(action_logits, dim=-1)
            # Clamp to avoid NaN
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            # Sample action
            action_idx = torch.multinomial(action_probs, 1)
            log_prob = torch.log(action_probs.gather(1, action_idx))

            self.hidden_state = new_hidden

        action_name = ACTION_NAMES[action_idx.item()]
        return action_name, action_idx.item(), log_prob, value.squeeze()

    def step(self, action: str, adam_x: int = None, adam_y: int = None,
             adam_last_action_idx: int = -1) -> dict:
        """
        Eve takes a step in the world.

        Args:
            action: The action to take
            adam_x, adam_y: Adam's position (for "other presence" sensing)
            adam_last_action_idx: Adam's last action index (0-7) for v1.0 first-contact
                                  mode. -1 = unknown/not seen.

        Returns:
            Result dict with reward, done, thought, emotion, etc.
        """
        if not self.alive:
            return {'done': True, 'reward': 0.0}

        self.tick += 1
        prev_stats = copy.deepcopy(self.stats)

        # Get current world state
        world_state = self.get_world_state(adam_x, adam_y)
        biome = world_state['biome']
        biome_data = BIOMES.get(biome, {})

        # Apply action effects (similar to WorldSim.step but for Eve)
        reward = self._apply_action(action, biome_data)

        # Apply vital decay
        self.stats['hunger'] += biome_data.get('hunger_rate', 0.5)
        self.stats['energy'] -= biome_data.get('energy_drain', 0.3)
        self.stats['stress'] += 0.1
        self.stats['health'] -= 0.05  # Natural decay

        # Weather effects
        weather = self.world_sim.weather_system.current
        if weather in ['rain', 'storm']:
            self.stats['energy'] -= 0.3
            if weather == 'storm':
                self.stats['stress'] += 0.2
        elif weather in ['snow', 'blizzard']:
            self.stats['energy'] -= 0.5
            self.stats['health'] -= 0.2
        elif weather == 'heatwave':
            self.stats['energy'] -= 0.8
            self.stats['hunger'] += 0.3

        # Clamp stats
        for k in self.stats:
            self.stats[k] = max(0, min(100, self.stats[k]))

        # Check death
        done = False
        if self.stats['health'] <= 0:
            done = True
            self.death_cause = "health depletion"
        elif self.stats['hunger'] >= 100:
            done = True
            self.death_cause = "starvation"
        elif self.stats['energy'] <= 0:
            done = True
            self.death_cause = "exhaustion"

        # Generate thought + emotion using Eve's own thought engine
        experience = self.thought_engine.experience(
            world_state=world_state,
            adam_stats=self.stats,
            action=action,
            prev_stats=prev_stats,
            tick=self.tick,
            reward=reward,
        )
        rule_based_thought = experience.get('thought', 'quiet. still.')
        self.latest_emotion = experience.get('emotion', 'uncertain')

        # Record experience for learned thinker
        if self.sensory_input is not None:
            self.learned_thinker.record_experience(self.sensory_input, rule_based_thought)

        # Blend rule-based and learned thoughts
        if self.learned_thinker.is_ready():
            learned_thought = self.learned_thinker.generate_thought(self.sensory_input)
            self.latest_thought = blend_thoughts(
                rule_based_thought, learned_thought, self.learned_thinker.confidence
            )
        else:
            self.latest_thought = rule_based_thought

        # Reflection
        self.recent_actions.append(action)
        if self.tick % self.REFLECTION_INTERVAL == 0 and self.tick > 0:
            reflection = self.thought_engine.reflect(
                world_state, self.stats,
                recent_actions=self.recent_actions
            )
            if reflection.get('has_reflection'):
                self.latest_reflection = reflection

        # Dreaming during SLEEP
        dream_data = None
        if action == "SLEEP" and self.tick > 1:
            dream = self.dream_engine.dream(
                self.thought_engine.persistent_memory,
                self.thought_engine.memory
            )
            if dream.get('dream_type') != 'empty':
                self.latest_dream = dream
                dream_data = {
                    "type": dream.get('dream_type', ''),
                    "thoughts": dream.get('thoughts', []),
                }

        # Update action counts
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        # Compute curiosity reward
        curiosity_reward = self.curiosity.compute_intrinsic_reward(world_state, self.stats)
        reward += curiosity_reward
        self.episode_intrinsic_reward += curiosity_reward

        # Encode next observation
        next_world_state = self.get_world_state(adam_x, adam_y)
        next_phase5 = self.thought_engine.get_phase5_signals(next_world_state, self.stats)
        # v1.0: pass other_presence and adam's last action for first-contact mode
        other_presence = next_world_state.get('other_presence', 0.0)
        self.sensory_input = self._encode_fn(
            next_world_state, self.stats,
            fear_signal=next_phase5['fear_signal'],
            pleasure_signal=next_phase5['pleasure_signal'],
            pattern_confidence=next_phase5['pattern_confidence'],
            other_presence=other_presence,
            other_last_action_idx=adam_last_action_idx,
        ).to(self.device)

        self.total_reward += reward

        # PPO buffer
        # (Simplified — full PPO update is handled by the simulator)

        if done:
            self.alive = False

        return {
            'done': done,
            'reward': reward,
            'thought': self.latest_thought,
            'emotion': self.latest_emotion,
            'dream': dream_data,
            'death_cause': self.death_cause,
        }

    def _apply_action(self, action: str, biome_data: dict) -> float:
        """Apply an action and return the reward. Updates Eve's state."""
        reward = 0.0

        if action == "EXPLORE":
            reward = 0.1
            # Random small movement
            dx = random.choice([-1, 0, 0, 1])
            dy = random.choice([-1, 0, 0, 1])
            self._move(dx, dy)
            # Chance to find food/water
            if biome_data.get('food_availability', 0) > 0.3 and random.random() < 0.3:
                self.stats['hunger'] = max(0, self.stats['hunger'] - 10)
                reward += 0.3

        elif action == "EAT":
            food_avail = biome_data.get('food_availability', 0)
            if food_avail > 0.3 and random.random() < food_avail:
                self.stats['hunger'] = max(0, self.stats['hunger'] - 20)
                self.stats['health'] = min(100, self.stats['health'] + 2)
                reward = 1.0
            else:
                reward = -0.1

        elif action == "DRINK":
            water_avail = biome_data.get('water_availability', 0)
            if water_avail > 0.3 and random.random() < water_avail:
                self.stats['thirst'] = max(0, self.stats['thirst'] - 20)
                reward = 0.2
            else:
                reward = -0.05

        elif action == "SLEEP":
            shelter = biome_data.get('shelter_quality', 0)
            if shelter > 0.5:
                self.stats['energy'] = min(100, self.stats['energy'] + 15)
                reward = 0.3
            else:
                self.stats['energy'] = min(100, self.stats['energy'] + 5)
                reward = 0.1

        elif action == "HIDE":
            if biome_data.get('danger_level', 0) > 0.3:
                reward = 0.5
                self.stats['stress'] = max(0, self.stats['stress'] - 5)
            else:
                reward = -0.05

        elif action == "MOVE":
            # Move in facing direction
            dx, dy = self._facing_delta()
            self._move(dx, dy)
            reward = 0.05

        elif action == "FLEE":
            # Move away from danger (random direction)
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            self._move(dx, dy)
            if biome_data.get('danger_level', 0) > 0.3:
                reward = 0.3
                self.stats['stress'] = max(0, self.stats['stress'] - 3)
            else:
                reward = -0.2

        elif action == "IDLE":
            reward = -0.05

        return reward

    def _move(self, dx: int, dy: int):
        """Move Eve by (dx, dy) with world wrapping."""
        if dx != 0 or dy != 0:
            self.x = (self.x + dx) % self.world_sim.world_map.width
            self.y = (self.y + dy) % self.world_sim.world_map.height
            self.current_biome = self.world_sim.world_map.get_biome(self.x, self.y)
            # Update facing
            if abs(dx) > abs(dy):
                self.facing = "east" if dx > 0 else "west"
            elif dy != 0:
                self.facing = "south" if dy > 0 else "north"

    def _facing_delta(self) -> Tuple[int, int]:
        """Get movement delta based on current facing direction."""
        return {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0),
        }.get(self.facing, (0, 0))

    def get_personality(self) -> dict:
        """Get Eve's personality summary."""
        return self.thought_engine.get_personality()

    def get_stats(self) -> dict:
        """Get Eve's stats for dashboard display."""
        cs = self.curiosity.get_curiosity_stats()
        ds = self.dream_engine.get_dream_stats()
        return {
            'name': self.name,
            'alive': self.alive,
            'tick': self.tick,
            'position': [self.x, self.y],
            'biome': self.current_biome,
            'health': round(self.stats['health'], 1),
            'hunger': round(self.stats['hunger'], 1),
            'energy': round(self.stats['energy'], 1),
            'stress': round(self.stats['stress'], 1),
            'pain': round(self.stats['pain'], 1),
            'total_reward': round(self.total_reward, 2),
            'thought': self.latest_thought,
            'emotion': self.latest_emotion,
            'action_counts': self.action_counts,
            'vocabulary_size': len(self.thought_engine.get_vocabulary()),
            'personality': self.get_personality().get('disposition', 'uncertain'),
            'fear_triggers': self.get_personality().get('fear_triggers', 0),
            'good_memories': self.get_personality().get('good_memories', 0),
            'patterns_learned': self.get_personality().get('patterns_learned', 0),
            'curiosity_states': cs.get('total_states_discovered', 0),
            'dreams_total': ds.get('total_dreams', 0),
            'nightmares': ds.get('nightmares', 0),
            'peaceful_dreams': ds.get('peaceful_dreams', 0),
            'learned_thinking': self.learned_thinker.get_stats(),
            'death_cause': self.death_cause,
        }

    def die(self):
        """Mark Eve as dead."""
        self.alive = False
        if not self.death_cause:
            self.death_cause = "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing EveAgent...")

    from world_sim import WorldSim

    # Create a world
    env = WorldSim()
    env.reset()

    # Create Eve
    eve = EveAgent(env, device='cpu')
    eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)

    print(f"Eve born at ({eve.x}, {eve.y}) in {eve.current_biome}")
    print(f"Adam is at ({env.adam_x}, {env.adam_y})")
    dist = abs(eve.x - env.adam_x) + abs(eve.y - env.adam_y)
    print(f"Distance from Adam: {dist} tiles")
    assert dist >= 10, "Eve should spawn at least 10 tiles from Adam"

    # Run a few ticks
    print("\nRunning 20 ticks...")
    for i in range(20):
        action_name, _, _, _ = eve.choose_action(eve.get_world_state(env.adam_x, env.adam_y))
        result = eve.step(action_name, env.adam_x, env.adam_y)
        if i % 5 == 0:
            print(f"  Tick {eve.tick}: {action_name} | HP:{eve.stats['health']:.0f} "
                  f"H:{eve.stats['hunger']:.0f} E:{eve.stats['energy']:.0f} "
                  f"| thought: '{eve.latest_thought}'")

        if result['done']:
            print(f"  Eve died at tick {eve.tick}: {result.get('death_cause')}")
            break

    stats = eve.get_stats()
    print(f"\nFinal stats:")
    print(f"  Position: ({stats['position'][0]}, {stats['position'][1]})")
    print(f"  Total reward: {stats['total_reward']}")
    print(f"  Personality: {stats['personality']}")
    print(f"  Vocab size: {stats['vocabulary_size']}")
    print(f"  Action counts: {stats['action_counts']}")

    print("\n✓ EveAgent works correctly")
