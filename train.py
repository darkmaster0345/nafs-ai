"""
Nafs AI — Single-Life Simulation (No Episodes)
"Adam lives once. When he dies, he dies. Run the program again."

Architecture:
  - NO episodes. Adam has ONE continuous life.
  - Every tick is displayed in real-time with a clear, readable GUI.
  - Minecraft-like world with biomes and random weather.
  - PPO+GRU learns online during Adam's single lifetime.
  - TensorBoard logging for every significant tick.
  - When Adam dies, the program ends. No restart. No memory saved.
  - Run again to birth a new Adam in a new world.

Phases 1-6 (all preserved):
  - Inner Voice (thought, emotion, world description)
  - Vocabulary Discovery (naming from experience)
  - Dialogue Gap (subconscious)
  - Persistent Memory + Personality (within one life)
  - Fear/Pleasure Maps
  - Dual-Speed Processing

Run:     python train.py
TB:      tensorboard --logdir runs/
Test:    pytest tests/ -v
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
import time
import json
import traceback
import copy
import sys

from sensory_encoder import encode_sensory_input, INPUT_DIM
from baby_brain_model import BabyBrain
from world_sim import WorldSim, BIOMES, WEATHER_TYPES
from thought_engine import ThoughtEngine
from curiosity import CuriosityModule
from dreaming import DreamEngine
from tb_logger import TBLogger

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

HIDDEN_DIM = 256
NUM_ACTIONS = len(WorldSim.ACTIONS)
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LEARNING_RATE = 3e-4
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
ENTROPY_COEF = 0.05
DIVERSITY_PENALTY = 0.25
DIVERSITY_WINDOW = 5

# PPO update frequency — learn every N ticks
PPO_UPDATE_INTERVAL = 64

# Display settings
TICK_DELAY = 0.02          # Seconds between ticks (for readability)
COMPACT_EVERY = 1          # Show compact display every N ticks
FULL_DISPLAY_EVERY = 20    # Show full stats display every N ticks

# Phase 6 reflection
REFLECTION_INTERVAL = 20   # Reflect every N ticks
REFLECTION_FOLLOW_BONUS = 0.05
REFLECTION_IGNORE_PENALTY = 0.02
PATTERN_CONFIDENCE_THRESHOLD = 0.5

# Curiosity
CURIOSITY_BONUS = 0.15
CURIOSITY_DECAY = 0.98
CURIOSITY_MIN = 0.01

# Checkpoint save
CHECKPOINT_INTERVAL = 500  # Save model every N ticks
MODEL_DIR = "checkpoints"

# TensorBoard log interval
TB_LOG_INTERVAL = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Display Functions — The Improved GUI
# ═══════════════════════════════════════════════════════════════════════════════

def health_bar(value, max_val=100, width=10):
    """Create a visual health bar like [=======   ]"""
    filled = int(value / max_val * width)
    empty = width - filled
    if value > 70:
        color = "\033[92m"  # Green
    elif value > 40:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red
    reset = "\033[0m"
    return f"{color}[{'=' * filled}{' ' * empty}]{reset}"


def compact_tick_display(tick, world_state, adam_stats, action, reward,
                         thought, emotion, biome_data, weather_data):
    """Show a compact one-line display for each tick."""
    biome_emoji = biome_data.get('emoji', '?')
    weather_emoji = weather_data.get('emoji', '?')
    biome_name = world_state.get('biome', '?')[:6]
    weather_name = world_state.get('weather', '?')[:5]
    
    hp = adam_stats.get('health', 100)
    hunger = adam_stats.get('hunger', 0)
    energy = adam_stats.get('energy', 100)
    
    hp_bar = health_bar(hp, 100, 8)
    
    # Time display
    hour = world_state.get('time_of_day', 12)
    time_icon = "\u2600\ufe0f" if 6 <= hour < 18 else "\U0001f319"
    
    # Action with color
    action_colors = {
        "EXPLORE": "\033[96m", "EAT": "\033[92m", "DRINK": "\033[94m",
        "SLEEP": "\033[35m", "HIDE": "\033[33m", "MOVE": "\033[36m",
        "FLEE": "\033[91m", "IDLE": "\033[90m",
    }
    ac = action_colors.get(action, "")
    reset = "\033[0m"
    
    # Reward color
    if reward > 0.5:
        r_color = "\033[92m"
    elif reward < -0.5:
        r_color = "\033[91m"
    else:
        r_color = ""
    
    # Hunger/energy indicators
    hunger_icon = "\U0001f356" if hunger < 30 else ("\U0001f374" if hunger < 60 else "\U0001f922")
    energy_icon = "\u26a1" if energy > 50 else ("\U0001f614" if energy > 20 else "\U0001f634")
    
    thought_short = thought[:30] + ".." if len(thought) > 32 else thought
    
    print(
        f"  {tick:>5} | {biome_emoji}{biome_name:<6} {weather_emoji}{weather_name:<5} {time_icon} "
        f"| {ac}{action:<7}{reset} | {r_color}{reward:>+5.2f}{reset} "
        f"| HP:{hp_bar}{hp:>3.0f} {hunger_icon}{hunger:>3.0f} {energy_icon}{energy:>3.0f} "
        f"| \033[3m{thought_short}\033[0m",
        flush=True
    )


def full_stats_display(tick, world_state, adam_stats, action, reward,
                       thought_engine, curiosity, dream_engine, total_reward,
                       action_counts, env, model, episode_intrinsic_reward):
    """Show detailed stats display periodically."""
    biome_name = world_state.get('biome', '?')
    weather_name = world_state.get('weather', '?')
    biome_data = BIOMES.get(biome_name, {})
    weather_data = WEATHER_TYPES.get(weather_name, {})
    
    hp = adam_stats.get('health', 100)
    hunger = adam_stats.get('hunger', 0)
    energy = adam_stats.get('energy', 100)
    stress = adam_stats.get('stress', 0)
    pain = adam_stats.get('pain', 0)
    
    personality = thought_engine.get_personality()
    vocab = thought_engine.get_vocabulary()
    discovered = thought_engine.get_discovered_vocabulary()
    cs = curiosity.get_curiosity_stats()
    ds = dream_engine.get_dream_stats()
    
    total_actions = sum(action_counts.values())
    
    print(f"\n{'=' * 70}", flush=True)
    print(f"  \U0001f4ca FULL STATS — Tick {tick} | Total Reward: {total_reward:.2f}", flush=True)
    print(f"{'─' * 70}", flush=True)
    
    # Location
    x = world_state.get('adam_x', 0)
    y = world_state.get('adam_y', 0)
    facing = world_state.get('facing', '?')
    print(f"  \U0001f5fa Location: ({x},{y}) facing {facing} | "
          f"{biome_data.get('emoji', '?')} {biome_name} | "
          f"{weather_data.get('emoji', '?')} {weather_name}", flush=True)
    
    # Vital signs
    print(f"  \U0001f493 Vitals:  HP {health_bar(hp, 100, 15)} {hp:.0f}/100", flush=True)
    print(f"  \U0001f356 Hunger:  {health_bar(100 - hunger, 100, 15)} {hunger:.0f}/100", flush=True)
    print(f"  \u26a1 Energy:  {health_bar(energy, 100, 15)} {energy:.0f}/100", flush=True)
    print(f"  \U0001f630 Stress:  {health_bar(stress, 100, 15)} {stress:.0f}/100", flush=True)
    print(f"  \U0001f4a5 Pain:    {pain:.1f}/10", flush=True)
    
    # Personality
    print(f"  \U0001f464 Personality: {personality['disposition']} | "
          f"Fears:{personality['fear_triggers']} Joys:{personality['good_memories']} "
          f"Patterns:{personality['patterns_learned']}", flush=True)
    
    # Action distribution
    if total_actions > 0:
        dist_parts = []
        for a in WorldSim.ACTIONS:
            cnt = action_counts.get(a, 0)
            pct = cnt / total_actions * 100
            if pct > 0:
                dist_parts.append(f"{a}:{pct:.0f}%")
        print(f"  \U0001f3b2 Actions: {' '.join(dist_parts)}", flush=True)
    
    # Vocabulary
    if discovered:
        print(f"  \U0001f4da Words discovered: {discovered}", flush=True)
    print(f"  \U0001f4d6 Total vocab: {len(vocab)} words | Thoughts: {thought_engine.thought_gen.total_thoughts}", flush=True)
    
    # Curiosity & Dreams
    print(f"  \U0001f50d Curiosity: {cs['total_states_discovered']} states "
          f"({cs['novel_states']} novel, {cs['familiar_states']} familiar) | "
          f"Intrinsic: {episode_intrinsic_reward:.2f}", flush=True)
    print(f"  \U0001f4a4 Dreams: {ds['total_dreams']} ({ds['nightmares']} nightmares, {ds['peaceful_dreams']} peaceful)", flush=True)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  \U0001f9e0 Brain: {total_params} params | Input dim: {INPUT_DIM} | Hidden: {HIDDEN_DIM}", flush=True)
    
    print(f"{'=' * 70}\n", flush=True)


def birth_display(env, world_state):
    """Show the birth/spawn screen."""
    biome_name = world_state.get('biome', '?')
    biome_data = BIOMES.get(biome_name, {})
    weather_name = world_state.get('weather', '?')
    weather_data = WEATHER_TYPES.get(weather_name, {})
    x = world_state.get('adam_x', 0)
    y = world_state.get('adam_y', 0)
    hour = world_state.get('time_of_day', 12)
    
    print(f"\n{'=' * 70}", flush=True)
    print(f"  \U0001f476 ADAM IS BORN", flush=True)
    print(f"{'─' * 70}", flush=True)
    print(f"  He opens his eyes for the first time.", flush=True)
    print(f"  He knows nothing. Not even his own name.", flush=True)
    print(f"  He will learn to think. He will learn to feel.", flush=True)
    print(f"  He will name things. He will dream.", flush=True)
    print(f"  He has ONE life. When he dies, it is over.", flush=True)
    print(f"{'─' * 70}", flush=True)
    print(f"  {biome_data.get('emoji', '?')} Born in: {biome_name}", flush=True)
    print(f"  {biome_data.get('desc', '')}", flush=True)
    print(f"  {weather_data.get('emoji', '?')} Weather: {weather_name} — {weather_data.get('desc', '')}", flush=True)
    print(f"  \U0001f5fa Position: ({x}, {y}) in a {env.world_map.width}x{env.world_map.height} world", flush=True)
    print(f"  \U0001f552 Time: {hour}:00", flush=True)
    print(f"{'=' * 70}\n", flush=True)


def death_display(tick, total_reward, adam_stats, thought_engine, curiosity,
                  dream_engine, action_counts, world_state):
    """Show the death screen — Adam's life summary."""
    biome_name = world_state.get('biome', '?')
    biome_data = BIOMES.get(biome_name, {})
    total_actions = sum(action_counts.values())
    
    print(f"\n{'=' * 70}", flush=True)
    print(f"  \U0001f480 ADAM HAS DIED", flush=True)
    print(f"{'─' * 70}", flush=True)
    print(f"  He lived for {tick} ticks.", flush=True)
    print(f"  He accumulated {total_reward:.2f} total reward.", flush=True)
    print(f"  He died in {biome_data.get('emoji', '?')} {biome_name}.", flush=True)
    print(f"{'─' * 70}", flush=True)
    
    # Final stats
    personality = thought_engine.get_personality()
    vocab = thought_engine.get_vocabulary()
    discovered = thought_engine.get_discovered_vocabulary()
    cs = curiosity.get_curiosity_stats()
    ds = dream_engine.get_dream_stats()
    
    print(f"  \U0001f464 Final Personality: {personality['disposition']}", flush=True)
    print(f"     Fears: {personality['fear_triggers']} | Joys: {personality['good_memories']}", flush=True)
    print(f"     Patterns learned: {personality['patterns_learned']}", flush=True)
    
    if discovered:
        print(f"  \U0001f4da Words discovered in life: {discovered}", flush=True)
    print(f"  \U0001f4d6 Total vocabulary: {len(vocab)} words", flush=True)
    print(f"  \U0001f50d States explored: {cs['total_states_discovered']} "
          f"({cs['novel_states']} novel, {cs['familiar_states']} familiar)", flush=True)
    print(f"  \U0001f4a4 Dreams: {ds['total_dreams']} ({ds['nightmares']} nightmares)", flush=True)
    print(f"  \U0001f3b2 Action distribution:", flush=True)
    for a in WorldSim.ACTIONS:
        cnt = action_counts.get(a, 0)
        pct = cnt / max(total_actions, 1) * 100
        bar = '#' * int(pct / 2)
        print(f"     {a:>8}: {bar} {pct:.1f}%", flush=True)
    
    print(f"{'─' * 70}", flush=True)
    print(f"  There is no afterlife. There is no save file.", flush=True)
    print(f"  Run the program again to birth a new Adam.", flush=True)
    print(f"{'=' * 70}\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Training Loop — ONE LIFE
# ═══════════════════════════════════════════════════════════════════════════════

def run_life():
    """
    Adam lives once. This is his entire life.
    No episodes. No restarts. One continuous experience.
    """
    DEVICE = torch.device("cpu")
    
    # Initialize the brain
    model = BabyBrain(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Initialize the world
    env = WorldSim()
    world_state, adam_stats_dict = env.reset()
    
    # Show birth screen
    birth_display(env, world_state)
    
    # Initialize all the Phase modules
    thought_engine = ThoughtEngine(memory_size=10)
    curiosity = CuriosityModule(
        curiosity_bonus=CURIOSITY_BONUS,
        curiosity_decay=CURIOSITY_DECAY,
        min_curiosity=CURIOSITY_MIN,
    )
    dream_engine = DreamEngine()
    tb_logger = TBLogger(log_dir="runs/nafs_single_life")
    
    # Encode initial observation
    init_phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)
    sensory_input = encode_sensory_input(
        world_state, adam_stats_dict,
        fear_signal=init_phase5['fear_signal'],
        pleasure_signal=init_phase5['pleasure_signal'],
        pattern_confidence=init_phase5['pattern_confidence'],
    ).to(DEVICE)
    hidden_state = model.init_hidden(1).to(DEVICE)
    
    # Tracking variables
    action_history = []
    all_action_counts = {a: 0 for a in WorldSim.ACTIONS}
    total_reward = 0.0
    tick = 0
    episode_intrinsic_reward = 0.0
    recent_actions = []
    
    # PPO buffer
    obs_list, actions_list, old_log_probs = [], [], []
    rewards_list, values_list, masks_list = [], [], []
    
    # Latest thought/reflection for display
    latest_thought = "quiet. still."
    latest_emotion = "uncertain"
    latest_reflection = None
    latest_dream = None
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    life_start = time.time()
    
    # ── Adam's single life begins ──────────────────────────────────────────
    alive = True
    
    while alive:
        try:
            tick += 1
            
            # Save previous stats for outcome comparison
            prev_stats = copy.deepcopy(adam_stats_dict)
            
            # Phase 5: Compute fear/pleasure/pattern signals
            phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)
            
            # Get action from brain
            with torch.no_grad():
                obs_list.append(sensory_input.clone())
                action_logits, state_value, hidden_state = model(
                    sensory_input.unsqueeze(0), hidden_state
                )
                action_logits_sq = action_logits.squeeze(0)
                state_value_sq = state_value.squeeze()
                action_dist = torch.distributions.Categorical(logits=action_logits_sq)
                action_idx = action_dist.sample()
                action = WorldSim.ACTIONS[action_idx.item()]
            
            # Step the world
            next_world_state, next_adam_stats_dict, reward, done = env.step(action)
            
            # Curiosity intrinsic reward
            intrinsic_reward = curiosity.compute_intrinsic_reward(
                world_state, adam_stats_dict
            )
            reward += intrinsic_reward
            episode_intrinsic_reward += intrinsic_reward
            
            # Action history for diversity penalty
            action_history.append(action_idx.item())
            if len(action_history) > 10:
                action_history.pop(0)
            if len(action_history) >= DIVERSITY_WINDOW:
                if len(set(action_history[-DIVERSITY_WINDOW:])) == 1:
                    reward -= DIVERSITY_PENALTY
            
            # Phase 6: Reflection feedback
            suggested = phase5.get('suggested_action')
            confidence = phase5.get('pattern_confidence', 0)
            if suggested and confidence >= PATTERN_CONFIDENCE_THRESHOLD:
                if action == suggested:
                    reward += REFLECTION_FOLLOW_BONUS
                else:
                    reward -= REFLECTION_IGNORE_PENALTY
            
            # Track action counts
            all_action_counts[action] = all_action_counts.get(action, 0) + 1
            actions_list.append(action_idx.item())
            old_log_probs.append(action_dist.log_prob(action_idx).item())
            rewards_list.append(reward)
            values_list.append(state_value_sq.item())
            masks_list.append(1.0 - done)
            
            # Phase 1: Generate inner experience (thought + emotion)
            experience = thought_engine.experience(
                world_state=next_world_state,
                adam_stats=next_adam_stats_dict,
                action=action,
                prev_stats=prev_stats,
                tick=tick,
                reward=reward,
            )
            latest_thought = experience.get('thought', 'quiet. still.')
            latest_emotion = experience.get('emotion', 'uncertain')
            
            # Phase 2: Log vocabulary discoveries
            if "new_words" in experience:
                for word, meaning in experience["new_words"]:
                    print(f"  \U0001f4dd NEW WORD: \"{word}\" = {meaning}", flush=True)
            
            # Phase 6: Reflection every N ticks
            recent_actions.append(action)
            if tick % REFLECTION_INTERVAL == 0 and tick > 0:
                reflection = thought_engine.reflect(
                    world_state, adam_stats_dict,
                    recent_actions=recent_actions
                )
                if reflection.get('has_reflection'):
                    latest_reflection = reflection
            
            # Dreaming: Memory consolidation during SLEEP
            if action == "SLEEP" and tick > 1:
                dream = dream_engine.dream(
                    thought_engine.persistent_memory,
                    thought_engine.memory
                )
                if dream.get('dream_type') != 'empty':
                    latest_dream = dream
                    dream_thoughts = ' '.join(dream.get('thoughts', []))
                    print(f"  \U0001f4a4 Dream ({dream['dream_type']}): {dream_thoughts}", flush=True)
            
            # Encode next observation with fear/pleasure/pattern signals
            next_phase5 = thought_engine.get_phase5_signals(next_world_state, next_adam_stats_dict)
            sensory_input = encode_sensory_input(
                next_world_state, next_adam_stats_dict,
                fear_signal=next_phase5['fear_signal'],
                pleasure_signal=next_phase5['pleasure_signal'],
                pattern_confidence=next_phase5['pattern_confidence'],
            ).to(DEVICE)
            
            total_reward += reward
            
            # ── Display ────────────────────────────────────────────────
            biome_name = world_state.get('biome', '?')
            biome_data = BIOMES.get(biome_name, {})
            weather_name = world_state.get('weather', '?')
            weather_data = WEATHER_TYPES.get(weather_name, {})
            
            # Show reflection if available
            if latest_reflection and latest_reflection.get('has_reflection'):
                print(f"  \U0001fa9e Reflect: \"{latest_reflection['reflection']}\" ({latest_reflection['personality']})", flush=True)
                latest_reflection = None
            
            # Compact tick display
            compact_tick_display(
                tick, world_state, adam_stats_dict, action, reward,
                latest_thought, latest_emotion, biome_data, weather_data
            )
            
            # Full stats display periodically
            if tick % FULL_DISPLAY_EVERY == 0:
                full_stats_display(
                    tick, world_state, adam_stats_dict, action, reward,
                    thought_engine, curiosity, dream_engine, total_reward,
                    all_action_counts, env, model, episode_intrinsic_reward
                )
            
            # ── PPO Update (online learning during life) ─────────────
            if tick % PPO_UPDATE_INTERVAL == 0 and len(obs_list) > 0:
                # Compute GAE
                R = 0.0
                if not done:
                    with torch.no_grad():
                        _, last_val, _ = model(sensory_input.unsqueeze(0), hidden_state)
                        R = last_val.item()
                
                advantages = []
                gae = 0.0
                next_values = values_list[1:] + [R]
                for i in reversed(range(len(rewards_list))):
                    delta = rewards_list[i] + GAMMA * next_values[i] * masks_list[i] - values_list[i]
                    gae = delta + GAMMA * GAE_LAMBDA * masks_list[i] * gae
                    advantages.insert(0, gae)
                
                advantages_t = torch.FloatTensor(advantages)
                returns_t = advantages_t + torch.FloatTensor(values_list)
                if len(advantages_t) > 1:
                    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
                
                old_log_probs_t = torch.FloatTensor(old_log_probs)
                actions_t = torch.LongTensor(actions_list)
                
                # PPO Update with Sequential GRU
                new_log_probs, new_entropies, new_values = [], [], []
                h = model.init_hidden(1).to(DEVICE)
                for i in range(len(obs_list)):
                    a_logits, s_value, h = model(obs_list[i].unsqueeze(0), h.detach())
                    a_logits_sq = a_logits.squeeze(0)
                    s_value_sq = s_value.squeeze()
                    dist = torch.distributions.Categorical(logits=a_logits_sq)
                    new_log_probs.append(dist.log_prob(actions_t[i]))
                    new_entropies.append(dist.entropy())
                    new_values.append(s_value_sq)
                
                new_log_probs_t = torch.stack(new_log_probs)
                new_entropies_t = torch.stack(new_entropies)
                new_values_t = torch.stack(new_values)
                
                ratio = torch.exp(new_log_probs_t - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values_t, returns_t)
                entropy_loss = -ENTROPY_COEF * new_entropies_t.mean()
                loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                # Clear the PPO buffer after update
                obs_list, actions_list, old_log_probs = [], [], []
                rewards_list, values_list, masks_list = [], [], []
                
                # TensorBoard logging
                if tick % TB_LOG_INTERVAL == 0:
                    total_actions = sum(all_action_counts.values())
                    action_dist = {}
                    for a in WorldSim.ACTIONS:
                        cnt = all_action_counts.get(a, 0)
                        pct = cnt / max(total_actions, 1) * 100
                        action_dist[a] = pct
                    
                    personality = thought_engine.get_personality()
                    max_pct = max(cnt / max(total_actions, 1) for cnt in all_action_counts.values()) if total_actions > 0 else 0
                    
                    tb_data = {
                        "reward": total_reward / max(tick, 1),
                        "total_reward": total_reward,
                        "tick": tick,
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": new_entropies_t.mean().item(),
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                        "action_dist": action_dist,
                        "dominant_pct": max_pct * 100,
                        "vocabulary_size": len(thought_engine.get_vocabulary()),
                        "discovered_words_count": len(discovered) if (discovered := thought_engine.get_discovered_vocabulary()) else 0,
                        "fear_triggers_count": personality['fear_triggers'],
                        "good_memories_count": personality['good_memories'],
                        "patterns_count": personality['patterns_learned'],
                        "curiosity_intrinsic_reward": episode_intrinsic_reward,
                        "curiosity_states_discovered": cs['total_states_discovered'] if (cs := curiosity.get_curiosity_stats()) else 0,
                        "dream_count": ds['total_dreams'] if (ds := dream_engine.get_dream_stats()) else 0,
                        "nightmare_count": ds['nightmares'] if (ds := dream_engine.get_dream_stats()) else 0,
                        "personality_disposition": personality['disposition'],
                    }
                    tb_logger.log_episode(tick, tb_data)
            
            # Save checkpoint periodically
            if tick % CHECKPOINT_INTERVAL == 0:
                model_path = os.path.join(MODEL_DIR, f"adam_tick{tick}.pt")
                torch.save({
                    'tick': tick, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_reward': total_reward,
                    'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM,
                    'num_actions': NUM_ACTIONS,
                    'entropy_coef': ENTROPY_COEF,
                    'diversity_penalty': DIVERSITY_PENALTY,
                }, model_path)
                print(f"  \U0001f4be Saved {model_path}", flush=True)
            
            # Check for death
            if done:
                alive = False
                break
            
            # Update state for next tick
            world_state = next_world_state
            adam_stats_dict = next_adam_stats_dict
            
            # Tick delay for readability
            if TICK_DELAY > 0:
                time.sleep(TICK_DELAY)
        
        except KeyboardInterrupt:
            print(f"\n  \u26a0\ufe0f Program interrupted by user at tick {tick}.", flush=True)
            alive = False
            break
        except Exception as e:
            print(f"\n  \u274c ERROR at tick {tick}: {e}", flush=True)
            traceback.print_exc()
            print(f"  Continuing...", flush=True)
            continue
    
    # ── Death ──────────────────────────────────────────────────────────────
    death_display(
        tick, total_reward, adam_stats_dict, thought_engine, curiosity,
        dream_engine, all_action_counts, world_state
    )
    
    # Save final model
    final_path = os.path.join(MODEL_DIR, "adam_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM,
        'num_actions': NUM_ACTIONS,
        'actions_list': WorldSim.ACTIONS,
        'ticks_lived': tick, 'total_reward': total_reward,
        'action_counts': all_action_counts,
    }, final_path)
    print(f"  Final brain saved: {final_path}", flush=True)
    
    # Save life log
    elapsed_min = (time.time() - life_start) / 60
    log_data = {
        "mode": "single_life",
        "ticks_lived": tick,
        "total_reward": total_reward,
        "real_time_min": elapsed_min,
        "action_counts": all_action_counts,
        "vocabulary": thought_engine.get_vocabulary(),
        "total_thoughts": thought_engine.thought_gen.total_thoughts,
        "curiosity_stats": curiosity.get_curiosity_stats(),
        "dream_stats": dream_engine.get_dream_stats(),
    }
    with open("life_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    # NOTE: We do NOT save memory.json — Adam is dead. No memory persists.
    # The philosophical constraint: when Adam dies, everything dies with him.
    # Running the program again creates a completely new Adam in a new world.
    
    # Close TensorBoard
    tb_logger.close()


if __name__ == '__main__':
    run_life()
