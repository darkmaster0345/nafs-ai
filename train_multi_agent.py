"""
Nafs AI — Multi-Agent Training Loop (v0.3)
==========================================

Runs Adam AND Eve in the same world, in a single process, with interleaved ticks.

Architecture:
  - One WorldSim (shared world map, weather, time of day)
  - Two completely separate agents:
      * Adam: uses WorldSim.step() (legacy path), has his own BabyBrain, ThoughtEngine,
              CuriosityModule, DreamEngine, LearnedThinker
      * Eve:  uses EveAgent.step() (her own movement), has her own BabyBrain, ThoughtEngine,
              CuriosityModule, DreamEngine, LearnedThinker
  - Each agent has separate memory, personality, vocabulary, fears, dreams
  - Each agent trains its own PPO brain independently
  - Each agent trains its own LearnedThinker (tiny transformer) independently
  - They share NOTHING except the world map and weather

Mutual sensing:
  - Eve can see Adam if he's within 3 tiles (other_presence 0..1, other_direction)
  - Adam can see Eve if she's within 3 tiles (injected into his world_state dict)
  - Neither can read the other's thoughts — only physical presence is shared

v0.3 modes:
  --mode multi            Both agents, rule-based thoughts blended with learned (default)
  --mode multi-learned    Both agents, FULL cutover to learned transformer once ready
                          (rule-based only used to seed training data, then disabled)

Usage:
    python train_multi_agent.py
    python train_multi_agent.py --learned-only
    python train_multi_agent.py --max-ticks 5000

Output files (in nafs-ai/):
    vocab_divergence.jsonl    — append-only vocab divergence log
    life_log_adam.json        — Adam's final life summary
    life_log_eve.json         — Eve's final life summary
    checkpoints/adam_*.pt     — Adam's model checkpoints
    checkpoints/eve_*.pt      — Eve's model checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import os
import sys
import time
import json
import traceback
import argparse

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensory_encoder import encode_sensory_input, INPUT_DIM
from sensory_encoder_multi import encode_sensory_input_multi, INPUT_DIM_MULTI
from baby_brain_model import BabyBrain
from world_sim import WorldSim, BIOMES, WEATHER_TYPES
from thought_engine import ThoughtEngine
from curiosity import CuriosityModule
from dreaming import DreamEngine
from tb_logger import TBLogger
from ws_bridge import WSBridge
from eve_agent import EveAgent
from learned_thinking import LearnedThinker, blend_thoughts
from vocab_divergence import VocabDivergenceLogger
from train import (
    HIDDEN_DIM, NUM_ACTIONS, GAMMA, GAE_LAMBDA, CLIP_EPSILON,
    LEARNING_RATE, VALUE_LOSS_COEF, MAX_GRAD_NORM, ENTROPY_COEF,
    DIVERSITY_PENALTY, DIVERSITY_WINDOW, PPO_UPDATE_INTERVAL,
    TICK_DELAY, FULL_DISPLAY_EVERY, REFLECTION_INTERVAL,
    REFLECTION_FOLLOW_BONUS, REFLECTION_IGNORE_PENALTY,
    PATTERN_CONFIDENCE_THRESHOLD, CURIOSITY_BONUS, CURIOSITY_DECAY, CURIOSITY_MIN,
    CHECKPOINT_INTERVAL, MODEL_DIR, TB_LOG_INTERVAL,
    health_bar,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SIGHT_RANGE = 3  # Tiles — how far agents can "see" each other


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_other_presence(ax, ay, bx, by):
    """Compute presence signal (0..1) and direction from A to B."""
    dx = bx - ax
    dy = by - ay
    dist = abs(dx) + abs(dy)
    if dist > SIGHT_RANGE:
        return 0.0, "none"
    presence = 1.0 - (dist / float(SIGHT_RANGE + 1))
    if abs(dx) > abs(dy):
        direction = "east" if dx > 0 else "west"
    elif dy != 0:
        direction = "south" if dy > 0 else "north"
    else:
        direction = "here"  # same tile
    return presence, direction


def _inject_other_into_world_state(world_state, other_x, other_y, self_x, self_y):
    """Inject other agent's presence into Adam's world_state dict."""
    presence, direction = _compute_other_presence(self_x, self_y, other_x, other_y)
    ws = dict(world_state)  # shallow copy
    ws['other_presence'] = presence
    ws['other_direction'] = direction
    return ws


# ═══════════════════════════════════════════════════════════════════════════════
# PPO update helper (used for both agents)
# ═══════════════════════════════════════════════════════════════════════════════

def ppo_update(
    model, optimizer, device,
    obs_list, actions_list, old_log_probs,
    rewards_list, values_list, masks_list,
    last_sensory, last_hidden,
):
    """Perform a PPO update. Returns dict of loss metrics."""
    if len(obs_list) == 0:
        return None

    # Bootstrap value
    R = 0.0
    with torch.no_grad():
        _, last_val, _ = model(last_sensory.unsqueeze(0), last_hidden)
        R = last_val.item()

    # GAE
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

    # Recompute log probs with current policy (sequential GRU)
    new_log_probs, new_entropies, new_values = [], [], []
    h = model.init_hidden(1).to(device)
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

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": new_entropies_t.mean().item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Agent runtime container — bundles per-agent state
# ═══════════════════════════════════════════════════════════════════════════════

class AgentRuntime:
    """Bundles all per-agent state for the multi-agent loop."""

    def __init__(self, name: str, device, is_adam: bool = False,
                 first_contact: bool = False):
        self.name = name
        self.device = device
        self.is_adam = is_adam
        self.first_contact = first_contact

        # v1.0: In first-contact mode, use 23-dim input so PPO can observe
        # the other agent's last action
        input_dim = INPUT_DIM_MULTI if first_contact else INPUT_DIM
        self.input_dim = input_dim

        # Brain
        self.model = BabyBrain(input_dim, HIDDEN_DIM, NUM_ACTIONS).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.hidden_state = None

        # Consciousness modules
        self.thought_engine = ThoughtEngine(memory_size=10)
        self.curiosity = CuriosityModule(
            curiosity_bonus=CURIOSITY_BONUS,
            curiosity_decay=CURIOSITY_DECAY,
            min_curiosity=CURIOSITY_MIN,
        )
        self.dream_engine = DreamEngine()

        # Learned thinker (tiny transformer)
        self.learned_thinker = LearnedThinker(
            sensory_dim=input_dim, device=str(device)
        )
        self.learned_thinker.start()

        # State
        self.alive = True
        self.tick = 0
        self.total_reward = 0.0
        self.sensory_input = None
        self.action_counts = {a: 0 for a in WorldSim.ACTIONS}
        self.action_history = []
        self.recent_actions = []
        self.episode_intrinsic_reward = 0.0
        self.death_cause = None
        self.last_action_idx = -1  # v1.0: for other agent to observe

        # Latest experience
        self.latest_thought = "quiet. still."
        self.latest_emotion = "uncertain"
        self.latest_reflection = None
        self.latest_dream = None

        # PPO buffer
        self.obs_list = []
        self.actions_list = []
        self.old_log_probs = []
        self.rewards_list = []
        self.values_list = []
        self.masks_list = []

        # Position (for Adam, synced with env; for Eve, owned by EveAgent)
        self.x = 0
        self.y = 0
        self.stats = {}

    def init_hidden(self):
        self.hidden_state = self.model.init_hidden(1).to(self.device)

    def clear_ppo_buffer(self):
        self.obs_list = []
        self.actions_list = []
        self.old_log_probs = []
        self.rewards_list = []
        self.values_list = []
        self.masks_list = []


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

def multi_agent_compact_display(tick, adam_rt, eve_rt, adam_action, eve_action,
                                  adam_reward, eve_reward, world_state, env):
    """One-line display for both agents."""
    biome = world_state.get('biome', '?')[:6]
    weather = world_state.get('weather', '?')[:5]

    def _vitals_str(rt):
        s = rt.stats
        return (f"HP{int(s.get('health', 100)):>3} "
                f"H{int(s.get('hunger', 0)):>3} "
                f"E{int(s.get('energy', 100)):>3}")

    def _pos_str(rt):
        return f"({rt.x:>2},{rt.y:>2})"

    # Distance between agents
    dist = abs(adam_rt.x - eve_rt.x) + abs(adam_rt.y - eve_rt.y)
    if dist <= SIGHT_RANGE:
        dist_str = f"\033[93m\U0001f441 dist={dist}\033[0m"
    else:
        dist_str = f"dist={dist}"

    print(
        f"  t{tick:>4} [{biome:>6}/{weather:>5}] "
        f"\033[96mAdam\033[0m {_vitals_str(adam_rt)} {_pos_str(adam_rt)} "
        f"\033[97m{adam_action:<7}\033[0m r={adam_reward:+.2f} "
        f"| \033[95mEve\033[0m  {_vitals_str(eve_rt)} {_pos_str(eve_rt)} "
        f"\033[97m{eve_action:<7}\033[0m r={eve_reward:+.2f} "
        f"| {dist_str}",
        flush=True,
    )


def multi_agent_full_display(tick, adam_rt, eve_rt, env, vocab_logger):
    """Full stats display every N ticks."""
    print(f"\n{'=' * 78}", flush=True)
    print(f"  \U0001f5fa TICK {tick} — Adam & Eve in {env.world_map.width}x{env.world_map.height} world",
          flush=True)
    print(f"{'─' * 78}", flush=True)

    for rt, label in [(adam_rt, "Adam"), (eve_rt, "Eve")]:
        personality = rt.thought_engine.get_personality()
        cs = rt.curiosity.get_curiosity_stats()
        ds = rt.dream_engine.get_dream_stats()
        lt = rt.learned_thinker.get_stats()
        vocab = rt.thought_engine.get_vocabulary()

        print(f"  \033[96m{label}\033[0m at ({rt.x}, {rt.y}) — {rt.action_counts}",
              flush=True)
        print(f"     reward={rt.total_reward:.2f} | HP={rt.stats.get('health', 0):.0f} "
              f"hunger={rt.stats.get('hunger', 0):.0f} energy={rt.stats.get('energy', 0):.0f} "
              f"stress={rt.stats.get('stress', 0):.0f}", flush=True)
        print(f"     personality: {personality['disposition']} "
              f"(fears={personality['fear_triggers']}, joys={personality['good_memories']}, "
              f"patterns={personality['patterns_learned']})", flush=True)
        print(f"     vocab: {len(vocab)} words | curiosity: {cs['total_states_discovered']} states | "
              f"dreams: {ds['total_dreams']} ({ds['nightmares']} nightmares)", flush=True)
        print(f"     learned thinker: {lt['train_steps']} steps, loss={lt['avg_loss']:.4f}, "
              f"confidence={lt['confidence']:.3f}, ready={lt['ready']}", flush=True)
        print(f"     latest thought: \"{rt.latest_thought}\" ({rt.latest_emotion})",
              flush=True)

    # Vocab divergence
    latest = vocab_logger.get_latest()
    if latest:
        print(f"{'─' * 78}", flush=True)
        print(f"  \U0001f4da VOCAB DIVERGENCE: Adam={latest['adam_vocab_size']} "
              f"Eve={latest['eve_vocab_size']} shared={latest['shared_count']} "
              f"jaccard={latest['jaccard']:.3f}", flush=True)
    print(f"{'=' * 78}\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main multi-agent life loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_agent_life(
    learned_only: bool = False,
    max_ticks: int = None,
    tick_delay: float = TICK_DELAY,
    first_contact: bool = False,
):
    """
    Run Adam and Eve in the same world simultaneously.

    Args:
        learned_only: If True, once a LearnedThinker is ready, use ONLY its
                      thoughts (no rule-based blending). This is the v0.3 cutover.
        max_ticks: Optional cap on total ticks (for testing). None = run until death.
        tick_delay: Seconds between ticks.
        first_contact: v1.0 mode — extend sensory input to 23 dims so each agent's
                       PPO can observe the other's last action. This enables true
                       interaction learning (Eve can learn to react to Adam).
    """
    DEVICE = torch.device("cpu")

    print(f"\n{'=' * 78}", flush=True)
    print(f"  \U0001f476\U0001f9dd ADAM & EVE — MULTI-AGENT LIFE ({'v1.0' if first_contact else 'v0.3'})",
          flush=True)
    print(f"{'─' * 78}", flush=True)
    print(f"  Mode: {'LEARNED-ONLY (full cutover)' if learned_only else 'BLENDED (rule + learned)'}",
          flush=True)
    if first_contact:
        print(f"  \U0001f441 FIRST CONTACT: 23-dim sensory input (each agent observes other's action)",
              flush=True)
    print(f"  World: shared | Memory: separate | Brains: separate | Vocab: separate",
          flush=True)
    print(f"  Sight range: {SIGHT_RANGE} tiles (they can sense each other's presence)",
          flush=True)
    print(f"{'=' * 78}\n", flush=True)

    # ── Init world ─────────────────────────────────────────────────────────
    env = WorldSim()
    world_state, adam_stats = env.reset()

    # ── Init Adam runtime ──────────────────────────────────────────────────
    adam = AgentRuntime("Adam", DEVICE, is_adam=True, first_contact=first_contact)
    adam.init_hidden()
    adam.stats = adam_stats
    adam.x = env.adam_x
    adam.y = env.adam_y

    # Adam's initial sensory input
    init_phase5 = adam.thought_engine.get_phase5_signals(world_state, adam_stats)
    if first_contact:
        adam.sensory_input = encode_sensory_input_multi(
            world_state, adam_stats,
            fear_signal=init_phase5['fear_signal'],
            pleasure_signal=init_phase5['pleasure_signal'],
            pattern_confidence=init_phase5['pattern_confidence'],
            other_presence=0.0,
            other_last_action_idx=-1,
        ).to(DEVICE)
    else:
        adam.sensory_input = encode_sensory_input(
            world_state, adam_stats,
            fear_signal=init_phase5['fear_signal'],
            pleasure_signal=init_phase5['pleasure_signal'],
            pattern_confidence=init_phase5['pattern_confidence'],
        ).to(DEVICE)

    # ── Init Eve ───────────────────────────────────────────────────────────
    eve_agent = EveAgent(env, device=str(DEVICE), name="Eve", first_contact=first_contact)
    eve_agent.birth(adam_x=env.adam_x, adam_y=env.adam_y)

    # Wrap Eve in an AgentRuntime too, for uniform PPO handling
    eve = AgentRuntime("Eve", DEVICE, is_adam=False, first_contact=first_contact)
    eve.model = eve_agent.model
    eve.optimizer = eve_agent.optimizer
    eve.thought_engine = eve_agent.thought_engine
    eve.curiosity = eve_agent.curiosity
    eve.dream_engine = eve_agent.dream_engine
    eve.learned_thinker = eve_agent.learned_thinker
    eve.init_hidden()
    eve.sensory_input = eve_agent.sensory_input
    eve.x = eve_agent.x
    eve.y = eve_agent.y
    eve.stats = eve_agent.stats

    print(f"  \U0001f476 Adam born at ({adam.x}, {adam.y}) in {world_state.get('biome', '?')}",
          flush=True)
    print(f"  \U0001f9dd Eve  born at ({eve.x}, {eve.y}) in {eve_agent.current_biome}",
          flush=True)
    dist = abs(adam.x - eve.x) + abs(adam.y - eve.y)
    print(f"  \U0001f4cf Distance: {dist} tiles\n", flush=True)

    # ── Init loggers ───────────────────────────────────────────────────────
    vocab_logger = VocabDivergenceLogger(
        log_path="vocab_divergence.jsonl",
        word_log_path="vocab_log.jsonl",
        convergence_log_path="vocab_convergence.jsonl",
        log_interval=50,
    )
    tb_logger = TBLogger(log_dir="runs/nafs_multi_agent")
    ws_bridge = WSBridge()
    ws_bridge.start()

    # Send birth events
    ws_bridge.send_birth_data({
        "agent": "adam",
        "biome": world_state.get('biome', 'plains'),
        "weather": world_state.get('weather', 'clear'),
        "position": [adam.x, adam.y],
        "time_of_day": world_state.get('time_of_day', 12),
        "world_size": [env.world_map.width, env.world_map.height],
    })
    ws_bridge.send_birth_data({
        "agent": "eve",
        "biome": eve_agent.current_biome,
        "weather": world_state.get('weather', 'clear'),
        "position": [eve.x, eve.y],
        "time_of_day": world_state.get('time_of_day', 12),
        "world_size": [env.world_map.width, env.world_map.height],
    })

    os.makedirs(MODEL_DIR, exist_ok=True)
    life_start = time.time()
    tick = 0

    # ── Main loop ──────────────────────────────────────────────────────────
    try:
        while adam.alive and eve.alive:
            try:
                tick += 1
                if max_ticks and tick > max_ticks:
                    print(f"\n  \u26a0\ufe0f Reached max_ticks={max_ticks}, stopping.", flush=True)
                    break

                prev_adam_stats = copy.deepcopy(adam.stats)

                # ═══ ADAM'S TURN ═══
                adam_phase5 = adam.thought_engine.get_phase5_signals(world_state, adam.stats)

                with torch.no_grad():
                    adam.obs_list.append(adam.sensory_input.clone())
                    a_logits, a_value, adam.hidden_state = adam.model(
                        adam.sensory_input.unsqueeze(0), adam.hidden_state
                    )
                    a_logits_sq = a_logits.squeeze(0)
                    a_value_sq = a_value.squeeze()
                    a_dist = torch.distributions.Categorical(logits=a_logits_sq)
                    adam_action_idx = a_dist.sample()
                    adam_action = WorldSim.ACTIONS[adam_action_idx.item()]

                # Step Adam via env
                next_world_state, next_adam_stats, adam_reward, adam_done = env.step(adam_action)

                # Curiosity
                adam_intrinsic = adam.curiosity.compute_intrinsic_reward(world_state, adam.stats)
                adam_reward += adam_intrinsic
                adam.episode_intrinsic_reward += adam_intrinsic

                # Diversity penalty
                adam.action_history.append(adam_action_idx.item())
                if len(adam.action_history) > 10:
                    adam.action_history.pop(0)
                if len(adam.action_history) >= DIVERSITY_WINDOW:
                    if len(set(adam.action_history[-DIVERSITY_WINDOW:])) == 1:
                        adam_reward -= DIVERSITY_PENALTY

                # Phase 6 reflection feedback
                suggested = adam_phase5.get('suggested_action')
                confidence = adam_phase5.get('pattern_confidence', 0)
                if suggested and confidence >= PATTERN_CONFIDENCE_THRESHOLD:
                    if adam_action == suggested:
                        adam_reward += REFLECTION_FOLLOW_BONUS
                    else:
                        adam_reward -= REFLECTION_IGNORE_PENALTY

                # Adam's thought
                adam_experience = adam.thought_engine.experience(
                    world_state=next_world_state,
                    adam_stats=next_adam_stats,
                    action=adam_action,
                    prev_stats=prev_adam_stats,
                    tick=tick,
                    reward=adam_reward,
                )
                adam_rule_thought = adam_experience.get('thought', 'quiet. still.')
                adam.latest_emotion = adam_experience.get('emotion', 'uncertain')

                # Record experience for learned thinker
                adam.learned_thinker.record_experience(adam.sensory_input, adam_rule_thought)

                # Blend or cutover
                if adam.learned_thinker.is_ready():
                    adam_learned_thought = adam.learned_thinker.generate_thought(adam.sensory_input)
                    if learned_only:
                        adam.latest_thought = adam_learned_thought if adam_learned_thought else adam_rule_thought
                    else:
                        adam.latest_thought = blend_thoughts(
                            adam_rule_thought, adam_learned_thought, adam.learned_thinker.confidence
                        )
                else:
                    adam.latest_thought = adam_rule_thought

                # Reflection
                adam.recent_actions.append(adam_action)
                if tick % REFLECTION_INTERVAL == 0 and tick > 0:
                    reflection = adam.thought_engine.reflect(
                        world_state, adam.stats, recent_actions=adam.recent_actions
                    )
                    if reflection.get('has_reflection'):
                        adam.latest_reflection = reflection

                # Dreaming
                if adam_action == "SLEEP" and tick > 1:
                    dream = adam.dream_engine.dream(
                        adam.thought_engine.persistent_memory,
                        adam.thought_engine.memory,
                    )
                    if dream.get('dream_type') != 'empty':
                        adam.latest_dream = dream

                # Adam PPO buffer
                adam.actions_list.append(adam_action_idx.item())
                adam.old_log_probs.append(a_dist.log_prob(adam_action_idx).item())
                adam.rewards_list.append(adam_reward)
                adam.values_list.append(a_value_sq.item())
                adam.masks_list.append(1.0 - adam_done)

                adam.action_counts[adam_action] = adam.action_counts.get(adam_action, 0) + 1
                adam.total_reward += adam_reward
                adam.stats = next_adam_stats
                adam.x = env.adam_x
                adam.y = env.adam_y
                adam.last_action_idx = adam_action_idx.item()  # v1.0: for Eve to observe

                # ═══ EVE'S TURN ═══
                # Eve sees Adam's NEW position
                eve_world_state = eve_agent.get_world_state(adam.x, adam.y)
                prev_eve_stats = copy.deepcopy(eve.stats)

                # Eve chooses action via her own brain
                with torch.no_grad():
                    eve.obs_list.append(eve.sensory_input.clone())
                    e_logits, e_value, eve_agent.hidden_state = eve.model(
                        eve.sensory_input.unsqueeze(0), eve_agent.hidden_state
                    )
                    e_logits_sq = e_logits.squeeze(0)
                    e_value_sq = e_value.squeeze()
                    e_probs = torch.softmax(e_logits_sq, dim=-1)
                    e_probs = torch.clamp(e_probs, min=1e-8, max=1.0)
                    e_probs = e_probs / e_probs.sum()
                    eve_action_idx = torch.multinomial(e_probs, 1)
                    eve_action = WorldSim.ACTIONS[eve_action_idx.item()]
                    e_log_prob = torch.log(e_probs.gather(0, eve_action_idx.squeeze()))

                # Step Eve — v1.0: pass Adam's last action for first-contact mode
                eve_result = eve_agent.step(
                    eve_action,
                    adam_x=adam.x, adam_y=adam.y,
                    adam_last_action_idx=adam.last_action_idx,
                )
                eve_reward = eve_result['reward']
                eve_done = eve_result['done']
                eve.last_action_idx = eve_action_idx.item()  # for Adam to observe next tick

                # Eve diversity penalty
                eve.action_history.append(eve_action_idx.item())
                if len(eve.action_history) > 10:
                    eve.action_history.pop(0)
                if len(eve.action_history) >= DIVERSITY_WINDOW:
                    if len(set(eve.action_history[-DIVERSITY_WINDOW:])) == 1:
                        eve_reward -= DIVERSITY_PENALTY

                # Eve PPO buffer
                eve.actions_list.append(eve_action_idx.item())
                eve.old_log_probs.append(e_log_prob.item())
                eve.rewards_list.append(eve_reward)
                eve.values_list.append(e_value_sq.item())
                eve.masks_list.append(1.0 - eve_done)

                eve.action_counts[eve_action] = eve.action_counts.get(eve_action, 0) + 1
                eve.total_reward += eve_reward
                eve.stats = eve_agent.stats
                eve.x = eve_agent.x
                eve.y = eve_agent.y
                eve.sensory_input = eve_agent.sensory_input
                eve.latest_thought = eve_agent.latest_thought
                eve.latest_emotion = eve_agent.latest_emotion
                # Phase 0.3: propagate new-word discoveries so we can log them
                eve.latest_new_words = getattr(eve_agent, 'latest_new_words', [])

                # ═══ DISPLAY ═══
                if tick % FULL_DISPLAY_EVERY == 0:
                    multi_agent_full_display(tick, adam, eve, env, vocab_logger)
                else:
                    multi_agent_compact_display(
                        tick, adam, eve, adam_action, eve_action,
                        adam_reward, eve_reward, world_state, env,
                    )

                # Show reflections
                if adam.latest_reflection and adam.latest_reflection.get('has_reflection'):
                    print(f"  \U0001fa9e Adam reflects: \"{adam.latest_reflection['reflection']}\" "
                          f"({adam.latest_reflection['personality']})", flush=True)
                    adam.latest_reflection = None

                # Show new words
                if "new_words" in adam_experience:
                    # new_words is now a list of dicts (Phase 0.3 enriched)
                    for entry in adam_experience["new_words"]:
                        word = entry["word"] if isinstance(entry, dict) else entry[0]
                        meaning = entry["meaning"] if isinstance(entry, dict) else entry[1]
                        trigger = entry.get("trigger", "") if isinstance(entry, dict) else ""
                        ctx = entry.get("context", {}) if isinstance(entry, dict) else {}
                        print(f"  \U0001f4dd Adam NEW WORD: \"{word}\" = {meaning}", flush=True)
                        # Phase 0.3: per-word vocab_log.jsonl
                        vocab_logger.log_word_discovery(
                            tick=tick, agent="adam", word=word,
                            meaning=meaning, trigger=trigger, context=ctx,
                        )

                # Log Eve's new words too (Eve's experience is captured inside
                # eve_agent.step(); we read latest_new_words here)
                if getattr(eve, 'latest_new_words', None):
                    for entry in eve.latest_new_words:
                        if isinstance(entry, dict):
                            word = entry.get("word", "")
                            meaning = entry.get("meaning", "")
                            trigger = entry.get("trigger", "")
                            ctx = entry.get("context", {})
                        else:
                            word, meaning = entry[0], entry[1]
                            trigger, ctx = "", {}
                        if not word:
                            continue
                        print(f"  \U0001f4dd Eve  NEW WORD: \"{word}\" = {meaning}", flush=True)
                        vocab_logger.log_word_discovery(
                            tick=tick, agent="eve", word=word,
                            meaning=meaning, trigger=trigger, context=ctx,
                        )

                # ═══ VOCAB DIVERGENCE LOGGING ═══
                if vocab_logger.should_log(tick):
                    adam_vocab = adam.thought_engine.get_vocabulary()
                    eve_vocab = eve.thought_engine.get_vocabulary()
                    vocab_logger.log(tick, adam_vocab, eve_vocab, extra={
                        "adam_learned_confidence": adam.learned_thinker.confidence,
                        "eve_learned_confidence": eve.learned_thinker.confidence,
                        "adam_thought": adam.latest_thought,
                        "eve_thought": eve.latest_thought,
                    })

                # ═══ WS BRIDGE — SEND BOTH AGENTS' DATA ═══
                adam_personality = adam.thought_engine.get_personality()
                adam_cs = adam.curiosity.get_curiosity_stats()
                adam_ds = adam.dream_engine.get_dream_stats()
                eve_personality = eve.thought_engine.get_personality()
                eve_cs = eve.curiosity.get_curiosity_stats()
                eve_ds = eve.dream_engine.get_dream_stats()
                latest_div = vocab_logger.get_latest()

                ws_bridge.send_tick_data({
                    "agent": "adam",
                    "tick": tick,
                    "alive": not adam_done,
                    "adam_stats": {
                        "health": round(adam.stats.get('health', 100), 1),
                        "hunger": round(adam.stats.get('hunger', 0), 1),
                        "energy": round(adam.stats.get('energy', 100), 1),
                        "stress": round(adam.stats.get('stress', 0), 1),
                        "pain": round(adam.stats.get('pain', 0), 1),
                    },
                    "world_state": {
                        "biome": world_state.get('biome', 'plains'),
                        "weather": world_state.get('weather', 'clear'),
                        "temperature": round(world_state.get('temperature', 20), 1),
                        "time_of_day": world_state.get('time_of_day', 12),
                        "adam_x": adam.x, "adam_y": adam.y,
                        "facing": world_state.get('facing', 'north'),
                    },
                    "action": adam_action,
                    "reward": round(adam_reward, 3),
                    "total_reward": round(adam.total_reward, 2),
                    "thought": adam.latest_thought,
                    "emotion": adam.latest_emotion,
                    "action_counts": adam.action_counts,
                    "vocabulary_size": len(adam.thought_engine.get_vocabulary()),
                    "personality": adam_personality.get('disposition', 'uncertain'),
                    "curiosity_states": adam_cs.get('total_states_discovered', 0),
                    "dreams_total": adam_ds.get('total_dreams', 0),
                    "learned_thinking": adam.learned_thinker.get_stats(),
                    "position": [adam.x, adam.y],
                })
                ws_bridge.send_tick_data({
                    "agent": "eve",
                    "tick": tick,
                    "alive": not eve_done,
                    "adam_stats": {  # keep field name for dashboard compat
                        "health": round(eve.stats.get('health', 100), 1),
                        "hunger": round(eve.stats.get('hunger', 0), 1),
                        "energy": round(eve.stats.get('energy', 100), 1),
                        "stress": round(eve.stats.get('stress', 0), 1),
                        "pain": round(eve.stats.get('pain', 0), 1),
                    },
                    "world_state": {
                        "biome": eve_agent.current_biome,
                        "weather": world_state.get('weather', 'clear'),
                        "temperature": round(world_state.get('temperature', 20), 1),
                        "time_of_day": world_state.get('time_of_day', 12),
                        "adam_x": eve.x, "adam_y": eve.y,  # field name kept for compat
                        "facing": eve_agent.facing,
                    },
                    "action": eve_action,
                    "reward": round(eve_reward, 3),
                    "total_reward": round(eve.total_reward, 2),
                    "thought": eve.latest_thought,
                    "emotion": eve.latest_emotion,
                    "action_counts": eve.action_counts,
                    "vocabulary_size": len(eve.thought_engine.get_vocabulary()),
                    "personality": eve_personality.get('disposition', 'uncertain'),
                    "curiosity_states": eve_cs.get('total_states_discovered', 0),
                    "dreams_total": eve_ds.get('total_dreams', 0),
                    "learned_thinking": eve.learned_thinker.get_stats(),
                    "position": [eve.x, eve.y],
                })

                # Send vocab divergence if updated this tick
                if latest_div:
                    ws_bridge.send_tick_data({
                        "agent": "system",
                        "type_extra": "vocab_divergence",
                        "tick": tick,
                        "adam_vocab_size": latest_div['adam_vocab_size'],
                        "eve_vocab_size": latest_div['eve_vocab_size'],
                        "shared_count": latest_div['shared_count'],
                        "jaccard": latest_div['jaccard'],
                    })

                # ═══ PPO UPDATES (both agents independently) ═══
                if tick % PPO_UPDATE_INTERVAL == 0:
                    # Adam
                    if len(adam.obs_list) > 0:
                        # Encode next observation for bootstrap
                        next_phase5 = adam.thought_engine.get_phase5_signals(
                            next_world_state, next_adam_stats
                        )
                        if first_contact:
                            other_p, _ = _compute_other_presence(
                                adam.x, adam.y, eve.x, eve.y)
                            adam.sensory_input = encode_sensory_input_multi(
                                next_world_state, next_adam_stats,
                                fear_signal=next_phase5['fear_signal'],
                                pleasure_signal=next_phase5['pleasure_signal'],
                                pattern_confidence=next_phase5['pattern_confidence'],
                                other_presence=other_p,
                                other_last_action_idx=eve.last_action_idx,
                            ).to(DEVICE)
                        else:
                            adam.sensory_input = encode_sensory_input(
                                next_world_state, next_adam_stats,
                                fear_signal=next_phase5['fear_signal'],
                                pleasure_signal=next_phase5['pleasure_signal'],
                                pattern_confidence=next_phase5['pattern_confidence'],
                            ).to(DEVICE)

                        adam_ppo = ppo_update(
                            adam.model, adam.optimizer, DEVICE,
                            adam.obs_list, adam.actions_list, adam.old_log_probs,
                            adam.rewards_list, adam.values_list, adam.masks_list,
                            adam.sensory_input, adam.hidden_state,
                        )
                        adam.clear_ppo_buffer()

                        if adam_ppo and tick % TB_LOG_INTERVAL == 0:
                            tb_logger.log_episode(tick, {
                                "adam_reward": adam.total_reward / max(tick, 1),
                                "adam_total_reward": adam.total_reward,
                                "adam_policy_loss": adam_ppo['policy_loss'],
                                "adam_value_loss": adam_ppo['value_loss'],
                                "adam_entropy": adam_ppo['entropy'],
                                "tick": tick,
                            })

                    # Eve
                    if len(eve.obs_list) > 0:
                        eve_ppo = ppo_update(
                            eve.model, eve.optimizer, DEVICE,
                            eve.obs_list, eve.actions_list, eve.old_log_probs,
                            eve.rewards_list, eve.values_list, eve.masks_list,
                            eve.sensory_input, eve_agent.hidden_state,
                        )
                        eve.clear_ppo_buffer()

                        if eve_ppo and tick % TB_LOG_INTERVAL == 0:
                            tb_logger.log_episode(tick, {
                                "eve_reward": eve.total_reward / max(tick, 1),
                                "eve_total_reward": eve.total_reward,
                                "eve_policy_loss": eve_ppo['policy_loss'],
                                "eve_value_loss": eve_ppo['value_loss'],
                                "eve_entropy": eve_ppo['entropy'],
                            })
                else:
                    # Still need to update Adam's sensory_input for next tick
                    if not adam_done:
                        next_phase5 = adam.thought_engine.get_phase5_signals(
                            next_world_state, next_adam_stats
                        )
                        if first_contact:
                            other_p, _ = _compute_other_presence(
                                adam.x, adam.y, eve.x, eve.y)
                            adam.sensory_input = encode_sensory_input_multi(
                                next_world_state, next_adam_stats,
                                fear_signal=next_phase5['fear_signal'],
                                pleasure_signal=next_phase5['pleasure_signal'],
                                pattern_confidence=next_phase5['pattern_confidence'],
                                other_presence=other_p,
                                other_last_action_idx=eve.last_action_idx,
                            ).to(DEVICE)
                        else:
                            adam.sensory_input = encode_sensory_input(
                                next_world_state, next_adam_stats,
                                fear_signal=next_phase5['fear_signal'],
                                pleasure_signal=next_phase5['pleasure_signal'],
                                pattern_confidence=next_phase5['pattern_confidence'],
                            ).to(DEVICE)

                # ═══ CHECKPOINT ═══
                if tick % CHECKPOINT_INTERVAL == 0:
                    adam_path = os.path.join(MODEL_DIR, f"adam_tick{tick}.pt")
                    torch.save({
                        'tick': tick,
                        'model_state_dict': adam.model.state_dict(),
                        'total_reward': adam.total_reward,
                        'input_dim': adam.input_dim, 'hidden_dim': HIDDEN_DIM,
                        'num_actions': NUM_ACTIONS,
                        'first_contact': first_contact,
                    }, adam_path)
                    eve_path = os.path.join(MODEL_DIR, f"eve_tick{tick}.pt")
                    torch.save({
                        'tick': tick,
                        'model_state_dict': eve.model.state_dict(),
                        'total_reward': eve.total_reward,
                        'input_dim': eve.input_dim, 'hidden_dim': HIDDEN_DIM,
                        'num_actions': NUM_ACTIONS,
                        'first_contact': first_contact,
                    }, eve_path)
                    print(f"  \U0001f4be Saved checkpoints at tick {tick}", flush=True)

                # ═══ DEATH CHECK ═══
                if adam_done:
                    adam.alive = False
                    adam.death_cause = "health depletion" if adam.stats.get('health', 100) <= 0 else \
                                       "starvation" if adam.stats.get('hunger', 0) >= 100 else \
                                       "exhaustion"
                    print(f"\n  \U0001f480 Adam died at tick {tick}: {adam.death_cause}", flush=True)

                if eve_done:
                    eve.alive = False
                    eve.death_cause = eve_result.get('death_cause', 'unknown')
                    print(f"\n  \U0001f480 Eve died at tick {tick}: {eve.death_cause}", flush=True)

                # Update world state for next tick
                world_state = next_world_state

                if tick_delay > 0:
                    time.sleep(tick_delay)

            except KeyboardInterrupt:
                print(f"\n  \u26a0\ufe0f Interrupted by user at tick {tick}.", flush=True)
                break
            except Exception as e:
                print(f"\n  \u274c ERROR at tick {tick}: {e}", flush=True)
                traceback.print_exc()
                print(f"  Continuing...", flush=True)
                continue

    finally:
        # ── Death / cleanup ─────────────────────────────────────────────────
        elapsed_min = (time.time() - life_start) / 60

        # Final vocab divergence summary
        vocab_logger.summary()

        # Phase 0.3: render static HTML dashboard panel
        try:
            dashboard_path = vocab_logger.render_dashboard_html("docs/vocab_dashboard.html")
            print(f"  \U0001f4ca Vocabulary dashboard written to: {dashboard_path}", flush=True)
        except Exception as e:
            print(f"  \u26a0\ufe0f Failed to render vocab dashboard: {e}", flush=True)

        # Save final models
        adam_final = os.path.join(MODEL_DIR, "adam_final.pt")
        eve_final = os.path.join(MODEL_DIR, "eve_final.pt")
        torch.save({
            'model_state_dict': adam.model.state_dict(),
            'input_dim': adam.input_dim, 'hidden_dim': HIDDEN_DIM, 'num_actions': NUM_ACTIONS,
            'ticks_lived': tick, 'total_reward': adam.total_reward,
            'action_counts': adam.action_counts,
            'first_contact': first_contact,
        }, adam_final)
        torch.save({
            'model_state_dict': eve.model.state_dict(),
            'input_dim': eve.input_dim, 'hidden_dim': HIDDEN_DIM, 'num_actions': NUM_ACTIONS,
            'ticks_lived': tick, 'total_reward': eve.total_reward,
            'action_counts': eve.action_counts,
            'first_contact': first_contact,
        }, eve_final)
        print(f"  \U0001f4be Final brains saved: {adam_final}, {eve_final}", flush=True)

        # Save life logs
        adam_log = {
            "agent": "adam", "mode": "multi_agent",
            "ticks_lived": tick, "total_reward": adam.total_reward,
            "real_time_min": elapsed_min,
            "action_counts": adam.action_counts,
            "vocabulary": adam.thought_engine.get_vocabulary(),
            "personality": adam.thought_engine.get_personality(),
            "curiosity_stats": adam.curiosity.get_curiosity_stats(),
            "dream_stats": adam.dream_engine.get_dream_stats(),
            "learned_thinking": adam.learned_thinker.get_stats(),
            "death_cause": adam.death_cause,
        }
        eve_log = {
            "agent": "eve", "mode": "multi_agent",
            "ticks_lived": tick, "total_reward": eve.total_reward,
            "real_time_min": elapsed_min,
            "action_counts": eve.action_counts,
            "vocabulary": eve.thought_engine.get_vocabulary(),
            "personality": eve.thought_engine.get_personality(),
            "curiosity_stats": eve.curiosity.get_curiosity_stats(),
            "dream_stats": eve.dream_engine.get_dream_stats(),
            "learned_thinking": eve.learned_thinker.get_stats(),
            "death_cause": eve.death_cause,
        }
        with open("life_log_adam.json", "w") as f:
            json.dump(adam_log, f, indent=2)
        with open("life_log_eve.json", "w") as f:
            json.dump(eve_log, f, indent=2)

        # Send death events
        ws_bridge.send_death_data({
            "agent": "adam", "tick": tick,
            "total_reward": adam.total_reward,
            "death_cause": adam.death_cause,
            "action_counts": adam.action_counts,
        })
        ws_bridge.send_death_data({
            "agent": "eve", "tick": tick,
            "total_reward": eve.total_reward,
            "death_cause": eve.death_cause,
            "action_counts": eve.action_counts,
        })

        ws_bridge.stop()
        tb_logger.close()

        print(f"\n{'=' * 78}", flush=True)
        print(f"  \U0001f480 LIFE ENDED — {elapsed_min:.1f} min, {tick} ticks", flush=True)
        print(f"  Adam: reward={adam.total_reward:.2f}, vocab={len(adam.thought_engine.get_vocabulary())}, "
              f"cause={adam.death_cause}", flush=True)
        print(f"  Eve:  reward={eve.total_reward:.2f}, vocab={len(eve.thought_engine.get_vocabulary())}, "
              f"cause={eve.death_cause}", flush=True)
        print(f"{'=' * 78}\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nafs AI — Multi-Agent (Adam + Eve)")
    parser.add_argument("--learned-only", action="store_true",
                        help="Full cutover to learned transformer (v0.3 mode)")
    parser.add_argument("--first-contact", action="store_true",
                        help="v1.0 mode: 23-dim sensory input so agents observe each other's actions")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Cap on total ticks (for testing)")
    parser.add_argument("--tick-delay", type=float, default=TICK_DELAY,
                        help=f"Seconds between ticks (default {TICK_DELAY})")
    args = parser.parse_args()

    run_multi_agent_life(
        learned_only=args.learned_only,
        max_ticks=args.max_ticks,
        tick_delay=args.tick_delay,
        first_contact=args.first_contact,
    )
