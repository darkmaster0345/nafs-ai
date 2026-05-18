"""
Nafs AI — Training Run 3 (PPO with Sequential GRU + Inner Voice)
"What emerges when code has no memory of the world?"

Fixes applied:
  Fix 1: Entropy bonus (coef=0.05)
  Fix 2: Action diversity penalty (0.25)
  Fix 3: Proper PPO clipped surrogate objective
  Fix 4: GAE for advantage estimation
  Fix 5: Single unified optimizer
  Fix 6: Balanced world rewards

Phases 1-6:
  - Inner Voice (thought, emotion, world description)
  - Vocabulary Discovery (naming from experience)
  - Dialogue Gap (subconscious)
  - Persistent Memory + Personality
  - Fear/Pleasure Maps
  - Dual-Speed Processing

New features:
  - Curiosity-driven exploration (intrinsic motivation)
  - Dreaming / memory consolidation during SLEEP
  - TensorBoard logging
  - Deeper Phase 6 reflection

Run: python train.py
TensorBoard: tensorboard --logdir runs/
Evaluate: python evaluate.py
Test: pytest tests/ -v
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

from sensory_encoder import encode_sensory_input
from baby_brain_model import BabyBrain
from world_sim import WorldSim
from thought_engine import ThoughtEngine
from curiosity import CuriosityModule
from dreaming import DreamEngine
from tb_logger import TBLogger

INPUT_DIM = 15  # 12 base sensory + 3 Phase 5 signals (fear, pleasure, pattern_confidence)
HIDDEN_DIM = 256
NUM_ACTIONS = len(WorldSim.ACTIONS)
EPISODES = 500
STEPS_PER_EPISODE = 200
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LEARNING_RATE = 3e-4
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
ENTROPY_COEF = 0.05
DIVERSITY_PENALTY = 0.25
DIVERSITY_WINDOW = 5
LOG_INTERVAL = 50
SAVE_INTERVAL = 50
MODEL_DIR = "checkpoints"
TRAINING_LOG_PATH = "training_log_run3.json"

# Phase 1: Thought sampling — show Adam's inner life every N episodes
THOUGHT_SAMPLE_INTERVAL = 50   # show a sample thought every 50 episodes
THOUGHT_SAMPLE_TICK = 10       # show the thought at this tick within the episode


# Phase 6: Reflection bonus — small extra reward when Adam follows his
# own learned patterns. This creates the feedback loop: slow reflection
# influences future PPO decisions via reward shaping.
REFLECTION_FOLLOW_BONUS = 0.05    # Bonus for following a learned pattern
REFLECTION_IGNORE_PENALTY = 0.02  # Penalty for ignoring a strong pattern
PATTERN_CONFIDENCE_THRESHOLD = 0.5  # Only apply bonus/penalty above this confidence

# Curiosity: Intrinsic motivation — reward for visiting novel states
CURIOSITY_BONUS = 0.15             # Max intrinsic reward per step
CURIOSITY_DECAY = 0.98            # How fast curiosity fades per episode
CURIOSITY_MIN = 0.01              # Minimum curiosity (never fully dies)


def train(start_episode=0):
    DEVICE = torch.device("cpu")
    model = BabyBrain(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    env = WorldSim()

    # Phase 1: Initialize Thought Engine — Adam's inner voice
    thought_engine = ThoughtEngine(memory_size=10)

    # Curiosity: Intrinsic motivation module
    curiosity = CuriosityModule(
        curiosity_bonus=CURIOSITY_BONUS,
        curiosity_decay=CURIOSITY_DECAY,
        min_curiosity=CURIOSITY_MIN,
    )

    # Dreaming: Memory consolidation during SLEEP
    dream_engine = DreamEngine()

    # TensorBoard logger (graceful fallback if not installed)
    tb_logger = TBLogger(log_dir="runs/nafs_run")

    episode_rewards = []
    survival_lengths = []
    all_action_counts = {a: 0 for a in WorldSim.ACTIONS}
    best_survival = 0
    training_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load from checkpoint if resuming
    if start_episode > 0:
        cp_path = os.path.join(MODEL_DIR, f"baby_brain_run3_ep{start_episode}.pt")
        if os.path.exists(cp_path):
            ckpt = torch.load(cp_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            best_survival = ckpt.get('best_survival', 0)
            print(f"  Loaded checkpoint from ep {start_episode}, best_survival={best_survival}", flush=True)
        else:
            print(f"  WARNING: Checkpoint {cp_path} not found, starting fresh", flush=True)
            start_episode = 0

    print("=" * 70, flush=True)
    print("  NAFS AI — Training Run 3 (Sequential PPO + GRU + Inner Voice)", flush=True)
    print(f"  entropy_coef={ENTROPY_COEF}, diversity_penalty={DIVERSITY_PENALTY}", flush=True)
    print(f"  curiosity_bonus={CURIOSITY_BONUS}, curiosity_decay={CURIOSITY_DECAY}", flush=True)
    print(f"  Phase 1+2: Thought + Emotion + Vocabulary Discovery", flush=True)
    print(f"  Phase 3: Dialogue Gap (Subconscious)", flush=True)
    print(f"  Phase 4: Persistent Memory + Personality", flush=True)
    print(f"  Phase 5: Fear Maps + Pleasure Maps", flush=True)
    print(f"  Phase 6: Dual-Speed Processing (Fast PPO + Slow Reflection)", flush=True)
    print(f"  NEW: Curiosity-driven exploration + Dreaming + TensorBoard", flush=True)
    print("=" * 70, flush=True)
    print(f"\nStarting training for {EPISODES} episodes...", flush=True)
    print(f"  Adam wakes up. He knows nothing.", flush=True)
    print(f"  He will learn to think. He will learn to feel. He will name things.", flush=True)
    print(f"  He will dream. He will be curious.\n", flush=True)

    for episode in range(start_episode, EPISODES):
        try:
            world_state, adam_stats_dict = env.reset()
            # Phase 5: Include fear/pleasure/pattern signals from the start
            init_phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)
            sensory_input = encode_sensory_input(
                world_state, adam_stats_dict,
                fear_signal=init_phase5['fear_signal'],
                pleasure_signal=init_phase5['pleasure_signal'],
                pattern_confidence=init_phase5['pattern_confidence'],
            ).to(DEVICE)
            hidden_state = model.init_hidden(1).to(DEVICE)
            action_history = []

            obs_list, actions_list, old_log_probs = [], [], []
            rewards_list, values_list, masks_list = [], [], []

            episode_reward = 0
            episode_steps = 0
            done = False

            # Phase 1: Reset thought engine for new episode
            thought_engine.reset_episode()
            curiosity.reset_episode()

            # Track last sample experience for display
            sample_experience = None
            sample_reflection = None
            sample_dream = None

            # Phase 6: Track recent actions for reflection
            recent_episode_actions = []
            episode_intrinsic_reward = 0.0

            # Rollout
            with torch.no_grad():
                for step in range(STEPS_PER_EPISODE):
                    # Save previous stats for outcome description
                    prev_stats = copy.deepcopy(adam_stats_dict)

                    # Phase 5: Compute fear/pleasure/pattern signals for PPO observation
                    phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)

                    obs_list.append(sensory_input.clone())
                    action_logits, state_value, hidden_state = model(sensory_input.unsqueeze(0), hidden_state)
                    action_logits_sq = action_logits.squeeze(0)
                    state_value_sq = state_value.squeeze()
                    action_dist = torch.distributions.Categorical(logits=action_logits_sq)
                    action_idx = action_dist.sample()
                    action = WorldSim.ACTIONS[action_idx.item()]

                    next_world_state, next_adam_stats_dict, reward, done = env.step(action)

                    # Curiosity: Add intrinsic reward for novel states
                    intrinsic_reward = curiosity.compute_intrinsic_reward(
                        world_state, adam_stats_dict
                    )
                    reward += intrinsic_reward
                    episode_intrinsic_reward += intrinsic_reward

                    action_history.append(action_idx.item())
                    if len(action_history) > 10:
                        action_history.pop(0)
                    if len(action_history) >= DIVERSITY_WINDOW:
                        if len(set(action_history[-DIVERSITY_WINDOW:])) == 1:
                            reward -= DIVERSITY_PENALTY

                    # Phase 6: Reflection feedback — reward shaping based on learned patterns
                    suggested = phase5.get('suggested_action')
                    confidence = phase5.get('pattern_confidence', 0)
                    if suggested and confidence >= PATTERN_CONFIDENCE_THRESHOLD:
                        if action == suggested:
                            reward += REFLECTION_FOLLOW_BONUS
                        else:
                            reward -= REFLECTION_IGNORE_PENALTY

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
                        tick=step,
                        reward=reward,
                    )

                    # Phase 2: Log vocabulary discoveries
                    if "new_words" in experience:
                        for word, meaning in experience["new_words"]:
                            print(f"  \U0001f4dd NEW WORD: \"{word}\" = {meaning}", flush=True)

                    # Phase 6: Dual-speed — fast action (PPO) done, now slow reflection
                    recent_episode_actions.append(action)
                    if step % 20 == 0 and step > 0:  # Reflect every 20 ticks
                        reflection = thought_engine.reflect(
                            world_state, adam_stats_dict,
                            recent_actions=recent_episode_actions
                        )
                        if step == THOUGHT_SAMPLE_TICK + 10:
                            sample_reflection = reflection

                    # Capture sample experience for display
                    if step == THOUGHT_SAMPLE_TICK:
                        sample_experience = experience

                    # Dreaming: Memory consolidation during SLEEP
                    if action == "SLEEP" and step > 0:
                        dream = dream_engine.dream(
                            thought_engine.persistent_memory,
                            thought_engine.memory
                        )
                        if dream.get('dream_type') != 'empty':
                            sample_dream = dream

                    # Phase 5: Encode next observation WITH fear/pleasure/pattern signals
                    next_phase5 = thought_engine.get_phase5_signals(next_world_state, next_adam_stats_dict)
                    sensory_input = encode_sensory_input(
                        next_world_state, next_adam_stats_dict,
                        fear_signal=next_phase5['fear_signal'],
                        pleasure_signal=next_phase5['pleasure_signal'],
                        pattern_confidence=next_phase5['pattern_confidence'],
                    ).to(DEVICE)
                    episode_reward += reward
                    episode_steps += 1
                    if done:
                        break

            episode_rewards.append(episode_reward)
            survival_lengths.append(episode_steps)
            if episode_steps > best_survival:
                best_survival = episode_steps

            # Phase 4: End episode in persistent memory
            thought_engine.end_episode(episode_steps)
            curiosity.end_episode()

            # GAE
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
                action_logits, state_value, h = model(obs_list[i].unsqueeze(0), h.detach())
                action_logits_sq = action_logits.squeeze(0)
                state_value_sq = state_value.squeeze()
                dist = torch.distributions.Categorical(logits=action_logits_sq)
                new_log_probs.append(dist.log_prob(actions_t[i]))
                new_entropies.append(dist.entropy())
                new_values.append(state_value_sq)

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

            # Logging
            if (episode + 1) % LOG_INTERVAL == 0:
                avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
                avg_survival = np.mean(survival_lengths[-LOG_INTERVAL:])
                elapsed_min = (time.time() - training_start) / 60
                total_actions = sum(all_action_counts.values())
                dist_parts = []
                for a in WorldSim.ACTIONS:
                    cnt = all_action_counts.get(a, 0)
                    pct = cnt / max(total_actions, 1) * 100
                    dist_parts.append(f"{a}:{pct:.0f}%")

                print(f"Ep {episode+1:>5}/{EPISODES} | R:{avg_reward:>7.2f} | Surv:{avg_survival:>5.1f} | Best:{best_survival:>3} | {elapsed_min:>5.1f}m | {' '.join(dist_parts)}", flush=True)

                # Phase 1: Show sample inner experience
                if sample_experience:
                    compact = thought_engine.format_experience(sample_experience, compact=True)
                    print(f"{compact}", flush=True)

                # Phase 6: Show reflection
                if sample_reflection and sample_reflection.get('has_reflection'):
                    print(f"  \U0001fa9e Reflect: \"{sample_reflection['reflection']}\" ({sample_reflection['personality']})", flush=True)

                # Show dream if one occurred
                if sample_dream and sample_dream.get('dream_type') != 'empty':
                    dream_thoughts = ' '.join(sample_dream.get('thoughts', []))
                    print(f"  \U0001f4a4 Dream ({sample_dream['dream_type']}): {dream_thoughts}", flush=True)

                # Show personality every 200 episodes
                if (episode + 1) % 200 == 0:
                    personality = thought_engine.get_personality()
                    vocab = thought_engine.get_vocabulary()
                    discovered = thought_engine.get_discovered_vocabulary()
                    print(f"  \U0001f464 Personality: {personality['disposition']} | "
                          f"Fears:{personality['fear_triggers']} Joys:{personality['good_memories']} | "
                          f"Patterns:{personality['patterns_learned']}", flush=True)
                    if discovered:
                        print(f"  \U0001f4da Vocab discovered: {discovered}", flush=True)
                    print(f"  \U0001f4d6 Total vocab: {len(vocab)} words", flush=True)
                    cs = curiosity.get_curiosity_stats()
                    print(f"  \U0001f50d Curiosity: {cs['total_states_discovered']} states, "
                          f"{cs['novel_states']} novel, {cs['familiar_states']} familiar", flush=True)
                    ds = dream_engine.get_dream_stats()
                    print(f"  \U0001f4a4 Dreams: {ds['total_dreams']} total "
                          f"({ds['nightmares']} nightmares, {ds['peaceful_dreams']} peaceful)", flush=True)

                if total_actions > 0:
                    max_pct = max(cnt / total_actions for cnt in all_action_counts.values())
                    if max_pct > 0.60:
                        dominant = max(all_action_counts, key=all_action_counts.get)
                        print(f"  \u26a0\ufe0f MONO: {dominant} at {max_pct*100:.0f}%", flush=True)

                # TensorBoard logging
                action_dist = {}
                for a in WorldSim.ACTIONS:
                    cnt = all_action_counts.get(a, 0)
                    pct = cnt / max(total_actions, 1) * 100
                    action_dist[a] = pct

                personality = thought_engine.get_personality()
                discovered = thought_engine.get_discovered_vocabulary()
                cs = curiosity.get_curiosity_stats()
                ds = dream_engine.get_dream_stats()

                tb_data = {
                    "reward": avg_reward,
                    "survival_length": avg_survival,
                    "best_survival": best_survival,
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": new_entropies_t.mean().item(),
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    "action_dist": action_dist,
                    "dominant_pct": max_pct * 100 if total_actions > 0 else 0,
                    "vocabulary_size": len(thought_engine.get_vocabulary()),
                    "discovered_words_count": len(discovered) if discovered else 0,
                    "fear_triggers_count": personality['fear_triggers'],
                    "good_memories_count": personality['good_memories'],
                    "patterns_count": personality['patterns_learned'],
                    "curiosity_intrinsic_reward": episode_intrinsic_reward,
                    "curiosity_states_discovered": cs['total_states_discovered'],
                    "dream_count": ds['total_dreams'],
                    "nightmare_count": ds['nightmares'],
                    "personality_disposition": personality['disposition'],
                    "thought_sample": thought_engine.format_experience(sample_experience, compact=True) if sample_experience else "",
                    "reflection_sample": sample_reflection.get('reflection', '') if sample_reflection else "",
                }
                tb_logger.log_episode(episode + 1, tb_data)

            # Save checkpoint
            if (episode + 1) % SAVE_INTERVAL == 0:
                model_path = os.path.join(MODEL_DIR, f"baby_brain_run3_ep{episode+1}.pt")
                torch.save({
                    'episode': episode, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_survival': best_survival, 'input_dim': INPUT_DIM,
                    'hidden_dim': HIDDEN_DIM, 'num_actions': NUM_ACTIONS,
                    'entropy_coef': ENTROPY_COEF, 'diversity_penalty': DIVERSITY_PENALTY,
                }, model_path)
                print(f"  \U0001f4be Saved {model_path}", flush=True)

        except Exception as e:
            print(f"\n  \u274c ERROR at episode {episode+1}: {e}", flush=True)
            traceback.print_exc()
            print(f"  Continuing to next episode...", flush=True)
            continue

    # Final Results
    elapsed_min = (time.time() - training_start) / 60
    total_actions = sum(all_action_counts.values())
    print("\n" + "=" * 70, flush=True)
    print("  TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"  Training time: {elapsed_min:.1f} min", flush=True)
    print(f"  Best survival: {best_survival} ticks", flush=True)
    print(f"  Avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}", flush=True)
    print(f"  Avg survival (last 100): {np.mean(survival_lengths[-100:]):.2f}", flush=True)
    print(f"\n  Action Distribution:", flush=True)
    for a in WorldSim.ACTIONS:
        cnt = all_action_counts.get(a, 0)
        pct = cnt / max(total_actions, 1) * 100
        print(f"    {a:>8}: {pct:>5.1f}%", flush=True)

    # Show final sample thought
    print(f"\n  Adam's Last Thoughts:", flush=True)
    if sample_experience:
        full = thought_engine.format_experience(sample_experience, adam_stats=adam_stats_dict)
        print(full, flush=True)

    print(f"\n  Vocabulary: {thought_engine.get_vocabulary()}", flush=True)
    discovered = thought_engine.get_discovered_vocabulary()
    if discovered:
        print(f"  Discovered words: {discovered}", flush=True)
    print(f"  Total thoughts generated: {thought_engine.thought_gen.total_thoughts}", flush=True)

    # ALWAYS save persistent memory at end of training
    memory_path = os.path.join(os.path.dirname(MODEL_DIR) or ".", "memory.json")
    thought_engine.persistent_memory.save_to_disk(memory_path)

    # Also save curiosity state in the memory.json
    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            mem_data = json.load(f)
        mem_data['curiosity_state'] = curiosity.save_state()
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(mem_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    print(f"  Memory saved to memory.json", flush=True)

    # Curiosity stats
    cs = curiosity.get_curiosity_stats()
    print(f"  Curiosity: {cs['total_states_discovered']} states discovered, "
          f"{cs['novel_states']} novel, {cs['familiar_states']} familiar", flush=True)

    # Dream stats
    ds = dream_engine.get_dream_stats()
    print(f"  Dreams: {ds['total_dreams']} total ({ds['nightmares']} nightmares, "
          f"{ds['peaceful_dreams']} peaceful)", flush=True)

    # Success criteria
    max_pct = max(cnt / max(total_actions,1) for cnt in all_action_counts.values())
    num_used = sum(1 for cnt in all_action_counts.values() if cnt > total_actions * 0.01)
    print(f"\n  Success Criteria:", flush=True)
    print(f"    No action >60%: {'✅' if max_pct <= 0.60 else '❌'} ({max_pct*100:.0f}%)", flush=True)
    print(f"    ≥3 actions used: {'✅' if num_used >= 3 else '❌'} ({num_used})", flush=True)
    print(f"    Survival >102: {'✅' if best_survival > 102 else '❌'} ({best_survival})", flush=True)

    # Save final
    final_path = os.path.join(MODEL_DIR, "baby_brain_run3_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(), 'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM, 'num_actions': NUM_ACTIONS,
        'actions_list': WorldSim.ACTIONS, 'entropy_coef': ENTROPY_COEF,
        'diversity_penalty': DIVERSITY_PENALTY, 'training_time_min': elapsed_min,
        'best_survival': best_survival,
        'final_avg_reward': float(np.mean(episode_rewards[-100:])),
        'final_avg_survival': float(np.mean(survival_lengths[-100:])),
        'action_counts': all_action_counts,
    }, final_path)
    print(f"\n  Model: {final_path}", flush=True)

    log_data = {
        "run": 3, "total_episodes": len(episode_rewards),
        "training_time_min": elapsed_min, "best_survival": best_survival,
        "final_avg_reward": float(np.mean(episode_rewards[-100:])),
        "final_avg_survival": float(np.mean(survival_lengths[-100:])),
        "action_counts": all_action_counts, "entropy_coef": ENTROPY_COEF,
        "diversity_penalty": DIVERSITY_PENALTY, "max_action_pct": float(max_pct),
        "vocabulary": thought_engine.get_vocabulary(),
        "total_thoughts": thought_engine.thought_gen.total_thoughts,
        "curiosity_stats": curiosity.get_curiosity_stats(),
        "dream_stats": dream_engine.get_dream_stats(),
    }
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2)

    # Close TensorBoard
    tb_logger.close()


if __name__ == '__main__':
    train()
