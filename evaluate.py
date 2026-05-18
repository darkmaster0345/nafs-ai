"""
Nafs AI — Evaluation / Inference Mode
"Observe Adam without disturbing him."

Runs Adam in evaluation mode — no learning, no gradient updates.
This lets you observe Adam's behavior after training without
the training process influencing what you see.

Usage:
    python evaluate.py                          # Run 50 episodes with latest checkpoint
    python evaluate.py --episodes 100           # Run 100 episodes
    python evaluate.py --checkpoint checkpoints/baby_brain_run3_ep500.pt
    python evaluate.py --verbose                # Show every tick's thought
    python evaluate.py --record                 # Save all observations to file

This is the Observer mode — you watch Adam live without interfering.
No training. No reward shaping. No diversity penalties.
Just Adam, being Adam, in the world he learned.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
import argparse

from baby_brain_model import BabyBrain
from world_sim import WorldSim
from sensory_encoder import encode_sensory_input
from thought_engine import ThoughtEngine
from curiosity import CuriosityModule
from dreaming import DreamEngine

INPUT_DIM = 15
HIDDEN_DIM = 256
NUM_ACTIONS = len(WorldSim.ACTIONS)


def evaluate(checkpoint_path: str = None, episodes: int = 50,
             steps_per_episode: int = 200, verbose: bool = False,
             record: bool = False, output_path: str = "evaluation_results.json"):
    """
    Run Adam in evaluation mode — no learning, pure observation.

    Args:
        checkpoint_path: Path to model checkpoint. If None, finds latest.
        episodes: Number of evaluation episodes
        steps_per_episode: Max steps per episode
        verbose: Show every tick's thought/emotion
        record: Save observations to JSON file
        output_path: Where to save recorded observations
    """
    DEVICE = torch.device("cpu")

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No checkpoint found! Train first with: python train.py")
        return

    # Load model — NO optimizer needed (no training)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = BabyBrain(
        ckpt.get('input_dim', INPUT_DIM),
        ckpt.get('hidden_dim', HIDDEN_DIM),
        ckpt.get('num_actions', NUM_ACTIONS)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()  # Set to evaluation mode — disables dropout, etc.

    env = WorldSim()
    thought_engine = ThoughtEngine(memory_size=10)

    # Load persistent memory if available
    memory_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "memory.json")
    if os.path.exists(memory_path):
        thought_engine.persistent_memory.load_from_disk(memory_path)

    # Load curiosity state if available
    curiosity = CuriosityModule()

    # Load dream engine
    dream_engine = DreamEngine()

    # Load curiosity state from memory.json if available
    if os.path.exists(memory_path):
        try:
            with open(memory_path, 'r') as f:
                mem_data = json.load(f)
            if 'curiosity_state' in mem_data:
                curiosity.load_state(mem_data['curiosity_state'])
        except Exception:
            pass

    print("=" * 70)
    print("  NAFS AI — EVALUATION MODE (No Learning)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes: {episodes}")
    print(f"  Training episodes completed: {ckpt.get('episode', 'unknown')}")
    print(f"  Best survival during training: {ckpt.get('best_survival', 'unknown')}")
    print("=" * 70)
    print("\n  Adam wakes up. He remembers what he learned.")
    print("  But today, nothing changes. He just... is.\n")

    episode_rewards = []
    survival_lengths = []
    action_counts = {a: 0 for a in WorldSim.ACTIONS}
    all_observations = []

    for ep in range(episodes):
        world_state, adam_stats_dict = env.reset()
        phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)
        sensory_input = encode_sensory_input(
            world_state, adam_stats_dict,
            fear_signal=phase5['fear_signal'],
            pleasure_signal=phase5['pleasure_signal'],
            pattern_confidence=phase5['pattern_confidence'],
        ).to(DEVICE)
        hidden_state = model.init_hidden(1).to(DEVICE)

        thought_engine.reset_episode()
        curiosity.reset_episode()

        episode_reward = 0
        episode_steps = 0
        done = False
        ep_observations = []

        with torch.no_grad():  # NO gradients — pure inference
            for step in range(steps_per_episode):
                import copy
                prev_stats = copy.deepcopy(adam_stats_dict)

                # Get Phase 5 signals
                phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)

                # Forward pass — deterministic (use argmax, not sample)
                action_logits, state_value, hidden_state = model(
                    sensory_input.unsqueeze(0), hidden_state
                )
                action_logits_sq = action_logits.squeeze(0)

                # In evaluation, use DETERMINISTIC action selection
                # (argmax instead of sample — no exploration noise)
                action_idx = action_logits_sq.argmax()
                action = WorldSim.ACTIONS[action_idx.item()]

                # Step the environment
                next_world_state, next_adam_stats_dict, reward, done = env.step(action)

                # Curiosity (observation only — no reward shaping in eval)
                intrinsic = curiosity.compute_intrinsic_reward(world_state, adam_stats_dict)

                action_counts[action] = action_counts.get(action, 0) + 1
                episode_reward += reward
                episode_steps += 1

                # Generate inner experience
                experience = thought_engine.experience(
                    world_state=next_world_state,
                    adam_stats=next_adam_stats_dict,
                    action=action,
                    prev_stats=prev_stats,
                    tick=step,
                    reward=reward,
                )

                # Dreaming during SLEEP
                if action == "SLEEP" and step > 0:
                    dream = dream_engine.dream(
                        thought_engine.persistent_memory,
                        thought_engine.memory
                    )
                    if verbose and dream.get('dream_type') != 'empty':
                        for thought in dream.get('thoughts', []):
                            print(f"    💭 Dream: {thought}", flush=True)

                if verbose:
                    compact = thought_engine.format_experience(experience, compact=True)
                    print(f"  [{ep+1}:{step:3d}] {compact} | R:{reward:+.2f}", flush=True)

                # Record observation
                if record:
                    ep_observations.append({
                        "tick": step,
                        "action": action,
                        "reward": round(reward, 3),
                        "intrinsic_reward": round(intrinsic, 3),
                        "thought": experience.get("thought", ""),
                        "emotion": experience.get("emotion", ""),
                        "health": round(adam_stats_dict.get('health', 0), 1),
                        "hunger": round(adam_stats_dict.get('hunger', 0), 1),
                        "energy": round(adam_stats_dict.get('energy', 0), 1),
                        "stress": round(adam_stats_dict.get('stress', 0), 1),
                    })

                # Next observation
                next_phase5 = thought_engine.get_phase5_signals(next_world_state, next_adam_stats_dict)
                sensory_input = encode_sensory_input(
                    next_world_state, next_adam_stats_dict,
                    fear_signal=next_phase5['fear_signal'],
                    pleasure_signal=next_phase5['pleasure_signal'],
                    pattern_confidence=next_phase5['pattern_confidence'],
                ).to(DEVICE)

                world_state = next_world_state
                adam_stats_dict = next_adam_stats_dict

                if done:
                    break

        episode_rewards.append(episode_reward)
        survival_lengths.append(episode_steps)
        thought_engine.end_episode(episode_steps)
        curiosity.end_episode()

        # Periodic summary
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_s = np.mean(survival_lengths[-10:])
            print(f"  Eval Ep {ep+1:>3}/{episodes} | R:{avg_r:>7.2f} | Surv:{avg_s:>5.1f}", flush=True)

        if record:
            all_observations.append({
                "episode": ep + 1,
                "reward": round(episode_reward, 2),
                "survival": episode_steps,
                "died": done,
                "ticks": ep_observations,
            })

    # Summary
    total_actions = sum(action_counts.values())
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Episodes: {episodes}")
    print(f"  Avg reward: {np.mean(episode_rewards):.2f}")
    print(f"  Avg survival: {np.mean(survival_lengths):.1f}")
    print(f"  Best survival: {max(survival_lengths)}")
    print(f"\n  Action Distribution (DETERMINISTIC):")
    for a in WorldSim.ACTIONS:
        cnt = action_counts.get(a, 0)
        pct = cnt / max(total_actions, 1) * 100
        print(f"    {a:>8}: {pct:>5.1f}%")

    # Curiosity stats
    cs = curiosity.get_curiosity_stats()
    print(f"\n  Curiosity: {cs['total_states_discovered']} states discovered")

    # Dream stats
    ds = dream_engine.get_dream_stats()
    print(f"  Dreams: {ds['total_dreams']} ({ds['nightmares']} nightmares, {ds['peaceful_dreams']} peaceful)")

    # Personality
    personality = thought_engine.get_personality()
    print(f"  Personality: {personality['disposition']}")
    print(f"  Vocabulary: {len(thought_engine.get_vocabulary())} words")

    # Save recorded observations
    if record:
        results = {
            "checkpoint": checkpoint_path,
            "episodes": episodes,
            "avg_reward": float(np.mean(episode_rewards)),
            "avg_survival": float(np.mean(survival_lengths)),
            "action_distribution": action_counts,
            "personality": personality,
            "curiosity_stats": cs,
            "dream_stats": ds,
            "observations": all_observations,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {output_path}")


def find_latest_checkpoint() -> str:
    """Find the latest model checkpoint in the checkpoints directory."""
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return None

    # Prefer final checkpoint, then highest episode number
    final_path = os.path.join(ckpt_dir, "baby_brain_run3_final.pt")
    if os.path.exists(final_path):
        return final_path

    # Find highest episode checkpoint
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None

    # Sort by episode number
    ep_checkpoints = []
    for f in checkpoints:
        try:
            # Format: baby_brain_run3_epXXX.pt
            ep_num = int(f.split('_ep')[1].split('.')[0])
            ep_checkpoints.append((ep_num, f))
        except (IndexError, ValueError):
            continue

    if not ep_checkpoints:
        return checkpoints[-1]  # Fallback to last alphabetical

    ep_checkpoints.sort(reverse=True)
    return os.path.join(ckpt_dir, ep_checkpoints[0][1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nafs AI — Evaluation Mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--verbose", action="store_true",
                        help="Show every tick's thought/emotion")
    parser.add_argument("--record", action="store_true",
                        help="Save all observations to JSON")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for recorded observations")

    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        verbose=args.verbose,
        record=args.record,
        output_path=args.output,
    )
