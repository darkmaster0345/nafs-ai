"""
Nafs AI — Observer Mode (Single-Life)
"Watch Adam live. No training. Just observation."

Runs Adam with a trained brain but NO learning.
You observe Adam's behavior without the training process
influencing what you see.

Usage:
    python evaluate.py                          # Observe Adam's life with latest checkpoint
    python evaluate.py --checkpoint checkpoints/adam_tick1000.pt
    python evaluate.py --verbose                # Show every tick's thought
    python evaluate.py --record                 # Save all observations to file
"""

import torch
import numpy as np
import os
import json
import time
import argparse
import copy

from baby_brain_model import BabyBrain
from world_sim import WorldSim, BIOMES, WEATHER_TYPES
from sensory_encoder import encode_sensory_input, INPUT_DIM
from thought_engine import ThoughtEngine
from curiosity import CuriosityModule
from dreaming import DreamEngine
from train import birth_display, death_display, compact_tick_display, full_stats_display

HIDDEN_DIM = 256
NUM_ACTIONS = len(WorldSim.ACTIONS)


def observe(checkpoint_path: str = None, verbose: bool = False,
            record: bool = False, output_path: str = "observation_results.json",
            tick_delay: float = 0.05):
    """
    Observe Adam's life with a trained brain — no learning, pure observation.
    """
    DEVICE = torch.device("cpu")

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No checkpoint found! Train first with: python train.py")
        return

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = BabyBrain(
        ckpt.get('input_dim', INPUT_DIM),
        ckpt.get('hidden_dim', HIDDEN_DIM),
        ckpt.get('num_actions', NUM_ACTIONS)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    env = WorldSim()
    world_state, adam_stats_dict = env.reset()
    thought_engine = ThoughtEngine(memory_size=10)
    curiosity = CuriosityModule()
    dream_engine = DreamEngine()

    # Birth display
    birth_display(env, world_state)
    print(f"  Brain loaded from: {checkpoint_path}")
    print(f"  Training ticks: {ckpt.get('tick', ckpt.get('episode', 'unknown'))}")
    print()

    # Initialize
    phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)
    sensory_input = encode_sensory_input(
        world_state, adam_stats_dict,
        fear_signal=phase5['fear_signal'],
        pleasure_signal=phase5['pleasure_signal'],
        pattern_confidence=phase5['pattern_confidence'],
    ).to(DEVICE)
    hidden_state = model.init_hidden(1).to(DEVICE)

    action_counts = {a: 0 for a in WorldSim.ACTIONS}
    total_reward = 0.0
    tick = 0
    all_observations = []
    latest_dream = None

    while True:
        try:
            tick += 1
            prev_stats = copy.deepcopy(adam_stats_dict)

            phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)

            with torch.no_grad():
                action_logits, state_value, hidden_state = model(
                    sensory_input.unsqueeze(0), hidden_state
                )
                action_logits_sq = action_logits.squeeze(0)

                # Deterministic action (argmax — no exploration)
                action_idx = action_logits_sq.argmax()
                action = WorldSim.ACTIONS[action_idx.item()]

            next_world_state, next_adam_stats_dict, reward, done = env.step(action)

            action_counts[action] = action_counts.get(action, 0) + 1
            total_reward += reward

            # Generate inner experience
            experience = thought_engine.experience(
                world_state=next_world_state,
                adam_stats=next_adam_stats_dict,
                action=action,
                prev_stats=prev_stats,
                tick=tick,
                reward=reward,
            )

            # Vocabulary discovery
            if "new_words" in experience:
                for word, meaning in experience["new_words"]:
                    print(f"  NEW WORD: \"{word}\" = {meaning}", flush=True)

            # Dreaming
            if action == "SLEEP" and tick > 1:
                dream = dream_engine.dream(
                    thought_engine.persistent_memory,
                    thought_engine.memory
                )
                if dream.get('dream_type') != 'empty':
                    latest_dream = dream

            # Display
            biome_name = world_state.get('biome', '?')
            biome_data = BIOMES.get(biome_name, {})
            weather_name = world_state.get('weather', '?')
            weather_data = WEATHER_TYPES.get(weather_name, {})

            if verbose:
                compact_tick_display(
                    tick, world_state, adam_stats_dict, action, reward,
                    experience.get('thought', ''), experience.get('emotion', ''),
                    biome_data, weather_data
                )

            # Full stats every 50 ticks
            if tick % 50 == 0:
                full_stats_display(
                    tick, world_state, adam_stats_dict, action, reward,
                    thought_engine, curiosity, dream_engine, total_reward,
                    action_counts, env, model, 0.0
                )

            # Record
            if record:
                all_observations.append({
                    "tick": tick,
                    "action": action,
                    "reward": round(reward, 3),
                    "thought": experience.get("thought", ""),
                    "emotion": experience.get("emotion", ""),
                    "biome": biome_name,
                    "weather": weather_name,
                    "health": round(adam_stats_dict.get('health', 0), 1),
                    "hunger": round(adam_stats_dict.get('hunger', 0), 1),
                    "energy": round(adam_stats_dict.get('energy', 0), 1),
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

            if tick_delay > 0:
                time.sleep(tick_delay)

        except KeyboardInterrupt:
            print(f"\n  Observation interrupted at tick {tick}.")
            break
        except Exception as e:
            print(f"  ERROR at tick {tick}: {e}")
            continue

    # Death display
    death_display(
        tick, total_reward, adam_stats_dict, thought_engine, curiosity,
        dream_engine, action_counts, world_state
    )

    # Save observations
    if record:
        results = {
            "checkpoint": checkpoint_path,
            "ticks_lived": tick,
            "total_reward": total_reward,
            "action_counts": action_counts,
            "observations": all_observations,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to: {output_path}")


def find_latest_checkpoint() -> str:
    """Find the latest model checkpoint."""
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return None

    # Prefer final, then highest tick number
    final_path = os.path.join(ckpt_dir, "adam_final.pt")
    if os.path.exists(final_path):
        return final_path

    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None

    tick_checkpoints = []
    for f in checkpoints:
        try:
            tick_num = int(f.split('tick')[1].split('.')[0])
            tick_checkpoints.append((tick_num, f))
        except (IndexError, ValueError):
            continue

    if not tick_checkpoints:
        return os.path.join(ckpt_dir, checkpoints[-1])

    tick_checkpoints.sort(reverse=True)
    return os.path.join(ckpt_dir, tick_checkpoints[0][1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nafs AI — Observer Mode")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=str, default="observation_results.json")
    parser.add_argument("--delay", type=float, default=0.05)

    args = parser.parse_args()
    observe(
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
        record=args.record,
        output_path=args.output,
        tick_delay=args.delay,
    )
