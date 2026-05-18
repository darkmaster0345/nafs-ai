"""
Nafs AI — Training Run 3 (PPO)
"What emerges when code has no memory of the world?"

This is Adam's training loop. He starts with zero knowledge and
learns purely through sensory experience — pain, hunger, warmth, fear.

Run: python main.py
"""

import time
import sys
import os
import json
import random
import numpy as np

from config import (
    PPO_CONFIG, TRAINING_CONFIG, SIM_CONFIG, ACTION_NAMES,
    OBS_DIM, ACTION_DIM
)
from world import World
from brain import PPOAgent, build_observation, compute_reward


# ── Terminal Colors ───────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print(f"{C.BOLD}{C.CYAN}")
    print("  ███╗   ██╗ █████╗ ███████╗███████╗     █████╗ ██╗")
    print("  ████╗  ██║██╔══██╗██╔════╝██╔════╝    ██╔══██╗██║")
    print("  ██╔██╗ ██║███████║█████╗  ███████╗    ███████║██║")
    print("  ██║╚██╗██║██╔══██║██╔══╝  ╚════██║    ██╔══██║██║")
    print("  ██║ ╚████║██║  ██║██║     ███████║    ██║  ██║██║")
    print("  ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═╝  ╚═╝╚═╝")
    print(f"{C.RESET}")
    print(f"  {C.DIM}\"What emerges when code has no memory of the world?\"{C.RESET}")
    print(f"  {C.DIM}Training Run 3 — PPO with Entropy Bonus + Diversity Penalty{C.RESET}\n")


# ── Simple Adam State Tracker ─────────────────────────────────────────────────
# Lightweight version for training speed — no text/LLM overhead

class AdamState:
    """Tracks Adam's internal state during training. Lightweight for speed."""

    def __init__(self):
        self.health = 100
        self.hunger = 10
        self.energy = 80
        self.stress = 5
        self.age_ticks = 0
        self.is_alive = True
        self.last_action = "IDLE"

    def apply_time_passage(self, hunger_rate, energy_drain):
        self.hunger = min(100, self.hunger + hunger_rate)
        self.energy = max(0, self.energy - energy_drain)
        self.age_ticks += 1

        if self.hunger >= 90:
            self.health = max(0, self.health - 3)
            self.stress = min(100, self.stress + 5)
        if self.energy <= 10:
            self.stress = min(100, self.stress + 3)
        if self.hunger < 40 and self.health > 70:
            self.stress = max(0, self.stress - 3)
        if self.health <= 0:
            self.is_alive = False

    def apply_outcome(self, outcome):
        self.health = max(0, min(100, self.health + outcome.get("health_delta", 0)))
        self.hunger = max(0, min(100, self.hunger + outcome.get("hunger_delta", 0)))
        self.energy = max(0, min(100, self.energy + outcome.get("energy_delta", 0)))

        outcome_text = outcome.get("outcome_text", "").lower()
        if "light again" in outcome_text or "rest" in outcome_text:
            self.stress = max(0, self.stress - 20)
        if self.health <= 0:
            self.is_alive = False

    def reset(self):
        self.health = 100
        self.hunger = 10
        self.energy = 80
        self.stress = 5
        self.age_ticks = 0
        self.is_alive = True
        self.last_action = "IDLE"

    def status_line(self):
        return (
            f"Health:{self.health:>3}  "
            f"Hunger:{self.hunger:>3}  "
            f"Energy:{self.energy:>3}  "
            f"Stress:{self.stress:>3}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    clear()
    print_header()

    # Initialize
    agent = PPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    world = World()
    adam = AdamState()

    # Training stats
    total_ticks = 0
    total_episodes = 0
    episode_ticks = 0
    episode_reward = 0.0
    episode_rewards = []
    action_counts = {name: 0 for name in ACTION_NAMES}
    best_survival = 0

    # Timing
    training_start = time.time()
    last_log_time = training_start

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Load existing model if available
    model_path = TRAINING_CONFIG["model_save_path"]
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"  {C.GREEN}[Loaded existing model]{C.RESET}")

    print(f"  {C.GREEN}Training starting...{C.RESET}")
    print(f"  {C.DIM}Adam wakes up. He knows nothing.{C.RESET}")
    print(f"  {C.DIM}entropy_coef={PPO_CONFIG['entropy_coef']}, diversity_penalty={PPO_CONFIG['diversity_penalty']}{C.RESET}\n")

    try:
        while True:
            # ── Episode Loop ────────────────────────────────────────────
            adam.reset()
            world = World()  # fresh world each episode
            episode_ticks = 0
            episode_reward = 0.0
            done = False

            while not done and episode_ticks < TRAINING_CONFIG["max_ticks_per_episode"]:
                # World advances
                world.tick_forward()
                adam.apply_time_passage(
                    hunger_rate=SIM_CONFIG["hunger_rate"],
                    energy_drain=SIM_CONFIG["energy_drain"],
                )

                # Build observation
                obs = build_observation(adam, world)

                # Select action from policy
                action_idx, log_prob, value = agent.select_action(obs)
                action_name = ACTION_NAMES[action_idx]

                # Apply action through world
                outcome = world.apply_action(action_name, "", adam.__dict__)
                adam.apply_outcome(outcome)
                adam.last_action = action_name

                # Compute reward
                reward = compute_reward(adam, outcome, action_idx)

                # Apply diversity penalty (Fix 2)
                diversity_pen = agent.compute_diversity_penalty(action_idx)
                reward -= diversity_pen

                # Check if episode is done
                done = not adam.is_alive

                # Store experience in buffer
                agent.buffer.store(obs, action_idx, log_prob, reward, value, float(done))

                # Track stats
                episode_ticks += 1
                episode_reward += reward
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
                total_ticks += 1

                # PPO update when buffer is full
                if len(agent.buffer) >= TRAINING_CONFIG["update_interval"]:
                    # Bootstrap value for last state
                    last_obs = build_observation(adam, world)
                    with torch.no_grad():
                        _, last_value = agent.actor_critic(
                            torch.FloatTensor(last_obs).unsqueeze(0)
                        )
                    update_stats = agent.update(last_value.item())

                    if update_stats and total_ticks % TRAINING_CONFIG["log_interval"] < TRAINING_CONFIG["update_interval"]:
                        elapsed = time.time() - training_start
                        print(
                            f"  {C.DIM}[Update {update_stats['updates']:>4}] "
                            f"Policy: {update_stats['policy_loss']:>7.3f}  "
                            f"Value: {update_stats['value_loss']:>7.3f}  "
                            f"Entropy: {update_stats['entropy']:>5.3f}  "
                            f"Time: {elapsed:>6.0f}s{C.RESET}"
                        )

                # Save model periodically
                if total_ticks % TRAINING_CONFIG["save_interval"] == 0 and total_ticks > 0:
                    agent.save(model_path)

            # ── Episode End ─────────────────────────────────────────────
            total_episodes += 1
            episode_rewards.append(episode_reward)

            if episode_ticks > best_survival:
                best_survival = episode_ticks

            # Log episode results
            elapsed = time.time() - training_start
            elapsed_min = elapsed / 60

            print(f"\n  {C.BOLD}{'─' * 60}{C.RESET}")
            print(f"  {C.YELLOW}Episode {total_episodes} Complete{C.RESET}")
            print(f"  {C.DIM}Survival: {episode_ticks} ticks  |  "
                  f"Reward: {episode_reward:>7.2f}  |  "
                  f"Best: {best_survival}  |  "
                  f"Time: {elapsed_min:>5.1f} min{C.RESET}")

            # Action distribution for this episode
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                dist_str = "  ".join(
                    f"{name}:{count/total_actions*100:>4.0f}%"
                    for name, count in action_counts.items()
                    if count > 0
                )
                print(f"  {C.CYAN}Actions: {dist_str}{C.RESET}")

            # Check for mono-behavior collapse
            if total_actions > 0:
                max_pct = max(count / total_actions for count in action_counts.values())
                if max_pct > 0.60:
                    dominant = max(action_counts, key=action_counts.get)
                    print(f"  {C.RED}{C.BOLD}[WARNING] Mono-behavior detected: "
                          f"{dominant} at {max_pct*100:.0f}%{C.RESET}")

            print(f"  {C.BOLD}{'─' * 60}{C.RESET}\n")

            # Save training log
            log_data = {
                "total_ticks": total_ticks,
                "total_episodes": total_episodes,
                "best_survival": best_survival,
                "last_episode_reward": episode_reward,
                "last_episode_ticks": episode_ticks,
                "elapsed_seconds": elapsed,
                "action_counts": action_counts,
                "entropy_coef": PPO_CONFIG["entropy_coef"],
                "diversity_penalty": PPO_CONFIG["diversity_penalty"],
            }
            with open(TRAINING_CONFIG["training_log_path"], "w") as f:
                json.dump(log_data, f, indent=2)

    except KeyboardInterrupt:
        elapsed = time.time() - training_start
        elapsed_min = elapsed / 60

        print(f"\n\n  {C.YELLOW}[Observer] Training paused by user.{C.RESET}")
        print(f"  {C.DIM}Total training time: {elapsed_min:.1f} minutes{C.RESET}")
        print(f"  {C.DIM}Total ticks: {total_ticks}{C.RESET}")
        print(f"  {C.DIM}Total episodes: {total_episodes}{C.RESET}")
        print(f"  {C.DIM}Best survival: {best_survival} ticks{C.RESET}")

        # Final action distribution
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            print(f"\n  {C.BOLD}Final Action Distribution:{C.RESET}")
            for name, count in action_counts.items():
                pct = count / total_actions * 100
                bar = "█" * int(pct / 2)
                print(f"    {name:>8}: {pct:>5.1f}% {C.CYAN}{bar}{C.RESET}")

        # Save model
        agent.save(model_path)

        # Final training log
        log_data = {
            "total_ticks": total_ticks,
            "total_episodes": total_episodes,
            "best_survival": best_survival,
            "last_episode_reward": episode_reward,
            "last_episode_ticks": episode_ticks,
            "elapsed_seconds": elapsed,
            "action_counts": action_counts,
            "entropy_coef": PPO_CONFIG["entropy_coef"],
            "diversity_penalty": PPO_CONFIG["diversity_penalty"],
        }
        with open(TRAINING_CONFIG["training_log_path"], "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"\n  {C.DIM}Model saved. Run again to continue training.{C.RESET}\n")


if __name__ == "__main__":
    # Import torch here to avoid issues
    import torch
    train()
