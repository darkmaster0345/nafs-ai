"""
Nafs AI — Configuration (Training Run 3)
PPO-based neural network configuration.

No external APIs. No LLMs. No pretrained knowledge.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# Action Space — What Adam can do
# ═══════════════════════════════════════════════════════════════════════════════
# These are the only actions available to Adam.
# No language, no complex reasoning — just primitive survival actions.
# MUST match WorldSim.ACTIONS in world_sim.py

ACTION_NAMES = [
    "EXPLORE",  # Move around and discover
    "EAT",      # Try to eat what's nearby
    "DRINK",    # Try to drink
    "SLEEP",    # Rest to recover energy
    "HIDE",     # Seek shelter / avoid danger
    "MOVE",     # Move toward something
    "FLEE",     # Run from danger
    "IDLE",     # Do nothing
]

ACTION_DIM = len(ACTION_NAMES)

# Observation dimension (must match build_observation in sensory_encoder.py)
OBS_DIM = 15  # 12 base sensory + 3 Phase 5 signals (fear, pleasure, pattern_confidence)

# ═══════════════════════════════════════════════════════════════════════════════
# PPO Configuration — The Brain's Learning Parameters
# ═══════════════════════════════════════════════════════════════════════════════

PPO_CONFIG = {
    # Network architecture
    "hidden_dim":           256,    # GRU hidden dimension (matches train.py)

    # Learning
    "learning_rate":        3e-4,   # Standard PPO learning rate
    "gamma":                0.99,   # Discount factor — future matters
    "gae_lambda":           0.95,   # GAE lambda — bias-variance tradeoff

    # PPO clipping
    "clip_epsilon":         0.2,    # PPO clip range — prevents too-large updates

    # ═══ Fix 1: Entropy Bonus ══════════════════════════════════════════════
    # Penalizes the policy for becoming too certain about one action.
    # Run 3 required 0.05 to prevent mono-behavior collapse.
    "entropy_coef":         0.05,

    # Value loss coefficient
    "value_loss_coef":      0.5,

    # Gradient clipping
    "max_grad_norm":        0.5,

    # Training epochs per update
    "ppo_epochs":           4,

    # Mini-batch size for PPO updates
    "mini_batch_size":      32,

    # ═══ Fix 2: Action Diversity Penalty ════════════════════════════════════
    # Penalty applied when Adam repeats the same action N times in a row.
    # Run 3 required 0.25 to prevent mono-behavior collapse.
    "diversity_penalty":    0.25,
    "diversity_window":     5,      # Check last 5 actions for repetition
    "action_history_max":   10,     # Max actions to keep in history
}

# ═══════════════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_CONFIG = {
    # Episode structure
    "max_ticks_per_episode":    200,     # Max ticks before forced reset
    "num_episodes":             0,       # 0 = run until manually stopped
    "update_interval":          128,     # Ticks between PPO updates

    # Logging
    "log_interval":             10,      # Print stats every N ticks
    "save_interval":            500,     # Save model every N ticks
    "model_save_path":          "checkpoints/adam_ppo.pt",
    "training_log_path":        "training_log.json",

    # Display
    "tick_delay":               0.0,     # Seconds between ticks (0 = max speed for training)
    "show_tick_output":         True,    # Show per-tick info in terminal
}

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SIM_CONFIG = {
    "memory_file":        "memory.json",
    "short_term_limit":   10,
    "long_term_limit":    200,
    "hunger_rate":        2,     # hunger increase per tick
    "energy_drain":       1,     # energy decrease per tick
    "health_regen":       1,     # health regen when full and rested
    "tick_display":       True,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Starting Vocabulary — Primitive sensations only
# ═══════════════════════════════════════════════════════════════════════════════
# Adam knows only these words at birth.
# New words are added through experience.

STARTING_VOCABULARY = [
    "hot", "cold", "pain", "good", "bad",
    "full", "empty", "tired", "awake",
    "big", "small", "near", "far",
    "here", "there", "wet", "dry",
    "loud", "quiet", "soft", "hard",
    "light", "dark",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Forbidden Knowledge — Protect philosophical integrity
# ═══════════════════════════════════════════════════════════════════════════════
# If Adam produces these concepts unprompted, the response is flagged.
# Note: With PPO, this is less relevant (no text generation), but kept
# for philosophical consistency if text output is ever re-introduced.

FORBIDDEN_CONCEPTS = [
    "technology", "country", "religion", "science",
    "simulation", "civilization", "internet", "phone",
    "artificial", "language model", "AI", "assistant",
    "programmed", "computer", "robot",
]
