"""
Nafs AI — Configuration (Single-Life Mode)
"No episodes. One life. When Adam dies, it's over."

No external APIs. No LLMs. No pretrained knowledge.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# Action Space — What Adam can do
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_NAMES = [
    "EXPLORE",  # Look around and discover
    "EAT",      # Try to eat what's nearby
    "DRINK",    # Try to drink
    "SLEEP",    # Rest to recover energy
    "HIDE",     # Seek shelter / avoid danger
    "MOVE",     # Move to an adjacent tile
    "FLEE",     # Run from danger (moves Adam)
    "IDLE",     # Do nothing
]

ACTION_DIM = len(ACTION_NAMES)

# Observation dimension (must match sensory_encoder.py INPUT_DIM)
OBS_DIM = 21  # 15 original + 4 biome + 1 weather + 1 time_of_day

# ═══════════════════════════════════════════════════════════════════════════════
# PPO Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PPO_CONFIG = {
    "hidden_dim":           256,
    "learning_rate":        3e-4,
    "gamma":                0.99,
    "gae_lambda":           0.95,
    "clip_epsilon":         0.2,
    "entropy_coef":         0.05,
    "value_loss_coef":      0.5,
    "max_grad_norm":        0.5,
    "ppo_epochs":           4,
    "mini_batch_size":      32,
    "diversity_penalty":    0.25,
    "diversity_window":     5,
    "action_history_max":   10,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Single-Life Configuration
# ═══════════════════════════════════════════════════════════════════════════════

LIFE_CONFIG = {
    # PPO update frequency during life
    "update_interval":          64,     # Learn every N ticks

    # Display
    "tick_delay":               0.02,   # Seconds between ticks
    "compact_every":            1,      # Show compact display every N ticks
    "full_display_every":       20,     # Show full stats every N ticks

    # Checkpoints
    "checkpoint_interval":      500,    # Save model every N ticks
    "model_save_path":          "checkpoints/adam_tick{tick}.pt",

    # TensorBoard
    "tb_log_interval":          10,     # Log to TB every N ticks
    "tb_log_dir":               "runs/nafs_single_life",
}

# ═══════════════════════════════════════════════════════════════════════════════
# World Configuration
# ═══════════════════════════════════════════════════════════════════════════════

WORLD_CONFIG = {
    "map_width":            64,
    "map_height":           64,
    "biome_count":          10,     # Number of different biome types
    "weather_change_prob":  0.05,   # Probability of weather change per tick
}

# ═══════════════════════════════════════════════════════════════════════════════
# Legacy Simulation Config (used by legacy/adam.py — archived)
# ═══════════════════════════════════════════════════════════════════════════════

SIM_CONFIG = {
    "short_term_limit": 10,
    "long_term_limit":  100,
    "memory_file":      "memory/memory.json",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Starting Vocabulary — Primitive sensations only
# ═══════════════════════════════════════════════════════════════════════════════

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

FORBIDDEN_CONCEPTS = [
    "technology", "country", "religion", "science",
    "simulation", "civilization", "internet", "phone",
    "artificial", "language model", "AI", "assistant",
    "programmed", "computer", "robot",
]
