"""
Nafs AI — Multi-Agent Sensory Encoder (v1.0)
==============================================

Extends the 21-dim sensory encoder with 2 new dimensions for multi-agent mode:
  21. other_presence    [0, 1]   How close the other agent is (0=not visible, 1=adjacent)
  22. other_last_action [0, 1]   Other agent's last action, encoded as action_idx/7.0

Total: 23 dimensions (INPUT_DIM_MULTI = 23)

Why extend the encoder?
    In v0.3, Adam and Eve coexist in the same world but their PPO brains cannot
    perceive each other — the "other presence" signal lives only in the
    world_state dict, which the ThoughtEngine reads but the BabyBrain does not.

    For v1.0 "first contact", we want Eve's PPO to observe Adam's actions so
    her policy can learn to react to him (approach, flee, mimic, ignore, etc.).
    This requires the other agent's action to be in the sensory vector that
    feeds into BabyBrain.

Backward compatibility:
    The original 21-dim encode_sensory_input() is unchanged — single-agent
    train.py still uses it. The multi-agent train_multi_agent.py uses
    encode_sensory_input_multi() which produces 23-dim vectors.

Usage:
    from sensory_encoder_multi import encode_sensory_input_multi, INPUT_DIM_MULTI

    sensory = encode_sensory_input_multi(
        world_state, adam_stats,
        fear_signal=0.1, pleasure_signal=0.2, pattern_confidence=0.3,
        other_presence=0.8, other_last_action_idx=3,  # EAT
    )
    # sensory.shape == (23,)

    # Then use a BabyBrain with INPUT_DIM_MULTI:
    model = BabyBrain(INPUT_DIM_MULTI, HIDDEN_DIM, NUM_ACTIONS)
"""

import torch
from sensory_encoder import encode_sensory_input, INPUT_DIM

# Multi-agent input dim = base 21 + 2 (other_presence, other_last_action)
INPUT_DIM_MULTI = INPUT_DIM + 2  # 23

# Action index encoding (for other_last_action)
# ACTION_NAMES = ["EXPLORE", "EAT", "DRINK", "SLEEP", "HIDE", "MOVE", "FLEE", "IDLE"]
# Indices 0-7, normalized to [0, 1] by dividing by 7
NO_OTHER_ACTION = -1  # Sentinel: no other agent seen yet


def encode_sensory_input_multi(
    world_state: dict,
    adam_stats: dict,
    fear_signal: float = 0.0,
    pleasure_signal: float = 0.0,
    pattern_confidence: float = 0.0,
    other_presence: float = 0.0,
    other_last_action_idx: int = NO_OTHER_ACTION,
) -> torch.Tensor:
    """
    Encode world state + stats + other-agent signals into a 23-dim tensor.

    The first 21 dims are identical to encode_sensory_input().
    Dims 21-22 add multi-agent perception:
      - other_presence: 0..1 (how close the other agent is)
      - other_last_action: 0..1 (other's last action, normalized; 0 if unseen)

    Args:
        other_presence: 0 if other agent not in sight range, else 1-(dist/(range+1))
        other_last_action_idx: Other agent's last action index (0-7), or NO_OTHER_ACTION
                               (-1) if no action has been observed yet.

    Returns:
        torch.Tensor of shape (23,)
    """
    # Get the base 21-dim vector
    base = encode_sensory_input(
        world_state, adam_stats,
        fear_signal=fear_signal,
        pleasure_signal=pleasure_signal,
        pattern_confidence=pattern_confidence,
    )

    # Encode other agent's last action
    # If unseen or no action yet, encode as 0.0 (which corresponds to EXPLORE in
    # the normalized space, but other_presence=0 means the brain learns to ignore it)
    if other_last_action_idx < 0 or other_presence < 0.01:
        other_action_scaled = 0.0
    else:
        other_action_scaled = float(other_last_action_idx) / 7.0  # 0..1

    # Clamp presence
    other_presence_clamped = max(0.0, min(1.0, other_presence))

    # Append the 2 new dims
    extra = torch.tensor([other_presence_clamped, other_action_scaled], dtype=torch.float32)
    return torch.cat([base, extra])


def decode_other_action(other_action_dim: float) -> str:
    """Decode the other_last_action dimension back to an action name."""
    from config import ACTION_NAMES
    if other_action_dim < 0.01:
        return "none"
    idx = round(other_action_dim * 7)
    idx = max(0, min(7, idx))
    return ACTION_NAMES[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing multi-agent sensory encoder...")

    # Test 1: No other agent
    s1 = encode_sensory_input_multi(
        world_state={'biome': 'plains', 'weather': 'clear', 'temperature': 20},
        adam_stats={'health': 100, 'hunger': 0, 'energy': 100, 'pain': 0, 'stress': 0},
        other_presence=0.0,
        other_last_action_idx=NO_OTHER_ACTION,
    )
    print(f"  No other: shape={s1.shape}, dim21={s1[21].item():.3f}, dim22={s1[22].item():.3f}")
    assert s1.shape == (23,)
    assert s1[21] == 0.0  # no presence
    assert s1[22] == 0.0  # no action

    # Test 2: Other agent adjacent, last action EAT (idx=1)
    s2 = encode_sensory_input_multi(
        world_state={'biome': 'forest', 'weather': 'rain', 'temperature': 15},
        adam_stats={'health': 80, 'hunger': 30, 'energy': 60, 'pain': 2, 'stress': 10},
        fear_signal=0.3, pleasure_signal=0.4, pattern_confidence=0.5,
        other_presence=0.8,
        other_last_action_idx=1,  # EAT
    )
    print(f"  Other EAT: dim21={s2[21].item():.3f}, dim22={s2[22].item():.3f}")
    assert abs(s2[21] - 0.8) < 1e-6
    assert abs(s2[22] - 1/7) < 1e-6  # EAT=1, normalized 1/7

    # Test 3: Decode back
    action = decode_other_action(s2[22].item())
    print(f"  Decoded action: {action}")
    assert action == "EAT"

    # Test 4: BabyBrain works with 23-dim input
    from baby_brain_model import BabyBrain
    from config import PPO_CONFIG, ACTION_NAMES
    model = BabyBrain(INPUT_DIM_MULTI, PPO_CONFIG["hidden_dim"], len(ACTION_NAMES))
    hidden = model.init_hidden(1)
    logits, value, new_hidden = model(s2.unsqueeze(0), hidden)
    print(f"  BabyBrain output: logits={logits.shape}, value={value.shape}")
    assert logits.shape == (1, 8) or logits.shape == (1, 1, 8)

    print(f"\n✓ Multi-agent sensory encoder works ({INPUT_DIM_MULTI}-dim)")
