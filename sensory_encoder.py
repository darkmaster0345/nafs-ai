
import torch

def encode_sensory_input(world_state: dict, adam_stats: dict,
                        fear_signal: float = 0.0, pleasure_signal: float = 0.0,
                        pattern_confidence: float = 0.0) -> torch.Tensor:
    """
    Converts NAFS world state and Adam's stats into a 15-dimensional numerical vector.
    Dimensions: temperature, pain, hunger, energy, light_level, smell_food, smell_danger,
                sound_level, wetness, touch_softness, proximity_entity, internal_stress,
                fear_signal (Phase 5), pleasure_signal (Phase 5), pattern_confidence (Phase 5).
    Each dimension is mapped to a float in [-1, 1] or [0, 1] range.
    """

    # Default values for world_state and adam_stats to avoid KeyError if a key is missing
    # and to provide a baseline for mapping.
    default_world_state = {
        'temperature': 25.0,  # Celsius
        'light_level': 0.5,   # 0.0 (dark) to 1.0 (bright)
        'smell_food': 0.0,    # 0.0 (none) to 1.0 (strong)
        'smell_danger': 0.0,  # 0.0 (none) to 1.0 (strong)
        'sound_level': 0.0,   # 0.0 (silent) to 1.0 (loud)
        'wetness': 0.0,       # 0.0 (dry) to 1.0 (soaked)
        'proximity_entity': 0.0, # 0.0 (far) to 1.0 (close)
        'weather': 'clear',   # 'clear', 'rainy', 'cold', 'hot'
        'event': 'none'       # 'none', 'food_nearby', 'danger_nearby', 'water_nearby'
    }
    default_adam_stats = {
        'health': 100.0,      # 0.0 (dead) to 100.0 (full health)
        'hunger': 0.0,        # 0.0 (full) to 100.0 (starving)
        'energy': 100.0,      # 0.0 (exhausted) to 100.0 (full energy)
        'pain': 0.0,          # 0.0 (none) to 10.0 (severe)
        'stress': 0.0         # 0.0 (calm) to 100.0 (high stress)
    }

    # Update defaults with actual state, ensuring all keys are present
    ws = {**default_world_state, **world_state}
    stats = {**default_adam_stats, **adam_stats}

    # --- Map world_state and adam_stats to a 12-dimensional vector ---

    # 1. Temperature: Map from typical range (e.g., -10 to 40 Celsius) to [-1, 1]
    # Assuming -10C is very cold (-1) and 40C is very hot (1), 15C is 0.
    temperature_scaled = max(-1.0, min(1.0, (ws['temperature'] - 15.0) / 25.0)) # -10C -> -1, 15C -> 0, 40C -> 1

    # 2. Pain: Map from [0, 10] to [0, 1]
    pain_scaled = max(0.0, min(1.0, stats['pain'] / 10.0))

    # 3. Hunger: Map from [0, 100] to [0, 1] (0=full, 1=starving)
    hunger_scaled = max(0.0, min(1.0, stats['hunger'] / 100.0))

    # 4. Energy: Map from [0, 100] to [0, 1] (0=exhausted, 1=full)
    energy_scaled = max(0.0, min(1.0, stats['energy'] / 100.0))

    # 5. Light Level: Already [0, 1]
    light_level_scaled = max(0.0, min(1.0, ws['light_level']))

    # 6. Smell Food: Already [0, 1]
    smell_food_scaled = max(0.0, min(1.0, ws['smell_food']))

    # 7. Smell Danger: Already [0, 1]
    smell_danger_scaled = max(0.0, min(1.0, ws['smell_danger']))

    # 8. Sound Level: Already [0, 1]
    sound_level_scaled = max(0.0, min(1.0, ws['sound_level']))

    # 9. Wetness: Already [0, 1]
    wetness_scaled = max(0.0, min(1.0, ws['wetness']))

    # 10. Touch Softness: Assuming this is a property of the environment Adam is currently in.
    # For simplicity, let's assume it's a direct input from world_state, mapped to [0, 1].
    # If not present, default to 0.5 (neutral).
    touch_softness_scaled = max(0.0, min(1.0, ws.get('touch_softness', 0.5)))

    # 11. Proximity Entity: Already [0, 1]
    proximity_entity_scaled = max(0.0, min(1.0, ws['proximity_entity']))

    # 12. Internal Stress: Map from [0, 100] to [0, 1]
    internal_stress_scaled = max(0.0, min(1.0, stats['stress'] / 100.0))

    # Phase 5: Fear signal — how much past fear is relevant to current situation
    # Scaled to [0, 1]
    fear_signal_scaled = max(0.0, min(1.0, fear_signal))

    # Phase 5: Pleasure signal — how much past reward is relevant to current situation
    # Scaled to [0, 1]
    pleasure_signal_scaled = max(0.0, min(1.0, pleasure_signal))

    # Phase 5: Pattern confidence — how confident Adam is about the best action
    # for this situation (based on past experience). Scaled to [0, 1].
    # 0 = no experience, 1 = very confident
    pattern_confidence_scaled = max(0.0, min(1.0, pattern_confidence))

    # Combine into a single tensor (15 dimensions)
    sensory_vector = torch.tensor([
        temperature_scaled,
        pain_scaled,
        hunger_scaled,
        energy_scaled,
        light_level_scaled,
        smell_food_scaled,
        smell_danger_scaled,
        sound_level_scaled,
        wetness_scaled,
        touch_softness_scaled,
        proximity_entity_scaled,
        internal_stress_scaled,
        fear_signal_scaled,         # Phase 5: dimension 13
        pleasure_signal_scaled,     # Phase 5: dimension 14
        pattern_confidence_scaled,  # Phase 5: dimension 15
    ], dtype=torch.float32)

    return sensory_vector
